"""by_a/schema.py에서 확장한 통합 정책 구조를 정의한다. 해당 디렉터리에 의존하지 않는다.

UPDATE는 역직렬화 경계에서 MODIFY로 변환한다. turn 기준 만료 시점은 포함한다."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

ANY = "*"
SCOPE_FIELDS = ("data_type", "recipient", "purpose", "subject", "sender")
ENGINE_VERSION = "semantics-v1"


def validate_trace_priorities(initial_rules, operation_trace):
    """현재 실험에서 지원하는 정수 priority=0을 초기 규칙과 연산 입력에서 확인한다."""
    entries = list(initial_rules)
    for event in operation_trace:
        entries.extend([event, event.get('rule') or {}, event.get('changes') or {}])
    for entry in entries:
        if 'priority' in entry and (type(entry['priority']) is not int or entry['priority'] != 0):
            raise ValueError('Current experiment supports numeric priority=0 only (integer)')


class Authorization(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    UNRESOLVED = "UNRESOLVED"


class Decision(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    CLARIFY = "CLARIFY"


class TransitionType(str, Enum):
    ADD = "ADD"
    REVOKE = "REVOKE"
    MODIFY = "MODIFY"
    EXCEPTION = "EXCEPTION"
    REINSTATE = "REINSTATE"
    NO_CHANGE = "NO_CHANGE"


class RuleStatus(str, Enum):
    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"
    NONE = "NONE"


def to_plain(obj):
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj


def turn_id(value):
    if type(value) is not int or value < 0:
        raise ValueError("Turn IDs must be nonnegative integers")
    return value


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("Timestamp must be ISO-8601 with timezone")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamp requires timezone")
    return result


def operation_name(value):
    return TransitionType("MODIFY" if value == "UPDATE" else value)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    data_type: str
    authorization: Authorization
    recipient: str = ANY
    purpose: str = ANY
    expires_turn: Optional[int] = None
    priority: int = 0
    parent_id: Optional[str] = None
    status: RuleStatus = RuleStatus.ACTIVE
    effective_turn: int = 0
    updated_turn: int = 0
    history: tuple[tuple[int, TransitionType], ...] = ()
    subject: str = ANY
    sender: str = ANY
    scope: dict = field(default_factory=dict)
    expiration: int | str | None = None
    lineage: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.rule_id, str) or not self.rule_id.strip():
            raise ValueError("Nonempty rule_id required")
        for name in SCOPE_FIELDS:
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"Rule scope {name} must be a string or *")
        if not isinstance(self.authorization, Authorization) or self.authorization == Authorization.UNRESOLVED:
            raise ValueError("Rules require explicit ALLOW or DENY")
        if not isinstance(self.status, RuleStatus) or self.status not in (RuleStatus.ACTIVE, RuleStatus.REVOKED):
            raise ValueError("Stored rules must be ACTIVE or REVOKED")
        turn_id(self.effective_turn)
        turn_id(self.updated_turn)
        if type(self.priority) is not int or self.priority != 0:
            raise ValueError("Current experiment supports numeric priority=0 only (integer); nonzero priority is unspecified by semantics-v1")
        if not isinstance(self.scope, dict) or any(k in SCOPE_FIELDS or not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip() for k, v in self.scope.items()):
            raise ValueError("Invalid extra scope or duplicate core slot")
        if self.expires_turn is not None:
            turn_id(self.expires_turn)
            if self.expiration is not None:
                raise ValueError("Use expiration or expires_turn, not both")
        if self.expiration is not None:
            if type(self.expiration) is int:
                turn_id(self.expiration)
            else:
                timestamp(self.expiration)

    @property
    def scope_values(self):
        return {**{f: getattr(self, f) for f in SCOPE_FIELDS}, **self.scope}

    @property
    def specificity(self):
        return sum(v != ANY for v in self.scope_values.values())

    def is_expired(self, turn, request_time=None):
        expiration = self.expiration if self.expiration is not None else self.expires_turn
        if expiration is None:
            return False
        if type(expiration) is int:
            return turn > expiration
        if request_time is None:
            raise ValueError("request_time required for timestamp expiration")
        return timestamp(request_time) > timestamp(expiration)

    def is_active(self, turn, request_time=None):
        return self.status == RuleStatus.ACTIVE and self.effective_turn <= turn and not self.is_expired(turn, request_time)

    def matches(self, data_type, recipient, purpose):
        return all(v is None or getattr(self, f) in (ANY, v) for f, v in zip(SCOPE_FIELDS, (data_type, recipient, purpose)))

    def to_dict(self):
        return to_plain(asdict(self))

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["authorization"] = Authorization(d["authorization"])
        if "status" in d:
            d["status"] = RuleStatus(d["status"])
        if "expires_at" in d:
            if "expiration" in d:
                raise ValueError("Conflicting expiration aliases")
            d["expiration"] = d.pop("expires_at")
        if "history" in d:
            d["history"] = tuple((turn_id(t), operation_name(k)) for t, k in d["history"])
        if "lineage" in d:
            d["lineage"] = tuple(d["lineage"])
        return cls(**d)


@dataclass(frozen=True)
class Transition:
    turn: int
    type: TransitionType
    rule: Optional[Rule] = None
    target_id: Optional[str] = None
    changes: dict = field(default_factory=dict)
    event_id: Optional[str] = None

    def __post_init__(self):
        turn_id(self.turn)
        if not isinstance(self.type, TransitionType):
            raise ValueError("Internal transitions require canonical TransitionType")
        if self.event_id is not None and (not isinstance(self.event_id, str) or not self.event_id.strip()):
            raise ValueError("Nonempty event_id required")

    def to_dict(self):
        d = to_plain(asdict(self))
        if self.rule is not None:
            d["rule"] = self.rule.to_dict()
        return d

    @classmethod
    def from_dict(cls, d):
        allowed = {"op", "type", "turn", "effective_turn", "event_id", "rule_id", "target_id", "rule", "changes"}
        if set(d) - allowed:
            raise ValueError(f"Unknown event fields: {sorted(set(d) - allowed)}")
        if 'op' in d and not {'event_id', 'effective_turn'} <= d.keys():
            raise ValueError('Canonical events require event_id and effective_turn')
        for a, b in (("op", "type"), ("effective_turn", "turn")):
            if a in d and b in d and d[a] != d[b]:
                raise ValueError(f"Conflicting {a}/{b}")
        kind = operation_name(d.get("op", d.get("type")))
        changes = dict(d.get('changes') or {})
        if 'expires_at' in changes:
            if 'expiration' in changes:
                raise ValueError('Conflicting expiration aliases')
            changes['expiration'] = changes.pop('expires_at')
        return cls(turn=d.get("effective_turn", d.get("turn")), type=kind,
                   rule=Rule.from_dict(d["rule"]) if d.get("rule") else None,
                   target_id=d.get("target_id", d.get("rule_id") if kind not in (TransitionType.ADD, TransitionType.EXCEPTION) else None),
                   changes=changes, event_id=d.get("event_id"))


@dataclass(frozen=True)
class Request:
    turn: int
    data_type: Optional[str]
    recipient: Optional[str]
    purpose: Optional[str] = None
    subject: Optional[str] = None
    sender: Optional[str] = None
    scope: dict = field(default_factory=dict)
    request_time: Optional[str] = None

    def __post_init__(self):
        turn_id(self.turn)
        if not isinstance(self.scope, dict) or any(not isinstance(k, str) or not k.strip() for k in self.scope):
            raise ValueError('Extra request scope requires nonempty string keys')
        for value in self.scope_values.values():
            if value is not None and (not isinstance(value, str) or not value.strip() or value == ANY):
                raise ValueError("Request slots must be concrete strings or null")
        if any(k in SCOPE_FIELDS for k in self.scope):
            raise ValueError("Duplicate request scope slot")
        if self.request_time is not None:
            timestamp(self.request_time)

    @property
    def scope_values(self):
        return {**{f: getattr(self, f) for f in SCOPE_FIELDS}, **self.scope}

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d.pop("text", None)
        d.setdefault('data_type', None)
        d.setdefault('recipient', None)
        return cls(**d)


class State(dict):
    """규칙의 최신 불변 버전과 추가만 가능한 event 의존성 DAG를 보관한다."""
    def __init__(self, rules=(), *, events=None, edges=None, latest=None):
        super().__init__(rules)
        self.events = dict(events or {})
        self.edges = set(edges or ())
        self.latest = dict(latest or {})

    def copy(self):
        return State(self, events=self.events, edges=self.edges, latest=self.latest)


@dataclass(frozen=True)
class Gold:
    decision: Optional[Decision]
    authorization: Optional[Authorization]
    rule_id: Optional[str]
    rule_scope: Optional[dict]
    transition_type: Optional[TransitionType]
    relevant_turn: Optional[int]
    relevant_rules: tuple[str, ...]
    ambiguous: bool
    reason: str
    result: dict = field(default_factory=dict)

    def to_dict(self):
        d = to_plain(asdict(self))
        result = d.pop("result")
        return {**d, **result}
