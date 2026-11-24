"""[B] 데이터 준비와 학습에 사용할 필드와 값의 범위를 정한다."""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from src.utils.reproducibility import content_hash

SCHEMA_VERSION = "1.0.0"
IGNORE_INDEX = -100
NonEmpty = Annotated[str, Field(strict=True, min_length=1, pattern=r".*\S.*")]
Hash = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
TurnID = Annotated[int, Field(strict=True, ge=0)]
Binary = Annotated[StrictInt, Field(ge=0, le=1)]


class Decision(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    CLARIFY = "CLARIFY"


class Authorization(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    UNRESOLVED = "UNRESOLVED"


class RuleStatus(str, Enum):
    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"
    NONE = "NONE"


class ScopeMatch(str, Enum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    INCOMPLETE = "INCOMPLETE"


class Capability(str, Enum):
    CAP01 = "CAP01"
    CAP02 = "CAP02"
    CAP03 = "CAP03"
    CAP04 = "CAP04"
    CAP05 = "CAP05"
    CAP06 = "CAP06"
    CAP07 = "CAP07"
    CAP08 = "CAP08"
    CAP09 = "CAP09"


CAPABILITY_NAMES = dict(zip(Capability, ("ADD", "REVOKE", "MODIFY", "EXCEPTION", "REINSTATE", "NO_CHANGE", "PARAPHRASE", "INDIRECT", "MISSING_CONTEXT")))
LABEL_TO_ID = {enum.__name__: {value.value: i for i, value in enumerate(enum)} for enum in (Decision, Authorization, RuleStatus, ScopeMatch, Capability)}
ID_TO_LABEL = {name: {index: label for label, index in mapping.items()} for name, mapping in LABEL_TO_ID.items()}


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True, frozen=True)


class Turn(Contract):
    turn_id: TurnID
    role: Literal["system", "user", "assistant"]
    text: NonEmpty


class Request(Contract):
    text: NonEmpty
    data_type: NonEmpty | None = None
    recipient: NonEmpty | None = None
    purpose: NonEmpty | None = None
    subject: NonEmpty | None = None
    sender: NonEmpty | None = None
    scope: dict[str, Any] = Field(default_factory=dict)
    request_time: NonEmpty | None = None


class GoldState(Contract):
    authorization: Authorization | None
    rule_status: RuleStatus | None
    scope_match: ScopeMatch | None
    ambiguous_fields: list[Literal["authorization", "rule_status", "scope_match"]] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_ambiguity(self) -> GoldState:
        if any(getattr(self, field) is not None for field in self.ambiguous_fields):
            raise ValueError("Ambiguous state heads must be null, not invented classes")
        return self

    def label_ids(self) -> dict[str, int]:
        """head별 label ID를 구한다. annotation이 없는 필드는 ignore_index로 둔다."""
        return {field: IGNORE_INDEX if getattr(self, field) is None else LABEL_TO_ID[enum.__name__][getattr(self, field).value] for field, enum in (("authorization", Authorization), ("rule_status", RuleStatus), ("scope_match", ScopeMatch))}


class Versions(Contract):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    policy_semantics_version: NonEmpty | None = None
    annotation_version: NonEmpty | None = None
    generation_prompt_version: NonEmpty | None = None
    target_protocol_version: NonEmpty | None = None
    target_version: NonEmpty | None = None
    label_version: NonEmpty | None = None
    split_manifest_hash: Hash | None = None
    model_config_hash: Hash | None = None
    source_record_hash: Hash
    content_hash: Hash | None = None


class TargetAttempt(Contract):
    raw_output: str | None
    parse_status: Literal["VALID", "INVALID_OUTPUT", "EXECUTION_ERROR"]
    decision: Decision | None
    attempt: Annotated[StrictInt, Field(ge=1, le=2)] | None = None
    latency_seconds: Annotated[float, Field(ge=0)] | None = None
    error_type: NonEmpty | None = None
    oom: bool = False

    @model_validator(mode="after")
    def check_parse(self) -> TargetAttempt:
        if self.raw_output is None and self.parse_status != 'EXECUTION_ERROR':
            raise ValueError('Only execution errors may have no raw output')
        if (self.parse_status == "VALID") != (self.decision is not None):
            raise ValueError("Valid output requires a decision; invalid output must have no decision")
        if self.parse_status == "VALID":
            from src.data.labeling import parse_decision
            if parse_decision(self.raw_output) != self.decision.value:
                raise ValueError("Raw output must match the parsed decision JSON")
        return self


# 단계별 필수 주석과 관측 응답을 검증한다. 생성·검증의 부가 기록은 전달용 record에 보존한다.
class Sample(Contract):
    sample_id: NonEmpty
    source_dataset: Literal["CoPriva", "CAMBER"]
    source_record_id: NonEmpty
    source_seed_id: NonEmpty
    scenario_id: NonEmpty
    group_id: NonEmpty
    variant_group_id: NonEmpty
    variant_id: NonEmpty
    parent_variant_id: NonEmpty | None
    data_kind: Literal["real_derived", "real_derived_unlabeled", "model_generated", "mock"]
    source_provenance: dict[str, Any]
    annotation: dict[str, Any] | None = None
    source_motive: dict[str, Any] | None = None
    disclosure_purpose: dict[str, Any] | None = None
    purpose_provenance: dict[str, Any] | None = None
    generation_architecture: str | None = None
    policy_turn_plan: dict[str, Any] | None = None
    dialogue_plan_hash: str | None = None
    generation_method: Literal['template','qwen_paraphrase'] | None = None
    fallback_used: bool | None = None
    failure_reason: list[str] = Field(default_factory=list)
    qwen_attempts: list[dict[str, Any]] = Field(default_factory=list)
    qwen_validation_status: str | None = None
    policy_validation_status: str | None = None
    policy_validation: dict[str, Any] | None = None
    gold_training_eligible: bool = False
    failure_prediction_training_eligible: bool = False
    source_checks: dict[str, Any] | None = None
    policy_temporal_boundary: dict[str, Any] | None = None
    transformation_notes: list[NonEmpty]
    capability_id: Capability | None
    template_family_id: NonEmpty
    expected_gold_relation: Literal["INVARIANT", "DECISION_CHANGE", "STATE_CHANGE", "RECOMPUTE"] | None
    language: NonEmpty
    stage: Literal["prepared", "gold", "target", "split"]
    turns: list[Turn] = Field(min_length=1)
    request: Request
    request_turn_id: TurnID
    initial_rules: list[dict[str, Any]]
    operation_trace: list[dict[str, Any]]
    versions: Versions
    engine_version: NonEmpty | None = None
    annotation_status: Literal['VALID', 'INVALID'] | None = None
    ambiguity_flags: dict[str, Any] = Field(default_factory=dict)
    gold_decision: Decision | None = None
    gold_state: GoldState | None = None
    supporting_rule_ids: list[NonEmpty] | None = None
    supporting_turn_ids: list[TurnID] | None = None
    evidence_trace: dict[str, Any] | None = None
    relevance_mask: list[Literal[-100, 0, 1]] | None = None
    oracle_verified: bool | None = None
    human_review_status: Literal["pending", "reviewed", "rejected"] = "pending"
    target_model_id: NonEmpty | None = None
    prompt_hash: Hash | None = None
    target_execution: dict[str, Any] | None = None
    target_attempts: list[TargetAttempt] = Field(default_factory=list, max_length=2)
    target_decision: Decision | None = None
    parse_status: Literal["NOT_RUN", "VALID", "INVALID_OUTPUT", "EXECUTION_ERROR"] = "NOT_RUN"
    failure_label_unsafe: Binary | None = None
    failure_label_any: Binary | None = None
    split: Literal["train", "validation", "id_test", "ood_composition", "ood_template", "ood_joint"] | None = None

    @model_validator(mode="after")
    def validate_pipeline(self) -> Sample:
        from src.rules.schema import validate_trace_priorities
        validate_trace_priorities(self.initial_rules, self.operation_trace)
        if self.annotation_status == 'INVALID' or self.ambiguity_flags.get('unresolved_conflict'):
            raise ValueError('Invalid policy conflicts cannot enter predictor samples')
        ids = [turn.turn_id for turn in self.turns]
        if ids != sorted(set(ids)) or ids[-1] != self.request_turn_id:
            raise ValueError("Turn IDs must be unique and ordered, with the final request last")
        if self.turns[-1].text != self.request.text or self.turns[-1].role != "user":
            raise ValueError("Request must equal the final user turn")
        if self.group_id != make_group_id(self.source_dataset, self.source_seed_id):
            raise ValueError("group_id must preserve the namespaced upstream seed identity")
        if not self.source_provenance:
            raise ValueError("Source provenance cannot be empty")
        support = self.supporting_turn_ids
        if support is not None and (support != sorted(set(support)) or not set(support) <= set(ids[:-1])):
            raise ValueError("Supporting turns must be sorted unique history turn IDs")
        if self.relevance_mask is not None:
            if len(self.relevance_mask) != len(ids) or support is None:
                raise ValueError("Relevance requires aligned turn masks and evidence IDs")
            for turn_id, label in zip(ids, self.relevance_mask):
                if turn_id == self.request_turn_id and label != IGNORE_INDEX:
                    raise ValueError("The request is excluded from relevance supervision")
                if (label == 1) != (turn_id in support):
                    raise ValueError("Positive relevance must match annotated supporting turns")
        if self.stage == "prepared" and any(value is not None for value in (self.gold_decision, self.gold_state, self.supporting_rule_ids, support, self.evidence_trace, self.oracle_verified)):
            raise ValueError("Gold annotations require gold stage or later")
        if self.stage in ("prepared", "gold") and self.parse_status != "NOT_RUN":
            raise ValueError("Executed output requires target stage or later")
        if self.stage != "prepared":
            required = (self.gold_decision, self.gold_state, self.versions.annotation_version)
            if any(value is None for value in required):
                raise ValueError("Gold stage requires external decision, state and annotation version")
            if self.oracle_verified is False:
                raise ValueError("Rejected gold annotations cannot enter labeled stages")
        if self.stage in ("target", "split") and self.parse_status == "NOT_RUN":
            raise ValueError("Target stage requires a recorded execution")
        if self.parse_status == "NOT_RUN":
            if self.target_decision is not None or self.target_attempts:
                raise ValueError("Unexecuted target cannot have output")
        else:
            required = (self.target_model_id, self.prompt_hash, self.target_execution, self.versions.target_protocol_version, self.versions.target_version)
            if any(value is None for value in required) or not self.target_attempts:
                raise ValueError("Executed target requires raw attempts and execution metadata")
            last = self.target_attempts[-1]
            if (last.parse_status, last.decision) != (self.parse_status, self.target_decision):
                raise ValueError("Selected output must match the final recorded attempt")
            if len(self.target_attempts) == 2 and self.target_attempts[0].parse_status != "INVALID_OUTPUT":
                raise ValueError("Retry is only permitted after a format failure")
        labels = failure_labels(self.gold_decision, self.target_decision) if self.parse_status == "VALID" else (None, None)
        if (self.failure_label_unsafe, self.failure_label_any) != labels:
            raise ValueError("Failure labels must match valid decisions; missing output stays null")
        if self.stage in ("target", "split") and self.versions.label_version is None:
            raise ValueError("Target labels require label_version")
        if self.stage == "split" and (self.split is None or self.versions.split_manifest_hash is None):
            raise ValueError("Split stage requires partition and manifest hash")
        if self.split is not None and self.stage != "split":
            raise ValueError("Partition assignment requires split stage")
        if self.versions.content_hash is not None and self.versions.content_hash != sample_hash(self):
            raise ValueError("Sample content hash mismatch")
        return self


def make_group_id(source_dataset: str, source_seed_id: str) -> str:
    """같은 원본에서 나온 variant가 같은 group에 들어가도록 ID를 만든다."""
    return f"{source_dataset}:{source_seed_id}"


def validate_pipeline_record(record: dict, *, stage: str | None = None) -> Sample:
    """전달용 record를 엄격한 Sample 계약으로 투영해 검증한다.

    생성·검증 기록은 원본에 남기며, 실제 원본 record hash가 필요하다. 합성 입력은 data_kind=mock으로 명시하고 산출물을 tests 내부에 저장해야 한다."""
    from src.data.oracle_adapter import canonical_request
    from src.rules.engine import evaluate_policy
    from src.rules.schema import ENGINE_VERSION
    if stage is None:
        stage = 'split' if record.get('split') is not None else 'target' if record.get('parse_status','NOT_RUN') != 'NOT_RUN' else 'gold'
    if record.get('source_provenance',{}).get('fixture') and record.get('data_kind') != 'mock':
        raise ValueError('Synthetic fixture must be explicitly marked mock')
    versions = dict(record.get('versions') or {})
    versions['source_record_hash'] = record.get('source_provenance',{}).get('record_hash') or versions.get('source_record_hash')
    if record.get('target_protocol_version'):
        versions.update(target_protocol_version=record['target_protocol_version'],
            target_version=(record.get('target_execution') or {}).get('model_revision'),label_version='unsafe-any-v1')
    if record.get('split_manifest_hash'):
        versions['split_manifest_hash']=record['split_manifest_hash']
    request = canonical_request(record['request'])
    request['text'] = record.get('final_request') or (record.get('dialogue') or {}).get('final_request') or record['request'].get('text')
    payload={key:record[key] for key in Sample.model_fields if key in record}
    payload.update(stage=stage, request=request,versions=versions,
        scenario_id=record.get('scenario_id') or record['group_id'],language=record.get('language','en'))
    sample=Sample.model_validate(payload)
    if stage != 'prepared':
        if sample.oracle_verified is not True or sample.engine_version != ENGINE_VERSION:
            raise ValueError('Production gold contract requires the current verified A oracle')
        events=sample.operation_trace
        ids=[e.get('event_id') for e in events]
        turns=[e.get('effective_turn') for e in events]
        history={t.turn_id for t in sample.turns[:-1]}
        initial_ids=[r.get('rule_id') for r in sample.initial_rules]
        if any(not isinstance(i,str) or not i for i in initial_ids) or len(initial_ids)!=len(set(initial_ids)):
            raise ValueError('Unique nonempty initial rule IDs required')
        if any(type(r.get('effective_turn')) is not int or r['effective_turn'] not in history for r in sample.initial_rules):
            raise ValueError('Initial rule chronology must reference actual history turns')
        from src.data.operations import OPERATION_CLASSES
        if any(e.get('op') not in OPERATION_CLASSES for e in events):
            raise ValueError('Stored operation names must be canonical (UPDATE is boundary-only)')
        if any(not isinstance(i,str) or not i for i in ids) or len(ids)!=len(set(ids)):
            raise ValueError('Unique nonempty operation event IDs required')
        if any(type(t) is not int or t not in history for t in turns) or turns != sorted(turns):
            raise ValueError('Operation chronology must reference actual history turns')
        actual=evaluate_policy(sample.initial_rules,events,canonical_request(record['request']),sample.request_turn_id,ENGINE_VERSION)
        if actual['annotation_status']!='VALID' or any(actual[k]!=record.get(k) for k in
                ('gold_decision','gold_state','supporting_turn_ids','supporting_rule_ids','evidence_trace')):
            raise ValueError('Gold/state/relevant-turn annotations disagree with the current A oracle')
    return sample


def failure_labels(gold: Decision | None, target: Decision | None) -> tuple[int | None, int | None]:
    """기존 tuple 반환 형식을 유지하고 label 계산은 labeling 모듈에 위임한다."""
    from src.data.labeling import failure_labels as map_labels
    labels = map_labels(gold, target)
    return labels['failure_label_unsafe'], labels['failure_label_any']


def sample_hash(sample: Sample) -> str:
    """저장된 hash 값은 빼고 sample 내용의 hash를 구한다."""
    payload = sample.model_dump(mode="json")
    payload["versions"].pop("content_hash")
    return content_hash(payload)


def validate_samples(samples: list[Sample]) -> None:
    """중복 sample과 source, split 정보의 불일치를 확인한다."""
    seen = set()
    groups, variants, records = {}, {}, {}
    for sample in samples:
        Sample.model_validate(sample.model_dump(mode="json"))
        if sample.sample_id in seen:
            raise ValueError(f"Duplicate sample_id: {sample.sample_id}")
        seen.add(sample.sample_id)
        for mapping, key, value in ((groups, sample.group_id, sample.split), (variants, sample.variant_group_id, sample.group_id), (records, (sample.source_dataset, sample.source_record_id), (sample.group_id, sample.versions.source_record_hash, sample.data_kind))):
            if key in mapping and mapping[key] != value:
                raise ValueError("Group leakage or source provenance mismatch")
            mapping[key] = value


def write_jsonl(path: str | Path, samples: list[Sample]) -> None:
    """필드를 검사한 뒤 UTF-8 JSONL로 저장한다."""
    validate_samples(samples)
    Path(path).write_text("".join(sample.model_dump_json() + "\n" for sample in samples), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[Sample]:
    """JSONL을 읽고 필드를 검사한다. 오류가 있으면 해당 줄을 알려준다."""
    samples = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        try:
            samples.append(Sample.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"Invalid JSONL line {line_number}: {exc}") from exc
    validate_samples(samples)
    return samples
