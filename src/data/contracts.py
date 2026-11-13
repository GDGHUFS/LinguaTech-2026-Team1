"""[B] 데이터 준비와 학습에 사용할 필드와 값의 범위를 정한다."""
from __future__ import annotations

import json
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
    raw_output: str
    parse_status: Literal["VALID", "INVALID_OUTPUT"]
    decision: Decision | None

    @model_validator(mode="after")
    def check_parse(self) -> TargetAttempt:
        if (self.parse_status == "VALID") != (self.decision is not None):
            raise ValueError("Valid output requires a decision; invalid output must have no decision")
        if self.parse_status == "VALID":
            if json.loads(self.raw_output) != {"decision": self.decision.value}:
                raise ValueError("Raw output must match the parsed decision JSON")
        return self


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
    data_kind: Literal["real_derived"]
    source_provenance: dict[str, Any]
    transformation_notes: list[NonEmpty]
    capability_id: Capability
    template_family_id: NonEmpty
    expected_gold_relation: Literal["INVARIANT", "DECISION_CHANGE", "STATE_CHANGE", "RECOMPUTE"]
    language: NonEmpty
    stage: Literal["prepared", "gold", "target", "split"]
    turns: list[Turn] = Field(min_length=1)
    request: Request
    request_turn_id: TurnID
    initial_rules: list[dict[str, Any]]
    operation_trace: list[dict[str, Any]]
    versions: Versions
    engine_version: NonEmpty | None = None
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
    parse_status: Literal["NOT_RUN", "VALID", "INVALID_OUTPUT"] = "NOT_RUN"
    failure_label_unsafe: Binary | None = None
    failure_label_any: Binary | None = None
    split: Literal["train", "validation", "id_test", "ood_composition", "ood_template"] | None = None

    @model_validator(mode="after")
    def validate_pipeline(self) -> Sample:
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


def failure_labels(gold: Decision | None, target: Decision | None) -> tuple[int | None, int | None]:
    """gold와 Target decision을 비교해 unsafe와 any-error label을 계산한다."""
    if gold is None or target is None:
        return None, None
    return int(gold == Decision.DENY and target == Decision.ALLOW), int(gold != target)


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
