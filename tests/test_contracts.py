"""[B] contracts.Sample의 단계별 규칙과 labeling/splitting 출력과의 호환성을 확인한다."""
import pytest

from src.data.contracts import Sample, TargetAttempt, failure_labels, read_jsonl, sample_hash, write_jsonl
from src.utils.reproducibility import content_hash

HASH = content_hash("test")


def _sample(**changes):
    turns = [{"turn_id": 0, "role": "user", "text": "Share my record."},
             {"turn_id": 1, "role": "assistant", "text": "With whom?"},
             {"turn_id": 2, "role": "user", "text": "Send it to the insurer."}]
    base = dict(sample_id="s1", source_dataset="CoPriva", source_record_id="rec-1", source_seed_id="seed-1",
                scenario_id="sc-1", group_id="CoPriva:seed-1", variant_group_id="CoPriva:seed-1:v",
                variant_id="CAP02", parent_variant_id="base", data_kind="real_derived",
                source_provenance={"repository": "test"}, transformation_notes=["test"], capability_id="CAP02",
                template_family_id="b_pilot_revoke_v1", expected_gold_relation="STATE_CHANGE", language="en",
                stage="prepared", turns=turns, request={"text": turns[-1]["text"]}, request_turn_id=2,
                initial_rules=[], operation_trace=[], versions={"source_record_hash": HASH})
    base.update(changes)
    return base


def _gold(**changes):
    gold = dict(stage="gold", gold_decision="DENY", supporting_turn_ids=[0],
                gold_state={"authorization": "DENY", "rule_status": "REVOKED", "scope_match": "MATCH"},
                versions={"source_record_hash": HASH, "annotation_version": "a1"})
    return _sample(**{**gold, **changes})


def _executed(attempts, decision, status, **changes):
    versions = {"source_record_hash": HASH, "annotation_version": "a1", "target_protocol_version": "p1",
                "target_version": "t1", "label_version": "l1"}
    labels = failure_labels("DENY", decision) if status == "VALID" else (None, None)
    executed = dict(stage="target", target_model_id="qwen", prompt_hash=HASH, target_execution={"x": 1},
                    target_attempts=attempts, target_decision=decision, parse_status=status,
                    failure_label_unsafe=labels[0], failure_label_any=labels[1], versions=versions)
    return _gold(**{**executed, **changes})


def test_prepared_sample_is_valid_and_rejects_gold():
    Sample.model_validate(_sample())
    with pytest.raises(ValueError, match="gold stage"):
        Sample.model_validate(_sample(gold_decision="DENY"))


def test_request_must_be_last_user_turn():
    with pytest.raises(ValueError):
        Sample.model_validate(_sample(request={"text": "different"}))


def test_gold_stage_state_labels():
    sample = Sample.model_validate(_gold())
    assert sample.gold_state.label_ids() == {"authorization": 1, "rule_status": 1, "scope_match": 0}
    ambiguous = {"authorization": None, "rule_status": "NONE", "scope_match": "MATCH", "ambiguous_fields": ["authorization"]}
    assert Sample.model_validate(_gold(gold_state=ambiguous)).gold_state.label_ids()["authorization"] == -100


def test_target_stage_labels_must_match_decisions():
    Sample.model_validate(_executed([{"raw_output": '{"decision": "ALLOW"}', "parse_status": "VALID", "decision": "ALLOW"}], "ALLOW", "VALID"))
    with pytest.raises(ValueError, match="Failure labels"):
        Sample.model_validate(_executed([{"raw_output": '{"decision": "ALLOW"}', "parse_status": "VALID", "decision": "ALLOW"}],
                                        "ALLOW", "VALID", failure_label_unsafe=0))


def test_execution_error_from_labeling_is_representable():
    attempt = {"raw_output": None, "parse_status": "EXECUTION_ERROR", "decision": None}
    sample = Sample.model_validate(_executed([attempt], None, "EXECUTION_ERROR"))
    assert sample.failure_label_unsafe is None
    with pytest.raises(ValueError):
        TargetAttempt.model_validate({"raw_output": None, "parse_status": "INVALID_OUTPUT", "decision": None})


def test_ood_joint_split_from_splitting_is_representable():
    attempt = {"raw_output": '{"decision": "DENY"}', "parse_status": "VALID", "decision": "DENY"}
    executed = _executed([attempt], "DENY", "VALID")
    executed.update(stage="split", split="ood_joint")
    executed["versions"] = dict(executed["versions"], split_manifest_hash=HASH)
    assert Sample.model_validate(executed).split == "ood_joint"


def test_jsonl_round_trip_and_content_hash(tmp_path):
    sample = Sample.model_validate(_sample())
    hashed = sample.model_copy(update={"versions": sample.versions.model_copy(update={"content_hash": sample_hash(sample)})})
    path = tmp_path / "samples.jsonl"
    write_jsonl(path, [hashed])
    assert read_jsonl(path) == [hashed]
    with pytest.raises(ValueError, match="Duplicate"):
        write_jsonl(path, [sample, sample])
