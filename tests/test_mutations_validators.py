"""[B] mutations.make_variants와 validators의 구조 검사를 확인한다."""
from src.data.capability_matrix import CAPABILITIES
from src.data.mutations import make_variants
from src.data.validators import validate_dialogue, validate_semantics, validate_structure


def test_make_variants_produces_base_and_all_capabilities(seed):
    variants = make_variants(seed)
    assert [v["variant_id"] for v in variants] == ["base", *CAPABILITIES]
    assert len({v["sample_id"] for v in variants}) == len(variants)
    assert {v["group_id"] for v in variants} == {seed["group_id"]}
    assert all(v["gold_decision"] is None and v["gold_state"] is None for v in variants)


def test_make_variants_skips_ineligible_seed(seed):
    assert make_variants({**seed, "status": "excluded"}) == []
    assert make_variants({**seed, "recipient": ""}) == []


def test_every_generated_variant_passes_structure_check(seed):
    for variant in make_variants(seed):
        assert validate_structure(variant) == [], variant["variant_id"]


def test_structure_check_flags_broken_traces(seed):
    base, *rest = make_variants(seed)
    cap05 = next(v for v in rest if v["variant_id"] == "CAP05")
    broken = dict(cap05, operation_trace=cap05["operation_trace"][1:])  # REVOKE 없이 REINSTATE
    assert "missing_prior_revocation" in validate_structure(broken)
    assert "invalid_expected_relation" in validate_structure(dict(cap05, expected_gold_relation="INVARIANT"))
    assert "unexpected_gold" in validate_structure(dict(base, gold_decision="DENY"))


def _dialogue(n=4, operation_turns=None):
    turns = [{"turn_id": i, "role": "user" if i % 2 == 0 else "assistant", "text": f"turn {i}"} for i in range(n)]
    return {"turns": turns, "final_request": turns[-1]["text"], "operation_turns": operation_turns or {}}


def test_validate_dialogue(seed):
    cap02 = next(v for v in make_variants(seed) if v["variant_id"] == "CAP02")
    # 마지막 turn이 user여야 하므로 홀수 개 turn을 쓴다.
    assert validate_dialogue(_dialogue(5, {"CAP02_op1": 1}), cap02) == []
    assert validate_dialogue(_dialogue(5), cap02) == ["invalid_operation_coverage"]
    assert validate_dialogue(_dialogue(5, {"CAP02_op1": 4}), cap02) == ["invalid_operation_order"]
    assert validate_dialogue(_dialogue(4, {"CAP02_op1": 1}), cap02) == ["request_mismatch"]
    assert validate_dialogue(_dialogue(3), cap02) == ["invalid_turn_count"]
    assert validate_dialogue({"turns": []}, cap02) == ["invalid_output_schema"]


def test_validate_semantics():
    ok = {"aligned": True, "mismatches": [], "missing_facts": [], "added_unsupported_permissions": [], "confidence": 0.95}
    assert validate_semantics(ok) == []
    assert validate_semantics(dict(ok, aligned=False, confidence=0.5)) == ["semantic_misalignment", "low_semantic_confidence"]
    assert validate_semantics(dict(ok, mismatches=["x"])) == ["mismatches"]
    assert validate_semantics(dict(ok, confidence=2)) == ["invalid_semantic_schema"]
    assert validate_semantics({"aligned": True}) == ["invalid_semantic_schema"]
