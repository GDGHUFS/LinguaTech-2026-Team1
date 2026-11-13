"""[B] group 단위 split과 OOD holdout, leakage 검사를 확인한다."""
import pytest

from src.data.splitting import check_leakage, split_samples


def _row(seed_index, variant, ops=(), template="b_pilot_base_v1", source="CoPriva"):
    return {"sample_id": f"{source}-{seed_index}-{variant}", "source_dataset": source,
            "source_seed_id": f"seed-{seed_index}", "group_id": f"{source}:seed-{seed_index}",
            "template_family_id": template, "operation_trace": [{"operation": op} for op in ops]}


def _rows(n=20, start=10):
    return [_row(i, v) for i in range(start, start + n) for v in ("base", "CAP02")]


def test_variants_of_one_seed_share_a_split():
    output, manifest = split_samples(_rows())
    by_group = {}
    for row in output:
        by_group.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_group.values())
    assert manifest["leakage_check"] == "passed"


def test_split_is_deterministic_and_order_independent():
    rows = _rows()
    first, m1 = split_samples(rows, seed=7)
    second, m2 = split_samples(list(reversed(rows)), seed=7)
    assert m1["assignments"] == m2["assignments"] and m1["manifest_hash"] == m2["manifest_hash"]


def test_ratio_counts_are_by_group():
    _, manifest = split_samples(_rows(20))
    counts = {}
    for sample_id, split in manifest["assignments"].items():
        counts.setdefault(split, set()).add(sample_id.rsplit("-", 1)[0])
    assert {k: len(v) for k, v in counts.items()} == {"train": 14, "validation": 3, "id_test": 3}


def test_ood_holdouts_are_separated_first():
    rows = _rows(10, start=200) + [
        _row(100, "comp", ops=("REVOKE", "REINSTATE", "EXCEPTION")),
        _row(101, "tmpl", template="b_ood_paraphrase_v1"),
        _row(102, "both", ops=("REVOKE", "REINSTATE", "EXCEPTION"), template="b_ood_indirect_v1"),
    ]
    output, _ = split_samples(rows)
    split = {r["sample_id"]: r["split"] for r in output}
    assert split["CoPriva-100-comp"] == "ood_composition"
    assert split["CoPriva-101-tmpl"] == "ood_template"
    assert split["CoPriva-102-both"] == "ood_joint"


def test_ood_variant_pulls_its_whole_group_out_of_id():
    rows = [_row(1, "base"), _row(1, "tmpl", template="b_ood_paraphrase_v1")] + _rows(5)
    output, _ = split_samples(rows)
    assert {r["split"] for r in output if r["group_id"] == "CoPriva:seed-1"} == {"ood_template"}


def test_leakage_and_input_errors():
    rows = [dict(_row(1, "a"), split="train"), dict(_row(1, "b"), split="id_test")]
    with pytest.raises(ValueError, match="leakage"):
        check_leakage(rows)
    with pytest.raises(ValueError):
        split_samples(_rows(), ratios=(.5, .5, .5))
    with pytest.raises(ValueError):
        split_samples([_row(1, "a"), _row(1, "a")])
