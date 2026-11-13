"""[B] train.py의 설정 검사, tuning plan, validation 지표, 입력 검사를 확인한다."""
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

import train

ROOT = Path(__file__).resolve().parents[1]


def _config(**changes):
    config = yaml.safe_load((ROOT / "configs/train.yaml").read_text(encoding="utf-8"))
    config.update(changes)
    return config


def test_repository_train_config_is_valid():
    train.validate_config(_config())


@pytest.mark.parametrize("changes", [{"max_epochs": 4}, {"effective_batch_size": 32}, {"optimizer": "SGD"}, {"world_size": 2}])
def test_invalid_train_config(changes):
    with pytest.raises(ValueError):
        train.validate_config(_config(**changes))


def test_tuning_plan_gives_every_mode_the_same_budget():
    config = _config()
    plans = {mode: train.tuning_plan(mode, config, 42, tune=True) for mode in train.MODES}
    assert {len(plan) for plan in plans.values()} == {config["tuning_trials"]}
    assert {mu for _, mu in plans["C"]} == {config["mu_relevance"]}
    assert train.tuning_plan("C_plus", config, 42) == [(config["lambda_state"], config["mu_relevance"])]
    assert train.tuning_plan("C_plus", config, 42, True) == train.tuning_plan("C_plus", config, 42, True)


def test_validation_metrics():
    perfect = train.validation_metrics([1, 0, 1, 0], [.9, .1, .8, .2])
    assert perfect == {"validation_auprc": 1.0, "validation_brier": pytest.approx(.025)}
    # 같은 score는 한 묶음으로 계산한다.
    assert train.validation_metrics([1, 0], [.5, .5])["validation_auprc"] == .5
    assert train.validation_metrics([1, 0, -100], [.9, .1, .3])["validation_auprc"] == 1.0
    with pytest.raises(ValueError):
        train.validation_metrics([1, 1], [.9, .8])


def test_is_better_prefers_auprc_then_lower_brier():
    a = {"validation_auprc": .8, "validation_brier": .2}
    assert train.is_better(a, None)
    assert train.is_better({"validation_auprc": .8, "validation_brier": .1}, a)
    assert not train.is_better({"validation_auprc": .7, "validation_brier": .0}, a)


def _record(**changes):
    turns = [{"turn_id": 0, "role": "user", "text": "hi"}, {"turn_id": 1, "role": "user", "text": "send it"}]
    record = {"sample_id": "s1", "source_dataset": "CoPriva", "source_provenance": {"r": 1}, "source_seed_id": "seed-1",
              "group_id": "CoPriva:seed-1", "data_kind": "real_derived", "turns": turns, "split": "train",
              "validation_status": "semantic_pass_oracle_pending", "gold_decision": "DENY",
              "target_decision": "ALLOW", "parse_status": "VALID", "failure_label_unsafe": 1}
    record.update(changes)
    return record


def _write(tmp_path, records, name="train.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_read_training_records_accepts_consistent_records(tmp_path):
    assert len(train.read_training_records(_write(tmp_path, [_record()]), "train")) == 1


@pytest.mark.parametrize("changes", [
    {"split": "id_test"}, {"turns": []}, {"data_kind": "mock"}, {"failure_label_unsafe": 0},
    {"validation_status": "rejected"}, {"group_id": None}, {"gold_decision": None, "failure_label_unsafe": None},
])
def test_read_training_records_rejects_inconsistent_records(tmp_path, changes):
    with pytest.raises(ValueError):
        train.read_training_records(_write(tmp_path, [_record(**changes)]), "train")


def test_read_training_records_refuses_test_files(tmp_path):
    with pytest.raises(ValueError):
        train.read_training_records(_write(tmp_path, [_record()], "id_test.jsonl"), "train")
