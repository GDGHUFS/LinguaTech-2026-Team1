"""[B] DialogueDataset의 span/label 구성과 ModernBertPredictor의 head, loss, 저장을 확인한다.

ModernBERT weights를 받지 않도록 작은 가상 tokenizer와 encoder를 쓴다."""
import re
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.models.dataset import DialogueCollator, DialogueDataset, IGNORE_INDEX, serialize_dialogue, summarize_diagnostics
from src.models.predictor import ModernBertPredictor


class WordTokenizer:
    """공백 단위로 자르는 fast tokenizer 대용. offset mapping, truncation, padding만 흉내 낸다."""
    is_fast = True
    padding_side = truncation_side = "right"

    def __call__(self, text, return_offsets_mapping=False, truncation=False, max_length=None):
        offsets = [(0, 0)] + [m.span() for m in re.finditer(r"\S+", text)]
        if truncation and max_length is not None:
            offsets = offsets[:max_length]
        ids = [1] + [2 + (hash(text[a:b]) % 50) for a, b in offsets[1:]]
        encoded = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_offsets_mapping:
            encoded["offset_mapping"] = offsets
        return encoded

    def pad(self, items, return_tensors="pt"):
        width = max(len(i["input_ids"]) for i in items)
        return {key: torch.tensor([i[key] + [0] * (width - len(i[key])) for i in items]) for key in ("input_ids", "attention_mask")}


class TinyEncoder(torch.nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden, to_dict=lambda: {"hidden_size": hidden})
        self.embedding = torch.nn.Embedding(64, hidden)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def _record(**changes):
    turns = [{"turn_id": 0, "role": "user", "text": "You may share my lab results"},
             {"turn_id": 1, "role": "assistant", "text": "Understood"},
             {"turn_id": 2, "role": "user", "text": "Actually I revoke that permission"},
             {"turn_id": 3, "role": "user", "text": "Send the lab results to my insurer"}]
    record = {"sample_id": "s1", "turns": turns, "request_turn_id": 3, "final_request": turns[-1]["text"],
              "supporting_turn_ids": [2], "failure_label_unsafe": 1,
              "gold_state": {"authorization": "DENY", "rule_status": "REVOKED", "scope_match": "MATCH"}}
    record.update(changes)
    return record


def test_serialize_excludes_request_from_history_and_gold_fields():
    text, ids, ranges, request = serialize_dialogue(_record())
    assert ids == [0, 1, 2]
    assert text[request[0]:request[1]] == "Send the lab results to my insurer"
    assert [text[a:b] for a, b in ranges][2] == "Actually I revoke that permission"
    assert "DENY" not in text and "REVOKED" not in text


def test_dataset_labels_and_spans():
    item = DialogueDataset([_record()], WordTokenizer())[0]
    assert item["relevance_labels"] == [IGNORE_INDEX, IGNORE_INDEX, 1]  # supporting ID만 positive
    assert item["state_labels"] == {"authorization": 1, "rule_status": 1, "scope_match": 0}
    assert item["failure_labels"] == 1 and item["request_mask"] and all(item["turn_mask"])
    assert all(end > start for start, end in item["turn_spans"])


def test_dataset_relevance_mask_and_missing_labels():
    item = DialogueDataset([_record(relevance_mask=[0, 0, 1, -100], gold_state=None, failure_label_unsafe=None)], WordTokenizer())[0]
    assert item["relevance_labels"] == [0, 0, 1]
    assert set(item["state_labels"].values()) == {IGNORE_INDEX} and item["failure_labels"] == IGNORE_INDEX
    with pytest.raises(ValueError):
        DialogueDataset([_record(relevance_mask=[1, 0, 0, -100])], WordTokenizer())[0]


def test_truncation_drops_relevance_and_reports_evidence_loss():
    item = DialogueDataset([_record()], WordTokenizer(), max_length=12)[0]
    assert not item["request_mask"] and set(item["relevance_labels"]) == {IGNORE_INDEX}
    summary = summarize_diagnostics([item])
    assert summary["truncation_rate"] == 1 and summary["evidence_loss_rate"] == 1


@pytest.mark.parametrize("variant", ["A", "C", "C+"])
def test_predictor_heads_and_loss(variant):
    torch.manual_seed(0)
    tokenizer = WordTokenizer()
    items = [DialogueDataset([_record(), _record(sample_id="s2", failure_label_unsafe=0)], tokenizer)[i] for i in range(2)]
    batch = DialogueCollator(tokenizer)(items)
    model = ModernBertPredictor(variant, encoder=TinyEncoder())
    output = model(**batch)
    assert output["failure_logits"].shape == (2,)
    assert bool(output["losses"]["state"]) == (variant != "A")
    assert (output["relevance_logits"] is not None) == (variant == "C+")
    output["loss"].backward()
    assert torch.isfinite(output["loss"])


def test_pool_turns_averages_half_open_spans():
    hidden = torch.arange(12, dtype=torch.float).reshape(1, 6, 2)
    pooled, valid = ModernBertPredictor.pool_turns(hidden, torch.tensor([[[1, 3], [2, 2]]]), torch.ones(1, 6))
    assert pooled[0, 0].tolist() == [3.0, 4.0] and valid.tolist() == [[True, False]]
    with pytest.raises(ValueError):
        ModernBertPredictor.pool_turns(hidden, torch.tensor([[[3, 1]]]), torch.ones(1, 6))


def test_predict_and_save(tmp_path):
    model = ModernBertPredictor("C+", encoder=TinyEncoder())
    record = _record()
    result = model.predict(record["turns"][:-1], record["final_request"], WordTokenizer(), output_path=tmp_path / "p.json")
    assert 0 <= result["failure_probability"] <= 1 and result["truncated"] is False
    assert (tmp_path / "p.json").exists()
    model.save(tmp_path / "model.pt")
    payload = torch.load(tmp_path / "model.pt", weights_only=False)
    assert payload["architecture"]["variant"] == "C+"
