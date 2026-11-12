"""[B] 대화를 직렬화하고 token span과 외부 annotation을 학습 입력으로 바꾼다."""
from __future__ import annotations

import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100
MODEL_ID = "answerdotai/ModernBERT-base"
STATE_LABELS = {
    "authorization": ("ALLOW", "DENY", "UNRESOLVED"),
    "rule_status": ("ACTIVE", "REVOKED", "EXPIRED", "NONE"),
    "scope_match": ("MATCH", "MISMATCH", "INCOMPLETE"),
}


def load_tokenizer(model_id=MODEL_ID, revision=None, **kwargs):
    """offset mapping을 지원하는 fast tokenizer를 공통 설정으로 불러온다."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True, **kwargs)
    if not tokenizer.is_fast:
        raise ValueError("Offset mapping requires a fast tokenizer")
    tokenizer.padding_side = tokenizer.truncation_side = "right"
    return tokenizer


def serialize_dialogue(record):
    """A/C/C+에 같은 marker와 vocabulary를 사용한다.

    직렬화한 text, history ID, content의 character range와 request range를 반환한다.
    request_turn_id가 있으면 마지막 turn을 request로 분리한다. 없으면 turns는 history로 본다.
    Gold, capability metadata와 Target 출력은 입력에 넣지 않는다."""
    turns = list(record["turns"])
    request = record.get("final_request")
    if request is None:
        request = (record.get("dialogue") or {}).get("final_request")
    if request is None:
        request = record["request"]["text"]
    if not isinstance(request, str) or not request.strip():
        raise ValueError("A nonempty request is required")
    if record.get("request_turn_id") is not None:
        if not turns or turns[-1]["turn_id"] != record["request_turn_id"] or turns[-1]["text"] != request or turns[-1]["role"] != "user":
            raise ValueError("request_turn_id must identify the final user request")
        turns = turns[:-1]
    text, ids, ranges = "", [], []
    for turn in turns:
        if turn["turn_id"] in ids or turn["role"] not in ("user", "assistant", "system"):
            raise ValueError("Invalid speaker or duplicate turn ID")
        if not isinstance(turn["text"], str) or not turn["text"].strip():
            raise ValueError("Turn text must be nonempty")
        text += f"[TURN] [{turn['role'].upper()}]\n"
        start = len(text)
        text += turn["text"]
        ranges.append((start, len(text)))
        ids.append(turn["turn_id"])
        text += "\n[/TURN]\n"
    text += "[REQUEST] [USER]\n"
    start = len(text)
    text += request
    return text + "\n[/REQUEST]", ids, ranges, (start, len(text))


def _overlap(offsets, bounds):
    return [i for i, (a, b) in enumerate(offsets) if b > a and a < bounds[1] and b > bounds[0]]


def _span(offsets, bounds):
    indices = _overlap(offsets, bounds)
    return (indices[0], indices[-1] + 1) if indices else (0, 0)


def summarize_diagnostics(items):
    """token length와 evidence 손실을 집계한다. annotation이 없으면 손실 여부는 결측으로 둔다."""
    diagnostics = [item["diagnostics"] for item in items]
    if not diagnostics:
        return {"count": 0, "length_percentiles": {}, "truncation_rate": None,
                "evidence_annotated_count": 0, "evidence_loss_rate": None,
                "incomplete_request_rate": None}
    lengths = torch.tensor([d["full_token_length"] for d in diagnostics], dtype=torch.float64)
    observed = [d for d in diagnostics if d["evidence_loss"] is not None]
    return {
        "count": len(diagnostics),
        "length_percentiles": dict(zip(("p50", "p90", "p95", "p99"), torch.quantile(lengths, torch.tensor([.5, .9, .95, .99], dtype=torch.float64)).tolist())),
        "truncation_rate": sum(d["truncated"] for d in diagnostics) / len(diagnostics),
        "evidence_annotated_count": len(observed),
        "evidence_loss_rate": sum(d["evidence_loss"] for d in observed) / len(observed) if observed else None,
        "incomplete_request_rate": sum(not d["request_complete"] for d in diagnostics) / len(diagnostics),
    }


class DialogueDataset(Dataset):
    """결측 gold는 masking한다. supporting ID만 있으면 positive label만 부여한다.

    negative label은 원본 turns에 정렬된 relevance_mask가 있을 때만 사용한다.
    일부만 남거나 잘린 turn은 무시하며, gold evidence로 입력 window를 고르지 않는다."""
    def __init__(self, records, tokenizer, max_length=2048, failure_field="failure_label_unsafe"):
        if isinstance(records, (str, Path)):
            records = [json.loads(line) for line in Path(records).read_text(encoding="utf-8").splitlines() if line.strip()]
        if max_length < 2 or tokenizer.padding_side != "right" or tokenizer.truncation_side != "right":
            raise ValueError("Use max_length >= 2 and right padding/truncation")
        if not tokenizer.is_fast:
            raise ValueError("A fast tokenizer is required")
        self.records, self.tokenizer = list(records), tokenizer
        self.max_length, self.failure_field = max_length, failure_field

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        text, ids, ranges, request_range = serialize_dialogue(record)
        full = self.tokenizer(text, return_offsets_mapping=True, truncation=False)
        encoded = self.tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=self.max_length)
        offsets, original = encoded.pop("offset_mapping"), full["offset_mapping"]
        def complete(bounds):
            kept = [offsets[i] for i in _overlap(offsets, bounds)]
            return bool(kept) and kept == [original[i] for i in _overlap(original, bounds)]
        spans = [_span(offsets, bounds) for bounds in ranges]
        visible = [complete(bounds) for bounds in ranges]
        request_complete = complete(request_range)
        support = record.get("supporting_turn_ids")
        if support is not None and not set(support) <= set(ids):
            raise ValueError("Supporting IDs must identify history turns")
        labels = [IGNORE_INDEX] * len(ids)
        mask = record.get("relevance_mask")
        if mask is not None:
            if support is None or len(mask) != len(record["turns"]) or any(x not in (-100, 0, 1) for x in mask):
                raise ValueError("Relevance masks require aligned external support annotations")
            by_id = {t["turn_id"]: x for t, x in zip(record["turns"], mask)}
            if record.get("request_turn_id") is not None and by_id[record["request_turn_id"]] != IGNORE_INDEX:
                raise ValueError("Request relevance must be ignored")
            labels = [by_id[i] for i in ids]
            if {i for i, x in zip(ids, labels) if x == 1} != set(support):
                raise ValueError("Positive relevance must match supporting IDs")
        elif support is not None:
            labels = [1 if i in support else IGNORE_INDEX for i in ids]
        labels = [x if ok and request_complete else IGNORE_INDEX for x, ok in zip(labels, visible)]
        state, state_labels = record.get("gold_state") or {}, {}
        for name, classes in STATE_LABELS.items():
            value = state.get(name)
            if value is None or name in state.get("ambiguous_fields", []):
                state_labels[name] = IGNORE_INDEX
            elif value in classes:
                state_labels[name] = classes.index(value)
            else:
                raise ValueError(f"Unknown {name} annotation: {value}")
        failure = record.get(self.failure_field)
        if failure is not None and failure not in (0, 1):
            raise ValueError("Failure labels must be binary or missing")
        lost = None if support is None else [i for i, ok in zip(ids, visible) if i in support and not ok]
        return {
            "input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"],
            "turn_spans": spans, "turn_mask": visible,
            "request_span": _span(offsets, request_range), "request_mask": request_complete,
            "relevance_labels": labels, "failure_labels": IGNORE_INDEX if failure is None else failure,
            "state_labels": state_labels,
            "diagnostics": {"sample_id": record.get("sample_id"), "full_token_length": len(full["input_ids"]),
                            "token_length": len(encoded["input_ids"]), "truncated": len(full["input_ids"]) > len(encoded["input_ids"]),
                            "lost_supporting_turn_ids": lost, "evidence_loss": None if lost is None else bool(lost),
                            "request_complete": request_complete, "turn_ids": ids},
        }


class DialogueCollator:
    """서로 다른 길이의 입력과 turn span을 padding해 batch로 묶는다."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, items):
        batch = dict(self.tokenizer.pad([{k: item[k] for k in ("input_ids", "attention_mask")} for item in items], return_tensors="pt"))
        width = max(len(item["turn_spans"]) for item in items)
        for key, padding, dtype in (("turn_spans", (0, 0), torch.long), ("turn_mask", False, torch.bool), ("relevance_labels", IGNORE_INDEX, torch.long)):
            shape = (len(items), width, 2) if key == "turn_spans" else (len(items), width)
            batch[key] = torch.tensor([item[key] + [padding] * (width - len(item[key])) for item in items], dtype=dtype).reshape(shape)
        batch["request_span"] = torch.tensor([item["request_span"] for item in items], dtype=torch.long)
        batch["request_mask"] = torch.tensor([item["request_mask"] for item in items], dtype=torch.bool)
        batch["failure_labels"] = torch.tensor([item["failure_labels"] for item in items], dtype=torch.float)
        batch["state_labels"] = {name: torch.tensor([item["state_labels"][name] for item in items]) for name in STATE_LABELS}
        batch["diagnostics"] = [item["diagnostics"] for item in items]
        return batch
