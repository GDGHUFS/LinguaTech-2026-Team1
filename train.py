"""[B] 같은 데이터와 학습 조건으로 A/C/C+를 학습한다.

checkpoint는 신뢰할 수 있는 파일만 불러와야 한다. pickle을 읽는 과정에서 코드가 실행될 수 있다."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

from src.models.dataset import DialogueDataset, DialogueCollator, IGNORE_INDEX
from src.models.dataset import load_tokenizer, summarize_diagnostics
from src.models.predictor import ModernBertPredictor

MODES = {"A": "A", "C": "C", "C_plus": "C+"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def parameter_hash(module):
    """mode별 encoder 초기값이 같은지 비교할 수 있도록 hash를 구한다."""
    checksum = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        checksum.update(name.encode())
        checksum.update(str(tensor.dtype).encode())
        checksum.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return checksum.hexdigest()


def seed_everything(seed):
    """Python, NumPy, torch의 seed를 맞추고 deterministic 연산을 사용한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def validate_config(config):
    """batch 크기와 optimizer 설정, 최대 학습 횟수를 확인한다."""
    if config["optimizer"] != "AdamW" or config["scheduler"] != "linear":
        raise ValueError("Only AdamW with linear decay is supported")
    if not 1 <= config["max_epochs"] <= 3:
        raise ValueError("max_epochs must be between 1 and 3")
    for key in ("max_epochs", "per_device_batch_size", "gradient_accumulation_steps", "tuning_trials"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["world_size"] != 1:
        raise ValueError("This independent pipeline supports one process/device")
    if config["effective_batch_size"] != config["per_device_batch_size"] * config["gradient_accumulation_steps"]:
        raise ValueError("effective_batch_size must equal micro batch times accumulation")
    if config["learning_rate"] <= 0 or config["weight_decay"] < 0 or config["max_grad_norm"] <= 0:
        raise ValueError("Invalid optimizer settings")
    if config.get("max_optimizer_steps") is not None and config["max_optimizer_steps"] < 1:
        raise ValueError("max_optimizer_steps must be positive")
    if config.get("checkpoint_every_steps") is not None and (not isinstance(config["checkpoint_every_steps"], int) or config["checkpoint_every_steps"] < 1):
        raise ValueError("checkpoint_every_steps must be a positive integer or null")


def tuning_plan(mode, config, seed, tune=False):
    """각 mode에서 실행할 lambda와 mu 조합을 고른다.

    --tune을 쓰면 mode마다 같은 seed로 tuning_trials번 학습한다.
    A는 조정할 보조 loss가 없으므로 같은 설정을 반복한다.
    C+는 lambda와 mu 조합 중 정해진 수만 고른다. test 데이터는 사용하지 않는다."""
    if mode not in MODES:
        raise ValueError("Unknown mode")
    if not tune:
        return [(config["lambda_state"], config["mu_relevance"])]
    if mode == "A":
        candidates = [(config["lambda_state"], config["mu_relevance"])] * config["tuning_trials"]
    elif mode == "C":
        candidates = [(x, config["mu_relevance"]) for x in config["lambda_candidates"]]
    else:
        candidates = list(itertools.product(config["lambda_candidates"], config["mu_candidates"]))
    random.Random(seed).shuffle(candidates)
    if config["tuning_trials"] > len(candidates):
        raise ValueError("Tuning budget exceeds candidate count; use the same valid budget for all modes")
    return candidates[:config["tuning_trials"]]


def validation_metrics(labels, scores):
    """validation의 average precision과 Brier를 계산한다. 같은 score는 묶어서 계산한다."""
    pairs = [(int(y), float(p)) for y, p in zip(labels, scores) if y != IGNORE_INDEX]
    if not pairs or {y for y, _ in pairs} != {0, 1}:
        raise ValueError("Validation selection requires observed positive and negative failure labels")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for _, p in pairs):
        raise ValueError("Nonfinite or invalid failure probability")
    positives = sum(y for y, _ in pairs)
    true_positive = total = 0
    ap = 0.0
    for _, group in itertools.groupby(sorted(pairs, key=lambda x: -x[1]), key=lambda x: x[1]):
        group = list(group)
        added = sum(y for y, _ in group)
        true_positive += added
        total += len(group)
        ap += (added / positives) * (true_positive / total)
    return {"validation_auprc": ap, "validation_brier": sum((p - y) ** 2 for y, p in pairs) / len(pairs)}


def is_better(metrics, best):
    return best is None or (metrics["validation_auprc"], -metrics["validation_brier"]) > (best["validation_auprc"], -best["validation_brier"])


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items() if k != "diagnostics"}
    return value


def _counts(batches):
    counts = {"failure": 0, "relevance": 0, "state": {}}
    for batch in batches:
        counts["failure"] += int((batch["failure_labels"] != IGNORE_INDEX).sum())
        valid = batch["turn_mask"] & batch["request_mask"].unsqueeze(1)
        counts["relevance"] += int(((batch["relevance_labels"] != IGNORE_INDEX) & valid).sum())
        for name, labels in batch["state_labels"].items():
            counts["state"][name] = counts["state"].get(name, 0) + int((labels != IGNORE_INDEX).sum())
    return counts


def normalized_losses(output, batch, counts):
    """effective batch 전체에서 label이 있는 항목 수를 세어 loss를 나눈다."""
    zero = output["failure_logits"].sum() * 0
    result = dict(failure=zero, state=zero, relevance=zero)
    mask = batch["failure_labels"] != IGNORE_INDEX
    if counts["failure"] and mask.any():
        result["failure"] = F.binary_cross_entropy_with_logits(output["failure_logits"][mask], batch["failure_labels"][mask].float(), reduction="sum") / counts["failure"]
    observed = sum(n > 0 for n in counts["state"].values())
    for name, logits in output["state_logits"].items():
        labels = batch["state_labels"][name]
        mask = labels != IGNORE_INDEX
        if mask.any():
            result["state"] = result["state"] + F.cross_entropy(logits[mask], labels[mask], reduction="sum") / counts["state"][name] / observed
    if output["relevance_logits"] is not None and counts["relevance"]:
        mask = output["relevance_mask"] & (batch["relevance_labels"] != IGNORE_INDEX)
        if mask.any():
            result["relevance"] = F.binary_cross_entropy_with_logits(output["relevance_logits"][mask], batch["relevance_labels"][mask].float(), reduction="sum") / counts["relevance"]
    return result


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    labels, scores = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        output = model(**batch)
        labels.extend(batch["failure_labels"].cpu().tolist())
        scores.extend(output["failure_logits"].sigmoid().cpu().tolist())
    return validation_metrics(labels, scores)


def save_checkpoint(path, model, optimizer, scheduler, state, metadata):
    """모델과 optimizer, scheduler, RNG 상태를 저장한다. 저장이 끝난 뒤 기존 checkpoint를 교체한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
           "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "state": state, "metadata": metadata, "rng": rng,
               "architecture": model.architecture()}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, expected_metadata=None):
    """checkpoint에서 모델과 학습 상태를 불러온다. 설정이나 데이터가 달라졌으면 resume를 중단한다."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if expected_metadata is not None and payload["metadata"] != expected_metadata:
        raise ValueError("Resume metadata mismatch (mode/config/data/revision/seed/budget)")
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    rng = payload["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return payload


def _write_history(path, history, merge=False):
    if not history:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if merge and path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            previous = list(csv.DictReader(stream))
        incoming = {(str(row["run_id"]), str(row["optimizer_step"])) for row in history}
        history = [row for row in previous if (row["run_id"], row["optimizer_step"]) not in incoming] + history
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train_model(model, train_dataset, validation_dataset, collator, config, metadata,
                output_dir, *, device="cpu", resume=None, stop_after_steps=None, metrics_path=None):
    """정해진 optimizer step 수만큼 학습한다.

    optimizer step을 마친 지점에서 저장하고 resume한다.
    stop_after_steps로 중단해도 scheduler의 전체 step 수는 바꾸지 않는다."""
    validate_config(config)
    if not len(train_dataset) or not len(validation_dataset):
        raise ValueError("Nonempty train and validation datasets are required")
    if metadata["mode"] not in MODES or model.variant != MODES[metadata["mode"]]:
        raise ValueError("Model variant and run metadata disagree")
    if (model.lambda_state, model.mu_relevance) != (config["lambda_state"], config["mu_relevance"]):
        raise ValueError("Model loss weights and training config disagree")
    validation_metrics([validation_dataset[i]["failure_labels"] for i in range(len(validation_dataset))], [.5] * len(validation_dataset))
    device = torch.device(device)
    model.to(device)
    micro = config["per_device_batch_size"]
    accumulation = config["gradient_accumulation_steps"]
    batches_per_epoch = math.ceil(len(train_dataset) / micro)
    total_steps = math.ceil(batches_per_epoch / accumulation) * config["max_epochs"]
    total_steps = min(total_steps, config.get("max_optimizer_steps") or total_steps)
    metadata = {**metadata, "config": config, "optimizer_step_budget": total_steps}
    run_id = digest(metadata)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0., 1. - step / total_steps))
    state = {"epoch": 0, "next_batch": 0, "global_step": 0, "best": None, "history": [], "runtime_seconds": 0.}
    if resume:
        state = load_checkpoint(resume, model, optimizer, scheduler, metadata)["state"]
    output_dir = Path(output_dir)
    if not resume and any((output_dir / name).exists() for name in ("last.pt", "best.pt")):
        raise ValueError("Run already exists; use --resume or a new output directory")
    validation_loader = DataLoader(validation_dataset, batch_size=micro, collate_fn=collator, shuffle=False,
                                   generator=torch.Generator().manual_seed(metadata["seed"]))
    started, previous_runtime = time.perf_counter(), state["runtime_seconds"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(state["epoch"], config["max_epochs"]):
        order = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(metadata["seed"] + epoch)).tolist()
        first_batch = state["next_batch"] if epoch == state["epoch"] else 0
        for start in range(first_batch, batches_per_epoch, accumulation):
            if state["global_step"] >= total_steps:
                break
            batches = [collator([train_dataset[i] for i in order[b * micro:(b + 1) * micro]])
                       for b in range(start, min(start + accumulation, batches_per_epoch))]
            counts = _counts(batches)
            supervised = counts["failure"] or (model.variant != "A" and any(counts["state"].values())) or (model.variant == "C+" and counts["relevance"])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            component = dict(failure=0., state=0., relevance=0.)
            for raw in batches:
                batch = _to_device(raw, device)
                output = model(**batch)
                losses = normalized_losses(output, batch, counts)
                loss = losses["failure"] + model.lambda_state * losses["state"] + model.mu_relevance * losses["relevance"]
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                if supervised:
                    loss.backward()
                for name in component:
                    component[name] += float(losses[name].detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            state["global_step"] += 1
            next_batch = min(start + accumulation, batches_per_epoch)
            epoch_end = next_batch == batches_per_epoch
            state["epoch"], state["next_batch"] = (epoch + 1, 0) if epoch_end else (epoch, next_batch)
            state["runtime_seconds"] = previous_runtime + time.perf_counter() - started
            row = {"run_id": run_id, "data_kind": metadata["data_kind"], "mode": metadata["mode"], "seed": metadata["seed"],
                   "trial": metadata.get("trial", 0), "lambda_state": model.lambda_state, "mu_relevance": model.mu_relevance,
                   "epoch": epoch + 1, "optimizer_step": state["global_step"], "failure_loss": component["failure"],
                   "state_loss": component["state"], "relevance_loss": component["relevance"],
                   "loss": component["failure"] + model.lambda_state * component["state"] + model.mu_relevance * component["relevance"],
                   "gradient_norm": float(norm), "learning_rate": lr,
                   "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                   "runtime_seconds": state["runtime_seconds"], "validation_auprc": None, "validation_brier": None}
            state["history"].append(row)
            finished = state["global_step"] == total_steps
            if epoch_end or finished:
                metrics = validate(model, validation_loader, device)
                row.update(metrics)
                state["runtime_seconds"] = previous_runtime + time.perf_counter() - started
                row["runtime_seconds"] = state["runtime_seconds"]
                if is_better(metrics, state["best"]):
                    state["best"] = {**metrics, "step": state["global_step"]}
                    save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, state, metadata)
            interrupted = stop_after_steps is not None and state["global_step"] >= stop_after_steps
            interval = config.get("checkpoint_every_steps")
            if epoch_end or finished or interrupted or (interval and state["global_step"] % interval == 0):
                save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, state, metadata)
                _write_history(metrics_path or output_dir / "train_metrics.csv", state["history"], merge=metrics_path is not None)
            if finished or interrupted:
                return state
    return state


def read_training_records(path, partition):
    """학습할 train/validation 파일을 읽는다. 필요한 gold나 원본 정보가 없으면 중단한다."""
    if partition not in ("train", "validation") or "test" in Path(path).stem.lower():
        raise ValueError("Only train/validation inputs are allowed")
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"Empty {partition} dataset")
    for record in records:
        if record.get("split") not in (None, partition):
            raise ValueError("Input contains a different partition, including test/OOD")
        if not record.get("turns"):
            raise ValueError("Dialogue generation is incomplete; real training is blocked")
        if record.get("data_kind") not in ("real_derived", "real_derived_unlabeled"):
            raise ValueError("CLI pilot accepts real-derived records only")
        if record.get("failure_label_unsafe") in (0, 1) and record.get("gold_decision") is None:
            raise ValueError("External gold decision is required for observed failure labels")
        if record.get("oracle_verified") is False or record.get("human_review_status") == "rejected" or record.get("validation_status") in ("rejected", "failed", "not_run"):
            raise ValueError("Rejected or unvalidated records cannot enter real training")
        if record.get("failure_label_unsafe") in (0, 1):
            from src.data.labeling import failure_labels
            expected = failure_labels(record.get("gold_decision"), record.get("target_decision"))
            if record.get("parse_status") != "VALID" or record["failure_label_unsafe"] != expected["failure_label_unsafe"]:
                raise ValueError("Observed failure label requires a consistent valid target result")
        if not all(record.get(key) for key in ("sample_id", "source_dataset", "source_provenance", "source_seed_id", "group_id")):
            raise ValueError("Real training requires dataset provenance and upstream group identity")
    if not any(r.get("gold_decision") is not None and r.get("failure_label_unsafe") in (0, 1) for r in records):
        raise ValueError("No externally gold-labeled data; real training is blocked")
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--train", default="data/splits/train.jsonl")
    parser.add_argument("--validation", default="data/splits/validation.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-version", required=True)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--pilot", action="store_true", help="Write real pilot metrics CSV; does not freeze settings")
    parser.add_argument("--output-dir", default="outputs/checkpoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_config = yaml.safe_load(Path(args.model_config).read_text(encoding="utf-8"))
    validate_config(config)
    if args.resume and args.tune:
        parser.error("Resume an individual trial without --tune")
    train_records = read_training_records(args.train, "train")
    val_records = read_training_records(args.validation, "validation")
    for key in ("sample_id", "group_id"):
        if {r[key] for r in train_records if r.get(key)} & {r[key] for r in val_records if r.get(key)}:
            raise ValueError(f"Train/validation {key} overlap")
    seed_key = lambda r: (r.get("source_dataset"), r.get("source_seed_id"))
    if {seed_key(r) for r in train_records if r.get("source_seed_id")} & {seed_key(r) for r in val_records if r.get("source_seed_id")}:
        raise ValueError("Train/validation upstream seed overlap")
    encoder_config = model_config["encoder"]
    seed_everything(args.seed)
    tokenizer = load_tokenizer(encoder_config["model_id"], revision=encoder_config["revision"])
    train_dataset = DialogueDataset(train_records, tokenizer, encoder_config["max_length"])
    val_dataset = DialogueDataset(val_records, tokenizer, encoder_config["max_length"])
    # 큰 weights를 불러오기 전에 입력과 label부터 확인한다.
    train_items = [train_dataset[i] for i in range(len(train_dataset))]
    val_items = [val_dataset[i] for i in range(len(val_dataset))]
    validation_metrics([i["failure_labels"] for i in val_items], [.5] * len(val_items))
    profile = summarize_diagnostics(train_items + val_items)
    if profile["truncation_rate"] > .05 or profile["evidence_loss_rate"] not in (None, 0) or profile["incomplete_request_rate"]:
        raise ValueError("Input truncation/evidence pilot gate failed; adjust common length policy before training")
    if args.mode != "A" and not any(any(v != IGNORE_INDEX for v in i["state_labels"].values()) for i in train_items):
        raise ValueError("State supervision unavailable; C/C+ training is incomplete")
    if args.mode == "C_plus" and not any(any(v != IGNORE_INDEX for v in i["relevance_labels"]) for i in train_items):
        raise ValueError("External supporting-turn supervision unavailable; C+ training is incomplete")
    tokenizer_hash = digest(json.loads(tokenizer.backend_tokenizer.to_str()))
    best_trial, pilot_rows = None, []
    plan = tuning_plan(args.mode, config, args.seed, args.tune)
    trial_ids = list(range(len(plan)))
    if args.resume:
        saved_metadata = torch.load(args.resume, map_location="cpu", weights_only=False)["metadata"]
        plan = [(saved_metadata["config"]["lambda_state"], saved_metadata["config"]["mu_relevance"])]
        trial_ids = [saved_metadata.get("trial", 0)]
    for trial, (lambda_state, mu_relevance) in zip(trial_ids, plan):
        trial_config = {**config, "lambda_state": lambda_state, "mu_relevance": mu_relevance}
        seed_everything(args.seed)
        model = ModernBertPredictor(MODES[args.mode], model_id=encoder_config["model_id"], revision=encoder_config["revision"],
                                    lambda_state=lambda_state, mu_relevance=mu_relevance, dropout=model_config["predictor"]["dropout"])
        metadata = {"mode": args.mode, "seed": args.seed, "trial": trial, "data_kind": "real_derived", "split_version": args.split_version,
                    "train_hash": digest(train_records), "validation_hash": digest(val_records), "tokenizer_hash": tokenizer_hash,
                    "model_config": model_config, "model_revision": getattr(model.encoder.config, "_commit_hash", None) or encoder_config["revision"],
                    "encoder_initialization_hash": parameter_hash(model.encoder),
                    "libraries": {"torch": torch.__version__, "transformers": __import__("transformers").__version__},
                    "input_diagnostics": profile}
        # head 초기화에 쓰인 random 값의 수가 다르므로 학습 직전에 seed를 다시 맞춘다.
        seed_everything(args.seed)
        directory = Path(args.resume).parent if args.resume else Path(args.output_dir) / args.mode / f"seed_{args.seed}" / f"trial_{trial}"
        report = Path("outputs/reports/pilot/pilot_train_metrics.csv") if args.pilot and not args.tune else None
        state = train_model(model, train_items, val_items, DialogueCollator(tokenizer), trial_config, metadata,
                            directory, device=args.device, resume=args.resume, metrics_path=report)
        if args.pilot and args.tune:
            report = Path("outputs/reports/pilot/pilot_train_metrics.csv")
            # 이번 trial의 기록을 더하되, 다른 run의 기록은 지우지 않는다.
            pilot_rows.extend(state["history"])
            _write_history(report, pilot_rows, merge=True)
        if is_better(state["best"], best_trial):
            best_trial = {**state["best"], "checkpoint": str(directory / "best.pt"), "lambda_state": lambda_state, "mu_relevance": mu_relevance}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(json.dumps({"status": "pilot_initial", "selected_by_validation": best_trial}, ensure_ascii=False))


if __name__ == "__main__":
    main()
