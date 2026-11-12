"""[B] ModernBERT 기반 A/C/C+를 구현한다. 외부 annotation은 loss target으로만 쓴다."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .dataset import IGNORE_INDEX, MODEL_ID, STATE_LABELS
import json
from pathlib import Path


class ModernBertPredictor(nn.Module):
    def __init__(self, variant="A", *, model_id=MODEL_ID, revision=None,
                 encoder=None, lambda_state=0.5, mu_relevance=0.2, dropout=0.1):
        super().__init__()
        if variant not in ("A", "C", "C+"):
            raise ValueError("variant must be A, C, or C+")
        if lambda_state < 0 or mu_relevance < 0:
            raise ValueError("Loss weights must be nonnegative")
        if encoder is None:
            from transformers import AutoModel
            encoder = AutoModel.from_pretrained(model_id, revision=revision)
        self.encoder, self.variant = encoder, variant
        self.lambda_state, self.mu_relevance = lambda_state, mu_relevance
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.failure_head = nn.Linear(hidden, 1)
        self.state_heads = nn.ModuleDict({name: nn.Linear(hidden, len(classes)) for name, classes in STATE_LABELS.items()} if variant != "A" else {})
        self.relevance_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1)) if variant == "C+" else None

    def architecture(self):
        """base weights를 다시 받지 않고 모델을 복원할 수 있도록 구조 설정을 반환한다."""
        return {"encoder_config": self.encoder.config.to_dict(), "variant": self.variant,
                "lambda_state": self.lambda_state, "mu_relevance": self.mu_relevance,
                "dropout": self.dropout.p}

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        """신뢰할 수 있는 B checkpoint에서 모델을 복원한다. base weights는 내려받지 않는다."""
        from transformers import AutoConfig, AutoModel
        payload = torch.load(path, map_location="cpu", weights_only=False)
        architecture = dict(payload["architecture"])
        config = dict(architecture.pop("encoder_config"))
        model_type = config.pop("model_type")
        encoder = AutoModel.from_config(AutoConfig.for_model(model_type, **config))
        model = cls(encoder=encoder, **architecture)
        model.load_state_dict(payload["model"])
        return model.to(device).eval()

    def save(self, path):
        """모델 구조와 weights만 저장한다. 같은 tokenizer가 필요하면 별도로 보관한다."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({"architecture": self.architecture(), "model": self.state_dict()}, temporary)
        temporary.replace(path)

    @torch.no_grad()
    def predict(self, turns, final_request, tokenizer, *, max_length=2048, output_path=None):
        """history turns와 현재 request만으로 failure probability를 계산한다.

        turns에는 마지막 request를 넣지 않는다. 결과에 truncation 여부를 함께 남기며,
        request가 잘렸으면 잘못된 확률을 반환하지 않고 중단한다."""
        from .dataset import DialogueDataset, DialogueCollator
        item = DialogueDataset([{"turns": turns, "final_request": final_request}], tokenizer, max_length)[0]
        if not item["request_mask"]:
            raise ValueError("Request was truncated; increase the common max_length")
        batch = DialogueCollator(tokenizer)([item])
        device = next(self.parameters()).device
        inputs = {key: value.to(device) for key, value in batch.items()
                  if key in ("input_ids", "attention_mask", "turn_spans", "turn_mask", "request_span", "request_mask")}
        training = self.training
        self.eval()
        try:
            probability = float(self(**inputs)["failure_logits"].sigmoid()[0])
        finally:
            self.train(training)
        result = {"failure_probability": probability, "truncated": item["diagnostics"]["truncated"]}
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, allow_nan=False) + "\n", encoding="utf-8")
        return result

    @staticmethod
    def pool_turns(hidden, spans, attention_mask):
        """prefix sum으로 반개구간 span의 평균을 구한다. padding token은 제외한다."""
        if spans.ndim != 3 or spans.shape[-1] != 2 or spans.shape[0] != hidden.shape[0]:
            raise ValueError("Expected spans [batch, turns, 2]")
        start, end = spans.unbind(-1)
        if ((start < 0) | (end < start) | (end > hidden.shape[1])).any():
            raise ValueError("Invalid half-open token span")
        prefix = F.pad((hidden * attention_mask.unsqueeze(-1)).cumsum(1), (0, 0, 1, 0))
        counts = F.pad(attention_mask.cumsum(1), (1, 0))
        def gather(indices):
            return prefix.gather(1, indices.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
        count = counts.gather(1, end) - counts.gather(1, start)
        return (gather(end) - gather(start)) / count.clamp_min(1).unsqueeze(-1), count > 0

    def extract_cls(self, input_ids, attention_mask):
        """마지막 layer의 CLS를 반환한다. 고정된 표현을 뽑을 때는 호출부에서 eval/no_grad를 설정한다."""
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]

    def forward(self, input_ids, attention_mask, *, turn_spans=None, turn_mask=None,
                request_span=None, request_mask=None, failure_labels=None,
                state_labels=None, relevance_labels=None, diagnostics=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls = hidden[:, 0]
        features = self.dropout(cls)
        failure = self.failure_head(features).squeeze(-1)
        states = {name: head(features) for name, head in self.state_heads.items()}
        relevance, valid = None, None
        if self.relevance_head is not None:
            if turn_spans is None or request_span is None:
                raise ValueError("C+ requires turn and request spans")
            pooled, valid = self.pool_turns(hidden, turn_spans, attention_mask)
            request, request_valid = self.pool_turns(hidden, request_span.unsqueeze(1), attention_mask)
            valid = valid & request_valid
            if turn_mask is not None:
                valid = valid & turn_mask
            if request_mask is not None:
                valid = valid & request_mask.unsqueeze(1)
            relevance = self.relevance_head(self.dropout(torch.cat((pooled, request.expand(-1, pooled.shape[1], -1)), dim=-1))).squeeze(-1)
        zero = failure.sum() * 0
        losses = {"failure": zero, "state": zero, "relevance": zero}
        if failure_labels is not None:
            mask = failure_labels != IGNORE_INDEX
            if mask.any():
                losses["failure"] = F.binary_cross_entropy_with_logits(failure[mask], failure_labels[mask].to(failure.dtype))
        if states and state_labels is not None:
            per_head = []
            for name, logits in states.items():
                labels = state_labels.get(name)
                if labels is not None:
                    mask = labels != IGNORE_INDEX
                    if mask.any():
                        per_head.append(F.cross_entropy(logits[mask], labels[mask]))
            if per_head:
                losses["state"] = torch.stack(per_head).mean()
        if relevance is not None and relevance_labels is not None:
            mask = valid & (relevance_labels != IGNORE_INDEX)
            if mask.any():
                losses["relevance"] = F.binary_cross_entropy_with_logits(relevance[mask], relevance_labels[mask].to(relevance.dtype))
        loss = losses["failure"] + self.lambda_state * losses["state"] + self.mu_relevance * losses["relevance"]
        return {"loss": loss, "losses": losses, "failure_logits": failure,
                "state_logits": states, "relevance_logits": relevance,
                "relevance_mask": valid, "cls_representation": cls}
