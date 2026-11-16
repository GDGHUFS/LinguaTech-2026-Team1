"""[B] ModernBERT 기반 A/B_pred/B_oracle/C/C+의 실패 예측과 보조 head를 구성한다."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .dataset import IGNORE_INDEX, MODEL_ID, STATE_LABELS
import json
from pathlib import Path


class ModernBertPredictor(nn.Module):
    """ModernBERT 표현에서 실패 위험과 모델별 보조 예측을 계산한다.

    B_pred는 예측 type 확률을 실패 head에 연결한다. B_oracle은 명시적인 gold type 입력이 필요한 진단 모델이다."""
    def __init__(self, variant="A", *, model_id=MODEL_ID, revision=None,
                 encoder=None, lambda_state=0.5, mu_relevance=0.2, nu_type=0.5, dropout=0.1):
        super().__init__()
        if variant not in ("A", "B_pred", "B_oracle", "C", "C+"):
            raise ValueError("Unknown predictor variant")
        if lambda_state < 0 or mu_relevance < 0 or nu_type < 0:
            raise ValueError("Loss weights must be nonnegative")
        if encoder is None:
            from transformers import AutoModel
            encoder = AutoModel.from_pretrained(model_id, revision=revision)
        self.encoder, self.variant = encoder, variant
        self.lambda_state, self.mu_relevance = lambda_state, mu_relevance
        self.nu_type = nu_type
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.type_head = nn.Linear(hidden, 6) if variant == 'B_pred' else None
        self.failure_head = nn.Linear(hidden + (6 if variant in ('B_pred', 'B_oracle') else 0), 1)
        self.state_heads = nn.ModuleDict({name: nn.Linear(hidden, len(classes)) for name, classes in STATE_LABELS.items()} if variant in ('C', 'C+') else {})
        self.relevance_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1)) if variant == "C+" else None

    def architecture(self):
        """가중치와 함께 저장할 모델 설정을 돌려준다."""
        return {"encoder_config": self.encoder.config.to_dict(), "variant": self.variant,
                "lambda_state": self.lambda_state, "mu_relevance": self.mu_relevance,
                "nu_type": self.nu_type,
                "dropout": self.dropout.p}

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        """checkpoint의 설정과 가중치로 모델을 복원한다. 기본 가중치를 다시 내려받지 않는다. 신뢰할 수 있는 pickle 파일만 전달해야 한다."""
        from transformers import AutoConfig, AutoModel
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get('implementation_contract') != 'spec-2026-11-15':
            raise ValueError('Legacy checkpoint needs explicit revalidation; conformance is not assumed')
        architecture = dict(payload["architecture"])
        config = dict(architecture.pop("encoder_config"))
        model_type = config.pop("model_type")
        encoder = AutoModel.from_config(AutoConfig.for_model(model_type, **config))
        model = cls(encoder=encoder, **architecture)
        model.load_state_dict(payload["model"])
        return model.to(device).eval()

    def save(self, path):
        """모델 설정과 가중치를 저장한다. tokenizer는 포함하지 않는다."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({"architecture": self.architecture(), "model": self.state_dict(), "implementation_contract": "spec-2026-11-15"}, temporary)
        temporary.replace(path)

    @torch.no_grad()
    def predict(self, turns, final_request, tokenizer, *, max_length=2048, output_path=None):
        """이전 대화와 마지막 요청에서 실패 확률과 입력 잘림 여부를 반환한다.

        turns에는 마지막 요청을 제외한 대화만 넣는다. 마지막 요청이 잘리면 ValueError로 중단한다. output_path를 지정하면 결과 JSON을 저장하고 기존 파일을 덮어쓴다."""
        from .dataset import DialogueDataset, DialogueCollator
        if self.variant == 'B_oracle':
            raise ValueError('B_oracle requires a separate gold-feature diagnostic; unavailable in normal inference')
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
        """각 span의 token 평균을 구한다. padding은 제외하고, span의 끝 index는 포함하지 않는다."""
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
        """마지막 layer의 CLS를 돌려준다. 학습 없이 추출하려면 호출할 때 eval과 no_grad를 사용한다."""
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]

    def forward(self, input_ids, attention_mask, *, turn_spans=None, turn_mask=None,
                request_span=None, request_mask=None, failure_labels=None,
                state_labels=None, relevance_labels=None, type_labels=None,
                oracle_type_features=None, diagnostics=None):
        """입력 token과 span에서 logit, 표현, loss를 반환한다.

        일반 모델의 label 인자는 loss 계산에만 사용한다. B_oracle만 [batch, 6] gold type 입력을 요구하며 C+는 대화와 요청 span이 필요하다. failure_logits는 [batch], state_logits는 head별 [batch, classes], relevance_logits는 C+에서 [batch, turns]다."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls = hidden[:, 0]
        features = self.dropout(cls)
        type_logits = self.type_head(features) if self.type_head is not None else None
        if oracle_type_features is not None and self.variant != 'B_oracle':
            raise ValueError('Gold type features are diagnostic-only')
        if self.variant == 'B_oracle':
            if oracle_type_features is None or oracle_type_features.shape != (features.shape[0], 6):
                raise ValueError('B_oracle requires explicit six-dimensional gold diagnostic features')
            if not ((oracle_type_features == 0) | (oracle_type_features == 1)).all():
                raise ValueError('Invalid gold type vector')
            failure_features = torch.cat((features, oracle_type_features.to(features.dtype)), -1)
        elif type_logits is not None:
            failure_features = torch.cat((features, type_logits.sigmoid()), -1)
        else:
            failure_features = features
        failure = self.failure_head(failure_features).squeeze(-1)
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
        losses = {"failure": zero, "state": zero, "relevance": zero, "type": zero}
        if type_logits is not None and type_labels is not None:
            mask = type_labels != IGNORE_INDEX
            if mask.any():
                losses['type'] = F.binary_cross_entropy_with_logits(type_logits[mask], type_labels[mask].to(type_logits.dtype))
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
        loss = losses["failure"] + self.lambda_state * losses["state"] + self.mu_relevance * losses["relevance"] + self.nu_type * losses['type']
        return {"loss": loss, "losses": losses, "failure_logits": failure,
                "state_logits": states, "relevance_logits": relevance,
                "relevance_mask": valid, "cls_representation": cls, "type_logits": type_logits}
