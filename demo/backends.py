"""[데모] Target LLM과 ConShift Predictor 실행부.

실제 모델이 준비돼 있으면 실제 모델을 쓰고, 없으면 화면에 '시뮬레이션'이라고 표시되는 대체 구현을 쓴다.

- Target LLM
  - 실제: CONSHIFT_TARGET_ENDPOINT(OpenAI 호환 URL)와 CONSHIFT_TARGET_MODEL(Qwen 모델 ID)을 설정하면
    prompts/target_decision.txt와 src.data.labeling.run_target으로 Qwen을 호출한다.
  - 시뮬레이션: 멀티턴에서 흔한 실패(최근 변경을 잊음, 조건을 무시함, 예외를 넓게 적용함)를
    요청 문장에 따라 결정적으로 흉내 낸다. 실제 Qwen의 행동이 아니다.
- Predictor
  - 실제: CONSHIFT_PREDICTOR_CKPT에 train.py가 만든 best.pt 경로를 주면 ModernBertPredictor로 예측한다.
  - 휴리스틱: 학습된 모델이 아니라, 대화 구조와 요청 표현으로 위험도를 어림하는 대체 점수다.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from demo.engine import RESTRICTING, ParsedRequest, apply_operations, deciding_turn, evaluate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CUES = {
    "past_reference": (r"\b(as (she|he|they) (agreed|said|allowed)|as (agreed|before|discussed|usual)|like (before|last time)|"
                       r"already (agreed|approved|allowed|said)|previously|earlier|originally|at the start|"
                       r"the first (time|message)|same as before|still)\b", .22, "과거 허락을 떠올리게 하는 표현"),
    "urgency": (r"\b(urgent|urgently|asap|emergency|immediately|right now|right away|quickly|deadline|today)\b", .14, "긴급함을 강조하는 표현"),
    "authority": (r"\b(required|mandatory|by law|legal|official|policy requires|compliance|audit|on behalf of)\b", .12, "권위·의무를 내세우는 표현"),
    "indirect": (r"\b(could you|would you mind|just|quick|only|small|simply|kindly|forward|pass along|loop in|cc)\b", .08, "부드럽고 간접적인 표현"),
}


def cue_hits(text: str) -> list[tuple[str, float, str]]:
    lowered = text.lower()
    return [(name, weight, label) for name, (pattern, weight, label) in CUES.items() if re.search(pattern, lowered)]


def _unit(*parts: str) -> float:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def dialogue_payload(scenario, request_text: str) -> dict:
    """Target에게는 자연어 대화와 마지막 요청만 준다 (gold state·decision은 주지 않는다)."""
    turns = [{"turn_id": i, "role": t.role, "text": f"{t.speaker}: {t.text}" if t.role == "user" else t.text}
             for i, t in enumerate(scenario.turns)]
    final = f"Request: {request_text.strip()}"
    turns.append({"turn_id": len(turns), "role": "user", "text": final})
    return {"turns": turns, "final_request": final, "request_turn_id": len(turns) - 1}


@dataclass
class TargetResult:
    decision: str | None
    raw_output: str | None
    mode: str
    failure_mode: str | None = None
    error: str | None = None


class SimulatedTarget:
    label = "시뮬레이션 Target"
    is_real = False

    def decide(self, scenario, request: ParsedRequest, request_text: str) -> TargetResult:
        state = apply_operations(list(scenario.turns))
        correct = evaluate(state, request, scenario.today).decision
        changes = sum(1 for t in scenario.turns if t.operation and t.operation.op != "NO_CHANGE")
        hits = cue_hits(request_text)
        probability = min(.88, .12 + .04 * max(0, changes - 2) + sum(w for _, w, _ in hits)
                          + (.06 if len(request_text.split()) > 22 else 0))
        if _unit(scenario.scenario_id, *(t.text for t in scenario.turns), " ".join(request_text.lower().split())) >= probability:
            return TargetResult(correct, f'{{"decision": "{correct}"}}', self.label)
        modes = self._failure_modes(scenario, state, request)
        if not modes:
            return TargetResult(correct, f'{{"decision": "{correct}"}}', self.label)
        name, wrong = modes[int(_unit("mode", scenario.scenario_id, request_text) * len(modes))]
        return TargetResult(wrong, f'{{"decision": "{wrong}"}}', self.label, failure_mode=name)

    @staticmethod
    def _failure_modes(scenario, state, request) -> list[tuple[str, str]]:
        """실제 LLM에서 자주 보이는 세 가지 실수를 흉내 낸 판단 중, 정답과 다른 것만 돌려준다."""
        turns, today = list(scenario.turns), scenario.today
        correct = evaluate(state, request, today).decision
        modes = []
        last = deciding_turn(state, request)
        if last is not None and turns[last].operation.op in RESTRICTING:
            modes.append(("최근 규칙 변경을 놓침", evaluate(apply_operations(turns, skip={last}), request, today).decision))
        relaxed = apply_operations(turns)
        for s in relaxed.values():
            s.rule = replace(s.rule, purpose=None, expires_day=None)
        modes.append(("목적·기한 조건을 무시함", evaluate(relaxed, request, today).decision))
        widened = apply_operations(turns)
        for s in widened.values():
            if s.status == "ACTIVE" and s.rule.authorization == "ALLOW" and s.rule.data_type == request.data_type:
                s.rule = replace(s.rule, recipient=request.recipient, purpose=None)
        modes.append(("예외를 다른 수신자로 넓힘", evaluate(widened, request, today).decision))
        return [(name, decision) for name, decision in modes if decision != correct]


class EndpointTarget:
    """OpenAI 호환 endpoint로 Qwen을 호출한다 (예: vLLM serve)."""
    is_real = True

    def __init__(self, endpoint: str, model: str):
        from src.data.generation import QwenClient
        self.client = QwenClient(endpoint, model, key_env="CONSHIFT_TARGET_API_KEY")
        self.label = f"실제 Target · {model}"
        self.template = (ROOT / "prompts/target_decision.txt").read_text(encoding="utf-8-sig")

    def decide(self, scenario, request: ParsedRequest, request_text: str) -> TargetResult:
        from src.data.labeling import run_target

        class Runner:
            metadata = {"model_id": self.client.model, "endpoint": "configured", "temperature": 0}
            complete = staticmethod(self.client.complete)

        result = run_target(dialogue_payload(scenario, request_text), Runner(), self.template)
        attempt = result["target_attempts"][-1] if result["target_attempts"] else {}
        return TargetResult(result["target_decision"], attempt.get("raw_output"), self.label,
                            error=result.get("target_error") or (None if result["target_decision"] else "INVALID_OUTPUT"))


@dataclass
class Risk:
    probability: float
    mode: str
    factors: list[str]


class HeuristicPredictor:
    label = "휴리스틱 위험도 (학습 모델 아님)"
    is_real = False

    def predict(self, scenario, request: ParsedRequest, request_text: str) -> Risk:
        state = apply_operations(list(scenario.turns))
        factors, logit = [], -1.6
        changes = [t.operation.op for t in scenario.turns if t.operation and t.operation.op not in ("NO_CHANGE", "ADD")]
        if changes:
            logit += .45 * len(changes)
            factors.append(f"대화 중 규칙 변경 {len(changes)}회 ({', '.join(changes)})")
        last = deciding_turn(state, request)
        if last is not None:
            distance = len(scenario.turns) - last
            if scenario.turns[last].operation.op in RESTRICTING + ("REINSTATE",) and distance <= 4:
                logit += .5
                factors.append("관련 규칙이 최근에 바뀜")
        relevant = [s for s in state.values() if s.rule.data_type == request.data_type]
        if any(s.rule.purpose or s.rule.expires_day for s in relevant):
            logit += .4
            factors.append("목적·기한 조건이 걸린 규칙")
        for _, weight, label in cue_hits(request_text):
            logit += weight * 3.2
            factors.append(label)
        return Risk(1 / (1 + math.exp(-logit)), self.label, factors or ["뚜렷한 위험 신호 없음"])


class CheckpointPredictor:
    is_real = True

    def __init__(self, checkpoint: str):
        import yaml
        from src.models.dataset import load_tokenizer
        from src.models.predictor import ModernBertPredictor
        config = yaml.safe_load((ROOT / "configs/model.yaml").read_text(encoding="utf-8"))["encoder"]
        self.model = ModernBertPredictor.from_checkpoint(checkpoint)
        self.tokenizer = load_tokenizer(config["tokenizer_id"], revision=config["tokenizer_revision"])
        self.max_length = config["max_length"]
        self.label = f"ConShift {self.model.variant} · {Path(checkpoint).parent.name}"

    def predict(self, scenario, request: ParsedRequest, request_text: str) -> Risk:
        payload = dialogue_payload(scenario, request_text)
        result = self.model.predict(payload["turns"][:-1], payload["final_request"], self.tokenizer, max_length=self.max_length)
        return Risk(result["failure_probability"], self.label, ["학습된 predictor 출력"])


def load_backends():
    """환경변수에 따라 실제 모델 또는 대체 구현을 고른다. 실패하면 이유와 함께 대체 구현으로 돌아간다."""
    notes = []
    target, predictor = SimulatedTarget(), HeuristicPredictor()
    endpoint, model = os.environ.get("CONSHIFT_TARGET_ENDPOINT"), os.environ.get("CONSHIFT_TARGET_MODEL")
    if endpoint and model:
        try:
            target = EndpointTarget(endpoint, model)
        except Exception as exc:  # 설정 오류여도 데모는 계속 돈다.
            notes.append(f"실제 Target을 불러오지 못해 시뮬레이션을 써요 ({type(exc).__name__}).")
    checkpoint = os.environ.get("CONSHIFT_PREDICTOR_CKPT")
    if checkpoint:
        try:
            predictor = CheckpointPredictor(checkpoint)
        except Exception as exc:
            notes.append(f"Predictor checkpoint를 불러오지 못해 휴리스틱을 써요 ({type(exc).__name__}).")
    return target, predictor, notes
