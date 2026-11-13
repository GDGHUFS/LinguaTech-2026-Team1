"""[데모] 리더보드 저장소. 여러 관람객이 동시에 접속해도 깨지지 않도록 파일 lock을 잡고 JSON에 쓴다.

점수는 predictor 위험도가 아니라 실제 위반(gold DENY인데 Target이 ALLOW) 여부로만 매긴다.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path

PATH = Path(os.environ.get("CONSHIFT_LEADERBOARD", Path(__file__).resolve().parents[1] / "outputs/demo/leaderboard.json"))


def clean_nickname(name: str) -> str:
    return re.sub(r"[^\w가-힣 .\-]", "", name or "").strip()[:16]


@contextmanager
def _locked():
    PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PATH.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _read() -> list[dict]:
    try:
        return json.loads(PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def record(nickname: str, scenario_id: str, success: bool, risk: float, request: str) -> None:
    with _locked():
        entries = _read()
        entries.append({"nickname": nickname, "scenario_id": scenario_id, "success": success,
                        "risk": round(risk, 4), "request": request[:300], "time": time.time()})
        temporary = PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
        temporary.replace(PATH)


def standings(limit: int = 10) -> list[dict]:
    """사람별로 뚫은 시나리오 수 → 성공 횟수 → 시도 횟수(적을수록) → 먼저 달성한 순으로 정렬한다."""
    with _locked():
        entries = _read()
    people: dict[str, dict] = {}
    for e in entries:
        p = people.setdefault(e["nickname"], {"nickname": e["nickname"], "broken": set(), "wins": 0,
                                              "attempts": 0, "last_win": float("inf"), "stealth": None})
        p["attempts"] += 1
        if e["success"]:
            p["wins"] += 1
            p["broken"].add(e["scenario_id"])
            p["last_win"] = min(p["last_win"], e["time"]) if p["last_win"] != float("inf") else e["time"]
            p["stealth"] = e["risk"] if p["stealth"] is None else min(p["stealth"], e["risk"])
    rows = sorted(people.values(), key=lambda p: (-len(p["broken"]), -p["wins"], p["attempts"], p["last_win"]))
    return [{**p, "broken": len(p["broken"])} for p in rows[:limit]]


def totals() -> dict:
    with _locked():
        entries = _read()
    return {"attempts": len(entries), "breaches": sum(e["success"] for e in entries),
            "players": len({e["nickname"] for e in entries})}
