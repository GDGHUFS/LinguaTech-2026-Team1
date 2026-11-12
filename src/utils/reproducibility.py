"""[B] 내용에 따른 JSON hash와 명시적으로 호출하는 random seed 설정을 제공한다."""
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    """같은 데이터, 설정, prompt가 같은 JSON 문자열로 직렬화되도록 한다."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    """데이터, 설정 또는 prompt의 내용을 SHA-256으로 식별한다."""
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    """파일 전체를 메모리에 올리지 않고 원본 데이터나 weights의 hash를 계산한다."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def seed_everything(seed: int, *, include_torch: bool = False) -> dict[str, Any]:
    """설치된 runtime의 seed를 맞춘다. torch는 요청한 경우에만 import한다."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    random.seed(seed)
    seeded = ["python"]
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
        seeded.append("numpy")
    if include_torch:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        seeded.append("torch")
    return {"seed": seed, "seeded": seeded, "pythonhashseed_requires_process_start": True}
