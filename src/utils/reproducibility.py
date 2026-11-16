"""[B] 파일과 내용의 hash를 계산하고 난수 seed를 설정한다."""
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    """같은 내용은 항상 같은 JSON 문자열이 되도록 정렬해서 변환한다."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    """내용을 JSON으로 변환한 뒤 SHA-256 hash를 구한다."""
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    """파일을 조금씩 읽으면서 hash를 구한다. 큰 파일도 한꺼번에 메모리에 올리지 않는다."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def runtime_metadata():
    """현재 실행 환경을 기록한다. 실험 검증 여부를 나타내지는 않는다."""
    import platform
    import subprocess
    from importlib.metadata import version, PackageNotFoundError
    libraries = {}
    for name in ('numpy', 'torch', 'transformers', 'scikit-learn', 'PyYAML', 'pydantic'):
        try:
            libraries[name] = version(name)
        except PackageNotFoundError:
            libraries[name] = None
    result = subprocess.run(['git','rev-parse','HEAD'], capture_output=True, text=True, check=False)
    working = subprocess.run(['git','status','--porcelain'], capture_output=True, text=True, check=False)
    root=Path(__file__).resolve().parents[2]
    paths=[]
    for folder in ('src','scripts','configs','prompts'):
        paths.extend(p for p in (root/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.txt','.json'))
    paths.extend(p for p in root.iterdir() if p.is_file() and (p.suffix=='.py' or p.name.startswith('requirements')))
    # 소스 파일의 바이트를 포함하므로 주석만 바뀌어도 구현 hash는 달라진다.
    implementation_hash=content_hash([(p.relative_to(root).as_posix(),file_hash(p)) for p in sorted(set(paths))])
    return dict(python=platform.python_version(), platform=platform.platform(), processor=platform.processor(),
                libraries=libraries, git_commit=result.stdout.strip() if result.returncode == 0 else None,
                git_dirty=bool(working.stdout.strip()) if working.returncode==0 else None,
                implementation_hash=implementation_hash)


def seed_everything(seed: int, *, include_torch: bool = False) -> dict[str, Any]:
    """설치된 라이브러리의 seed를 맞춘다. torch는 include_torch가 True일 때만 불러온다."""
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
