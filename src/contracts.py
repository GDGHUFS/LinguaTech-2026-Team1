"""Shared strict I/O, provenance and immutable artifact contracts."""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import yaml

ROOT = Path(__file__).resolve().parents[1]


class Blocked(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if Path(temp).exists():
            Path(temp).unlink()


def write_json(path, obj, immutable=False):
    if immutable and Path(path).exists():
        if digest(read_json(path)) != digest(obj):
            raise Blocked(f'Immutable manifest differs: {path}')
        return
    atomic_text(path, json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_jsonl(path, rows):
    atomic_text(path, ''.join(canonical(r) + '\n' for r in rows))


def config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if cfg['seed'] != 42 or cfg['training']['seed'] != 42:
        raise Blocked('Only seed 42 is authorized')
    t = cfg['target']
    fraction = t.get('gpu_memory_fraction')
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not 0 < fraction <= 1):
        raise Blocked('gpu_memory_fraction must be in (0, 1]')
    if (not t['freeze'] or t['do_sample'] or t['batch_size'] != 1
            or t['trust_remote_code'] or len(t['revision']) != 40):
        raise Blocked('Invalid frozen Target contract')
    expected = {'model_id':'Qwen/Qwen3-4B-Instruct-2507', 'dtype':'bfloat16',
                'max_new_tokens':1024, 'max_context_tokens':57344, 'use_cache':True,
                'system_prompt':None}
    if any(t.get(k) != v for k,v in expected.items()):
        raise Blocked('Target configuration differs from registered protocol')
    expected_training = {'optimizer':'Adam', 'learning_rate':.001, 'batch_size':64,
        'max_epochs':200, 'patience':10, 'l2_lambda':.0001,
        'l2_candidates':[.0001,.001,.01,.1], 'class_weighting':False,
        'oversampling':False, 'scheduler':None, 'gradient_clipping':None,
        'precision':'float32', 'selection_metric':'validation_ap',
        'selection_tiebreaker':'validation_brier', 'early_stopping_min_delta':0.001}
    if any(cfg['training'].get(k) != v for k,v in expected_training.items()):
        raise Blocked('Training configuration differs from registered protocol')
    if cfg['evaluation'] != {'bootstrap_repetitions':2000, 'bootstrap_seed':42, 'ece_bins':10}:
        raise Blocked('Evaluation configuration differs from registered protocol')
    for key, rel in cfg['paths'].items():
        cfg['paths'][key] = str((ROOT / rel).resolve())
    return cfg


def paths(cfg):
    return Path(cfg['paths']['data']), Path(cfg['paths']['output'])


def code_hash():
    return digest({p.relative_to(ROOT).as_posix(): file_hash(p)
                   for p in sorted((ROOT / 'src').glob('*.py'))})


def verify_manifest(manifest):
    body = {k: v for k, v in manifest.items() if k != 'manifest_hash'}
    if manifest.get('manifest_hash') != digest(body):
        raise Blocked('Manifest hash mismatch')


def gate(cfg, name, status, **details):
    _, output = paths(cfg)
    path = output / 'reports/progress.json'
    current = read_json(path) if path.exists() else {}
    current[name] = {'status': status, **details}
    write_json(path, current)
