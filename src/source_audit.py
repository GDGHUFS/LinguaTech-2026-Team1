"""Verify local evidence against official pinned file hashes and census it."""
from collections import Counter
from .contracts import (Blocked, ROOT, digest, file_hash, paths, read_json,
                        write_json, write_jsonl, gate)
from .evolif_adapter import normalize, DETERMINISTIC, JUDGE
from .checker import OfficialChecker, MAPPING, args_valid


def audit(cfg):
    data, output = paths(cfg)
    raw = data / 'raw'
    lock = read_json(ROOT / 'configs/sources.json')
    if (lock['dataset_revision'] != cfg['source']['dataset_revision']
            or lock['github_commit'] != cfg['source']['repository_commit']):
        raise Blocked('Source revision mismatch')
    checked = []
    for f in lock['files']:
        if not f['path'].startswith(('official_data/', 'official_code/')):
            continue
        p = raw / f['path']
        if not p.is_file() or file_hash(p) != f['sha256']:
            raise Blocked(f'Source hash mismatch: {p}')
        checked.append(f)
    listed = {f['path'] for f in checked if f['path'].endswith('.jsonl') and 'dialog_v0.1/' in f['path']}
    actual = {p.relative_to(raw).as_posix() for p in (raw / 'official_data/dialog_v0.1').glob('dialog_*.jsonl')}
    if actual != listed:
        raise Blocked('Original file set differs from official lock')
    # Verify the official evaluator actually scopes history per file and reads verified queries.
    evaluator = (raw / 'official_code/src/eval.py').read_text(encoding='utf-8')
    for marker in ('history_msgs: List[Dict[str, str]] = []', 'turn.get("user_query_verified")', 'dialog_{file_id}.jsonl'):
        if marker not in evaluator:
            raise Blocked('Official trajectory evidence changed')
    rows = normalize(raw / 'official_data/dialog_v0.1', lock['dataset_revision'])
    prompt_source = 'No system message; target.system_prompt is null.'
    if cfg['target']['system_prompt'] is not None:
        raise Blocked('This experiment requires system_prompt: null')
    target_files = []
    for name in ('config.json', 'tokenizer_config.json', 'generation_config.json', 'README.md'):
        p = raw / 'target_metadata' / name
        original = next(f for f in lock['files'] if f['path'].endswith('Qwen3-4B-Instruct-2507/' + name))
        if '/resolve/' + cfg['target']['revision'] + '/' not in original['url']:
            raise Blocked('Target metadata URL is not pinned to the model revision')
        if file_hash(p) != original['sha256']:
            raise Blocked('Target metadata hash mismatch')
        target_files.append({'path': name, 'sha256': file_hash(p)})
    checker = OfficialChecker(raw / 'official_code')
    if set(checker.classes) != DETERMINISTIC:
        raise Blocked('Official deterministic checker mapping is incomplete')
    invalid = [r['sample_id'] for r in rows for c in r['active_constraints']
               if c['instruction_id'] in DETERMINISTIC and not args_valid(c['instruction_id'], c['args'])]
    if invalid:
        raise Blocked(f'Invalid official args: {len(invalid)}')
    try:
        checker.helper._get_sentence_tokenizer()
    except Exception as exc:
        raise Blocked('Install NLTK punkt and punkt_tab resources before preparing the experiment') from exc
    tids = {r['trajectory_id'] for r in rows}
    census = {'rows': len(rows), 'trajectories': len(tids), 'groups': len({r['group_id'] for r in rows}),
              'trajectory_lengths': dict(Counter(sum(r['trajectory_id'] == t for r in rows) for t in tids)),
              'deterministic_candidate_turns': sum(bool(r['active_constraints']) and set(r['composition_signature']) <= DETERMINISTIC for r in rows),
              'judge_turns': sum(bool(set(r['composition_signature']) & JUDGE) for r in rows),
              'instruction_counts': dict(Counter(i for r in rows for i in r['composition_signature']))}
    manifest = {'dataset_revision': lock['dataset_revision'], 'repository_commit': lock['github_commit'],
                'license': cfg['source']['license'], 'files': checked, 'target_files': target_files,
                'model_id': cfg['target']['model_id'], 'model_revision': cfg['target']['revision'],
                'tokenizer_revision': cfg['target']['revision'], 'census': census,
                'checker_mapping': MAPPING,
                'system_prompt_source': prompt_source, 'normalized_hash': digest(rows)}
    manifest['manifest_hash'] = digest(manifest)
    write_json(data / 'manifests/revision_lock.json', manifest, immutable=True)
    write_jsonl(data / 'normalized/turns.jsonl', rows)
    gate(cfg, 'G0', 'PASS', **census)
    return manifest
