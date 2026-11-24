"""Normalize original file-bounded EvolIF trajectories without synthesizing text."""
from collections import defaultdict
from .contracts import Blocked, digest, read_jsonl

DETERMINISTIC = {'case', 'countableItems', 'endwith', 'existence', 'forbidden',
                 'format', 'length', 'punctuation', 'startwith'}
JUDGE = {'emotion', 'reader_age', 'style'}


def normalize(folder, revision):
    files = sorted(folder.glob('dialog_*.jsonl'))
    if not files:
        raise Blocked('Official dialog files are missing')
    raw = {p.stem: read_jsonl(p) for p in files}
    parent = {t: t for t in raw}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    seen = {}
    for tid, rows in raw.items():
        if not rows or [r['turn'] for r in rows] != list(range(1, len(rows) + 1)):
            raise Blocked(f'Unverified turn sequence: {tid}')
        identities = [('full', digest(rows))]
        for r in rows:
            if not isinstance(r.get('user_query_verified'), str) or not r['user_query_verified'].strip():
                raise Blocked(f'Missing verified query: {tid}')
            identities.append(('query', digest(r['user_query_verified'])))
            # Shared source persona is a conservative family link, not a dialogue ID.
            if r.get('style', {}).get('uuid'):
                identities.append(('persona', r['style']['uuid']))
        for identity in identities:
            if identity in seen:
                union(tid, seen[identity])
            else:
                seen[identity] = tid
    families = defaultdict(list)
    for tid in raw:
        families[find(tid)].append(tid)
    groups = {tid: digest(sorted(family)) for family in families.values() for tid in family}
    result = []
    for p in files:
        for idx, r in enumerate(raw[p.stem]):
            constraints = [{'instruction_id': i['id'], 'args': i['args']} for i in r['instructions']]
            ids = [i['instruction_id'] for i in constraints]
            if len(ids) != len(set(ids)):
                raise Blocked('Duplicate active instruction ID')
            result.append({'sample_id': digest([revision, p.name, idx]),
                           'trajectory_id': p.stem, 'group_id': groups[p.stem],
                           'turn_index': r['turn'], 'source_file': p.name,
                           'source_record_index': idx, 'user_query': r['user_query_verified'],
                           'active_constraints': constraints, 'composition_signature': sorted(set(ids)),
                           'source_revision': revision})
    return result
