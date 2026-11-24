"""Response-blind family holdout with one precommitted OOD composition."""
from collections import defaultdict
import random
from .contracts import Blocked, digest, verify_manifest


def partition(groups):
    groups = sorted(groups)
    random.Random(42).shuffle(groups)
    n = len(groups)
    a, b = int(.70 * n), int(.15 * n)
    return {'train': groups[:a], 'validation': groups[a:a+b], 'id_test': groups[a+b:]}


def make_split(rows, source_hash):
    signatures = defaultdict(set)
    groups = {r['group_id'] for r in rows}
    for r in rows:
        signatures[tuple(r['composition_signature'])].add(r['group_id'])
    selected = None
    for sig in sorted(signatures, key=lambda s: (-len(signatures[s]), s)):
        if len(sig) < 2 or len(signatures[sig]) < 20:
            continue
        parts = partition(groups - signatures[sig])
        if not all(parts.values()):
            continue
        train_ids = {i for r in rows if r['group_id'] in parts['train'] for i in r['composition_signature']}
        if set(sig) <= train_ids:
            selected = sig
            break
    ood = signatures[selected] if selected else set()
    parts = partition(groups - ood)
    if not all(parts.values()):
        raise Blocked('Insufficient independent groups for three ID splits')
    parts['ood'] = sorted(ood)
    membership = {g: split for split, gs in parts.items() for g in gs}
    result = {'seed': 42, 'source_hash': source_hash, 'normalized_hash': digest(rows),
              'ood_signature': list(selected) if selected else None,
              'ood_status': 'ENABLED' if selected else 'DISABLED_INSUFFICIENT_STRUCTURE',
              'membership': membership,
              'counts': {s: {'groups': len(gs), 'fraction_all_groups': len(gs)/len(groups),
                             'trajectories': len({r['trajectory_id'] for r in rows if r['group_id'] in gs}),
                             'turns': sum(r['group_id'] in gs for r in rows)} for s, gs in parts.items()}}
    result['manifest_hash'] = digest(result)
    audit_split(rows, result)
    return result


def audit_split(rows, manifest):
    verify_manifest(manifest)
    if digest(rows) != manifest['normalized_hash']:
        raise Blocked('Normalized source changed since split')
    trajectories = defaultdict(set)
    membership = manifest['membership']
    if set(membership.values()) - {'train', 'validation', 'id_test', 'ood'}:
        raise Blocked('Unknown split assignment')
    if set(membership) != {r['group_id'] for r in rows}:
        raise Blocked('Group membership mismatch')
    selected = manifest['ood_signature']
    for r in rows:
        if r['composition_signature'] != sorted({i['instruction_id'] for i in r['active_constraints']}):
            raise Blocked('Composition signature differs from active IDs')
        split = membership[r['group_id']]
        trajectories[r['trajectory_id']].add((split, r['group_id']))
        if selected and r['composition_signature'] == selected and split != 'ood':
            raise Blocked('OOD signature leakage')
    if any(len(v) != 1 for v in trajectories.values()):
        raise Blocked('Trajectory split/family leakage')


def pilot_trajectories(rows, manifest, limit=20):
    train_groups = sorted(g for g, s in manifest['membership'].items() if s == 'train')
    random.Random(42).shuffle(train_groups)
    return [t for g in train_groups for t in sorted({r['trajectory_id'] for r in rows if r['group_id'] == g})][:limit]
