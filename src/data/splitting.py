"""[B] 연결된 원본 그룹(group)을 묶고 OOD부터 격리한다."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import random

DEFAULT_COMPOSITIONS = (('REVOKE', 'REINSTATE', 'EXCEPTION'),)
DEFAULT_TEMPLATES = ('b_ood_paraphrase_v1', 'b_ood_indirect_v1')


def identities(row):
    source = row.get('source_dataset')
    if not source:
        raise ValueError('Missing source_dataset')
    keys = [(source, field, str(row[field])) for field in
            ('source_seed_id', 'scenario_id', 'group_id', 'variant_group_id', 'seed_id', 'source_record_id') if row.get(field)]
    if not any(row.get(k) for k in ('source_seed_id','scenario_id','group_id')):
        raise ValueError('Missing upstream seed/scenario/group identity')
    return keys


def connected_groups(rows):
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for key in identities(row):
            if key in seen:
                parent[root(i)] = root(seen[key])
            seen[key] = i
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[root(i)].append(i)
    return list(groups.values())


def holdout_kind(row, compositions, templates):
    sequence = tuple(o['operation'] for o in row.get('operation_trace', []))
    composition = any(tuple(sequence[i:i+len(c)]) == tuple(c) for c in compositions for i in range(len(sequence)-len(c)+1))
    template = row.get('template_family_id') in templates
    return composition, template


def check_leakage(rows, compositions=DEFAULT_COMPOSITIONS, templates=DEFAULT_TEMPLATES):
    for group in connected_groups(rows):
        if len({rows[i]['split'] for i in group}) != 1:
            raise ValueError('Cross-split upstream group leakage')
    for row in rows:
        c, t = holdout_kind(row, compositions, templates)
        if (c or t) and not row['split'].startswith('ood_'):
            raise ValueError('OOD holdout leaked into ID')
        if c and row['split'] not in ('ood_composition','ood_joint'):
            raise ValueError('Composition assigned to wrong holdout')
        if t and row['split'] not in ('ood_template','ood_joint'):
            raise ValueError('Template assigned to wrong holdout')


def split_samples(rows, *, seed=42, ratios=(.7,.15,.15), compositions=DEFAULT_COMPOSITIONS, templates=DEFAULT_TEMPLATES):
    if len(ratios) != 3 or any(r < 0 for r in ratios) or abs(sum(ratios)-1) > 1e-9:
        raise ValueError('ID ratios must be three nonnegative values summing to one')
    if any(not c or any(not isinstance(op,str) for op in c) for c in compositions):
        raise ValueError('Invalid holdout composition')
    ids = [r['sample_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate sample_id')
    output = [dict(r) for r in rows]
    # 순서와 failure 값에 무관한 그룹 정렬(group ordering)을 사용한다.
    groups = sorted(connected_groups(rows), key=lambda g: sorted(ids[i] for i in g))
    id_groups = defaultdict(list)
    for group in groups:
        flags = [holdout_kind(rows[i], compositions, templates) for i in group]
        c, t = any(f[0] for f in flags), any(f[1] for f in flags)
        partition = 'ood_joint' if c and t else 'ood_composition' if c else 'ood_template' if t else None
        if partition:
            for i in group:
                output[i]['split'] = partition
        else:
            id_groups[rows[group[0]]['source_dataset']].append(group)
    rng = random.Random(seed)
    for source in sorted(id_groups):
        groups = id_groups[source]
        rng.shuffle(groups)
        # 최대 나머지 배분(largest remainder)으로 작은 group 수도 재현 가능하게 나눈다.
        exact = [len(groups)*r for r in ratios]
        counts = [int(n) for n in exact]
        for k in sorted(range(3), key=lambda k: (-(exact[k]-counts[k]), k))[:len(groups)-sum(counts)]:
            counts[k] += 1
        offset = 0
        for partition, count in zip(('train','validation','id_test'), counts):
            for group in groups[offset:offset+count]:
                for i in group:
                    output[i]['split'] = partition
            offset += count
    check_leakage(output, compositions, templates)
    assignments = {r['sample_id']:r['split'] for r in output}
    manifest = {'status':'provisional_not_final_research_dataset', 'seed':seed, 'id_ratios':list(ratios),
        'ood_compositions':[list(c) for c in compositions], 'ood_templates':list(templates),
        'assignment_unit':'connected_upstream_groups', 'assignments':assignments,
        'leakage_check':'passed', 'target_failure_used_for_assignment':False,
        'empty_partitions':[p for p in ('train','validation','id_test','ood_composition','ood_template','ood_joint') if p not in assignments.values()]}
    manifest['manifest_hash'] = 'sha256:' + hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    for row in output:
        row['split_manifest_hash'] = manifest['manifest_hash']
    return output, manifest
