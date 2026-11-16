"""[B] 같은 원본에서 나온 항목을 함께 묶고, OOD를 먼저 분리한다."""
from __future__ import annotations

from collections import defaultdict
from collections import Counter
import hashlib
import json
from src.utils.experiment import seed_metadata
import random

DEFAULT_COMPOSITIONS = (('REVOKE', 'REINSTATE', 'EXCEPTION'),)
DEFAULT_TEMPLATES = ('b_ood_paraphrase_v1', 'b_ood_indirect_v1', 'b_ood_question_embedding_v1')


def template_validation_scope(rows, declared_templates):
    """구현된 구조 검사 범위를 반환한다. 의미적 일반화가 검증됐다는 뜻은 아니다."""
    return dict(supported_generation_family='b_ood_question_embedding_v1',
                supported_request_style='question_embedding',
                declared_families=list(declared_templates),
                observed_eligible_ood_families=dict(Counter(r.get('template_family_id') for r in rows
                    if r.get('split') in ('ood_template', 'ood_joint') and r.get('evaluation_eligible', True))),
                checks=['connected_group_separation', 'family_id_separation',
                        'known_expression_exclusion', 'normalized_exact_request_overlap'],
                semantic_generalization_status='NOT_ESTABLISHED',
                claim_limit='Only implemented held-out families with actual validated samples; no broad paraphrase/indirect generalization claim.')


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
    """공유하는 원본 식별자를 추이적으로 연결해 분할 단위를 만든다."""
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
    sequence = operation_signature(row)
    composition = any(tuple(sequence[i:i+len(c)]) == tuple(c) for c in compositions for i in range(len(sequence)-len(c)+1))
    template = row.get('template_family_id') in templates
    return composition, template


def operation_signature(row):
    return tuple('MODIFY' if op == 'UPDATE' else op for op in
                 (o.get('op', o.get('operation')) for o in row.get('operation_trace', [])))


def check_leakage(rows, compositions=DEFAULT_COMPOSITIONS, templates=DEFAULT_TEMPLATES, fewshot_templates=()):
    """원본 group과 사전 지정한 holdout의 교차 분할 누수를 검사한다. 위반 시 ValueError를 발생시킨다."""
    for group in connected_groups(rows):
        if len({rows[i]['split'] for i in group}) != 1:
            raise ValueError('Cross-split upstream group leakage')
    from src.data.fewshots import normalized_request, uses_heldout_expression
    learned_requests = {normalized_request(r.get('final_request') or '') for r in rows
                        if r['split'] in ('train', 'validation')} - {''}
    for row in rows:
        c, t = holdout_kind(row, compositions, templates)
        text = row.get('final_request') or ''
        if t and normalized_request(text) in learned_requests:
            raise ValueError('OOD final request leaked into training/validation')
        if not t and (uses_heldout_expression(text) or
                      any(uses_heldout_expression(turn.get('text', '')) for turn in row.get('turns') or [])):
            raise ValueError('Held-out expression leaked outside OOD template')
        if (c or t) and not row['split'].startswith('ood_'):
            raise ValueError('OOD holdout leaked into ID')
        if c and row['split'] not in ('ood_composition','ood_joint'):
            raise ValueError('Composition assigned to wrong holdout')
        if t and row['split'] not in ('ood_template','ood_joint'):
            raise ValueError('Template assigned to wrong holdout')
    train = [r for r in rows if r['split'] in ('train', 'validation')]
    train_templates = {r.get('template_family_id') for r in train} | set(fewshot_templates)
    train_signatures = {operation_signature(r) for r in train}
    for row in rows:
        c, t = holdout_kind(row, compositions, templates)
        if t and row.get('template_family_id') in train_templates:
            raise ValueError('OOD template leaked into training/validation/few-shot examples')
        if c and operation_signature(row) in train_signatures:
            raise ValueError('OOD composition signature leaked into training/validation')


def split_samples(rows, *, seed=42, ratios=(.7,.15,.15), compositions=DEFAULT_COMPOSITIONS, templates=DEFAULT_TEMPLATES, fewshot_templates=()):
    """연결된 원본 group 단위로 OOD를 먼저 배정한 뒤 소스별 ID를 나눈다.

    배정에는 failure label을 사용하지 않는다. OOD group의 대조 항목은 함께 보관하되 evaluation_eligible=False로 표시한다. 복사한 record 목록과 분할 manifest를 반환한다."""
    if len(ratios) != 3 or any(r < 0 for r in ratios) or abs(sum(ratios)-1) > 1e-9:
        raise ValueError('ID ratios must be three nonnegative values summing to one')
    if any(not c or any(not isinstance(op,str) for op in c) for c in compositions):
        raise ValueError('Invalid holdout composition')
    ids = [r['sample_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate sample_id')
    output = [dict(r) for r in rows]
    # 입력 순서나 failure label이 달라도 group 배정이 같도록 먼저 정렬한다.
    groups = sorted(connected_groups(rows), key=lambda g: sorted(ids[i] for i in g))
    id_groups = defaultdict(list)
    for group in groups:
        flags = [holdout_kind(rows[i], compositions, templates) for i in group]
        c, t = any(f[0] for f in flags), any(f[1] for f in flags)
        partition = 'ood_joint' if c and t else 'ood_composition' if c else 'ood_template' if t else None
        if partition:
            for i, flag in zip(group,flags):
                output[i]['split'] = partition
                # 함께 옮긴 대조 variant는 평가에서 제외하고 실제 holdout 항목만 채점한다.
                output[i]['evaluation_eligible'] = any(flag)
        else:
            id_groups[rows[group[0]]['source_dataset']].append(group)
    rng = random.Random(seed)
    for source in sorted(id_groups):
        groups = id_groups[source]
        rng.shuffle(groups)
        # 비율대로 나눈 뒤 남는 group은 소수 부분이 큰 split부터 배정한다.
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
    check_leakage(output, compositions, templates, fewshot_templates)
    assignments = {r['sample_id']:r['split'] for r in output}
    manifest = {'experiment': seed_metadata([seed]), 'status':'provisional_not_final_research_dataset', 'seed':seed, 'id_ratios':list(ratios),
        'ood_compositions':[list(c) for c in compositions], 'ood_templates':list(templates),
        'assignment_unit':'connected_upstream_groups', 'assignments':assignments,
        'leakage_check':'passed', 'target_failure_used_for_assignment':False,
        'empty_partitions':[p for p in ('train','validation','id_test','ood_composition','ood_template','ood_joint') if p not in assignments.values()]}
    manifest.update(groups={p: sorted({r['group_id'] for r in output if r['split'] == p}) for p in set(assignments.values())},
        source_counts={p: dict(Counter(r['source_dataset'] for r in output if r['split'] == p)) for p in set(assignments.values())},
        class_prevalence={p: {label: sum(r.get(label) == 1 for r in output if r['split'] == p and r.get('evaluation_eligible',True)) / n if n else None
            for label in ('failure_label_unsafe', 'failure_label_any')
            for n in [sum(r.get(label) in (0, 1) for r in output if r['split'] == p and r.get('evaluation_eligible',True))]} for p in set(assignments.values())},
        template_families={p: sorted({r['template_family_id'] for r in output if r['split'] == p and r.get('template_family_id') and r.get('evaluation_eligible',True)}) for p in set(assignments.values())},
        operation_signatures={p: sorted({operation_signature(r) for r in output if r['split'] == p and r.get('evaluation_eligible',True)}) for p in set(assignments.values())},
        scoring_source_counts={p: dict(Counter(r['source_dataset'] for r in output if r['split']==p and r.get('evaluation_eligible',True))) for p in set(assignments.values())},
        excluded_control_ids=sorted(r['sample_id'] for r in output if r.get('evaluation_eligible') is False),
        fewshot_template_families=list(fewshot_templates))
    manifest['template_composition_hash'] = hashlib.sha256(json.dumps([manifest['template_families'], manifest['operation_signatures']], sort_keys=True).encode()).hexdigest()
    train_ops = {op for r in output if r['split'] == 'train' for op in operation_signature(r)}
    heldout_ops = {op for r in output if holdout_kind(r, compositions, templates)[0] for op in operation_signature(r)}
    manifest['ood_unseen_primitives'] = sorted(heldout_ops - train_ops)
    manifest['ood_primitive_coverage'] = not bool(heldout_ops - train_ops)
    manifest['manifest_hash'] = 'sha256:' + hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    for row in output:
        row['split_manifest_hash'] = manifest['manifest_hash']
    return output, manifest
