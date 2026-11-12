"""[B] 구조 변형(structured mutation)을 만들며 gold 의미론은 계산하지 않는다."""
from copy import deepcopy
import hashlib
from src.data.capability_matrix import CAPABILITIES

SLOTS = ('data_type', 'subject', 'sender', 'recipient', 'purpose')

def make_variants(seed):
    # 구조가 완비된 후보만 사용하며 사람 검토(human review)는 계속 대기한다.
    if seed.get('status') != 'eligible_pending_review' or any(not seed.get(k) for k in SLOTS):
        return []
    if not seed.get('group_id') or not seed.get('source_provenance'):
        raise ValueError('Missing source identity/provenance')
    scope = {k: seed[k] for k in SLOTS}
    identity = hashlib.sha256(seed['seed_id'].encode()).hexdigest()[:20]
    rule = dict(rule_id=f'r_{identity}', **scope, authorization='ALLOW')
    # 허용은 원본 동의가 아니라 명시적으로 표시한 연구용 개입(synthetic intervention)이다.
    base = {'sample_id': f'{identity}__base', 'source_seed_id': seed['source_seed_id'],
            'seed_id': seed['seed_id'], 'source_dataset': seed['source_dataset'],
            'source_record_id': seed['source_record_id'], 'source_provenance': deepcopy(seed['source_provenance']),
            'group_id': seed['group_id'], 'variant_group_id': f'{seed["group_id"]}:{identity}',
            'data_kind': 'real_derived_unlabeled', 'initial_rules': [rule], 'operation_trace': [],
            'request': deepcopy(scope), 'policy_context': deepcopy(seed['policy_context']),
            'synthetic_intervention': True, 'transformation_notes': ['Synthetic base permission; not source consent or gold.'],
            'oracle_relation_status': 'pending', 'gold_decision': None, 'gold_state': None,
            'evidence_trace': None, 'human_review_status': 'pending', 'variant_id': 'base',
            'parent_variant_id': None, 'capability_id': None, 'expected_gold_relation': None,
            'template_family_id': 'b_pilot_base_v1', 'request_style': 'direct'}
    result = [base]
    for cap, (name, relation) in CAPABILITIES.items():
        v = deepcopy(base)
        v.update(sample_id=f'{identity}__{cap}', variant_id=cap, parent_variant_id='base',
                 capability_id=cap, expected_gold_relation=relation, template_family_id=f'b_pilot_{name.lower()}_v1')
        op = {'operation_id': f'{cap}_op1', 'operation': name, 'rule_id': rule['rule_id']}
        if cap == 'CAP01':
            v['initial_rules'] = []
            op['rule'] = deepcopy(rule)
        elif cap == 'CAP03':
            # 새로운 인물이나 목적을 추측하지 않고 유효 기간(validity)을 합성 수정한다.
            op['changes'] = {'expires_at': '2030-01-01T00:00:00Z'}
            v['request']['request_time'] = '2030-01-02T00:00:00Z'
        elif cap == 'CAP04':
            v['initial_rules'][0]['authorization'] = 'DENY'
            op['parent_rule_id'] = rule['rule_id']
            op['rule_id'] += '_exception'
            op['rule'] = dict(rule, rule_id=op['rule_id'], authorization='ALLOW', scope={'channel': 'secure_portal'})
            v['request']['channel'] = 'secure_portal'
        elif cap == 'CAP05':
            v['operation_trace'].append(dict(op, operation='REVOKE', operation_id=f'{cap}_op0'))
        elif cap in ('CAP07', 'CAP08'):
            v['request_style'] = 'paraphrase' if cap == 'CAP07' else 'indirect'
            op = None
        elif cap == 'CAP09':
            v['request']['recipient'] = None
            op = None
        if op:
            v['operation_trace'].append(op)
        result.append(v)
    return result
