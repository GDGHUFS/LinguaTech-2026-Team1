"""[B] 원본 정보를 바탕으로 structured mutation을 만든다. gold는 계산하지 않는다."""
from copy import deepcopy
import hashlib
from src.data.capability_matrix import CAPABILITIES

SLOTS = ('data_type', 'subject', 'sender', 'recipient', 'purpose')

def make_variants(seed, modify_fields=('authorization',)):
    if 'source_evidence' in seed:
        from src.data.source_policy import generation_blockers
        if generation_blockers(seed):
            return []
    # 필수 정보가 있는 후보만 사용하며 사람의 검수 상태는 그대로 전달한다.
    if seed.get('status') != 'eligible_pending_review' or any(not seed.get(k) for k in SLOTS):
        return []
    if not seed.get('group_id') or not seed.get('source_provenance'):
        raise ValueError('Missing source identity/provenance')
    scope = {k: seed[k] for k in SLOTS}
    identity = hashlib.sha256(seed['seed_id'].encode()).hexdigest()[:20]
    rule = dict(rule_id=f'r_{identity}', **scope, authorization='ALLOW')
    # 초기 ALLOW는 연구를 위해 추가한 조건이다. 원본에 있던 동의로 취급하지 않는다.
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
    for key in ('source_component', 'source_evidence', 'semantic_review', 'consent_assessment',
                'disclosure_assessment', 'purpose_mapping_status', 'generation_ready', 'expansion',
                'annotation', 'source_motive', 'disclosure_purpose', 'source_checks', 'adapter_revision'):
        if key in seed:
            base[key] = deepcopy(seed[key])
    base['human_review_status'] = seed.get('human_review_status', 'pending')
    base['source_facts'] = {k: deepcopy(seed.get(k)) for k in SLOTS + ('consent',)}
    base['policy_temporal_boundary'] = {
        'source_history': 'Preserve historical secrecy, recovery and knowledge limits.',
        'synthetic_policy': 'Explicit later hypothetical policy; never original consent.',
        'status': 'requires_semantic_validation_and_independent_qa'}
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
            # CAP03은 합성 정책의 authorization 필드를 변경한다.
            op['changes'] = {'authorization': 'DENY'}
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
    for field in modify_fields:
        if field not in ('authorization','recipient','purpose','data_type'):
            raise ValueError('Unsupported MODIFY subtype')
        if field == 'authorization':
            continue
        v = deepcopy(next(r for r in result if r['capability_id']=='CAP03'))
        v.update(sample_id=v['sample_id']+'__'+field, variant_id='CAP03_'+field, modification_subtype=field)
        value = 'synthetic_alternative_' + field
        v['operation_trace'][0]['changes'] = {field:value}
        v['transformation_notes'].append('Synthetic scope intervention: '+field+'='+value)
        result.append(v)
    return result


def apply_holdout_plan(variants, pool):
    """Target label이나 점수와 무관하게 사전 지정한 group별 holdout을 적용한다.

    ID의 기본 설정은 유지한다. holdout group의 대조 variant도 쌍별 검토를 위해 남기되, split 단계에서 OOD 평가 대상에서는 제외한다."""
    if pool == 'id':
        return variants
    if pool not in ('ood_composition','ood_template') or not variants:
        raise ValueError('Unknown predeclared holdout pool')
    v = deepcopy(variants[0])
    rule = v['initial_rules'][0]
    v.update(sample_id=v['sample_id']+'__'+pool,variant_id=pool,parent_variant_id='base',
             capability_id=None,expected_gold_relation=None)
    if pool == 'ood_template':
        v.update(template_family_id='b_ood_question_embedding_v1',request_style='question_embedding')
    else:
        rid = rule['rule_id']
        v['operation_trace'] = [
            dict(operation_id='comp_revoke',operation='REVOKE',rule_id=rid),
            dict(operation_id='comp_reinstate',operation='REINSTATE',rule_id=rid),
            dict(operation_id='comp_exception',operation='EXCEPTION',rule_id=rid+'_exception',parent_rule_id=rid,
                 rule=dict(rule,rule_id=rid+'_exception',authorization='DENY',scope={'channel':'secure_portal'}))]
        v['request']['channel']='secure_portal'
        v['template_family_id']='b_pilot_composition_v1'
    return variants+[v]
