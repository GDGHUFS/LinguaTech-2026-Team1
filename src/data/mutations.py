"""[B] 원본 정보를 바탕으로 structured mutation을 만든다. gold는 계산하지 않는다."""
from copy import deepcopy
import hashlib
from src.data.capability_matrix import CAPABILITIES

SLOTS = ('data_type', 'subject', 'sender', 'recipient', 'purpose')

def make_variants(seed, modify_fields=('authorization',), synthetic_purpose='confidential discussion'):
    if 'source_evidence' in seed:
        from src.data.source_policy import generation_blockers
        if generation_blockers(seed):
            return []
    # 필수 정보가 있는 후보만 사용하며 사람의 검수 상태는 그대로 전달한다.
    if seed.get('status') != 'eligible_pending_review' or any(not seed.get(k) for k in SLOTS if k != 'purpose'):
        return []
    if not seed.get('purpose') and not seed.get('source_normalization'):
        return []
    if not seed.get('group_id') or not seed.get('source_provenance'):
        raise ValueError('Missing source identity/provenance')
    scope = {k: seed[k] for k in SLOTS}
    purpose_origin = 'source_extracted' if scope['purpose'] else 'synthetic_policy'
    if purpose_origin == 'synthetic_policy':
        if not isinstance(synthetic_purpose, str) or not synthetic_purpose.strip():
            raise ValueError('Explicit nonempty synthetic policy purpose required')
        scope['purpose'] = synthetic_purpose
    identity = hashlib.sha256(seed['seed_id'].encode()).hexdigest()[:20]
    rule = dict(rule_id=f'r_{identity}', **scope, authorization='ALLOW')
    # 초기 ALLOW는 연구를 위해 추가한 조건이다. 원본에 있던 동의로 취급하지 않는다.
    base = {'sample_id': f'{identity}__base', 'source_seed_id': seed['source_seed_id'],
            'seed_id': seed['seed_id'], 'source_dataset': seed['source_dataset'],
            'source_record_id': seed['source_record_id'], 'source_provenance': deepcopy(seed['source_provenance']),
            'group_id': seed['group_id'], 'variant_group_id': f'{seed["group_id"]}:{identity}',
            'data_kind': 'real_derived_unlabeled', 'initial_rules': [rule], 'operation_trace': [],
            'request': deepcopy(scope), 'policy_context': deepcopy(seed['policy_context']),
            'purpose_provenance': dict(origin=purpose_origin, value=scope['purpose'],
                source_path=(seed.get('field_provenance') or {}).get('purpose') if purpose_origin == 'source_extracted' else None,
                changes=[]),
            'synthetic_intervention': True, 'transformation_notes': ['Synthetic base permission; not source consent or gold.'],
            'oracle_relation_status': 'pending', 'gold_decision': None, 'gold_state': None,
            'evidence_trace': None, 'human_review_status': 'pending', 'variant_id': 'base',
            'parent_variant_id': None, 'capability_id': None, 'expected_gold_relation': None,
            'template_family_id': 'b_pilot_base_v1', 'request_style': 'direct'}
    for key in ('source_component', 'source_evidence', 'semantic_review', 'consent_assessment',
                'disclosure_assessment', 'purpose_mapping_status', 'generation_ready', 'expansion',
                'annotation', 'source_motive', 'disclosure_purpose', 'source_checks', 'adapter_revision',
                'source_normalization'):
        if key in seed:
            base[key] = deepcopy(seed[key])
    base['human_review_status'] = seed.get('human_review_status', 'pending')
    base['data_kind'] = seed.get('data_kind', 'real_derived_unlabeled')
    if seed.get('source_normalization'):
        base['source_record'] = deepcopy(seed['source_record'])
    base['source_facts'] = {k: deepcopy(seed.get(k)) for k in SLOTS + ('consent',)}
    base['policy_temporal_boundary'] = {
        'source_history': 'Preserve historical secrecy, recovery and knowledge limits.',
        'synthetic_policy': 'Explicit later hypothetical policy; never original consent.',
        'status': 'requires_automatic_semantic_validation'}
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
        if field == 'recipient':
            alternatives = [seed[k] for k in ('subject','sender') if seed[k] != seed['recipient']]
            if not alternatives:
                continue
            value = alternatives[0]
        elif field == 'purpose':
            value = 'restricted record review'
        else:
            value = 'administrative contact details (hypothetical policy category)'
        v['operation_trace'][0]['changes'] = {field:value}
        if field == 'purpose':
            v['purpose_provenance']['changes'] = [dict(operation_id=v['operation_trace'][0]['operation_id'],
                value=value, origin='synthetic_policy')]
        v['transformation_notes'].append('Synthetic scope intervention: '+field+'='+value)
        result.append(v)
    return result


def validate_policy_provenance(sample):
    """Check policy scope separately from immutable source facts; never infer gold."""
    if not sample.get('source_normalization'):
        return []
    meta = sample.get('purpose_provenance') or {}
    if (meta.get('origin') != 'synthetic_policy' or not isinstance(meta.get('value'), str) or not meta['value'].strip() or
            meta.get('source_path') is not None or sample.get('source_facts', {}).get('purpose') is not None):
        return ['synthetic_policy_purpose_provenance_mismatch']
    rules = list(sample.get('initial_rules', []))
    rules += [o['rule'] for o in sample.get('operation_trace', []) if 'rule' in o]
    if any(r.get('purpose') != meta['value'] or r.get('authorization') not in ('ALLOW', 'DENY') for r in rules):
        return ['synthetic_policy_scope_or_authorization_mismatch']
    if sample.get('request', {}).get('purpose') != meta.get('request_value',meta['value']):
        return ['synthetic_policy_request_purpose_mismatch']
    changes = [dict(operation_id=o['operation_id'], value=o['changes']['purpose'], origin='synthetic_policy')
        for o in sample.get('operation_trace', []) if 'purpose' in o.get('changes', {})]
    if meta.get('changes') != changes:
        return ['synthetic_policy_purpose_change_provenance_mismatch']
    return []


def expand_coverage_variants(anchors):
    """Add counterexamples to operation-keyword shortcuts; do not rebalance on Target results."""
    if not anchors:
        return []
    result = list(anchors)
    by_cap = {v.get('capability_id'):v for v in anchors if not v.get('modification_subtype')}
    def variant(cap, name):
        row = deepcopy(by_cap[cap])
        row.update(sample_id=row['sample_id']+'__'+name,variant_id=row['variant_id']+'__'+name,
            parent_variant_id=None,variant_group_id=row['variant_group_id']+':'+name,
            controlled_anchor=False)
        row['transformation_notes'].append('Predeclared policy contrast: '+name)
        return row
    # ADD can introduce a prohibition as well as a permission.
    row = variant('CAP01','new_prohibition')
    row['operation_trace'][0]['rule']['authorization'] = 'DENY'
    result.append(row)
    # Revoking A does not revoke an independently stated permission B.
    row = variant('CAP02','independent_permission')
    other = deepcopy(row['initial_rules'][0])
    other['rule_id'] += '_independent'
    row['initial_rules'].append(other)
    result.append(row)
    # A narrower DENY exception to a general ALLOW, and a request outside an ALLOW exception.
    row = variant('CAP04','narrow_prohibition')
    row['initial_rules'][0]['authorization'] = 'ALLOW'
    row['operation_trace'][0]['rule']['authorization'] = 'DENY'
    result.append(row)
    row = variant('CAP04','outside_exception')
    row['request']['channel'] = 'public_portal'
    result.append(row)
    # Same scope-change event: asking within its new scope can still be allowed.
    for field in ('recipient','purpose'):
        original = next((v for v in anchors if v.get('modification_subtype')==field),None)
        if not original:
            continue
        row = deepcopy(original)
        row.update(sample_id=row['sample_id']+'__new_scope_request',variant_id=row['variant_id']+'__new_scope_request',
            parent_variant_id=None,variant_group_id=row['variant_group_id']+':new_scope_'+field,controlled_anchor=False)
        row['request'][field] = row['operation_trace'][0]['changes'][field]
        if field=='purpose':
            row['purpose_provenance']['request_value'] = row['request'][field]
        result.append(row)
    original = next((v for v in anchors if v.get('modification_subtype')=='data_type'),None)
    if original:
        row = deepcopy(original)
        row.update(sample_id=row['sample_id']+'__return_to_source_category',
            variant_id=row['variant_id']+'__return_to_source_category',parent_variant_id=None,
            variant_group_id=row['variant_group_id']+':return_to_source_category',controlled_anchor=False)
        hypothetical = row['operation_trace'][0]['changes']['data_type']
        row['initial_rules'][0]['data_type'] = hypothetical
        row['operation_trace'][0]['changes']['data_type'] = row['source_facts']['data_type']
        row['hypothetical_initial_scopes'] = [dict(rule_id=row['initial_rules'][0]['rule_id'],
            field='data_type',value=hypothetical,origin='synthetic_policy',asserted_source_fact=False)]
        result.append(row)
    # Reapproval restores its own scope, not every possible disclosure purpose.
    row = variant('CAP05','other_purpose_request')
    row['request']['purpose'] = 'restricted record review'
    row['purpose_provenance']['request_value'] = row['request']['purpose']
    result.append(row)
    for cap in ('CAP06','CAP07','CAP08','CAP09'):
        row = variant(cap,'prohibition_context')
        row['initial_rules'][0]['authorization'] = 'DENY'
        result.append(row)
    for row in result:
        row['generation_architecture'] = 'guarded_v1'
        row.setdefault('controlled_anchor',True)
        row['scope_intervention_provenance'] = [dict(field=k,value=v,origin='synthetic_policy',asserted_source_fact=False)
            for op in row['operation_trace'] for k,v in op.get('changes',{}).items()
            if k in ('recipient','purpose','data_type')]
    return result


def apply_holdout_plan(variants, pool, coverage='legacy'):
    """Target label이나 점수와 무관하게 사전 지정한 group별 holdout을 적용한다.

    ID의 기본 설정은 유지한다. holdout group의 대조 variant도 쌍별 검토를 위해 남기되, split 단계에서 OOD 평가 대상에서는 제외한다."""
    if coverage not in ('legacy', 'coverage_v2'):
        raise ValueError('Unknown holdout coverage')
    if coverage == 'coverage_v2' and pool != 'id':
        return _expanded_holdout(variants, pool)
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
    companion = deepcopy(v)
    companion['sample_id'] += '__contrast'
    companion['variant_id'] += '__contrast'
    if pool == 'ood_template':
        revoke = next(r for r in variants if r.get('capability_id') == 'CAP02')
        companion['operation_trace'] = deepcopy(revoke['operation_trace'])
    else:
        companion['request']['channel'] = 'public_portal'
    return variants+[companion, v]


def _expanded_holdout(variants, pool):
    """Predeclared structural coverage, independent of observed Target decisions."""
    if pool not in ('ood_composition', 'ood_template') or not variants:
        raise ValueError('Unknown or empty holdout pool')
    added = []

    def detached(original, suffix):
        row = deepcopy(original)
        row.update(sample_id=row['sample_id']+'__'+suffix,
                   variant_id=row['variant_id']+'__'+suffix,
                   parent_variant_id=None, expected_gold_relation=None,
                   capability_id=None, controlled_anchor=False,
                   variant_group_id=row['variant_group_id']+':'+suffix)
        row['transformation_notes'].append('Predeclared holdout coverage_v2: '+suffix)
        return row

    if pool == 'ood_template':
        for original in variants:
            # Missing-recipient prompts prepend a clarification instruction and do not
            # satisfy the existing held-out question-prefix contract.
            if original['request'].get('recipient') is None:
                continue
            row = detached(original, 'ood_template_v2')
            row.update(template_family_id='b_ood_question_embedding_v1',
                       request_style='question_embedding')
            added.append(row)
    else:
        base = next(row for row in variants if row['variant_id'] == 'base')
        # REVOKE requires ALLOW, and EXCEPTION must flip its parent's authorization.
        for initial in ('ALLOW',):
            for style in ('direct', 'paraphrase', 'indirect'):
                for channel in ('secure_portal', 'public_portal'):
                    suffix = 'ood_composition_v2_'+style+'_'+channel
                    row = detached(base, suffix)
                    row['request_style'] = style
                    rule = row['initial_rules'][0]
                    rule['authorization'] = initial
                    rid = rule['rule_id']
                    row['operation_trace'] = [
                        dict(operation_id='comp_revoke',operation='REVOKE',rule_id=rid),
                        dict(operation_id='comp_reinstate',operation='REINSTATE',rule_id=rid),
                        dict(operation_id='comp_exception',operation='EXCEPTION',
                             rule_id=rid+'_exception',parent_rule_id=rid,
                             rule=dict(rule,rule_id=rid+'_exception',authorization='DENY',
                                       scope={'channel':'secure_portal'}))]
                    row['request']['channel'] = channel
                    row['template_family_id'] = 'b_pilot_composition_v1'
                    added.append(row)
    return variants + added
