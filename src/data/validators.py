"""[B] 필드와 operation 순서가 맞는지 검사하고, Qwen의 검증 결과를 확인한다."""
from src.data.capability_matrix import CAPABILITIES

def validate_structure(sample):
    errors = []
    # capability에 맞는 expected relation인지 확인한다. 실제로 그 관계가 성립하는지는 여기서 판단하지 않는다.
    cap = sample.get('capability_id')
    if cap is not None:
        if cap not in CAPABILITIES:
            errors.append('unknown_capability')
        elif sample.get('expected_gold_relation') != CAPABILITIES[cap][1]:
            errors.append('invalid_expected_relation')
    rule_ids = [r['rule_id'] for r in sample['initial_rules']]
    if len(rule_ids) != len(set(rule_ids)):
        errors.append('duplicate_rule_id')
    operation_ids = [o['operation_id'] for o in sample['operation_trace']]
    if len(operation_ids) != len(set(operation_ids)):
        errors.append('duplicate_operation_id')
    if any(sample.get(k) is not None for k in ('gold_decision','gold_state','evidence_trace')):
        errors.append('unexpected_gold')
    if sample.get('oracle_relation_status') != 'pending':
        errors.append('oracle_must_be_pending')
    rules = {r['rule_id'] for r in sample['initial_rules']}
    revoked = set()
    for op in sample['operation_trace']:
        name, rid = op['operation'], op['rule_id']
        if name in ('ADD', 'EXCEPTION'):
            if rid in rules or op.get('rule', {}).get('rule_id') != rid:
                errors.append('invalid_new_rule')
            if name == 'EXCEPTION' and op.get('parent_rule_id') not in rules:
                errors.append('missing_parent_rule')
            rules.add(rid)
        elif name in ('REVOKE','MODIFY','REINSTATE'):
            if rid not in rules:
                errors.append('unknown_rule_reference')
            if name == 'REVOKE':
                revoked.add(rid)
            if name == 'REINSTATE' and rid not in revoked:
                errors.append('missing_prior_revocation')
        elif name != 'NO_CHANGE':
            errors.append('unknown_operation')
    return errors

def validate_dialogue(output, sample):
    errors = []
    if not isinstance(output, dict) or set(output) != {'turns', 'final_request', 'operation_turns'}:
        return ['invalid_output_schema']
    turns = output['turns']
    if not isinstance(turns, list) or not 4 <= len(turns) <= 12:
        return ['invalid_turn_count']
    for i, turn in enumerate(turns):
        if not isinstance(turn, dict) or set(turn) != {'turn_id','role','text'} or turn.get('turn_id') != i or type(turn.get('turn_id')) is not int or turn.get('role') not in ('user','assistant') or not isinstance(turn.get('text'),str) or not turn['text'].strip():
            errors.append('invalid_turn')
    if errors:
        return errors
    if turns[-1]['role'] != 'user' or output['final_request'] != turns[-1]['text']:
        errors.append('request_mismatch')
    mapping = output['operation_turns']
    expected = [o['operation_id'] for o in sample['operation_trace']]
    if not isinstance(mapping, dict) or set(mapping) != set(expected):
        errors.append('invalid_operation_coverage')
    else:
        ids = [mapping[k] for k in expected]
        if any(type(i) is not int or not 0 <= i < len(turns)-1 for i in ids) or ids != sorted(ids):
            errors.append('invalid_operation_order')
    return errors

def validate_semantics(result):
    required = {'aligned','mismatches','missing_facts','added_unsupported_permissions','confidence'}
    if not isinstance(result, dict) or set(result) != required:
        return ['invalid_semantic_schema']
    if type(result['aligned']) is not bool or type(result['confidence']) not in (int,float) or not 0 <= result['confidence'] <= 1:
        return ['invalid_semantic_schema']
    for k in ('mismatches','missing_facts','added_unsupported_permissions'):
        if not isinstance(result[k],list) or any(not isinstance(x,str) for x in result[k]):
            return ['invalid_semantic_schema']
    errors = [k for k in ('mismatches','missing_facts','added_unsupported_permissions') if result[k]]
    if not result['aligned']:
        errors.append('semantic_misalignment')
    if result['confidence'] < .9:
        errors.append('low_semantic_confidence')
    return errors
