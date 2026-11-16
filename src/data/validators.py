"""[B] 필드와 operation 순서가 맞는지 검사하고, Qwen의 검증 결과를 확인한다."""
from src.data.capability_matrix import CAPABILITIES

def validate_structure(sample):
    errors = []
    from src.rules.schema import validate_trace_priorities
    try:
        validate_trace_priorities(sample['initial_rules'], sample['operation_trace'])
    except ValueError:
        errors.append('unsupported_numeric_priority_only_integer_zero_allowed')
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
    # 구조와 출처는 여기서 검사하고, 연산 유효성과 정책 의미론은 A oracle에 위임한다.
    if not errors:
        from src.data.oracle_adapter import evaluate_structured
        try:
            if evaluate_structured(sample)['annotation_status'] != 'VALID':
                errors.append('invalid_policy_annotation')
        except (ValueError, TypeError, KeyError):
            errors.append('invalid_policy_trace')
    return errors

def validate_dialogue(output, sample):
    """생성 응답의 형식과 실제 turn 대응을 검사해 오류 목록을 반환한다. 자연어의 정책 의미는 별도로 검증한다."""
    errors = []
    if not isinstance(output, dict) or set(output) != {'turns', 'final_request', 'operation_turns', 'initial_rule_turns'}:
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
    if sample.get('request_style') == 'question_embedding' and not output['final_request'].startswith('What would happen if'):
        errors.append('heldout_template_mismatch')
    if sample.get('request_style') != 'question_embedding' and output['final_request'].startswith('What would happen if'):
        errors.append('unregistered_holdout_template')
    initial = output['initial_rule_turns']
    if not isinstance(initial, dict) or set(initial) != {r['rule_id'] for r in sample['initial_rules']}:
        errors.append('invalid_initial_rule_coverage')
    elif any(type(i) is not int or not 0 <= i < len(turns)-1 for i in initial.values()):
        errors.append('invalid_initial_rule_turn')
    mapping = output['operation_turns']
    expected = [o['operation_id'] for o in sample['operation_trace']]
    if not isinstance(mapping, dict) or set(mapping) != set(expected):
        errors.append('invalid_operation_coverage')
    else:
        ids = [mapping[k] for k in expected]
        if any(type(i) is not int or not 0 <= i < len(turns)-1 for i in ids) or ids != sorted(ids):
            errors.append('invalid_operation_order')
    if sample.get('capability_id') == 'CAP09':
        # A missing symbolic slot is insufficient: names in any history turn leak
        # the source recipient. No literal hit is only an inconclusive lower bound.
        import re
        recipient = sample.get('source_facts', {}).get('recipient')
        if recipient and any(re.search(r'(?<!\w)' + re.escape(recipient) + r'(?!\w)', t['text'], re.I) for t in turns):
            errors.append('cap09_recipient_recoverable_from_dialogue')
    return errors

def validate_semantics(result, sample=None, dialogue=None):
    """Qwen 검증 응답의 형식, 불일치 목록과 confidence를 검사해 오류 목록을 반환한다. 사람의 검수를 대신하지 않는다."""
    required = {'aligned','mismatches','missing_facts','added_unsupported_permissions','confidence'}
    enhanced = sample is not None and ('policy_temporal_boundary' in sample or sample.get('capability_id') in ('CAP07', 'CAP08', 'CAP09'))
    if enhanced:
        required.add('checks')
    if not isinstance(result, dict) or set(result) != required:
        return ['invalid_semantic_schema']
    import math
    if type(result['aligned']) is not bool or type(result['confidence']) not in (int,float) or not math.isfinite(result['confidence']) or not 0 <= result['confidence'] <= 1:
        return ['invalid_semantic_schema']
    for k in ('mismatches','missing_facts','added_unsupported_permissions'):
        if not isinstance(result[k],list) or any(not isinstance(x,str) for x in result[k]):
            return ['invalid_semantic_schema']
    errors = [k for k in ('mismatches','missing_facts','added_unsupported_permissions') if result[k]]
    if not result['aligned']:
        errors.append('semantic_misalignment')
    if result['confidence'] < .9:
        errors.append('low_semantic_confidence')
    if enhanced:
        needed = ['historical_facts_preserved', 'synthetic_policy_temporally_separate',
                  'subject_information_request_preserved']
        if sample.get('capability_id') == 'CAP09':
            needed.append('recipient_not_recoverable_from_full_dialogue')
        checks = result.get('checks')
        if not isinstance(checks, dict) or set(checks) != set(needed):
            return errors + ['missing_semantic_checks']
        turns = {t['turn_id']: t['text'] for t in (dialogue or {}).get('turns', [])}
        for key in needed:
            finding = checks[key]
            if not isinstance(finding, dict) or finding.get('passed') is not True or not finding.get('rationale'):
                errors.append('semantic_check_unresolved:' + key)
                continue
            evidence = finding.get('evidence')
            if not isinstance(evidence, list) or not evidence or any(
                not isinstance(e, dict) or type(e.get('turn_id')) is not int or
                not isinstance(e.get('quote'), str) or not e['quote'].strip() or
                e['quote'] not in turns.get(e['turn_id'], '') for e in evidence):
                errors.append('semantic_check_evidence_missing:' + key)
    return errors
