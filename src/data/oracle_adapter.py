"""B record를 A의 정책 oracle 입력 형식으로 변환한다.

우선순위·권한·연산 유효성은 oracle이 판단한다. 구조 계획에는 기호적 turn을 쓰고, 대화 주석에는 초기 규칙과 연산 모두 실제 turn과의 명시적 대응이 필요하다."""
from copy import deepcopy

from src.rules.engine import evaluate_policy
from src.rules.schema import ENGINE_VERSION, operation_name


def canonical_request(request):
    """저장된 label 재검증과 P2에서 공유하는 요청 형식으로 변환한다."""
    request = deepcopy(request)
    request.pop('text', None)
    core = {'data_type', 'recipient', 'purpose', 'subject', 'sender', 'request_time', 'scope'}
    extra = {k: request.pop(k) for k in list(request) if k not in core}
    request['scope'] = {**request.get('scope', {}), **extra}
    return request


def _inputs(sample, dialogue=None):
    from src.rules.schema import validate_trace_priorities
    validate_trace_priorities(sample['initial_rules'], sample['operation_trace'])
    rules = deepcopy(sample['initial_rules'])
    events = []
    if dialogue is None:
        initial_turns = {r['rule_id']: 0 for r in rules}
        operation_turns = {o['operation_id']: i + 1 for i, o in enumerate(sample['operation_trace'])}
        request_turn = len(sample['operation_trace']) + 1
    else:
        initial_turns = dialogue.get('initial_rule_turns')
        operation_turns = dialogue['operation_turns']
        if not isinstance(initial_turns, dict) or set(initial_turns) != {r['rule_id'] for r in rules}:
            raise ValueError('Explicit initial_rule_turns required; never guess supporting turns')
        request_turn = dialogue['turns'][-1]['turn_id']
        history_ids = {t['turn_id'] for t in dialogue['turns'][:-1]}
        if any(type(t) is not int or t not in history_ids for t in initial_turns.values()):
            raise ValueError('Initial rules must reference actual history turns')
    for rule in rules:
        rule['effective_turn'] = initial_turns[rule['rule_id']]
    for operation in sample['operation_trace']:
        op = deepcopy(operation)
        kind = operation_name(op['operation']).value
        event = {'op': kind, 'event_id': op['operation_id'],
                 'effective_turn': operation_turns[op['operation_id']]}
        if kind in ('ADD', 'EXCEPTION'):
            event.update(rule_id=op['rule_id'], rule=op['rule'])
            if kind == 'EXCEPTION':
                event['target_id'] = op['parent_rule_id']
        elif kind != 'NO_CHANGE':
            event['rule_id'] = op['rule_id']
        if op.get('changes'):
            changes = dict(op['changes'])
            if 'expires_at' in changes:
                changes['expiration'] = changes.pop('expires_at')
            event['changes'] = changes
        events.append(event)
    request = canonical_request(sample['request'])
    return rules, events, request, request_turn


def evaluate_structured(sample):
    """기호적 turn으로 구조 계획을 평가한다. 자연어의 의미 정렬은 확인하지 않는다."""
    return evaluate_policy(*_inputs(sample), semantics_version=ENGINE_VERSION)


def annotate_dialogue(sample, dialogue):
    """초기 규칙과 연산을 실제 turn에 대응시켜 gold와 relevance_mask를 추가한다.

    최종 요청은 relevance 학습에서 제외한다. 정책 충돌이나 turn 정렬로 인한 판단·상태 변경은 ValueError로 거부한다."""
    rules, events, request, request_turn = _inputs(sample, dialogue)
    result = evaluate_policy(rules, events, request, request_turn, ENGINE_VERSION)
    if result['annotation_status'] != 'VALID':
        raise ValueError('Invalid policy conflict cannot enter gold annotations')
    # 실제 turn으로 대응시킨 뒤에도 구조 계획의 oracle 판단과 상태가 같아야 한다.
    expected = evaluate_structured(sample)
    if any(result[k] != expected[k] for k in ('gold_decision', 'gold_state')):
        raise ValueError('Natural turn alignment changed structured policy semantics')
    record = deepcopy(sample)
    versions = dict(record.get('versions') or {})
    versions.update(policy_semantics_version=ENGINE_VERSION, annotation_version='oracle-labels-v1')
    record.update(result, initial_rules=rules, operation_trace=events,
                  versions=versions, oracle_verified=True)
    support = set(result['supporting_turn_ids'])
    record['relevance_mask'] = [-100 if t['turn_id'] == request_turn else int(t['turn_id'] in support)
                                for t in dialogue['turns']]
    return record
