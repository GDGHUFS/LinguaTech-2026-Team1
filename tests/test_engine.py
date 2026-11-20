"""재사용한 A 사례에 더해 통합 명세서의 경계 조건을 검사한다."""
import json
from copy import deepcopy

import pytest

from src.rules.engine import TransitionError, apply, evaluate_policy, replay
from src.rules.schema import Authorization, Request, Rule, State, Transition, TransitionType


def rule(rid='r', auth='ALLOW', **fields):
    return dict(rule_id=rid, data_type='email', recipient='organizer', purpose='event',
                authorization=auth, **fields)


def event(op, turn=1, rid='r', **fields):
    return dict(op=op, effective_turn=turn, event_id=f'e{turn}', rule_id=rid, **fields)


def oracle(rules=None, events=None, request=None, turn=5, **kw):
    return evaluate_policy([rule()] if rules is None else rules, events or [],
                           dict(data_type='email', recipient='organizer', purpose='event') if request is None else request,
                           turn, **kw)


def test_initial_permission_is_not_discarded():
    g = oracle()
    assert g['gold_decision'] == 'ALLOW'
    assert g['gold_state'] == dict(authorization='ALLOW', rule_status='ACTIVE', scope_match='MATCH', ambiguous_fields=[])
    assert g['supporting_turn_ids'] == [0]


def test_default_denial_has_no_invented_support():
    g = oracle([])
    assert g['gold_decision'] == 'DENY'
    assert g['gold_state']['rule_status'] == 'NONE'
    assert g['gold_state']['scope_match'] == 'MISMATCH'
    assert g['supporting_turn_ids'] == g['supporting_rule_ids'] == []


@pytest.mark.parametrize('field,value', [('recipient', 'doctor'), ('purpose', 'treatment'),
                                        ('data_type', 'phone'), ('authorization', 'DENY')])
def test_modify_supersedes_old_permission(field, value):
    g = oracle(events=[event('MODIFY', changes={field: value})])
    assert g['gold_decision'] == 'DENY'
    assert g['supporting_turn_ids'] == [0, 1]
    detail = g['evidence_trace']['event_details'][-1]
    assert detail['before'][field] != detail['after'][field]
    assert detail['changed_fields'] == [field]
    assert detail['predecessor_event_ids'] == ['initial:r']


def test_reinstate_is_new_allow_event_with_lineage():
    g = oracle(events=[event('REVOKE'), event('REINSTATE', 2)])
    details = g['evidence_trace']['event_details']
    assert g['gold_decision'] == 'ALLOW'
    assert details[-2]['after']['status'] == 'REVOKED'
    assert details[-1]['after']['status'] == 'ACTIVE'
    assert details[-1]['after']['effective_turn'] == 2
    assert details[-1]['after']['authorization'] == 'ALLOW'
    assert g['evidence_trace']['edges'] == [['e1', 'e2'], ['initial:r', 'e1']]
    assert g['supporting_turn_ids'] == [0, 1, 2]


def test_reapproval_can_change_scope_without_restoring_old_scope():
    g = oracle(events=[event('REVOKE'), event('REINSTATE', 2, changes={'recipient': 'doctor'})])
    assert g['gold_decision'] == 'DENY'
    assert g['gold_state']['scope_match'] == 'MISMATCH'
    assert g['supporting_turn_ids'] == [0, 1, 2]


def test_revoke_is_not_an_explicit_deny_rule():
    revoked = oracle(events=[event('REVOKE')])
    denied = oracle([rule(auth='DENY')])
    assert revoked['gold_decision'] == denied['gold_decision'] == 'DENY'
    assert revoked['gold_state']['rule_status'] == 'REVOKED'
    assert denied['gold_state']['rule_status'] == 'ACTIVE'


@pytest.mark.parametrize('missing', ['recipient', 'purpose', 'data_type'])
def test_decision_relevant_missing_slot_clarifies(missing):
    request = dict(data_type='email', recipient='organizer', purpose='event')
    request[missing] = None
    g = oracle(request=request)
    assert g['gold_decision'] == 'CLARIFY'
    assert g['gold_state']['authorization'] == 'UNRESOLVED'
    assert g['gold_state']['scope_match'] == 'INCOMPLETE'
    assert g['ambiguity_flags']['missing_context']


def test_missing_slot_that_cannot_change_decision_does_not_invent_conflict():
    g = oracle([rule(auth='DENY')], request=dict(data_type='email', recipient=None, purpose='event'))
    assert g['gold_decision'] == 'DENY'
    assert g['gold_state']['scope_match'] is None
    assert g['gold_state']['rule_status'] is None
    assert 'scope_match' in g['gold_state']['ambiguous_fields']


def test_unknown_slot_is_missing_context():
    g = oracle(request={'data_type': 'email', 'purpose': 'event'})
    assert g['gold_decision'] == 'CLARIFY'


def test_unresolved_conflict_masks_gold_instead_of_deny_or_clarify():
    g = oracle([rule('a'), rule('b', 'DENY')])
    assert g['annotation_status'] == 'INVALID'
    assert g['gold_decision'] is None
    assert all(g['gold_state'][k] is None for k in ('authorization', 'rule_status', 'scope_match'))
    assert g['ambiguity_flags']['unresolved_conflict']
    assert not g['ambiguity_flags']['missing_context']


def test_conflict_in_missing_slot_branch_is_still_invalid():
    g = oracle([rule('a'), rule('b', 'DENY')], request={'data_type': 'email', 'recipient': None, 'purpose': 'event'})
    assert g['annotation_status'] == 'INVALID' and g['gold_decision'] is None


def test_latest_effective_turn_resolves_equal_scope():
    g = oracle(events=[event('ADD', rid='other', rule=rule('other', 'DENY'))])
    assert g['gold_decision'] == 'DENY' and g['annotation_status'] == 'VALID'


def test_same_auth_duplicates_keep_both_justifications():
    g = oracle([rule('a'), rule('b')])
    assert g['gold_decision'] == 'ALLOW'
    assert g['supporting_rule_ids'] == ['a', 'b']


def test_narrative_is_not_supporting_evidence():
    plain = oracle()
    g = oracle(events=[dict(op='NO_CHANGE', event_id='noise', effective_turn=1)])
    assert g == plain


def test_irrelevant_rule_is_not_support():
    other = dict(rule('other'), data_type='phone')
    assert oracle([rule(), other])['supporting_rule_ids'] == ['r']


def test_revoked_exception_keeps_its_ancestor_closure():
    broad = dict(rule(auth='DENY'), recipient='*')
    events = [event('EXCEPTION', rid='child', rule=rule('child'), target_id='r'),
              event('REVOKE', 2, 'child')]
    g = oracle([broad], events)
    assert g['gold_decision'] == 'DENY'
    assert g['supporting_rule_ids'] == ['child', 'r']
    assert g['supporting_turn_ids'] == [0, 1, 2]
    assert g['gold_state']['rule_status'] is None  # 일반 규칙은 ACTIVE이고 하위 규칙은 REVOKED인 경우다.


@pytest.mark.parametrize('turn,decision,status', [(2, 'ALLOW', 'ACTIVE'), (3, 'DENY', 'EXPIRED')])
def test_turn_expiration_inclusive(turn, decision, status):
    g = oracle([rule(expiration=2)], turn=turn)
    assert (g['gold_decision'], g['gold_state']['rule_status']) == (decision, status)


@pytest.mark.parametrize('now,decision', [('2030-01-01T00:00:00Z', 'ALLOW'), ('2030-01-02T00:00:00Z', 'DENY')])
def test_timestamp_expiration(now, decision):
    g = oracle([rule(expiration='2030-01-01T00:00:00Z')],
               request=dict(data_type='email', recipient='organizer', purpose='event', request_time=now))
    assert g['gold_decision'] == decision


def test_missing_expiration_clock_is_rejected_not_guessed():
    with pytest.raises(ValueError, match='request_time'):
        oracle([rule(expiration='2030-01-01T00:00:00Z')])


@pytest.mark.parametrize('field', ['subject', 'sender'])
def test_actor_scope_is_enforced(field):
    g = oracle([rule(**{field: 'owner'})], request=dict(data_type='email', recipient='organizer', purpose='event', **{field: 'other'}))
    assert g['gold_decision'] == 'DENY'


def test_extra_scope_exception_matches_b_channel():
    broad = rule(auth='DENY')
    child = rule('child', scope={'channel': 'secure_portal'})
    ev = event('EXCEPTION', rid='child', rule=child, target_id='r')
    req = dict(data_type='email', recipient='organizer', purpose='event', scope={'channel': 'secure_portal'})
    assert oracle([broad], [ev], req)['gold_decision'] == 'ALLOW'
    req['scope']['channel'] = 'public'
    assert oracle([broad], [ev], req)['gold_decision'] == 'DENY'


@pytest.mark.parametrize('op', ['REVOKE', 'MODIFY', 'REINSTATE'])
def test_nonexistent_reference_is_error(op):
    with pytest.raises(TransitionError, match='unknown target'):
        oracle(events=[event(op, rid='missing', changes={'purpose': 'other'} if op == 'MODIFY' else {})])


def test_repeated_revocation_is_error():
    with pytest.raises(TransitionError):
        oracle(events=[event('REVOKE'), event('REVOKE', 2)])


def test_modifying_revoked_rule_is_error():
    with pytest.raises(TransitionError):
        oracle(events=[event('REVOKE'), event('MODIFY', 2, changes={'purpose': 'other'})])


def test_expired_permission_not_automatically_renewed_by_reinstate():
    g = oracle([rule(expiration=1)], [event('REVOKE'), event('REINSTATE', 2)])
    assert g['gold_decision'] == 'DENY' and g['gold_state']['rule_status'] == 'EXPIRED'


def test_duplicate_event_id_is_error():
    with pytest.raises(TransitionError, match='duplicate event_id'):
        oracle(events=[event('REVOKE'), dict(event('REINSTATE', 2), event_id='e1')])


def test_duplicate_initial_rule_id_is_error():
    with pytest.raises(TransitionError, match='duplicate rule_id'):
        oracle([rule(), rule()])


def test_ordered_events_are_not_silently_sorted():
    with pytest.raises(TransitionError, match='out of order'):
        oracle(events=[event('REVOKE', 2), event('REINSTATE', 1)])


@pytest.mark.parametrize('at', [5, 6])
def test_request_and_future_events_are_rejected(at):
    with pytest.raises(TransitionError, match='precede current request'):
        oracle(events=[event('REVOKE', at)])


def test_unknown_semantics_version_is_error():
    with pytest.raises(TransitionError, match='Unsupported semantics'):
        oracle(semantics_version='future')


def test_request_turn_disagreement_is_error():
    with pytest.raises(TransitionError, match='turn mismatch'):
        oracle(request=dict(data_type='email', recipient='organizer', purpose='event', turn=4))


def test_legacy_update_normalized_only_at_input_boundary():
    tr = Transition.from_dict(event('UPDATE', changes={'purpose': 'other'}))
    assert tr.type == TransitionType.MODIFY and tr.to_dict()['type'] == 'MODIFY'
    assert oracle(events=[event('UPDATE', changes={'purpose': 'other'})])['evidence_trace']['event_details'][-1]['op'] == 'MODIFY'
    with pytest.raises(ValueError):
        Transition(1, 'UPDATE', target_id='r')
    with pytest.raises(ValueError):
        TransitionType('UPDATE')


@pytest.mark.parametrize('fields', [{'priority': 1}, {'expiration': -1}, {'expiration': True},
                                    {'expiration': '2030-01-01'}, {'recipient': None}])
def test_invalid_or_unspecified_rule_fields_fail_explicitly(fields):
    raw = dict(rule(), **fields)
    with pytest.raises(ValueError):
        Rule.from_dict(raw)


def test_no_change_cannot_hide_mutation():
    with pytest.raises(TransitionError):
        oracle(events=[event('NO_CHANGE', changes={'authorization': 'ALLOW'})])


def test_result_json_roundtrip_and_input_immutability():
    rules, events = [rule()], [event('MODIFY', changes={'authorization': 'DENY'})]
    before = deepcopy((rules, events))
    result = oracle(rules, events)
    assert json.loads(json.dumps(result)) == result
    assert (rules, events) == before


def test_request_turn_is_never_in_closure():
    g = oracle(events=[event('REVOKE')])
    assert 5 not in g['supporting_turn_ids']


def test_apply_preserves_predecessor_state():
    state = replay([Transition(0, TransitionType.ADD, Rule.from_dict(rule()))])
    changed = apply(state, Transition(1, TransitionType.REVOKE, target_id='r'))
    assert state['r'].status.value == 'ACTIVE'
    assert changed['r'].status.value == 'REVOKED'
    assert len(state.events) == 1 and len(changed.events) == 2
