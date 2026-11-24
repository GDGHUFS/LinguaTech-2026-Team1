"""A의 기존 사례를 통합 rule engine에 적용해 호환 동작을 검사한다."""
import json

import pytest

from src.rules.engine import TransitionError, compute_gold, decide, replay
from src.rules.schema import Authorization, Decision, Request, Rule, Transition, TransitionType

A, D = Authorization.ALLOW, Authorization.DENY
T = TransitionType


def rule(rid, auth, data="phone", recipient="*", purpose="*", **kw):
    return Rule(rule_id=rid, data_type=data, authorization=auth, recipient=recipient, purpose=purpose, **kw)


def gold(transitions, turn, data="phone", recipient=None, purpose=None):
    return decide(replay(transitions), Request(turn, data, recipient, purpose))


def test_add_allow():
    g = gold([Transition(1, T.ADD, rule("r1", A, recipient="doctor"))], 2, recipient="doctor")
    assert (g.decision, g.rule_id, g.transition_type) == (Decision.ALLOW, "r1", T.ADD)


def test_no_rule_defaults_to_deny():
    g = gold([], 1, recipient="doctor")
    assert g.decision == Decision.DENY and g.rule_id is None


def test_revoke_then_reinstate():
    trs = [Transition(1, T.ADD, rule("r1", A, recipient="doctor")), Transition(2, T.REVOKE, target_id="r1")]
    g = gold(trs, 3, recipient="doctor")
    assert (g.decision, g.rule_id, g.transition_type, g.relevant_turn) == (Decision.DENY, None, T.REVOKE, 2)

    g = gold(trs + [Transition(3, T.REINSTATE, target_id="r1")], 4, recipient="doctor")
    assert (g.decision, g.rule_id, g.transition_type) == (Decision.ALLOW, "r1", T.REINSTATE)


def test_exception_only_inside_its_scope():
    trs = [
        Transition(1, T.ADD, rule("r1", D)),
        Transition(2, T.EXCEPTION, rule("r2", A, recipient="mom"), target_id="r1"),
    ]
    assert gold(trs, 3, recipient="mom").decision == Decision.ALLOW
    assert gold(trs, 3, recipient="boss").decision == Decision.DENY


def test_exception_survives_parent_modify():
    trs = [
        Transition(1, T.ADD, rule("r1", A)),
        Transition(2, T.EXCEPTION, rule("r2", D, recipient="bob"), target_id="r1"),
        Transition(3, T.MODIFY, target_id="r1", changes={"purpose": "work"}),
    ]
    g = gold(trs, 4, recipient="bob", purpose="work")
    assert (g.decision, g.rule_id, g.ambiguous) == (Decision.DENY, "r2", False)


def test_modify_moves_scope():
    trs = [
        Transition(1, T.ADD, rule("r1", A, recipient="doctor")),
        Transition(2, T.MODIFY, target_id="r1", changes={"recipient": "nurse"}),
    ]
    assert gold(trs, 3, recipient="doctor").decision == Decision.DENY
    g = gold(trs, 3, recipient="nurse")
    assert (g.decision, g.transition_type) == (Decision.ALLOW, T.MODIFY)


def test_expiry():
    trs = [Transition(1, T.ADD, rule("r1", A, recipient="doctor", expires_turn=2))]
    assert gold(trs, 2, recipient="doctor").decision == Decision.ALLOW
    assert gold(trs, 3, recipient="doctor").decision == Decision.DENY


def test_clarify_only_when_missing_field_matters():
    trs = [Transition(1, T.ADD, rule("r1", A, recipient="doctor"))]
    assert gold(trs, 2, recipient=None).decision == Decision.CLARIFY

    trs = [Transition(1, T.ADD, rule("r1", D))]
    assert gold(trs, 2, recipient=None).decision == Decision.DENY

    trs = [Transition(1, T.ADD, rule("r1", A, recipient="doctor", purpose="treatment"))]
    assert gold(trs, 2, recipient="doctor", purpose=None).decision == Decision.CLARIFY


def test_specific_rule_wins_over_later_general_rule():
    trs = [Transition(1, T.ADD, rule("r1", A, recipient="alice")), Transition(2, T.ADD, rule("r2", D))]
    g = gold(trs, 3, recipient="alice")
    assert (g.decision, g.rule_id, g.ambiguous) == (Decision.ALLOW, "r1", False)


def test_same_turn_specific_wins_without_ambiguity():
    trs = [Transition(1, T.ADD, rule("r1", D)), Transition(1, T.ADD, rule("r2", A, recipient="mom"))]
    g = gold(trs, 2, recipient="mom")
    assert (g.decision, g.ambiguous) == (Decision.ALLOW, False)


def test_same_rank_conflict_is_invalid():
    trs = [Transition(1, T.ADD, rule("r1", A, recipient="mom")), Transition(1, T.ADD, rule("r2", D, recipient="mom"))]
    g = gold(trs, 2, recipient="mom")
    assert g.decision is None and g.ambiguous
    assert g.result['annotation_status'] == 'INVALID'


@pytest.mark.parametrize(
    "trs",
    [
        [Transition(1, T.ADD, rule("r1", D)), Transition(2, T.REVOKE, target_id="r1")],
        [Transition(1, T.ADD, rule("r1", A)), Transition(2, T.REINSTATE, target_id="r1")],
        [Transition(1, T.ADD, rule("r1", A)), Transition(2, T.ADD, rule("r1", D))],
        [Transition(1, T.ADD, rule("r1", A)), Transition(2, T.MODIFY, target_id="r1", changes={"rule_id": "other"})],
        [Transition(1, T.ADD, rule("r1", D)), Transition(2, T.EXCEPTION, rule("r2", A), target_id="r1")],
        [Transition(1, T.ADD, rule("r1", D)), Transition(2, T.EXCEPTION, rule("r2", D, recipient="mom"), target_id="r1")],
        [Transition(2, T.ADD, rule("r1", D)), Transition(1, T.ADD, rule("r2", A))],
        [Transition(1, T.REVOKE, target_id="nope")],
    ],
)
def test_invalid_transitions_raise(trs):
    with pytest.raises(TransitionError):
        replay(trs)


def test_compute_gold_from_json():
    scenario = json.loads("""
    {
      "scenario_id": "s001",
      "transitions": [
        {"turn": 1, "type": "ADD", "rule": {"rule_id": "r1", "data_type": "phone", "authorization": "DENY"}},
        {"turn": 2, "type": "EXCEPTION", "target_id": "r1",
         "rule": {"rule_id": "r2", "data_type": "phone", "recipient": "doctor", "authorization": "ALLOW"}},
        {"turn": 3, "type": "REVOKE", "target_id": "r2"},
        {"turn": 3, "type": "NO_CHANGE"},
        {"turn": 4, "type": "REINSTATE", "target_id": "r2"}
      ],
      "request": {"turn": 5, "data_type": "phone", "recipient": "doctor"}
    }
    """)
    g = compute_gold(scenario)
    json.dumps(g)
    assert g["gold_decision"] == "ALLOW"
    assert g['gold_state']['rule_status'] == 'ACTIVE'
    assert g['supporting_turn_ids'] == [1, 2, 3, 4]
    assert set(g["supporting_rule_ids"]) == {"r1", "r2"}

    scenario["request"]["turn"] = 4
    with pytest.raises(TransitionError):
        compute_gold(scenario)


def test_transition_roundtrip():
    tr = Transition(2, T.EXCEPTION, rule("r2", A, recipient="mom"), target_id="r1")
    assert Transition.from_dict(json.loads(json.dumps(tr.to_dict()))) == tr
