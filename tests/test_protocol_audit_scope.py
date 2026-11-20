"""오프라인으로 검사 범위와 감사 관련 회귀를 확인한다."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.rules.schema import Rule, validate_trace_priorities
from src.data.canonicalize import screening_summary
from src.data.fewshots import load_fewshots
from src.data.oracle_adapter import annotate_dialogue
from src.data.splitting import template_validation_scope


@pytest.mark.parametrize('value', [1, -1, True, False, 0.0, '0', None])
@pytest.mark.parametrize('location', ['initial', 'event', 'new_rule', 'changes'])
def test_numeric_priority_is_strict_integer_zero(value, location):
    initial, events = [], []
    entry = {'priority': value}
    if location == 'initial': initial = [entry]
    elif location == 'event': events = [entry]
    else: events = [{'rule' if location == 'new_rule' else 'changes': entry}]
    with pytest.raises(ValueError, match='priority=0 only'):
        validate_trace_priorities(initial, events)
    validate_trace_priorities([{'priority': 0}], [{'rule': {'priority': 0}, 'changes': {}}])


def test_cap06_narrative_is_input_supported_and_policy_invariant():
    shots = load_fewshots(Path('prompts/naturalization_shots.json'))['shots']
    shot = next(s for s in shots if s['template_family_id'] == 'shot_nochange_v2')
    assert shot['example_kind'] == 'CAP06_unrelated_narrative_policy_invariance'
    output, sample = shot['output'], shot['input']
    assert 'delivery updates online' in sample['policy_context']
    assert output['turns'][2]['text'] == 'By the way, I usually check delivery updates online.'
    observed = annotate_dialogue(sample, output)
    control = deepcopy(sample)
    control['operation_trace'] = []
    control_output = deepcopy(output)
    control_output['operation_turns'] = {}
    baseline = annotate_dialogue(control, control_output)
    assert (observed['gold_decision'], observed['gold_state']) == (baseline['gold_decision'], baseline['gold_state'])


def test_adapter_cannot_silently_drop_nonzero_priority():
    from src.data.oracle_adapter import evaluate_structured
    shots = load_fewshots(Path('prompts/naturalization_shots.json'))['shots']
    sample = deepcopy(shots[0]['input'])
    sample['operation_trace'][0]['priority'] = 1
    with pytest.raises(ValueError, match='priority=0 only'):
        evaluate_structured(sample)
    with pytest.raises(ValueError, match='priority=0 only'):
        Rule.from_dict(dict(rule_id='r1', data_type='email', authorization='ALLOW', priority=True))


def test_audit_counts_overlapping_exclusions_without_claiming_approved_seeds():
    rows = [dict(status='excluded', group_id='one', exclusion_reasons=['missing_subject','missing_purpose'],
                 human_review_status='pending', generation_ready=False)]
    summary = screening_summary(rows)
    assert summary['records'] == 1 and sum(summary['exclusion_reasons'].values()) == 2
    assert summary['eligible_pending_review'] == summary['human_reviewed'] == summary['generation_ready'] == 0


def test_empty_ood_scope_does_not_claim_generalization():
    scope = template_validation_scope([], ['b_ood_paraphrase_v1'])
    assert scope['supported_request_style'] == 'question_embedding'
    assert scope['observed_eligible_ood_families'] == {}
    assert scope['semantic_generalization_status'] == 'NOT_ESTABLISHED'
