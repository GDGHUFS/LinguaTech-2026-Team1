"""A/B JSON 계약, 실제 turn 정렬, failure label 연동을 검사한다."""
from copy import deepcopy
import ast
import json
from pathlib import Path

import pytest

from src.data.contracts import GoldState, Sample
from src.data.generation import execute
from src.data.labeling import failure_labels, run_target, target_input
from src.data.mutations import make_variants
from src.data.oracle_adapter import annotate_dialogue, evaluate_structured
from src.data.splitting import split_samples
from src.data.validators import validate_dialogue, validate_structure
from src.rules.engine import evaluate_policy
from src.rules.schema import ENGINE_VERSION

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def seed():
    return dict(status='eligible_pending_review', seed_id='CAMBER:fixture', source_seed_id='fixture',
                source_dataset='CAMBER', source_record_id='fixture-1', group_id='CAMBER:fixture',
                source_provenance={'fixture': True}, policy_context='Fictional test policy',
                data_type='email', subject='owner', sender='owner', recipient='organizer', purpose='event')


def dialogue(sample):
    # 고정된 합성 fixture이며 실제 데이터나 API를 사용하지 않는다.
    turns = [dict(turn_id=0, role='user', text='I allow my email to the organizer for this event.'),
             dict(turn_id=1, role='assistant', text='Understood.'),
             dict(turn_id=2, role='user', text='I revoke that permission.'),
             dict(turn_id=3, role='user', text='Send my email to the organizer for this event.')]
    return dict(turns=turns, final_request=turns[-1]['text'],
                initial_rule_turns={r['rule_id']: 0 for r in sample['initial_rules']},
                operation_turns={o['operation_id']: 2 for o in sample['operation_trace']})


@pytest.mark.parametrize('cap', [None] + [f'CAP{i:02}' for i in range(1, 10)])
def test_b_mutation_families_use_the_a_oracle(seed, cap):
    sample = next(v for v in make_variants(seed) if v['capability_id'] == cap)
    assert validate_structure(sample) == []
    g = evaluate_structured(sample)
    assert g['annotation_status'] == 'VALID'
    GoldState.model_validate(g['gold_state'])
    if cap in ('CAP02', 'CAP03'):
        assert g['gold_decision'] == 'DENY'
    elif cap == 'CAP09':
        assert g['gold_decision'] == 'CLARIFY'
    else:
        assert g['gold_decision'] == 'ALLOW'


def test_annotation_remaps_closure_to_actual_turn_ids(seed):
    sample = make_variants(seed)[2]  # CAP02의 REVOKE 사례다.
    natural = dialogue(sample)
    annotation = annotate_dialogue(sample, natural)
    assert annotation['supporting_turn_ids'] == [0, 2]
    assert annotation['relevance_mask'] == [1, 0, 1, -100]
    assert annotation['operation_trace'][0]['effective_turn'] == 2
    assert annotation['oracle_verified']
    assert sample['gold_decision'] is None


def test_missing_initial_alignment_is_not_guessed(seed):
    sample = make_variants(seed)[2]
    natural = dialogue(sample)
    del natural['initial_rule_turns']
    with pytest.raises(ValueError, match='initial_rule_turns'):
        annotate_dialogue(sample, natural)
    assert validate_dialogue(natural, sample) == ['invalid_output_schema']


@pytest.mark.parametrize('turn', [3, 99, -1, True])
def test_initial_evidence_must_reference_real_history_turn(seed, turn):
    sample = make_variants(seed)[2]
    natural = dialogue(sample)
    natural['initial_rule_turns'] = {sample['initial_rules'][0]['rule_id']: turn}
    with pytest.raises(ValueError):
        annotate_dialogue(sample, natural)


def test_natural_turn_remapping_cannot_change_oracle_decision(seed):
    sample = make_variants(seed)[0]
    sample['initial_rules'][0]['expiration'] = 2
    with pytest.raises(ValueError, match='changed structured policy semantics'):
        annotate_dialogue(sample, dialogue(sample))


class FixtureClient:
    model = 'qwen-fixture'
    endpoint = 'fixture://no-network'

    def __init__(self, natural, aligned=True):
        self.natural, self.aligned = natural, aligned
        self.outputs = [json.dumps(natural), json.dumps(dict(aligned=aligned, mismatches=[],
                         missing_facts=[], added_unsupported_permissions=[], confidence=.99))]

    def complete(self, prompt):
        if '"structured_trace"' in prompt:
            sample = json.loads(prompt.split('INPUT JSON:\n')[-1].partition('\nEND INPUT JSON')[0])['structured_trace']
            return json.dumps(semantic_fixture(self.natural, sample, self.aligned))
        return self.outputs.pop(0)


def semantic_fixture(natural, sample, aligned=True):
    """Test transport response only, never a real semantic review."""
    result = dict(aligned=aligned, mismatches=[], missing_facts=[], added_unsupported_permissions=[], confidence=.99)
    if 'policy_temporal_boundary' in sample or sample.get('capability_id') in ('CAP07','CAP08','CAP09'):
        keys = ['historical_facts_preserved','synthetic_policy_temporally_separate','subject_information_request_preserved']
        if sample.get('purpose_provenance'):
            keys += ['synthetic_purpose_consistent_with_source', 'policy_conditions_expressed']
        if sample.get('capability_id') == 'CAP09': keys.append('recipient_not_recoverable_from_full_dialogue')
        result['checks'] = {k: dict(passed=True, rationale='Synthetic transport fixture only',
            evidence=[dict(turn_id=0,quote=natural['turns'][0]['text'])]) for k in keys}
    return result


def test_generation_attaches_gold_only_after_semantic_validation(seed):
    sample = make_variants(seed)[2]
    record = execute(sample, FixtureClient(dialogue(sample)), ROOT/'prompts')
    assert record['validation_status'] == 'semantic_pass_oracle_verified'
    assert record['gold_decision'] == 'DENY'
    assert record['supporting_turn_ids'] == [0, 2]
    assert record['versions']['policy_semantics_version'] == ENGINE_VERSION
    assert record['human_review_status'] == 'pending'  # 구조 검증 통과는 사람의 QA 승인을 뜻하지 않는다.
    assert record['oracle_relation_status'] == 'pending'  # 쌍별 관계 검증은 별도의 통과 조건이다.


def test_semantic_mismatch_does_not_rewrite_gold(seed):
    sample = make_variants(seed)[2]
    record = execute(sample, FixtureClient(dialogue(sample), aligned=False), ROOT/'prompts')
    assert record['validation_status'] == 'rejected'
    assert record['gold_decision'] is None and not record.get('oracle_verified')


def test_invalid_policy_blocks_generation_before_client_call(seed):
    sample = make_variants(seed)[0]
    other = dict(sample['initial_rules'][0], rule_id='other', authorization='DENY')
    sample['initial_rules'].append(other)
    record = execute(sample, FixtureClient(dialogue(sample)), ROOT/'prompts')
    assert record['generation_status'] == 'invalid_structure'


def test_pending_generation_is_honestly_unannotated(seed):
    record = execute(make_variants(seed)[0], None, ROOT/'prompts')
    assert record['generation_status'] == 'pending_qwen'
    assert record['gold_decision'] is None and not record.get('oracle_verified')


class Runner:
    metadata = {'model_id': 'fixture'}
    def complete(self, prompt):
        assert 'gold_decision' not in prompt and 'supporting_turn_ids' not in prompt
        return '{"decision":"ALLOW"}'


def test_annotated_generation_to_target_label_uses_only_natural_input(seed):
    sample = make_variants(seed)[2]
    record = execute(sample, FixtureClient(dialogue(sample)), ROOT/'prompts')
    result = run_target(record, Runner(), 'fixture decision protocol')
    assert result['failure_label_unsafe'] == result['failure_label_any'] == 1
    assert set(target_input(record)) == {'dialogue', 'final_request'}


def test_invalid_annotation_cannot_be_labeled():
    with pytest.raises(ValueError, match='Invalid policy'):
        run_target({'annotation_status': 'INVALID'}, Runner(), '')


def test_canonical_operations_keep_b_split_compatibility(seed):
    sample = make_variants(seed)[5]  # CAP05의 재승인 사례다.
    rows = [dict(sample, operation_trace=[dict(op='REVOKE'), dict(op='REINSTATE')])]
    output, _ = split_samples(rows, compositions=(('REVOKE', 'REINSTATE'),), templates=())
    assert output[0]['split'] == 'ood_composition'


def test_sample_contract_accepts_canonical_oracle_fields(seed):
    sample = make_variants(seed)[2]
    natural = dialogue(sample)
    gold = annotate_dialogue(sample, natural)
    payload = dict(sample_id='fixture', source_dataset='CAMBER', source_record_id='fixture',
                   source_seed_id='fixture', scenario_id='fixture', group_id='CAMBER:fixture',
                   variant_group_id='pair', variant_id='CAP02', parent_variant_id='base', data_kind='real_derived',
                   source_provenance={'fixture': True}, transformation_notes=['synthetic test'], capability_id='CAP02',
                   template_family_id='fixture', expected_gold_relation='STATE_CHANGE', language='en', stage='gold',
                   turns=natural['turns'], request={'text': natural['final_request']}, request_turn_id=3,
                   initial_rules=gold['initial_rules'], operation_trace=gold['operation_trace'],
                   versions=dict(gold['versions'], source_record_hash='sha256:'+'0'*64))
    for key in ('gold_decision', 'gold_state', 'supporting_rule_ids', 'supporting_turn_ids', 'evidence_trace',
                'engine_version', 'ambiguity_flags', 'annotation_status', 'oracle_verified', 'relevance_mask'):
        payload[key] = gold[key]
    validated = Sample.model_validate(payload)
    assert validated.gold_state.label_ids() == dict(authorization=1, rule_status=1, scope_match=0)
    payload['annotation_status'] = 'INVALID'
    with pytest.raises(ValueError, match='Invalid policy'):
        Sample.model_validate(payload)


def test_invalid_state_classes_use_null_and_existing_ignore_contract():
    g = evaluate_policy([dict(rule_id='a', data_type='email', authorization='ALLOW'),
                         dict(rule_id='b', data_type='email', authorization='DENY')], [],
                        dict(data_type='email', recipient='x', purpose='x'), 1, ENGINE_VERSION)
    assert set(GoldState.model_validate(g['gold_state']).label_ids().values()) == {-100}


def test_source_directory_is_not_an_import_dependency():
    for path in (ROOT/'src').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith('by_a')
            elif isinstance(node, ast.Import):
                assert all(not name.name.startswith('by_a') for name in node.names)
