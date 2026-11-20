"""다운로드, GPU 학습, Qwen 요청 없이 seed와 생성 예시 관련 회귀를 검사한다."""
import ast
from copy import deepcopy
import itertools
import json
from pathlib import Path
import random

import pytest
import yaml

from src.data.fewshots import load_fewshots, check_fewshot_overlap, expression_audit
from src.data.generation import execute
from src.data.oracle_adapter import _inputs, annotate_dialogue
from src.data.splitting import DEFAULT_TEMPLATES, check_leakage
from src.evaluation.protocol import freeze_protocol, evaluate_predictions

SHOTS = Path('prompts/naturalization_shots.json')


def test_seven_shots_preserve_operations_and_oracle_bridge():
    shots = load_fewshots(SHOTS, DEFAULT_TEMPLATES)['shots']
    for shot in shots:
        sample, output = shot['input'], shot['output']
        events = _inputs(sample, output)[1]
        assert [e['event_id'] for e in events] == [o['operation_id'] for o in sample['operation_trace']]
        assert [e['op'] for e in events] == [o['operation'] for o in sample['operation_trace']]
    missing = annotate_dialogue(shots[-1]['input'], shots[-1]['output'])
    assert missing['gold_decision'] == 'CLARIFY'


def test_generation_actually_passes_all_seven_examples_to_client():
    examples = load_fewshots(SHOTS)
    sample = dict(deepcopy(examples['shots'][0]['input']), oracle_relation_status='pending')
    class FakeClient:
        model, endpoint = 'Qwen-synthetic-test-only', 'offline://fixture'
        def __init__(self): self.prompts = []
        def complete(self, prompt):
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                passed = json.loads(prompt.split('SEVEN SYNTHETIC EXAMPLES:\n')[1].split('\nINPUT JSON:\n')[0])
                assert passed == examples['shots']
                return json.dumps(examples['shots'][0]['output'])
            return json.dumps(dict(aligned=True, mismatches=[], missing_facts=[],
                                   added_unsupported_permissions=[], confidence=.95))
    client = FakeClient()
    result = execute(sample, client, SHOTS.parent)
    assert len(client.prompts) == 2
    assert result['validation_status'] == 'semantic_pass_oracle_verified'
    assert result['execution']['shot_count'] == 7
    # 가짜 client로 연결만 확인한다. 실제 의미 검증은 수행하지 않는다.


@pytest.mark.parametrize('problem', ['count', 'alias', 'expression', 'family'])
def test_bad_or_leaking_shots_are_rejected(tmp_path, problem):
    examples = json.loads(SHOTS.read_text(encoding='utf-8'))
    if problem == 'count': examples['shots'].pop()
    elif problem == 'alias':
        op = examples['shots'][0]['input']['operation_trace'][0]
        op['event_id'] = op.pop('operation_id')
    elif problem == 'expression':
        examples['shots'][0]['output']['turns'][1]['text'] = 'WHAT would happen if this were shared?'
    else: examples['shots'][0]['template_family_id'] = DEFAULT_TEMPLATES[-1]
    path = tmp_path/'shots.json'
    path.write_text(json.dumps(examples), encoding='utf-8')
    with pytest.raises(ValueError): load_fewshots(path, DEFAULT_TEMPLATES)


def test_ood_expression_and_request_overlap_are_checked_beyond_family_ids():
    row = dict(source_dataset='fixture', source_seed_id='one', group_id='one', split='train',
               template_family_id='renamed_id', turns=[dict(text='What would happen if I shared it?')])
    with pytest.raises(ValueError, match='expression'): check_leakage([row])
    shots = load_fewshots(SHOTS)['shots']
    heldout = dict(row, split='ood_template', template_family_id=DEFAULT_TEMPLATES[-1],
                   final_request=shots[0]['output']['final_request'], turns=[])
    with pytest.raises(ValueError, match='few-shot'): check_fewshot_overlap([heldout], shots, DEFAULT_TEMPLATES)
    row.update(turns=[], final_request=heldout['final_request'])
    heldout.update(source_seed_id='two', group_id='two')
    with pytest.raises(ValueError, match='final request'): check_leakage([row, heldout])


def test_cap07_08_audit_does_not_claim_semantic_diversity():
    audit = expression_audit([dict(capability_id='CAP07', final_request='Can I share it?')]*2)
    assert audit['CAP07']['distinct_final_requests'] == 1
    assert audit['CAP07']['generated_count'] == 2
    assert audit['CAP08']['semantic_diversity_status'].startswith('NOT_MEASURED')


def test_single_seed_freeze_evaluation_and_baseline_config():
    from test_evaluation_protocol import fixture
    from src.evaluation.baselines_pre import validate_config
    rows, predictions, manifest, plan = fixture()
    plan['seeds'] = [42]
    predictions = [p for p in predictions if p['seed'] == 42]
    frozen = freeze_protocol(predictions, rows, manifest, plan,test_only=True)
    result = evaluate_predictions(predictions, manifest, frozen, records=rows,test_only=True)
    assert frozen['experiment']['single_seed_experiment'] is True
    assert result['experiment']['seed_variability_estimated'] is False
    assert result['models']['C']['splits']['id_test']['summary']['auprc']['sd'] is None
    assert result['contrasts']['id_test:C-A']['repetitions'] == 2000
    config = yaml.safe_load(Path('configs/baselines.yaml').read_text(encoding='utf-8'))
    validate_config(config)
    for name in ('train', 'eval', 'baselines'):
        assert yaml.safe_load(Path(f'configs/{name}.yaml').read_text(encoding='utf-8'))['seeds'] == [42]


def test_tuning_has_unique_effective_trials_without_importing_gpu_runtime():
    # torch나 transformers가 없어도 실제 계획 함수를 검사한다.
    tree = ast.parse(Path('train.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'tuning_plan')
    namespace = dict(random=random, itertools=itertools, MODES=dict.fromkeys(['A','B_oracle','B_pred','C','C_plus','C+']))
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'train.py', 'exec'), namespace)
    planner = namespace['tuning_plan']
    config = yaml.safe_load(Path('configs/train.yaml').read_text(encoding='utf-8'))
    for mode, count in [('A',1),('B_oracle',1),('B_pred',3),('C',3),('C_plus',3)]:
        assert len(planner(mode, config, 42)) == 1
        plan = planner(mode, config, 42, tune=True)
        assert len(plan) == len(set(plan)) == count
        assert plan == planner(mode, config, 42, tune=True)
    config['nu_candidates'] = [.5, .5, .5]
    assert len(planner('B_pred', config, 42, tune=True)) == 1
