"""합성 저장 점수로 평가 흐름을 검사한다. 모델 호출이나 연구 산출물은 만들지 않는다."""
from copy import deepcopy
import hashlib
import json

import pytest
import yaml

from src.evaluation.protocol import digest,evaluate_predictions,freeze_protocol


def fixture():
    rows = []
    for split in ('train','validation','id_test','ood_composition','ood_template'):
        for group in range(2):
            for label in (0,1):
                sid = f'{split}-{group}-{label}'
                rows.append(dict(sample_id=sid,split=split,group_id=f'CoPriva:{split}-{group}',source_dataset='CoPriva',
                    source_seed_id=f'{split}-{group}',capability_id='CAP02',parse_status='VALID',gold_decision='DENY',
                    target_decision='ALLOW' if label else 'DENY',failure_label_unsafe=label,
                    gold_state=dict(authorization='DENY',rule_status='REVOKED',scope_match='MATCH')))
    manifest = dict(assignments={r['sample_id']:r['split'] for r in rows})
    manifest['manifest_hash'] = 'sha256:'+hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    predictions = []
    for model in ('A','C','C+'):
        for seed in (42,43,44):
            for row in rows:
                label = row['failure_label_unsafe']
                score = (.9 if label else .1) if model != 'A' else (.1 if label else .9)
                predictions.append(dict(sample_id=row['sample_id'],split=row['split'],group_id=row['group_id'],model=model,seed=seed,
                                        checkpoint_hash=f'{model}:{seed}',failure_probability=score))
    plan = dict(revision='synthetic-test-only',models=['A','C','C+','G1','P1','B_pred'],seeds=[42,43,44])
    return rows,predictions,manifest,plan


def test_full_fair_comparison_three_splits_three_seeds_and_holm():
    rows,predictions,manifest,plan = fixture()
    protocol = freeze_protocol(predictions,rows,manifest,plan,test_only=True)
    result = evaluate_predictions(predictions,manifest,protocol,records=rows,test_only=True)
    assert result['models']['G1']['status'] == 'SKIPPED'
    assert result['status'] == 'INCOMPLETE'
    for split in ('id_test','ood_composition','ood_template'):
        performance = result['models']['C']['splits'][split]
        assert performance['summary']['auprc']['mean'] == 1
        assert performance['summary']['auprc']['sd'] == 0
        assert performance['subgroup_summary']['gold=DENY']['auprc']['mean']==1
        assert len(performance['seeds']) == 3
        assert performance['seeds']['42']['metrics']['threshold'] == .9
        assert 'gold=DENY' in performance['seeds']['42']['subgroups']
        contrast = result['contrasts'][f'{split}:C-A']
        assert contrast['estimate'] == .5 and contrast['repetitions'] == 2000
        assert contrast['holm_family_size'] == 6
    assert result['optional']['representation_probe']['status'] == 'SKIPPED'
    assert result['models']['C']['id_ood_gap']['ood_template']['summary']['mean']==0


def test_test_scores_never_select_threshold():
    rows,predictions,manifest,plan = fixture()
    protocol = freeze_protocol(predictions,rows,manifest,plan,test_only=True)
    changed = deepcopy(predictions)
    for p in changed:
        if p['split']=='id_test':
            p['failure_probability'] = .2
    frozen_again = freeze_protocol(changed,rows,manifest,plan,test_only=True)
    assert protocol['thresholds'] == frozen_again['thresholds']
    assert protocol['protocol_hash'] == frozen_again['protocol_hash']
    result = evaluate_predictions(changed,manifest,protocol,records=rows,test_only=True)
    assert result['models']['C']['splits']['id_test']['seeds']['42']['metrics']['threshold'] == .9


@pytest.mark.parametrize('problem',['missing','nan','split','group','duplicate','seed','checkpoint'])
def test_partial_or_misaligned_predictions_fail_closed(problem):
    rows,predictions,manifest,plan = fixture()
    protocol = freeze_protocol(predictions,rows,manifest,plan,test_only=True)
    index = next(i for i,p in enumerate(predictions) if p['split']=='id_test')
    if problem=='missing':
        predictions.pop(index)
    elif problem=='duplicate':
        predictions.append(dict(predictions[index]))
    else:
        key,value = {'nan':('failure_probability',float('nan')),'split':('split','train'),
                     'group':('group_id','wrong'),'seed':('seed',99),'checkpoint':('checkpoint_hash','wrong')}[problem]
        predictions[index][key] = value
    with pytest.raises(ValueError):
        evaluate_predictions(predictions,manifest,protocol,records=rows,test_only=True)


def test_invalid_target_is_common_mask_and_never_a_negative():
    rows,predictions,manifest,plan = fixture()
    selected = next(r for r in rows if r['split']=='id_test')
    selected.update(parse_status='INVALID_OUTPUT',target_decision=None,failure_label_unsafe=None)
    for p in predictions:
        if p['sample_id']==selected['sample_id']:
            p['failure_probability']=None
    protocol = freeze_protocol(predictions,rows,manifest,plan,test_only=True)
    result = evaluate_predictions(predictions,manifest,protocol,records=rows,test_only=True)
    for model in ('A','C','C+'):
        part = result['models'][model]['splits']['id_test']
        assert part['excluded_target_count']==1
        assert selected['sample_id'] not in part['common_sample_ids']


def test_references_and_validation_are_locked():
    rows,predictions,manifest,plan = fixture()
    protocol = freeze_protocol(predictions,rows,manifest,plan,test_only=True)
    rows[0]['gold_decision']='ALLOW'
    with pytest.raises(ValueError,match='reference data'):
        evaluate_predictions(predictions,manifest,protocol,records=rows,test_only=True)
    rows,predictions,manifest,plan = fixture()
    protocol = freeze_protocol(predictions,rows,manifest,plan,test_only=True)
    next(p for p in predictions if p['split']=='validation')['failure_probability']=.4
    with pytest.raises(ValueError,match='Validation scores'):
        evaluate_predictions(predictions,manifest,protocol,records=rows,test_only=True)


@pytest.mark.parametrize('change',[{'calibration':'isotonic'},{'models':['B_oracle']},
                                 {'seeds':[]},{'seeds':[42,42]},{'seeds':[True]},{'bootstrap_repetitions':100}])
def test_unregistered_protocol_choices_are_rejected(change):
    rows,predictions,manifest,plan = fixture()
    plan.update(change)
    with pytest.raises(ValueError):
        freeze_protocol(predictions,rows,manifest,plan,test_only=True)


def test_unfrozen_or_tampered_protocol_is_rejected():
    rows,predictions,manifest,plan = fixture()
    with pytest.raises(ValueError,match='Frozen protocol'):
        evaluate_predictions(predictions,manifest,dict(plan,frozen=False),records=rows,test_only=True)
    protocol = freeze_protocol(predictions,rows,manifest,plan,test_only=True)
    protocol['thresholds']['C']['42']['threshold'] = .1
    with pytest.raises(ValueError,match='Frozen protocol'):
        evaluate_predictions(predictions,manifest,protocol,records=rows,test_only=True)


def test_masked_state_and_turn_relevance_diagnostics():
    rows,predictions,manifest,plan = fixture()
    for row in rows:
        row.update(turns=[dict(turn_id=0),dict(turn_id=1),dict(turn_id=2)],request_turn_id=2,relevance_mask=[1,0,-100])
        row['gold_state']['scope_match']=None
    for p in predictions:
        if p['model']!='A':
            p['predicted_state'] = dict(authorization='DENY',rule_status='REVOKED',scope_match='MATCH')
        if p['model']=='C+':
            p['turn_scores']={'0':.9,'1':.1}
    protocol = freeze_protocol(predictions,rows,manifest,plan,test_only=True)
    result = evaluate_predictions(predictions,manifest,protocol,records=rows,test_only=True)
    diagnostics = result['models']['C+']['splits']['id_test']['seeds']['42']['supervision']
    assert diagnostics['state']['scope_match']['status']=='NA'
    assert diagnostics['relevance']['f1']==1
    assert diagnostics['relevance']['score_coverage']==1


def test_cli_freeze_then_evaluate(tmp_path):
    from evaluate import main
    rows,predictions,manifest,plan = fixture()
    # --test-only로 평가 CLI 연결을 검사한다. 아래 검수 metadata는 합성이며 실제 승인 검증은 별도 테스트에서 확인한다.
    selection = {'CoPriva': []}
    (tmp_path/'data.yaml').write_text(yaml.safe_dump(dict(sources=['CoPriva'], source_selection=selection)), encoding='utf-8')
    for row in rows:
        row.update(human_review_status='reviewed', semantic_review=dict(reviewer='test-only', synthetic_policy_compatible=True),
            purpose_mapping_status='confirmed', generation_ready=True,
            source_evidence={k: dict(status='confirmed') for k in ('subject','sender','recipient','data_type','purpose')},
            consent_assessment=dict(status='reviewed'), disclosure_assessment=dict(status='reviewed'))
    manifest.pop('manifest_hash')
    manifest['source_selection'] = selection
    manifest['manifest_hash'] = digest(manifest)
    for name,content in [('records',rows),('predictions',predictions)]:
        (tmp_path/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in content),encoding='utf-8')
    (tmp_path/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    (tmp_path/'plan.yaml').write_text(yaml.safe_dump(plan),encoding='utf-8')
    args=['--test-only','--records',str(tmp_path/'records.jsonl'),'--predictions',str(tmp_path/'predictions.jsonl'),
          '--split-manifest',str(tmp_path/'manifest.json'),'--data-config',str(tmp_path/'data.yaml')]
    assert main(args+['--protocol',str(tmp_path/'plan.yaml'),'--freeze','--out',str(tmp_path/'frozen.json')])==0
    assert main(args+['--protocol',str(tmp_path/'frozen.json'),'--out',str(tmp_path/'report.json')])==0
    assert json.loads((tmp_path/'report.json').read_text())['status']=='INCOMPLETE'
