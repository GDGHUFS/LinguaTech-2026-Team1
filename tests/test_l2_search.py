"""CPU-only candidate fitting and deterministic resumption, no Target calls."""
import numpy as np
import pytest
import torch
from src.contracts import ROOT, config
from src.features import temporal_features
from src.training import search_probe, predict_probe


def tensors():
    torch.manual_seed(17)
    f = temporal_features(torch.randn(24,4),torch.randn(24,4))
    return tuple(t[:16] for t in f), torch.tensor([0.,1.]*8), tuple(t[16:] for t in f), torch.tensor([0.,1.]*4)


@pytest.mark.parametrize('name',['p1','t1','conshift'])
def test_equal_search_grid_cpu_and_finished_resume(tmp_path,name):
    f,y,v,z = tensors()
    cfg = {**config(ROOT/'configs/experiment.yaml')['training'],'max_epochs':2}
    model,meta,report = search_probe(name,f,y,v,z,cfg,tmp_path)
    assert next(model.parameters()).device.type == 'cpu'
    assert [c['l2_lambda'] for c in report['candidates']] == [.0001,.001,.01,.1]
    assert meta['selected_l2'] == report['selected_l2']
    assert meta['training_config']['l2_lambda'] == meta['selected_l2']
    assert meta['hyperparameter_selection_source'] == 'validation'
    assert len(list((tmp_path/'search'/name).glob('*/checkpoints/*.last.pt'))) == 4
    expected = predict_probe(model,v)
    model2,meta2,report2 = search_probe(name,f,y,v,z,cfg,tmp_path,resume=True)
    assert np.array_equal(expected,predict_probe(model2,v))
    assert meta2['selected_l2'] == meta['selected_l2']


def test_interrupted_search_keeps_candidates_isolated(tmp_path,monkeypatch):
    import src.training as training
    f,y,v,z = tensors()
    cfg = {**config(ROOT/'configs/experiment.yaml')['training'],'max_epochs':2}
    expected,_,_ = search_probe('conshift',f,y,v,z,cfg,tmp_path/'reference')
    original = training.fit_probe
    def interrupt(*args,**kwargs):
        if args[5]['l2_lambda'] == .01:
            raise InterruptedError('Before third candidate')
        return original(*args,**kwargs)
    monkeypatch.setattr(training,'fit_probe',interrupt)
    with pytest.raises(InterruptedError):
        search_probe('conshift',f,y,v,z,cfg,tmp_path/'interrupted')
    monkeypatch.setattr(training,'fit_probe',original)
    actual,_,_ = search_probe('conshift',f,y,v,z,cfg,tmp_path/'interrupted',resume=True)
    assert np.array_equal(predict_probe(expected,v),predict_probe(actual,v))


def test_cpu_pipeline_reuses_collection_with_separate_output(tmp_path,monkeypatch):
    from src.contracts import digest, write_json, write_jsonl, read_json
    from src.representation import SampleCache
    from src.training import train
    import src.evaluation as evaluation
    from src.target import FrozenTarget
    def forbid_target(*args,**kwargs):
        raise AssertionError('Probe-only pipeline must never construct Target')
    monkeypatch.setattr(FrozenTarget,'__init__',forbid_target)
    monkeypatch.setattr(evaluation,'figures',lambda *args:None)
    cfg = config(ROOT/'configs/experiment.yaml')
    cfg['paths'] = {'data':str(tmp_path/'data'),'output':str(tmp_path/'probe'),
                    'collection_output':str(tmp_path/'collection')}
    cfg['training']['max_epochs'] = 1
    cfg['evaluation']['bootstrap_repetitions'] = 20
    cache = SampleCache(tmp_path/'data')
    torch.manual_seed(3)
    rows = []
    for i in range(40):
        previous = {'sample_id':f'previous-{i}','cache_fingerprint':'fp'}
        cache.commit(previous,torch.randn(4))
        row = dict(sample_id=str(i),trajectory_id=f't-{i}',group_id=f'g-{i//2}',
            turn_index=i+2,split='train' if i<16 else 'validation' if i<24 else 'id_test' if i<32 else 'ood',
            failure_label=i%2 if i<32 else int(i>32),eligible=True,cache_fingerprint='fp',
            prompt_hash='p',prompt_token_ids_hash='t',prompt_text='common sample '+('yes' if i%2 else 'no'),
            prompt_token_count=20+i,prefill_seconds=0.,generation_seconds=0.,
            total_seconds_including_retries=0.)
        cache.commit(row,torch.randn(4))
        row['previous_representation_key'] = previous['representation_key']
        rows.append(row)
    write_jsonl(tmp_path/'data/labeled/turns.jsonl',rows)
    write_json(tmp_path/'data/manifests/collection_lock.json',dict(target=cfg['target'],
        fingerprint='fp',dtype='torch.bfloat16',template='template',code='original-collection-code',versions={}))
    write_json(tmp_path/'data/manifests/split_manifest.json',dict(manifest_hash='split',ood_status='ENABLED'))
    write_json(tmp_path/'collection/reports/label_coverage.json',dict(status='BLOCKED',
        all_original_turns_collected=True,labeled_hash=digest(rows),
        cohort_hash=digest([r['sample_id'] for r in rows]),
        id_test={'status':'PASS'},ood={'status':'BLOCKED'}))
    first = train(cfg)
    assert all(first[name]['selected_l2'] in cfg['training']['l2_candidates']
               for name in ('p1','t1','conshift'))
    assert len(read_json(tmp_path/'probe/reports/p1_l2_search.json')['candidates']) == 4
    resumed = train(cfg,resume=True)
    assert first == resumed
    result = evaluation.evaluate(cfg)
    assert result['id_test']['status'] == 'PASS'
    assert result['ood']['status'] == 'PASS'
    assert result['ood']['coverage']['turns'] == {'0': 1, '1': 7}
    assert result['ood']['models']['conshift']['n'] == 8
    assert read_json(tmp_path/'probe/reports/evaluation_coverage.json')['policy'] == 'both_classes_present_v2'
    assert set(result['id_test']['models']) == {'p1','t1','conshift','g1','l1'}
    assert not (tmp_path/'collection/checkpoints').exists()
