"""Interrupted tensor-only runs must equal uninterrupted runs exactly."""
from pathlib import Path
import numpy as np
import pytest
import torch
from src.contracts import ROOT, Blocked, config, read_json
from src.features import temporal_features
from src.training import fit_probe, predict_probe, load_probe
from src.metrics import paired_bootstrap
from src.checkpointing import save_torch


def inputs():
    torch.manual_seed(9)
    features = temporal_features(torch.randn(18,4),torch.randn(18,4))
    labels = torch.tensor([0.,1.]*9)
    cfg = config(ROOT / 'configs/experiment.yaml')['training']
    return features,labels,{**cfg,'max_epochs':5,'patience':3,'batch_size':4}


def test_interrupted_training_restores_optimizer_rng_shuffle_and_best(tmp_path,monkeypatch):
    import src.training as training
    features,labels,cfg = inputs()
    expected,expected_meta,expected_history = fit_probe('conshift',features,labels,features,labels,cfg)
    original_save = training.save_torch
    def interrupt(path,value):
        original_save(path,value)
        if str(path).endswith('.last.pt') and value['epoch'] == 2:
            raise InterruptedError('Stop after committing epoch two')
    monkeypatch.setattr(training,'save_torch',interrupt)
    with pytest.raises(InterruptedError):
        fit_probe('conshift',features,labels,features,labels,cfg,checkpoint_dir=tmp_path)
    state = torch.load(tmp_path/'conshift.last.pt',weights_only=True)
    assert state['epoch'] == 2 and state['optimizer_state']['state']
    monkeypatch.setattr(training,'save_torch',original_save)
    actual,meta,history = fit_probe('conshift',features,labels,features,labels,cfg,checkpoint_dir=tmp_path,resume=True)
    assert np.array_equal(predict_probe(expected,features),predict_probe(actual,features))
    assert expected_history == history
    assert meta['epoch'] == expected_meta['epoch'] and meta['threshold'] == expected_meta['threshold']
    best,best_meta = load_probe(tmp_path/'conshift.best.pt')
    assert np.array_equal(predict_probe(best,features),predict_probe(actual,features))
    assert best_meta['threshold'] == meta['threshold']
    # A finished checkpoint reload performs no further optimizer updates.
    loaded,loaded_meta,loaded_history = fit_probe('conshift',features,labels,features,labels,cfg,checkpoint_dir=tmp_path,resume=True)
    assert history == loaded_history and loaded_meta['epochs_completed'] == len(history)
    assert np.array_equal(predict_probe(actual,features),predict_probe(loaded,features))


def test_training_resume_rejects_changed_data_or_settings(tmp_path):
    features,labels,cfg = inputs()
    fit_probe('p1',features,labels,features,labels,cfg,checkpoint_dir=tmp_path)
    with pytest.raises(Blocked,match='use --resume'):
        fit_probe('p1',features,labels,features,labels,cfg,checkpoint_dir=tmp_path)
    with pytest.raises(Blocked,match='mismatch'):
        fit_probe('p1',features,1-labels,features,labels,cfg,checkpoint_dir=tmp_path,resume=True)
    with pytest.raises(Blocked,match='mismatch'):
        fit_probe('p1',features,labels,features,labels,{**cfg,'patience':1},checkpoint_dir=tmp_path,resume=True)


def test_atomic_model_write_keeps_previous_checkpoint_on_error(tmp_path):
    from src.checkpointing import atomic_dump
    p = tmp_path/'last.pt'
    save_torch(p,{'epoch':1})
    def broken(value,path):
        Path(path).write_bytes(b'partial')
        raise OSError('interrupted write')
    with pytest.raises(OSError):
        atomic_dump(p,{'epoch':2},broken)
    assert torch.load(p,weights_only=True)['epoch'] == 1
    assert not list(tmp_path.glob('*.tmp'))


def prediction_fixture():
    return [{'group_id':g,'failure_label':y,'probabilities':{'p1':p,'t1':p*.9,'conshift':p*.8}}
            for g in ('a','b','c') for y,p in ((0,.1),(1,.9))]


def test_bootstrap_resume_exact_random_sequence(tmp_path,monkeypatch):
    import src.metrics as metrics_module
    rows = prediction_fixture()
    expected = paired_bootstrap(rows,repetitions=120)
    path = tmp_path/'bootstrap.json'
    original_write = metrics_module.write_json
    def interrupt(path,value):
        original_write(path,value)
        if value['completed_repetitions'] == 50:
            raise InterruptedError('Stop after fifty replicates')
    monkeypatch.setattr(metrics_module,'write_json',interrupt)
    with pytest.raises(InterruptedError):
        paired_bootstrap(rows,repetitions=120,checkpoint_path=path)
    assert read_json(path)['completed_repetitions'] == 50
    monkeypatch.setattr(metrics_module,'write_json',original_write)
    assert paired_bootstrap(rows,repetitions=120,checkpoint_path=path,resume=True) == expected
    with pytest.raises(Blocked,match='mismatch'):
        paired_bootstrap(rows,repetitions=121,checkpoint_path=path,resume=True)


def test_evaluation_model_checkpoints_skip_completed_predictions(tmp_path,monkeypatch):
    import src.evaluation as evaluation
    from src.contracts import write_json
    from src.probes import PROBES
    cfg = config(ROOT/'configs/experiment.yaml')
    cfg['paths'] = {'data':str(tmp_path/'data'),'output':str(tmp_path/'output')}
    output = Path(cfg['paths']['output'])
    meta = {'threshold_source':'validation','threshold':.6,'cohort_hash':'cohort',
            'collection_fingerprint':'fp','parameter_count':5}
    write_json(output/'reports/training_manifest.json',{'cohort_hash':'cohort',
        'collection_fingerprint':'fp','target_config':cfg['target']})
    write_json(Path(cfg['paths']['data'])/'manifests/collection_lock.json',{'fingerprint':'fp'})
    for name in ('g1','l1'):
        write_json(output/'checkpoints'/f'{name}.json',meta)
    features,_,_ = inputs()
    monkeypatch.setattr(evaluation,'feature_tensors',lambda rows,cache:tuple(t[:2] for t in features))
    class BaselineFixture:
        def predict_proba(self,x):
            return np.array([[.8,.2],[.2,.8]])
    monkeypatch.setattr(evaluation.joblib,'load',lambda p:BaselineFixture())
    models = {name:cls(4) for name,cls in PROBES.items()}
    monkeypatch.setattr(evaluation,'load_probe',lambda p:(models[p.stem],meta))
    rows = [{'sample_id':str(i),'group_id':'g','trajectory_id':'t','turn_index':i+2,
        'failure_label':i,'split':'id_test','prompt_text':'','prompt_token_count':2,
        'prefill_seconds':0.,'generation_seconds':0.,'total_seconds_including_retries':0.} for i in (0,1)]
    checkpoint = output/'evaluation/id_test'
    first = evaluation.score_rows(rows,cfg,None,checkpoint_dir=checkpoint,run_signature='sig')
    def no_more_model_inference(_):
        raise AssertionError('Completed predictions must not rerun')
    monkeypatch.setattr(evaluation.joblib,'load',no_more_model_inference)
    monkeypatch.setattr(evaluation,'load_probe',no_more_model_inference)
    second = evaluation.score_rows(rows,cfg,None,checkpoint_dir=checkpoint,resume=True,run_signature='sig')
    assert first[0] == second[0] and first[1] == second[1] and first[3:] == second[3:]
    assert len(list(checkpoint.glob('*.json'))) == 5
    with pytest.raises(Blocked,match='mismatch'):
        evaluation.score_rows(rows,cfg,None,checkpoint_dir=checkpoint,resume=True,run_signature='changed')
