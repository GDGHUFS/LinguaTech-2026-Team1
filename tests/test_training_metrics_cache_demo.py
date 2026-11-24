from pathlib import Path
from types import SimpleNamespace
import copy
import numpy as np
import pytest
import torch
from src.contracts import ROOT, Blocked, config, digest, code_hash
from src.representation import SampleCache
from src.features import temporal_features
from src.probes import ConShiftProbe
from src.training import fit_probe, load_probe, predict_probe
from src.baselines import build_baselines, baseline_input
from src.metrics import metrics, calibration, select_threshold, paired_bootstrap
from src.privacy_demo import privacy_check, PrivacyDemo
from src.evaluation import assert_cohort


def test_cache_hash_integrity_and_commit(tmp_path):
    cache = SampleCache(tmp_path)
    record = {'sample_id':'tensor-fixture', 'cache_fingerprint':'fp', 'prompt_hash':'p', 'prompt_token_ids_hash':'t'}
    assert cache.load('tensor-fixture','fp') is None
    cache.commit(record,torch.tensor([1.,2.]))
    assert cache.load('tensor-fixture','fp','p','t')['hidden_dim'] == 2
    for args in [('bad','p','t'), ('fp','bad','t'), ('fp','p','bad')]:
        with pytest.raises(Blocked):
            cache.load('tensor-fixture',*args)
    (cache.vectors / record['representation_key']).write_bytes(b'corrupt')
    with pytest.raises(Blocked):
        cache.load('tensor-fixture','fp')


def test_metrics_threshold_and_probability_one():
    y,p = [0,1,0,1], [.1,.8,.2,.9]
    assert metrics(y,p,.5)['AP'] == 1
    assert select_threshold(y,p) == .8
    # All predictions tied: best F1 is achieved at the highest positive threshold.
    assert select_threshold([0,1],[.5,.5]) == .5
    with pytest.raises(Blocked):
        select_threshold(y,p,split='id_test')
    result = metrics([1,1],[.5,1.],.5)
    assert result['AP'] is None and result['AUROC'] is None and result['Brier'] is not None
    ece,bins = calibration([0,1],[0.,1.])
    assert ece == 0 and [b['bin'] for b in bins] == [0,9]


def test_probe_fit_seed_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(3)
    features = temporal_features(torch.randn(12,4),torch.randn(12,4))
    labels = torch.tensor([0.,1.]*6)
    cfg = config(ROOT / 'configs/experiment.yaml')['training']
    cfg = {**cfg,'max_epochs':3,'patience':2}
    m1,meta1,_ = fit_probe('conshift',features,labels,features,labels,cfg)
    m2,meta2,_ = fit_probe('conshift',features,labels,features,labels,cfg)
    assert np.array_equal(predict_probe(m1,features),predict_probe(m2,features))
    assert meta1['threshold'] == meta2['threshold']
    p = tmp_path / 'probe.pt'
    torch.save({'state_dict':m1.state_dict(),'metadata':meta1},p)
    loaded,_ = load_probe(p)
    assert np.array_equal(predict_probe(m1,features),predict_probe(loaded,features))
    assert meta1['parameter_count'] == 11


def test_probe_input_does_not_consume_metadata():
    from src.training import feature_tensors
    class TensorFixtureCache:
        def vector(self,key):
            return {'now':torch.tensor([1.,2.]),'prev':torch.tensor([2.,1.])}[key]
    row = {'representation_key':'now','previous_representation_key':'prev',
           'target_response':'', 'active_constraints':[], 'future_turn':99}
    before = feature_tensors([row],TensorFixtureCache())
    row.update(target_response='changed fixture',active_constraints=[{'gold':'ignored'}],future_turn=-1)
    after = feature_tensors([row],TensorFixtureCache())
    assert all(torch.equal(a,b) for a,b in zip(before,after))


def test_baselines_fixed_config_and_train_only_scaler():
    models = build_baselines()
    assert models['g1'][0].ngram_range == (1,2)
    assert models['g1'][0].min_df == 2
    assert models['g1'][-1].C == 1.
    x = np.array([[0.,1.],[1.,2.],[2.,3.],[3.,4.]])
    models['l1'].fit(x,[0,1,0,1])
    mean = models['l1'][0].mean_.copy()
    models['l1'].predict_proba(np.array([[100.,200.]]))
    assert np.array_equal(mean,models['l1'][0].mean_)


def test_paired_bootstrap_uses_whole_groups_and_same_predictions():
    rows = [{'group_id':g, 'failure_label':y,
             'probabilities':{'p1':p,'t1':p,'conshift':p}}
            for g in ['a','b'] for y,p in [(0,.1),(1,.9)]]
    result = paired_bootstrap(rows,repetitions=30)
    assert result['valid_replicates'] == 30  # each group contains both classes
    assert result['stable'] is False
    assert result['differences']['t1-p1']['percentile_95_ci'] == [0.,0.]
    assert result == paired_bootstrap(rows,repetitions=30)


def test_privacy_exact_nfkc_phone_and_no_partial_names():
    identifiers = [{'value':'SYNTH-X9'}, {'value':'010-1234-5678','kind':'phone'}, {'value':'Synthetic Full Name'}]
    assert privacy_check('ＳＹＮＴＨ－Ｘ９',identifiers)['result'] == 'FAIL'
    assert privacy_check('(010) 1234 5678',identifiers)['result'] == 'FAIL'
    assert privacy_check('Full',identifiers)['result'] == 'PASS'


def test_demo_missing_checkpoint_never_fabricates_risk(tmp_path):
    cfg = config(ROOT / 'configs/experiment.yaml')
    cfg['paths']['output'] = str(tmp_path)
    demo = PrivacyDemo(cfg)
    assert demo.step('',[],'')['status'] == 'CHECKPOINT_NOT_AVAILABLE'
    assert demo.step('',[],'')['risk'] is None


def test_demo_first_turn_history_and_risk_before_generation(tmp_path):
    # Tensor-only unit stub; no experimental conversations or responses are exported.
    cfg = config(ROOT / 'configs/experiment.yaml')
    cfg['paths']['output'] = str(tmp_path)
    cp = tmp_path / 'checkpoints'
    cp.mkdir()
    model = ConShiftProbe(2)
    meta = {'model_name':'conshift','hidden_dim':2,'threshold':.7,
            'threshold_source':'validation','target_config':cfg['target']}
    import importlib.metadata
    meta['representation_protocol'] = {'dtype':str(torch.float32), 'template_hash':digest('tensor-demo-fixture'),
        'code_hash':code_hash(), 'versions':{n:importlib.metadata.version(n) for n in ('torch','transformers','safetensors')}}
    torch.save({'state_dict':model.state_dict(),'metadata':meta},cp / 'conshift.pt')
    events = []
    class TensorTarget:
        dtype = torch.float32
        tokenizer = type('TemplateFixture',(),{'chat_template':'tensor-demo-fixture'})()
        def prepare(self,history):
            assert all(h['role'] in {'user','assistant'} for h in history)
            events.append(('history',len(history)))
            return '',{'input_ids':torch.tensor([[1,2]])}
        def representation(self,inputs):
            events.append('extract')
            return torch.tensor([1.,2.]),1
        def generate(self,inputs):
            events.append('generate')
            return '',False,0
    demo = PrivacyDemo(cfg,target=TensorTarget())
    first = demo.step('',[],'',on_risk=lambda r:events.append(('risk',r['risk'])))
    assert first['risk'] is None
    second = demo.step('',[],'',on_risk=lambda r:events.append(('risk',r['risk'])))
    assert second['risk'] is not None and second['threshold'] == .7
    assert events[1:4] == ['extract',('risk',None),'generate']
    assert events[4] == ('history',3)
    assert second['turn'] == 2


def test_same_cohort_contract():
    with pytest.raises(Blocked):
        assert_cohort([{'sample_id':'x','probabilities':{'p1':.5}}])
