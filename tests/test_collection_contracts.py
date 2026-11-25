"""Unit protocol tests use original user turns and tensor-only inference doubles.

All stub outputs stay in pytest tmp_path; they are never research artifacts.
"""
from pathlib import Path
import copy
import pytest
import torch
from src.contracts import ROOT, config, digest, read_json, read_jsonl, write_json, write_jsonl
from src.evolif_adapter import normalize
from src.splitting import make_split
from src.target import collect, FrozenTarget


@pytest.fixture
def setup_collection(tmp_path, monkeypatch):
    cfg = config(ROOT / 'configs/experiment.yaml')
    cfg['paths'] = {'data':str(tmp_path / 'data'), 'output':str(tmp_path / 'output')}
    data = Path(cfg['paths']['data'])
    # Two real original turns per trajectory suffice to check replay contracts.
    original = normalize(ROOT / 'data/raw/official_data/dialog_v0.1', cfg['source']['dataset_revision'])
    trajectories = sorted({r['trajectory_id'] for r in original})[:10]
    rows = [r for r in original if r['trajectory_id'] in trajectories and r['turn_index'] <= 2]
    lock = {'dataset_revision':cfg['source']['dataset_revision']}
    lock['manifest_hash'] = digest(lock)
    write_json(data / 'manifests/revision_lock.json',lock)
    manifest = make_split(rows,lock['manifest_hash'])
    write_json(data / 'manifests/split_manifest.json',manifest)
    write_jsonl(data / 'normalized/turns.jsonl',rows)
    write_json(Path(cfg['paths']['output'])/'reports/progress.json',{'G2':{'status':'PASS'}})
    events = []
    class TensorTarget:
        dtype = torch.float32
        tokenizer = type('TemplateFixture',(),{'chat_template':'tensor-unit-fixture'})()
        def __init__(self,cfg):
            pass
        def prepare(self,history):
            events.append(('prepare',copy.deepcopy(history)))
            return str(history), {'input_ids':torch.tensor([[1,2]]), 'attention_mask':torch.tensor([[1,1]])}
        def representation(self,inputs):
            events.append('extract')
            return torch.tensor([1.,2.]),1
        def generate(self,inputs):
            events.append('generate')
            return '',False,0
    monkeypatch.setattr('src.target.FrozenTarget',TensorTarget)
    return cfg,data,rows,events,TensorTarget


def test_cache_replay_skips_inference_and_retains_target_history(setup_collection):
    cfg,data,rows,events,_ = setup_collection
    collect(cfg,resume=True)
    first_count = events.count('generate')
    assert first_count == len(rows)
    assert events[1:3] == ['extract','generate']
    histories = [x[1] for x in events if isinstance(x,tuple)]
    assert len(histories[0]) == 1 and len(histories[1]) == 3
    assert histories[1][1] == {'role':'assistant','content':''}
    assert all(message['role'] != 'system' for history in histories for message in history)
    records = [read_json(p) for p in (data / 'responses').glob('*.json')]
    assert all(r['model_revision'] == cfg['target']['revision'] for r in records)
    assert all(r['extraction_token_index'] == 1 for r in records)
    collect(cfg,resume=True)
    assert events.count('generate') == first_count


def test_generation_retries_once_and_blocks_dependents(setup_collection,monkeypatch):
    cfg,data,rows,events,target = setup_collection
    def fail(self,inputs):
        events.append('failed')
        raise RuntimeError('Tensor-only technical-error unit fixture')
    monkeypatch.setattr(target,'generate',fail)
    collect(cfg)
    ntraj = len({r['trajectory_id'] for r in rows})
    assert events.count('failed') == 2*ntraj
    assert all(read_json(p)['status'] == 'TECHNICAL_ERROR' for p in (data / 'responses').glob('*.json'))


def test_context_overflow_never_truncates_or_generates(setup_collection,monkeypatch):
    cfg,data,rows,events,target = setup_collection
    def too_long(self,history):
        length = cfg['target']['max_context_tokens']
        return '',{'input_ids':torch.ones(1,length,dtype=torch.long),'attention_mask':torch.ones(1,length,dtype=torch.long)}
    monkeypatch.setattr(target,'prepare',too_long)
    collect(cfg)
    assert 'generate' not in events and 'extract' not in events
    assert all(read_json(p)['status'] == 'CONTEXT_OVERFLOW' for p in (data / 'responses').glob('*.json'))


def test_truncation_stays_in_history_without_blocking_next_turn(setup_collection,monkeypatch):
    cfg,data,rows,events,target = setup_collection
    def truncated(self,inputs):
        events.append('generate')
        return '',True,cfg['target']['max_new_tokens']
    monkeypatch.setattr(target,'generate',truncated)
    collect(cfg)
    assert events.count('generate') == len(rows)
    assert all(read_json(p)['status'] == 'OK' and read_json(p)['truncated'] for p in (data / 'responses').glob('*.json'))


def test_real_target_refuses_cpu_before_weight_loading(monkeypatch):
    # Keep a direct reference to the real class despite inference doubles in other tests.
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA_GPU_NOT_AVAILABLE'):
        FrozenTarget(config(ROOT / 'configs/experiment.yaml'))


def test_dead_worker_stops_collection_and_closes_without_poisoning_cache(setup_collection, monkeypatch):
    from src.contracts import Blocked
    cfg, data, rows, events, target = setup_collection
    def fail(self, inputs):
        raise Blocked('vLLM worker exited unexpectedly')
    monkeypatch.setattr(target, 'generate', fail)
    monkeypatch.setattr(target, 'close', lambda self: events.append('closed'), raising=False)
    with pytest.raises(Blocked, match='worker exited'):
        collect(cfg)
    assert events[-1] == 'closed'
    assert not list((data / 'responses').glob('*.json'))


def test_successful_collection_closes_worker_and_locks_runtime(setup_collection, monkeypatch):
    cfg, data, rows, events, target = setup_collection
    versions = {'vllm': 'unit-version'}
    monkeypatch.setattr(target, 'runtime_versions', versions, raising=False)
    monkeypatch.setattr(target, 'close', lambda self: events.append('closed'), raising=False)
    collect(cfg)
    assert events[-1] == 'closed'
    assert read_json(data / 'manifests/collection_lock.json')['versions'] == versions
