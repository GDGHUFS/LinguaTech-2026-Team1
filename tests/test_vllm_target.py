"""CPU checks of token identity, extraction equivalence and worker isolation."""
import copy
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from src.contracts import ROOT, Blocked, config
from src.representation import extract
from src.target import FrozenTarget
from src import vllm_worker


@pytest.mark.parametrize('update', [
    {'generation_gpus': [1, 1]}, {'generation_gpus': [True, 1]},
    {'generation_gpus': [0]}, {'generation_gpus': [0, -1]},
    {'tensor_parallel_size': 1}, {'tensor_parallel_size': True},
    {'gpu_memory_utilization': .8},
    {'gpu_memory_utilization': 0}, {'gpu_memory_utilization': float('nan')},
    {'max_num_batched_tokens': -1}, {'request_timeout_seconds': 0},
    {'enforce_eager': 'false'},
])
def test_invalid_two_gpu_configuration(tmp_path, update):
    cfg = config(ROOT / 'configs/experiment.yaml')
    cfg['target']['vllm'].update(update)
    path = tmp_path / 'invalid.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    with pytest.raises(Blocked):
        config(path)


def test_scheduler_visible_device_mapping(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-first, GPU-second')
    assert vllm_worker.visible_device(0) == 'GPU-first'
    assert vllm_worker.visible_device(1) == 'GPU-second'
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    assert vllm_worker.visible_device(1) == '1'


def test_backbone_vector_matches_original_causal_lm_hidden_state():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(42)
    cfg = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    model = Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    inputs = {'input_ids': torch.tensor([[2, 4, 8, 0]]),
              'attention_mask': torch.tensor([[1, 1, 1, 0]])}
    with torch.inference_mode():
        reference = model(**inputs, output_hidden_states=True, use_cache=False).hidden_states[-1][0, 2]
    vector, index = extract(model.model, inputs)
    torch.testing.assert_close(vector, reference.float(), rtol=0, atol=0)
    assert index == 2 and not vector.requires_grad


@pytest.mark.parametrize('ids,finish,truncated', [
    ([5, 6], 'stop', False), ([5, 6], 'length', True), ([5, 9], 'length', False),
])
def test_vllm_generation_preserves_token_ids_and_eos(ids, finish, truncated):
    target = FrozenTarget.__new__(FrozenTarget)
    target.eos_ids = [9]
    sent = []
    target.worker = SimpleNamespace(generate=lambda prompt: (sent.append(prompt) or ids, finish))
    target.tokenizer = SimpleNamespace(decode=lambda tokens, **kw: str(tokens))
    text, cut, count = target.generate({'input_ids': torch.tensor([[2, 3, 4]])})
    assert sent == [[2, 3, 4]]
    assert (text, cut, count) == (str(ids), truncated, len(ids))


def test_vllm_child_uses_two_gpu_tensor_parallel_and_exact_tokens(monkeypatch):
    settings = config(ROOT / 'configs/experiment.yaml')['target']
    events = []
    class Engine:
        def __init__(self, **kwargs):
            assert vllm_worker.os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-first,GPU-second'
            events.append(kwargs)
            self.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(shutdown=lambda: events.append('shutdown')))
        def generate(self, prompts, params, **kwargs):
            assert prompts == [{'prompt_token_ids': [2, 3, 4]}]
            assert params['temperature'] == 0 and params['stop_token_ids'] == [9, 10]
            assert params['max_tokens'] == 1024 and not params['detokenize']
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[5, 9], finish_reason='stop')])]
    class Connection:
        def __init__(self):
            self.requests = iter([[2, 3, 4], None])
            self.messages = []
        def recv(self):
            return next(self.requests)
        def send(self, message):
            self.messages.append(message)
        def close(self):
            events.append('closed')
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=Engine, SamplingParams=lambda **kw: kw))
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _: 'test-device')
    monkeypatch.setattr(vllm_worker.importlib.metadata, 'version', lambda _: 'unit-version')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'old-mask')
    monkeypatch.setenv('VLLM_WORKER_MULTIPROC_METHOD', 'old-value')
    if hasattr(vllm_worker.os, 'setsid'):
        monkeypatch.setattr(vllm_worker.os, 'setsid', lambda: None)
    connection = Connection()
    vllm_worker._serve(connection, copy.deepcopy(settings), 'GPU-first,GPU-second', [9, 10], 42)
    engine_settings = events[0]
    assert engine_settings['revision'] == settings['revision']
    assert engine_settings['tokenizer_revision'] == settings['revision']
    assert engine_settings['tensor_parallel_size'] == 2
    assert engine_settings['distributed_executor_backend'] == 'mp'
    assert connection.messages[0][1]['tensor_parallel_size'] == 2
    assert len(connection.messages[0][1]['device_names']) == 2
    assert engine_settings['enable_prefix_caching'] and engine_settings['enable_chunked_prefill']
    assert connection.messages[0][0] == 'ready'
    assert connection.messages[1] == ('result', ([5, 9], 'stop'))
    assert events[-2:] == ['shutdown', 'closed']


def test_worker_timeout_is_fatal_and_closes():
    worker = vllm_worker.VllmWorker.__new__(vllm_worker.VllmWorker)
    worker.connection = SimpleNamespace(poll=lambda _: False)
    events = []
    worker.close = lambda: events.append('close')
    with pytest.raises(Blocked, match='timed out'):
        worker._receive('result', 1)
    assert events == ['close']
