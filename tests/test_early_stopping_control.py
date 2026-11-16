"""가짜 optimizer와 모델로 중단 및 checkpoint 처리를 검사한다. 실제 학습은 하지 않는다."""
import ast
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time

import pytest
import yaml

from src.models.training_control import (CORE_VARIANTS, checkpoint_is_better, update_early_stopping,
                                         write_training_summary, write_training_comparison)


def config():
    return yaml.safe_load(Path('configs/train.yaml').read_text(encoding='utf-8'))


def metric(ap, brier):
    return dict(validation_auprc=ap, validation_brier=brier)


def test_patience_monitors_strict_auprc_brier_only_selects_checkpoint():
    state = {}
    best = None
    for epoch, values in enumerate([(.6,.2),(.6,.1),(.5,.05),(.6,.08)], 1):
        metrics = metric(*values)
        if checkpoint_is_better(metrics, best): best = dict(metrics, epoch=epoch)
        assert update_early_stopping(state, metrics, 3) is (epoch == 4)
    assert best['epoch'] == 4
    assert state['epochs_without_auprc_improvement'] == 3
    assert not update_early_stopping(state, metric(.61,.3), 3)
    assert state['epochs_without_auprc_improvement'] == 0
    assert not checkpoint_is_better(metric(.61,.3), metric(.61,.3))


def extract_function(name, namespace):
    tree = ast.parse(Path('train.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'train.py', 'exec'), namespace)
    return namespace[name]


def test_max_epoch_and_stopping_config_validation():
    validate = extract_function('validate_config', {})
    c = config()
    validate(c)
    assert c['max_epochs'] == 10 and c['seeds'] == [42]
    assert c['early_stopping']['patience'] == 3
    with pytest.raises(ValueError): validate(dict(c, seeds=[42,43,44]))
    for limit in (0, 11, True):
        with pytest.raises(ValueError): validate(dict(c, max_epochs=limit))
    for changes in ({'patience': 0}, {'monitor': 'validation_brier'}, {'tie_break_resets_patience': True}):
        with pytest.raises(ValueError):
            validate(dict(c, early_stopping={**c['early_stopping'], **changes}))


class Scalar:
    def __init__(self, value=0.): self.value = value
    def __add__(self, other): return Scalar(self.value + float(other))
    __radd__ = __add__
    def __mul__(self, other): return Scalar(self.value * float(other))
    __rmul__ = __mul__
    def __float__(self): return float(self.value)
    def detach(self): return self
    def backward(self): pass


def fake_training_namespace(tmp_path, metrics):
    snapshots = {}
    class Generator:
        def manual_seed(self, seed): return self
    class Optimizer:
        def __init__(self, parameters, lr, weight_decay): self.param_groups = [{'lr': lr}]
        def zero_grad(self, **kwargs): pass
        def step(self): pass
    class Scheduler:
        def __init__(self, *args): pass
        def step(self): pass
    class Model:
        variant = 'A'
        lambda_state, mu_relevance, nu_type = .5, .2, .5
        def to(self, device): pass
        def parameters(self): return []
        def train(self): pass
        def __call__(self, **batch): return {}
    def save(path, model, optimizer, scheduler, state, metadata):
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshots[str(path)] = dict(state=deepcopy(state), metadata=deepcopy(metadata))
    def load(path, model, optimizer, scheduler, metadata):
        assert metadata == snapshots[str(path)]['metadata']
        return deepcopy(snapshots[str(path)])
    metrics = iter(metrics)
    namespace = dict(
        math=math, Path=Path, time=time, digest=lambda _: 'fake-run',
        MODES={'A':'A', 'B_pred':'B_pred', 'C':'C', 'C_plus':'C+', 'C+':'C+', 'B_oracle':'B_oracle'},
        validate_config=extract_function('validate_config', {}),
        validation_metrics=lambda *args: None,
        torch=SimpleNamespace(device=lambda _: SimpleNamespace(type='cpu'), Generator=Generator,
            randperm=lambda n, **kw: SimpleNamespace(tolist=lambda: list(range(n))),
            isfinite=lambda value: True,
            optim=SimpleNamespace(AdamW=Optimizer, lr_scheduler=SimpleNamespace(LambdaLR=Scheduler)),
            nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=lambda *args, **kw: 0.))),
        DataLoader=lambda *args, **kwargs: [],
        _counts=lambda _: dict(failure=1, state={}, relevance=0),
        _to_device=lambda batch, device: batch,
        normalized_losses=lambda *args: {k: Scalar(.1) for k in ('failure','state','relevance','type')},
        validate=lambda *args: next(metrics), is_better=checkpoint_is_better,
        save_checkpoint=save, load_checkpoint=load, _write_history=lambda *args, **kw: None,
        CORE_VARIANTS=CORE_VARIANTS, update_early_stopping=update_early_stopping,
        write_training_summary=write_training_summary)
    return extract_function('train_model', namespace), Model, snapshots


@pytest.mark.parametrize('mode', ['A', 'B_pred', 'C', 'C_plus'])
def test_real_training_loop_stops_saves_tie_winner_and_resumes_without_updates(tmp_path, mode):
    values = [metric(.7,.2), metric(.7,.1), metric(.69,.05), metric(.7,.15)]
    train, Model, snapshots = fake_training_namespace(tmp_path, values)
    model = Model()
    model.variant = {'C_plus':'C+'}.get(mode, mode)
    c = config()
    c.update(per_device_batch_size=1, gradient_accumulation_steps=2, effective_batch_size=2)
    rows = [dict(failure_labels=0), dict(failure_labels=1)]
    metadata = dict(mode=mode, seed=42, data_kind='synthetic_test_only')
    directory = tmp_path/mode/'seed_42'/'trial_0'
    state = train(model, rows, rows, lambda _: {}, c, metadata, directory)
    assert state['epoch'] == state['global_step'] == 4
    assert state['best']['epoch'] == 2 and state['best']['validation_brier'] == .1
    assert state['stop_reason'] == 'early_stopping'
    assert snapshots[str(directory/'best.pt')]['state']['best']['epoch'] == 2
    assert snapshots[str(directory/'last.pt')]['state']['early_stopped'] is True
    summary = json.loads((directory/'training_summary.json').read_text())
    assert summary['best_epoch'] == 2 and summary['actual_optimizer_steps'] == 4
    assert summary['max_epochs'] == 10 and summary['training_time_seconds'] >= 0
    # 종료된 checkpoint를 재개할 때는 추가 갱신이나 검증이 없어야 한다.
    resumed = train(model, rows, rows, lambda _: {}, c, metadata, directory, resume=directory/'last.pt')
    assert resumed['global_step'] == 4
    comparison = write_training_comparison(tmp_path)
    assert comparison['runs'][0]['actual_optimizer_steps'] == 4
    assert 'different actual optimizer update steps' in comparison['comparison_note']


def test_interrupted_resume_preserves_patience_and_max_epoch_cap(tmp_path):
    train, Model, snapshots = fake_training_namespace(tmp_path, [metric(.5,.2)]*4)
    c = config()
    c.update(per_device_batch_size=1, gradient_accumulation_steps=2, effective_batch_size=2)
    rows = [dict(failure_labels=0), dict(failure_labels=1)]
    metadata = dict(mode='A', seed=42, data_kind='synthetic_test_only')
    first = train(Model(), rows, rows, lambda _: {}, c, metadata, tmp_path, stop_after_steps=2)
    assert first['epochs_without_auprc_improvement'] == 1
    resumed = train(Model(), rows, rows, lambda _: {}, c, metadata, tmp_path, resume=tmp_path/'last.pt')
    assert resumed['global_step'] == 4 and resumed['early_stopped']


def test_improving_validation_runs_to_ten_epochs(tmp_path):
    train, Model, _ = fake_training_namespace(tmp_path, [metric(i/10,.2) for i in range(1,11)])
    c = config()
    rows = [dict(failure_labels=0), dict(failure_labels=1)]
    state = train(Model(), rows, rows, lambda _: {}, c,
                  dict(mode='A', seed=42, data_kind='synthetic_test_only'), tmp_path)
    assert state['epoch'] == 10 and state['best']['epoch'] == 10
    assert state['stop_reason'] == 'max_epochs' and not state['early_stopped']


def test_partial_epoch_interrupt_does_not_consume_patience(tmp_path):
    train, Model, _ = fake_training_namespace(tmp_path, [])
    c = config()
    c.update(per_device_batch_size=1, gradient_accumulation_steps=1, effective_batch_size=1)
    rows = [dict(failure_labels=0), dict(failure_labels=1)]
    state = train(Model(), rows, rows, lambda _: {}, c,
                  dict(mode='A', seed=42, data_kind='synthetic_test_only'), tmp_path, stop_after_steps=1)
    summary = json.loads((tmp_path/'training_summary.json').read_text())
    assert state['epoch'] == 0 and state['next_batch'] == 1
    assert state['epochs_without_auprc_improvement'] == 0 and state['best'] is None
    assert summary['completed_epochs'] == 0 and summary['epochs_started'] == 1


def test_stopping_can_be_disabled_for_preserved_protocol(tmp_path):
    train, Model, _ = fake_training_namespace(tmp_path, [metric(.5,.2)]*10)
    c = config()
    c['early_stopping']['enabled'] = False
    rows = [dict(failure_labels=0), dict(failure_labels=1)]
    state = train(Model(), rows, rows, lambda _: {}, c,
                  dict(mode='A', seed=42, data_kind='synthetic_test_only'), tmp_path)
    assert state['epoch'] == 10 and not state['early_stopped']
