"""Interrupt actual CLI loops and verify completed inference is not repeated."""
import importlib.util
import csv
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from src.data.progress import Journal, CachedFailure
from test_automatic_generation import ROOT, MODEL, SHA, QwenTransport, canonical, family
from test_pipeline_integration import SyntheticTarget


def load(name, script):
    spec = importlib.util.spec_from_file_location(name, ROOT/script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_journal_recovers_torn_tail_and_rejects_changed_inputs(tmp_path):
    path = tmp_path/'progress.jsonl'
    journal = Journal(path, 'signature')
    assert journal.run('one', lambda: {'ok': True}) == {'ok': True}
    with path.open('ab') as stream:
        stream.write(b'{"incomplete":')
    resumed = Journal(path, 'signature', resume=True)
    assert resumed.run('one', lambda: pytest.fail('must not repeat')) == {'ok': True}
    assert resumed.run('two', lambda: 2) == 2
    with pytest.raises(ValueError, match='changed'):
        Journal(path, 'different', resume=True)


def test_journal_preserves_failures_and_can_retry_unfinished_operations(tmp_path):
    journal = Journal(tmp_path/'progress.jsonl', 'signature')
    def fail():
        raise TimeoutError('private text')
    with pytest.raises(TimeoutError):
        journal.run('failed', fail)
    with pytest.raises(CachedFailure):
        journal.run('failed', lambda: pytest.fail('must not repeat'))
    with pytest.raises(TimeoutError):
        journal.run('retry', fail, cache_failures=False)
    assert journal.run('retry', lambda: 42) == 42
    assert 'private text' not in journal.path.read_text()


def test_generation_restart_reuses_normalization_and_completed_samples(tmp_path, monkeypatch):
    module = load('resume_prepare', 'scripts/02_prepare.py')
    monkeypatch.setattr(module, 'QwenClient', QwenTransport)
    import src.data.source_policy as policy
    monkeypatch.setattr(policy, 'enforce_source_policy', lambda *args: None)
    import src.data.automatic as automatic
    normalizations = []
    original_seed = automatic.create_seed
    def create(seed, client):
        normalizations.append(seed['seed_id'])
        return original_seed(seed, client)
    monkeypatch.setattr(automatic, 'create_seed', create)
    original_execute = module.execute
    attempts = []
    interrupt = [True]
    def execute(row, *args):
        attempts.append(row['sample_id'])
        if len(attempts) == 2 and interrupt[0]:
            raise KeyboardInterrupt()
        return original_execute(row, *args)
    monkeypatch.setattr(module, 'execute', execute)
    source, out, audit = tmp_path/'input.jsonl', tmp_path/'generated.jsonl', tmp_path/'audit.csv'
    source.write_text(''.join(json.dumps(canonical(i))+'\n' for i in range(3)))
    argv = ['02_prepare.py', '--input', str(source), '--endpoint', 'fixture://offline', '--model', MODEL,
            '--revision', SHA, '--limit-seeds', '3', '--output', str(out), '--audit', str(audit)]
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(KeyboardInterrupt):
        module.main()
    assert len(out.read_text(encoding='utf-8').splitlines()) == 1
    with audit.open(encoding='utf-8', newline='') as stream:
        live_audit = list(csv.DictReader(stream))
    assert len(live_audit) == 1
    assert live_audit[0]['sample_id'] == attempts[0]
    assert live_audit[0]['controlled_family_status'] == 'pending'
    assert json.loads(out.read_text(encoding='utf-8'))['controlled_family_status'] == 'pending'
    # Start naturalization after the first accepted source, before normalizing all sources.
    assert len(normalizations) == 1
    assert audit.with_suffix('.events.csv').exists()
    assert audit.with_suffix('.stages.csv').exists()
    raw_events = [json.loads(line) for line in audit.with_suffix('.model.jsonl').read_text(encoding='utf-8').splitlines()]
    assert {r['stage'] for r in raw_events} == {'normalization','qwen_paraphrase'}
    first = attempts[0]
    interrupt[0] = False
    monkeypatch.setattr(sys, 'argv', argv+['--resume'])
    module.main()
    rows = [json.loads(line) for line in out.read_text(encoding='utf-8').splitlines()]
    assert attempts.count(first) == 1
    assert len(normalizations) == 3
    assert len({r['sample_id'] for r in rows}) == len(rows)
    assert all(r['controlled_family_status'] == 'semantic_complete_oracle_verified' for r in rows)
    with audit.open(encoding='utf-8', newline='') as stream:
        completed_audit = list(csv.DictReader(stream))
    samples = [r for r in completed_audit if r['row_type'] == 'sample']
    assert len(samples) == len(rows)
    assert all(r['controlled_family_status'] == 'semantic_complete_oracle_verified' for r in samples)
    before = len(attempts)
    module.main()
    assert len(attempts) == before


def test_prediction_export_restart_keeps_journal_out_of_evaluation_inputs(tmp_path, monkeypatch):
    import export_predictions as module
    import src.data.approvals as approvals
    import src.evaluation.protocol as protocol
    monkeypatch.setattr(approvals, 'research_rows', lambda *args: None)
    monkeypatch.setattr(approvals, 'enforce_execution', lambda *args, **kwargs: None)
    monkeypatch.setattr(protocol, '_valid', lambda row: True)
    rows = [dict(sample_id=str(i)) for i in range(3)]
    monkeypatch.setattr(module, 'read_rows', lambda path: rows)
    fake_torch = ModuleType('torch')
    fake_torch.load = lambda *args, **kwargs: dict(metadata=dict(seed=42, mode='C', data_kind='model_generated',
        model_config=dict(encoder=dict(model_id='fixture', revision=SHA, max_length=128))))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    dataset = ModuleType('src.models.dataset')
    dataset.load_tokenizer = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, 'src.models.dataset', dataset)
    predictor = ModuleType('src.models.predictor')
    predictor.ModernBertPredictor = SimpleNamespace(from_checkpoint=lambda *args, **kwargs: SimpleNamespace(variant='C'))
    monkeypatch.setitem(sys.modules, 'src.models.predictor', predictor)
    exporter = ModuleType('src.evaluation.export')
    attempts = []
    def predict(model, tokenizer, items, **kwargs):
        sid = items[0]['sample_id']
        attempts.append(sid)
        if len(attempts) == 2:
            raise KeyboardInterrupt()
        yield dict(sample_id=sid, failure_probability=.5), dict(sample_id=sid, cls=[1.])
    exporter.predict_records = predict
    monkeypatch.setitem(sys.modules, 'src.evaluation.export', exporter)
    checkpoint = tmp_path/'best.pt'
    checkpoint.write_bytes(b'fixture checkpoint')
    out, embeddings = tmp_path/'predictions/C.jsonl', tmp_path/'embeddings/C.jsonl'
    argv = ['--checkpoint', str(checkpoint), '--records', 'unused', '--out', str(out),
            '--embeddings-out', str(embeddings)]
    with pytest.raises(KeyboardInterrupt):
        module.main(argv)
    assert len(out.read_text().splitlines()) == 1
    module.main(argv+['--resume'])
    assert attempts.count('0') == 1
    assert len(out.read_text().splitlines()) == len(embeddings.read_text().splitlines()) == 3
    assert list(out.parent.glob('*.jsonl')) == [out]


def test_activation_restart_reuses_saved_features(tmp_path, monkeypatch):
    import yaml
    module = load('resume_activations', 'scripts/04_export_target_pregen.py')
    import evaluate
    import src.data.approvals as approvals
    monkeypatch.setattr(approvals, 'research_rows', lambda *args: None)
    monkeypatch.setattr(approvals, 'enforce_execution', lambda *args, **kwargs: None)
    metadata = dict(model_id=MODEL, model_revision=SHA, chat_template_hash='fixture')
    rows = [dict(sample_id=str(i), target_execution=metadata, prompt_hash=module.sha(str(i))) for i in range(3)]
    monkeypatch.setattr(evaluate, 'read_rows', lambda path: rows)
    monkeypatch.setattr(module, 'partitions', lambda *args: (None, None, rows))
    monkeypatch.setattr(module, 'target_prompt', lambda row, template: row['sample_id'])
    monkeypatch.setattr(module, 'validate_activations', lambda *args: None)
    attempts = []
    def extract(prompt):
        attempts.append(prompt)
        if len(attempts) == 2:
            raise KeyboardInterrupt()
        return [1.], {}
    monkeypatch.setattr(module, 'LocalTarget', lambda revision: SimpleNamespace(metadata=metadata, pre_generation=extract))
    fake_torch = ModuleType('torch')
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    out, config, manifest, prompt = (tmp_path/name for name in ['activations.jsonl','config.yaml','manifest.json','prompt.txt'])
    prompt.write_text('fixture')
    manifest.write_text('{}')
    config.write_text(yaml.safe_dump(dict(activations=str(out), manifest=str(manifest), records='unused',
        target_revision=SHA, target_prompt=str(prompt))))
    with pytest.raises(KeyboardInterrupt):
        module.main(['--config', str(config)])
    assert len(out.read_text().splitlines()) == 1
    module.main(['--config', str(config), '--resume'])
    assert attempts.count('0') == 1
    assert len(out.read_text().splitlines()) == 3


def test_target_restart_does_not_repeat_saved_observations(tmp_path, monkeypatch):
    module = load('resume_target', 'scripts/03_label_and_split.py')
    import src.data.approvals as approvals
    monkeypatch.setattr(approvals, 'enforce_execution', lambda *args, **kwargs: None)
    monkeypatch.setattr(module, 'LocalTarget', lambda revision: SyntheticTarget())
    original = module.run_target
    attempts = []
    def observe(row, *args):
        attempts.append(row['sample_id'])
        if len(attempts) == 2:
            raise KeyboardInterrupt()
        return original(row, *args)
    monkeypatch.setattr(module, 'run_target', observe)
    rows = family()
    source, out, report = tmp_path/'input.jsonl', tmp_path/'target.jsonl', tmp_path/'report.csv'
    source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    argv = ['03_label_and_split.py', '--input', str(source), '--execute', '--full-run', '--revision', SHA,
            '--labeled-output', str(out), '--report', str(report), '--splits-dir', str(tmp_path/'splits')]
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(KeyboardInterrupt):
        module.main()
    assert len(out.read_text(encoding='utf-8').splitlines()) == 1
    first = attempts[0]
    monkeypatch.setattr(sys, 'argv', argv+['--resume'])
    module.main()
    assert attempts.count(first) == 1
    assert len(out.read_text(encoding='utf-8').splitlines()) == len(rows)
    before = len(attempts)
    module.main()
    assert len(attempts) == before
