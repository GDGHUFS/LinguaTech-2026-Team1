"""Fault injection: a bad item must not stop later healthy items."""
import importlib.util
import json
import sys

import pytest

from src.data.batch import read_items
from src.data.generation import execute
from src.data.mutations import make_variants
from src.data.automatic import create_seed
from test_automatic_generation import ROOT, SHA, MODEL, QwenTransport, canonical
from test_automatic_generation import family
from test_pipeline_integration import SyntheticTarget


def test_bad_json_and_identity_do_not_discard_later_rows(tmp_path, capsys):
    path = tmp_path/'input.jsonl'
    path.write_text('{bad json}\n[]\n{"seed_id": null}\n{"seed_id": "ok"}\n')
    failures = []
    assert read_items(path, failures, required=('seed_id',)) == [{'seed_id': 'ok'}]
    assert [r['line'] for r in failures] == [1, 2, 3]
    assert capsys.readouterr().err.count('WARNING') == 3


@pytest.mark.parametrize('raw', ['broken JSON', 'null', '[]', '{"turns": 42}'])
def test_bad_model_response_does_not_stop_next_call(raw):
    seed = create_seed(canonical(), QwenTransport())
    sample = make_variants(seed)[0]
    bad = QwenTransport()
    bad.complete = lambda prompt: raw
    assert execute(sample, bad, ROOT/'prompts')['validation_status'] != 'semantic_pass_oracle_verified'
    assert execute(sample, QwenTransport(), ROOT/'prompts')['validation_status'] == 'semantic_pass_oracle_verified'


@pytest.mark.parametrize('stage', ['preparation', 'structure', 'family', 'gold_contract'])
def test_prepare_cli_keeps_other_families_after_unexpected_error(tmp_path, monkeypatch, capsys, stage):
    spec = importlib.util.spec_from_file_location('resilient_prepare', ROOT/'scripts/02_prepare.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'QwenClient', QwenTransport)
    import src.data.source_policy as policy
    monkeypatch.setattr(policy, 'enforce_source_policy', lambda *args: None)
    seeds = [canonical(i) for i in range(3)]
    bad_id = seeds[0]['seed_id']
    if stage == 'preparation':
        original = module.make_variants
        def failing(seed, *args, **kwargs):
            if seed['seed_id'] == bad_id:
                raise RuntimeError('injected')
            return original(seed, *args, **kwargs)
        monkeypatch.setattr(module, 'make_variants', failing)
    elif stage == 'structure':
        import src.data.validators as validators
        original = validators.validate_structure
        def failing(row):
            if row['seed_id'] == bad_id:
                raise AttributeError('injected')
            return original(row)
        monkeypatch.setattr(validators, 'validate_structure', failing)
    elif stage == 'family':
        original = module.validate_controlled_family
        def failing(rows):
            if rows[0]['seed_id'] == bad_id:
                raise RuntimeError('injected')
            return original(rows)
        monkeypatch.setattr(module, 'validate_controlled_family', failing)
    else:
        import src.data.contracts as contracts
        original = contracts.validate_pipeline_record
        def failing(row, **kwargs):
            if row['seed_id'] == bad_id:
                raise RuntimeError('injected')
            return original(row, **kwargs)
        monkeypatch.setattr(contracts, 'validate_pipeline_record', failing)
    source, output, audit = tmp_path/'canonical.jsonl', tmp_path/'generated.jsonl', tmp_path/'audit.csv'
    source.write_text(''.join(json.dumps(r)+'\n' for r in seeds))
    monkeypatch.setattr(sys, 'argv', ['02_prepare.py', '--input', str(source), '--endpoint', 'fixture://offline',
        '--model', MODEL, '--revision', SHA, '--limit-seeds', '3', '--execution-mode', 'final',
        '--output', str(output), '--audit', str(audit)])
    module.main()
    rows = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
    good = [r for r in rows if r['controlled_family_status'] == 'semantic_complete_oracle_verified']
    assert {r['seed_id'] for r in good} == {s['seed_id'] for s in seeds[1:]}
    assert audit.is_file() and audit.with_suffix('.source.jsonl').is_file()
    assert 'WARNING' in capsys.readouterr().err


@pytest.mark.parametrize('stage', ['input_validation', 'target', 'contract'])
def test_target_cli_continues_after_item_failure(tmp_path, monkeypatch, stage):
    spec = importlib.util.spec_from_file_location('resilient_target', ROOT/'scripts/03_label_and_split.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = family()
    import src.data.approvals as approvals
    def gate(items, **kwargs):
        if stage == 'input_validation' and items[0].get('variant_group_id') == 'bad-family':
            raise RuntimeError('injected')
    monkeypatch.setattr(approvals, 'enforce_execution', gate)
    if stage == 'input_validation':
        bad = dict(rows[0], sample_id='bad', variant_group_id='bad-family')
        rows.insert(0, bad)
    elif stage == 'target':
        original = module.run_target
        def failing(row, *args):
            if row['sample_id'] == rows[0]['sample_id']:
                raise RuntimeError('injected')
            return original(row, *args)
        monkeypatch.setattr(module, 'run_target', failing)
    else:
        import src.data.contracts as contracts
        original = contracts.validate_pipeline_record
        def failing(row, **kwargs):
            if kwargs.get('stage') == 'target' and row['sample_id'] == rows[0]['sample_id']:
                raise RuntimeError('injected')
            return original(row, **kwargs)
        monkeypatch.setattr(contracts, 'validate_pipeline_record', failing)
    monkeypatch.setattr(module, 'LocalTarget', lambda revision: SyntheticTarget())
    source, output, report = tmp_path/'input.jsonl', tmp_path/'target.jsonl', tmp_path/'report.csv'
    source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    monkeypatch.setattr(sys, 'argv', ['03_label_and_split.py', '--input', str(source),
        '--revision', SHA, '--execute', '--full-run', '--labeled-output', str(output),
        '--splits-dir', str(tmp_path/'splits'), '--report', str(report)])
    module.main()
    results = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
    assert len(results) == len(rows)-1
    assert all(r['parse_status'] == 'VALID' for r in results)
    failures = [json.loads(line) for line in report.with_suffix('.errors.jsonl').read_text().splitlines()]
    assert len(failures) == 1
    assert (tmp_path/'splits/split_manifest.json').is_file()
