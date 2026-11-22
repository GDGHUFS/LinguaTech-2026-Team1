"""Multi-turn selection must preserve policy meaning and replayable model evidence."""
import json

import pytest

from test_guarded_dialogue import seed213, variants
from test_automatic_generation import QwenTransport, ROOT
from src.data.generation import execute
from src.data.guarded_dialogue import blueprint, with_schedule, validate_record
from src.data.contracts import validate_pipeline_record


class BatchSelector:
    model = QwenTransport.model
    serving_provenance = QwenTransport().serving_provenance

    def __init__(self, damage=None):
        self.damage = damage
        self.calls = 0

    def verify_revision(self):
        return self.serving_provenance

    def complete_json(self, prompt, schema):
        self.calls += 1
        fields = schema['properties']['turns']['properties']
        turns = {key:value['enum'][0] for key,value in fields.items()}
        first = next(iter(turns))
        if self.damage == 'missing':
            turns.pop(first)
        elif self.damage == 'extra':
            turns['0'] = 'The source background is replaced.'
        elif self.damage == 'meaning':
            turns[first] = 'All sharing is allowed.'
        elif self.damage == 'type':
            turns[first] = [turns[first]]
        return json.dumps(dict(turns=turns))


def batch_sample(seed):
    # REVOKE then REINSTATE has multiple policy updates plus an initial rule.
    sample = next(v for v in variants(seed) if v['variant_id'] == 'CAP05')
    return dict(sample, paraphrase_scope='all_policy_turns')


def test_batch_changes_every_eligible_policy_turn_in_one_call(seed213):
    sample = batch_sample(seed213)
    plan = blueprint(with_schedule(sample))
    client = BatchSelector()
    row = execute(sample, client, ROOT/'prompts')
    assert client.calls == 1
    assert row['generation_method'] == 'qwen_paraphrase'
    coverage = row['paraphrase_coverage']
    assert len(coverage['changed_turn_ids']) >= 3
    assert coverage['changed_turn_ids'] == coverage['eligible_turn_ids']
    for index, turn in enumerate(row['turns']):
        if index in coverage['eligible_turn_ids']:
            assert turn['text'] != plan['dialogue']['turns'][index]['text']
        else:
            assert turn == plan['dialogue']['turns'][index]
    baseline = execute(sample, None, ROOT/'prompts')
    for key in ('gold_decision','gold_state','operation_trace','initial_rules','policy_turn_plan','final_request'):
        assert row[key] == baseline[key]
    validate_record(row)
    validate_pipeline_record(row, stage='gold')


@pytest.mark.parametrize('damage',['missing','extra','meaning','type'])
def test_batch_rejects_entire_invalid_output_and_reports_template_fallback(seed213, damage):
    sample = batch_sample(seed213)
    row = execute(sample, BatchSelector(damage), ROOT/'prompts')
    baseline = execute(sample, None, ROOT/'prompts')
    assert row['fallback_used'] and row['generation_method'] == 'template'
    assert row['qwen_validation_status'] == 'UNVERIFIED'
    assert row['turns'] == baseline['turns']
    assert row['paraphrase_coverage']['changed_turn_ids'] == []
    validate_record(row)


@pytest.mark.parametrize('damage',['coverage','mapping','raw','scope','sentence'])
def test_batch_replay_rejects_tampering(seed213, damage):
    row = execute(batch_sample(seed213), BatchSelector(), ROOT/'prompts')
    if damage == 'coverage': row['paraphrase_coverage']['changed_turn_ids'] = []
    if damage == 'mapping': row['qwen_attempts'][0]['turn_ids'] = [0]
    if damage == 'raw': row['qwen_attempts'][0]['raw'] = '{"turns":{}}'
    if damage == 'scope': row['paraphrase_scope'] = 'single_policy_turn'
    if damage == 'sentence': row['dialogue']['turns'][1]['text'] = 'Everyone may share anything.'
    with pytest.raises(ValueError):
        validate_record(row)


def test_batch_template_only_makes_no_model_claim(seed213):
    row = execute(batch_sample(seed213), None, ROOT/'prompts')
    assert not row['qwen_attempts'] and not row['fallback_used']
    assert row['paraphrase_coverage']['changed_turn_ids'] == []
    validate_record(row)


def test_cpu_batch_reports_explicit_unsupported_mode_before_loading_model():
    from src.data.generation import QwenClient
    client = object.__new__(QwenClient)
    client.endpoint = 'transformers://cpu'
    with pytest.raises(ValueError, match='single_policy_turn'):
        client.complete_json('test', {'properties':{'turns':{}}})


def test_cli_batch_budget_and_resume_preserve_all_dialogues(seed213,tmp_path,monkeypatch):
    from test_progress_resume import load
    import src.data.source_policy as source_policy
    import src.data.generation as generation
    import sys
    module = load('multiturn_prepare','scripts/02_prepare.py')
    monkeypatch.setattr(source_policy,'enforce_source_policy',lambda *args:None)
    client = BatchSelector()
    monkeypatch.setattr(generation,'QwenClient',lambda *args,**kwargs:client)
    # The CLI imported QwenClient when its module was loaded.
    monkeypatch.setattr(module,'QwenClient',lambda *args,**kwargs:client)
    source = tmp_path/'normalized.jsonl'
    source.write_text(json.dumps(seed213)+'\n',encoding='utf-8')
    output,audit = tmp_path/'rows.jsonl',tmp_path/'audit.csv'
    argv = ['02_prepare.py','--input',str(source),'--endpoint','fixture://offline',
        '--model',client.model,'--limit-seeds','1','--paraphrase-budget','2',
        '--output',str(output),'--audit',str(audit)]
    monkeypatch.setattr(sys,'argv',argv)
    module.main()
    expected = output.read_text(encoding='utf-8')
    rows = [json.loads(line) for line in expected.splitlines()]
    assert len(rows) == 25 and client.calls == 2
    assert sum(r['generation_method']=='qwen_paraphrase' for r in rows) == 2
    assert all(r['paraphrase_scope']=='all_policy_turns' for r in rows)
    for row in rows: validate_record(row)
    monkeypatch.setattr(sys,'argv',argv+['--resume'])
    module.main()
    assert client.calls == 2 and output.read_text(encoding='utf-8') == expected
