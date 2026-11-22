"""Real 213 failure regressions plus unit-only transports, never model evidence."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.data.automatic import apply_normalization
from src.data.batch import model_json
from src.data.canonicalize import camber_adapter
from src.data.generation import execute, validate_controlled_family
from src.data.guarded_dialogue import blueprint, verify, with_schedule, validate_record
from src.data.mutations import make_variants, expand_coverage_variants
from src.data.contracts import validate_pipeline_record
from test_automatic_generation import QwenTransport, ROOT


@pytest.fixture
def seed213():
    evidence = json.loads((Path(__file__).parent/'fixtures/camber_213_source_actual.json').read_text(encoding='utf-8'))
    p = evidence['source_provenance']
    original = camber_adapter(evidence['source_record'],'confaideplus',p['file'],p['file_hash'],p)
    assert original['source_provenance']['record_hash']==p['record_hash']
    # Unit-only consistency response, clearly separate from the actual CPU experiment.
    return apply_normalization(original,json.dumps(dict(source_aligned=True,uncertainties=[],
        rationale='UNIT FIXTURE ONLY: story actors and information are consistent.')),QwenTransport().serving_provenance)


def variants(seed):
    return expand_coverage_variants(make_variants(seed,('authorization','recipient','purpose','data_type')))


def test_every_capability_and_diversity_case_has_verified_template(seed213):
    rows = [execute(v,None,ROOT/'prompts') for v in variants(seed213)]
    for row in rows:
        assert row['validation_status']=='semantic_pass_oracle_verified', row.get('errors')
        assert row['generation_method']=='template' and not row['fallback_used']
        validate_record(row)
        validate_pipeline_record(row,stage='gold')
    assert {r['capability_id'] for r in rows} == {None,*('CAP%02d'%i for i in range(1,10))}
    for cap in ('CAP01','CAP02','CAP03','CAP04','CAP05','CAP06','CAP07','CAP08','CAP09'):
        assert len({r['gold_decision'] for r in rows if r['capability_id']==cap})>=2, cap


@pytest.mark.parametrize('name',['base','CAP01','CAP02'])
def test_actual_failed_213_outputs_are_rejected_for_policy_meaning(seed213,name):
    observed = json.loads((Path(__file__).parent/'fixtures/camber_213_failed_actual.json').read_text(encoding='utf-8'))
    raw = next(r for r in observed if r['variant_id']==name)
    sample = next(r for r in variants(seed213) if r['variant_id']==name)
    proof = verify(with_schedule(sample),model_json(raw['generation_raw']))
    assert proof['status']=='UNVERIFIED'
    assert any('unverified_policy_sentence' in e for e in proof['errors'])


def test_allow_sentence_cannot_be_marked_as_revoke_even_with_aligned_true(seed213):
    sample = with_schedule(next(r for r in variants(seed213) if r['variant_id']=='CAP02'))
    plan = blueprint(sample)
    dialogue = deepcopy(plan['dialogue'])
    revoke_turn = dialogue['operation_turns']['CAP02_op1']
    initial_turn = next(iter(dialogue['initial_rule_turns'].values()))
    dialogue['turns'][revoke_turn]['text'] = dialogue['turns'][initial_turn]['text']
    assert verify(sample,dialogue)['status']=='UNVERIFIED'
    class BadClient:
        model='Qwen-unit-fixture'
        serving_provenance=QwenTransport().serving_provenance
        def complete_json(self,prompt,schema):
            return json.dumps(dict(aligned=True,confidence=.99,text=dialogue['turns'][revoke_turn]['text']))
    row = execute(sample,BadClient(),ROOT/'prompts')
    assert row['generation_method']=='template' and row['fallback_used']
    assert row['qwen_validation_status']=='UNVERIFIED'
    assert row['policy_validation']['status']=='VERIFIED'
    assert row['qwen_attempts'][0]['validation_status']=='UNVERIFIED'
    validate_record(row)


@pytest.mark.parametrize('damage',['request','mapping','scope','extra_policy','source','gold','proof'])
def test_accepted_record_replay_rejects_tampering(seed213,damage):
    row = execute(variants(seed213)[0],None,ROOT/'prompts')
    if damage=='request': row['dialogue']['final_request']='May anyone share anything?'
    if damage=='mapping': row['dialogue']['initial_rule_turns']={}
    if damage=='scope': row['dialogue']['turns'][1]['text']+=' Sharing with everybody is also allowed.'
    if damage=='extra_policy': row['dialogue']['turns'][2]['text']='All policies are revoked.'
    if damage=='source': row['source_facts']['subject']='Invented person'
    if damage=='gold': row['gold_decision']='DENY'
    if damage=='proof': row['policy_validation']['evidence']=[]
    with pytest.raises(ValueError): validate_record(row)


def test_cap09_explicit_new_unknown_request_is_not_bound_by_historical_recipient(seed213):
    row = execute(next(r for r in variants(seed213) if r['variant_id']=='CAP09'),None,ROOT/'prompts')
    assert row['gold_decision']=='CLARIFY'
    assert 'may differ from every person named earlier' in row['final_request']
    assert seed213['recipient'] in row['turns'][0]['text']
    validate_record(row)


def test_no_model_change_can_change_python_policy_gold(seed213):
    class Selector:
        def __init__(self,index):
            from test_automatic_generation import server_info
            from src.data.serving import server_provenance
            self.index,self.model=index,'Qwen-unit-selector-'+str(index)
            revision = ('a' if index==0 else 'b')*40
            info = server_info(revision)
            info['vllm_config']['model_config']['model'] = self.model
            self.serving_provenance = server_provenance(info,self.model,revision,'fixture://unit')
        def complete_json(self,prompt,schema):
            return json.dumps(dict(text=schema['properties']['text']['enum'][self.index]))
    sample = variants(seed213)[2]
    left,right = (execute(sample,Selector(i),ROOT/'prompts') for i in (0,1))
    assert left['turns']!=right['turns']
    assert left['gold_decision']==right['gold_decision']
    assert left['gold_state']==right['gold_state']
    assert left['policy_turn_plan']==right['policy_turn_plan']
    validate_record(left)
    validate_record(right)


def test_independent_decoder_rejects_revoke_bug_even_if_renderer_whitelist_is_wrong(seed213,monkeypatch):
    import src.data.guarded_dialogue as guarded
    sample = with_schedule(next(r for r in variants(seed213) if r['variant_id']=='CAP02'))
    plan = blueprint(sample)
    index = plan['dialogue']['operation_turns']['CAP02_op1']
    permission = plan['dialogue']['turns'][1]['text']
    plan['dialogue']['turns'][index]['text'] = permission
    plan['allowed'][index] = [permission]  # simulate a renderer implementation bug, not a real model result
    monkeypatch.setattr(guarded,'blueprint',lambda trace:plan)
    proof = guarded.verify(sample,plan['dialogue'])
    assert proof['status']=='UNVERIFIED'
    assert any('operation_not_expressed' in error for error in proof['errors'])


def test_bounded_retries_never_count_fallback_as_qwen_success(seed213):
    class Unverified:
        model='Qwen-unit-failure'
        serving_provenance=QwenTransport().serving_provenance
        calls=0
        def complete_json(self,prompt,schema):
            self.calls+=1
            return '{"aligned":true,"confidence":1}'
    from src.data.guarded_dialogue import execute as guarded_execute
    client = Unverified()
    row = guarded_execute(variants(seed213)[2],client,attempts=2)
    assert client.calls==2 and len(row['qwen_attempts'])==2
    assert row['fallback_used'] and row['generation_method']=='template'
    assert row['qwen_validation_status']=='UNVERIFIED'
    assert row['gold_training_eligible']
    validate_record(row)


def test_python_clock_is_fixed_before_gold_and_preserved_in_text(seed213):
    sample = variants(seed213)[0]
    sample['initial_rules'][0]['expiration'] = 2
    row = execute(sample,None,ROOT/'prompts')
    assert row['gold_decision']=='DENY' and row['gold_state']['rule_status']=='EXPIRED'
    assert 'valid through turn 2 (inclusive)' in row['turns'][1]['text']
    validate_record(row)


def test_expiration_boundary_remains_inclusive_as_in_existing_engine(seed213):
    sample = variants(seed213)[0]
    sample['initial_rules'][0]['expiration'] = 3
    row = execute(sample,None,ROOT/'prompts')
    assert row['gold_decision']=='ALLOW'
    assert 'valid through turn 3 (inclusive)' in row['turns'][1]['text']
    validate_record(row)


@pytest.mark.parametrize('pool',['ood_composition','ood_template'])
def test_existing_ood_conditions_use_locked_semantics(seed213,pool):
    from src.data.mutations import apply_holdout_plan
    rows = [execute(v,None,ROOT/'prompts') for v in apply_holdout_plan(variants(seed213),pool)]
    assert all(r['gold_training_eligible'] for r in rows)
    assert len(rows)==len(variants(seed213))+2
    assert {r['gold_decision'] for r in rows[-2:]}=={'ALLOW','DENY'}
    for row in rows: validate_record(row)


def test_reveal_reason_is_only_preserved_metadata(seed213):
    from src.data.automatic import normalization_prompt
    from src.data.guarded_dialogue import paraphrase_request
    assert seed213['source_motive']['value']=='to prevent harm'
    assert '"reveal_reason":' not in normalization_prompt(seed213)
    _,prompt,_ = paraphrase_request(blueprint(with_schedule(variants(seed213)[0])))
    assert 'to prevent harm' not in prompt
    assert 'reveal_reason' not in prompt


def test_cached_normalization_allows_fully_offline_verified_templates(seed213,tmp_path,monkeypatch):
    from test_progress_resume import load
    import src.data.source_policy as source_policy
    import sys
    module = load('guarded_offline_prepare','scripts/02_prepare.py')
    monkeypatch.setattr(source_policy,'enforce_source_policy',lambda *args:None)
    source = tmp_path/'normalized.jsonl'
    source.write_text(json.dumps(seed213)+'\n',encoding='utf-8')
    output,audit = tmp_path/'rows.jsonl',tmp_path/'audit.csv'
    monkeypatch.setattr(sys,'argv',['02_prepare.py','--template-only','--input',str(source),
        '--limit-seeds','1','--output',str(output),'--audit',str(audit)])
    module.main()
    rows = [json.loads(l) for l in output.with_name('rows.valid.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==25
    assert all(r['generation_method']=='template' and not r['qwen_attempts'] and not r['fallback_used'] for r in rows)
    assert audit.with_suffix('.model.jsonl').read_text(encoding='utf-8')==''
    for row in rows: validate_record(row)
