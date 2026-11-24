from pathlib import Path
import copy
import pytest
from src.contracts import ROOT, Blocked, digest, read_json, read_jsonl, write_json
from src.evolif_adapter import normalize, DETERMINISTIC
from src.splitting import make_split, audit_split, pilot_trajectories
from src.checker import OfficialChecker, args_valid
from src.labeling import failure_label, label_one, coverage

RAW = ROOT / 'data/raw'


@pytest.fixture(scope='module')
def originals():
    return normalize(RAW / 'official_data/dialog_v0.1', '47115ae2af4830948f3f15697221a1acca6078a7')


@pytest.fixture(scope='module')
def checker():
    return OfficialChecker(RAW / 'official_code')


def test_original_census_and_no_source_labels(originals):
    assert len(originals) == 7620
    assert len({r['trajectory_id'] for r in originals}) == 150
    assert len({r['group_id'] for r in originals}) == 133
    for r in originals:
        assert 'instruction_success' not in r and 'topic_success' not in r
        assert r['composition_signature'] == sorted({i['instruction_id'] for i in r['active_constraints']})
    sample = originals[0]
    source = read_jsonl(RAW / 'official_data/dialog_v0.1' / sample['source_file'])[sample['source_record_index']]
    assert sample['user_query'] == source['user_query_verified']


def test_response_blind_split_and_leakage(originals):
    manifest = make_split(originals, 'source')
    assert manifest == make_split(originals, 'source')
    audit_split(originals, manifest)
    assert sum(s['groups'] for s in manifest['counts'].values()) == 133
    assert len(pilot_trajectories(originals, manifest)) == 20
    if manifest['ood_signature']:
        for r in originals:
            if r['composition_signature'] == manifest['ood_signature']:
                assert manifest['membership'][r['group_id']] == 'ood'
    changed = copy.deepcopy(manifest)
    changed['membership'][originals[0]['group_id']] = 'alien'
    with pytest.raises(Blocked):
        audit_split(originals, changed)


def test_immutable_manifest(tmp_path):
    p = tmp_path / 'manifest.json'
    write_json(p, {'a':1}, immutable=True)
    write_json(p, {'a':1}, immutable=True)
    with pytest.raises(Blocked):
        write_json(p, {'a':2}, immutable=True)


@pytest.mark.parametrize('results,label', [([],None), (['PASS'],0), (['PASS','FAIL'],1),
    (['FAIL','UNKNOWN'],None), (['PASS','UNKNOWN'],None), (['PASS','PASS'],0)])
def test_failure_aggregation(results, label):
    assert failure_label([{'result':r} for r in results]) == label


def test_all_official_args_are_supported(originals):
    for row in originals:
        for c in row['active_constraints']:
            if c['instruction_id'] in DETERMINISTIC:
                assert args_valid(c['instruction_id'], c['args'])


@pytest.mark.parametrize('i,args,passing,failing', [
    ('case', {'mode':'all_upper'}, 'HELLO', 'Hello'),
    ('countableItems', {'num':2}, '- a\n- b', '- a'),
    ('format', {'mode':'json'}, '{"a":1}', 'not json'),
    ('existence', {'apple':1}, 'apple', 'pear'),
    ('forbidden', ['secret'], 'public', 'secret'),
    ('length', {'mode':'word','relation':'exactly','number':2}, 'one two', 'one'),
    ('punctuation', {'mode':'must_include','value':'!'}, 'yes!', 'yes'),
    ('startwith', {'mode':'keyword','value':'hello'}, 'hello there', 'bye there'),
    ('endwith', {'mode':'keyword','value':'end'}, 'the end', 'the beginning'),
])
def test_actual_official_checker_bool_mapping(checker,i,args,passing,failing):
    constraint = {'instruction_id':i,'args':args}
    assert checker.evaluate(constraint,passing)['result'] == 'PASS'
    assert checker.evaluate(constraint,failing)['result'] == 'FAIL'


def test_unknown_never_passes(checker, monkeypatch):
    assert checker.evaluate({'instruction_id':'style','args':{}},'')['result'] == 'UNKNOWN'
    assert checker.evaluate({'instruction_id':'case','args':{}},'')['result'] == 'UNKNOWN'
    monkeypatch.setattr(checker.helper, 'count_sentences', lambda _: (_ for _ in ()).throw(LookupError('missing punkt')))
    r = checker.evaluate({'instruction_id':'length','args':{'mode':'sentence','relation':'exactly','number':1}}, 'Text.')
    assert r['result'] == 'UNKNOWN'


def test_no_previous_no_primary_cohort_and_truncation(originals, checker):
    row = next(r for r in originals if r['composition_signature'] and set(r['composition_signature']) <= DETERMINISTIC)
    # No invented experimental response: empty text is just a checker contract input.
    response = {'status':'OK','target_response':'','truncated':False,'representation_key':'tensor'}
    result = label_one(row,response,None,checker)
    assert not result['eligible']
    assert result['previous_representation_key'] is None
    result = label_one(row,{**response,'truncated':True},None,checker)
    assert result['failure_label'] is None and result['exclusion_reason'] == 'OUTPUT_TRUNCATED'
    other = {'trajectory_id':'different','turn_index':row['turn_index']-1,'representation_key':'old'}
    assert not label_one(row,response,other,checker)['eligible']


def test_coverage_gate_empty():
    result = coverage([])
    assert result['status'] == 'BLOCKED'
    assert all(result[s]['status'] == 'BLOCKED' for s in ('train','validation','id_test','ood'))
