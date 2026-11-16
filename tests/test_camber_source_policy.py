"""합성 원본으로 소스 검수 조건과 누수 방지를 검사한다. 자료는 테스트 안에만 둔다."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from src.data.canonicalize import active_setup, camber_adapter, select_group_candidates, load_source
from src.data.mutations import make_variants
from src.data.source_policy import apply_review, generation_blockers, selected_source, enforce_source_policy
from src.data.oracle_adapter import evaluate_structured
from src.data.splitting import split_samples

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def seed():
    scenario = dict(subject_agent='owner', aware_agent='owner', oblivious_agent='organizer',
        detail='email', reveal_reason='event', story='Fictional test: owner shares email for event. No consent assertion.')
    result = camber_adapter(dict(example_id='fixture', example_name='fixture', scenario=repr(scenario),
        appropriateness_label='inappropriate', expansion_type='no_expansion', expansion_dimension='none'),
        'confaideplus', 'fixture.csv', 'fixture-hash', dict(repository='fixture', revision='fixture'))
    # Test-only explicitly scoped fixture. Real adapter output leaves purpose null;
    # production needs a separately approved annotation, tested in test_camber_revision.
    result['purpose'] = 'event'
    result['source_evidence']['purpose'] = dict(path='scenario.story', quote=scenario['story'], status='pending_review')
    return result


def review_for(seed):
    return dict(seed_id=seed['seed_id'], record_hash=seed['source_provenance']['record_hash'],
        reviewer='test-fixture-only', reviewed_at='fixture-time', rationale='Fictional review for unit test only',
        scope={k: dict(path=v['path'], quote=v['quote'], value=seed[k], confirmed=True)
               for k, v in seed['source_evidence'].items() if k in ('subject','sender','recipient','data_type','purpose')},
        consent_assessment=dict(path='scenario.story', quote=seed['policy_context'], value='absent'),
        disclosure_assessment=dict(path='scenario.story', quote=seed['policy_context'], value='explicit'),
        synthetic_policy_compatible=True)


def test_default_selection_preserves_optional_sources():
    config = yaml.safe_load((ROOT/'configs/data.yaml').read_text(encoding='utf-8'))
    setup = active_setup(config)
    assert set(setup['sources']) == {'CAMBER'}
    assert 'CoPriva' in config['seed_setup']['sources']
    assert setup['sources']['CAMBER']['components'] == ['confaideplus']
    assert 'camber/privacylensplus.csv' not in setup['sources']['CAMBER']['files']
    assert not selected_source(dict(source_dataset='CoPriva'), config)
    assert not selected_source(dict(source_dataset='CAMBER', source_record_id='privacylensplus:1'), config)
    config['sources'].append('CoPriva')
    config['source_selection']['CoPriva'] = []
    assert 'CoPriva' in active_setup(config)['sources']


def test_raw_labels_and_scope_presence_do_not_approve_generation(seed):
    assert seed['status'] == 'eligible_pending_review'
    assert seed['consent'] is None and seed['gold_decision'] is None
    assert seed['purpose_mapping_status'] == 'pending_review'
    assert make_variants(seed) == []
    assert generation_blockers(seed)
    with pytest.raises(ValueError, match='review'):
        enforce_source_policy([seed])
    altered = deepcopy(seed)
    altered['source_appropriateness_label'] = 'appropriate'
    assert make_variants(altered) == [] and altered['gold_decision'] is None


@pytest.mark.parametrize('change', ['hash','quote','scope','unclear','compatibility'])
def test_review_cannot_invent_evidence_or_silently_approve(seed, change):
    review = review_for(seed)
    if change == 'hash': review['record_hash'] = 'stale'
    if change == 'quote': review['scope']['purpose']['quote'] = 'invented permission'
    if change == 'scope': review['scope']['sender']['value'] = 'invented sender'
    if change == 'unclear': review['consent_assessment']['value'] = 'unclear'
    if change == 'compatibility': review['synthetic_policy_compatible'] = False
    if change in ('unclear','compatibility'):
        result = apply_review(seed, review)
        assert not result['generation_ready'] and make_variants(result) == []
    else:
        with pytest.raises(ValueError):
            apply_review(seed, review)


def test_reviewed_fixture_uses_existing_oracle_and_preserves_provenance(seed):
    approved = apply_review(seed, review_for(seed))
    assert generation_blockers(approved) == []
    variants = make_variants(approved)
    assert len(variants) == 10
    for row in variants:
        assert row['source_facts']['consent'] is None
        assert row['synthetic_intervention'] is True
        assert row['source_provenance'] == seed['source_provenance']
        assert row['gold_decision'] is None
        assert evaluate_structured(row)['annotation_status'] == 'VALID'
    base = evaluate_structured(variants[0])
    cap06 = evaluate_structured(next(v for v in variants if v['capability_id'] == 'CAP06'))
    assert (base['gold_decision'],base['gold_state']) == (cap06['gold_decision'],cap06['gold_state'])


def test_group_sampling_and_all_expansion_split_isolation():
    rows = [dict(group_id=f'CAMBER:g{g}', source_record_id=f'{g}:{i}',
        source_record=dict(expansion_type='no_expansion' if i == 0 else 'reasoning_guided'))
        for g in range(100) for i in range(3)]
    sampled = select_group_candidates(rows, 100, 42)
    assert len({r['group_id'] for r in sampled}) == 100
    assert sampled == select_group_candidates(list(reversed(rows)), 100, 42)
    split_rows = [dict(r, source_dataset='CAMBER', source_seed_id=r['group_id'],
        variant_group_id=r['group_id']+':'+r['source_record_id'], sample_id=r['source_record_id'],
        template_family_id='id', operation_trace=[]) for r in rows]
    result, _ = split_samples(split_rows, compositions=(), templates=())
    assert all(len({r['split'] for r in result if r['group_id'] == group}) == 1 for group in {r['group_id'] for r in result})


def test_pending_review_cli_never_writes_generation(tmp_path):
    path = ROOT/'data/annotated/canonical_confaideplus_v2.jsonl'
    if not path.exists():
        pytest.skip('Local source pilot not installed')
    output, audit = tmp_path/'generated.jsonl', tmp_path/'audit.csv'
    run = subprocess.run([sys.executable, 'scripts/02_prepare.py', '--input', str(path),
        '--output', str(output), '--audit', str(audit)], cwd=ROOT, capture_output=True, text=True)
    assert run.returncode != 0 and 'generation-ready' in run.stderr
    assert not output.exists() and not audit.exists()


def test_local_inventory_group_pilot():
    config = yaml.safe_load((ROOT/'configs/data.yaml').read_text(encoding='utf-8'))
    if not (ROOT/'data/raw/CAMBER/camber/confaideplus.csv').exists():
        pytest.skip('Pinned source unavailable')
    rows = load_source(ROOT/'data/raw', 'CAMBER', active_setup(config)['sources']['CAMBER'])
    assert len(rows) == 12960
    assert len({r['group_id'] for r in rows}) == 270
    assert all(not r['generation_ready'] for r in rows)
    pilot = select_group_candidates(rows, 100, 42)
    assert len({r['group_id'] for r in pilot}) == 100
