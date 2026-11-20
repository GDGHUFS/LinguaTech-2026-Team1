"""오프라인 반례로 실행 승인 조건을 검사한다. 실제 승인이나 연구 산출물은 만들지 않는다."""
from copy import deepcopy
import pytest
from test_camber_source_policy import seed, review_for
from src.data.source_policy import apply_review, validate_source_approval
from src.data.approvals import validate_qa, validate_freeze, qa_payload, research_rows
from src.utils.reproducibility import content_hash, file_hash


def test_exact_source_review_revalidated(seed):
    approved=apply_review(seed,review_for(seed))
    assert validate_source_approval(approved,{seed['seed_id']:seed})['generation_ready']


@pytest.mark.parametrize('damage',['hash','group','seed','quote','unclear','reviewer','facts'])
def test_source_review_replay_and_corruption_rejected(seed,damage):
    row=apply_review(seed,review_for(seed))
    if damage=='hash': row['source_provenance']['record_hash']='invalid'
    if damage=='group': row['group_id']='another-expansion-group'
    if damage=='seed': row['seed_id']='sibling'
    if damage=='quote': row['semantic_review']['scope']['purpose']['quote']='invented'
    if damage=='unclear': row['semantic_review']['consent_assessment']['value']='unclear'
    if damage=='reviewer': row['semantic_review']['reviewer']=''
    if damage=='facts': row['purpose']='invented purpose'
    with pytest.raises(ValueError): validate_source_approval(row,{seed['seed_id']:seed})


def qa_fixture():
    row=dict(sample_id='test-only',group_id='g',turns=[dict(text='fictional')],
             validation_status='semantic_pass_oracle_verified',controlled_family_status='semantic_complete_oracle_verified')
    human=dict(approved=True,reviewer='unit-test-only',reviewed_at='fixture',rationale='test only')
    entry=dict(human,content_hash=content_hash(qa_payload(row)),semantic_aligned=True,support_verified=True)
    return row,dict(human,samples={row['sample_id']:entry})


def test_generated_qa_accepts_bound_evidence_and_rejects_edits():
    row,qa=qa_fixture(); validate_qa([row],qa)
    row['turns'][0]['text']='changed after review'
    with pytest.raises(ValueError): validate_qa([row],qa)


@pytest.mark.parametrize('key',['approved','semantic_aligned','support_verified'])
def test_pending_generated_qa_rejected(key):
    row,qa=qa_fixture(); qa['samples'][row['sample_id']][key]=False
    with pytest.raises(ValueError): validate_qa([row],qa)


def test_final_freeze_binds_config_plan_and_records(tmp_path):
    row,qa=qa_fixture(); plan={'g':'id'}
    row.update(split='train',holdout_plan_hash=content_hash(plan))
    config=tmp_path/'config.yaml'; config.write_text('test only')
    manifest={k:v for k,v in qa.items() if k!='samples'}
    manifest.update(source_selection={'CAMBER':['confaideplus']},holdout_plan=plan,
                    records={row['sample_id']:content_hash(row)},config_hashes={str(config):file_hash(config)})
    manifest['audit_summary']=dict(independent_support_annotations=50,semantic_alignment=.95,
                                  support_f1=.95,support_exact_match=.9,leave_one_turn_out_checked=True)
    data={'source_selection':manifest['source_selection']}
    validate_freeze([row],manifest,data,[config])
    config.write_text('changed')
    with pytest.raises(ValueError): validate_freeze([row],manifest,data,[config])


def test_production_rejects_fixture_and_test_protocol():
    with pytest.raises(ValueError): research_rows([dict(data_kind='mock')])
    from test_evaluation_protocol import fixture
    from src.evaluation.protocol import freeze_protocol,evaluate_predictions
    rows,predictions,manifest,plan=fixture()
    with pytest.raises(ValueError,match='registry'): freeze_protocol(predictions,rows,manifest,plan)
    protocol=freeze_protocol(predictions,rows,manifest,plan,test_only=True)
    with pytest.raises(ValueError,match='Test-only'): evaluate_predictions(predictions,manifest,protocol,records=rows)


def test_modify_scope_subtypes_keep_oracle(seed):
    from src.data.mutations import make_variants
    from src.data.oracle_adapter import evaluate_structured
    rows=make_variants(apply_review(seed,review_for(seed)),('authorization','recipient','purpose','data_type'))
    assert len(rows)==13
    for row in rows:
        assert evaluate_structured(row)['annotation_status']=='VALID'


def test_source_readiness_can_advance_without_fabricated_model_results(seed):
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('readiness_stage',Path('scripts/06_camber_readiness.py'))
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    approved=apply_review(seed,review_for(seed))
    result=module.audit([approved],[approved],{'source_selection':{'CAMBER':['confaideplus']}})
    assert result['status']=='READY_FOR_GENERATION'
    assert result['target_calls']==0 and result['split_feasibility']['actual_id_ood_feasible'] is False
