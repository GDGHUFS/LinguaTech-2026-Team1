"""합성 점수와 표현으로 선택 분석의 순위·group 순열·학습 분할 제한을 검사한다."""
import pytest
from src.evaluation.failure_guided import failure_guided
from src.evaluation.ood_analysis import grouped_mmd
from src.evaluation.protocol import freeze_protocol
from src.evaluation.representation_probe import run_probes,validate_embeddings
from test_evaluation_protocol import fixture


def test_optional_ranking_matches_equal_budget_and_is_deterministic():
    rows=[dict(sample_id=str(i),failure_probability=i/10,failure_label_unsafe=int(i>=5)) for i in range(10)]
    result=failure_guided(rows,3)
    assert result['found']==3 and result['recall_at_b']==.6
    assert result==failure_guided(rows,3)
    with pytest.raises(ValueError):
        failure_guided(rows,3,access_timing='post_run')
    with pytest.raises(ValueError):
        failure_guided(rows,11)


def test_optional_grouped_mmd_identical_and_shifted_features():
    identical=grouped_mmd([[0],[0]],[[0],[0]],['a','b'],['c','d'],sigma=1,permutations=30)
    assert identical['mmd2_unbiased']==0 and identical['p_value']==1
    shifted=grouped_mmd([[0],[0],[0]],[[10],[10],[10]],['a','b','c'],['d','e','f'],sigma=1,permutations=30)
    assert shifted['mmd2_unbiased']>1.9
    with pytest.raises(ValueError):
        grouped_mmd([[0],[0]],[[1],[1]],['a','b'],['a','c'],sigma=1)


def embeddings_fixture():
    rows,predictions,manifest,plan=fixture()
    protocol=freeze_protocol(predictions,rows,manifest,plan,test_only=True)
    embeddings=[]
    for model in ('A','C','C+'):
        for seed in (42,43,44):
            for r in rows:
                embeddings.append(dict(sample_id=r['sample_id'],split=r['split'],group_id=r['group_id'],
                    model=model,seed=seed,checkpoint_hash=f'{model}:{seed}',cls=[float(r['failure_label_unsafe']),1.],
                    encoder_frozen=True,representation='last_layer_cls'))
    return embeddings,rows,manifest,protocol


def test_probe_contract_uses_common_ids_and_no_test_layer_selection():
    embeddings,rows,manifest,protocol=embeddings_fixture()
    refs,indexed,active=validate_embeddings(embeddings,rows,manifest,protocol,test_only=True)
    assert len(refs)==20 and set(active)=={'A','C','C+'}
    embeddings.pop()
    with pytest.raises(ValueError,match='common probe'):
        validate_embeddings(embeddings,rows,manifest,protocol,test_only=True)


def test_probe_not_executed_is_skipped():
    _,rows,manifest,protocol=embeddings_fixture()
    assert run_probes([],rows,manifest,protocol,test_only=True)['status']=='SKIPPED'


def test_actual_sklearn_probe_train_only_scaler_and_val_selection():
    pytest.importorskip('sklearn')
    embeddings,rows,manifest,protocol=embeddings_fixture()
    # 두 state class를 쓰며, 테스트 표현의 큰 분포 이동이 scaler 학습에 반영되지 않는지 확인한다.
    for r in rows:
        r['gold_state']['authorization']='ALLOW' if r['failure_label_unsafe'] else 'DENY'
    for e in embeddings:
        if e['split'].startswith('ood'):
            e['cls']=[100.,1.]
    _,predictions,_,plan=fixture()
    protocol=freeze_protocol(predictions,rows,manifest,plan,test_only=True)
    result=run_probes(embeddings,rows,manifest,protocol,test_only=True)
    fit=result['models']['A']['seeds']['42']['authorization']
    assert fit['scaler_mean']==[.5,1.]
    assert fit['selected_C'] in [.01,.1,1.,10.]
    assert fit['test']['id_test']['accuracy']==1.
