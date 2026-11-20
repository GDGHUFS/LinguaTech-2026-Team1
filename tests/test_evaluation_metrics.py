"""torch 의존성 없이 C의 AP와 Brier 회귀 사례를 검사한다."""
import math
import pytest
from src.evaluation.metrics import (average_precision,binary_metrics,classification_metrics,
                                    precision_recall,select_threshold)
from src.evaluation.statistics import holm_correction,mean_sd,paired_bootstrap


def test_c_validation_metrics_reused():
    # hge-1113-b-tests의 tests/test_train_utils.py::test_validation_metrics에서 가져온 사례다.
    perfect = binary_metrics([1,0,1,0],[.9,.1,.8,.2],threshold=.5)
    assert perfect['auprc'] == 1.
    assert perfect['brier'] == pytest.approx(.025)
    assert average_precision([1,0],[.5,.5]) == .5
    assert average_precision([1,1],[.9,.8]) is None


def test_ranking_ties_and_ece_endpoints():
    result = binary_metrics([0,1,0,1],[.5,.5,.5,.5],threshold=.5)
    assert result['auprc'] == result['auroc'] == .5
    assert result['brier'] == .25 and result['ece'] == 0
    assert binary_metrics([0,1],[0.,1.],threshold=.5)['ece'] == 0


def test_auc_with_a_partial_tie_is_pairwise_concordance():
    assert binary_metrics([0,1,0,1],[.1,.3,.3,.4])['auroc'] == pytest.approx(.875)


@pytest.mark.parametrize('score',[None,float('nan'),float('inf'),-1,1.1,True])
def test_missing_or_invalid_scores_are_errors(score):
    with pytest.raises(ValueError):
        binary_metrics([0,1],[.1,score])


@pytest.mark.parametrize('label',[None,-100,2,True])
def test_metrics_require_explicitly_filtered_valid_labels(label):
    with pytest.raises(ValueError):
        binary_metrics([0,label],[.1,.8])


def test_empty_and_single_class_are_explicit():
    assert binary_metrics([],[])['status'] == 'NA'
    single = binary_metrics([0,0],[.1,.2],threshold=.5)
    assert single['auprc'] is single['auroc'] is single['recall'] is single['precision'] is None
    assert single['brier'] == pytest.approx(.025)
    assert single['status'] == 'PARTIAL'


def test_threshold_selection_ties_prefer_higher_threshold():
    assert select_threshold([0,1,0,1],[.1,.9,.2,.8])['threshold'] == .8
    assert select_threshold([1,1],[.4,.5])['status'] == 'NA'


def test_masked_classification_fixed_class_axis():
    result = classification_metrics(['A',None,'B',-100],['A',None,'A',None],('A','B','C'))
    assert result['n'] == 2 and result['coverage'] == .5
    assert result['accuracy'] == .5 and result['macro_f1'] == pytest.approx(2/9)
    assert result['balanced_accuracy'] == .5
    with pytest.raises(ValueError):
        classification_metrics(['A'],[None],('A','B'))


def test_three_seed_sample_sd_does_not_hide_missing_seeds():
    assert mean_sd([1.,2.,3.])['sd'] == 1.
    assert mean_sd([1.,None,3.])['mean'] is None


def test_paired_bootstrap_uses_whole_groups_and_same_seeds():
    labels = [0,1]*3
    groups = ['g1','g1','g2','g2','g3','g3']
    good = {s:[.1,.9]*3 for s in (42,43,44)}
    bad = {s:[.9,.1]*3 for s in (42,43,44)}
    result = paired_bootstrap(labels,groups,good,bad)
    assert result['estimate'] == .5
    assert result['ci95'] == [.5,.5]
    assert result['valid_replicates'] == 2000
    assert result['n_groups'] == 3
    assert result['p_value'] == pytest.approx(1/2001)
    assert result == paired_bootstrap(labels,groups,good,bad)


def test_bootstrap_degenerate_draws_are_counted():
    result = paired_bootstrap([0,1],['a','b'],{42:[.1,.9]},{42:[.2,.8]},repetitions=100)
    assert result['status'] == 'NA'
    assert result['degenerate_replicates'] > 0 and result['ci95'] is None
    with pytest.raises(ValueError):
        paired_bootstrap([0,1],['a','b'],{42:[.1,.9]},{43:[.1,.9]})


def test_holm_keeps_planned_missing_hypotheses():
    result = holm_correction({'a':.01,'b':.03,'c':None})
    assert result == {'a':.03,'b':.06,'c':None}
