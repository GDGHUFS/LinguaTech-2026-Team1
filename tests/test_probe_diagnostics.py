import numpy as np
import pytest
from src.probe_diagnostics import diagnose


def fixture():
    u = np.array([[1.,0.],[0.,1.],[1.,0.],[0.,1.]])
    delta = np.array([[.2,0.],[0.,.3],[-.2,0.],[0.,-.3]])
    d = np.array([[.1],[.2],[.4],[.8]])
    rows = [dict(sample_id=str(i),group_id=str(i//2),trajectory_id='t',turn_index=i+2)
            for i in range(4)]
    return (u,delta,d), np.array([0,1,0,1]), rows


def test_gate_contributions_and_same_head_ablations():
    f,y,rows = fixture()
    weight = np.array([1.,2.,3.,4.])
    report, predictions = diagnose(f,y,weight,.1,{'a':0.,'c':0.},.25,
        rows=rows,cutpoints={'distance':[.2,.4],'turn':[2.,3.]})
    assert report['gate']['mean'] == .5 and report['gate']['std'] == 0
    assert report['groups'] == 2
    for i,r in enumerate(predictions):
        assert r['gated_delta_logit'] == pytest.approx(.5 * (f[1][i] @ weight[2:]))
        logit = f[0][i] @ weight[:2] + .5 * (f[1][i] @ weight[2:]) + .1
        assert r['probability'] == pytest.approx(1/(1+np.exp(-logit)))
        assert r['sample_id'] == str(i)
    assert report['ablations']['gate_train_mean']['gate'] == .25
    assert report['metrics']['Brier'] == report['ablations']['gate_05']['Brier']
    assert report['ablations']['gate_0']['Brier'] != report['metrics']['Brier']


def test_p1_has_no_delta_and_single_class_subgroups_are_explicit():
    f,y,rows = fixture()
    report, predictions = diagnose(f,y,np.array([1.,2.]),0.,rows=rows,
        cutpoints={'distance':[.2,.4]})
    assert report['ablations'] == {}
    assert all(r['gated_delta_logit'] == 0 for r in predictions)
    assert report['subgroups']['distance']['bins']['middle']['undefined_reason'] == 'SINGLE_CLASS'


def test_diagnostics_do_not_modify_inputs_and_identical_cuts_are_marked():
    f,y,rows = fixture()
    before = [a.copy() for a in f]
    report,_ = diagnose(f,y,np.ones(4),0.,{'a':2.,'c':-.5},rows=rows,
                       cutpoints={'distance':[.2,.2]})
    assert all(np.array_equal(a,b) for a,b in zip(f,before))
    assert report['subgroups']['distance']['status'] == 'IDENTICAL_TRAIN_CUTPOINTS'
    assert report['gate']['std'] > 0
