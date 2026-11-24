from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F
from src.features import temporal_features, previous_record
from src.probes import P1Probe, T1Probe, ConShiftProbe
from src.representation import extract, last_nonpadding


def test_temporal_math_and_detachment():
    h = torch.tensor([[3., 4.]], requires_grad=True)
    old = torch.tensor([[0., 5.]], requires_grad=True)
    u, delta, distance = temporal_features(h, old)
    assert torch.allclose(u, torch.tensor([[.6, .8]]))
    assert torch.allclose(delta, torch.tensor([[.6, -.2]]))
    assert distance.item() == pytest.approx(.2)
    assert not u.requires_grad and not delta.requires_grad
    u, delta, d = temporal_features(h, h)
    assert delta.count_nonzero() == 0
    assert d.item() == pytest.approx(0, abs=1e-6)
    _, _, d = temporal_features(torch.tensor([[1., 0.]]), torch.tensor([[-1., 0.]]))
    assert d.item() == 2.


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), 0.])
def test_invalid_vectors(bad):
    with pytest.raises(ValueError):
        temporal_features(torch.tensor([[bad, 0.]]), torch.ones(1, 2))


def test_previous_requires_immediate_same_trajectory():
    current = {'trajectory_id': 'a', 'turn_index': 3}
    assert previous_record(current, {'trajectory_id':'a', 'turn_index':2})
    assert not previous_record(current, {'trajectory_id':'b', 'turn_index':2})
    assert not previous_record(current, {'trajectory_id':'a', 'turn_index':1})
    assert not previous_record(current, None)


@pytest.mark.parametrize('cls,count', [(P1Probe, 5), (T1Probe, 9), (ConShiftProbe, 11)])
def test_shapes_and_parameter_counts(cls, count):
    model = cls(4)
    assert sum(p.numel() for p in model.parameters()) == count
    x = torch.randn(3, 4)
    assert model(x, x, torch.ones(3, 1)).shape == (3,)


def test_gate_equivalence_gradients_and_loss():
    torch.manual_seed(42)
    con = ConShiftProbe(4)
    t1 = T1Probe(4)
    t1.head.load_state_dict(con.head.state_dict())
    u, delta, d = temporal_features(torch.randn(3, 4), torch.randn(3, 4))
    assert con.gate_a.item() == con.gate_c.item() == 0
    assert torch.equal(con.gate_value(d), torch.full((3, 1), .5))
    assert torch.allclose(con(u, delta, d, force_gate=1), t1(u, delta, d))
    assert torch.allclose(con(u, delta, d, force_gate=0), con.head(torch.cat([u, torch.zeros_like(delta)], -1)).squeeze(-1))
    loss = F.binary_cross_entropy_with_logits(con(u, delta, d), torch.tensor([0., 1., 0.]))
    loss.backward()
    assert con.gate_a.grad is not None and con.gate_c.grad is not None
    assert con.gate_a.grad.isfinite().all()
    assert ((con.gate_value(d) >= 0) & (con.gate_value(d) <= 1)).all()


def test_padding_selection_and_pregeneration_extraction():
    assert last_nonpadding(torch.tensor([[0, 1, 1, 0], [1, 1, 0, 0]])).tolist() == [2, 1]
    with pytest.raises(ValueError):
        last_nonpadding(torch.zeros(1, 2))
    class TensorOnlyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=2)
            self.parameter = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        def forward(self, **kwargs):
            assert kwargs['use_cache'] is False and kwargs['output_hidden_states']
            assert not torch.is_grad_enabled()
            return SimpleNamespace(hidden_states=(torch.tensor([[[1., 2.], [3., 4.], [5., 6.]]]),))
    model = TensorOnlyModel()
    h, idx = extract(model, {'input_ids':torch.tensor([[1, 2, 0]]), 'attention_mask':torch.tensor([[1, 1, 0]])})
    assert idx == 1 and h.tolist() == [3., 4.]
    assert h.dtype == torch.float32 and h.device.type == 'cpu'
    assert model.parameter.grad is None and not h.requires_grad
    model.config.hidden_size = 3
    with pytest.raises(RuntimeError):
        extract(model, {'input_ids':torch.tensor([[1, 2, 0]]), 'attention_mask':torch.tensor([[1, 1, 0]])})
