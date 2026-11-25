"""Patience tolerates small validation fluctuations without losing best fits."""
import pytest
from src.training_selection import patience_improved, select_l2_candidate
from src.contracts import ROOT, Blocked, config


def test_patience_accumulates_small_ap_gains_from_last_reset():
    reference = (.5, -.2)
    assert not patience_improved((.5004, -.19), reference, .001)
    assert not patience_improved((.5008, -.18), reference, .001)
    assert patience_improved((.5012, -.17), reference, .001)


def test_brier_tiebreak_does_not_extend_positive_delta_training():
    assert not patience_improved((.5, -.1), (.5, -.2), .001)
    assert patience_improved((.5, -.1), (.5, -.2), 0.)
    assert not patience_improved((.49, -.1), (.5, -.2), 0.)
    assert patience_improved((.1, -.4), None, .001)


@pytest.mark.parametrize('delta', [-.1, float('nan'), float('inf')])
def test_invalid_delta_rejected(delta):
    with pytest.raises(ValueError):
        patience_improved((.5, -.2), None, delta)


def test_context56k_protocol_and_isolated_paths(tmp_path):
    import yaml
    cfg = config(ROOT / 'configs/experiment.yaml')
    assert cfg['target']['max_context_tokens'] == 57344
    assert cfg['target']['gpu_memory_fraction'] == .40
    assert cfg['training']['early_stopping_min_delta'] == .001
    assert cfg['training']['max_epochs'] == 200
    assert cfg['training']['patience'] == 10
    assert cfg['training']['l2_candidates'] == [.0001,.001,.01,.1]
    assert cfg['paths']['data'] == str(ROOT / 'data/runs/context56k-vllm-2gpu')
    assert cfg['paths']['output'] == str(ROOT / 'outputs/runs/context56k-vllm-2gpu')
    cfg['target']['max_context_tokens'] = 8192
    path = tmp_path / 'old.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    with pytest.raises(Blocked, match='registered protocol'):
        config(path)


def test_l2_selection_uses_validation_then_brier_epoch_and_grid_order():
    candidates = [dict(validation_ap=.7,validation_brier=.2,epoch=3),
                  dict(validation_ap=.8,validation_brier=.3,epoch=5),
                  dict(validation_ap=.8,validation_brier=.2,epoch=4),
                  dict(validation_ap=.8,validation_brier=.2,epoch=2),
                  dict(validation_ap=.8,validation_brier=.2,epoch=2)]
    assert select_l2_candidate(candidates) == 3


@pytest.mark.parametrize('fraction', [0, -.1, 1.01, True, '0.98', None, float('nan')])
def test_invalid_gpu_memory_fraction_rejected(tmp_path, fraction):
    import yaml
    cfg = config(ROOT / 'configs/experiment.yaml')
    cfg['target']['gpu_memory_fraction'] = fraction
    path = tmp_path / 'invalid.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    with pytest.raises(Blocked, match='gpu_memory_fraction'):
        config(path)
