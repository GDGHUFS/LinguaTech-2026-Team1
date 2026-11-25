"""Patience tolerates small validation fluctuations without losing best fits."""
import pytest
from src.training_selection import patience_improved
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


def test_context128k_protocol_and_isolated_paths(tmp_path):
    import yaml
    cfg = config(ROOT / 'configs/experiment.yaml')
    assert cfg['target']['max_context_tokens'] == 131072
    assert cfg['training']['early_stopping_min_delta'] == .001
    assert cfg['paths']['data'] == str(ROOT / 'data/runs/context128k')
    assert cfg['paths']['output'] == str(ROOT / 'outputs/runs/context128k')
    cfg['target']['max_context_tokens'] = 8192
    path = tmp_path / 'old.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    with pytest.raises(Blocked, match='registered protocol'):
        config(path)
