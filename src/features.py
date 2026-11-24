"""The only neural probe inputs: current and immediately previous hidden vectors."""
import torch
import torch.nn.functional as F


def valid_vector(h):
    return h.ndim >= 1 and bool(torch.isfinite(h).all()) and bool((h.float().norm(dim=-1) > 0).all())


def temporal_features(h_t, h_prev):
    if h_t.shape != h_prev.shape or not valid_vector(h_t) or not valid_vector(h_prev):
        raise ValueError('Invalid hidden representation')
    u_t = F.normalize(h_t.detach().float(), dim=-1, eps=1e-12)
    prev = F.normalize(h_prev.detach().float(), dim=-1, eps=1e-12)
    delta = u_t - prev
    distance = (1 - (u_t * prev).sum(dim=-1, keepdim=True)).clamp(0, 2)
    return u_t, delta, distance


def previous_record(current, previous):
    return (previous is not None and current['trajectory_id'] == previous['trajectory_id']
            and current['turn_index'] == previous['turn_index'] + 1)
