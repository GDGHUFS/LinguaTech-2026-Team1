"""Validation-only patience decisions, independent of best-checkpoint ranking."""
import math


def select_l2_candidate(candidates):
    """Validation AP, then Brier, then epoch, then predeclared grid order."""
    if not candidates:
        raise ValueError('Empty L2 search')
    return max(range(len(candidates)), key=lambda i: (
        candidates[i]['validation_ap'], -candidates[i]['validation_brier'],
        -candidates[i]['epoch'], -i))


def patience_improved(rank, reference, min_delta):
    """Accumulate AP gains from the last patience reset, not the last epoch.

    With positive min_delta, Brier tie-breaks still select the best checkpoint
    but do not extend training. Zero preserves the original AP/Brier rule.
    """
    if not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError('early_stopping_min_delta must be finite and nonnegative')
    if reference is None:
        return True
    if min_delta == 0:
        return rank > reference
    return rank[0] - reference[0] >= min_delta
