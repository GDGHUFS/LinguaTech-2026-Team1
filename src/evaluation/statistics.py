"""고정된 학습 seed에서 scenario group별 쌍을 유지해 불확실성을 추정한다. seed 간 표준편차는 별도로 계산한다."""
from collections import defaultdict
import math
import random
import statistics

from .metrics import average_precision


def mean_sd(values):
    good = [v for v in values if v is not None]
    return dict(mean=statistics.mean(good) if len(good) == len(values) and good else None,
                sd=statistics.stdev(good) if len(good) == len(values) and len(good) > 1 else None,
                n_valid=len(good), n_expected=len(values), status='OK' if len(good) == len(values) and good else 'NA')


def quantile(values, q):
    values = sorted(values)
    index = (len(values)-1)*q
    lo = int(index)
    hi = min(lo+1, len(values)-1)
    return values[lo]*(hi-index)+values[hi]*(index-lo) if hi != lo else values[lo]


def paired_bootstrap(labels, groups, scores_left, scores_right, *, repetitions=2000, seed=42, min_valid_fraction=.8):
    """같은 group을 모든 모델과 seed에 재표집해 seed별 AP 차이의 평균을 추정한다.

    신뢰구간은 등록된 학습 seed에 조건부다. 양측 p값에는 중심화한 paired bootstrap 귀무 근사와 plus-one 보정을 쓴다. 한 class만 남은 표집은 제외 횟수로 보고한다."""
    if type(repetitions) is not int or repetitions < 1 or not 0 < min_valid_fraction <= 1:
        raise ValueError('Invalid bootstrap protocol')
    if set(scores_left) != set(scores_right) or not scores_left:
        raise ValueError('Paired training seeds differ or are empty')
    if len(labels) != len(groups) or any(len(v) != len(labels) for v in list(scores_left.values())+list(scores_right.values())):
        raise ValueError('Paired samples differ')
    result = dict(status='NA', estimate=None, ci95=None, p_value=None, repetitions=repetitions,
                  valid_replicates=0, degenerate_replicates=0, n_groups=len(set(groups)),
                  uncertainty_unit='scenario_groups_conditional_on_fixed_training_seeds',
                  p_method='two_sided_centered_paired_bootstrap_plus_one')
    def difference(indices):
        y = [labels[i] for i in indices]
        deltas = []
        for training_seed in sorted(scores_left):
            a = average_precision(y, [scores_left[training_seed][i] for i in indices])
            b = average_precision(y, [scores_right[training_seed][i] for i in indices])
            if a is None or b is None:
                return None
            deltas.append(a-b)
        return statistics.mean(deltas)
    estimate = difference(list(range(len(labels))))
    if estimate is None or result['n_groups'] < 2:
        return dict(result, reason='two_classes_and_two_groups_required')
    buckets = defaultdict(list)
    for i, group in enumerate(groups):
        buckets[group].append(i)
    keys = sorted(buckets)
    rng, draws = random.Random(seed), []
    for _ in range(repetitions):
        indices = [i for key in rng.choices(keys, k=len(keys)) for i in buckets[key]]
        delta = difference(indices)
        if delta is not None:
            draws.append(delta)
    result.update(estimate=estimate, valid_replicates=len(draws), degenerate_replicates=repetitions-len(draws))
    if len(draws) < max(2, math.ceil(repetitions*min_valid_fraction)):
        return dict(result, reason='insufficient_non_degenerate_bootstrap_replicates')
    result.update(status='PARTIAL' if len(draws) != repetitions else 'OK',
                  ci95=[quantile(draws,.025), quantile(draws,.975)],
                  p_value=(1+sum(abs(d-estimate) >= abs(estimate) for d in draws))/(1+len(draws)))
    return result


def holm_correction(p_values):
    """p값이 NA인 가설도 사전 계획한 전체 검정 수에 포함해 Holm 보정한다."""
    for p in p_values.values():
        if p is not None and (not math.isfinite(p) or not 0 <= p <= 1):
            raise ValueError('Invalid p-value')
    ordered = sorted(p_values, key=lambda key: (p_values[key] if p_values[key] is not None else 1., key))
    previous, result = 0., {}
    for rank, key in enumerate(ordered):
        p = p_values[key]
        if p is None:
            result[key] = None
        else:
            previous = max(previous, min(1., p*(len(ordered)-rank)))
            result[key] = previous
    return result
