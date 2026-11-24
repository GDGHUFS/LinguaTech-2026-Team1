"""고정 대역폭 RBF MMD와 group 단위 label 순열 검정을 수행한다."""
import math
import random


def grouped_mmd(x, y, groups_x, groups_y, *, sigma, permutations=2000, seed=42):
    """사전 고정한 sigma로 비편향 MMD 제곱값과 group 단위 순열 p값을 구한다.

    group 전체를 함께 재배정한다. 풀별 독립 group이 둘 미만이면 NA를 반환하며, 비편향 추정값은 음수일 수도 있다."""
    import numpy as np
    if not math.isfinite(sigma) or sigma <= 0 or type(permutations) is not int or permutations < 1:
        raise ValueError('Pre-frozen positive bandwidth and permutation count required')
    x,y = np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite embeddings with common feature dimension required')
    if len(x) != len(groups_x) or len(y) != len(groups_y):
        raise ValueError('Embedding/group length mismatch')
    if set(groups_x) & set(groups_y):
        raise ValueError('Train/OOD upstream groups overlap')
    if min(len(set(groups_x)),len(set(groups_y))) < 2 or min(len(x),len(y)) < 2:
        return dict(status='NA',reason='two_independent_groups_per_pool_required')
    z = np.concatenate((x,y))
    distances = ((z[:,None,:]-z[None,:,:])**2).sum(-1)
    kernel = np.exp(-distances/(2*sigma*sigma))
    groups = list(groups_x)+list(groups_y)
    keys = sorted(set(groups))
    buckets = {g:[i for i,v in enumerate(groups) if v==g] for g in keys}
    def statistic(a,b):
        aa,bb,ab = kernel[np.ix_(a,a)],kernel[np.ix_(b,b)],kernel[np.ix_(a,b)]
        return float((aa.sum()-np.trace(aa))/(len(a)*(len(a)-1)) +
                     (bb.sum()-np.trace(bb))/(len(b)*(len(b)-1))-2*ab.mean())
    observed = statistic(list(range(len(x))),list(range(len(x),len(z))))
    rng,exceed = random.Random(seed),0
    for _ in range(permutations):
        left = set(rng.sample(keys,len(set(groups_x))))
        a = [i for g in keys if g in left for i in buckets[g]]
        b = [i for g in keys if g not in left for i in buckets[g]]
        exceed += statistic(a,b) >= observed
    return dict(status='OK',mmd2_unbiased=observed,p_value=(1+exceed)/(1+permutations),
                permutations=permutations,sigma=sigma,resampling_unit='whole_upstream_groups',
                interpretation='MMD/permutation representation shift only; not a full H2ST implementation')
