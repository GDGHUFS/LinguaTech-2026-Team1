"""AP uses sklearn's non-interpolated average_precision_score."""
import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from .contracts import Blocked, digest, read_json, write_json


def arrays(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid binary labels or prediction shape')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid probabilities')
    return y, p


def calibration(y, p):
    y, p = arrays(y, p)
    bins = np.minimum((p * 10).astype(int), 9)
    rows = []
    ece = 0.
    for b in range(10):
        mask = bins == b
        if not mask.any():
            continue
        confidence, frequency = float(p[mask].mean()), float(y[mask].mean())
        ece += float(mask.mean()) * abs(confidence - frequency)
        rows.append({'bin': b, 'count': int(mask.sum()), 'mean_probability': confidence,
                     'failure_frequency': frequency})
    return ece, rows


def metrics(y, p, threshold):
    y, p = arrays(y, p)
    if not len(y):
        return {'n': 0, 'AP': None, 'AUROC': None, 'F1': None, 'Brier': None, 'ECE': None, 'prevalence': None, 'undefined_reason': 'EMPTY_COHORT'}
    both = len(np.unique(y)) == 2
    ece, bins = calibration(y, p)
    return {'n': len(y), 'AP': float(average_precision_score(y, p)) if both else None,
            'AUROC': float(roc_auc_score(y, p)) if both else None,
            'F1': float(f1_score(y, p >= threshold, zero_division=0)),
            'Brier': float(np.mean((p - y)**2)), 'ECE': ece,
            'prevalence': float(y.mean()), 'threshold': float(threshold),
            'calibration_bins': bins, 'undefined_reason': None if both else 'SINGLE_CLASS'}


def select_threshold(y, p, split='validation'):
    if split != 'validation':
        raise Blocked('Threshold selection is allowed on validation only')
    y, p = arrays(y, p)
    if len(np.unique(y)) != 2:
        raise Blocked('Validation must contain both classes')
    candidates = np.unique(np.r_[0., p, 1.])
    return float(max(candidates, key=lambda t: (f1_score(y, p >= t, zero_division=0), t)))


def paired_bootstrap(rows, repetitions=2000, seed=42, checkpoint_path=None, resume=False):
    if not rows:
        raise Blocked('No predictions for paired bootstrap')
    groups = sorted({r['group_id'] for r in rows})
    members = {g: np.array([i for i, r in enumerate(rows) if r['group_id'] == g]) for g in groups}
    y = np.array([r['failure_label'] for r in rows])
    probabilities = {m: np.array([r['probabilities'][m] for r in rows]) for m in ('p1', 't1', 'conshift')}
    differences = {'t1-p1': [], 'conshift-t1': [], 'conshift-p1': []}
    rng = np.random.default_rng(seed)
    undefined = 0
    signature = digest({'rows':rows,'repetitions':repetitions,'seed':seed})
    start = 0
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        if not resume:
            raise Blocked('Bootstrap checkpoint exists; use --resume')
        state = read_json(checkpoint_path)
        checksum = state.pop('checkpoint_hash')
        if digest(state) != checksum or state['signature'] != signature:
            raise Blocked('Bootstrap checkpoint/prediction/settings mismatch')
        start,undefined,differences = state['completed_repetitions'],state['undefined'],state['differences']
        rng.bit_generator.state = state['rng_state']
    def checkpoint(completed):
        if checkpoint_path is not None:
            state = {'signature':signature,'completed_repetitions':completed,'undefined':undefined,
                'differences':differences,'rng_state':rng.bit_generator.state}
            state['checkpoint_hash'] = digest(state)
            write_json(checkpoint_path,state)
    for iteration in range(start,repetitions):
        chosen = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([members[g] for g in chosen])
        if len(np.unique(y[indices])) != 2:
            undefined += 1
        else:
            ap = {m: float(average_precision_score(y[indices], p[indices])) for m, p in probabilities.items()}
            for key in differences:
                a, b = key.split('-')
                differences[key].append(ap[a] - ap[b])
        if (iteration+1) % 50 == 0 or iteration+1 == repetitions:
            checkpoint(iteration+1)
    result = {'repetitions': repetitions, 'seed': seed, 'sampling_unit': 'group',
              'undefined_replicates': undefined, 'valid_replicates': repetitions - undefined,
              'stable': repetitions - undefined >= 1800, 'differences': {}}
    for key, values in differences.items():
        ci = np.quantile(values, [.025, .975]).tolist() if values else None
        interpretation = 'UNDEFINED' if ci is None else ('IMPROVEMENT' if ci[0] > 0 else 'DEGRADATION' if ci[1] < 0 else 'UNCERTAIN')
        result['differences'][key] = {'percentile_95_ci': ci, 'interpretation': interpretation,
                                      'replicate_differences': values}
    return result
