"""Read-only train/validation diagnostics; never used for fitting or selection."""
import numpy as np
from .metrics import metrics


def describe(values):
    x = np.asarray(values, dtype=float).ravel()
    return {'mean':float(x.mean()), 'std':float(x.std()),
            'min':float(x.min()), 'max':float(x.max()),
            'quantiles':dict(zip(('p05','p25','p50','p75','p95'),
                                np.quantile(x,[.05,.25,.5,.75,.95]).tolist()))}


def sigmoid(x):
    return np.exp(-np.logaddexp(0., -np.asarray(x, dtype=float)))


def scores(y, logits, threshold):
    result = metrics(y, sigmoid(logits), threshold)
    result['BCE'] = float(np.mean(np.logaddexp(0., logits) - y * logits))
    result['predicted_positive_count'] = int((sigmoid(logits) >= threshold).sum())
    return result


def diagnose(features, y, weight, bias, gate_parameters=None, fixed_gate=.5,
             threshold=.5, rows=None, cutpoints=None):
    """Ablations reuse this fitted head: they are NOT refitted baselines."""
    u, delta, distance = [np.asarray(x, dtype=float) for x in features]
    y = np.asarray(y, dtype=float)
    weight = np.asarray(weight, dtype=float).ravel()
    d = distance.ravel()
    current = u @ weight[:u.shape[1]]
    raw_delta = delta @ weight[u.shape[1]:] if weight.size > u.shape[1] else np.zeros(len(y))
    if gate_parameters is None:
        gate = np.ones(len(y)) if weight.size > u.shape[1] else np.zeros(len(y))
    else:
        gate = sigmoid(gate_parameters['a'] * d + gate_parameters['c'])
    contribution = gate * raw_delta
    logits = current + contribution + float(bias)
    report = {'metrics':scores(y,logits,threshold), 'distance':describe(d),
              'gate':describe(gate), 'gate_parameters':gate_parameters,
              'gate_fraction_below_005':float((gate < .05).mean()),
              'gate_fraction_above_095':float((gate > .95).mean()),
              'current_logit':describe(current), 'raw_delta_logit':describe(raw_delta),
              'gated_delta_logit':describe(contribution), 'bias':float(bias),
              'delta_to_current_mean_abs_ratio':float(np.abs(contribution).mean() /
                                                       max(np.abs(current).mean(),1e-12)),
              'delta_current_opposite_sign_fraction':float((contribution * current < 0).mean()),
              'class_conditioned':{}, 'subgroups':{},
              'fixed_gate_source':'train mean of this epoch; applied unchanged to validation',
              'ablation_protocol':'Same fitted head, no retraining; diagnostic only',
              'ablations':{}}
    for label in (0,1):
        mask = y == label
        if mask.any():
            report['class_conditioned'][str(label)] = {
                'n':int(mask.sum()), 'gate':describe(gate[mask]),
                'distance':describe(d[mask]), 'gated_delta_logit':describe(contribution[mask])}
    if weight.size > u.shape[1]:
        for name, value in (('gate_0',0.),('gate_05',.5),('gate_1',1.),('gate_train_mean',fixed_gate)):
            report['ablations'][name] = {'gate':float(value),
                **scores(y, current + value * raw_delta + float(bias), threshold)}
    groups = {'distance':d}
    if rows is not None:
        groups['turn'] = np.array([r['turn_index'] for r in rows])
        report['groups'] = len({r['group_id'] for r in rows})
    for key, values in groups.items():
        cuts = (cutpoints or {}).get(key)
        if cuts is None:
            continue
        if cuts[0] == cuts[1]:
            report['subgroups'][key] = {'status':'IDENTICAL_TRAIN_CUTPOINTS','cutpoints':cuts}
            continue
        bins = np.searchsorted(cuts, values, side='left')
        report['subgroups'][key] = {'cutpoints':cuts, 'cutpoints_source':'train','bins':{}}
        for index, name in enumerate(('low','middle','high')):
            mask = bins == index
            entry = scores(y[mask],logits[mask],threshold) if mask.any() else {'n':0}
            if rows is not None:
                entry['groups'] = len({r['group_id'] for r,m in zip(rows,mask) if m})
            report['subgroups'][key]['bins'][name] = entry
    records = []
    for i in range(len(y)):
        record = {k:rows[i][k] for k in ('sample_id','group_id','trajectory_id','turn_index')} if rows is not None else {'row_index':i}
        record.update(failure_label=int(y[i]), probability=float(sigmoid(logits[i])),
                      threshold=float(threshold), gate=float(gate[i]), distance=float(d[i]),
                      current_logit=float(current[i]), raw_delta_logit=float(raw_delta[i]),
                      gated_delta_logit=float(contribution[i]), bias=float(bias))
        records.append(record)
    return report, records
