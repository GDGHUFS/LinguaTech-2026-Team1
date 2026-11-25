"""Five-model common-cohort evaluation, paired inference and descriptive plots."""
import time
from pathlib import Path
import joblib
import numpy as np
import torch
from .contracts import (Blocked, digest, file_hash, code_hash, paths, read_json, read_jsonl, write_json,
                        write_jsonl, gate)
from .training import feature_tensors, load_probe, predict_probe
from .representation import SampleCache
from .baselines import baseline_input
from .metrics import metrics, paired_bootstrap

MODELS = ('g1', 'l1', 'p1', 't1', 'conshift')


def assert_cohort(rows):
    ids = [r['sample_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise Blocked('Duplicate evaluation sample')
    if any(set(r['probabilities']) != set(MODELS) for r in rows):
        raise Blocked('Models do not share exactly the same cohort')


def score_rows(rows, cfg, cache, checkpoint_dir=None, resume=False, run_signature=None):
    _, output = paths(cfg)
    cp = output / 'checkpoints'
    trained = read_json(output / 'reports/training_manifest.json')
    cohort_fingerprint = read_json(paths(cfg)[0] / 'manifests/collection_lock.json')['fingerprint']
    if trained['collection_fingerprint'] != cohort_fingerprint or trained['target_config'] != cfg['target']:
        raise Blocked('Checkpoint/Target extraction protocol mismatch')
    started = time.perf_counter()
    features = feature_tensors(rows, cache)
    feature_seconds = time.perf_counter() - started
    predictions, thresholds, latency, counts = {}, {}, {}, {}
    gates, gate_parameters = None,None
    for name in MODELS:
        checkpoint_path = Path(checkpoint_dir) / (name+'.json') if checkpoint_dir is not None else None
        if checkpoint_path is not None and checkpoint_path.exists():
            if not resume:
                raise Blocked('Evaluation model checkpoint exists; use --resume')
            saved = read_json(checkpoint_path)
            checksum = saved.pop('checkpoint_hash')
            if (digest(saved) != checksum or saved['run_signature'] != run_signature
                    or saved['model_name'] != name
                    or saved['sample_hash'] != digest([r['sample_id'] for r in rows])):
                raise Blocked('Evaluation checkpoint/cohort/settings mismatch')
            predictions[name] = np.asarray(saved['probabilities'])
            thresholds[name],counts[name],latency[name] = saved['threshold'],saved['parameter_count'],saved['timing']
            if name == 'conshift':
                gates,gate_parameters = np.asarray(saved['gates']),saved['gate_parameters']
            continue
        if name in ('g1', 'l1'):
            model = joblib.load(cp / (name + '.joblib'))  # own local training artifacts only
            meta = read_json(cp / (name + '.json'))
            started = time.perf_counter()
            predictions[name] = model.predict_proba(baseline_input(name, rows))[:, 1]
        else:
            model, meta = load_probe(cp / (name + '.pt'))
            started = time.perf_counter()
            predictions[name] = predict_probe(model, features)
            if name == 'conshift':
                with torch.inference_mode():
                    gates = model.gate_value(features[2]).numpy().ravel()
                gate_parameters = {'a':float(model.gate_a.detach()[0]),'c':float(model.gate_c.detach()[0])}
        elapsed = time.perf_counter() - started
        if (meta['cohort_hash'] != trained['cohort_hash'] or meta['threshold_source'] != 'validation'
                or meta['collection_fingerprint'] != cohort_fingerprint):
            raise Blocked('Checkpoint metadata/common cohort mismatch')
        thresholds[name] = meta['threshold']
        counts[name] = meta['parameter_count']
        latency[name] = {'prediction_seconds': elapsed, 'seconds_per_turn': elapsed / len(rows)}
        if checkpoint_path is not None:
            saved = {'model_name':name,'run_signature':run_signature,'sample_hash':digest([r['sample_id'] for r in rows]),
                'probabilities':predictions[name].tolist(),'threshold':thresholds[name],
                'parameter_count':counts[name],'timing':latency[name]}
            if name == 'conshift':
                saved.update(gates=gates.tolist(),gate_parameters=gate_parameters)
            saved['checkpoint_hash'] = digest(saved)
            write_json(checkpoint_path,saved)
    result = []
    for idx, r in enumerate(rows):
        result.append({k: r[k] for k in ('sample_id', 'group_id', 'trajectory_id', 'turn_index', 'failure_label', 'split')})
        result[-1].update(distance=float(features[2][idx, 0]), gate=float(gates[idx]),
                          probabilities={m: float(predictions[m][idx]) for m in MODELS},
                          thresholds=thresholds)
    assert_cohort(result)
    latency['feature_assembly_seconds_including_disk'] = feature_seconds
    latency['target_prefill_seconds'] = sum(r['prefill_seconds'] for r in rows)
    latency['target_generation_seconds'] = sum(r['generation_seconds'] for r in rows)
    latency['target_total_including_retries_seconds'] = sum(r['total_seconds_including_retries'] for r in rows)
    latency['protocol'] = 'Separate hidden-state prefill forward plus generation prefill; both included. Probe timed in one CPU batch; no warmup or hardware-independent claim.'
    return result, thresholds, latency, counts, gate_parameters


def table(rows, thresholds):
    return {m: metrics([r['failure_label'] for r in rows], [r['probabilities'][m] for r in rows], thresholds[m]) for m in MODELS}


def subgroup(rows, key, cuts, thresholds):
    if cuts[0] == cuts[1]:
        return {'status': 'EXCLUDED_IDENTICAL_CUTPOINTS', 'cutpoints': cuts}
    result = {'cutpoints': cuts, 'cutpoints_source': 'train', 'bins': {}}
    names = ('early', 'middle', 'late') if key == 'turn_index' else ('low', 'middle', 'high')
    for index, name in enumerate(names):
        selected = [r for r in rows if int(np.searchsorted(cuts, r[key], side='left')) == index]
        report = table(selected, thresholds)
        ap_c, ap_t = report['conshift']['AP'], report['t1']['AP']
        result['bins'][name] = {'n': len(selected), 'groups': len({r['group_id'] for r in selected}),
                               'prevalence': float(np.mean([r['failure_label'] for r in selected])) if selected else None,
                               'mean_gate': float(np.mean([r['gate'] for r in selected])) if selected else None,
                               'metrics': report, 'conshift_minus_t1_ap': ap_c - ap_t if ap_c is not None and ap_t is not None else None}
    return result


def figures(rows, output, split):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    y = np.array([r['failure_label'] for r in rows])
    out = output / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    for kind in ('pr', 'roc', 'reliability'):
        fig, ax = plt.subplots(figsize=(6, 5))
        for m in MODELS:
            p = np.array([r['probabilities'][m] for r in rows])
            if kind == 'pr':
                precision, recall, _ = precision_recall_curve(y, p)
                ax.step(recall, precision, where='post', label=m)
            elif kind == 'roc':
                fpr, tpr, _ = roc_curve(y, p)
                ax.plot(fpr, tpr, label=m)
            else:
                from .metrics import calibration
                _, bins = calibration(y, p)
                ax.plot([b['mean_probability'] for b in bins], [b['failure_frequency'] for b in bins], marker='o', label=m)
        ax.set(xlabel={'pr':'Recall', 'roc':'False positive rate', 'reliability':'Mean predicted failure probability'}[kind],
               ylabel={'pr':'Precision', 'roc':'True positive rate', 'reliability':'Observed failure frequency'}[kind],
               title=f'{split}: {kind}', xlim=(0, 1), ylim=(0, 1))
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f'{split}_{kind}.png', dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([r['distance'] for r in rows], [r['gate'] for r in rows], s=8, alpha=.5)
    ax.set(xlabel='Cosine distance', ylabel='Learned gate', ylim=(0, 1), title=f'{split}: gate vs distance')
    fig.tight_layout()
    fig.savefig(out / f'{split}_gate_distance.png', dpi=160)
    plt.close(fig)


def evaluate(cfg,resume=False):
    data, output = paths(cfg)
    cov = read_json(Path(cfg['paths'].get('collection_output',output)) / 'reports/label_coverage.json')
    if cov['status'] != 'PASS':
        raise Blocked('G3 label coverage did not pass')
    labeled = read_jsonl(data / 'labeled/turns.jsonl')
    if digest(labeled) != cov['labeled_hash']:
        raise Blocked('Labeled data changed after coverage audit')
    all_rows = [r for r in labeled if r['eligible']]
    training = read_json(output / 'reports/training_manifest.json')
    if (digest([r['sample_id'] for r in all_rows]) != training['cohort_hash']
            or training['labeled_hash'] != cov['labeled_hash']):
        raise Blocked('Evaluation cohort changed after training')
    cache = SampleCache(data)
    run = read_json(data / 'manifests/collection_lock.json')
    for r in all_rows:
        cache.load(r['sample_id'], run['fingerprint'], r['prompt_hash'], r['prompt_token_ids_hash'])
    cuts = read_json(output / 'checkpoints/analysis_cutpoints.json')
    split_manifest = read_json(data / 'manifests/split_manifest.json')
    state_dir = output / 'evaluation'
    state_dir.mkdir(parents=True,exist_ok=True)
    if any(p.name != '.gitkeep' for p in state_dir.iterdir()) and not resume:
        raise Blocked('Evaluation checkpoints already exist; use --resume')
    artifacts = [output / 'checkpoints' / (name+suffix) for name in MODELS
                 for suffix in (('.joblib','.json') if name in ('g1','l1') else ('.pt',))]
    evaluation_lock = {'labeled_hash':cov['labeled_hash'],'training_manifest':digest(training),
        'settings':cfg['evaluation'],'split_hash':split_manifest['manifest_hash'],
        'code_hash':code_hash(),'model_artifacts':{p.name:file_hash(p) for p in artifacts}}
    signature = digest(evaluation_lock)
    write_json(state_dir / 'run.json',evaluation_lock,immutable=True)
    result, latency, analysis = {}, {}, {}
    for split in ('id_test', 'ood'):
        if split == 'ood' and split_manifest['ood_status'] != 'ENABLED':
            result[split] = {'status': split_manifest['ood_status']}
            continue
        if cov[split]['status'] != 'PASS':
            result[split] = {'status': 'EXCLUDED_INSUFFICIENT_LABEL_COVERAGE'}
            continue
        rows = [r for r in all_rows if r['split'] == split]
        predictions, thresholds, timing, counts, gate_parameters = score_rows(rows,cfg,cache,
            checkpoint_dir=state_dir / split,resume=resume,run_signature=signature)
        write_jsonl(output / 'predictions' / ('id_predictions.jsonl' if split == 'id_test' else 'ood_predictions.jsonl'), predictions)
        result[split] = {'status': 'PASS', 'models': table(predictions, thresholds), 'parameter_counts': counts}
        latency[split] = timing
        analysis[split] = {'gate_parameters': gate_parameters,
                           'turn': subgroup(predictions, 'turn_index', cuts['turn'], thresholds),
                           'shift': subgroup(predictions, 'distance', cuts['distance'], thresholds)}
        if split == 'id_test':
            write_json(output / 'reports/paired_bootstrap.json', paired_bootstrap(predictions,
                cfg['evaluation']['bootstrap_repetitions'],cfg['evaluation']['bootstrap_seed'],
                checkpoint_path=state_dir / 'bootstrap.json',resume=resume))
        figures(predictions, output, split)
        write_json(state_dir / 'progress.json',{'run_signature':signature,'status':'RUNNING',
            'completed_splits':list(result),'last_completed_split':split})
    if result.get('ood', {}).get('status') == 'PASS':
        result['ood_minus_id'] = {m: {metric: result['ood']['models'][m][metric] - result['id_test']['models'][m][metric]
            for metric in ('AP', 'AUROC', 'F1', 'Brier', 'ECE')} for m in MODELS}
    write_json(output / 'reports/metrics.json', result)
    write_json(output / 'reports/latency_report.json', latency)
    write_json(output / 'reports/gate_analysis.json', analysis)
    write_json(state_dir / 'progress.json',{'run_signature':signature,'status':'COMPLETE','completed_splits':list(result)})
    gate(cfg, 'G5', 'PASS', ood=result['ood']['status'], paired_group_bootstrap=True,
         raw_predictions_saved=True, same_cohort=True)
    return result
