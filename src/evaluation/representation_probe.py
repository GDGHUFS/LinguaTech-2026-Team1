"""A/C/C+ frozen ModernBERT state probes, separate from Target P1.

Only train fits scaler/LR; validation selects C by macro F1, then lower C.
Dependencies are lazy so a missing probe runtime cannot disable core eval.
"""
import argparse
import json
import math
from pathlib import Path

from .metrics import classification_metrics
from .protocol import STATE_CLASSES, SPLITS, _manifest, _valid, digest
from .statistics import mean_sd
from src.utils.experiment import seed_metadata


def validate_embeddings(embeddings, records, manifest, protocol, *, test_only=False):
    if bool(protocol.get('test_only')) != test_only:
        raise ValueError('Test-only protocol cannot enter production probe')
    if not test_only:
        from src.data.approvals import enforce_execution, research_rows
        research_rows(records)
        enforce_execution(records, mode='final', **protocol.get('execution_approval',{}))
    frozen = dict(protocol)
    checksum = frozen.pop('protocol_hash',None)
    if not frozen.get('frozen') or checksum != digest(frozen):
        raise ValueError('Frozen evaluation/probe protocol required')
    if frozen.get('probe') != {'representation':'last_layer_cls', 'C_grid':[.01,.1,1.,10.],
                               'selection':'validation_macro_f1_then_lower_C','class_weight':None}:
        raise ValueError('Probe extraction/selection settings must be frozen before test')
    refs = _manifest(manifest,records)
    if manifest['manifest_hash'] != frozen['split_manifest_hash'] or frozen['record_hashes'] != {sid:digest(r) for sid,r in sorted(refs.items())}:
        raise ValueError('Frozen probe records/manifest mismatch')
    eligible = {sid:r for sid,r in refs.items() if _valid(r)}
    indexed, dimensions, checkpoint_ids = {},set(),{}
    for row in embeddings:
        if row.get('model') not in ('A','C','C+') or row.get('seed') not in protocol['seeds']:
            raise ValueError('Unexpected probe encoder/seed')
        if row['sample_id'] not in refs or any(row.get(k) != refs[row['sample_id']][k] for k in ('split','group_id')):
            raise ValueError('Embedding sample/split/group mismatch')
        if row.get('encoder_frozen') is not True or row.get('representation') != 'last_layer_cls':
            raise ValueError('Only predeclared frozen last-layer CLS features accepted')
        feature = row.get('cls')
        if not isinstance(feature,list) or not feature or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in feature):
            raise ValueError('Finite embedding vector required')
        dimensions.add(len(feature))
        run = (row['model'],row['seed'])
        checkpoint = row.get('checkpoint_hash')
        if not checkpoint or checkpoint_ids.setdefault(run,checkpoint) != checkpoint:
            raise ValueError('Inconsistent/missing encoder checkpoint provenance')
        key = run+(row['sample_id'],)
        if key in indexed:
            raise ValueError('Duplicate probe embedding ID')
        indexed[key] = row
    if len(dimensions) > 1:
        raise ValueError('Common encoder feature dimension required')
    active = [m for m in ('A','C','C+') if any(k[0]==m for k in indexed)]
    for model in active:
        for seed in protocol['seeds']:
            if any((model,seed,sid) not in indexed for sid in eligible):
                raise ValueError('Missing common probe sample IDs/seed')
    return eligible,indexed,active


def run_probes(embeddings,records,manifest,protocol, *, test_only=False):
    """고정된 CLS 표현에서 state를 예측하는 선형 probe를 학습한다.

    scaler와 분류기는 학습 분할에만 맞추고 C는 검증 macro F1로 선택한다. 의존성이 없으면 SKIPPED를 반환한다."""
    refs,indexed,active = validate_embeddings(embeddings,records,manifest,protocol,test_only=test_only)
    if not embeddings:
        return dict(status='SKIPPED',reason='no_exported_encoder_embeddings')
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return dict(status='SKIPPED',reason='scikit_learn_runtime_missing')
    report = dict(status='OK',experiment=seed_metadata(protocol['seeds']),purpose='frozen_A_C_C+_state_decodability_not_Target_P1',models={})
    for model in ('A','C','C+'):
        if model not in active:
            report['models'][model] = dict(status='SKIPPED',reason='encoder_not_exported')
            continue
        runs = {}
        for seed in protocol['seeds']:
            fields = {}
            for field,classes in STATE_CLASSES.items():
                partitions = {}
                for partition in ('train','validation')+SPLITS:
                    ids = sorted(sid for sid,r in refs.items() if r['split']==partition and
                                 (r.get('gold_state') or {}).get(field) is not None and
                                 field not in (r.get('gold_state') or {}).get('ambiguous_fields',[]))
                    partitions[partition] = (ids,[indexed[(model,seed,sid)]['cls'] for sid in ids],
                                             [refs[sid]['gold_state'][field] for sid in ids])
                train_ids,xtrain,ytrain = partitions['train']
                val_ids,xval,yval = partitions['validation']
                if len(set(ytrain)) < 2 or not yval:
                    fields[field] = dict(status='NA',reason='insufficient_train_classes_or_validation_labels')
                    continue
                if any(y not in classes for p in partitions.values() for y in p[2]):
                    raise ValueError('Unknown state label in frozen probe data')
                scaler = StandardScaler().fit(xtrain)
                ztrain,zval = scaler.transform(xtrain),scaler.transform(xval)
                selected = None
                for regularization in protocol['probe']['C_grid']:
                    classifier = LogisticRegression(C=regularization,max_iter=2000,random_state=seed,class_weight=None).fit(ztrain,ytrain)
                    quality = classification_metrics(yval,classifier.predict(zval).tolist(),classes)['macro_f1']
                    if selected is None or quality > selected[0]:
                        selected = quality,regularization,classifier
                quality,regularization,classifier = selected
                tests = {}
                for partition in SPLITS:
                    ids,x,y = partitions[partition]
                    predicted = classifier.predict(scaler.transform(x)).tolist() if x else []
                    tests[partition] = dict(classification_metrics(y,predicted,classes),sample_ids=ids)
                fields[field] = dict(status='OK',selected_C=regularization,validation_macro_f1=quality,
                    train_ids=train_ids,validation_ids=val_ids,test=tests,
                    scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),
                    classes=classifier.classes_.tolist(),coef=classifier.coef_.tolist(),intercept=classifier.intercept_.tolist())
            runs[str(seed)] = fields
        summary = {field:{partition:{metric:mean_sd([runs[str(s)][field].get('test',{}).get(partition,{}).get(metric) for s in protocol['seeds']])
                    for metric in ('macro_f1','balanced_accuracy')} for partition in SPLITS} for field in STATE_CLASSES}
        report['models'][model] = dict(status='OK',seeds=runs,summary=summary)
    return report


def main(argv=None):
    from evaluate import read_rows
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('embeddings','records','manifest','protocol','out'):
        parser.add_argument('--'+option,required=True)
    parser.add_argument('--test-only',action='store_true')
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        parser.error('Use a new output path/revision')
    result = run_probes(read_rows(args.embeddings),read_rows(args.records),
                        json.loads(Path(args.manifest).read_text(encoding='utf-8')),
                        json.loads(Path(args.protocol).read_text(encoding='utf-8')),test_only=args.test_only)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status']}))


if __name__ == '__main__':
    main()
