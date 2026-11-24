"""P1: validate frozen Target activation provenance, then train-only scaler/LR."""
import argparse
import json
import math
import pickle
from pathlib import Path
from .baselines_pre import partitions, select_lr, prediction_rows, validate_config
from .protocol import digest


def validate_activations(records, activations):
    """공통 표본 ID와 label 수집 당시 Target·입력·추출 방식이 같은지 확인한다. 응답 생성 이후의 표현은 ValueError로 거부한다."""
    indexed, dimension, policy = {}, None, None
    for a in activations:
        sid = a['sample_id']
        if sid in indexed:
            raise ValueError('Duplicate activation ID')
        indexed[sid] = a
    if set(indexed) != {r['sample_id'] for r in records}:
        raise ValueError('P1 must cover exactly the common valid-Target sample IDs')
    for r in records:
        a = indexed[r['sample_id']]
        execution = r.get('target_execution') or {}
        for field in ('model_id', 'model_revision', 'chat_template_hash'):
            if not execution.get(field) or a.get(field) != execution[field]:
                raise ValueError('P1 Target does not match label-collection Target')
        if a.get('prompt_hash') != r.get('prompt_hash') or not a.get('prompt_hash'):
            raise ValueError('P1 prompt differs from label collection')
        if a.get('generation_tokens') != 0 or a.get('representation') != 'target_pregen_middle_final_nonpadding' or a.get('encoder_frozen') is not True:
            raise ValueError('P1 requires frozen zero-generation input activations')
        if a.get('layer_index', 0) != a.get('layer_depth', 0) // 2 or a.get('layer_index', 0) < 1:
            raise ValueError('Unexpected P1 extraction layer')
        if a.get('token_position', -1) < 0 or not a.get('input_ids_hash') or a.get('truncation') is not False:
            raise ValueError('Missing P1 input/token provenance')
        signature = tuple(a.get(k) for k in ('model_id','model_revision','chat_template_hash','layer_index','layer_depth','padding_side','dtype','truncation'))
        policy = policy or signature
        if signature != policy or a.get('padding_side') not in ('left','right') or not a.get('dtype'):
            raise ValueError('Inconsistent P1 extraction policy')
        feature = a.get('hidden_state')
        if not isinstance(feature, list) or not feature or any(type(v) not in (float, int) or not math.isfinite(v) for v in feature):
            raise ValueError('Invalid P1 hidden vector')
        dimension = dimension or len(feature)
        if len(feature) != dimension:
            raise ValueError('P1 feature dimension mismatch')
    return indexed


def fit_p1(records, manifest, activations, seed=42, c_grid=(.01, .1, 1., 10.)):
    """학습 표현에만 scaler와 분류기를 맞추고 검증 점수로 C를 선택한다. 학습 객체, 출처 정보와 공통 표본 예측을 반환한다."""
    from sklearn.preprocessing import StandardScaler
    train, val, all_rows = partitions(records, manifest)
    indexed = validate_activations(all_rows, activations)
    x = lambda rows: [indexed[r['sample_id']]['hidden_state'] for r in rows]
    scaler = StandardScaler().fit(x(train))
    classifier, c, quality = select_lr(scaler.transform(x(train)), [r['failure_label_unsafe'] for r in train],
        scaler.transform(x(val)), [r['failure_label_unsafe'] for r in val], seed, c_grid)
    metadata = dict(model='P1', seed=seed, selected_C=c, validation=quality,
        train_ids=[r['sample_id'] for r in train], validation_ids=[r['sample_id'] for r in val],
        split_manifest_hash=manifest['manifest_hash'], records_hash=digest(records), activations_hash=digest(activations),
        scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
        coef=classifier.coef_.tolist(), intercept=classifier.intercept_.tolist(),
        calibration='none', class_weight=None, information_access='target_pre_generation_hidden_states')
    predictions = prediction_rows(all_rows, classifier.predict_proba(scaler.transform(x(all_rows)))[:, 1], 'P1', seed, digest(metadata))
    return dict(scaler=scaler, classifier=classifier), metadata, predictions


def main(argv=None):
    import yaml
    from evaluate import read_rows
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/baselines.yaml')
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    validate_config(config)
    records, activations = read_rows(config['records']), read_rows(config['activations'])
    from src.data.source_policy import enforce_source_policy
    from src.data.approvals import enforce_execution, research_rows
    research_rows(records)
    enforce_execution(records, mode=config.get('execution_mode','final'), qa_path=config.get('qa_manifest'), freeze_path=config.get('freeze_manifest'), data_config=config.get('data_config','configs/data.yaml'), config_paths=[config.get('data_config','configs/data.yaml'),'configs/model.yaml','configs/train.yaml',args.config])
    from src.data.labeling import validate_observed_provenance
    from .protocol import _valid
    template = Path(config['target_prompt']).read_text(encoding='utf-8-sig')
    for row in records:
        if row.get('data_kind')=='mock' or row.get('source_provenance',{}).get('fixture'):
            raise ValueError('Synthetic fixtures are test-only, not research baseline inputs')
        if _valid(row):
            validate_observed_provenance(row,template)
    manifest = json.loads(Path(config['manifest']).read_text(encoding='utf-8'))
    for seed in config['seeds']:
        out = Path(config['p1_output']) / f'seed_{seed}'
        if out.exists():
            raise ValueError('Use a new artifact directory')
        fitted, metadata, predictions = fit_p1(records, manifest, activations, seed, config['C_grid'])
        from src.utils.reproducibility import runtime_metadata
        from .registry import load_registry
        from src.utils.experiment import seed_metadata
        metadata.update(experiment=seed_metadata(config['seeds']),runtime=runtime_metadata(),config_hash=digest(config),model_registry_hash=load_registry()[1])
        out.mkdir(parents=True)
        (out/'fit.pkl').write_bytes(pickle.dumps(fitted))
        from src.utils.reproducibility import file_hash
        metadata['checkpoint_hash']=file_hash(out/'fit.pkl')
        for row in predictions:
            row['checkpoint_hash']=metadata['checkpoint_hash']
        (out/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
        (out/'predictions.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in predictions), encoding='utf-8')


if __name__ == '__main__':
    main()
