"""Required G1 and independent prevalence sanity. Fits never receive held-out rows."""
import argparse
import json
import pickle
from pathlib import Path

from src.data.labeling import target_prompt, validate_observed_provenance, validate_target_cohort
from src.evaluation.metrics import binary_metrics
from src.evaluation.protocol import _manifest, _valid, digest


def validate_config(config):
    if config.get('calibration') != 'none' or config.get('class_weight') is not None or config['C_grid'] != [.01,.1,1.,10.]:
        raise ValueError('Freeze supported baseline protocol: no calibration, no class weighting, specified C grid')
    from src.utils.experiment import seed_metadata
    seed_metadata(config['seeds'])
    import yaml
    training = yaml.safe_load(Path(config.get('training_config','configs/train.yaml')).read_text(encoding='utf-8'))
    if config['seeds'] != training['seeds']:
        raise ValueError('External baselines and core predictors must share training seeds')


def partitions(records, manifest):
    refs = _manifest(manifest, records)
    valid = [r for r in refs.values() if _valid(r)]
    if any('target_execution' in r for r in valid):
        validate_target_cohort(valid)
    train = sorted((r for r in valid if r['split'] == 'train'), key=lambda r: r['sample_id'])
    val = sorted((r for r in valid if r['split'] == 'validation'), key=lambda r: r['sample_id'])
    if {r['failure_label_unsafe'] for r in train} != {0, 1} or {r['failure_label_unsafe'] for r in val} != {0, 1}:
        raise ValueError('GATE_FAILED: train and validation require both failure classes')
    return train, val, sorted(valid, key=lambda r: r['sample_id'])


def select_lr(xtrain, ytrain, xval, yval, seed, c_grid):
    """학습 데이터로 후보별 logistic regression을 학습하고 검증 AUPRC, 낮은 Brier, 낮은 C 순으로 선택한다."""
    from sklearn.linear_model import LogisticRegression
    selected = None
    for c in c_grid:
        model = LogisticRegression(C=c, max_iter=2000, random_state=seed, class_weight=None).fit(xtrain, ytrain)
        metrics = binary_metrics(yval, model.predict_proba(xval)[:, 1].tolist())
        rank = (metrics['auprc'], -metrics['brier'], -c)
        if selected is None or rank > selected[0]:
            selected = rank, model, c, metrics
    return selected[1:]


def prediction_rows(rows, probabilities, model, seed, artifact_hash):
    access = 'target_pre_generation_hidden_states' if model == 'P1' else 'text_pre_execution'
    return [dict(sample_id=r['sample_id'], split=r['split'], group_id=r['group_id'],
                 model=model, seed=seed, checkpoint_hash=artifact_hash,
                 failure_probability=float(p), information_access=access) for r, p in zip(rows, probabilities)]


def fit_g1(records, manifest, template, seed=42, c_grid=(.01, .1, 1., 10.)):
    """학습 데이터로만 TF-IDF와 분류기를 학습하고 검증 데이터로 C를 선택한다.

    학습된 객체, 출처 정보, 전체 유효 표본의 예측과 학습 양성 비율을 쓰는 대조 점수를 반환한다."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    train, val, all_rows = partitions(records, manifest)
    # 고정된 Target과 같은 자연어 입력만 사용하고 gold 필드는 제외한다.
    texts = lambda rows: [target_prompt(r, template) for r in rows]
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    xtrain = vectorizer.fit_transform(texts(train))
    classifier, c, quality = select_lr(xtrain, [r['failure_label_unsafe'] for r in train],
        vectorizer.transform(texts(val)), [r['failure_label_unsafe'] for r in val], seed, c_grid)
    metadata = dict(model='G1', seed=seed, train_ids=[r['sample_id'] for r in train],
        validation_ids=[r['sample_id'] for r in val], split_manifest_hash=manifest['manifest_hash'],
        records_hash=digest(records), template_hash=digest(template), selected_C=c, validation=quality,
        calibration='none', class_weight=None, vocabulary=vectorizer.vocabulary_, idf=vectorizer.idf_.tolist(),
        coef=classifier.coef_.tolist(), intercept=classifier.intercept_.tolist())
    checkpoint_hash = digest(metadata)
    predictions = prediction_rows(all_rows, classifier.predict_proba(vectorizer.transform(texts(all_rows)))[:, 1], 'G1', seed, checkpoint_hash)
    prior = sum(r['failure_label_unsafe'] for r in train) / len(train)
    sanity = prediction_rows(all_rows, [prior] * len(all_rows), 'majority', seed, digest(dict(train_ids=metadata['train_ids'], prior=prior)))
    return dict(vectorizer=vectorizer, classifier=classifier), metadata, predictions, sanity


def main(argv=None):
    import yaml
    from evaluate import read_rows
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/baselines.yaml')
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    validate_config(config)
    records = read_rows(config['records'])
    from src.data.source_policy import enforce_source_policy
    from src.data.approvals import enforce_execution, research_rows
    research_rows(records)
    enforce_execution(records, mode=config.get('execution_mode','final'), qa_path=config.get('qa_manifest'), freeze_path=config.get('freeze_manifest'), data_config=config.get('data_config','configs/data.yaml'), config_paths=[config.get('data_config','configs/data.yaml'),'configs/model.yaml','configs/train.yaml',args.config])
    manifest = json.loads(Path(config['manifest']).read_text(encoding='utf-8'))
    template = Path(config['target_prompt']).read_text(encoding='utf-8-sig')
    for row in records:
        if row.get('data_kind')=='mock' or row.get('source_provenance',{}).get('fixture'):
            raise ValueError('Synthetic fixtures are test-only, not research baseline inputs')
        if _valid(row):
            validate_observed_provenance(row,template)
    for seed in config['seeds']:
        out = Path(config['g1_output']) / f'seed_{seed}'
        if out.exists():
            raise ValueError('Use a new artifact directory; existing results are preserved')
        fitted, metadata, predictions, sanity = fit_g1(records, manifest, template, seed, config['C_grid'])
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
        for name, rows in (('predictions', predictions), ('majority_sanity', sanity)):
            (out/f'{name}.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows), encoding='utf-8')


if __name__ == '__main__':
    main()
