"""고정된 저장 예측으로 평가한다. 테스트 데이터로 모델을 학습하지 않는다."""
from collections import Counter
import hashlib
import json
from src.utils.experiment import seed_metadata

from src.data.splitting import connected_groups
from .metrics import binary_metrics, classification_metrics, probability, select_threshold
from .statistics import holm_correction, mean_sd, paired_bootstrap

SPLITS = ('id_test', 'ood_composition', 'ood_template')
STATE_CLASSES = {'authorization': ('ALLOW', 'DENY', 'UNRESOLVED'),
                 'rule_status': ('ACTIVE', 'REVOKED', 'EXPIRED', 'NONE'),
                 'scope_match': ('MATCH', 'MISMATCH', 'INCOMPLETE')}
FAIR_MODELS = {'majority', 'G1', 'P1', 'A', 'B_pred', 'C', 'C+', 'P2'}


def digest(value):
    return 'sha256:'+hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _manifest(manifest, records):
    if manifest.get('ood_primitive_coverage') is False:
        raise ValueError('OOD composition gate failed: primitive operations absent from train')
    payload = dict(manifest)
    recorded = payload.pop('manifest_hash', None)
    # B와 hash가 같도록 json.dumps의 기본 ensure_ascii=True를 유지한다.
    expected = 'sha256:'+hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if recorded != expected:
        raise ValueError('Split manifest content/hash mismatch')
    if 'ood_templates' in manifest and 'ood_compositions' in manifest:
        from src.data.splitting import check_leakage
        check_leakage(records,manifest['ood_compositions'],manifest['ood_templates'],manifest.get('fewshot_template_families',()))
    by_id = {}
    for row in records:
        sid = row['sample_id']
        if sid in by_id:
            raise ValueError('Duplicate reference sample ID')
        if manifest['assignments'].get(sid) != row.get('split'):
            raise ValueError('Reference split disagrees with manifest')
        if row.get('split_manifest_hash') not in (None,recorded):
            raise ValueError('Record belongs to a different split manifest')
        if row.get('split','').startswith('ood_') and 'ood_templates' in manifest:
            from src.data.splitting import holdout_kind
            member = any(holdout_kind(row,manifest['ood_compositions'],manifest['ood_templates']))
            if row.get('evaluation_eligible') is not member:
                raise ValueError('OOD evaluation membership disagrees with the predeclared holdout')
        if not all(row.get(k) for k in ('group_id', 'source_dataset', 'source_seed_id')):
            raise ValueError('Missing upstream group/source identity')
        by_id[sid] = row
    if set(by_id) != set(manifest['assignments']):
        raise ValueError('Reference sample IDs must exactly match manifest')
    for group in connected_groups(records):
        if len({records[i]['split'] for i in group}) != 1:
            raise ValueError('Cross-split scenario/seed leakage')
        if len({records[i]['group_id'] for i in group}) != 1:
            raise ValueError('Connected seed family has inconsistent bootstrap group IDs')
    return by_id


def _valid(row):
    if row.get('parse_status') != 'VALID':
        if any(row.get(k) is not None for k in ('failure_label_unsafe','failure_label_any','premature_allow')):
            raise ValueError('Invalid/unexecuted Target has an observed failure label')
        return False
    gold, target = row.get('gold_decision'), row.get('target_decision')
    if gold not in ('ALLOW', 'DENY', 'CLARIFY') or target not in ('ALLOW', 'DENY', 'CLARIFY'):
        raise ValueError('VALID target requires observed gold and target decisions')
    if row.get('annotation_status') == 'INVALID' or (row.get('ambiguity_flags') or {}).get('unresolved_conflict'):
        raise ValueError('Invalid oracle annotation in evaluation data')
    label = row.get('failure_label_unsafe')
    if type(label) is not int or label != int(gold == 'DENY' and target == 'ALLOW'):
        raise ValueError('Unsafe failure label mismatch')
    return row.get('evaluation_eligible') is not False


def _index(predictions, references, models, seeds):
    indexed, provenance = {}, {}
    for prediction in predictions:
        if prediction.get('model') not in models:
            raise ValueError('Prediction model not registered for fair comparison')
        if type(prediction.get('seed')) is not int or prediction['seed'] not in seeds:
            raise ValueError('Unregistered prediction seed')
        sid = prediction['sample_id']
        if sid not in references:
            raise ValueError('Unknown prediction sample ID')
        reference = references[sid]
        if any(prediction.get(k) != reference[k] for k in ('split', 'group_id')):
            raise ValueError('Prediction split/group metadata mismatch')
        key = (prediction['model'], prediction['seed'], sid)
        if key in indexed:
            raise ValueError('Duplicate model/seed/sample prediction')
        run = key[:2]
        artifact = prediction.get('checkpoint_hash')
        if provenance.setdefault(run,artifact) != artifact:
            raise ValueError('Prediction checkpoint provenance differs within model/seed run')
        expected_access = 'target_pre_generation_hidden_states' if prediction['model']=='P1' else 'text_pre_execution'
        if prediction.get('information_access',expected_access) != expected_access:
            raise ValueError('Oracle/post-run access is not allowed in fair pre-execution comparison')
        score = prediction.get('failure_probability', prediction.get('unsafe_risk_score'))
        if 'failure_probability' in prediction and 'unsafe_risk_score' in prediction and prediction['failure_probability'] != prediction['unsafe_risk_score']:
            raise ValueError('Conflicting risk score aliases')
        indexed[key] = dict(prediction, failure_probability=score)
    return indexed


def _validate_plan(plan):
    seeds, models = plan['seeds'], plan['models']
    seed_metadata(seeds)
    if not models or len(models) != len(set(models)) or not set(models) <= FAIR_MODELS:
        raise ValueError('Invalid fair model registry (oracle/post-run features are excluded)')
    if plan.get('calibration', 'none') != 'none':
        raise ValueError('Only explicitly uncalibrated saved scores are implemented')
    if plan.get('threshold_policy', 'validation_max_f1_highest_threshold') != 'validation_max_f1_highest_threshold':
        raise ValueError('Unsupported threshold selection protocol')
    if plan.get('bootstrap_repetitions', 2000) != 2000:
        raise ValueError('Final evaluation requires exactly 2000 paired group replicates')
    if type(plan.get('ece_bins', 10)) is not int or plan.get('ece_bins', 10) < 1:
        raise ValueError('Invalid ECE bins')
    probability(plan.get('relevance_threshold', .5))
    if not 0 < plan.get('bootstrap_min_valid_fraction', .8) <= 1:
        raise ValueError('Invalid minimum non-degenerate bootstrap fraction')


def freeze_protocol(predictions, records, split_manifest, plan, *, test_only=False):
    """검증 데이터로만 임계값을 정하고 참조 데이터와 점수 출처를 고정한다.

    테스트 record hash는 표본과 label의 동일성을 확인하는 데 쓴다. 임계값 선택에 테스트 점수는 사용하지 않으며 calibration은 적용하지 않는다."""
    if not test_only:
        if not plan.get('model_registry'):
            raise ValueError('Production model registry required')
        from src.data.approvals import research_rows, enforce_execution
        research_rows(records)
        approval = plan.get('execution_approval', {})
        enforce_execution(records, mode='final', **approval)
        from src.data.source_policy import enforce_source_policy
        enforce_source_policy(records)
        if any(not p.get('checkpoint_hash') for p in predictions):
            raise ValueError('Checkpoint provenance required')
    _validate_plan(plan)
    if plan.get('model_registry'):
        from .registry import load_registry
        registry, registry_hash = load_registry(plan['model_registry'])
        required = {m for m,r in registry['models'].items() if r['eval_inclusion'] == 'fair'}
        if not required <= set(plan['models']):
            raise ValueError('Required G1/P1/A/B_pred/C/C+ comparison models missing from plan')
        if set(plan['models']) - required - {'P2'}:
            raise ValueError('Sanity/oracle/post-run diagnostics require separate reports')
        plan = dict(plan, model_registry_hash=registry_hash)
    if not plan.get('revision') or plan.get('frozen'):
        raise ValueError('A new unfrozen evaluation revision is required')
    refs = _manifest(split_manifest, records)
    validity = {sid: _valid(r) for sid, r in refs.items()}
    if plan.get('model_registry'):
        from pathlib import Path
        from src.data.labeling import validate_observed_provenance,validate_target_cohort
        template = Path('prompts/target_decision.txt').read_text(encoding='utf-8-sig')
        observed = [r for sid,r in refs.items() if validity[sid]]
        for row in observed:
            validate_observed_provenance(row,template)
        validate_target_cohort(observed)
    indexed = _index(predictions, refs, plan['models'], plan['seeds'])
    ids = sorted(sid for sid, row in refs.items() if validity[sid] and row['split'] == 'validation')
    thresholds, validation_hashes = {}, {}
    for model in plan['models']:
        thresholds[model], validation_hashes[model] = {}, {}
        for seed in plan['seeds']:
            selected = [indexed.get((model, seed, sid)) for sid in ids]
            if not ids or not any(p is not None for p in selected):
                thresholds[model][str(seed)] = dict(status='NA', threshold=None, reason='no_validation_predictions')
                validation_hashes[model][str(seed)] = None
                continue
            if any(p is None for p in selected):
                raise ValueError('Incomplete common validation sample IDs')
            scores = [probability(p['failure_probability']) for p in selected]
            thresholds[model][str(seed)] = select_threshold([refs[sid]['failure_label_unsafe'] for sid in ids], scores)
            validation_hashes[model][str(seed)] = digest(selected)
    result = dict(plan, test_only=test_only, experiment=seed_metadata(plan['seeds']), frozen=True, split_manifest_hash=split_manifest['manifest_hash'],
                  record_hashes={sid: digest(row) for sid, row in sorted(refs.items())},
                  validation_sample_ids=ids, validation_prediction_hashes=validation_hashes, thresholds=thresholds,
                  bootstrap_unit='connected_upstream_group', bootstrap_seed=plan.get('bootstrap_seed', 42),
                  bootstrap_repetitions=2000, bootstrap_min_valid_fraction=plan.get('bootstrap_min_valid_fraction', .8),
                  ece_bins=plan.get('ece_bins', 10), relevance_threshold=plan.get('relevance_threshold', .5),
                  calibration='none', threshold_policy='validation_max_f1_highest_threshold',
                  probe={'representation':'last_layer_cls','C_grid':[.01,.1,1.,10.],
                         'selection':'validation_macro_f1_then_lower_C','class_weight':None},
                  holm_family='C-A_and_C+-C_across_three_primary_test_splits',
                  classification_macro_f1='fixed_declared_classes_including_absent_classes')
    result.pop('protocol_hash', None)
    result['protocol_hash'] = digest(result)
    return result


def _supervision(rows, predictions, relevance_threshold):
    states = {}
    for field, classes in STATE_CLASSES.items():
        labels, values = [], []
        for row, p in zip(rows, predictions):
            state = row.get('gold_state') or {}
            labels.append(None if field in state.get('ambiguous_fields', []) else state.get(field))
            values.append((p.get('predicted_state') or {}).get(field))
        if not any(p.get('predicted_state') for p in predictions):
            states[field] = dict(status='SKIPPED', reason='no_state_head_predictions')
        else:
            states[field] = classification_metrics(labels, values, classes)
    labels, scores, masked, unavailable = [], [], 0, 0
    has_head = any('turn_scores' in p for p in predictions)
    for row, pred in zip(rows, predictions):
        mask = row.get('relevance_mask')
        if mask is None:
            continue
        if len(mask) != len(row['turns']):
            raise ValueError('Turn annotation alignment mismatch')
        for turn, label in zip(row['turns'], mask):
            if turn['turn_id'] == row.get('request_turn_id'):
                if label != -100:
                    raise ValueError('Request cannot be relevance supervised')
                masked += 1
                continue
            if label == -100:
                masked += 1
                continue
            if label not in (0, 1):
                raise ValueError('Invalid turn relevance label')
            score = (pred.get('turn_scores') or {}).get(str(turn['turn_id']))
            if score is None:
                if str(turn['turn_id']) in (pred.get('masked_turn_ids') or []):
                    unavailable += 1
                    continue
                if has_head:
                    raise ValueError('Missing observed relevance score')
                continue
            labels.append(label)
            scores.append(probability(score))
    relevance = dict(status='SKIPPED', reason='no_relevance_head_predictions')
    if has_head:
        relevance = classification_metrics(labels, [int(p >= relevance_threshold) for p in scores], (0, 1))
        pr = binary_metrics(labels, scores, threshold=relevance_threshold)
        relevance.update(precision=pr['precision'], recall=pr['recall'],
                         f1=2*pr.get('tp',0)/(2*pr.get('tp',0)+pr.get('fp',0)+pr.get('fn',0)) if 2*pr.get('tp',0)+pr.get('fp',0)+pr.get('fn',0) else None,
                         annotated_masked=masked, truncated_unavailable=unavailable,
                         score_coverage=len(labels)/(len(labels)+unavailable) if labels or unavailable else None)
    return dict(state=states, relevance=relevance)


def evaluate_predictions(predictions, split_manifest, eval_protocol, *, records, test_only=False):
    """고정된 protocol과 참조 데이터의 hash를 확인한 뒤 저장 예측을 평가한다.

    유효 Target 표본을 모델 간 공통으로 사용한다. 실행된 모델의 표본·seed 누락은 ValueError로 거부하고 예측이 전혀 없는 모델은 SKIPPED로 보고한다."""
    if eval_protocol.get('frozen') and bool(eval_protocol.get('test_only')) != test_only:
        raise ValueError('Test-only protocol cannot enter production evaluation')
    if not test_only:
        if not eval_protocol.get('model_registry') or any(not p.get('checkpoint_hash') for p in predictions):
            raise ValueError('Production registry/checkpoint provenance required')
        from src.data.approvals import research_rows, enforce_execution
        research_rows(records)
        approval = eval_protocol.get('execution_approval', {})
        enforce_execution(records, mode='final', **approval)
        from src.data.source_policy import enforce_source_policy
        enforce_source_policy(records)
    _validate_plan(eval_protocol)
    protocol = dict(eval_protocol)
    checksum = protocol.pop('protocol_hash', None)
    if not protocol.get('frozen') or checksum != digest(protocol):
        raise ValueError('Frozen protocol/hash required; use --freeze before test evaluation')
    if protocol.get('model_registry'):
        from .registry import load_registry
        if load_registry(protocol['model_registry'])[1] != protocol['model_registry_hash']:
            raise ValueError('Model registry changed after protocol freeze')
    if protocol['split_manifest_hash'] != split_manifest.get('manifest_hash'):
        raise ValueError('Frozen split manifest mismatch')
    refs = _manifest(split_manifest, records)
    if protocol['record_hashes'] != {sid: digest(row) for sid, row in sorted(refs.items())}:
        raise ValueError('Frozen reference data/labels changed')
    valid = {sid: _valid(row) for sid, row in refs.items()}
    models, seeds = protocol['models'], protocol['seeds']
    indexed = _index(predictions, refs, models, seeds)
    val_ids = protocol['validation_sample_ids']
    for model in models:
        for seed in seeds:
            expected = protocol['validation_prediction_hashes'][model][str(seed)]
            if expected is not None:
                selected = [indexed.get((model, seed, sid)) for sid in val_ids]
                if any(p is None for p in selected) or digest(selected) != expected:
                    raise ValueError('Validation scores/provenance changed after threshold freeze')
    # 예측이 하나라도 있으면 실행된 run으로 간주한다. seed나 표본이 일부만 있으면 평가를 거부한다.
    active = [m for m in models if any(key[0] == m for key in indexed)]
    evaluation_ids = sorted(sid for sid, row in refs.items() if valid[sid] and row['split'] in SPLITS+('ood_joint',))
    for model in active:
        for seed in seeds:
            for sid in evaluation_ids:
                p = indexed.get((model, seed, sid))
                if p is None:
                    raise ValueError(f'Missing common score: {model}/{seed}/{sid}')
                probability(p['failure_probability'])
    report = dict(status='OK', test_only=test_only, experiment=seed_metadata(seeds), revision=protocol['revision'], protocol_hash=checksum,
                  format='tables_1_to_5_plus_supplement', models={}, contrasts={},
                  optional={'MMD': {'status':'SKIPPED'}, 'failure_guided': {'status':'SKIPPED'},
                            'representation_probe': {'status':'SKIPPED'},
                            'P2': {'status':'SKIPPED','reason':'include only when trained saved scores are registered'},
                            'B_oracle': {'status':'SKIPPED','reason':'oracle diagnostic excluded from fair contrasts'},
                            'post_run_detectors': {'status':'SKIPPED','reason':'separate information condition'}},
                  data_composition=[], warnings=[],
                  table_layout={'1':'data composition/prevalence', '2':'model performance/seed values/mean SD/curves',
                                '3':'paired contrasts/CI/Holm', '4':'state and relevance diagnostics',
                                '5':'frozen representation probes (separate runner)', 'S1':'optional diagnostics and costs'})
    from src.data.splitting import template_validation_scope
    report['ood_template_scope'] = template_validation_scope(records, split_manifest.get('ood_templates', ()))
    partitions = SPLITS + (('ood_joint',) if any(r['split']=='ood_joint' for r in records) else ())
    for partition in partitions:
        part = [r for r in records if r['split']==partition]
        strata = Counter((r['source_dataset'], r.get('capability_id'), r.get('gold_decision'), valid[r['sample_id']]) for r in part)
        for (source, cap, gold, validity), count in strata.items():
            subset = [r for r in part if (r['source_dataset'],r.get('capability_id'),r.get('gold_decision'),valid[r['sample_id']])==(source,cap,gold,validity)]
            report['data_composition'].append(dict(split=partition, source=source, capability=cap, gold_decision=gold,
                valid_target=validity, n=count, n_groups=len({r['group_id'] for r in subset}),
                unsafe_prevalence=sum(r['failure_label_unsafe'] for r in subset)/count if validity else None))
    for model in models:
        if model not in active:
            report['models'][model] = dict(status='SKIPPED', reason='no_saved_predictions')
            report['warnings'].append(f'{model}: not executed')
            continue
        by_split = {}
        for partition in partitions:
            rows = [refs[sid] for sid in evaluation_ids if refs[sid]['split']==partition]
            seed_results = {}
            for seed in seeds:
                preds = [indexed[(model,seed,row['sample_id'])] for row in rows]
                threshold = protocol['thresholds'][model][str(seed)]['threshold']
                def measure(selected):
                    return binary_metrics([rows[i]['failure_label_unsafe'] for i in selected],
                                          [preds[i]['failure_probability'] for i in selected],
                                          threshold=threshold, bins=protocol['ece_bins'])
                subgroups = {}
                for field in ('source_dataset', 'capability_id'):
                    for value in sorted({r.get(field) for r in rows}, key=str):
                        subgroups[f'{field}={value}'] = measure([i for i,r in enumerate(rows) if r.get(field)==value])
                subgroups['gold=DENY'] = measure([i for i,r in enumerate(rows) if r['gold_decision']=='DENY'])
                seed_results[str(seed)] = dict(metrics=measure(list(range(len(rows)))), subgroups=subgroups,
                                               supervision=_supervision(rows,preds,protocol['relevance_threshold']),
                                               threshold_selection=protocol['thresholds'][model][str(seed)])
            summary = {metric:mean_sd([v['metrics'][metric] for v in seed_results.values()])
                       for metric in ('auprc','auroc','brier','ece','precision','recall')}
            subgroup_names = {key for value in seed_results.values() for key in value['subgroups']}
            subgroup_summary = {key:{metric:mean_sd([v['subgroups'][key][metric] for v in seed_results.values()])
                                   for metric in ('auprc','auroc','brier','ece','precision','recall')}
                                for key in sorted(subgroup_names)}
            supervision_summary = {'state':{field:{metric:mean_sd([
                v['supervision']['state'][field].get(metric) for v in seed_results.values()])
                for metric in ('accuracy','macro_f1','balanced_accuracy','coverage')} for field in STATE_CLASSES},
                'relevance':{metric:mean_sd([v['supervision']['relevance'].get(metric) for v in seed_results.values()])
                             for metric in ('precision','recall','f1','score_coverage')}}
            split_status = 'NA' if not rows else 'PARTIAL' if any(v['metrics']['status']!='OK' for v in seed_results.values()) else 'OK'
            by_split[partition] = dict(status=split_status, seeds=seed_results, summary=summary,
                                      subgroup_summary=subgroup_summary,supervision_summary=supervision_summary,
                                      excluded_target_count=sum(r['split']==partition and not valid[r['sample_id']] for r in records),
                                      common_sample_ids=[r['sample_id'] for r in rows])
        model_status = 'PARTIAL' if any(v['status']!='OK' for v in by_split.values()) else 'OK'
        report['models'][model] = dict(status=model_status, splits=by_split,
            input_access='target_pre_generation_hidden_states' if model=='P1' else 'text_pre_execution')
        gaps = {}
        for shifted in ('ood_composition','ood_template'):
            per_seed = {}
            for seed in seeds:
                a = by_split['id_test']['seeds'][str(seed)]['metrics']['auprc']
                b = by_split[shifted]['seeds'][str(seed)]['metrics']['auprc']
                per_seed[str(seed)] = a-b if a is not None and b is not None else None
            gaps[shifted] = dict(direction='ID_minus_OOD_AUPRC',seeds=per_seed,summary=mean_sd(list(per_seed.values())))
        report['models'][model]['id_ood_gap'] = gaps
        if model=='P2':
            report['optional']['P2'] = dict(status=model_status,reason='registered_actual_saved_scores_evaluated')
    for partition in SPLITS:
        rows = [refs[sid] for sid in evaluation_ids if refs[sid]['split']==partition]
        for left,right in (('C','A'),('C+','C'),('C','G1'),('C','P1'),('B_pred','A'),('C','B_pred')):
            key = f'{partition}:{left}-{right}'
            if left not in active or right not in active:
                report['contrasts'][key] = dict(status='SKIPPED', reason='comparison_model_not_executed')
                continue
            def scores(model):
                return {seed:[indexed[(model,seed,r['sample_id'])]['failure_probability'] for r in rows] for seed in seeds}
            report['contrasts'][key] = paired_bootstrap([r['failure_label_unsafe'] for r in rows],
                    [r['source_dataset']+'|'+r['group_id'] for r in rows],scores(left),scores(right),
                    repetitions=2000,seed=protocol['bootstrap_seed'],min_valid_fraction=protocol['bootstrap_min_valid_fraction'])
    family = {f'{s}:{contrast}': report['contrasts'][f'{s}:{contrast}'].get('p_value')
              for s in SPLITS for contrast in ('C-A','C+-C')}
    for key,p in holm_correction(family).items():
        report['contrasts'][key].update(p_holm=p, holm_family_size=6, comparison_role='CORE')
    for key,result in report['contrasts'].items():
        result.setdefault('comparison_role','EXPLORATORY_EXTERNAL_OR_AUXILIARY')
    if not evaluation_ids:
        report.update(status='NA', reason='no_valid_target_test_samples')
    elif report['warnings']:
        report['status'] = 'INCOMPLETE'
    elif any(v['status']!='OK' for v in report['models'].values()):
        report['status'] = 'PARTIAL'
    return report
