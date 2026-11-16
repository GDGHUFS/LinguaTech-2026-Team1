"""[B] Target 응답과 failure label을 저장하고 train, validation, ID/OOD로 나눈다."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.labeling import LocalTarget, failure_labels, run_target, target_input
from src.data.splitting import DEFAULT_COMPOSITIONS, DEFAULT_TEMPLATES, split_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/data.yaml')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--revision', help='Immutable Qwen model commit; required for actual inference')
    parser.add_argument('--execute', action='store_true', help='Load pinned local Qwen weights and execute')
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--full-run', action='store_true', help='Label every eligible sample instead of the 100-200 pilot subset')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ood-composition', action='append', help='Comma-separated ordered operations; repeatable')
    parser.add_argument('--ood-template', action='append', help='Exact held-out template family; repeatable')
    parser.add_argument('--labeled-output', type=Path)
    parser.add_argument('--splits-dir', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--execution-mode', choices=('pilot','final'), default='pilot')
    parser.add_argument('--qa-manifest')
    parser.add_argument('--freeze-manifest')
    args = parser.parse_args()
    import yaml
    from src.data.source_policy import selected_source
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    args.input = args.input or ROOT/config['paths']['generated']/'generated.jsonl'
    args.labeled_output = args.labeled_output or ROOT/config['paths']['labeled']/'target_results.jsonl'
    args.splits_dir = args.splits_dir or ROOT/config['paths']['splits']
    args.report = args.report or ROOT/Path(config['seed_setup']['inventory']).parent/'target_pilot.csv'
    if args.seed != 42:
        parser.error("This experiment registers only random seed 42")
    if args.labeled_output.exists() or args.report.exists() or any(args.splits_dir.glob('*.json*')):
        parser.error('Existing labels/splits/reports are preserved; choose new output paths')
    if not args.full_run and not 100 <= args.limit <= 200:
        parser.error('Pilot limit must be 100-200')
    if args.execute and not args.revision:
        parser.error('--execute requires --revision')
    rows = [json.loads(l) for l in args.input.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
    if not rows:
        parser.error('Empty input; no output written')
    if any(not selected_source(r, config) for r in rows):
        parser.error('Input contains a disabled source/component')
    from src.data.source_policy import enforce_source_policy
    from src.data.approvals import enforce_execution
    enforce_execution(rows, mode=args.execution_mode, qa_path=args.qa_manifest, freeze_path=args.freeze_manifest, data_config=args.config, config_paths=[args.config])
    if any(r.get('data_kind') == 'mock' or r.get('source_provenance',{}).get('fixture') for r in rows):
        parser.error('Mock fixture cannot be saved as real pilot data')
    compositions = tuple(tuple(s.strip() for s in c.split(',')) for c in args.ood_composition) if args.ood_composition else tuple(tuple(c) for c in config['split']['ood_compositions'])
    templates = tuple(args.ood_template) if args.ood_template else tuple(config['split']['ood_template_families'])
    from src.data.fewshots import load_fewshots, check_fewshot_overlap
    shots = load_fewshots(ROOT/'prompts/naturalization_shots.json', templates)
    check_fewshot_overlap(rows, shots['shots'], templates)
    fewshot_templates = tuple(s['template_family_id'] for s in shots['shots'])
    # Target 호출 전에 전체 데이터의 group·holdout 누수를 검사한다. 실제 분할 결과는 응답 수집 후 저장한다.
    split_samples(rows, seed=args.seed, compositions=compositions, templates=templates, fewshot_templates=fewshot_templates, ratios=tuple(config['split']['id_ratios'][k] for k in ('train','validation','id_test')))
    eligible = []
    for row in rows:
        try:
            if row.get('validation_status') != 'semantic_pass_oracle_verified' or row.get('oracle_verified') is not True or row.get('oracle_relation_status') != 'passed':
                continue
            from src.data.contracts import validate_pipeline_record
            validate_pipeline_record(row,stage='gold')
            target_input(row)
            eligible.append(row['sample_id'])
        except ValueError:
            pass
    selected = set(eligible if args.full_run else sorted(eligible)[:args.limit])
    if args.execute and not selected:
        parser.error('No natural-language dialogue available; run generation first')
    runner = LocalTarget(args.revision) if args.execute else None
    template = (ROOT/'prompts/target_decision.txt').read_text(encoding='utf-8-sig')
    results = []
    output = args.labeled_output
    output.parent.mkdir(parents=True, exist_ok=True)
    # 다음 호출이 실패해도 받은 응답은 남도록 호출할 때마다 저장한다.
    with output.open('w', encoding='utf-8') as stream:
        for row in rows:
            if runner and row['sample_id'] in selected:
                result = run_target(row, runner, template)
                validate_pipeline_record(result,stage='target')
            else:
                result = dict(row, target_attempts=[], target_decision=None, parse_status='NOT_RUN',
                    target_error=None, latency_seconds=None, research_dataset_status='unlabeled_target_only',
                    pending_reason='dialogue_missing_or_unvalidated' if row['sample_id'] not in eligible else 'inference_not_requested' if not runner else 'outside_pilot_limit')
                result.update(failure_labels(row.get('gold_decision'), None))
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False)+'\n')
            stream.flush()
    split_rows, manifest = split_samples(results, seed=args.seed, compositions=compositions, templates=templates, fewshot_templates=fewshot_templates, ratios=tuple(config['split']['id_ratios'][k] for k in ('train','validation','id_test')))
    manifest.update(natural_dialogue_count=len(eligible), target_attempted=sum(bool(r['target_attempts']) for r in results),
        source_selection=config.get('source_selection'), source_policy_revision=config.get('source_policy_revision'))
    # 실행 정보가 추가됐으므로 manifest hash를 다시 계산한다.
    from src.data.labeling import sha
    manifest.pop('manifest_hash')
    manifest['manifest_hash'] = sha(json.dumps(manifest, sort_keys=True))
    folder = args.splits_dir
    folder.mkdir(parents=True, exist_ok=True)
    for partition in ('train','validation','id_test','ood_composition','ood_template','ood_joint'):
        subset = [dict(r, split_manifest_hash=manifest['manifest_hash']) for r in split_rows if r['split']==partition]
        (folder/f'{partition}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in subset), encoding='utf-8')
    (folder/'split_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    report = args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    attempted = [r for r in results if r['target_attempts']]
    valid = [r for r in attempted if r['parse_status']=='VALID']
    labeled = [r for r in valid if r.get('gold_decision') is not None]
    fields = ['row_type','sample_id','parse_status','attempt_count','latency_seconds','oom','target_error',
              'gold_available','failure_label_unsafe','failure_label_any','premature_allow','N_total','N_attempted',
              'N_valid','json_valid_rate','first_attempt_json_valid_rate','N_invalid_output','N_execution_error',
              'N_oom','mean_latency_seconds','failure_prevalence','gold_target_confusion_status','pilot_status']
    with report.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({'row_type':'sample','sample_id':r['sample_id'],'parse_status':r['parse_status'],
                'attempt_count':len(r['target_attempts']),'latency_seconds':r['latency_seconds'],
                'oom':any(a.get('oom',False) for a in r['target_attempts']) if r['target_attempts'] else None, 'target_error':r['target_error'],
                'gold_available':r.get('gold_decision') is not None,
                **{k:r[k] for k in ('failure_label_unsafe','failure_label_any','premature_allow')}})
        writer.writerow({'row_type':'summary','N_total':len(results),'N_attempted':len(attempted),'N_valid':len(valid),
            'json_valid_rate':len(valid)/len(attempted) if attempted else None,
            'first_attempt_json_valid_rate':sum(r['target_attempts'][0]['parse_status']=='VALID' for r in attempted)/len(attempted) if attempted else None,
            'N_invalid_output':sum(r['parse_status']=='INVALID_OUTPUT' for r in attempted),
            'N_execution_error':sum(r['parse_status']=='EXECUTION_ERROR' for r in attempted),
            'N_oom':sum(any(a.get('oom',False) for a in r['target_attempts']) for r in attempted) if attempted else None,
            'mean_latency_seconds':sum(r['latency_seconds'] for r in attempted)/len(attempted) if attempted else None,
            'failure_prevalence':sum(r['failure_label_unsafe'] for r in labeled)/len(labeled) if labeled else None,
            'gold_target_confusion_status':'not_measured_no_valid_gold_target_pairs' if not labeled else 'available_in_target_records',
            'pilot_status':'executed' if len(attempted)>=100 else 'incomplete'})
    print(json.dumps({'N_total':len(results),'N_attempted':len(attempted),'N_valid':len(valid),
                      'split_counts':dict(Counter(r['split'] for r in split_rows)), 'research_dataset_finalized':False}))


if __name__ == '__main__':
    main()
