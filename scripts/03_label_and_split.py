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
    parser.add_argument('--input', type=Path, default=ROOT/'data/generated/generated.jsonl')
    parser.add_argument('--revision', help='Immutable Qwen model commit; required for actual inference')
    parser.add_argument('--execute', action='store_true', help='Load pinned local Qwen weights and execute')
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ood-composition', action='append', help='Comma-separated ordered operations; repeatable')
    parser.add_argument('--ood-template', action='append', help='Exact held-out template family; repeatable')
    args = parser.parse_args()
    if not 100 <= args.limit <= 200:
        parser.error('Pilot limit must be 100-200')
    if args.execute and not args.revision:
        parser.error('--execute requires --revision')
    rows = [json.loads(l) for l in args.input.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
    if not rows:
        parser.error('Empty input; no output written')
    if any(r.get('data_kind') == 'mock' or r.get('source_provenance',{}).get('fixture') for r in rows):
        parser.error('Mock fixture cannot be saved as real pilot data')
    compositions = tuple(tuple(s.strip() for s in c.split(',')) for c in args.ood_composition) if args.ood_composition else DEFAULT_COMPOSITIONS
    templates = tuple(args.ood_template) if args.ood_template else DEFAULT_TEMPLATES
    # pilot 대상 수를 제한해도 같은 family가 나뉘지 않도록 전체 데이터의 split을 먼저 확인한다.
    split_samples(rows, seed=args.seed, compositions=compositions, templates=templates)
    eligible = []
    for row in rows:
        try:
            if row.get('validation_status') not in (None, 'semantic_pass_oracle_pending'):
                continue
            target_input(row)
            eligible.append(row['sample_id'])
        except ValueError:
            pass
    selected = set(sorted(eligible)[:args.limit])
    if args.execute and not selected:
        parser.error('No natural-language dialogue available; run generation first')
    runner = LocalTarget(args.revision) if args.execute else None
    template = (ROOT/'prompts/target_decision.txt').read_text(encoding='utf-8-sig')
    results = []
    output = ROOT/'data/labeled/target_results.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    # 다음 호출이 실패해도 받은 응답은 남도록 호출할 때마다 저장한다.
    with output.open('w', encoding='utf-8') as stream:
        for row in rows:
            if runner and row['sample_id'] in selected:
                result = run_target(row, runner, template)
            else:
                result = dict(row, target_attempts=[], target_decision=None, parse_status='NOT_RUN',
                    target_error=None, latency_seconds=None, research_dataset_status='unlabeled_target_only',
                    pending_reason='dialogue_missing_or_unvalidated' if row['sample_id'] not in eligible else 'inference_not_requested' if not runner else 'outside_pilot_limit')
                result.update(failure_labels(row.get('gold_decision'), None))
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False)+'\n')
            stream.flush()
    split_rows, manifest = split_samples(results, seed=args.seed, compositions=compositions, templates=templates)
    manifest.update(natural_dialogue_count=len(eligible), target_attempted=sum(bool(r['target_attempts']) for r in results))
    # 실행 정보가 추가됐으므로 manifest hash를 다시 계산한다.
    from src.data.labeling import sha
    manifest.pop('manifest_hash')
    manifest['manifest_hash'] = sha(json.dumps(manifest, sort_keys=True))
    folder = ROOT/'data/splits'
    folder.mkdir(parents=True, exist_ok=True)
    for partition in ('train','validation','id_test','ood_composition','ood_template','ood_joint'):
        subset = [dict(r, split_manifest_hash=manifest['manifest_hash']) for r in split_rows if r['split']==partition]
        (folder/f'{partition}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in subset), encoding='utf-8')
    (folder/'split_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    report = ROOT/'outputs/reports/pilot/target_pilot.csv'
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
