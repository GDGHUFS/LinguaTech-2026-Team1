"""[B] structured mutation을 만들고 Qwen에 대화 생성과 검증을 요청한다."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.mutations import make_variants
from src.data.generation import QwenClient, execute

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/annotated/canonical_seed.jsonl')
    parser.add_argument('--endpoint')
    parser.add_argument('--model')
    parser.add_argument('--key-env', default='QWEN_API_KEY')
    parser.add_argument('--limit-seeds', type=int, default=20)
    args = parser.parse_args()
    if bool(args.endpoint) != bool(args.model):
        parser.error('--endpoint and --model must be provided together')
    if args.limit_seeds < 1:
        parser.error('--limit-seeds must be positive')
    seeds = [json.loads(line) for line in args.input.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    selected = []
    for seed in seeds:
        variants = make_variants(seed)
        if variants:
            selected.append(variants)
    random.Random(42).shuffle(selected)
    selected = selected[:args.limit_seeds]
    if not selected:
        parser.error('No structurally eligible seeds; no output written')
    client = QwenClient(args.endpoint, args.model, args.key_env) if args.endpoint else None
    records = [execute(v, client, ROOT / 'prompts') for variants in selected for v in variants]
    # 같은 seed에서 나온 base와 variant는 생성에 실패한 항목도 함께 남긴다.
    groups = defaultdict(list)
    for record in records:
        groups[record['variant_group_id']].append(record)
    for family in groups.values():
        complete = all(r['validation_status'] == 'semantic_pass_oracle_pending' for r in family)
        for record in family:
            record['controlled_family_status'] = 'semantic_complete_oracle_pending' if complete else 'incomplete'
    review_ids = set()
    for cap in [None] + [f'CAP{i:02}' for i in range(1,10)]:
        pool = [r['sample_id'] for r in records if r['capability_id'] == cap]
        random.Random(42).shuffle(pool)
        review_ids.update(pool[:min(2,len(pool))])
    output = ROOT / 'data/generated/generated.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    for record in records:
        record['manual_review_selected'] = record['sample_id'] in review_ids
    output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    audit = ROOT / 'outputs/reports/pilot/generation_audit.csv'
    audit.parent.mkdir(parents=True, exist_ok=True)
    fields = ['row_type','sample_id','capability_id','group_id','variant_group_id','generation_status','validation_status','errors','manual_review_selected','oracle_relation_status','N_structured','N_attempted','N_generated','N_validated','generation_success_rate','validation_yield','controlled_family_status']
    with audit.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({**{k:r[k] for k in fields if k in r},'row_type':'sample','errors':json.dumps(r['errors'])})
        for cap in [None] + [f'CAP{i:02}' for i in range(1,10)]:
            subset = [r for r in records if r['capability_id'] == cap]
            attempted = sum(r['generation_status'] != 'pending_qwen' for r in subset)
            generated = sum(r['generation_status'] == 'generated' for r in subset)
            valid = sum(r['validation_status'] == 'semantic_pass_oracle_pending' for r in subset)
            writer.writerow({'row_type':'summary','capability_id':cap or 'BASE','N_structured':len(subset),'N_attempted':attempted,'N_generated':generated,'N_validated':valid,'generation_success_rate':generated/attempted if attempted else None,'validation_yield':valid/attempted if attempted else None,'oracle_relation_status':'pending'})
    print(json.dumps({'eligible_seed_pool':len([s for s in seeds if s.get('status')=='eligible_pending_review']), 'selected_seeds':len(selected),'structured_records':len(records),'qwen_executed':client is not None,'oracle_relation_status':'pending'}))

if __name__ == '__main__':
    main()
