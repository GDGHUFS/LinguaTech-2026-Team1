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
from src.data.mutations import make_variants, apply_holdout_plan
from src.data.generation import QwenClient, execute, validate_controlled_family

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/data.yaml')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--reviews', type=Path, help='Actual evidence-bound per-record human reviews (JSONL)')
    parser.add_argument('--annotations', type=Path, help='Approved derived annotation revisions (JSONL); still require exact source reviews')
    parser.add_argument('--endpoint')
    parser.add_argument('--model')
    parser.add_argument('--key-env', default='QWEN_API_KEY')
    parser.add_argument('--limit-seeds', type=int, default=20)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--execution-mode',choices=('pilot','final'),default='pilot')
    parser.add_argument('--holdout-plan', type=Path, help='Predeclared JSON group_id -> id/ood_composition/ood_template')
    args = parser.parse_args()
    import yaml
    from src.data.source_policy import selected_source, generation_blockers, apply_review
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    args.output = args.output or ROOT/config['paths']['generated']/'generated.jsonl'
    args.audit = args.audit or ROOT/Path(config['seed_setup']['inventory']).parent/'generation_audit.csv'
    args.input = args.input or ROOT/config['paths']['annotated']/config.get('canonical_filename', 'canonical_seed.jsonl')
    if args.output.exists() or args.audit.exists():
        parser.error('Existing generation/audit artifacts are preserved; provide new --output and --audit paths')
    if bool(args.endpoint) != bool(args.model):
        parser.error('--endpoint and --model must be provided together')
    if args.limit_seeds < 1:
        parser.error('--limit-seeds must be positive')
    seeds = [json.loads(line) for line in args.input.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if any(not selected_source(s, config) for s in seeds):
        parser.error('Input contains a disabled source/component; select it explicitly in config')
    args.annotations = args.annotations or (Path(config['approvals']['annotation_revisions']) if config.get('approvals',{}).get('annotation_revisions') else None)
    if args.annotations:
        from src.data.annotations import apply_revisions
        annotations = [json.loads(l) for l in args.annotations.read_text(encoding='utf-8').splitlines() if l.strip()]
        seeds = apply_revisions(seeds, annotations)
    args.reviews = args.reviews or (Path(config['approvals']['source_reviews']) if config.get('approvals',{}).get('source_reviews') else None)
    reviews = [json.loads(l) for l in args.reviews.read_text(encoding='utf-8').splitlines() if l.strip()] if args.reviews else []
    by_id = {r['seed_id']: r for r in reviews}
    if len(by_id) != len(reviews) or set(by_id) - {s['seed_id'] for s in seeds}:
        parser.error('Duplicate or unknown review seed IDs')
    seeds = [apply_review(s, by_id[s['seed_id']]) if s['seed_id'] in by_id else s for s in seeds]
    holdout_plan = json.loads(args.holdout_plan.read_text(encoding='utf-8')) if args.holdout_plan else {}
    if args.execution_mode == 'final' and (not holdout_plan or set(holdout_plan) != {s['group_id'] for s in seeds}):
        parser.error('Final generation requires complete predeclared group holdout plan')
    selected = []
    for seed in seeds:
        if generation_blockers(seed):
            continue
        from src.data.source_policy import enforce_source_policy
        enforce_source_policy([seed], args.config)
        variants = make_variants(seed, config.get('modify_subtypes', ['authorization']))
        if variants:
            variants = apply_holdout_plan(variants, holdout_plan.get(seed['group_id'],'id'))
            from src.utils.reproducibility import content_hash
            for variant in variants:
                variant['holdout_plan_hash'] = content_hash(holdout_plan)
            selected.append(variants)
    random.Random(42).shuffle(selected)
    selected = selected[:args.limit_seeds]
    if not selected:
        parser.error('No human-reviewed generation-ready seeds; no output written')
    from src.data.fewshots import load_fewshots, expression_audit, check_fewshot_overlap
    from src.data.splitting import DEFAULT_TEMPLATES
    shots = load_fewshots(ROOT/'prompts/naturalization_shots.json', DEFAULT_TEMPLATES)
    client = QwenClient(args.endpoint, args.model, args.key_env) if args.endpoint else None
    records = [execute(v, client, ROOT / 'prompts') for variants in selected for v in variants]
    # 같은 seed에서 나온 base와 variant는 생성에 실패한 항목도 함께 남긴다.
    check_fewshot_overlap(records, shots['shots'], DEFAULT_TEMPLATES)
    groups = defaultdict(list)
    for record in records:
        groups[record['variant_group_id']].append(record)
    for family in groups.values():
        validate_controlled_family(family)
        from src.data.contracts import validate_pipeline_record
        for record in family:
            if record.get('validation_status') == 'semantic_pass_oracle_verified':
                validate_pipeline_record(record,stage='gold')
        complete = all(r['validation_status'] == 'semantic_pass_oracle_verified' for r in family)
        for record in family:
            record['controlled_family_status'] = 'semantic_complete_oracle_verified' if complete else 'incomplete'
    review_ids = set()
    for cap in [None] + [f'CAP{i:02}' for i in range(1,10)]:
        pool = [r['sample_id'] for r in records if r['capability_id'] == cap]
        random.Random(42).shuffle(pool)
        review_ids.update(pool[:min(2,len(pool))])
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    for record in records:
        record['manual_review_selected'] = record['sample_id'] in review_ids
    output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    audit = args.audit
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
            valid = sum(r['validation_status'] == 'semantic_pass_oracle_verified' for r in subset)
            writer.writerow({'row_type':'summary','capability_id':cap or 'BASE','N_structured':len(subset),'N_attempted':attempted,'N_generated':generated,'N_validated':valid,'generation_success_rate':generated/attempted if attempted else None,'validation_yield':valid/attempted if attempted else None,'oracle_relation_status':'pending'})
    print(json.dumps({'eligible_seed_pool':len([s for s in seeds if s.get('status')=='eligible_pending_review']), 'selected_seeds':len(selected),'structured_records':len(records),'qwen_executed':client is not None,'experiment':{'training_seeds':[42],'single_seed_experiment':True},'expression_audit':expression_audit(records),'oracle_relation_status':'pending'}))

if __name__ == '__main__':
    main()
