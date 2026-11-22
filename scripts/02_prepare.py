"""[B] structured mutation을 만들고 Qwen에 대화 생성과 검증을 요청한다."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import io
import os
from pathlib import Path
import random
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.mutations import make_variants, apply_holdout_plan
from src.data.generation import QwenClient, execute, validate_controlled_family
from src.data.batch import read_items, warning, diagnostic
from src.data.progress import Journal, signature, atomic_text, append_json

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/data.yaml')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--reviews', type=Path, help='Actual evidence-bound per-record human reviews (JSONL)')
    parser.add_argument('--annotations', type=Path, help='Approved derived annotation revisions (JSONL); still require exact source reviews')
    parser.add_argument('--endpoint')
    parser.add_argument('--model')
    parser.add_argument('--revision', help='Immutable serving Qwen commit; checked against the running server')
    parser.add_argument('--source-audit', type=Path, help='Automatic source normalization/exclusion JSONL')
    parser.add_argument('--resume', action='store_true', help='Continue using the matching durable item journal')
    parser.add_argument('--key-env', default='QWEN_API_KEY')
    parser.add_argument('--limit-seeds', type=int, default=20)
    parser.add_argument('--limit-generate-seeds', type=int, help='Pilot only: normalize all selected sources but naturalize only this many source families')
    parser.add_argument('--limit-dialogues', type=int, help='Pilot only: cap actual naturalization calls; retain all planned variants as not-run records')
    parser.add_argument('--paraphrase-budget', type=int, help='Guarded mode: at most this many dialogues use Qwen; others remain verified templates')
    parser.add_argument('--paraphrase-scope', choices=('single_policy_turn','all_policy_turns'),
        help='Guarded wording coverage; defaults to all_policy_turns (one batched request per dialogue)')
    parser.add_argument('--accepted-output', type=Path, help='Only verified Gold-labelled records, written incrementally')
    parser.add_argument('--normalized-output', type=Path, help='Verified source records reusable without a model')
    parser.add_argument('--template-only', action='store_true', help='Guarded templates only; offline mode requires previously verified normalized sources')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--execution-mode',choices=('pilot','final'),default='pilot')
    parser.add_argument('--holdout-plan', type=Path, help='Optional JSON group plan; otherwise assigned automatically before Qwen calls')
    args = parser.parse_args()
    import yaml
    from src.data.source_policy import selected_source, generation_blockers, apply_review
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    automatic = config.get('generation_mode') == 'camber_automatic'
    guarded = config.get('dialogue_generation') == 'guarded_v1'
    if args.paraphrase_scope is not None and not guarded:
        parser.error('--paraphrase-scope requires guarded_v1')
    args.paraphrase_scope = args.paraphrase_scope or config.get('qwen_paraphrase_scope','all_policy_turns')
    if args.paraphrase_scope not in ('single_policy_turn','all_policy_turns'):
        parser.error('Unsupported qwen_paraphrase_scope')
    if (guarded and args.endpoint == 'transformers://cpu' and args.paraphrase_scope == 'all_policy_turns'
            and args.paraphrase_budget != 0 and not args.template_only):
        parser.error('CPU decoding requires --paraphrase-scope single_policy_turn; all_policy_turns requires vLLM')
    if guarded and config.get('free_generation',False):
        parser.error('Free generation is not supported by the verified closed grammar')
    if guarded and args.limit_dialogues is not None:
        parser.error('Guarded mode uses --paraphrase-budget; templates must not be skipped by --limit-dialogues')
    if args.template_only and not guarded:
        parser.error('--template-only requires guarded_v1')
    if bool(args.endpoint) != bool(args.model):
        parser.error('--endpoint and --model must be provided together')
    client = QwenClient(args.endpoint, args.model, args.key_env, revision=args.revision) if args.endpoint else None
    if automatic and client is None and not (guarded and args.template_only):
        parser.error('Qwen generation-ready scenarios require --endpoint and --model')
    if automatic and (args.reviews or args.annotations):
        parser.error('CAMBER automatic normalization does not use --reviews/--annotations')
    args.output = args.output or ROOT/config['paths']['generated']/'generated.jsonl'
    args.audit = args.audit or ROOT/Path(config['seed_setup']['inventory']).parent/'generation_audit.csv'
    args.source_audit = args.source_audit or args.audit.with_suffix('.source.jsonl')
    args.accepted_output = args.accepted_output or args.output.with_name(args.output.stem+'.valid.jsonl')
    args.normalized_output = args.normalized_output or args.output.with_name(args.output.stem+'.normalized.jsonl')
    args.input = args.input or ROOT/config['paths']['annotated']/config.get('canonical_filename', 'canonical_seed.jsonl')
    progress_path = args.output.with_suffix('.progress.jsonl')
    if any(p.exists() for p in (args.output,args.audit,args.source_audit,args.accepted_output,args.normalized_output)) and not (args.resume and progress_path.exists()):
        parser.error('Existing generation/audit artifacts are preserved; provide new --output and --audit paths')
    fingerprint = signature({k:v for k,v in vars(args).items() if k != 'resume'},
        [args.input, args.config, args.holdout_plan, args.annotations, args.reviews,
         *sorted((ROOT/'prompts').glob('*.*')), *sorted((ROOT/'src/data').glob('*.py')), Path(__file__)])
    journal = Journal(progress_path, fingerprint, resume=args.resume)
    if bool(args.endpoint) != bool(args.model):
        parser.error('--endpoint and --model must be provided together')
    if args.limit_seeds < 1:
        parser.error('--limit-seeds must be positive')
    if args.limit_generate_seeds is not None and (args.limit_generate_seeds < 1 or args.execution_mode != 'pilot'):
        parser.error('--limit-generate-seeds must be positive and is pilot-only')
    if args.limit_dialogues is not None and (args.limit_dialogues < 1 or args.execution_mode != 'pilot'):
        parser.error('--limit-dialogues must be positive and is pilot-only')
    if args.paraphrase_budget is not None and (args.paraphrase_budget<0 or not guarded):
        parser.error('--paraphrase-budget requires guarded mode and a nonnegative budget')
    # Create outputs before the first inference, even if every source is rejected.
    atomic_text(args.output, '')
    atomic_text(args.accepted_output, '')
    atomic_text(args.normalized_output, '')
    events_path = args.audit.with_suffix('.events.jsonl')
    model_output_path = args.audit.with_suffix('.model.jsonl')
    event_csv = args.audit.with_suffix('.events.csv')
    summary_csv = args.audit.with_suffix('.stages.csv')
    atomic_text(events_path, '')
    if not args.resume:
        atomic_text(model_output_path, '')
    if client is not None:
        client.stage_callback = lambda row, stage, raw: append_json(model_output_path,
            dict(diagnostic(row, stage, 'model_response'), raw=raw, serving_provenance=client.serving_provenance))
    event_fields = list(diagnostic({}, '', ''))
    buffer = io.StringIO(newline='')
    csv.DictWriter(buffer, fieldnames=event_fields).writeheader()
    atomic_text(event_csv, buffer.getvalue())
    counts = Counter()
    def emit(event):
        append_json(events_path, event)
        with event_csv.open('a', encoding='utf-8', newline='') as stream:
            csv.DictWriter(stream, fieldnames=event_fields, extrasaction='ignore').writerow(event)
            stream.flush()
            os.fsync(stream.fileno())
        counts[(event['stage'], event['status'], event.get('failure_code') or '')] += 1
        buffer = io.StringIO(newline='')
        writer = csv.writer(buffer)
        writer.writerow(['stage','status','failure_code','event_count'])
        writer.writerows([(*key, count) for key, count in sorted(counts.items())])
        atomic_text(summary_csv, buffer.getvalue())
        print(json.dumps({k:v for k,v in event.items() if k in event_fields}, ensure_ascii=False), flush=True)
    if automatic and client is not None:
        try:
            client.verify_revision()
        except ValueError as exc:
            parser.error(str(exc))
    source_audit = []
    seeds = read_items(args.input, source_audit, required=('seed_id', 'group_id'))
    def audit_add(finding):
        source_audit.append(finding)
        atomic_text(args.source_audit, ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in source_audit))
        emit(finding if finding.get('stage') else diagnostic(finding, 'normalization',
            'passed' if finding['status'] == 'normalized' else 'failed', finding.get('reason')))
    enabled = []
    for seed in seeds:
        try:
            if not selected_source(seed, config):
                raise ValueError('Disabled source')
        except Exception as exc:
            finding = dict(seed_id=seed['seed_id'], status='excluded', reason='source_selection:'+type(exc).__name__)
            audit_add(finding)
            warning('source', seed['seed_id'], finding['reason'])
            continue
        enabled.append(seed)
    seeds = enabled
    args.annotations = args.annotations or (Path(config['approvals']['annotation_revisions']) if not automatic and config.get('approvals',{}).get('annotation_revisions') else None)
    if args.annotations:
        from src.data.annotations import apply_revisions
        annotations = [json.loads(l) for l in args.annotations.read_text(encoding='utf-8').splitlines() if l.strip()]
        seeds = apply_revisions(seeds, annotations)
    args.reviews = args.reviews or (Path(config['approvals']['source_reviews']) if not automatic and config.get('approvals',{}).get('source_reviews') else None)
    reviews = [json.loads(l) for l in args.reviews.read_text(encoding='utf-8').splitlines() if l.strip()] if args.reviews else []
    by_id = {r['seed_id']: r for r in reviews}
    if len(by_id) != len(reviews) or set(by_id) - {s['seed_id'] for s in seeds}:
        parser.error('Duplicate or unknown review seed IDs')
    seeds = [apply_review(s, by_id[s['seed_id']]) if s['seed_id'] in by_id else s for s in seeds]
    holdout_plan = json.loads(args.holdout_plan.read_text(encoding='utf-8')) if args.holdout_plan else {}
    if automatic and not args.holdout_plan:
        from src.data.automatic import make_plan
        holdout_plan = make_plan(seeds)
    if any(v not in ('id','ood_composition','ood_template') for v in holdout_plan.values()):
        parser.error('Invalid holdout pool')
    if args.execution_mode == 'final' and (not holdout_plan or set(holdout_plan) != {s['group_id'] for s in seeds}):
        parser.error('Final generation requires complete predeclared group holdout plan')
    selected = []
    scenario_errors = []
    if automatic:
        # Select by stable IDs before model output; never select using Target labels.
        seeds = sorted(seeds, key=lambda s: s['seed_id'])
        random.Random(42).shuffle(seeds)
        seeds = seeds[:args.limit_seeds]
    def prepare_families():
        for seed in seeds:
            if automatic:
                from src.data.automatic import create_seed, SourceExcluded
                from src.data.progress import CachedFailure
                try:
                    reused = bool(seed.get('source_normalization'))
                    if reused:
                        from src.data.automatic import validate_scenario
                        validate_scenario(seed)
                    elif client is None:
                        raise SourceExcluded('Offline templates require a verified normalized source')
                    else:
                        seed = journal.run('normalize:'+seed['seed_id'], lambda: create_seed(seed, client))
                    append_json(args.normalized_output,seed)
                    audit_add(dict(diagnostic(seed, 'normalization', 'passed'), status='normalized',
                        normalization_origin='reused_verified' if reused else 'qwen_checked',
                        normalization=seed['source_normalization']))
                except (SourceExcluded, CachedFailure) as exc:
                    finding = {**exc.audit, **{k:v for k,v in diagnostic(seed, 'normalization', 'failed').items()
                        if k in ('seed_id','source_record_id','group_id','sample_id')},
                        'status':'excluded', 'reason':exc.audit.get('reason', 'cached_normalization_failure')}
                    audit_add(finding)
                    scenario_errors.append(finding)
                    warning('normalization', seed['seed_id'], finding['reason'])
                    continue
                except Exception as exc:
                    finding = dict(diagnostic(seed, 'normalization', 'failed', type(exc).__name__),
                        status='excluded', reason='normalization:'+type(exc).__name__)
                    audit_add(finding)
                    scenario_errors.append(finding)
                    warning('normalization', seed['seed_id'], finding['reason'])
                    continue
            if len(selected) >= (args.limit_generate_seeds or args.limit_seeds):
                emit(diagnostic(seed, 'synthetic_policy', 'skipped', 'pilot_generation_limit'))
                continue
            try:
                blockers = generation_blockers(seed)
                if blockers:
                    raise ValueError('Seed validation failed: '+','.join(blockers))
                from src.data.source_policy import enforce_source_policy
                enforce_source_policy([seed], args.config)
                variants = make_variants(seed, config.get('modify_subtypes', ['authorization']),
                    synthetic_purpose=config.get('synthetic_policy_purpose', 'confidential discussion'))
                if guarded:
                    if config.get('variant_policy')=='coverage_v1':
                        from src.data.mutations import expand_coverage_variants
                        variants = expand_coverage_variants(variants)
                    for variant in variants:
                        variant['generation_architecture'] = 'guarded_v1'
                if not variants:
                    raise ValueError('No variants')
                variants = apply_holdout_plan(variants, holdout_plan.get(seed['group_id'],'id'))
                from src.utils.reproducibility import content_hash
                for variant in variants:
                    variant['holdout_plan_hash'] = content_hash(holdout_plan)
                    variant['holdout_plan'] = holdout_plan
                    if guarded:
                        variant['paraphrase_attempt_limit'] = config.get('qwen_paraphrase_attempts',1)
                        variant['paraphrase_scope'] = args.paraphrase_scope
                selected.append(variants)
                emit(diagnostic(seed, 'synthetic_policy', 'passed'))
            except Exception as exc:
                finding = dict(diagnostic(seed, 'synthetic_policy', 'failed', 'synthetic_policy_preparation:'+type(exc).__name__),
                    status='excluded', reason='preparation:'+type(exc).__name__)
                audit_add(finding)
                scenario_errors.append(finding)
                warning('preparation', seed['seed_id'], finding['reason'])
                continue
            yield variants
    from src.data.fewshots import load_fewshots, expression_audit, check_fewshot_overlap
    from src.data.splitting import DEFAULT_TEMPLATES
    shots = load_fewshots(ROOT/'prompts/naturalization_shots.json', DEFAULT_TEMPLATES)
    records = []
    qwen_used = 0
    fields = ['row_type','sample_id','capability_id','group_id','variant_group_id','generation_status','validation_status','errors','manual_review_selected','oracle_relation_status','N_structured','N_attempted','N_generated','N_validated','generation_success_rate','validation_yield','controlled_family_status','generation_method','fallback_used','qwen_validation_status','N_qwen_attempted','N_qwen_verified','N_fallback','qwen_success_rate','fallback_rate','paraphrase_scope','paraphrase_coverage']
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.audit.open('w', encoding='utf-8', newline='') as stream:
        csv.DictWriter(stream, fieldnames=fields).writeheader()
        stream.flush()
        os.fsync(stream.fileno())
    for variants in prepare_families():
        for variant in variants:
            if args.limit_dialogues is not None and len(records) >= args.limit_dialogues:
                from src.data.oracle_adapter import evaluate_structured
                record = dict(variant, structured_trace=variant, turns=None,
                    generation_status='not_run_pilot_limit', validation_status='not_run', errors=[],
                    structured_oracle=evaluate_structured(variant),
                    diagnostics=[diagnostic(variant, 'generation', 'skipped', 'pilot_dialogue_limit')])
            else:
                verbalizer = None if args.template_only or (guarded and args.paraphrase_budget is not None and qwen_used>=args.paraphrase_budget) else client
                record = journal.run('generate:'+variant['sample_id'], lambda: execute(variant, verbalizer, ROOT/'prompts'))
                qwen_used += bool(record.get('qwen_attempts'))
            if guarded and record.get('validation_status')=='semantic_pass_oracle_verified':
                try:
                    from src.data.guarded_dialogue import validate_record
                    from src.data.contracts import validate_pipeline_record
                    validate_record(record)
                    validate_pipeline_record(record,stage='gold')
                    append_json(args.accepted_output,record)
                except Exception as exc:
                    record.update(validation_status='UNVERIFIED',gold_training_eligible=False)
                    record['errors'].append('admission:'+type(exc).__name__)
            record['controlled_family_status'] = 'pending'
            records.append(record)
            # A restart rebuilds this view from the durable journal without calling Qwen again.
            append_json(args.output, record)
            for event in record.get('diagnostics', []):
                emit(event)
            with args.audit.open('a', encoding='utf-8', newline='') as stream:
                csv.DictWriter(stream, fieldnames=fields).writerow({
                    **{k:record[k] for k in fields if k in record},
                    'row_type':'sample','errors':json.dumps(record['errors'])})
                stream.flush()
                os.fsync(stream.fileno())
            print(json.dumps(dict(stage='generation', sample_id=record['sample_id'],
                completed=len(records), generation_status=record['generation_status'],
                validation_status=record['validation_status'], controlled_family_status='pending',
                errors=record['errors']), ensure_ascii=False), flush=True)
    if not selected:
        parser.error('No generation-ready seeds; inspect source/events reports: '+str(args.source_audit))
    # 같은 seed에서 나온 base와 variant는 생성에 실패한 항목도 함께 남긴다.
    for record in records:
        try:
            check_fewshot_overlap([record], shots['shots'], DEFAULT_TEMPLATES)
        except Exception as exc:
            record['validation_status'] = 'rejected'
            record['errors'].append('fewshot_overlap:'+type(exc).__name__)
            warning('fewshot_overlap', record['sample_id'], record['errors'])
    groups = defaultdict(list)
    for record in records:
        groups[record['variant_group_id']].append(record)
    for family in groups.values():
        try:
            validate_controlled_family(family)
        except Exception as exc:
            for record in family:
                record['validation_status'] = 'rejected'
                record['errors'].append('family:'+type(exc).__name__)
            warning('family', family[0]['variant_group_id'], type(exc).__name__)
        from src.data.contracts import validate_pipeline_record
        for record in family:
            if record.get('validation_status') == 'semantic_pass_oracle_verified':
                try:
                    validate_pipeline_record(record,stage='gold')
                except Exception as exc:
                    record['validation_status'] = 'rejected'
                    record['errors'].append('gold_contract:'+type(exc).__name__)
                    warning('gold_contract', record['sample_id'], record['errors'])
        complete = all(r['validation_status'] == 'semantic_pass_oracle_verified' for r in family)
        for record in family:
            admitted = record['validation_status']=='semantic_pass_oracle_verified' and record.get('oracle_relation_status')=='passed' if guarded else complete
            record['controlled_family_status'] = 'semantic_complete_oracle_verified' if admitted else 'incomplete'
            emit(diagnostic(record, 'family_validation', 'passed' if admitted else 'failed',
                None if admitted else 'incomplete_controlled_family'))
            if record.get('validation_status') != 'semantic_pass_oracle_verified':
                warning('validation', record['sample_id'], record.get('errors') or record['validation_status'])
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    for record in records:
        record['manual_review_selected'] = False
    atomic_text(output, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    accepted = [r for r in records if r.get('validation_status')=='semantic_pass_oracle_verified'
        and r.get('controlled_family_status')=='semantic_complete_oracle_verified'
        and (not guarded or r.get('gold_training_eligible') is True)]
    atomic_text(args.accepted_output,''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in accepted))
    audit = args.audit
    audit.parent.mkdir(parents=True, exist_ok=True)
    with audit.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({**{k:r[k] for k in fields if k in r},'row_type':'sample','errors':json.dumps(r['errors'])})
        for cap in [None] + [f'CAP{i:02}' for i in range(1,10)]:
            subset = [r for r in records if r['capability_id'] == cap]
            attempted = sum(r['generation_status'] not in ('pending_qwen','not_run_pilot_limit') for r in subset)
            generated = sum(r['generation_status'] == 'generated' for r in subset)
            valid = sum(r['validation_status'] == 'semantic_pass_oracle_verified' for r in subset)
            model_attempted = sum(bool(r.get('qwen_attempts')) for r in subset)
            model_valid = sum(r.get('generation_method')=='qwen_paraphrase' for r in subset)
            fallback = sum(r.get('fallback_used',False) for r in subset)
            writer.writerow({'row_type':'summary','capability_id':cap or 'BASE','N_structured':len(subset),'N_attempted':attempted,'N_generated':generated,'N_validated':valid,'generation_success_rate':(model_valid/model_attempted if model_attempted else None) if guarded else generated/attempted if attempted else None,'validation_yield':valid/attempted if attempted else None,'oracle_relation_status':'pending',
                'N_qwen_attempted':model_attempted,'N_qwen_verified':model_valid,'N_fallback':fallback,
                'qwen_success_rate':model_valid/model_attempted if model_attempted else None,
                'fallback_rate':fallback/model_attempted if model_attempted else None})
    coverage = dict(
        eligible_policy_turns=sum(len(r.get('paraphrase_coverage',{}).get('eligible_turn_ids',[])) for r in records),
        changed_policy_turns=sum(len(r.get('paraphrase_coverage',{}).get('changed_turn_ids',[])) for r in records),
        fallback_dialogues=sum(bool(r.get('fallback_used')) for r in records))
    print(json.dumps({'eligible_seed_pool':len(seeds), 'selected_seeds':len(selected),'scenario_errors':scenario_errors,'structured_records':len(records),'qwen_executed':client is not None,'experiment':{'training_seeds':[42],'single_seed_experiment':True},'expression_audit':expression_audit(records),'paraphrase_coverage':coverage,'oracle_relation_status':'pending'}))

if __name__ == '__main__':
    main()
