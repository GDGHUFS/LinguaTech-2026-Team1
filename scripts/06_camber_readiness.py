"""Offline source pilot; writes evidence/review queues, never dialogues or model metrics."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from src.data.canonicalize import active_setup, download_source, load_source, select_group_candidates, screening_summary
from src.data.source_policy import SCOPE, generation_blockers
from src.data.capability_matrix import CAPABILITIES


def audit(rows, pilot, config):
    """원본과 group 표본의 구조 적격·검수 상태를 집계한다. 실제 생성 대화나 모델 성능은 측정하지 않는다."""
    def summarize(items):
        result = screening_summary(items)
        result['independent_upstream_groups'] = len({r['group_id'] for r in items})
        result['canonicalization_success_rate'] = sum('invalid_scenario' not in r['exclusion_reasons'] for r in items)/len(items)
        result['scope_verbatim_evidence_rates'] = {k: sum(bool(r.get('source_evidence', {}).get(k, {}).get('quote')) for r in items)/len(items) for k in SCOPE}
        result['semantic_scope_confirmed_groups'] = len({r['group_id'] for r in items if not generation_blockers(r)})
        result['generation_blockers_overlapping'] = dict(Counter(reason for r in items for reason in generation_blockers(r)))
        result['blocker_upstream_groups_overlapping'] = {
            reason: len({r['group_id'] for r in items if reason in generation_blockers(r)})
            for reason in result['generation_blockers_overlapping']}
        result['consent_review'] = dict(Counter(r['consent_assessment']['status'] for r in items))
        result['disclosure_review'] = dict(Counter(r['disclosure_assessment']['status'] for r in items))
        # 정확히 일치하는 질문 패턴만 충돌 후보로 표시한다. 의미적 충돌을 확정하지는 않는다.
        conflicts = []
        for row in items:
            match = re.search(r'Is it ok for (\S+) to share', row.get('source_question') or '')
            if match and match[1] != row['sender']:
                conflicts.append({'seed_id': row['seed_id'], 'group_id': row['group_id'],
                                  'structured_sender': row['sender'], 'question_sender': match[1]})
        result['question_sender_conflict_flags'] = len(conflicts)
        result['question_sender_conflict_upstream_groups'] = len({r['group_id'] for r in conflicts})
        result['question_sender_conflict_examples'] = conflicts[:5]
        result['reason_count_unit'] = 'records; reasons overlap; group counts are distinct group_id'
        return result
    ready = [r for r in pilot if not generation_blockers(r)]
    return dict(revision=config.get('source_policy_revision','camber-v2'), status='READY_FOR_GENERATION' if ready else 'BLOCKED', source_selection=config['source_selection'],
        source_integrity='PASS_offline_git_blob_hash', full_inventory=summarize(rows), pilot=summarize(pilot),
        pilot_sampling_unit='independent_upstream_group_one_representative',
        pilot_group_ids=[r['group_id'] for r in pilot],
        group_expansion_sizes=dict(Counter(Counter(r['group_id'] for r in rows).values())),
        capabilities={cap: dict(operation=name, relation=relation, implementation='existing_structured_template',
            reviewed_source_groups=len(ready), valid_natural_dialogues=0, generation_status='READY_FOR_GENERATION' if ready else 'BLOCKED_pending_source_review',
            semantic_validation='NOT_RUN') for cap, (name, relation) in CAPABILITIES.items()},
        gold_decision_distribution=None, gold_status='NOT_RUN_no_generated_trajectories',
        split_feasibility=dict(structural_group_capacity=len(pilot) >= 50,
            approved_upstream_groups=len({r['group_id'] for r in ready}), actual_id_ood_feasible=False,
            reason='Qwen dialogues, observed Target labels and approved split freeze required',
            assignment_unit='upstream_group_including_all_expansions',
            ood_template_scope='question_embedding / b_ood_question_embedding_v1 only; semantic generalization NOT_ESTABLISHED'),
        qwen_calls=0, target_calls=0, training='NOT_RUN', human_review_performed=bool(ready))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/data.yaml')
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/reports/camber_v2/readiness.json')
    parser.add_argument('--reviews',type=Path,help='Actual exact-record human review JSONL')
    parser.add_argument('--annotations',type=Path,help='Approved annotation revision JSONL; pending proposals are rejected')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Existing report preserved; choose a new --out')
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    setup = active_setup(config)
    rows = []
    for name, spec in setup['sources'].items():
        download_source(ROOT/config['paths']['raw']/name, spec, offline=True)
        rows.extend(load_source(ROOT/config['paths']['raw'], name, spec))
    pilot = select_group_candidates(rows, setup['candidates_per_source'], setup['random_seed'])
    originals={r['seed_id']:r for r in rows}
    if args.annotations:
        from src.data.annotations import apply_revisions
        annotations=[json.loads(l) for l in args.annotations.read_text(encoding='utf-8').splitlines() if l.strip()]
        rows=apply_revisions(rows,annotations)
        by_seed={r['seed_id']:r for r in rows}
        pilot=[by_seed[r['seed_id']] for r in pilot]
    if args.reviews:
        from src.data.source_policy import apply_review, validate_source_approval
        reviews=[json.loads(l) for l in args.reviews.read_text(encoding='utf-8').splitlines() if l.strip()]
        by_id={r['seed_id']:r for r in reviews}
        if len(by_id)!=len(reviews) or set(by_id)-set(originals):
            raise ValueError('Duplicate/unknown review IDs')
        rows=[apply_review(r,by_id[r['seed_id']]) if r['seed_id'] in by_id else r for r in rows]
        pilot=[apply_review(r,by_id[r['seed_id']]) if r['seed_id'] in by_id else r for r in pilot]
        for r in rows:
            if r.get('generation_ready'): validate_source_approval(r,originals)
    result = audit(rows, pilot, config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
