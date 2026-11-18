"""자동 Qwen/oracle 검증과 데이터 freeze, 선택적 기존 승인 기록을 검증한다."""
import json
from pathlib import Path
from src.utils.reproducibility import content_hash, file_hash


def identity(document):
    if not all(isinstance(document.get(k), str) and document[k].strip()
               for k in ('reviewer', 'reviewed_at', 'rationale')):
        raise ValueError('Human reviewer, reviewed_at and rationale required')
    if document.get('approved') is not True:
        raise ValueError('Approval missing/pending')


def qa_payload(row):
    keys = ('sample_id', 'seed_id', 'source_provenance', 'turns', 'final_request',
            'request', 'initial_rules', 'operation_trace', 'gold_decision', 'gold_state',
            'supporting_turn_ids', 'semantic_validation', 'execution', 'annotation',
            'source_facts', 'source_motive', 'disclosure_purpose', 'source_checks', 'policy_temporal_boundary')
    return {k: row.get(k) for k in keys}


def validate_qa(rows, manifest):
    identity(manifest)
    entries = manifest.get('samples', {})
    if not rows:
        raise ValueError('Empty QA cohort')
    for row in rows:
        entry = entries.get(row['sample_id'], {})
        identity(entry)
        if entry.get('content_hash') != content_hash(qa_payload(row)):
            raise ValueError('QA sample content/hash mismatch')
        if row.get('validation_status') != 'semantic_pass_oracle_verified':
            raise ValueError('Semantic validation missing')
        if row.get('controlled_family_status') != 'semantic_complete_oracle_verified':
            raise ValueError('Incomplete controlled family')
        if entry.get('semantic_aligned') is not True or entry.get('support_verified') is not True:
            raise ValueError('Independent semantic/support QA missing')
        if 'policy_temporal_boundary' in row or row.get('capability_id') in ('CAP07','CAP08','CAP09'):
            from src.data.validators import validate_semantics
            if validate_semantics(row.get('semantic_validation'), row, {'turns': row.get('turns', [])}):
                raise ValueError('Required CAP semantic checks unresolved')
            if entry.get('temporal_boundary_verified') is not True or entry.get('request_scope_verified') is not True:
                raise ValueError('Independent temporal/scope QA missing')
            if row.get('capability_id') == 'CAP09':
                if entry.get('full_dialogue_recipient_unrecoverable') is not True:
                    raise ValueError('Independent CAP09 context QA missing')
                import re
                recipient = row.get('source_facts', {}).get('recipient')
                if recipient and any(re.search(r'(?<!\w)' + re.escape(recipient) + r'(?!\w)', t['text'], re.I) for t in row['turns']):
                    raise ValueError('CAP09 recipient leaked in dialogue')


SPLIT_FIELDS = ('sample_id','source_dataset','source_record_id','source_seed_id','scenario_id','seed_id',
                'group_id','variant_group_id','split','split_manifest_hash','template_family_id',
                'operation_trace','final_request','turns','evaluation_eligible')


def split_membership(rows):
    return [{k:r[k] for k in SPLIT_FIELDS if k in r} for r in rows]


def validate_automatic_freeze(rows, manifest):
    import re
    import yaml
    from src.data.splitting import check_leakage
    from src.evaluation.protocol import _manifest
    from src.data.serving import validate_generation_cohort
    snapshot = manifest.get('split_manifest')
    membership = manifest.get('split_membership')
    if not snapshot or not membership:
        raise ValueError('Complete split manifest/membership required for automatic freeze')
    check_leakage(membership, snapshot['ood_compositions'], snapshot['ood_templates'],
                  snapshot.get('fewshot_template_families', ()))
    _manifest(snapshot, membership)
    indexed = {r['sample_id']:r for r in membership}
    if set(indexed) != set(manifest.get('records', {})):
        raise ValueError('Frozen record/membership IDs differ')
    for r in split_membership(rows):
        if indexed.get(r['sample_id']) != r:
            raise ValueError('Frozen split membership changed')
    generation = validate_generation_cohort(rows)
    if manifest.get('generation_model') != generation:
        raise ValueError('Frozen generation revision differs')
    model_path = manifest.get('model_config_path')
    if not model_path or manifest.get('config_hashes', {}).get(model_path) != file_hash(model_path):
        raise ValueError('Frozen model config required/changed')
    model = yaml.safe_load(Path(model_path).read_text(encoding='utf-8'))
    for role in ('encoder','target','generation'):
        if not re.fullmatch(r'[0-9a-f]{40}', model.get(role, {}).get('revision') or ''):
            raise ValueError('Unverified immutable model revision: '+role)
    if any(model['generation'].get(k) != generation[k] for k in ('model_id',)) or model['generation']['revision'] != generation['model_revision']:
        raise ValueError('Serving Qwen revision differs from frozen configuration')
    for r in rows:
        target = r.get('target_execution') or {}
        if target.get('model_id') != model['target']['model_id'] or target.get('model_revision') != model['target']['revision']:
            raise ValueError('Target revision differs from frozen configuration')


def validate_freeze(rows, manifest, data_config, config_paths):
    automatic = data_config.get('generation_mode') == 'camber_automatic'
    if automatic:
        if manifest.get('schema_version') != 'automatic-freeze-1':
            raise ValueError('Automatic freeze manifest required')
        from src.data.automatic import validate_generated
        validate_generated(rows)
        validate_automatic_freeze(rows, manifest)
    else:
        identity(manifest)
    audit = manifest.get('audit_summary', {})
    if not automatic and (audit.get('independent_support_annotations', 0) < 50 or
            audit.get('semantic_alignment', 0) < .95 or
            audit.get('support_f1', 0) < .95 or audit.get('support_exact_match', 0) < .90 or
            audit.get('leave_one_turn_out_checked') is not True):
        raise ValueError('Final independent human QA audit gates not met')
    if manifest.get('source_selection') != data_config['source_selection']:
        raise ValueError('Frozen source selection mismatch')
    plan = manifest.get('holdout_plan', {})
    if not plan or any(plan.get(r['group_id']) not in ('id','ood_composition','ood_template') for r in rows):
        raise ValueError('Complete predeclared group holdout plan required')
    for row in rows:
        if row.get('holdout_plan_hash') != content_hash(plan):
            raise ValueError('Holdout plan changed since generation')
        if manifest.get('records', {}).get(row['sample_id']) != content_hash(row):
            raise ValueError('Frozen sample changed')
        pool = plan[row['group_id']]
        if (pool == 'id' and row.get('split','').startswith('ood_')) or (pool != 'id' and row.get('split') != pool):
            raise ValueError('Split differs from approved holdout plan')
    if not config_paths:
        raise ValueError('Frozen configuration paths required')
    for path in config_paths:
        if manifest.get('config_hashes', {}).get(str(path)) != file_hash(path):
            raise ValueError('Frozen config changed: ' + str(path))


def enforce_execution(rows, *, mode='pilot', qa_path=None, freeze_path=None,
                      data_config='configs/data.yaml', config_paths=()):
    """설정된 생성 경로의 검증과 final record/config freeze를 확인한다."""
    from src.data.source_policy import enforce_source_policy
    config = enforce_source_policy(rows, data_config)
    if mode not in ('pilot','final'):
        raise ValueError('Execution mode must be pilot/final')
    if config.get('generation_mode') == 'camber_automatic':
        from src.data.automatic import validate_generated
        validate_generated(rows)
        if mode == 'final':
            freeze_path = freeze_path or config.get('approvals', {}).get('freeze_manifest')
            if not freeze_path:
                raise ValueError('Run scripts/08_freeze.py to create the automatic freeze manifest')
            validate_freeze(rows, json.loads(Path(freeze_path).read_text(encoding='utf-8')), config, config_paths)
        return config
    if not qa_path:
        raise ValueError('Generated human QA manifest required')
    validate_qa(rows, json.loads(Path(qa_path).read_text(encoding='utf-8')))
    if mode == 'final':
        if not freeze_path:
            raise ValueError('Final approved freeze manifest required')
        validate_freeze(rows, json.loads(Path(freeze_path).read_text(encoding='utf-8')), config, config_paths)
    return config


def research_rows(rows):
    from src.data.labeling import validate_observed_provenance, validate_target_cohort
    if not rows:
        raise ValueError('Research cohort empty')
    template = Path('prompts/target_decision.txt').read_text(encoding='utf-8-sig')
    for row in rows:
        if row.get('data_kind') not in ('real_derived','real_derived_unlabeled','model_generated') or row.get('source_provenance',{}).get('fixture'):
            raise ValueError('Mock/fixture/pending research data forbidden')
        if row.get('parse_status') == 'VALID':
            validate_observed_provenance(row, template)
        elif row.get('parse_status') not in ('INVALID_OUTPUT','EXECUTION_ERROR') or not row.get('target_attempts'):
            raise ValueError('Unobserved Target output forbidden')
    validate_target_cohort([r for r in rows if r.get('parse_status') == 'VALID'])
