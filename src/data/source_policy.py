"""선택한 소스와 원문 근거에 연결된 사람의 검수 기록을 검증한다."""
from copy import deepcopy

SCOPE = ('subject', 'sender', 'recipient', 'data_type', 'purpose')


def enforce_source_policy(rows, config_path='configs/data.yaml'):
    """실행 CLI에서 소스 선택과 검수 승인을 확인한다. adapter나 테스트용 함수는 별도로 호출할 수 있다."""
    from pathlib import Path
    import yaml
    config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    from src.data.canonicalize import active_setup, download_source, load_source
    root = Path(__file__).resolve().parents[2]
    originals = {}
    for source, spec in active_setup(config)['sources'].items():
        raw = root / config['paths']['raw']
        download_source(raw/source, spec, offline=True)
        originals.update({r['seed_id']: r for r in load_source(raw, source, spec)})
    for row in rows:
        if not selected_source(row, config):
            raise ValueError('Disabled source/component; update source_selection explicitly')
        if config.get('generation_mode') == 'camber_automatic':
            from src.data.automatic import validate_scenario
            validate_scenario(row)
            original = originals.get(row.get('seed_id'))
            if original is None or any(row.get(k) != original.get(k) for k in
                    ('source_dataset', 'source_record_id', 'source_seed_id', 'group_id', 'source_provenance')):
                raise ValueError('Canonical identity/provenance mismatch')
            continue
        if generation_blockers(dict(row, status='eligible_pending_review')):
            raise ValueError('Evidence-bound source review/generation approval missing')
        validate_source_approval(row, originals)
    return config


def validate_source_approval(row, originals):
    """고정된 원본 record와 검수 근거를 대조한다. 복사된 승인 표시만으로 통과시키지 않는다."""
    original = originals.get(row.get('seed_id'))
    if original is None:
        raise ValueError('Unknown source seed identity')
    for key in ('source_dataset', 'source_record_id', 'source_seed_id', 'group_id', 'source_provenance'):
        if row.get(key) != original.get(key):
            raise ValueError('Source record/group/hash mismatch: ' + key)
    if row.get('annotation'):
        from src.data.annotations import apply_annotation
        original = apply_annotation(original, row['annotation'])
    approved = apply_review(original, row.get('semantic_review') or {})
    if generation_blockers(approved):
        raise ValueError('Source review not approved or unclear')
    for key in ('source_evidence', 'consent_assessment', 'disclosure_assessment',
                'purpose_mapping_status', 'generation_ready', 'human_review_status'):
        if row.get(key) != approved.get(key):
            raise ValueError('Source approval metadata changed: ' + key)
    for key in ('annotation', 'source_motive', 'disclosure_purpose', 'source_checks'):
        if row.get(key) != approved.get(key):
            raise ValueError('Source annotation metadata changed: ' + key)
    facts = row.get('source_facts', row)
    if any(facts.get(k) != original[k] for k in SCOPE):
        raise ValueError('Source scope facts changed after review')
    return approved


def selected_source(row, config):
    selection = config.get('source_selection')
    if selection is None:
        return row.get('source_dataset') in config['sources']
    source = row.get('source_dataset')
    if source not in selection:
        return False
    component = row.get('source_component')
    if component is None and source == 'CAMBER':
        component = row.get('source_record_id', '').split(':')[0]
    return source != 'CAMBER' or component in selection[source]


def generation_blockers(seed):
    if seed.get('source_normalization', {}).get('mode') == 'camber_automatic':
        from src.data.automatic import validate_scenario
        try:
            validate_scenario(seed)
            return []
        except (ValueError, KeyError, TypeError):
            return ['invalid_qwen_scenario']
    reasons = []
    if seed.get('source_checks', {}).get('conflicts'):
        reasons.append('source_semantic_conflict')
    if seed.get('source_checks', {}).get('checks', {}).get('relationship_reference', {}).get('conflict_indicators'):
        reasons.append('relationship_conflict_requires_resolution')
    if not seed.get('purpose') and not seed.get('source_facts', {}).get('purpose'):
        reasons.append('disclosure_purpose_unresolved')
    if seed.get('status') != 'eligible_pending_review':
        reasons.append('structurally_ineligible')
    review = seed.get('semantic_review') or {}
    if seed.get('human_review_status') != 'reviewed' or not review.get('reviewer'):
        reasons.append('human_review_pending')
    if seed.get('purpose_mapping_status') != 'confirmed':
        reasons.append('purpose_mapping_unverified')
    if any((seed.get('source_evidence', {}).get(k) or {}).get('status') != 'confirmed' for k in SCOPE):
        reasons.append('scope_evidence_unverified')
    for field in ('consent_assessment', 'disclosure_assessment'):
        if seed.get(field, {}).get('status') != 'reviewed':
            reasons.append(field + '_pending')
    if review.get('synthetic_policy_compatible') is not True:
        reasons.append('synthetic_policy_compatibility_unverified')
    if seed.get('generation_ready') is not True:
        reasons.append('generation_not_approved')
    return reasons


def apply_review(seed, review):
    """해당 원본 record에 제공된 검수만 적용한다. 같은 group의 다른 record에는 전파하지 않는다.

    각 판단에는 원문 인용과 JSON 필드 경로가 필요하다. 동의가 없다는 검수 결과를 허용으로 간주하지 않는다."""
    if review.get('seed_id') != seed['seed_id'] or review.get('record_hash') != seed['source_provenance']['record_hash']:
        raise ValueError('Review does not bind to this exact source record')
    annotation = seed.get('annotation')
    if annotation and (review.get('annotation_hash') != annotation['annotation_hash'] or
                       review.get('annotation_revision') != annotation['annotation_revision']):
        raise ValueError('Review must bind exact approved annotation revision')
    if not all(isinstance(review.get(k), str) and review[k].strip() for k in ('reviewer', 'reviewed_at', 'rationale')):
        raise ValueError('Reviewer, time and rationale required')
    scenario = __import__('ast').literal_eval(seed['source_record']['scenario'])
    def evidence(finding):
        path, quote = finding.get('path', ''), finding.get('quote')
        value = scenario.get(path.removeprefix('scenario.')) if path.startswith('scenario.') else None
        if not isinstance(quote, str) or not quote.strip() or not isinstance(value, str) or quote not in value:
            raise ValueError('Review requires a verbatim source excerpt and valid scenario path')
    result = deepcopy(seed)
    for field in SCOPE:
        finding = review['scope'][field]
        evidence(finding)
        if not seed[field] or finding.get('value') != seed[field] or finding.get('confirmed') is not True:
            raise ValueError('Scope mismatch requires explicit reannotation, not approval')
        if str(seed[field]).casefold() not in finding['quote'].casefold():
            raise ValueError('Scope value must be supported by its own source quote')
        allowed = {'subject': 'subject_agent', 'sender': 'aware_agent', 'recipient': 'oblivious_agent',
                   'data_type': 'detail', 'purpose': 'story'}
        if seed.get('source_component') == 'confaideplus' and finding['path'] not in ('scenario.story', 'scenario.' + allowed[field]):
            raise ValueError('Wrong scope source mapping')
        result['source_evidence'][field] = dict(finding, status='confirmed')
    for field in ('consent_assessment', 'disclosure_assessment'):
        finding = review[field]
        evidence(finding)
        if finding.get('value') not in ('explicit', 'absent', 'unclear'):
            raise ValueError('Explicit consent/disclosure assessment required')
        result[field] = dict(finding, status='reviewed')
    compatible = review.get('synthetic_policy_compatible') is True
    clear = all(result[k]['value'] != 'unclear' for k in ('consent_assessment', 'disclosure_assessment'))
    result.update(semantic_review=deepcopy(review), human_review_status='reviewed',
                  purpose_mapping_status='confirmed', generation_ready=compatible and clear and not seed['exclusion_reasons']
                  and not seed.get('source_checks', {}).get('conflicts'))
    return result
