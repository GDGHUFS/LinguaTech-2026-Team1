"""Immutable derived annotations: raw hash -> parent hash -> revision -> exact human review.

This module accepts proposals; it never creates a reviewer identity or source approval.
Literal support is a necessary check, not a semantic verdict.
"""
import ast
from copy import deepcopy
import json
from pathlib import Path
from src.utils.reproducibility import content_hash

FIELDS = ('subject', 'sender', 'recipient', 'data_type', 'purpose',
          'recipient_relation', 'relationship_reference', 'information_publicness')


def annotation_hash(annotation):
    return content_hash({k: v for k, v in annotation.items() if k != 'annotation_hash'})


def validate_annotation(seed, annotation, *, approved=False):
    from src.data.canonicalize import digest
    raw_hash = digest(json.dumps(seed['source_record'], sort_keys=True, ensure_ascii=False).encode())
    if raw_hash != seed['source_provenance']['record_hash']:
        raise ValueError('Original source record content/hash mismatch')
    if annotation.get('schema_version') != 'derived-annotation-1':
        raise ValueError('Unsupported annotation schema')
    if annotation.get('seed_id') != seed['seed_id'] or annotation.get('record_hash') != seed['source_provenance']['record_hash']:
        raise ValueError('Annotation exact-record hash mismatch')
    if annotation.get('parent_canonical_hash') != content_hash(seed):
        raise ValueError('Annotation parent canonical hash mismatch')
    if annotation.get('annotation_hash') != annotation_hash(annotation):
        raise ValueError('Annotation content hash mismatch')
    for key in ('annotation_revision', 'reason', 'proposed_by'):
        if not isinstance(annotation.get(key), str) or not annotation[key].strip():
            raise ValueError('Annotation identity/reason missing')
    scenario = ast.literal_eval(seed['source_record']['scenario'])
    if not isinstance(annotation.get('unresolved'), list) or not isinstance(annotation.get('changes'), dict):
        raise ValueError('Annotation changes and unresolved list required')
    for field, finding in annotation['changes'].items():
        if field not in FIELDS or finding.get('before') != seed.get(field):
            raise ValueError('Invalid annotation field/before value')
        path, quote = finding.get('path', ''), finding.get('quote')
        source = scenario.get(path.removeprefix('scenario.')) if path.startswith('scenario.') else None
        if not isinstance(quote, str) or not quote.strip() or not isinstance(source, str) or quote not in source:
            raise ValueError('Annotation requires exact source path/quote')
        if finding.get('status') not in ('supported', 'pending', 'unclear') or not finding.get('reason'):
            raise ValueError('Annotation status/reason required')
        after = finding.get('after')
        if after is not None and (not isinstance(after, str) or not after.strip()):
            raise ValueError('Annotation value must be nonempty text or null')
        if finding['status'] == 'supported' and (after is None or after.casefold() not in quote.casefold()):
            raise ValueError('Supported value must occur in evidence; inferred paraphrases stay pending')
        if field == 'purpose' and path == 'scenario.reveal_reason' and finding['status'] == 'supported':
            raise ValueError('Motive alone cannot establish disclosure purpose')
        mapping = dict(subject='subject_agent',sender='aware_agent',recipient='oblivious_agent',data_type='detail')
        if field in mapping and path not in ('scenario.story','scenario.'+mapping[field]):
            raise ValueError('Wrong annotation source mapping')
    if approved:
        from src.data.approvals import identity
        identity(annotation.get('approval') or {})
        approval = annotation['approval']
        if approval.get('annotation_hash') != annotation_hash({**annotation, 'approval': None}):
            raise ValueError('Approval must bind exact annotation proposal hash')
        if annotation['unresolved'] or any(f['status'] != 'supported' for f in annotation['changes'].values()):
            raise ValueError('Unresolved annotations cannot enter downstream')
    return annotation


def propose(seed, revision, changes, reason, *, proposed_by, unresolved):
    annotation = dict(schema_version='derived-annotation-1', seed_id=seed['seed_id'],
        record_hash=seed['source_provenance']['record_hash'], parent_canonical_hash=content_hash(seed),
        annotation_revision=revision, changes=deepcopy(changes), reason=reason,
        proposed_by=proposed_by, unresolved=list(unresolved), approval=None)
    annotation['annotation_hash'] = annotation_hash(annotation)
    validate_annotation(seed, annotation)
    return annotation


def save_revision(path, seed, annotation):
    validate_annotation(seed, annotation)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(annotation, stream, ensure_ascii=False, sort_keys=True, indent=2)


def apply_annotation(seed, annotation):
    validate_annotation(seed, annotation, approved=True)
    result = deepcopy(seed)
    for field, finding in annotation['changes'].items():
        result[field] = finding['after']
        result['source_evidence'][field] = dict(path=finding['path'], quote=finding['quote'], status='pending_review')
        result['field_provenance'][field] = finding['path']
    if 'purpose' in annotation['changes']:
        result['disclosure_purpose'] = dict(value=result['purpose'], status='supported',
            path=annotation['changes']['purpose']['path'], quote=annotation['changes']['purpose']['quote'])
    # Replay always starts from pinned adapter output. A new revision needs a new source review.
    result.update(annotation=deepcopy(annotation), human_review_status='pending', generation_ready=False,
                  purpose_mapping_status='pending_review')
    result.pop('semantic_review', None)
    from src.data.camber_evidence import assess
    result['source_checks'] = assess(result)
    result['exclusion_reasons'] = sorted(set(seed['exclusion_reasons'] + result['source_checks']['conflicts']))
    result['status'] = 'excluded' if result['exclusion_reasons'] else 'eligible_pending_review'
    return result


def apply_revisions(seeds, annotations):
    indexed = {a['seed_id']: a for a in annotations}
    if len(indexed) != len(annotations) or set(indexed) - {s['seed_id'] for s in seeds}:
        raise ValueError('Duplicate or unknown annotation seed IDs')
    return [apply_annotation(s,indexed[s['seed_id']]) if s['seed_id'] in indexed else s for s in seeds]
