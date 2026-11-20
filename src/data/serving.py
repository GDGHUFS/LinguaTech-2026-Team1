"""Generation provenance bound to the serving engine's reported immutable revision."""
import re
from src.utils.reproducibility import content_hash


def server_provenance(info, model, revision, endpoint):
    config = info.get('vllm_config', {}).get('model_config', {})
    if not re.fullmatch(r'[0-9a-f]{40}', revision or ''):
        raise ValueError('Generation revision NOT VERIFIED: immutable expected SHA required')
    if (config.get('model') != model or config.get('revision') != revision or
            config.get('tokenizer_revision') != revision):
        raise ValueError('Generation revision NOT VERIFIED: serving engine model/revisions differ or are absent')
    loaded = (config.get('hf_config') or {}).get('_commit_hash')
    if loaded != revision:
        raise ValueError('Generation revision NOT VERIFIED: loaded config commit absent or different')
    evidence = {k: config.get(k) for k in ('model', 'revision', 'tokenizer_revision')}
    evidence['loaded_config_commit'] = loaded
    return dict(status='server_reported_engine_config', model_id=model, model_revision=revision,
                endpoint=endpoint, evidence=evidence, evidence_hash=content_hash(evidence))


def validate_serving(meta):
    evidence = meta.get('evidence') or {}
    revision = meta.get('model_revision')
    if (meta.get('status') != 'server_reported_engine_config' or
            not re.fullmatch(r'[0-9a-f]{40}', revision or '') or
            evidence.get('model') != meta.get('model_id') or
            evidence.get('revision') != revision or evidence.get('tokenizer_revision') != revision or
            evidence.get('loaded_config_commit') != revision or
            meta.get('evidence_hash') != content_hash(evidence)):
        raise ValueError('Generation model revision NOT VERIFIED')


def validate_generation_cohort(rows):
    signatures = set()
    for row in rows:
        for stage in (row.get('source_normalization') or {}, row.get('execution') or {}):
            meta = stage.get('serving_provenance') or {}
            validate_serving(meta)
            if stage.get('model_id') != meta['model_id']:
                raise ValueError('Generation model/provenance mismatch')
            signatures.add((meta['model_id'], meta['model_revision']))
    if not rows or len(signatures) != 1:
        raise ValueError('One verified generation model/revision required')
    model, revision = next(iter(signatures))
    return dict(model_id=model, model_revision=revision)
