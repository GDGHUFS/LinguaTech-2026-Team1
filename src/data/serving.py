"""Generation provenance bound to the serving engine's reported immutable revision."""
import re
from src.utils.reproducibility import content_hash


def _cached_snapshot_evidence(model, revision):
    """Read local cache only; corroboration is not proof of a remote model load."""
    import json
    from pathlib import Path
    from src.utils.reproducibility import file_hash
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(model, 'config.json', revision=revision)
        if not isinstance(cached, str):
            return None
        path = Path(cached)
        if path.parent.name != revision or path.parent.parent.name != 'snapshots':
            return None
        if not isinstance(json.loads(path.read_text(encoding='utf-8')), dict):
            return None
        return dict(model=model, revision=revision, config_sha256=file_hash(path))
    except (ImportError, OSError, ValueError):
        return None


def server_provenance(info, model, revision, endpoint):
    config = info.get('vllm_config', {}).get('model_config', {})
    if not re.fullmatch(r'[0-9a-f]{40}', revision or ''):
        raise ValueError('Generation revision NOT VERIFIED: immutable expected SHA required')
    if (config.get('model') != model or config.get('revision') != revision or
            config.get('tokenizer_revision') != revision):
        raise ValueError('Generation revision NOT VERIFIED: serving engine model/revisions differ or are absent')
    loaded = (config.get('hf_config') or {}).get('_commit_hash')
    if config.get('hf_config') is not None and loaded != revision:
        raise ValueError('Generation revision NOT VERIFIED: loaded config commit absent or different')
    evidence = {k: config.get(k) for k in ('model', 'revision', 'tokenizer_revision')}
    evidence['loaded_config_commit'] = loaded
    evidence['revision_binding'] = 'loaded_config_commit' if loaded else 'explicit_engine_revisions'
    if config.get('hf_config') is None:
        cached = _cached_snapshot_evidence(model, revision)
        if cached:
            evidence['local_cache_corroboration'] = cached
    return dict(status='server_reported_engine_config', model_id=model, model_revision=revision,
                endpoint=endpoint, evidence=evidence, evidence_hash=content_hash(evidence))


def validate_serving(meta):
    evidence = meta.get('evidence') or {}
    revision = meta.get('model_revision')
    if (meta.get('status') not in ('server_reported_engine_config', 'local_loaded_snapshot') or
            not re.fullmatch(r'[0-9a-f]{40}', revision or '') or
            evidence.get('model') != meta.get('model_id') or
            evidence.get('revision') != revision or evidence.get('tokenizer_revision') != revision or
            not (evidence.get('loaded_config_commit') == revision or
                 (meta.get('status') == 'server_reported_engine_config' and
                  evidence.get('revision_binding') == 'explicit_engine_revisions' and
                  evidence.get('loaded_config_commit') is None)) or
            meta.get('evidence_hash') != content_hash(evidence)):
        raise ValueError('Generation model revision NOT VERIFIED')
    if meta['status'] == 'local_loaded_snapshot':
        hashes = evidence.get('snapshot_hashes') or {}
        if (meta.get('endpoint') != 'transformers://cpu' or evidence.get('runtime') != 'transformers_cpu'
                or evidence.get('seed') != 42 or not hashes.get('config.json')
                or not hashes.get('tokenizer_config.json') or not any(k.endswith('.safetensors') for k in hashes)
                or any(not re.fullmatch(r'sha256:[0-9a-f]{64}', v) for v in hashes.values())):
            raise ValueError('CPU snapshot provenance incomplete')
    cached = evidence.get('local_cache_corroboration')
    if cached is not None and (not isinstance(cached, dict) or meta['status'] != 'server_reported_engine_config' or
            evidence.get('revision_binding') != 'explicit_engine_revisions' or
            cached.get('model') != meta['model_id'] or cached.get('revision') != revision or
            not re.fullmatch(r'sha256:[0-9a-f]{64}', cached.get('config_sha256', ''))):
        raise ValueError('Generation cache corroboration invalid')


def validate_generation_cohort(rows):
    if rows and all(r.get('generation_architecture')=='guarded_v1' for r in rows):
        sources, paraphrases = set(), set()
        for row in rows:
            meta = row['source_normalization']['serving_provenance']
            validate_serving(meta)
            sources.add((meta['model_id'],meta['model_revision']))
            for attempt in row.get('qwen_attempts',[]):
                if attempt.get('serving_provenance'):
                    meta = attempt['serving_provenance']
                    validate_serving(meta)
                    paraphrases.add((meta['model_id'],meta['model_revision']))
        selected = paraphrases or sources
        if len(selected)!=1:
            raise ValueError('One registered paraphrase model/revision per cohort required')
        model,revision = next(iter(selected))
        return dict(model_id=model,model_revision=revision,architecture='guarded_v1',
            source_normalization_models=[dict(model_id=m,model_revision=r) for m,r in sorted(sources)])
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
