"""Offline regressions for the GPU operator's compatibility fixes; no model calls."""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import pytest

from test_automatic_generation import canonical, QwenTransport, server_info, MODEL, SHA
from src.data.automatic import apply_normalization, normalization_prompt, validate_scenario, SourceExcluded
from src.data.serving import server_provenance, validate_serving
from src.utils.reproducibility import content_hash


@pytest.mark.parametrize('missing', ['absent', 'null'])
def test_missing_hf_config_records_cache_without_claiming_remote_load(tmp_path, monkeypatch, missing):
    import src.data.serving as serving
    snapshot = tmp_path/'snapshots'/SHA/'config.json'
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text('{"model_type":"qwen3"}', encoding='utf-8')
    monkeypatch.setitem(sys.modules,'huggingface_hub',SimpleNamespace(
        try_to_load_from_cache=lambda *args,**kwargs:str(snapshot)))
    info = server_info()
    if missing=='absent':
        info['vllm_config']['model_config'].pop('hf_config')
    else:
        info['vllm_config']['model_config']['hf_config']=None
    meta = server_provenance(info,MODEL,SHA,'http://localhost:8000/v1')
    validate_serving(meta)
    assert meta['evidence']['loaded_config_commit'] is None
    assert meta['evidence']['revision_binding']=='explicit_engine_revisions'
    assert meta['evidence']['local_cache_corroboration']['revision']==SHA
    # A cache match cannot excuse a server-reported wrong model revision.
    info['vllm_config']['model_config']['revision']='b'*40
    with pytest.raises(ValueError):
        server_provenance(info,MODEL,SHA,'http://localhost:8000/v1')
    damaged=deepcopy(meta)
    damaged['evidence']['local_cache_corroboration']['revision']='b'*40
    damaged['evidence_hash']=content_hash(damaged['evidence'])
    with pytest.raises(ValueError):
        validate_serving(damaged)


def test_wrong_snapshot_or_missing_cache_is_not_load_evidence(tmp_path,monkeypatch):
    from src.data.serving import _cached_snapshot_evidence
    path=tmp_path/'snapshots'/('b'*40)/'config.json'
    path.parent.mkdir(parents=True)
    path.write_text('{}',encoding='utf-8')
    monkeypatch.setitem(sys.modules,'huggingface_hub',SimpleNamespace(
        try_to_load_from_cache=lambda *args,**kwargs:str(path)))
    assert _cached_snapshot_evidence(MODEL,SHA) is None
    info=server_info()
    info['vllm_config']['model_config']['hf_config']=None
    meta=server_provenance(info,MODEL,SHA,'http://localhost:8000/v1')
    validate_serving(meta)
    assert 'local_cache_corroboration' not in meta['evidence']


def test_exact_json_key_quote_is_canonicalized_with_original_preserved():
    seed=canonical()
    transport=QwenTransport()
    output=json.loads(transport.complete(normalization_prompt(seed)))
    output['scope']['data_type']['quote']='"detail": "email"'
    raw=json.dumps(output)
    result=apply_normalization(seed,raw,transport.serving_provenance)
    assert result['source_evidence']['data_type']['quote']=='email'
    assert result['source_evidence']['data_type']['raw_quote']=='"detail": "email"'
    assert result['source_normalization']['raw']==raw
    validate_scenario(result)


@pytest.mark.parametrize('quote,path', [
    ('"wrong_key": "email"','scenario.detail'),
    ('"detail": "email address"','scenario.detail'),
    ('"detail": "email", "extra": "email"','scenario.detail'),
    ('"detail": "wrong", "detail": "email"','scenario.detail'),
    ('"story": "email"','scenario.story'),
])
def test_quote_repair_does_not_invent_or_truncate_evidence(quote,path):
    seed=canonical()
    transport=QwenTransport()
    output=json.loads(transport.complete(normalization_prompt(seed)))
    output['scope']['data_type'].update(quote=quote,path=path)
    with pytest.raises(SourceExcluded):
        apply_normalization(seed,json.dumps(output),transport.serving_provenance)


def test_legacy_prompt_uses_validator_keys_and_evidence_contract():
    from src.data.generation import generation_prompt, validation_prompt
    from src.data.validators import semantic_requirements
    sample=dict(request_style='question_embedding',capability_id='CAP09',
        policy_temporal_boundary={},purpose_provenance={"origin":"synthetic_policy"})
    required,checks=semantic_requirements(sample)
    prompt=validation_prompt('template',sample,{'turns':[]})
    assert all(json.dumps(k) in prompt for k in required|set(checks))
    assert 'turn_id, quote' in prompt
    assert 'policy_conditions_expressed' in checks
    prompt=generation_prompt('template',{'shots':[]},sample)
    assert '4 to 12' in prompt
    assert 'final_request must start' in prompt
