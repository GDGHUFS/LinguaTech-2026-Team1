"""[B] Qwen으로 대화를 생성하고 검증한다. 모델이 보낸 응답도 함께 저장한다."""
import json
import os
import hashlib
import urllib.request
from copy import deepcopy

class QwenClient:
    """Qwen의 chat completions API를 호출한다. 생성과 의미 검증에 같은 client를 사용한다."""
    def __init__(self, endpoint, model, key_env='QWEN_API_KEY', revision=None):
        if 'qwen' not in model.lower():
            raise ValueError('A Qwen model ID is required')
        self.endpoint, self.model = endpoint.rstrip('/'), model
        self.key = os.environ.get(key_env)
        self.key_env = key_env
        self.revision = revision
        self.serving_provenance = {'status': 'NOT VERIFIED'}

    def verify_revision(self):
        from src.data.serving import server_provenance
        from urllib.parse import urlsplit, urlunsplit
        url = urlsplit(self.endpoint)
        prefix = url.path.removesuffix('/v1').rstrip('/')
        address = urlunsplit((url.scheme, url.netloc, prefix+'/server_info', 'config_format=json', ''))
        headers = {'Authorization': f'Bearer {self.key}'} if self.key else {}
        try:
            with urllib.request.urlopen(urllib.request.Request(address, headers=headers), timeout=30) as response:
                info = json.load(response)
            self.serving_provenance = server_provenance(info, self.model, self.revision, self.endpoint)
        except Exception as exc:
            self.serving_provenance = {'status': 'NOT VERIFIED'}
            raise ValueError('Generation revision NOT VERIFIED: server_info unavailable or mismatched') from exc
        return self.serving_provenance

    def complete(self, prompt):
        before = self.verify_revision()
        payload = {'model': self.model, 'temperature': 0, 'max_tokens': 4096,
                   'messages': [{'role': 'user', 'content': prompt}],
                   'response_format': {'type': 'json_object'}}
        headers = {'Content-Type': 'application/json'}
        if self.key:
            headers['Authorization'] = f'Bearer {self.key}'
        request = urllib.request.Request(self.endpoint + '/chat/completions', data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=180) as response:
            envelope = json.load(response)
        if self.verify_revision() != before:
            raise ValueError('Generation serving revision changed during request')
        return envelope['choices'][0]['message']['content']


def validate_controlled_family(family):
    """전달된 family를 직접 갱신해 base와 INVARIANT 항목의 판단·상태·요청을 비교한다.

    base가 없거나 oracle 검증이 미완료면 pending으로 둔다. 다른 관계 유형에 대해 판단 변경을 강제하지는 않는다."""
    base = next((r for r in family if r.get('variant_id') == 'base'), None)
    from src.data.validators import validate_capability
    ready = base is not None and all(r.get('oracle_verified') is True for r in family)
    for record in family:
        trace = record.get('structured_trace') or record
        errors = validate_capability(trace)
        if base and record.get('capability_id'):
            base_trace = base.get('structured_trace') or base
            cap = record['capability_id']
            initial = deepcopy(trace['initial_rules'])
            expected_initial = deepcopy(base_trace['initial_rules'])
            if cap == 'CAP01':
                expected_initial = []
                if trace['operation_trace'] and trace['operation_trace'][0].get('rule') not in base_trace['initial_rules']:
                    errors.append('cap01_added_rule_differs_from_base')
            elif cap == 'CAP04':
                for rule in expected_initial:
                    rule['authorization'] = 'DENY'
            if initial != expected_initial:
                errors.append('controlled_initial_policy_mismatch')
            request = deepcopy(base_trace['request'])
            if cap == 'CAP04':
                request['channel'] = 'secure_portal'
            elif cap == 'CAP09':
                request['recipient'] = None
            if trace['request'] != request:
                errors.append('controlled_request_mismatch')
        if base and any(record.get(k) != base.get(k) for k in ('group_id','variant_group_id','seed_id','source_provenance')):
            errors.append('controlled_family_identity_mismatch')
        if errors:
            record['oracle_relation_status'] = 'failed'
            record['validation_status'] = 'rejected'
            record.setdefault('errors', []).extend(errors)
        elif not ready:
            record['oracle_relation_status'] = 'pending'
        elif record.get('expected_gold_relation') == 'INVARIANT' and any(record[k] != base[k] for k in ('gold_decision','gold_state','request')):
            record['oracle_relation_status'] = 'failed'
            record['validation_status'] = 'rejected'
            record['errors'].append('controlled_invariance_violation')
        else:
            record['oracle_relation_status'] = 'passed'


def naturalization_input(sample):
    payload = deepcopy(sample)
    if sample.get('source_normalization'):
        from src.data.automatic import source_scenario
        payload['source_scenario'] = source_scenario(sample)
        for key in ('source_record', 'source_appropriateness_label'):
            payload.pop(key, None)
    return payload


def execute(sample, client, prompt_dir):
    """Record any item-level failure, including structure validation errors."""
    from src.data.batch import warning
    try:
        record = _execute(sample, client, prompt_dir)
    except Exception as exc:
        record = dict(sample, structured_trace=deepcopy(sample), turns=None,
                      generation_status='failed', validation_status='failed',
                      generation_raw=None, validation_raw=None,
                      errors=['execution:'+type(exc).__name__])
    if record.get('errors'):
        warning('generation', sample.get('sample_id'), record['errors'])
    return record


def _execute(sample, client, prompt_dir):
    """구조 oracle을 확인한 뒤 대화 생성과 의미 검증을 요청하고 결과를 반환한다.

    client가 없으면 구조 검사까지만 수행한다. 생성·검증 중 오류는 상태와 오류 종류로 기록하며 사람의 검수 승인을 대신하지 않는다."""
    from src.data.validators import validate_structure, validate_dialogue, validate_semantics
    from src.data.oracle_adapter import evaluate_structured, annotate_dialogue
    record = dict(sample, turns=None, generation_status='pending_qwen', validation_status='not_run',
                  errors=validate_structure(sample), generation_raw=None, validation_raw=None)
    record['structured_trace'] = deepcopy(sample)
    if record['errors']:
        record['generation_status'] = 'invalid_structure'
        return record
    try:
        structured = evaluate_structured(sample)
        if structured['annotation_status'] != 'VALID':
            raise ValueError('Unresolved policy conflict')
        record['structured_oracle'] = structured
    except (ValueError, TypeError, KeyError) as exc:
        record.update(generation_status='invalid_structure', errors=[f'oracle:{type(exc).__name__}'])
        return record
    if client is None:
        return record
    record['execution'] = {'model_id': client.model, 'endpoint': client.endpoint, 'temperature': 0, 'max_tokens': 4096}
    stage = 'generation'
    try:
        template = (prompt_dir / 'generation.txt').read_text(encoding='utf-8-sig')
        from src.data.fewshots import load_fewshots
        from src.data.splitting import DEFAULT_TEMPLATES
        examples = load_fewshots(prompt_dir / 'naturalization_shots.json', DEFAULT_TEMPLATES)
        record['execution']['shot_count'] = 7
        record['execution']['fewshot_template_families'] = [s['template_family_id'] for s in examples['shots']]
        record['execution']['shots_hash'] = hashlib.sha256(json.dumps(examples, sort_keys=True).encode()).hexdigest()
        generation_input = naturalization_input(sample)
        prompt = template + '\nSEVEN SYNTHETIC EXAMPLES:\n' + json.dumps(examples['shots'], ensure_ascii=False) + '\nINPUT JSON:\n' + json.dumps(generation_input, ensure_ascii=False)
        record['execution']['generation_prompt_hash'] = hashlib.sha256(prompt.encode()).hexdigest()
        raw = client.complete(prompt)
        record['generation_raw'] = raw
        output = json.loads(raw)
        record['errors'] = validate_dialogue(output, sample)
        if record['errors']:
            record['generation_status'] = 'invalid_dialogue'
            return record
        record.update(turns=output['turns'], dialogue=output, final_request=output['final_request'],
                      request_turn_id=output['turns'][-1]['turn_id'], generation_status='generated')
        stage = 'validation'
        template = (prompt_dir / 'validation.txt').read_text(encoding='utf-8-sig')
        prompt = template + '\nINPUT JSON:\n' + json.dumps({'structured_trace': generation_input, 'dialogue': output}, ensure_ascii=False)
        record['execution']['validation_prompt_hash'] = hashlib.sha256(prompt.encode()).hexdigest()
        raw = client.complete(prompt)
        record['validation_raw'] = raw
        if sample.get('source_normalization'):
            record['execution']['serving_provenance'] = deepcopy(client.serving_provenance)
        result = json.loads(raw)
        record['semantic_validation'] = result
        record['errors'] = validate_semantics(result, sample, output)
        if record['errors']:
            record['validation_status'] = 'rejected'
        else:
            annotated = annotate_dialogue(sample, output)
            for key in ('gold_decision', 'gold_state', 'supporting_rule_ids', 'supporting_turn_ids',
                        'evidence_trace', 'ambiguity_flags', 'annotation_status', 'engine_version',
                        'initial_rules', 'operation_trace', 'versions', 'oracle_verified',
                        'oracle_relation_status', 'relevance_mask'):
                record[key] = annotated[key]
            record['validation_status'] = 'semantic_pass_oracle_verified'
    except Exception as exc:
        # 오류 메시지에 인증 정보가 들어갈 수 있어 오류 종류만 남긴다.
        record['errors'] = [f'{stage}:{type(exc).__name__}']
        if stage == 'generation':
            record['generation_status'] = 'failed'
        else:
            record['validation_status'] = 'failed'
    return record
