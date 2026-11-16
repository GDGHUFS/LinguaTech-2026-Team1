"""[B] Qwen으로 대화를 생성하고 검증한다. 모델이 보낸 응답도 함께 저장한다."""
import json
import os
import hashlib
import urllib.request

class QwenClient:
    """Qwen의 chat completions API를 호출한다. 생성과 의미 검증에 같은 client를 사용한다."""
    def __init__(self, endpoint, model, key_env='QWEN_API_KEY'):
        if 'qwen' not in model.lower():
            raise ValueError('A Qwen model ID is required')
        self.endpoint, self.model = endpoint.rstrip('/'), model
        self.key = os.environ.get(key_env)
        self.key_env = key_env

    def complete(self, prompt):
        payload = {'model': self.model, 'temperature': 0, 'max_tokens': 4096,
                   'messages': [{'role': 'user', 'content': prompt}],
                   'response_format': {'type': 'json_object'}}
        headers = {'Content-Type': 'application/json'}
        if self.key:
            headers['Authorization'] = f'Bearer {self.key}'
        request = urllib.request.Request(self.endpoint + '/chat/completions', data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=180) as response:
            envelope = json.load(response)
        return envelope['choices'][0]['message']['content']


def validate_controlled_family(family):
    """전달된 family를 직접 갱신해 base와 INVARIANT 항목의 판단·상태·요청을 비교한다.

    base가 없거나 oracle 검증이 미완료면 pending으로 둔다. 다른 관계 유형에 대해 판단 변경을 강제하지는 않는다."""
    base = next((r for r in family if r.get('variant_id') == 'base'), None)
    ready = base is not None and all(r.get('oracle_verified') is True for r in family)
    for record in family:
        if not ready:
            record['oracle_relation_status'] = 'pending'
        elif record.get('expected_gold_relation') == 'INVARIANT' and any(record[k] != base[k] for k in ('gold_decision','gold_state','request')):
            record['oracle_relation_status'] = 'failed'
            record['validation_status'] = 'rejected'
            record['errors'].append('controlled_invariance_violation')
        else:
            record['oracle_relation_status'] = 'passed'


def execute(sample, client, prompt_dir):
    """구조 oracle을 확인한 뒤 대화 생성과 의미 검증을 요청하고 결과를 반환한다.

    client가 없으면 구조 검사까지만 수행한다. 생성·검증 중 오류는 상태와 오류 종류로 기록하며 사람의 검수 승인을 대신하지 않는다."""
    from src.data.validators import validate_structure, validate_dialogue, validate_semantics
    from src.data.oracle_adapter import evaluate_structured, annotate_dialogue
    record = dict(sample, turns=None, generation_status='pending_qwen', validation_status='not_run',
                  errors=validate_structure(sample), generation_raw=None, validation_raw=None)
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
        prompt = template + '\nSEVEN SYNTHETIC EXAMPLES:\n' + json.dumps(examples['shots'], ensure_ascii=False) + '\nINPUT JSON:\n' + json.dumps(sample, ensure_ascii=False)
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
        prompt = template + '\nINPUT JSON:\n' + json.dumps({'structured_trace': sample, 'dialogue': output}, ensure_ascii=False)
        record['execution']['validation_prompt_hash'] = hashlib.sha256(prompt.encode()).hexdigest()
        raw = client.complete(prompt)
        record['validation_raw'] = raw
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
