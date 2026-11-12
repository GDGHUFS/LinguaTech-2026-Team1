"""[B] Qwen 호출(generation/semantic validation)과 원시 응답을 보존한다."""
import json
import os
import hashlib
import urllib.request

class QwenClient:
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


def execute(sample, client, prompt_dir):
    from src.data.validators import validate_structure, validate_dialogue, validate_semantics
    record = dict(sample, turns=None, generation_status='pending_qwen', validation_status='not_run',
                  errors=validate_structure(sample), generation_raw=None, validation_raw=None)
    if record['errors']:
        record['generation_status'] = 'invalid_structure'
        return record
    if client is None:
        return record
    record['execution'] = {'model_id': client.model, 'endpoint': client.endpoint, 'temperature': 0, 'max_tokens': 4096}
    stage = 'generation'
    try:
        template = (prompt_dir / 'generation.txt').read_text(encoding='utf-8-sig')
        prompt = template + '\nINPUT JSON:\n' + json.dumps(sample, ensure_ascii=False)
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
        record['errors'] = validate_semantics(result)
        record['validation_status'] = 'rejected' if record['errors'] else 'semantic_pass_oracle_pending'
    except Exception as exc:
        # 오류 종류(error type)만 기록하여 인증 정보가 응답에 유출되지 않게 한다.
        record['errors'] = [f'{stage}:{type(exc).__name__}']
        if stage == 'generation':
            record['generation_status'] = 'failed'
        else:
            record['validation_status'] = 'failed'
    return record
