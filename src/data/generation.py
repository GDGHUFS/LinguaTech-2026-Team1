"""[B] Qwen으로 대화를 생성하고 검증한다. 모델이 보낸 응답도 함께 저장한다."""
import json
import os
import hashlib
import urllib.request
from copy import deepcopy

GENERATION_TASK_SUFFIX = ('\nEND INPUT JSON\nGenerate ONLY the dialogue for this INPUT, not an example. '
    'Use its actual actors, information, purpose, initial rule IDs and operation IDs in the output mappings. '
    'Do not answer the source appropriateness question. Return only the requested dialogue JSON.')
VALIDATION_TASK_SUFFIX = ('\nEND INPUT JSON\nReturn only the semantic-validation JSON with all required checks '
    'and verbatim evidence. Compare this dialogue to its trace and original history; do not answer the disclosure request.')


def generation_prompt(template, examples, sample):
    """Retained legacy path: prompt and replay use identical format requirements."""
    request = ('must' if sample.get('request_style') == 'question_embedding' else 'must NOT')
    requirements = ('OUTPUT REQUIREMENTS: turns contains 4 to 12 turns with consecutive turn_id '
        '0, 1, 2, ... . The last turn has role user; final_request equals its text exactly. '
        'final_request '+request+' start with "What would happen if".')
    return (template+'\n'+requirements+'\nSEVEN SYNTHETIC EXAMPLES:\n'+
        json.dumps(examples['shots'],ensure_ascii=False)+'\nINPUT JSON:\n'+
        json.dumps(naturalization_input(sample),ensure_ascii=False)+GENERATION_TASK_SUFFIX)


def validation_prompt(template, sample, dialogue):
    from src.data.validators import semantic_requirements
    required, needed = semantic_requirements(sample)
    requirements = 'OUTPUT REQUIREMENTS: return exactly these top-level keys: '+json.dumps(sorted(required))+'. '
    requirements += ('checks contains exactly these keys: '+json.dumps(needed)+
        '. Each check requires passed, rationale, and evidence [{turn_id, quote}]; '
        'quotes must be verbatim from that turn.' if needed else 'Do not include checks.')
    return (template+'\n'+requirements+'\nINPUT JSON:\n'+
        json.dumps(dict(structured_trace=naturalization_input(sample),dialogue=dialogue),ensure_ascii=False)+
        VALIDATION_TASK_SUFFIX)

class QwenClient:
    """Qwen의 chat completions API를 호출한다. 생성과 의미 검증에 같은 client를 사용한다."""
    def __init__(self, endpoint, model, key_env='QWEN_API_KEY', revision=None):
        if 'qwen' not in model.lower():
            raise ValueError('A Qwen model ID is required')
        self.endpoint, self.model = endpoint.rstrip('/'), model
        self.key = os.environ.get(key_env)
        self.key_env = key_env
        self.revision = revision
        self.max_tokens = 1024 if self.endpoint == 'transformers://cpu' else 4096
        self.serving_provenance = {'status': 'NOT VERIFIED'}

    def _load_cpu(self):
        """Run the same prompts and validators for a small real CPU diagnostic."""
        import re
        from pathlib import Path
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.utils.reproducibility import file_hash, content_hash, seed_everything
        if not re.fullmatch(r'[0-9a-f]{40}', self.revision or ''):
            raise ValueError('Immutable CPU model revision required')
        seed_everything(42, include_torch=True)
        torch.set_num_threads(4)
        cache = Path(__file__).resolve().parents[2]/'outputs/model-cache'
        snapshot = cache/('models--'+self.model.replace('/', '--'))/'snapshots'/self.revision
        # A local path also prevents optional tokenizer heuristics from contacting the Hub.
        self.tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
        self.cpu_model = AutoModelForCausalLM.from_pretrained(self.model, revision=self.revision,
            cache_dir=str(cache), local_files_only=True, dtype=torch.float32).eval()
        evidence = dict(model=self.model, revision=self.revision, tokenizer_revision=self.revision,
            loaded_config_commit=self.cpu_model.config._commit_hash,
            runtime='transformers_cpu', torch_version=torch.__version__, seed=42,
            quantization='none', max_new_tokens=self.max_tokens, max_normalization_tokens=256,
            snapshot_hashes={p.name:file_hash(p) for p in sorted(snapshot.iterdir()) if p.is_file()})
        self.serving_provenance = dict(status='local_loaded_snapshot', model_id=self.model,
            model_revision=self.revision, endpoint=self.endpoint, evidence=evidence,
            evidence_hash=content_hash(evidence))

    def verify_revision(self):
        if self.endpoint == 'transformers://cpu':
            if not hasattr(self, 'cpu_model'):
                self._load_cpu()
            from src.data.serving import validate_serving
            validate_serving(self.serving_provenance)
            return self.serving_provenance
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
        if self.endpoint == 'transformers://cpu':
            import torch
            text = self.tokenizer.apply_chat_template([dict(role='user', content=prompt)],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = self.tokenizer(text, return_tensors='pt')
            with torch.inference_mode():
                output = self.cpu_model.generate(**inputs, do_sample=False,
                    max_new_tokens=256 if prompt.startswith('Normalize') else self.max_tokens,
                    pad_token_id=self.tokenizer.eos_token_id)
            return self.tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        payload = {'model': self.model, 'temperature': 0, 'seed':42, 'max_tokens': 4096,
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

    def complete_json(self, prompt, schema):
        """Constrained finite-language JSON on CPU; native JSON Schema on vLLM."""
        before = self.verify_revision()
        if self.endpoint == 'transformers://cpu':
            import torch
            choices = schema.get('properties',{}).get('text',{}).get('enum')
            if not choices or set(schema.get('properties',{})) != {'text'}:
                raise ValueError('CPU structured decoding requires a finite text enum')
            responses = [json.dumps(dict(text=text),ensure_ascii=False,separators=(',',':')) for text in choices]
            sequences = [self.tokenizer.encode(text,add_special_tokens=False) for text in responses]
            chat = self.tokenizer.apply_chat_template([dict(role='user',content=prompt)],
                tokenize=False,add_generation_prompt=True,enable_thinking=False)
            inputs = self.tokenizer(chat,return_tensors='pt')
            length = inputs['input_ids'].shape[1]
            eos = self.tokenizer.eos_token_id
            def allowed(batch_id, ids):
                prefix = ids[length:].tolist()
                result = {seq[len(prefix)] if len(prefix)<len(seq) else eos for seq in sequences
                    if seq[:len(prefix)] == prefix}
                if not result:
                    raise ValueError('No valid structured-decoding continuation')
                return sorted(result)
            with torch.inference_mode():
                output = self.cpu_model.generate(**inputs,do_sample=False,
                    max_new_tokens=min(self.max_tokens,max(map(len,sequences))+1),
                    prefix_allowed_tokens_fn=allowed,pad_token_id=eos)
            return self.tokenizer.decode(output[0,length:],skip_special_tokens=True).strip()
        payload = dict(model=self.model,temperature=0,seed=42,max_tokens=self.max_tokens,
            messages=[dict(role='user',content=prompt)],response_format=dict(type='json_schema',
                json_schema=dict(name='bounded_policy_paraphrase',strict=True,schema=schema)))
        headers = {'Content-Type':'application/json'}
        if self.key:
            headers['Authorization'] = 'Bearer '+self.key
        request = urllib.request.Request(self.endpoint+'/chat/completions',
            data=json.dumps(payload).encode(),headers=headers)
        with urllib.request.urlopen(request,timeout=180) as response:
            envelope = json.load(response)
        if self.verify_revision()!=before:
            raise ValueError('Serving revision changed')
        return envelope['choices'][0]['message']['content']


def validate_controlled_family(family):
    """전달된 family를 직접 갱신해 base와 INVARIANT 항목의 판단·상태·요청을 비교한다.

    base가 없거나 oracle 검증이 미완료면 pending으로 둔다. 다른 관계 유형에 대해 판단 변경을 강제하지는 않는다."""
    base = next((r for r in family if r.get('variant_id') == 'base'), None)
    from src.data.validators import validate_capability
    guarded = all(r.get('generation_architecture')=='guarded_v1' for r in family)
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
        elif not (record.get('oracle_verified') is True and (base is None or base.get('oracle_verified') is True) if guarded else ready):
            record['oracle_relation_status'] = 'pending'
        elif base and record.get('expected_gold_relation') == 'INVARIANT' and any(record[k] != base[k] for k in ('gold_decision','gold_state','request')):
            record['oracle_relation_status'] = 'failed'
            record['validation_status'] = 'rejected'
            record['errors'].append('controlled_invariance_violation')
        else:
            record['oracle_relation_status'] = 'passed'


def naturalization_input(sample):
    # Keep task facts and the complete policy trace; audit hashes and duplicate story
    # copies belong in the saved record, not in the model's transformation input.
    keys = ('sample_id','capability_id','request_style','initial_rules','operation_trace','request',
            'source_facts','policy_temporal_boundary','purpose_provenance')
    payload = {k:deepcopy(sample[k]) for k in keys if k in sample}
    if sample.get('source_normalization'):
        from src.data.automatic import source_scenario
        payload['source_scenario'] = source_scenario(sample)
    elif sample.get('policy_context'):
        payload['policy_context'] = deepcopy(sample['policy_context'])
    return payload


def execute(sample, client, prompt_dir):
    """Record any item-level failure, including structure validation errors."""
    from src.data.batch import warning
    try:
        if sample.get('generation_architecture') == 'guarded_v1':
            from src.data.guarded_dialogue import execute as guarded_execute
            return guarded_execute(sample,client,sample.get('paraphrase_attempt_limit',1))
        record = _execute(sample, client, prompt_dir)
    except Exception as exc:
        record = dict(sample, structured_trace=deepcopy(sample), turns=None,
                      generation_status='failed', validation_status='failed',
                      generation_raw=None, validation_raw=None,
                      errors=['execution:'+type(exc).__name__])
    if record.get('errors'):
        warning('generation', sample.get('sample_id'), record['errors'])
    from src.data.batch import diagnostic
    if any(code.startswith('oracle:') for code in record.get('errors', [])):
        stage, status = 'dialogue_oracle', 'failed'
    elif record.get('validation_status') == 'semantic_pass_oracle_verified':
        stage, status = 'semantic_validation', 'passed'
    elif record.get('generation_status') == 'generated':
        stage, status = 'semantic_validation', 'failed'
    elif record.get('generation_status') == 'invalid_structure':
        stage, status = 'rule_engine', 'failed'
    else:
        stage, status = 'generation', 'failed'
    record['diagnostics'] = []
    if record.get('structured_oracle'):
        record['diagnostics'].append(diagnostic(record, 'rule_engine', 'passed'))
    if record.get('generation_status') == 'generated':
        record['diagnostics'].append(diagnostic(record, 'generation', 'passed'))
    record['diagnostics'] += [diagnostic(record, stage, status, code) for code in record.get('errors', []) or [None]]
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
    record['execution'] = {'model_id': client.model, 'endpoint': client.endpoint, 'temperature': 0,
        'max_tokens': getattr(client, 'max_tokens', 4096), 'seed':42, 'serving_provenance':deepcopy(getattr(client, 'serving_provenance', {}))}
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
        prompt = generation_prompt(template, examples, sample)
        record['execution']['generation_prompt_hash'] = hashlib.sha256(prompt.encode()).hexdigest()
        raw = client.complete(prompt)
        record['generation_raw'] = raw
        if getattr(client, 'stage_callback', None):
            client.stage_callback(sample, 'generation', raw)
        from src.data.batch import model_json
        output = model_json(raw)
        record['errors'] = validate_dialogue(output, sample)
        if record['errors']:
            record['generation_status'] = 'invalid_dialogue'
            return record
        record.update(turns=output['turns'], dialogue=output, final_request=output['final_request'],
                      request_turn_id=output['turns'][-1]['turn_id'], generation_status='generated')
        stage = 'validation'
        template = (prompt_dir / 'validation.txt').read_text(encoding='utf-8-sig')
        prompt = validation_prompt(template, sample, output)
        record['execution']['validation_prompt_hash'] = hashlib.sha256(prompt.encode()).hexdigest()
        raw = client.complete(prompt)
        record['validation_raw'] = raw
        if getattr(client, 'stage_callback', None):
            client.stage_callback(sample, 'semantic_validation', raw)
        if sample.get('source_normalization'):
            record['execution']['serving_provenance'] = deepcopy(client.serving_provenance)
        result = model_json(raw)
        record['semantic_validation'] = result
        record['errors'] = validate_semantics(result, sample, output)
        if record['errors']:
            record['validation_status'] = 'rejected'
        else:
            stage = 'oracle'
            annotated = annotate_dialogue(sample, output)
            if any(annotated[k] != structured[k] for k in ('gold_decision','gold_state')):
                raise ValueError('Structured/dialogue oracle decision or state mismatch')
            for key in ('gold_decision', 'gold_state', 'supporting_rule_ids', 'supporting_turn_ids',
                        'evidence_trace', 'ambiguity_flags', 'annotation_status', 'engine_version',
                        'initial_rules', 'operation_trace', 'versions', 'oracle_verified',
                        'oracle_relation_status', 'relevance_mask'):
                record[key] = annotated[key]
            record['validation_status'] = 'semantic_pass_oracle_verified'
    except Exception as exc:
        # 오류 메시지에 인증 정보가 들어갈 수 있어 오류 종류만 남긴다.
        record['errors'] = [f'{stage}:json_parse_error' if isinstance(exc, json.JSONDecodeError) else f'{stage}:{type(exc).__name__}']
        if stage == 'generation':
            record['generation_status'] = 'failed'
        else:
            record['validation_status'] = 'failed'
    return record
