"""Frozen Qwen inference and resumable sequential original-trajectory replay."""
from collections import defaultdict
import time
import importlib.metadata
import torch
from .contracts import (Blocked, digest, text_hash, code_hash, paths, read_json, read_jsonl,
                        write_json, write_jsonl, gate, verify_manifest)
from .representation import SampleCache, extract
from .splitting import audit_split, pilot_trajectories


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class FrozenTarget:
    def __init__(self, cfg):
        if not torch.cuda.is_available():
            raise Blocked('CUDA_GPU_NOT_AVAILABLE: full Target inference is disabled on CPU')
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.cfg = cfg['target']
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg['model_id'], revision=self.cfg['revision'], trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg['model_id'], revision=self.cfg['revision'], trust_remote_code=False,
            torch_dtype=self.dtype).to('cuda').eval()
        self.model.requires_grad_(False)
        if any(p.requires_grad for p in self.model.parameters()):
            raise Blocked('Target parameters must be frozen')

    def prepare(self, history):
        prompt = self.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        # Chat template already includes all special tokens.
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        inputs = {k: v.to(self.model.device) for k, v in inputs.items() if k in ('input_ids', 'attention_mask')}
        return prompt, inputs

    def representation(self, inputs):
        return extract(self.model, inputs)

    def generate(self, inputs):
        with torch.inference_mode():
            sequence = self.model.generate(**inputs, do_sample=False,
                max_new_tokens=self.cfg['max_new_tokens'], use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)[0]
        generated = sequence[inputs['input_ids'].shape[-1]:]
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else (eos or [self.tokenizer.eos_token_id])
        truncated = len(generated) >= self.cfg['max_new_tokens'] and (not len(generated) or int(generated[-1]) not in eos)
        return self.tokenizer.decode(generated, skip_special_tokens=True), bool(truncated), len(generated)


def collect(cfg, pilot=False, resume=False):
    data, output = paths(cfg)
    lock = read_json(data / 'manifests/revision_lock.json')
    manifest = read_json(data / 'manifests/split_manifest.json')
    verify_manifest(lock)
    rows = read_jsonl(data / 'normalized/turns.jsonl')
    audit_split(rows, manifest)
    if manifest['source_hash'] != lock['manifest_hash']:
        raise Blocked('Split/source lock mismatch')
    if not pilot:
        readiness = read_json(output/'reports/progress.json') if (output/'reports/progress.json').exists() else {}
        if readiness.get('G2',{}).get('status') != 'PASS':
            raise Blocked('Run --pilot and pass G2 before full collection')
    target = FrozenTarget(cfg)
    template = target.tokenizer.chat_template
    run = {'target': cfg['target'], 'source': lock['manifest_hash'], 'split': manifest['manifest_hash'],
           'code': code_hash(), 'template': digest(template), 'dtype': str(target.dtype),
           'versions': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'safetensors')}}
    fingerprint = digest(run)
    write_json(data / 'manifests/collection_lock.json', {**run, 'fingerprint': fingerprint}, immutable=True)
    cache = SampleCache(data)
    by_traj = defaultdict(list)
    for r in rows:
        by_traj[r['trajectory_id']].append(r)
    chosen = pilot_trajectories(rows, manifest, 20) if pilot else sorted(by_traj)

    def replay(trajectories):
        for tid in trajectories:
            history = []
            blocked = None
            for r in sorted(by_traj[tid], key=lambda r: r['turn_index']):
                base = {k: r[k] for k in ('sample_id', 'trajectory_id', 'turn_index')}
                base.update(cache_fingerprint=fingerprint, model_revision=cfg['target']['revision'],
                            tokenizer_revision=cfg['target']['revision'], representation_key=None)
                if blocked:
                    old = cache.load(r['sample_id'], fingerprint)
                    if old is not None and old.get('status') != blocked:
                        raise Blocked('Dependent cache status mismatch')
                    if old is None:
                        cache.commit({**base, 'status': blocked})
                    continue
                history.append({'role': 'user', 'content': r['user_query']})
                prompt, inputs = target.prepare(history)
                prompt_hash = text_hash(prompt)
                token_hash = digest(inputs['input_ids'].cpu().tolist())
                base.update(prompt_text=prompt, prompt_hash=prompt_hash, prompt_token_ids_hash=token_hash,
                            prompt_token_count=int(inputs['attention_mask'].sum()),
                            extraction_layer='hidden_states[-1] (returned final decoder state)',
                            template_hash=digest(template))
                old = cache.load(r['sample_id'], fingerprint, prompt_hash, token_hash)
                if old is not None:
                    if old['status'] == 'OK':
                        history.append({'role': 'assistant', 'content': old['target_response']})
                    else:
                        blocked = old['status']
                    continue
                if base['prompt_token_count'] + cfg['target']['max_new_tokens'] > cfg['target']['max_context_tokens']:
                    blocked = 'CONTEXT_OVERFLOW'
                    cache.commit({**base, 'status': blocked})
                    continue
                errors = []
                total_start = time.perf_counter()
                for attempt in range(2):
                    try:
                        sync()
                        started = time.perf_counter()
                        h, idx = target.representation(inputs)
                        sync()
                        prefill = time.perf_counter() - started
                        started = time.perf_counter()
                        response, truncated, generated_count = target.generate(inputs)
                        sync()
                        generation = time.perf_counter() - started
                        record = {**base, 'status': 'OK', 'target_response': response,
                                  'truncated': truncated, 'generated_token_count': generated_count,
                                  'extraction_token_index': idx, 'extraction_token_id': int(inputs['input_ids'][0, idx]),
                                  'prefill_seconds': prefill, 'generation_seconds': generation,
                                  'total_seconds_including_retries': time.perf_counter() - total_start,
                                  'attempt_errors': errors}
                        cache.commit(record, h)
                        history.append({'role': 'assistant', 'content': response})
                        break
                    except Exception as exc:
                        errors.append(type(exc).__name__ + ': ' + str(exc))
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    blocked = 'TECHNICAL_ERROR'
                    cache.commit({**base, 'status': blocked, 'errors': errors})

    replay(chosen)
    if pilot:
        from .checker import OfficialChecker
        from .labeling import label_one
        checker = OfficialChecker(data / 'raw/official_code')
        def positives():
            prev = {}
            count = 0
            mapping_errors = []
            for r in rows:
                if r['trajectory_id'] not in chosen:
                    continue
                rec = cache.load(r['sample_id'], fingerprint)
                label = label_one(r, rec, prev.get(r['trajectory_id']), checker)
                count += label['eligible'] and label['failure_label'] == 1
                mapping_errors.extend(c for c in label['constraint_results']
                    if c['result'] == 'UNKNOWN' and c['error'] != 'LLM_JUDGE_REQUIRED')
                prev[r['trajectory_id']] = {**r, **(rec or {})}
            return count, mapping_errors
        positive_count, mapping_errors = positives()
        if not positive_count:
            expanded = pilot_trajectories(rows, manifest, 50)
            replay([t for t in expanded if t not in chosen])
            chosen = expanded
            positive_count, mapping_errors = positives()
        selected_rows = [r for r in rows if r['trajectory_id'] in chosen]
        records = [cache.load(r['sample_id'], fingerprint) for r in selected_rows]
        # Replay a second time verifies cache reuse and history/token hash consistency.
        replay(chosen)
        generated = [r for r in records if r and r['status'] == 'OK']
        ok = (bool(generated) and not mapping_errors
              and all(r and r['status'] in {'OK','CONTEXT_OVERFLOW'} for r in records))
        gate(cfg, 'G2', 'PASS' if ok else 'BLOCKED', trajectories=len(chosen),
             positives=positive_count, label_scarcity_warning=positive_count == 0,
             checker_mapping_errors=len(mapping_errors),
             generated_turns=len(generated),
             context_overflow_turns=sum(r is not None and r['status'] == 'CONTEXT_OVERFLOW' for r in records),
             cache_resume_verified=True, reason=None if ok else 'Pilot lacks usable generations or has technical/checker errors')
    write_jsonl(data / 'representations/index.jsonl',
                [read_json(p) for p in sorted(cache.responses.glob('*.json'))])
    return {'trajectories': len(chosen), 'pilot': pilot, 'resume_requested': resume, 'fingerprint': fingerprint}
