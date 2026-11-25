"""Atomic per-sample tensor cache paired with JSON provenance."""
from pathlib import Path
import os
import torch
from safetensors.torch import load_file, save_file
from .contracts import Blocked, digest, file_hash, read_json, write_json
from .features import valid_vector


def last_nonpadding(attention_mask):
    positions = torch.arange(attention_mask.shape[-1], device=attention_mask.device)
    indices = positions.expand_as(attention_mask).masked_fill(attention_mask == 0, -1).max(dim=-1).values
    if bool((indices < 0).any()):
        raise ValueError('Empty prompt')
    return indices


def extract(model, inputs):
    model.eval()
    with torch.inference_mode():
        # FrozenTarget passes the bare Qwen3 backbone: no LM head or all-layer tuple.
        result = model(**inputs, output_hidden_states=False, use_cache=False, return_dict=True)
        index = last_nonpadding(inputs['attention_mask'])
        h = result.last_hidden_state[torch.arange(index.shape[0], device=index.device), index]
        if h.shape[-1] != model.config.hidden_size or not valid_vector(h):
            raise Blocked('Invalid returned final-layer hidden state')
    return h[0].float().cpu().contiguous(), int(index[0])


class SampleCache:
    def __init__(self, data):
        self.responses = Path(data) / 'responses'
        self.vectors = Path(data) / 'representations'
        self.responses.mkdir(parents=True, exist_ok=True)
        self.vectors.mkdir(parents=True, exist_ok=True)

    def load(self, sample_id, fingerprint, prompt_hash=None, token_hash=None):
        p = self.responses / (sample_id + '.json')
        if not p.exists():
            return None
        record = read_json(p)
        if record.get('record_hash') != digest({k:v for k,v in record.items() if k != 'record_hash'}):
            raise Blocked('Response/provenance cache hash mismatch')
        if record['cache_fingerprint'] != fingerprint:
            raise Blocked('Cache configuration/source/code version changed')
        if prompt_hash is not None and record.get('prompt_hash') != prompt_hash:
            raise Blocked('Cache prompt mismatch')
        if token_hash is not None and record.get('prompt_token_ids_hash') != token_hash:
            raise Blocked('Cache prompt token IDs mismatch')
        if record.get('representation_key'):
            tensor_path = self.vectors / record['representation_key']
            if not tensor_path.exists() or file_hash(tensor_path) != record['representation_sha256']:
                raise Blocked('Incomplete or corrupt representation cache')
            h = load_file(tensor_path)['hidden']
            if h.dtype != torch.float32 or h.ndim != 1 or h.numel() != record['hidden_dim'] or not valid_vector(h):
                raise Blocked('Invalid cached hidden vector')
        return record

    def commit(self, record, vector=None):
        sid = record['sample_id']
        if vector is not None:
            if not valid_vector(vector) or vector.ndim != 1:
                raise Blocked('Invalid representation commit')
            key = sid + '.safetensors'
            dest = self.vectors / key
            temp = self.vectors / (key + '.tmp')
            save_file({'hidden': vector.detach().float().cpu().contiguous()}, str(temp))
            os.replace(temp, dest)
            record.update(representation_key=key, representation_sha256=file_hash(dest), hidden_dim=vector.numel())
        # JSON is the commit marker; orphan tensors after interruption are overwritten.
        record['record_hash'] = digest({k:v for k,v in record.items() if k != 'record_hash'})
        write_json(self.responses / (sid + '.json'), record)

    def vector(self, key):
        return load_file(self.vectors / key)['hidden']
