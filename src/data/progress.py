"""Durable item journal with input-bound restart and torn-tail recovery."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def signature(arguments, paths):
    files = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
             for p in paths if p is not None}
    payload = json.dumps(dict(arguments=arguments, files=files), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


class CachedFailure(Exception):
    def __init__(self, event):
        super().__init__(event['error_type'])
        self.audit = event.get('audit', {})


class Journal:
    def __init__(self, path, fingerprint, resume=False):
        self.path = Path(path)
        self.events = {}
        header = dict(schema='item-progress-1', signature=fingerprint)
        if self.path.exists():
            if not resume:
                raise ValueError('Progress exists; use --resume')
            # Only an unfinished final write is recoverable. Never ignore a corrupt full line.
            with self.path.open('r+b') as stream:
                first = stream.readline()
                if json.loads(first) != header:
                    raise ValueError('Resume inputs/configuration/code changed; use new output paths')
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    if not line.endswith(b'\n'):
                        stream.truncate(offset)
                        stream.flush()
                        os.fsync(stream.fileno())
                        break
                    event = json.loads(line)
                    if event['key'] in self.events:
                        raise ValueError('Duplicate progress key')
                    self.events[event['key']] = event
        else:
            atomic_text(self.path, json.dumps(header)+'\n')

    def run(self, key, operation, *, cache_failures=True):
        event = self.events.get(key)
        if event is not None:
            if 'error_type' in event:
                raise CachedFailure(event)
            return deepcopy(event['value'])
        try:
            value = operation()
        except Exception as exc:
            if cache_failures:
                self._append(dict(key=key, error_type=type(exc).__name__, audit=getattr(exc, 'audit', {})))
            raise
        self._append(dict(key=key, value=value))
        return value

    def _append(self, event):
        append_json(self.path, event)
        self.events[event['key']] = deepcopy(event)
