"""Fetch only the pinned official evidence listed in the tracked source lock."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import urllib.request
from .contracts import ROOT, Blocked, file_hash, paths, read_json, write_json


def fetch_sources(cfg):
    data, _ = paths(cfg)
    raw = data / 'raw'
    manifest = read_json(ROOT / 'configs/sources.json')
    def fetch(item):
        relative = item['path']
        if relative.startswith('models/Qwen3-4B-Instruct-2507/'):
            relative = 'target_metadata/' + relative.split('/')[-1]
        dest = (raw / relative).resolve()
        if not dest.is_relative_to(raw.resolve()):
            raise Blocked('Invalid official source path')
        if dest.exists() and file_hash(dest) == item['sha256']:
            return
        request = urllib.request.Request(item['url'], headers={'User-Agent':'ConShift-v3-source-audit'})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                content = response.read()
        except Exception as exc:
            raise Blocked(f'Official evidence download failed: {item["url"]}: {exc}') from exc
        if hashlib.sha256(content).hexdigest() != item['sha256']:
            raise Blocked('Downloaded official evidence changed: ' + item['url'])
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix(dest.suffix + '.tmp')
        temp.write_bytes(content)
        temp.replace(dest)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(fetch, manifest['files']))
    write_json(raw / 'source_manifest.json', manifest)
