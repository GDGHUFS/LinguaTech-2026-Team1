import importlib.util
import io
import json
from pathlib import Path
import tarfile


def test_prior_archive_and_canonical_groups_are_all_excluded(tmp_path):
    path = Path(__file__).resolve().parents[1]/'scripts/10_prepare_expansion.py'
    spec = importlib.util.spec_from_file_location('expansion_selection', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive_path = tmp_path/'prior.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for name, groups in [('normalized.jsonl',['a','b']),('gold.jsonl',['b','c'])]:
            data = ''.join(json.dumps({'group_id':g})+'\n' for g in groups).encode()
            member = tarfile.TarInfo('runs/prior/'+name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    canonical = tmp_path/'canonical.jsonl'
    canonical.write_text('{"group_id":"d"}\n', encoding='utf-8')
    assert module.excluded_groups([archive_path, canonical]) == {'a','b','c','d'}
