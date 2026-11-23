"""Prepare a fresh development cohort excluding every previously used source group."""
import argparse
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from src.data.canonicalize import active_setup, download_source, load_source, select_group_candidates


def excluded_groups(paths):
    groups = set()
    for path in paths:
        if path.name.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(path) as archive:
                # Read source-level artifacts without extracting the archive.
                members = [m for m in archive.getmembers() if m.isfile() and
                           m.name.endswith(('/normalized.jsonl', '/generated.jsonl',
                                            '/gold.jsonl', '/target-all.jsonl'))]
                if not members:
                    raise ValueError('No source identity artifacts in '+str(path))
                for member in members:
                    for line in archive.extractfile(member):
                        if line.strip():
                            row = json.loads(line)
                            groups.add(row['group_id'])
        else:
            for line in path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    groups.add(json.loads(line)['group_id'])
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/data.yaml')
    p.add_argument('--exclude', type=Path, action='append', required=True,
                   help='Prior archive or canonical/generated JSONL; repeat for every earlier cohort')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--count', type=int, default=100)
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--offline', action='store_true')
    a = p.parse_args()
    if not 50 <= a.count <= 100:
        p.error('Use 50-100 independent source groups per development cohort')
    if a.out_dir.exists():
        p.error('Use a new output directory')
    config = yaml.safe_load(a.config.read_text(encoding='utf-8'))
    setup = active_setup(config)
    excluded = excluded_groups(a.exclude)
    rows = []
    for source, spec in setup['sources'].items():
        raw = ROOT/config['paths']['raw']
        download_source(raw/source, spec, offline=a.offline)
        records = [r for r in load_source(raw, source, spec) if r['group_id'] not in excluded]
        selected = select_group_candidates(records, a.count, a.seed)
        if len(selected) != a.count:
            raise ValueError('Insufficient unused source groups; requested '+str(a.count))
        rows.extend(selected)
    config['holdout_coverage'] = 'coverage_v2'
    config['seed_setup']['random_seed'] = a.seed
    config['seed_setup']['candidates_per_source'] = a.count
    a.out_dir.mkdir(parents=True)
    (a.out_dir/'data.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    (a.out_dir/'canonical.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n'
                                                  for r in rows), encoding='utf-8')
    (a.out_dir/'selection.json').write_text(json.dumps(dict(
        purpose='development_only', seed=a.seed, excluded_groups=sorted(excluded),
        selected_groups=sorted(r['group_id'] for r in rows)), indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(selected=len(rows), excluded_groups=len(excluded), out=str(a.out_dir))))


if __name__ == '__main__':
    main()
