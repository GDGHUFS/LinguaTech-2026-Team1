"""[B] 지정한 revision의 원본을 받고 seed 후보를 추출한다."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from src.data.canonicalize import active_setup, download_source, run_pilot

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/data.yaml')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--canonical-out', type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    setup = active_setup(config)
    raw = ROOT / config['paths']['raw']
    inventory = args.out or ROOT/setup['inventory']
    canonical = args.canonical_out or ROOT/config['paths']['annotated']/config.get('canonical_filename', 'canonical_seed.jsonl')
    if inventory.exists() or canonical.exists():
        parser.error('Existing seed artifacts are preserved; provide new --out and --canonical-out')
    for source, spec in setup['sources'].items():
        download_source(raw / source, spec, offline=args.offline)
    result = run_pilot(raw, canonical, inventory, setup)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
