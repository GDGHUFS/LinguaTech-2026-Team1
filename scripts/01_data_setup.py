"""[B] 고정 revision의 원본을 받고 독립적으로 seed pilot을 실행한다."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from src.data.canonicalize import download_source, run_pilot

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/data.yaml')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    setup = config['seed_setup']
    raw = ROOT / config['paths']['raw']
    for source, spec in setup['sources'].items():
        download_source(raw / source, spec, offline=args.offline)
    result = run_pilot(raw, ROOT / config['paths']['annotated'] / 'canonical_seed.jsonl', ROOT / setup['inventory'], setup)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
