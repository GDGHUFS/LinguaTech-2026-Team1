"""Load both inference backends and exit without creating research artifacts."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from src.contracts import config
    from src.target import FrozenTarget
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiment.yaml')
    args = parser.parse_args()
    target = FrozenTarget(config(args.config))
    try:
        print(json.dumps({'status': 'LOADED', 'tensor_parallel_size': 2,
                          'hardware': target.hardware,
                          'versions': target.runtime_versions,
                          'scope': 'Startup only; no long-context inference tested'}, indent=2))
    finally:
        target.close()


if __name__ == '__main__':
    main()
