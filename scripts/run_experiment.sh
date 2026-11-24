#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
config="${1:-configs/experiment.yaml}"

python scripts/01_audit.py --config "$config" --fetch
python scripts/02_split.py --config "$config"
python scripts/03_collect.py --config "$config" --pilot --resume
python scripts/03_collect.py --config "$config" --resume
python scripts/04_label.py --config "$config"
python scripts/05_train.py --config "$config" --resume
python scripts/06_evaluate.py --config "$config" --resume
