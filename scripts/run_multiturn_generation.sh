#!/usr/bin/env bash
# Run from the repository root, with the pinned Qwen vLLM server already running.
# Usage: bash scripts/run_multiturn_generation.sh smoke
#        bash scripts/run_multiturn_generation.sh full100
set -euo pipefail

MODE="${1:-smoke}"
case "$MODE" in
  smoke) N_SOURCES=3; EXECUTION_MODE=pilot ;;
  full100) N_SOURCES=100; EXECUTION_MODE=final ;;
  *) echo 'Usage: bash scripts/run_multiturn_generation.sh {smoke|full100}' >&2; exit 2 ;;
esac

INPUT="${INPUT:-runs/l40s_20261121_053150/canonical.jsonl}"
GEN_REV="${GEN_REV:-b968826d9c46dd6066d109eabc6255188de91218}"
OUT="${OUT:-runs/multiturn_${MODE}_$(date +%Y%m%d_%H%M%S)}"
: "${QWEN_API_KEY:?Set QWEN_API_KEY to the running vLLM server API key}"
test -f "$INPUT" || { echo "Missing canonical input: $INPUT" >&2; exit 1; }
mkdir -p "$OUT"

# Omit --paraphrase-budget to attempt all eligible policy turns in every dialogue.
# Failed batches fall back to verified templates; inspect coverage and fallback counts.
python scripts/02_prepare.py \
  --input "$INPUT" \
  --endpoint http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3-8B \
  --revision "$GEN_REV" \
  --limit-seeds "$N_SOURCES" \
  --paraphrase-scope all_policy_turns \
  --execution-mode "$EXECUTION_MODE" \
  --output "$OUT/generated.jsonl" \
  --accepted-output "$OUT/gold.jsonl" \
  --normalized-output "$OUT/normalized.jsonl" \
  --audit "$OUT/generation-audit.csv" \
  2>&1 | tee "$OUT/generation.log"
