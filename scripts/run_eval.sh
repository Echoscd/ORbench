#!/bin/bash
# Evaluate a run directory
# Usage:
#   ./run_eval.sh <run_name> [arch]
#   ./run_eval.sh kimi-k2.5-openrouter_l2_20260320
#   ./run_eval.sh kimi-k2.5-openrouter_l2_agent_mt_20260320 sm_89
set -e
cd "$(dirname "$0")/.."

RUN="${1:?Usage: $0 <run_name> [arch]}"
ARCH="${2:-}"

EXTRA_ARGS=""
if [[ -n "$ARCH" ]]; then
    EXTRA_ARGS="--arch $ARCH"
fi

echo "=== Eval: run=$RUN ==="
python3 run.py eval --run "$RUN" $EXTRA_ARGS
