#!/bin/bash
# Generate solutions (non-agent, one-shot)
# Usage:
#   ./run_generate.sh <model> <task> [samples] [level]
#   ./run_generate.sh kimi-k2.5-openrouter thompson_sampling 10 2
#   ./run_generate.sh gpt-4.1 all 5           # all tasks
set -e
cd "$(dirname "$0")/.."

MODEL="${1:?Usage: $0 <model> <task> [samples] [level]}"
TASK="${2:?Usage: $0 <model> <task> [samples] [level]}"
SAMPLES="${3:-10}"
LEVEL="${4:-2}"

if [[ "$TASK" == "all" ]]; then
    TASK_ARG=""
else
    TASK_ARG="--tasks $TASK"
fi

echo "=== Generate: model=$MODEL task=$TASK samples=$SAMPLES level=$LEVEL ==="
python3 run.py generate-batch \
    --models "$MODEL" \
    $TASK_ARG \
    --levels "$LEVEL" \
    --samples "$SAMPLES" \
    --yes
