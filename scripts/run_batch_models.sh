#!/bin/bash
# Run agent experiments across multiple models for the same task
# Usage:
#   ./run_batch_models.sh <task> [repeats] [turns] [level] [models...]
#   ./run_batch_models.sh thompson_sampling 10 2 2 kimi-k2.5-openrouter gemini-3.1-pro-preview gpt-4.1
#   ./run_batch_models.sh collision_detection   # uses default models list below
set -e
cd "$(dirname "$0")/.."

TASK="${1:?Usage: $0 <task> [repeats] [turns] [level] [models...]}"
REPEATS="${2:-10}"
TURNS="${3:-2}"
LEVEL="${4:-2}"

# Default model list if none given
shift 4 2>/dev/null || true
if [[ $# -eq 0 ]]; then
    MODELS=(kimi-k2.5-openrouter gemini-3.1-pro-preview gpt-4.1 claude-sonnet-4 deepseek-r1)
else
    MODELS=("$@")
fi

echo "=== Batch agent: task=$TASK repeats=$REPEATS turns=$TURNS level=$LEVEL ==="
echo "=== Models: ${MODELS[*]} ==="

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo ">>> Starting: $MODEL x $TASK"
    python3 run.py agent-multiturn \
        --model "$MODEL" \
        --task "$TASK" \
        --level "$LEVEL" \
        --turns "$TURNS" \
        --repeats "$REPEATS" || echo "[WARN] $MODEL failed, continuing..."
    echo "<<< Done: $MODEL"
done

echo "=== All models complete ==="
