#!/bin/bash
# Run agent multi-turn experiment
# Usage:
#   ./run_agent.sh <model> <task> [repeats] [turns] [level] [--split] [--run-name NAME]
#   ./run_agent.sh kimi-k2.5-openrouter thompson_sampling 10
#   ./run_agent.sh gemini-3.1-pro-preview collision_detection 5 10 2 --split
set -e
cd "$(dirname "$0")/.."

MODEL="${1:?Usage: $0 <model> <task> [repeats] [turns] [level] [--split] [--run-name NAME]}"
TASK="${2:?Usage: $0 <model> <task> [repeats] [turns] [level] [--split] [--run-name NAME]}"
REPEATS="${3:-10}"
TURNS="${4:-2}"
LEVEL="${5:-2}"
shift 5 2>/dev/null || true

echo "=== Agent: model=$MODEL task=$TASK repeats=$REPEATS turns=$TURNS level=$LEVEL extra=$* ==="
python3 run.py agent-multiturn \
    --model "$MODEL" \
    --task "$TASK" \
    --level "$LEVEL" \
    --turns "$TURNS" \
    --repeats "$REPEATS" \
    "$@"
