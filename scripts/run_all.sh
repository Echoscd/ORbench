#!/bin/bash
# Run a full pipeline: generate (or agent) + eval + analyze for one model x task
# Usage:
#   ./run_all.sh <model> <task> [mode] [samples/repeats] [level]
#   ./run_all.sh kimi-k2.5-openrouter thompson_sampling agent 10
#   ./run_all.sh gpt-4.1 regex_match generate 10 2
set -e
cd "$(dirname "$0")/.."

MODEL="${1:?Usage: $0 <model> <task> [mode:generate|agent] [samples/repeats] [level]}"
TASK="${2:?Usage: $0 <model> <task> [mode:generate|agent] [samples/repeats] [level]}"
MODE="${3:-agent}"
COUNT="${4:-10}"
LEVEL="${5:-2}"

DATE=$(date +%Y%m%d)

if [[ "$MODE" == "agent" ]]; then
    echo "=== [1/3] Agent multi-turn: model=$MODEL task=$TASK repeats=$COUNT level=$LEVEL ==="
    python3 run.py agent-multiturn \
        --model "$MODEL" \
        --task "$TASK" \
        --level "$LEVEL" \
        --repeats "$COUNT"

    RUN_NAME="${MODEL}_l${LEVEL}_agent_mt_${DATE}"
else
    echo "=== [1/3] Generate: model=$MODEL task=$TASK samples=$COUNT level=$LEVEL ==="
    python3 run.py generate-batch \
        --models "$MODEL" \
        --tasks "$TASK" \
        --levels "$LEVEL" \
        --samples "$COUNT" \
        --yes

    RUN_NAME="${MODEL}_l${LEVEL}_${DATE}"
fi

echo "=== [2/3] Eval: run=$RUN_NAME ==="
python3 run.py eval --run "$RUN_NAME" || echo "[WARN] eval failed or partially failed"

echo "=== [3/3] Analyze: run=$RUN_NAME ==="
python3 run.py analyze --run "$RUN_NAME" || echo "[WARN] analyze failed or partially failed"

echo "=== Pipeline complete: $RUN_NAME ==="
