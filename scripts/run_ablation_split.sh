#!/bin/bash
# Run split vs no-split ablation experiment
# Runs the same model x task twice: once with --split, once without
# Usage:
#   ./run_ablation_split.sh <model> <task> [turns] [repeats] [level]
#   ./run_ablation_split.sh kimi-k2.5-openrouter network_rm_dp 10 1 2
set -e
cd "$(dirname "$0")/.."

MODEL="${1:?Usage: $0 <model> <task> [turns] [repeats] [level]}"
TASK="${2:?Usage: $0 <model> <task> [turns] [repeats] [level]}"
TURNS="${3:-10}"
REPEATS="${4:-1}"
LEVEL="${5:-2}"

DATE=$(date +%Y%m%d_%H%M)
RUN_NOSPLIT="${MODEL}_l${LEVEL}_agent_mt_nosplit_${DATE}"
RUN_SPLIT="${MODEL}_l${LEVEL}_agent_mt_split_${DATE}"

echo "============================================"
echo "  Ablation: split vs no-split"
echo "  model=$MODEL  task=$TASK"
echo "  turns=$TURNS  repeats=$REPEATS  level=$LEVEL"
echo "============================================"

echo ""
echo ">>> [1/2] Running WITHOUT --split (run=$RUN_NOSPLIT) ..."
python3 run.py agent-multiturn \
    --model "$MODEL" \
    --task "$TASK" \
    --level "$LEVEL" \
    --turns "$TURNS" \
    --repeats "$REPEATS" \
    --run-name "$RUN_NOSPLIT"
echo "<<< [1/2] Done: $RUN_NOSPLIT"

echo ""
echo ">>> [2/2] Running WITH --split (run=$RUN_SPLIT) ..."
python3 run.py agent-multiturn \
    --model "$MODEL" \
    --task "$TASK" \
    --level "$LEVEL" \
    --turns "$TURNS" \
    --repeats "$REPEATS" \
    --split \
    --run-name "$RUN_SPLIT"
echo "<<< [2/2] Done: $RUN_SPLIT"

echo ""
echo "============================================"
echo "  Results:"
echo "  no-split: runs/$RUN_NOSPLIT/$TASK/"
echo "  split:    runs/$RUN_SPLIT/$TASK/"
echo "============================================"
