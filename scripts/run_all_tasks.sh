#!/bin/bash
# run_all_tasks.sh — Run agent experiments on ALL tasks for a given model
#
# Usage:
#   bash scripts/run_all_tasks.sh <model> [repeats] [turns] [level]
#   bash scripts/run_all_tasks.sh gemini-3.1-pro-preview-openrouter 1 1 2
#   bash scripts/run_all_tasks.sh kimi-k2.5-openrouter 1 1 2

cd "$(dirname "$0")/.."

MODEL="${1:?Usage: $0 <model> [repeats] [turns] [level]}"
REPEATS="${2:-1}"
TURNS="${3:-1}"
LEVEL="${4:-2}"
DATE=$(date +%Y%m%d_%H%M)

# All tasks with medium data ready
TASKS=()
for task_dir in tasks/*/; do
    task=$(basename "$task_dir")
    [[ -f "$task_dir/data/medium/cpu_time_ms.txt" ]] || continue
    [[ -f "$task_dir/prompt_template.yaml" ]] || continue
    TASKS+=("$task")
done

echo "============================================================"
echo "  Model:   $MODEL"
echo "  Tasks:   ${#TASKS[@]}"
echo "  Repeats: $REPEATS"
echo "  Turns:   $TURNS"
echo "  Level:   $LEVEL"
echo "  Date:    $DATE"
echo "============================================================"
echo ""
echo "  Tasks: ${TASKS[*]}"
echo ""

RUN_NAME="${MODEL}_l${LEVEL}_agent_mt_${DATE}"
MAX_PARALLEL="${5:-4}"  # Max concurrent tasks (default 4)
FAILED=0
SUCCEEDED=0

echo "  Parallel: $MAX_PARALLEL"

# Track background PIDs
declare -A TASK_PIDS
declare -A TASK_LOGS
RUNNING=0

wait_for_slot() {
    # Wait until a slot opens up
    while (( RUNNING >= MAX_PARALLEL )); do
        for t in "${!TASK_PIDS[@]}"; do
            pid=${TASK_PIDS[$t]}
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid"
                rc=$?
                if [[ $rc -eq 0 ]]; then
                    echo ">>> [$t] done"
                    ((SUCCEEDED++))
                else
                    echo ">>> [$t] FAILED (exit=$rc)"
                    ((FAILED++)) || true
                fi
                unset "TASK_PIDS[$t]"
                ((RUNNING--))
            fi
        done
        if (( RUNNING >= MAX_PARALLEL )); then
            sleep 2
        fi
    done
}

for task in "${TASKS[@]}"; do
    wait_for_slot
    echo ""
    echo ">>> [$task] starting (parallel)..."
    LOG_FILE="/tmp/orbench_${RUN_NAME}_${task}.log"
    TASK_LOGS[$task]="$LOG_FILE"
    python3 run.py agent-multiturn \
        --model "$MODEL" \
        --task "$task" \
        --level "$LEVEL" \
        --turns "$TURNS" \
        --repeats "$REPEATS" \
        --run-name "$RUN_NAME" > "$LOG_FILE" 2>&1 &
    TASK_PIDS[$task]=$!
    ((RUNNING++))
done

# Wait for remaining tasks
for t in "${!TASK_PIDS[@]}"; do
    pid=${TASK_PIDS[$t]}
    wait "$pid"
    rc=$?
    if [[ $rc -eq 0 ]]; then
        echo ">>> [$t] done"
        ((SUCCEEDED++))
    else
        echo ">>> [$t] FAILED (exit=$rc)"
        ((FAILED++)) || true
    fi
done

echo ""
echo "============================================================"
echo "  Run complete: $RUN_NAME"
echo "  Succeeded: $SUCCEEDED / ${#TASKS[@]}"
echo "  Failed:    $FAILED / ${#TASKS[@]}"
echo "============================================================"

# Auto-summarize
echo ""
echo ">>> Generating summary..."
python3 scripts/summarize_run.py "runs/$RUN_NAME"
