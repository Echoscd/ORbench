#!/bin/bash
# gen_all_data.sh — Generate missing data for all tasks
#
# Usage:
#   bash scripts/gen_all_data.sh                    # Generate all missing sizes
#   bash scripts/gen_all_data.sh small              # Only generate small
#   bash scripts/gen_all_data.sh small medium       # Generate small and medium
#   bash scripts/gen_all_data.sh --parallel 4       # Run 4 tasks in parallel

set -uo pipefail
cd "$(dirname "$0")/.."

SIZES=("small" "medium" "large")
MAX_PARALLEL=1
TASKS_DIR="tasks"
FILTER_SIZES=()

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --parallel|-p)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        small|medium|large)
            FILTER_SIZES+=("$1")
            shift
            ;;
        *)
            echo "Unknown arg: $1"
            echo "Usage: $0 [small|medium|large ...] [--parallel N]"
            exit 1
            ;;
    esac
done

if [[ ${#FILTER_SIZES[@]} -gt 0 ]]; then
    SIZES=("${FILTER_SIZES[@]}")
fi

echo "=== ORBench Batch Data Generation ==="
echo "  Sizes: ${SIZES[*]}"
echo "  Parallel: $MAX_PARALLEL"
echo ""

# Collect jobs
declare -a JOBS=()
for task_dir in "$TASKS_DIR"/*/; do
    task=$(basename "$task_dir")
    [[ -f "$task_dir/gen_data.py" ]] || continue
    for size in "${SIZES[@]}"; do
        if [[ ! -f "$task_dir/data/$size/cpu_time_ms.txt" ]]; then
            JOBS+=("$task:$size")
        fi
    done
done

if [[ ${#JOBS[@]} -eq 0 ]]; then
    echo "All data already generated. Nothing to do."
    exit 0
fi

echo "  Jobs to run: ${#JOBS[@]}"
for job in "${JOBS[@]}"; do
    echo "    - $job"
done
echo ""

FAILED=0
SUCCEEDED=0

run_job() {
    local task="$1"
    local size="$2"
    local task_dir="$TASKS_DIR/$task"
    local data_dir="$task_dir/data/$size"
    local log_file="$task_dir/data/${size}_gen.log"
    mkdir -p "$task_dir/data"
    echo "[START] $task/$size"
    if python3 "$task_dir/gen_data.py" "$size" "$data_dir" --with-expected > "$log_file" 2>&1; then
        echo "[DONE]  $task/$size"
        return 0
    else
        echo "[FAIL]  $task/$size (see $log_file)"
        return 1
    fi
}

if [[ $MAX_PARALLEL -le 1 ]]; then
    for job in "${JOBS[@]}"; do
        task="${job%%:*}"
        size="${job##*:}"
        if run_job "$task" "$size"; then
            ((SUCCEEDED++))
        else
            ((FAILED++)) || true
        fi
    done
else
    # Parallel: launch up to MAX_PARALLEL at a time
    RUNNING=0
    declare -A PID_MAP=()

    for job in "${JOBS[@]}"; do
        task="${job%%:*}"
        size="${job##*:}"
        run_job "$task" "$size" &
        PID_MAP[$!]="$job"
        ((RUNNING++))

        if [[ $RUNNING -ge $MAX_PARALLEL ]]; then
            wait -n 2>/dev/null || true
            # Reap finished
            for pid in "${!PID_MAP[@]}"; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    if wait "$pid" 2>/dev/null; then
                        ((SUCCEEDED++))
                    else
                        ((FAILED++)) || true
                    fi
                    unset "PID_MAP[$pid]"
                    ((RUNNING--))
                fi
            done
        fi
    done

    # Wait for remaining
    for pid in "${!PID_MAP[@]}"; do
        if wait "$pid" 2>/dev/null; then
            ((SUCCEEDED++))
        else
            ((FAILED++)) || true
        fi
    done
fi

echo ""
echo "=== Summary ==="
echo "  Succeeded: $SUCCEEDED"
echo "  Failed:    $FAILED"
echo "  Total:     ${#JOBS[@]}"
