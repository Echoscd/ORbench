#!/bin/bash
# Generate test data for one or all tasks
# Usage:
#   ./gen_data.sh                          # all tasks, all sizes
#   ./gen_data.sh regex_match large        # single task, single size
#   ./gen_data.sh regex_match              # single task, all sizes
set -e
cd "$(dirname "$0")/.."

TASK="${1:-all}"
SIZE="${2:-all}"
SIZES=("small" "medium" "large")

gen_one() {
    local task="$1" size="$2"
    local out="tasks/${task}/data/${size}"
    local script="tasks/${task}/gen_data.py"
    if [[ ! -f "$script" ]]; then
        echo "[SKIP] $task: gen_data.py not found"
        return
    fi
    if [[ -f "${out}/input.bin" && -f "${out}/cpu_time_ms.txt" ]]; then
        echo "[SKIP] $task/$size: data already exists"
        return
    fi
    echo "[GEN] $task/$size ..."
    mkdir -p "$out"
    python3 "$script" "$size" "$out" --with-expected
    echo "[DONE] $task/$size"
}

if [[ "$TASK" == "all" ]]; then
    TASKS=($(ls tasks/))
else
    TASKS=("$TASK")
fi

if [[ "$SIZE" == "all" ]]; then
    for t in "${TASKS[@]}"; do
        for s in "${SIZES[@]}"; do
            gen_one "$t" "$s"
        done
    done
else
    for t in "${TASKS[@]}"; do
        gen_one "$t" "$SIZE"
    done
fi

echo "=== Data generation complete ==="
