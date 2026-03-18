#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) - Generate NFA + input strings + expected output for regex_match

Generates a random NFA with controlled state count and transition density,
then generates N input strings (mix of matching and non-matching),
and computes expected match results via Thompson NFA simulation.

Usage:
  python3 gen_data.py <size_name> <output_dir> [--with-expected]
"""

import os
import sys
import re
import shutil
import subprocess
from pathlib import Path
from collections import deque

import numpy as np

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))

from framework.orbench_io_py import write_input_bin

# ---------------------------------------------------------------------------
# Size definitions
# ---------------------------------------------------------------------------
SIZES = {
    "small":  {"num_strings": 1000,    "avg_str_len": 100, "num_states": 32,  "seed": 42},
    "medium": {"num_strings": 100000,  "avg_str_len": 200, "num_states": 64,  "seed": 42},
    "large":  {"num_strings": 1000000, "avg_str_len": 500, "num_states": 128, "seed": 42},
}

# Use lowercase letters as alphabet (26 symbols, mapped to 0-25)
NUM_SYMBOLS = 26


# ---------------------------------------------------------------------------
# Random NFA generation
# ---------------------------------------------------------------------------

def generate_random_nfa(num_states, num_symbols, seed, transition_density=2.0, eps_density=0.3):
    """
    Generate a random NFA with controlled properties.

    Args:
        num_states: Number of states (0 = start, num_states-1 = accept)
        num_symbols: Alphabet size
        seed: Random seed
        transition_density: Average number of transitions per (state, symbol)
        eps_density: Average number of epsilon transitions per state

    Returns:
        trans_offsets: int32[num_states * num_symbols + 1], CSR offsets
        trans_targets: int32[num_transitions], CSR targets
        eps_offsets: int32[num_states + 1], CSR offsets for epsilon transitions
        eps_targets: int32[num_eps_transitions], CSR targets for epsilon transitions
        accept_states: int32[], accept state IDs
    """
    rng = np.random.default_rng(seed)

    start_state = 0
    # Multiple accept states (last few states)
    num_accept = max(1, num_states // 8)
    accept_states = np.arange(num_states - num_accept, num_states, dtype=np.int32)

    # Build transition table: for each (state, symbol), generate some target states
    trans_lists = []
    for s in range(num_states):
        for c in range(num_symbols):
            # Poisson-distributed number of transitions
            n_trans = rng.poisson(transition_density)
            n_trans = min(n_trans, num_states)  # cap
            if n_trans > 0:
                targets = rng.choice(num_states, size=n_trans, replace=False)
                trans_lists.append(np.sort(targets).astype(np.int32))
            else:
                trans_lists.append(np.array([], dtype=np.int32))

    # Build CSR for transitions
    trans_offsets = np.zeros(num_states * num_symbols + 1, dtype=np.int32)
    all_targets = []
    offset = 0
    for i, targets in enumerate(trans_lists):
        trans_offsets[i] = offset
        all_targets.append(targets)
        offset += len(targets)
    trans_offsets[num_states * num_symbols] = offset
    trans_targets = np.concatenate(all_targets) if all_targets and offset > 0 else np.array([], dtype=np.int32)

    # Build epsilon transitions
    eps_lists = []
    for s in range(num_states):
        n_eps = rng.poisson(eps_density)
        n_eps = min(n_eps, num_states - 1)
        if n_eps > 0:
            # Don't add self-loops for epsilon
            candidates = [t for t in range(num_states) if t != s]
            if len(candidates) > 0:
                targets = rng.choice(candidates, size=min(n_eps, len(candidates)), replace=False)
                eps_lists.append(np.sort(targets).astype(np.int32))
            else:
                eps_lists.append(np.array([], dtype=np.int32))
        else:
            eps_lists.append(np.array([], dtype=np.int32))

    eps_offsets = np.zeros(num_states + 1, dtype=np.int32)
    all_eps = []
    offset = 0
    for i, targets in enumerate(eps_lists):
        eps_offsets[i] = offset
        all_eps.append(targets)
        offset += len(targets)
    eps_offsets[num_states] = offset
    eps_targets = np.concatenate(all_eps) if all_eps and offset > 0 else np.array([], dtype=np.int32)

    return (trans_offsets, trans_targets, eps_offsets, eps_targets,
            accept_states, start_state)


# ---------------------------------------------------------------------------
# Thompson NFA simulation (Python reference)
# ---------------------------------------------------------------------------

def epsilon_closure(eps_offsets, eps_targets, states):
    """Compute epsilon closure of a set of states via BFS."""
    result = set(states)
    queue = deque(states)
    while queue:
        s = queue.popleft()
        start = eps_offsets[s]
        end = eps_offsets[s + 1]
        for i in range(start, end):
            t = eps_targets[i]
            if t not in result:
                result.add(t)
                queue.append(t)
    return result


def nfa_match(trans_offsets, trans_targets, eps_offsets, eps_targets,
              accept_set, start_state, num_symbols, string):
    """
    Thompson NFA simulation: check if string is accepted.

    Args:
        string: sequence of symbol IDs (0-indexed)

    Returns:
        1 if accepted, 0 otherwise
    """
    # Start with epsilon closure of {start_state}
    current = epsilon_closure(eps_offsets, eps_targets, [start_state])

    for c in string:
        if not current:
            return 0
        # Compute next states
        next_states = set()
        for s in current:
            idx = s * num_symbols + c
            start = trans_offsets[idx]
            end = trans_offsets[idx + 1]
            for i in range(start, end):
                next_states.add(trans_targets[i])
        # Epsilon closure
        current = epsilon_closure(eps_offsets, eps_targets, next_states)

    return 1 if current & accept_set else 0


# ---------------------------------------------------------------------------
# String generation
# ---------------------------------------------------------------------------

def generate_matching_string(trans_offsets, trans_targets, eps_offsets, eps_targets,
                             accept_set, start_state, num_states, num_symbols,
                             target_len, rng, max_attempts=50):
    """Generate a string that matches the NFA via random walk."""
    for _ in range(max_attempts):
        # Random walk from start_state
        current_states = epsilon_closure(eps_offsets, eps_targets, [start_state])
        string = []

        for step in range(target_len):
            if not current_states:
                break

            # Try random symbols until we find one with transitions
            symbols_tried = 0
            moved = False
            symbol_order = rng.permutation(num_symbols)

            for c in symbol_order:
                next_states = set()
                for s in current_states:
                    idx = s * num_symbols + int(c)
                    start = trans_offsets[idx]
                    end = trans_offsets[idx + 1]
                    for i in range(start, end):
                        next_states.add(trans_targets[i])

                if next_states:
                    string.append(int(c))
                    current_states = epsilon_closure(eps_offsets, eps_targets, next_states)
                    moved = True
                    break

            if not moved:
                break

        # Check if we reached an accept state
        if current_states & accept_set and len(string) > 0:
            return string

    return None  # Failed to generate matching string


def generate_strings(trans_offsets, trans_targets, eps_offsets, eps_targets,
                     accept_set, start_state, num_states, num_symbols,
                     num_strings, avg_len, seed, match_ratio=0.4):
    """Generate a mix of matching and non-matching strings."""
    rng = np.random.default_rng(seed + 1000)

    strings = []
    expected = []
    num_match_target = int(num_strings * match_ratio)

    # Generate matching strings
    match_count = 0
    for i in range(num_match_target):
        str_len = max(1, int(rng.exponential(avg_len * 0.5)) + 1)
        str_len = min(str_len, avg_len * 3)  # cap
        s = generate_matching_string(
            trans_offsets, trans_targets, eps_offsets, eps_targets,
            accept_set, start_state, num_states, num_symbols,
            str_len, rng
        )
        if s is not None:
            strings.append(s)
            expected.append(1)
            match_count += 1
        else:
            # Fallback: random string (likely non-matching)
            str_len = max(1, int(rng.exponential(avg_len)) + 1)
            str_len = min(str_len, avg_len * 3)
            s = rng.integers(0, num_symbols, size=str_len).tolist()
            result = nfa_match(trans_offsets, trans_targets, eps_offsets, eps_targets,
                               accept_set, start_state, num_symbols, s)
            strings.append(s)
            expected.append(result)

    # Generate random strings (most will not match)
    for i in range(num_strings - len(strings)):
        str_len = max(1, int(rng.exponential(avg_len)) + 1)
        str_len = min(str_len, avg_len * 3)
        s = rng.integers(0, num_symbols, size=str_len).tolist()
        result = nfa_match(trans_offsets, trans_targets, eps_offsets, eps_targets,
                           accept_set, start_state, num_symbols, s)
        strings.append(s)
        expected.append(result)

    # Shuffle
    perm = rng.permutation(len(strings))
    strings = [strings[i] for i in perm]
    expected = [expected[i] for i in perm]

    print(f"  Match ratio: {sum(expected)}/{len(expected)} = {sum(expected)/len(expected):.1%}")
    return strings, expected


# ---------------------------------------------------------------------------
# Compile and run CPU baseline
# ---------------------------------------------------------------------------

def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "regex_match" / "solution_cpu"
    src = orbench_root / "tasks" / "regex_match" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "regex_match" / "task_io_cpu.c"
    harness = orbench_root / "framework" / "harness_cpu.c"

    sources = [src, task_io_cpu, harness]
    if exe.exists():
        try:
            exe_m = exe.stat().st_mtime
            if all(exe_m >= s.stat().st_mtime for s in sources):
                return exe
        except Exception:
            pass

    cmd = [
        "gcc", "-O2",
        "-I", str(orbench_root / "framework"),
        str(harness),
        str(task_io_cpu),
        str(src),
        "-o", str(exe),
        "-lm",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline compile failed:\n{r.stderr}")
    return exe


def run_cpu_time(exe: Path, data_dir: Path) -> float:
    import re as re_mod
    r = subprocess.run([str(exe), str(data_dir)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline run failed:\n{r.stderr}\n{r.stdout}")
    m = re_mod.search(r"TIME_MS:\s*([0-9.]+)", r.stdout)
    if not m:
        raise RuntimeError(f"TIME_MS not found in stdout:\n{r.stdout}")
    return float(m.group(1))


def run_cpu_expected_output(exe: Path, data_dir: Path) -> None:
    out_txt = data_dir / "output.txt"
    if out_txt.exists():
        out_txt.unlink()
    r = subprocess.run([str(exe), str(data_dir), "--validate"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline validate run failed:\n{r.stderr}\n{r.stdout}")
    if not out_txt.exists():
        raise RuntimeError("output.txt not produced by CPU baseline")
    expected = data_dir / "expected_output.txt"
    shutil.copy2(out_txt, expected)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 gen_data.py <size_name> <output_dir> [--with-expected]")
        sys.exit(1)

    size_name = sys.argv[1]
    out_dir = Path(sys.argv[2])
    with_expected = (len(sys.argv) == 4 and sys.argv[3] == "--with-expected")
    out_dir.mkdir(parents=True, exist_ok=True)

    if size_name not in SIZES:
        raise ValueError(f"Unknown size: {size_name}. Available: {list(SIZES.keys())}")

    cfg = SIZES[size_name]
    num_strings = cfg["num_strings"]
    avg_str_len = cfg["avg_str_len"]
    num_states = cfg["num_states"]
    seed = cfg["seed"]

    print(f"[gen_data] Generating {size_name}: {num_strings} strings, "
          f"avg_len={avg_str_len}, {num_states} NFA states...")

    # 1) Generate random NFA
    trans_offsets, trans_targets, eps_offsets, eps_targets, accept_states, start_state = \
        generate_random_nfa(num_states, NUM_SYMBOLS, seed,
                            transition_density=1.5, eps_density=0.2)

    num_trans = len(trans_targets)
    num_eps = len(eps_targets)
    num_accept = len(accept_states)

    print(f"  NFA: {num_states} states, {NUM_SYMBOLS} symbols, "
          f"{num_trans} transitions, {num_eps} eps-transitions, "
          f"{num_accept} accept states")

    # is_accept lookup
    accept_set = set(accept_states.tolist())

    # 2) Generate strings
    strings, expected_results = generate_strings(
        trans_offsets, trans_targets, eps_offsets, eps_targets,
        accept_set, start_state, num_states, NUM_SYMBOLS,
        num_strings, avg_str_len, seed
    )

    # Pack strings into flat arrays
    str_offsets = np.zeros(num_strings + 1, dtype=np.int32)
    offset = 0
    for i, s in enumerate(strings):
        str_offsets[i] = offset
        offset += len(s)
    str_offsets[num_strings] = offset
    total_chars = offset

    str_data = np.zeros(total_chars, dtype=np.int32)  # store as int32 for orbench format
    pos = 0
    for s in strings:
        for c in s:
            str_data[pos] = c
            pos += 1

    # is_accept array (boolean lookup)
    is_accept = np.zeros(num_states, dtype=np.int32)
    for a in accept_states:
        is_accept[a] = 1

    print(f"  Strings: {num_strings} total, {total_chars} chars")

    # 3) Write input.bin
    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("trans_offsets", "int32", trans_offsets),
            ("trans_targets", "int32", trans_targets),
            ("eps_offsets", "int32", eps_offsets),
            ("eps_targets", "int32", eps_targets),
            ("is_accept", "int32", is_accept),
            ("str_offsets", "int32", str_offsets),
            ("str_data", "int32", str_data),
        ],
        params={
            "num_states": num_states,
            "num_symbols": NUM_SYMBOLS,
            "start_state": start_state,
            "num_strings": num_strings,
            "total_chars": total_chars,
            "num_trans": num_trans,
            "num_eps": num_eps,
        },
    )

    # 4) Write requests.txt (one request per string — just the count)
    with open(out_dir / "requests.txt", "w") as f:
        for i in range(num_strings):
            f.write(f"{i}\n")

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        time_ms = run_cpu_time(exe, out_dir)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{time_ms:.3f}\n")
        run_cpu_expected_output(exe, out_dir)
        print(f"[gen_data] {size_name}: wrote all files in {out_dir}")
    else:
        print(f"[gen_data] {size_name}: wrote input.bin/requests.txt in {out_dir} "
              "(expected/cpu_time skipped; pass --with-expected)")


if __name__ == "__main__":
    main()
