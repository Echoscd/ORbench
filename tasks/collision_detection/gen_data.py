#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) - Generate 2D convex polygon collision detection scenes.

Generates clustered convex polygons, precomputes AABBs, and computes expected
collision counts via grid broad phase + SAT narrow phase.

Usage:
  python3 gen_data.py <size_name> <output_dir> [--with-expected]
"""

import os
import sys
import re as re_mod
import shutil
import subprocess
from pathlib import Path

import numpy as np

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))

from framework.orbench_io_py import write_input_bin

SIZES = {
    "small":  {"N": 1000,  "num_clusters": 10,  "cluster_radius": 8.0,
               "poly_r_min": 0.5, "poly_r_max": 1.5, "world_size": 100.0, "seed": 42},
    "medium": {"N": 10000, "num_clusters": 30,  "cluster_radius": 10.0,
               "poly_r_min": 0.3, "poly_r_max": 1.5, "world_size": 300.0, "seed": 42},
    "large":  {"N": 50000, "num_clusters": 80,  "cluster_radius": 12.0,
               "poly_r_min": 0.2, "poly_r_max": 2.0, "world_size": 600.0, "seed": 42},
}


# ---------------------------------------------------------------------------
# Convex polygon generation
# ---------------------------------------------------------------------------

def generate_convex_polygon(cx, cy, num_verts, r_min, r_max, rng):
    """Generate a random convex polygon centered at (cx, cy)."""
    angles = np.sort(rng.uniform(0, 2 * np.pi, num_verts))
    radii = rng.uniform(r_min, r_max, num_verts)
    xs = cx + radii * np.cos(angles)
    ys = cy + radii * np.sin(angles)
    return xs.astype(np.float32), ys.astype(np.float32)


def generate_scene(N, num_clusters, cluster_radius, poly_r_min, poly_r_max,
                   world_size, seed):
    """Generate N convex polygons in a clustered distribution."""
    rng = np.random.default_rng(seed)

    # Cluster centers
    centers = rng.uniform(cluster_radius, world_size - cluster_radius,
                          size=(num_clusters, 2))

    all_vx = []
    all_vy = []
    offsets = [0]
    aabbs = []

    for i in range(N):
        # Pick a cluster
        ci = i % num_clusters
        cx = centers[ci, 0] + rng.normal(0, cluster_radius * 0.5)
        cy = centers[ci, 1] + rng.normal(0, cluster_radius * 0.5)
        # Clamp to world
        cx = np.clip(cx, poly_r_max, world_size - poly_r_max)
        cy = np.clip(cy, poly_r_max, world_size - poly_r_max)

        num_verts = rng.integers(3, 9)  # 3 to 8 vertices
        vx, vy = generate_convex_polygon(cx, cy, num_verts, poly_r_min, poly_r_max, rng)

        all_vx.append(vx)
        all_vy.append(vy)
        offsets.append(offsets[-1] + len(vx))
        aabbs.append([vx.min(), vy.min(), vx.max(), vy.max()])

    poly_offsets = np.array(offsets, dtype=np.int32)
    vertices_x = np.concatenate(all_vx).astype(np.float32)
    vertices_y = np.concatenate(all_vy).astype(np.float32)
    aabb = np.array(aabbs, dtype=np.float32).ravel()  # N*4

    return poly_offsets, vertices_x, vertices_y, aabb


# ---------------------------------------------------------------------------
# SAT collision detection (Python reference)
# ---------------------------------------------------------------------------

def sat_test(vx_a, vy_a, vx_b, vy_b):
    """SAT test for two convex polygons. Returns True if they overlap."""
    for vx, vy in [(vx_a, vy_a), (vx_b, vy_b)]:
        n = len(vx)
        for i in range(n):
            j = (i + 1) % n
            # Edge normal (unnormalized)
            nx = -(vy[j] - vy[i])
            ny = vx[j] - vx[i]

            # Project A
            proj_a = nx * vx_a + ny * vy_a
            min_a, max_a = proj_a.min(), proj_a.max()

            # Project B
            proj_b = nx * vx_b + ny * vy_b
            min_b, max_b = proj_b.min(), proj_b.max()

            if max_a < min_b or max_b < min_a:
                return False  # Separating axis found
    return True


def aabb_overlap(aabb, i, j):
    """Check if AABBs of polygon i and j overlap."""
    # aabb layout: [min_x, min_y, max_x, max_y] per polygon
    ax0, ay0, ax1, ay1 = aabb[i*4], aabb[i*4+1], aabb[i*4+2], aabb[i*4+3]
    bx0, by0, bx1, by1 = aabb[j*4], aabb[j*4+1], aabb[j*4+2], aabb[j*4+3]
    return ax1 >= bx0 and bx1 >= ax0 and ay1 >= by0 and by1 >= ay0


def compute_collisions_python(N, poly_offsets, vertices_x, vertices_y, aabb,
                              world_size, cell_size):
    """Grid broad phase + SAT narrow phase. Returns collision counts."""
    # Build grid
    grid_dim = int(np.ceil(world_size / cell_size)) + 1
    grid = {}

    for i in range(N):
        gx0 = max(0, int(aabb[i*4+0] / cell_size))
        gy0 = max(0, int(aabb[i*4+1] / cell_size))
        gx1 = min(grid_dim - 1, int(aabb[i*4+2] / cell_size))
        gy1 = min(grid_dim - 1, int(aabb[i*4+3] / cell_size))
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                key = (gx, gy)
                if key not in grid:
                    grid[key] = []
                grid[key].append(i)

    # Collect unique candidate pairs
    candidate_set = set()
    for cell_polys in grid.values():
        n = len(cell_polys)
        for a in range(n):
            for b in range(a + 1, n):
                i, j = cell_polys[a], cell_polys[b]
                if i > j:
                    i, j = j, i
                candidate_set.add((i, j))

    print(f"  Broad phase: {len(candidate_set)} candidate pairs")

    # Narrow phase
    counts = np.zeros(N, dtype=np.int32)
    collision_count = 0

    for i, j in candidate_set:
        if not aabb_overlap(aabb, i, j):
            continue

        s_a, e_a = poly_offsets[i], poly_offsets[i + 1]
        s_b, e_b = poly_offsets[j], poly_offsets[j + 1]

        if sat_test(vertices_x[s_a:e_a], vertices_y[s_a:e_a],
                    vertices_x[s_b:e_b], vertices_y[s_b:e_b]):
            counts[i] += 1
            counts[j] += 1
            collision_count += 1

    match_ratio = np.sum(counts > 0) / N
    print(f"  Collisions: {collision_count} pairs, "
          f"match_ratio={match_ratio:.1%}")
    return counts


# ---------------------------------------------------------------------------
# CPU baseline compile/run
# ---------------------------------------------------------------------------

def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "collision_detection" / "solution_cpu"
    src = orbench_root / "tasks" / "collision_detection" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "collision_detection" / "task_io_cpu.c"
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
        str(harness), str(task_io_cpu), str(src),
        "-o", str(exe), "-lm",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline compile failed:\n{r.stderr}")
    return exe


def run_cpu_time(exe: Path, data_dir: Path) -> float:
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
    r = subprocess.run([str(exe), str(data_dir), "--validate"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline validate failed:\n{r.stderr}\n{r.stdout}")
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
    N = cfg["N"]
    seed = cfg["seed"]

    print(f"[gen_data] Generating {size_name}: {N} polygons...")

    poly_offsets, vertices_x, vertices_y, aabb = generate_scene(
        N, cfg["num_clusters"], cfg["cluster_radius"],
        cfg["poly_r_min"], cfg["poly_r_max"], cfg["world_size"], seed
    )

    total_verts = len(vertices_x)
    world_size = cfg["world_size"]

    # Cell size: ~2x average polygon diameter
    avg_diameter = np.mean(np.sqrt(
        (aabb[2::4] - aabb[0::4])**2 + (aabb[3::4] - aabb[1::4])**2
    ))
    cell_size = max(avg_diameter * 2.0, 1.0)

    # Store as x100 int (no float params in orbench)
    world_size_x100 = int(round(world_size * 100))
    cell_size_x100 = int(round(cell_size * 100))

    print(f"  {N} polygons, {total_verts} vertices, "
          f"world={world_size}, cell_size={cell_size:.2f}")

    # Write input.bin
    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("poly_offsets", "int32", poly_offsets),
            ("vertices_x", "float32", vertices_x),
            ("vertices_y", "float32", vertices_y),
            ("aabb", "float32", aabb),
        ],
        params={
            "N": N,
            "total_verts": total_verts,
            "world_size_x100": world_size_x100,
            "cell_size_x100": cell_size_x100,
        },
    )

    # Dummy requests.txt
    with open(out_dir / "requests.txt", "w") as f:
        f.write("solve\n")

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        time_ms = run_cpu_time(exe, out_dir)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{time_ms:.3f}\n")
        run_cpu_expected_output(exe, out_dir)
        print(f"[gen_data] {size_name}: wrote all files in {out_dir}")
    else:
        print(f"[gen_data] {size_name}: wrote input.bin in {out_dir}")


if __name__ == "__main__":
    main()
