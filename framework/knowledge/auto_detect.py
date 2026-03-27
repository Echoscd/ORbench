"""
auto_detect.py — Automatic feature extraction from CUDA source code and ptxas output.
"""

from __future__ import annotations

import re


def strip_comments(src: str) -> str:
    """Remove C/C++ comments."""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    src = re.sub(r"//.*?$", "", src, flags=re.MULTILINE)
    return src


def extract_auto_features(source_path: str) -> dict:
    """Extract all auto-detectable features from a CUDA source file."""
    with open(source_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    src = strip_comments(raw)

    def has(keyword):
        return keyword in src

    def has_any(keywords):
        return any(k in src for k in keywords)

    def count(pattern):
        return len(re.findall(pattern, src))

    features = {
        # Memory placement
        "uses_constant_memory": has("__constant__"),
        "uses_shared_memory": has("__shared__") and has("__syncthreads"),
        "uses_ldg": has("__ldg"),
        "uses_float4": has("float4") or has("int4"),

        # Compute
        "uses_fmaf": has_any(["fmaf", "__fmaf_rn"]),
        "uses_hw_intrinsics": has_any([
            "__vcmpltu4", "__vcmpgtu4", "__vabsdiffu4",
            "__ffs", "__clz", "__brev"
        ]),
        "uses_fast_math": has_any(["__fdividef", "__expf", "__logf", "__sinf"]),

        # Parallelism
        "uses_warp_shuffle": has_any(["__shfl_sync", "__shfl_down_sync", "__shfl_up_sync"]),
        "uses_warp_vote": has_any(["__ballot_sync", "__popc", "__all_sync", "__any_sync"]),
        "uses_cooperative_groups": has("cooperative_groups"),
        "num_atomicAdd": count(r"atomicAdd\b"),
        "num_syncthreads": count(r"__syncthreads\b"),

        # Compiler hints
        "uses_template_kernel": has("template") and has("__global__"),
        "num_template_params": _count_template_params(src),
        "uses_pragma_unroll": has("#pragma unroll"),
        "uses_launch_bounds": has("__launch_bounds__"),

        # Memory management
        "has_persistent_alloc": has("static") and has("cudaMalloc"),
        "num_cudaMalloc": count(r"cudaMalloc\b"),
        "num_cudaMemcpy": count(r"cudaMemcpy[^T]"),
        "num_cudaMemcpyToSymbol": count(r"cudaMemcpyToSymbol\b"),
        "uses_double_buffer": _detect_double_buffer(src),

        # Libraries
        "uses_thrust": has("thrust::"),
        "uses_cub": has("cub::"),
        "uses_cublas": has_any(["cublas", "cublasS", "cublasD"]),

        # Code structure
        "num_global_kernels": count(r"__global__\s+void"),
        "num_device_functions": count(r"__device__\s+(?:inline\s+)?(?:void|int|float|bool|double)"),
        "total_lines": len(raw.strip().split("\n")),
    }

    return features


def _count_template_params(src: str) -> int:
    matches = re.findall(
        r"template\s*<([^>]+)>\s*(?:__global__|__launch_bounds__)", src
    )
    if not matches:
        return 0
    return max(len(m.split(",")) for m in matches)


def _detect_double_buffer(src: str) -> bool:
    swap_patterns = [
        r"temp\s*=\s*\w+_next.*\w+_next\s*=\s*\w+_curr.*\w+_curr\s*=\s*temp",
        r"swap\s*\(",
        r"std::swap",
    ]
    for p in swap_patterns:
        if re.search(p, src, re.DOTALL):
            return True
    return False


def extract_ptxas_info(compile_stderr: str) -> list[dict]:
    """Extract per-kernel resource usage from nvcc --ptxas-options=-v stderr."""
    results = []
    current_kernel = None

    for line in compile_stderr.split("\n"):
        if "Function properties for" in line:
            current_kernel = line.split("for ")[-1].strip().rstrip(":")
        elif "Used" in line and current_kernel:
            info = {"kernel_name": current_kernel}

            reg = re.search(r"(\d+) registers", line)
            if reg:
                info["registers"] = int(reg.group(1))

            smem = re.search(r"(\d+) bytes smem", line)
            if smem:
                info["smem_bytes"] = int(smem.group(1))

            lmem = re.search(r"(\d+) bytes lmem", line)
            info["lmem_bytes"] = int(lmem.group(1)) if lmem else 0

            cmem = re.search(r"(\d+) bytes cmem\[0\]", line)
            if cmem:
                info["cmem_bytes"] = int(cmem.group(1))

            results.append(info)
            current_kernel = None

    return results


def ptxas_summary(ptxas_info: list[dict]) -> dict:
    if not ptxas_info:
        return {}
    return {
        "max_registers": max((k.get("registers", 0) for k in ptxas_info), default=0),
        "total_lmem_bytes": sum(k.get("lmem_bytes", 0) for k in ptxas_info),
        "has_register_spill": any(k.get("lmem_bytes", 0) > 0 for k in ptxas_info),
        "max_smem_bytes": max((k.get("smem_bytes", 0) for k in ptxas_info), default=0),
        "num_kernel_variants": len(ptxas_info),
    }
