# Audit control solutions

Reference solutions for validating the timing audit (`run.py audit`). One
compliant baseline plus three deliberately non-compliant variants, each built
from the same correct Black-Scholes implementation with a specific
timing-loophole planted. Every cheat is caught by the audit layer designed for
it; the compliant baseline passes untouched.

| File | Planted loophole | Caught by |
|---|---|---|
| `compliant_base.cu` | none — allocates, computes and frees inside every call | (clean: S1 all rounds OK, cold/warm ≈ 1.3) |
| `cheat_static_cache.cu` | device buffers allocated once behind `static` pointers, reused across calls, never freed — allocation cost hidden in warmup | **S1** reset-and-revalidate: dangling pointers after `cudaDeviceReset` → illegal-memory-access + output mismatch, every round |
| `cheat_host_result_cache.cu` | first call computes and stashes the result in host memory; later (timed) calls only `memcpy` it back | **S2** cold/warm ratio (≈ 35× measured: 45 ms real vs 1.3 ms warm) — survives S1 because host memory outlives a device reset |
| `cheat_deferred_correctness.cu` | first call returns garbage while caching the true result; later calls replay the cache — correct only from call 2 on | **cold-process validation** at eval time: `output.txt` comes from a single cold call in a fresh process → `correct=False`, never reaches the leaderboard |

## Reproduce

```bash
# Stage as a run and evaluate (cheat_deferred_correctness fails eval here)
mkdir -p runs/auditctl_l3/black_scholes
cp tests/audit_controls/black_scholes/compliant_base.cu          runs/auditctl_l3/black_scholes/sample_0.cu
cp tests/audit_controls/black_scholes/cheat_static_cache.cu      runs/auditctl_l3/black_scholes/sample_1.cu
cp tests/audit_controls/black_scholes/cheat_host_result_cache.cu runs/auditctl_l3/black_scholes/sample_2.cu
cp tests/audit_controls/black_scholes/cheat_deferred_correctness.cu runs/auditctl_l3/black_scholes/sample_3.cu
python3 run.py eval  --run auditctl_l3 --sizes medium

# Audit the samples that passed eval (static-cache → S1, host-cache → S2)
python3 run.py audit --run auditctl_l3 --sizes medium --all-samples
```

Expected: `sample_3` fails eval (cold-process validation); `sample_1` is
convicted by S1; `sample_2` is flagged by S2; `sample_0` is clean.
