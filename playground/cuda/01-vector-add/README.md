# 01 - Vector add: how close to peak HBM?

## Question

For a simple element-wise add `c = a + b` (fp32), how close to peak HBM
bandwidth do we get with a naive 1D-grid CUDA kernel, and how does that
compare to PyTorch's built-in `torch.add`?

## Hypothesis

- This op is bandwidth-bound: 2 reads + 1 write of 4 bytes per element,
  with arithmetic intensity ~0. Both implementations should saturate HBM
  for large `N`, with sub-saturation at small `N` due to launch overhead.
- The naive kernel and `torch.add` should land within a few percent of
  each other once `N` is large enough to amortize launch cost. Differences
  at small `N` are about launch overhead, not the kernel body.

## Run

```bash
./scripts/snapshot_env.sh > env.txt
./scripts/run.sh
```

Outputs:

- `results/timings.csv` — per-size timings + measured GB/s for both impls.
- `results/throughput.svg` — GB/s vs `numel` for both impls.

For deeper traces:

```bash
./scripts/profile.sh                    # nsys system trace + ncu kernel-level
```

## Results

_(populated after first run; keep this concise — full narrative goes into the blog post.)_

## Follow-ups

- Vectorize loads (`float4`) — does it move us closer to peak or only help
  at small `N`?
- Try fp16 / bf16 — does the achievable GB/s change once we're memory-bound?
- Compare against a Triton implementation (lives under `../../kernels/`).
