# Kernel experiments

Higher-level kernel work: Triton, CUTLASS, and fused / hand-tuned ops that
go beyond a single `__global__` function.

## Index

_(no experiments yet — this directory is a placeholder.)_

## Planned

| Slug                       | Question                                                |
|----------------------------|---------------------------------------------------------|
| `01-triton-vector-add`     | Match the `cuda/01-vector-add` numbers from Triton, then narrow the gap with `tl.load` vectorization. |
| `02-fused-bias-gelu`       | How much does fusing `linear → bias → gelu` save vs. the eager pipeline?       |
| `03-flash-attention-study` | Walk a minimal flash-attention forward, measure tile-size vs. throughput tradeoff. |

## Patterns

- For Triton: keep one autotune config per file. If you need many configs,
  the experiment is asking the wrong question — split it.
- Always cross-check against the `cuda/` equivalent when one exists. Triton
  and raw CUDA should bracket each other; if they diverge, the kernel is
  hiding a correctness or alignment issue.
