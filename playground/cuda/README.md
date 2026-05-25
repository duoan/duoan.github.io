# CUDA experiments

Hand-written CUDA kernels and the bandwidth / occupancy / latency studies
behind them. Most experiments here are intentionally minimal — the point is
to make a single mechanism (memory coalescing, vectorized loads, shared-mem
tiling, etc.) visible in isolation, not to ship a fast op.

## Index

| Slug                    | Question                                                  | Status |
|-------------------------|-----------------------------------------------------------|--------|
| `01-vector-add`         | How close does a naive vector-add get to peak HBM bandwidth, vs. `torch.add`? | seed   |

## Patterns

- Use `torch.utils.cpp_extension.load_inline` for one-file kernels so the
  experiment runs as a single `uv run python` invocation. Promote to a
  separate `setup.py` / CMake build only when the kernel grows beyond ~200 LoC
  or needs custom NVCC flags.
- Time with `torch.cuda.Event` (start/stop around N iters, `synchronize`,
  divide). Avoid `time.perf_counter()` — it measures launch latency, not
  device time.
- Always include a `torch.testing.assert_close` against a torch reference,
  even for "obviously correct" kernels. Bandwidth numbers from a kernel
  that's silently wrong will mislead future-you.
