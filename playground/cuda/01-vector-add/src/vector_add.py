"""Bandwidth study for an element-wise vector add.

Compares a hand-written CUDA kernel (JIT-compiled via
`torch.utils.cpp_extension.load_inline`) against `torch.add` across a sweep of
problem sizes, and writes the results to `results/timings.csv` plus a plot at
`results/throughput.svg`.

Run from the experiment directory:

    uv run python -m src.vector_add
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
__global__ void vec_add_kernel(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ c,
                               int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

torch::Tensor vec_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "this kernel is fp32-only");
    TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch");

    auto c = torch::empty_like(a);
    int n = static_cast<int>(a.numel());
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vec_add_kernel<<<blocks, threads>>>(a.data_ptr<float>(),
                                        b.data_ptr<float>(),
                                        c.data_ptr<float>(),
                                        n);
    return c;
}
"""

CPP_SRC = "torch::Tensor vec_add(torch::Tensor a, torch::Tensor b);"

EXP_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = EXP_DIR / "results"
SIZES = [1 << k for k in range(14, 27)]  # 16K .. 64M elements


def time_ms(fn, *args, iters: int = 200, warmup: int = 25) -> float:
    """Average device time per call in milliseconds, via cuda events."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def gbps_for_add_fp32(numel: int, ms: float) -> float:
    # 2 loads + 1 store of 4 bytes per element
    return 3 * 4 * numel / (ms * 1e-3) / 1e9


def maybe_plot(rows: list[dict[str, float]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed, skipping svg")
        return

    ns = [r["numel"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, [r["torch_gbps"] for r in rows], marker="o", label="torch.add")
    ax.plot(ns, [r["cuda_gbps"] for r in rows], marker="s", label="naive cuda kernel")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("numel")
    ax.set_ylabel("Achieved GB/s (3*4*N / time)")
    ax.set_title("Vector add: achieved bandwidth vs problem size (fp32)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out = RESULTS_DIR / "throughput.svg"
    fig.tight_layout()
    fig.savefig(out)
    print(f"[plot] wrote {out}")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this experiment requires a CUDA GPU.")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"[device] {props.name} sm_{props.major}{props.minor} "
          f"{props.total_memory // (1024**3)} GiB")

    print("[build] JIT-compiling vec_add via load_inline ...")
    mod = load_inline(
        name="vec_add_inline",
        cpp_sources=[CPP_SRC],
        cuda_sources=[CUDA_SRC],
        functions=["vec_add"],
        verbose=False,
    )

    rows: list[dict[str, float]] = []
    for n in SIZES:
        a = torch.randn(n, device=device, dtype=torch.float32)
        b = torch.randn(n, device=device, dtype=torch.float32)

        torch.testing.assert_close(mod.vec_add(a, b), a + b)

        t_torch = time_ms(torch.add, a, b)
        t_cuda = time_ms(mod.vec_add, a, b)
        bw_torch = gbps_for_add_fp32(n, t_torch)
        bw_cuda = gbps_for_add_fp32(n, t_cuda)

        print(f"n={n:>10}  torch={t_torch:7.4f}ms ({bw_torch:6.1f} GB/s)  "
              f"cuda={t_cuda:7.4f}ms ({bw_cuda:6.1f} GB/s)")
        rows.append({
            "numel": n,
            "torch_ms": t_torch,
            "cuda_ms": t_cuda,
            "torch_gbps": bw_torch,
            "cuda_gbps": bw_cuda,
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "timings.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv]  wrote {csv_path}")

    maybe_plot(rows)


if __name__ == "__main__":
    main()
