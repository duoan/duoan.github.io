"""Benchmark NCCL `all_reduce` algorithmic bandwidth across message sizes.

Run with torchrun (rank 0 writes the artifacts):

    torchrun --nproc_per_node=2 -m src.allreduce_bench

The script tolerates `world_size > visible_gpus` — extra ranks bind to
device 0. That's the single-workstation case this experiment targets.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import torch
import torch.distributed as dist

EXP_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = EXP_DIR / "results"
SIZES = [1 << k for k in range(12, 27)]  # 4K .. 64M elements


def time_collective_ms(numel: int, dtype: torch.dtype, iters: int, warmup: int) -> float:
    x = torch.empty(numel, dtype=dtype, device="cuda")
    for _ in range(warmup):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dist.all_reduce(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def algbw_gbps(numel: int, dtype: torch.dtype, ms: float, world_size: int) -> float:
    """Ring all-reduce algorithmic bandwidth, matching nccl-tests' definition."""
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    total = numel * bytes_per_elem
    factor = 2.0 * (world_size - 1) / max(world_size, 1)
    return factor * total / (ms * 1e-3) / 1e9


def maybe_plot(rows: list[dict[str, float]], world_size: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed, skipping svg")
        return

    ns = [r["numel"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, [r["algbw_gbps"] for r in rows], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("numel (fp32)")
    ax.set_ylabel("Achieved algbw (GB/s)")
    ax.set_title(f"all_reduce algorithmic bandwidth (world_size={world_size})")
    ax.grid(True, which="both", alpha=0.3)
    out = RESULTS_DIR / f"throughput_ws{world_size}.svg"
    fig.tight_layout()
    fig.savefig(out)
    print(f"[plot] wrote {out}")


def main() -> None:
    if "RANK" not in os.environ:
        raise SystemExit(
            "RANK env var not set. Run via:\n"
            "  torchrun --nproc_per_node=<N> -m src.allreduce_bench"
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this experiment requires NCCL.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    visible = torch.cuda.device_count()
    device_id = local_rank % max(visible, 1)
    torch.cuda.set_device(device_id)

    dist.init_process_group(backend="nccl")
    if rank == 0:
        props = torch.cuda.get_device_properties(0)
        print(f"[device] {props.name} sm_{props.major}{props.minor} "
              f"world_size={world_size} visible_gpus={visible}")

    iters, warmup = 50, 10
    dtype = torch.float32

    rows: list[dict[str, float]] = []
    for n in SIZES:
        ms = time_collective_ms(n, dtype, iters, warmup)
        bw = algbw_gbps(n, dtype, ms, world_size)
        if rank == 0:
            print(f"n={n:>10}  time={ms:7.4f}ms  algbw={bw:7.2f} GB/s")
            rows.append({
                "numel": n,
                "ms": ms,
                "algbw_gbps": bw,
                "world_size": world_size,
            })

    if rank == 0:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RESULTS_DIR / f"timings_ws{world_size}.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[csv]  wrote {csv_path}")
        maybe_plot(rows, world_size)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
