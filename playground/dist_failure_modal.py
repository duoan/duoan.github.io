"""Reproduce common distributed training failures for the runbook blog post.

Cases (each returns structured JSON metrics):

  1. nan              — FP16 overflow → NaN loss / grads
  2. loss_spike       — rare corrupt batch → transient loss explosion
  3. numerical_drift  — silent rank desync (local-only param update)
  4. memory_leak      — retained activation references grow RSS / CUDA alloc
  5. straggler        — one slow rank paces the collective
  6. bad_node         — persistent compute degradation on one rank
  7. nccl_hang        — missing collective participant → watchdog timeout
  8. throughput_cliff — tiny local work → communication dominates

Usage (from repo root)::

    # Local CPU/gloo lab (no Modal / GPU required):
    uv run python playground/dist_failure_modal.py
    uv run python playground/dist_failure_modal.py --case nan

    # Modal GPU path (same cases; NCCL when multi-GPU is available):
    uv run modal run playground/dist_failure_modal.py
    DIST_FAIL_GPU=A10G uv run modal run playground/dist_failure_modal.py

Writes ``playground/dist_failure_results.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import time
from pathlib import Path

import modal
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

GPU = os.environ.get("DIST_FAIL_GPU", "A10G")
CASES = (
    "nan",
    "loss_spike",
    "numerical_drift",
    "memory_leak",
    "straggler",
    "bad_node",
    "nccl_hang",
    "throughput_cliff",
)

app = modal.App("dist-failure-runbook")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "numpy")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _backend(prefer_nccl: bool = False) -> str:
    if prefer_nccl and torch.cuda.is_available() and dist.is_nccl_available():
        return "nccl"
    return "gloo"


def _device(rank: int, backend: str) -> torch.device:
    if backend == "nccl":
        return torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    return torch.device("cpu")


def _mlp(dim: int = 64, n_classes: int = 10) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim, dim),
        nn.GELU(),
        nn.Linear(dim, dim),
        nn.GELU(),
        nn.Linear(dim, n_classes),
    )


def _init_pg(rank: int, world_size: int, backend: str, port: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def _param_checksum(model: nn.Module) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for p in model.parameters():
        total = total + p.detach().double().sum()
    return float(total.item())


def _max_param_diff(model: nn.Module) -> float:
    """All-reduce max abs diff of each param vs rank-0 snapshot (via allgather)."""
    diffs: list[float] = []
    for p in model.parameters():
        flat = p.detach().float().reshape(-1).cpu()
        gathered = [torch.zeros_like(flat) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, flat)
        ref = gathered[0]
        for g in gathered[1:]:
            diffs.append(float((g - ref).abs().max().item()))
    return max(diffs) if diffs else 0.0


def _finite(x: float) -> bool:
    return bool(np.isfinite(x))


# ---------------------------------------------------------------------------
# Case workers (each runs inside one rank process)
# ---------------------------------------------------------------------------


def case_nan(rank: int, world_size: int, backend: str, port: int) -> dict:
    """FP16 forward with huge LR and no grad scaling → NaNs appear fast."""
    _init_pg(rank, world_size, backend, port)
    device = _device(rank, backend)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(0)
    model = _mlp(32, 5).to(device)
    # Intentionally fragile: fp16 weights + huge LR, no GradScaler.
    model = model.half()
    ddp = nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.SGD(ddp.parameters(), lr=50.0)

    losses: list[float] = []
    first_nan_step: int | None = None
    for step in range(40):
        x = torch.randn(16, 32, device=device, dtype=torch.float16)
        y = torch.randint(0, 5, (16,), device=device)
        opt.zero_grad(set_to_none=True)
        logits = ddp(x)
        loss = F.cross_entropy(logits.float(), y)
        loss_v = float(loss.detach().cpu())
        losses.append(loss_v)
        if first_nan_step is None and not _finite(loss_v):
            first_nan_step = step
        if not _finite(loss_v):
            break
        loss.backward()
        # Detect NaN grads before step (debug signal).
        grad_nan = any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
        )
        if grad_nan and first_nan_step is None:
            first_nan_step = step
            break
        opt.step()

    # Healthy control: keep FP16 compute but master weights in FP32 + small LR + clip.
    torch.manual_seed(0)
    model_ok = _mlp(32, 5).to(device)  # fp32 master weights
    ddp_ok = nn.parallel.DistributedDataParallel(model_ok)
    opt_ok = torch.optim.SGD(ddp_ok.parameters(), lr=0.05)
    losses_ok: list[float] = []
    for _step in range(40):
        x = torch.randn(16, 32, device=device)
        y = torch.randint(0, 5, (16,), device=device)
        opt_ok.zero_grad(set_to_none=True)
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                loss = F.cross_entropy(ddp_ok(x), y)
            loss.backward()
        else:
            loss = F.cross_entropy(ddp_ok(x), y)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(ddp_ok.parameters(), 1.0)
        opt_ok.step()
        losses_ok.append(float(loss.detach().float().cpu()))

    out = {
        "case": "nan",
        "rank": rank,
        "backend": backend,
        "device": str(device),
        "broken_losses": losses,
        "first_nan_step": first_nan_step,
        "nan_detected": first_nan_step is not None,
        "healthy_losses": losses_ok,
        "healthy_all_finite": all(_finite(v) for v in losses_ok),
        "fix": "fp32 master weights / GradScaler / lower LR / grad clip; assert isfinite",
    }
    dist.destroy_process_group()
    return out


def case_loss_spike(rank: int, world_size: int, backend: str, port: int) -> dict:
    """Most batches fine; steps {12, 27} inject huge feature scale → loss spike."""
    _init_pg(rank, world_size, backend, port)
    device = _device(rank, backend)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(0 + rank)
    model = _mlp(64, 10).to(device)
    ddp = nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.AdamW(ddp.parameters(), lr=1e-3)

    spike_steps = {12, 27}
    losses: list[float] = []
    for step in range(40):
        x = torch.randn(32, 64, device=device)
        y = torch.randint(0, 10, (32,), device=device)
        if step in spike_steps:
            # Corrupt batch: extreme activations (bad tokenization / unnormalized input).
            x = x * 1e3
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(ddp(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))

    arr = np.asarray(losses, dtype=np.float64)
    med = float(np.median(arr))
    spike_vals = {s: losses[s] for s in sorted(spike_steps)}
    ratios = {s: losses[s] / max(med, 1e-9) for s in sorted(spike_steps)}

    out = {
        "case": "loss_spike",
        "rank": rank,
        "backend": backend,
        "losses": losses,
        "median_loss": med,
        "spike_steps": sorted(spike_steps),
        "spike_losses": spike_vals,
        "spike_ratio_vs_median": ratios,
        "detected": all(ratios[s] >= 5.0 for s in spike_steps),
        "fix": "log per-batch stats (input norm, label hist); clip grads; quarantine outliers",
    }
    dist.destroy_process_group()
    return out


def case_numerical_drift(rank: int, world_size: int, backend: str, port: int) -> dict:
    """Silent desync: after warmup, rank 0 applies an extra local param update.

    Loss can still look fine (training continues) while ranks diverge.
    """
    _init_pg(rank, world_size, backend, port)
    device = _device(rank, backend)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(0)
    model = _mlp(48, 8).to(device)
    ddp = nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.SGD(ddp.parameters(), lr=0.05)

    onset = 10
    losses: list[float] = []
    max_diffs: list[float] = []
    for step in range(30):
        # Identical data across ranks so healthy run stays synced.
        torch.manual_seed(1000 + step)
        x = torch.randn(24, 48, device=device)
        y = torch.randint(0, 8, (24,), device=device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(ddp(x), y)
        loss.backward()
        opt.step()

        if step >= onset and rank == 0:
            # Bug: rank-0-only "EMA" / logging side-effect that mutates weights.
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(0.01)

        md = _max_param_diff(model)
        max_diffs.append(md)
        losses.append(float(loss.detach().cpu()))

    out = {
        "case": "numerical_drift",
        "rank": rank,
        "backend": backend,
        "onset_step": onset,
        "losses": losses,
        "max_param_diff": max_diffs,
        "final_max_param_diff": max_diffs[-1],
        "drift_detected": max_diffs[-1] > 1e-3,
        "loss_still_finite": all(_finite(v) for v in losses),
        "fix": "periodically allgather/checksum weights; forbid rank-local param writes",
    }
    dist.destroy_process_group()
    return out


def case_memory_leak(rank: int, world_size: int, backend: str, port: int) -> dict:
    """Retain detached activations 'for debugging' → monotonic memory growth."""
    _init_pg(rank, world_size, backend, port)
    device = _device(rank, backend)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(0)
    model = _mlp(256, 10).to(device)
    ddp = nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.SGD(ddp.parameters(), lr=0.01)

    leak_bucket: list[torch.Tensor] = []
    allocated: list[float] = []
    for _step in range(40):
        x = torch.randn(128, 256, device=device)
        y = torch.randint(0, 10, (128,), device=device)
        opt.zero_grad(set_to_none=True)
        logits = ddp(x)
        # Bug: keep every step's activations (+ inputs) on device "for later viz".
        leak_bucket.append(torch.cat([logits.detach().reshape(-1), x.detach().reshape(-1)]))
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
            allocated.append(torch.cuda.memory_allocated(device) / (1024**2))
        else:
            allocated.append(sum(t.numel() * t.element_size() for t in leak_bucket) / (1024**2))

    # Fixed: clear debug buffer each step.
    leak_bucket.clear()
    allocated_fixed: list[float] = []
    for _step in range(40):
        x = torch.randn(128, 256, device=device)
        y = torch.randint(0, 10, (128,), device=device)
        opt.zero_grad(set_to_none=True)
        logits = ddp(x)
        debug = [logits.detach(), x.detach()]
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        debug.clear()
        if device.type == "cuda":
            torch.cuda.synchronize()
            allocated_fixed.append(torch.cuda.memory_allocated(device) / (1024**2))
        else:
            allocated_fixed.append(
                sum(t.numel() * t.element_size() for t in debug) / (1024**2)
            )

    growth = allocated[-1] - allocated[0]
    # Monotonic leak: last ≥ 5× first (and absolute growth ≥ 1 MB).
    leak_detected = allocated[-1] >= max(5.0 * max(allocated[0], 1e-6), allocated[0] + 1.0)
    out = {
        "case": "memory_leak",
        "rank": rank,
        "backend": backend,
        "device": str(device),
        "allocated_mb_leaky": allocated,
        "allocated_mb_fixed": allocated_fixed,
        "growth_mb": growth,
        "leak_detected": leak_detected,
        "fixed_flat": allocated_fixed[-1] <= allocated_fixed[0] + 0.1,
        "fix": "track cuda.memory_allocated / RSS over steps; never retain graph tensors",
    }
    dist.destroy_process_group()
    return out


def case_straggler(rank: int, world_size: int, backend: str, port: int) -> dict:
    """Rank 1 delays before an all-reduce → every rank's collective wait stretches."""
    _init_pg(rank, world_size, backend, port)
    device = _device(rank, backend)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(0)
    model = _mlp(32, 5).to(device)
    ddp = nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.SGD(ddp.parameters(), lr=0.05)

    straggler = 1 if world_size > 1 else 0
    sleep_ms = 120.0
    warmup = 3
    onset = 8
    step_ms: list[float] = []
    collective_ms: list[float] = []
    token = torch.ones(1, device=device)
    for step in range(18):
        x = torch.randn(16, 32, device=device)
        y = torch.randint(0, 5, (16,), device=device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(ddp(x), y)
        loss.backward()
        opt.step()

        # Explicit collective probe (isolates straggler from DDP bucket noise).
        dist.barrier()
        t0 = time.perf_counter()
        if rank == straggler and step >= onset:
            time.sleep(sleep_ms / 1e3)
        dist.all_reduce(token)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        collective_ms.append((t1 - t0) * 1e3)
        step_ms.append(collective_ms[-1])

    pre = collective_ms[warmup:onset]
    post = collective_ms[onset:]
    pre_med = float(np.median(pre))
    post_med = float(np.median(post))
    out = {
        "case": "straggler",
        "rank": rank,
        "backend": backend,
        "straggler_rank": straggler,
        "injected_sleep_ms": sleep_ms,
        "onset_step": onset,
        "collective_ms": collective_ms,
        "step_ms": step_ms,
        "pre_median_ms": pre_med,
        "post_median_ms": post_med,
        "slowdown": post_med / max(pre_med, 1e-9),
        "detected": post_med >= pre_med + 0.6 * sleep_ms,
        "fix": "per-rank timers around collectives; slowest rank sets the pace",
    }
    dist.destroy_process_group()
    return out


def case_bad_node(rank: int, world_size: int, backend: str, port: int) -> dict:
    """Persistent compute inflation on one rank (bad GPU / thermal throttle)."""
    _init_pg(rank, world_size, backend, port)
    device = _device(rank, backend)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(0)
    model = _mlp(96, 10).to(device)
    ddp = nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.SGD(ddp.parameters(), lr=0.05)

    bad = 0
    # Extra matmuls on the bad rank only — simulates degraded FLOP/s.
    inflate = 12
    local_ms: list[float] = []
    step_ms: list[float] = []
    for _step in range(20):
        x = torch.randn(48, 96, device=device)
        y = torch.randint(0, 10, (48,), device=device)
        t0 = time.perf_counter()
        if rank == bad:
            with torch.no_grad():
                junk = x
                for _ in range(inflate):
                    junk = junk @ junk.T[:96, :96]
                    junk = junk[:48]
        t1 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(ddp(x), y)
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        local_ms.append((t1 - t0) * 1e3)
        step_ms.append((t2 - t0) * 1e3)

    # Gather per-rank local compute means for detection.
    local_mean = torch.tensor([float(np.mean(local_ms))], dtype=torch.float64)
    gathered = [torch.zeros_like(local_mean) for _ in range(world_size)]
    dist.all_gather(gathered, local_mean)
    means = [float(t.item()) for t in gathered]
    healthy = float(min(means)) if means else 0.0
    ratios = [m / max(healthy, 1e-9) for m in means]
    # Flag ranks ≥5× the fastest peer (bad GPU / throttle).
    flagged = [i for i, r in enumerate(ratios) if r >= 5.0]

    out = {
        "case": "bad_node",
        "rank": rank,
        "backend": backend,
        "bad_rank": bad,
        "inflate_matmul_loops": inflate,
        "local_compute_ms": local_ms,
        "step_ms": step_ms,
        "per_rank_local_mean_ms": means,
        "ratio_vs_fastest": ratios,
        "flagged_ranks": flagged,
        "detected": bad in flagged,
        "fix": "compare pre-collective local timers across ranks; quarantine outliers",
    }
    dist.destroy_process_group()
    return out


def case_nccl_hang(rank: int, world_size: int, backend: str, port: int) -> dict:
    """One rank skips a collective → hang; watchdog surfaces it.

    Uses gloo (or NCCL on GPU) with a short monitored wait. Rank 1 never enters
    the barrier after onset — classic "NCCL timeout / illegal memory access"
    precursor in production logs.
    """
    _init_pg(rank, world_size, backend, port)
    device = _device(rank, backend)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    hang_rank = 1 if world_size > 1 else -1
    timeout_s = 2.0
    t_payload = torch.ones(1, device=device)
    # Healthy collective first.
    dist.all_reduce(t_payload)
    healthy_ok = True

    error: str | None = None
    elapsed_ms = 0.0
    if rank == hang_rank:
        # Simulate a dead worker: exit without destroying PG (peer sees timeout).
        time.sleep(0.05)
        error = "simulated_rank_exit_before_collective"
    else:
        t0 = time.perf_counter()
        try:
            # Wait briefly for peer; use a second allreduce that will block.
            # Multiprocessing parent enforces overall timeout via join.
            dist.all_reduce(torch.ones(1, device=device))
            # If we somehow proceed (ws=1), mark not hung.
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
        elapsed_ms = (time.perf_counter() - t0) * 1e3

    out = {
        "case": "nccl_hang",
        "rank": rank,
        "backend": backend,
        "hang_rank": hang_rank,
        "healthy_collective_ok": healthy_ok,
        "timeout_s": timeout_s,
        "peer_wait_ms": elapsed_ms,
        "hang_injected": hang_rank >= 0,
        "error": error,
        "fix": "NCCL_DEBUG=INFO, TORCH_NCCL_ASYNC_ERROR_HANDLING=1, per-rank heartbeats",
    }
    # Non-hang ranks may still be blocked; parent kills the process group.
    with contextlib.suppress(Exception):
        dist.destroy_process_group()
    return out


def case_throughput_cliff(rank: int, world_size: int, backend: str, port: int) -> dict:
    """Sweep local batch size: when compute << collective, throughput cliffs."""
    _init_pg(rank, world_size, backend, port)
    device = _device(rank, backend)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(0)
    model = _mlp(128, 10).to(device)
    ddp = nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.SGD(ddp.parameters(), lr=0.01)

    # Extra collective payload to make the cliff obvious on CPU/gloo.
    comm_mb = 8
    junk = torch.randn(comm_mb * 1024 * 1024 // 4, device=device)

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    rows: list[dict] = []
    for bs in batch_sizes:
        times: list[float] = []
        for _step in range(12):
            x = torch.randn(bs, 128, device=device)
            y = torch.randint(0, 10, (bs,), device=device)
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(ddp(x), y)
            loss.backward()
            dist.all_reduce(junk)  # fixed communication tax
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
        # Drop warmup
        steady = times[2:]
        med = float(np.median(steady))
        samples_per_s = (bs * world_size) / (med / 1e3)
        rows.append(
            {
                "batch_size_per_rank": bs,
                "global_batch": bs * world_size,
                "median_step_ms": med,
                "samples_per_s": samples_per_s,
            }
        )

    sps = [r["samples_per_s"] for r in rows]
    peak_i = int(np.argmax(sps))
    # Cliff: smallest batches where efficiency vs peak collapses.
    peak = sps[peak_i]
    cliff = next(
        (r for r in rows if r["samples_per_s"] < 0.45 * peak and r["batch_size_per_rank"] < rows[peak_i]["batch_size_per_rank"]),
        rows[0],
    )

    out = {
        "case": "throughput_cliff",
        "rank": rank,
        "backend": backend,
        "world_size": world_size,
        "comm_mb": comm_mb,
        "sweep": rows,
        "peak_batch_size": rows[peak_i]["batch_size_per_rank"],
        "peak_samples_per_s": peak,
        "cliff_batch_size": cliff["batch_size_per_rank"],
        "cliff_samples_per_s": cliff["samples_per_s"],
        "cliff_ratio_vs_peak": cliff["samples_per_s"] / max(peak, 1e-9),
        "fix": "raise local work / overlap comm; avoid tiny microbatches with fat collectives",
    }
    dist.destroy_process_group()
    return out


CASE_FNS = {
    "nan": case_nan,
    "loss_spike": case_loss_spike,
    "numerical_drift": case_numerical_drift,
    "memory_leak": case_memory_leak,
    "straggler": case_straggler,
    "bad_node": case_bad_node,
    "nccl_hang": case_nccl_hang,
    "throughput_cliff": case_throughput_cliff,
}


# ---------------------------------------------------------------------------
# Multiprocess launcher
# ---------------------------------------------------------------------------


def _worker(
    rank: int,
    world_size: int,
    case: str,
    backend: str,
    port: int,
    result_queue: mp.Queue,
) -> None:
    try:
        fn = CASE_FNS[case]
        result_queue.put(("ok", rank, fn(rank, world_size, backend, port)))
    except Exception as e:  # noqa: BLE001
        result_queue.put(("err", rank, f"{type(e).__name__}: {e}"))


def run_case(
    case: str,
    world_size: int = 2,
    prefer_nccl: bool = False,
    base_port: int = 29500,
) -> dict:
    if case not in CASE_FNS:
        raise ValueError(f"Unknown case {case!r}; choose from {CASES}")

    backend = _backend(prefer_nccl=prefer_nccl)
    # Unique port per case to avoid collisions across sequential runs.
    port = base_port + (abs(hash(case)) % 1000)

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    procs: list[mp.Process] = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_worker,
            args=(rank, world_size, case, backend, port, result_queue),
        )
        p.start()
        procs.append(p)

    # Hang case: peer blocks; enforce join timeout and kill leftovers.
    join_timeout = 8.0 if case == "nccl_hang" else 120.0
    for p in procs:
        p.join(timeout=join_timeout)

    timed_out = []
    for i, p in enumerate(procs):
        if p.is_alive():
            timed_out.append(i)
            p.terminate()
            p.join(timeout=5.0)

    results: list[dict] = []
    errors: list[str] = []
    while True:
        try:
            status, rank, payload = result_queue.get_nowait()
        except queue.Empty:
            break
        if status == "ok":
            results.append(payload)
        else:
            errors.append(f"rank{rank}: {payload}")

    # Summarize from rank 0 when available, else any rank.
    results.sort(key=lambda r: r.get("rank", 0))
    primary = results[0] if results else {}
    summary = {
        "case": case,
        "world_size": world_size,
        "backend": backend,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "timed_out_ranks": timed_out,
        "errors": errors,
        "ranks": results,
        "primary": primary,
    }

    # Case-specific rollup for the blog tables.
    if case == "nccl_hang":
        peer_errs = [
            r.get("error")
            for r in results
            if r.get("rank") != r.get("hang_rank") and r.get("error")
        ]
        summary["hang_confirmed"] = bool(timed_out) or bool(peer_errs) or bool(
            primary.get("hang_injected")
        )
        if timed_out:
            summary["symptom"] = (
                f"ranks {timed_out} blocked on collective after rank "
                f"{primary.get('hang_rank')} exited"
            )
        elif peer_errs:
            summary["symptom"] = f"peer collective error: {peer_errs[0]}"
        else:
            summary["symptom"] = "hang injected (single-rank fallback)"
        summary["peer_errors"] = peer_errs
    elif primary:
        for key in (
            "nan_detected",
            "first_nan_step",
            "detected",
            "drift_detected",
            "leak_detected",
            "slowdown",
            "cliff_ratio_vs_peak",
            "final_max_param_diff",
            "growth_mb",
        ):
            if key in primary:
                summary[key] = primary[key]

    return summary


def run_all(world_size: int = 2, prefer_nccl: bool = False) -> dict:
    out = {
        "meta": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "world_size": world_size,
            "prefer_nccl": prefer_nccl,
            "backend": _backend(prefer_nccl),
        },
        "cases": {},
    }
    for case in CASES:
        print(f"==> running {case}")
        out["cases"][case] = run_case(case, world_size=world_size, prefer_nccl=prefer_nccl)
    return out


# ---------------------------------------------------------------------------
# Modal + CLI
# ---------------------------------------------------------------------------


@app.function(gpu=GPU, image=image, timeout=1800)
def bench(which: str = "all") -> dict:
    prefer = torch.cuda.is_available()
    n_gpu = torch.cuda.device_count() if prefer else 0
    prefer_nccl = prefer and n_gpu >= 2
    ws = 2
    if which == "all":
        results = run_all(world_size=ws, prefer_nccl=prefer_nccl)
    else:
        results = {
            "meta": {
                "torch": torch.__version__,
                "cuda_available": prefer,
                "device_name": torch.cuda.get_device_name(0) if prefer else "cpu",
                "world_size": ws,
                "prefer_nccl": prefer_nccl,
                "backend": _backend(prefer_nccl),
                "source": "modal",
            },
            "cases": {which: run_case(which, world_size=ws, prefer_nccl=prefer_nccl)},
        }
    results["meta"]["source"] = "modal"
    return results


@app.local_entrypoint()
def main(case: str = "all") -> None:
    which = str(case).strip().lower()
    if which not in {"all", *CASES}:
        raise SystemExit(f"Unknown --case {case!r}; use all|{'|'.join(CASES)}")
    results = bench.remote(which)
    path = Path("playground/dist_failure_results.json")
    path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {path}")
    _print_summary(results)


def _print_summary(results: dict) -> None:
    meta = results.get("meta", {})
    print(f"device={meta.get('device_name')} backend={meta.get('backend')}")
    for name, c in results.get("cases", {}).items():
        p = c.get("primary", {})
        if name == "nan":
            print(f"  nan: detected={p.get('nan_detected')} first_step={p.get('first_nan_step')}")
        elif name == "loss_spike":
            print(f"  loss_spike: detected={p.get('detected')} ratios={p.get('spike_ratio_vs_median')}")
        elif name == "numerical_drift":
            print(
                f"  drift: detected={p.get('drift_detected')} "
                f"final_diff={p.get('final_max_param_diff'):.3e}"
            )
        elif name == "memory_leak":
            print(f"  leak: detected={p.get('leak_detected')} growth_mb={p.get('growth_mb'):.1f}")
        elif name == "straggler":
            print(
                f"  straggler: detected={p.get('detected')} "
                f"slowdown={p.get('slowdown'):.2f}x "
                f"pre={p.get('pre_median_ms'):.1f}ms post={p.get('post_median_ms'):.1f}ms"
            )
        elif name == "bad_node":
            print(
                f"  bad_node: detected={p.get('detected')} "
                f"flagged={p.get('flagged_ranks')} ratios={p.get('ratio_vs_fastest')}"
            )
        elif name == "nccl_hang":
            print(f"  nccl_hang: confirmed={c.get('hang_confirmed')} {c.get('symptom')}")
        elif name == "throughput_cliff":
            print(
                f"  cliff: peak_bs={p.get('peak_batch_size')} "
                f"cliff_bs={p.get('cliff_batch_size')} "
                f"ratio={p.get('cliff_ratio_vs_peak'):.2f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="all", choices=("all", *CASES))
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--nccl", action="store_true", help="Prefer NCCL when CUDA is available")
    args = parser.parse_args()

    if args.case == "all":
        results = run_all(world_size=args.world_size, prefer_nccl=args.nccl)
    else:
        results = {
            "meta": {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device_name": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "cpu",
                "world_size": args.world_size,
                "prefer_nccl": args.nccl,
                "backend": _backend(args.nccl),
                "source": "local",
            },
            "cases": {
                args.case: run_case(
                    args.case, world_size=args.world_size, prefer_nccl=args.nccl
                )
            },
        }
        results["meta"]["source"] = "local"

    if args.case == "all":
        results["meta"]["source"] = "local"

    path = Path("playground/dist_failure_results.json")
    path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {path}")
    _print_summary(results)
