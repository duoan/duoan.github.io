"""Reproduce common distributed training failures for the runbook blog post.

Cases (each returns structured JSON metrics):

  1. nan              — catalog of common NaN/Inf recipes (AMP, data, masks, DDP, …)
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
from typing import Any

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


def _nan_hit(name: str, where: str, detail: str, fix: str, **extra: Any) -> dict:
    return {
        "name": name,
        "triggered": True,
        "where": where,
        "detail": detail,
        "fix": fix,
        **extra,
    }


def _nan_ok(name: str, detail: str, fix: str, **extra: Any) -> dict:
    return {
        "name": name,
        "triggered": False,
        "where": None,
        "detail": detail,
        "fix": fix,
        **extra,
    }


def _recipe_fp16_overflow(device: torch.device) -> dict:
    """FP16 weights + huge LR, no GradScaler."""
    torch.manual_seed(0)
    model = _mlp(32, 5).to(device).half()
    opt = torch.optim.SGD(model.parameters(), lr=50.0)
    for step in range(20):
        x = torch.randn(16, 32, device=device, dtype=torch.float16)
        y = torch.randint(0, 5, (16,), device=device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).float(), y)
        v = float(loss.detach().cpu())
        if not _finite(v):
            return _nan_hit(
                "fp16_overflow",
                "loss",
                f"non-finite loss at step {step} (value={v})",
                "FP32 master weights + GradScaler / lower LR / grad clip",
                first_step=step,
            )
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            return _nan_hit(
                "fp16_overflow",
                "grad",
                f"non-finite grad at step {step}",
                "FP32 master weights + GradScaler / lower LR / grad clip",
                first_step=step,
            )
        opt.step()
    return _nan_ok("fp16_overflow", "unexpectedly stayed finite", "GradScaler + lower LR")


def _recipe_huge_lr_fp32(device: torch.device) -> dict:
    """Even FP32 blows up with an absurd LR (no AMP involved)."""
    torch.manual_seed(0)
    model = _mlp(32, 5).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e3)
    for step in range(30):
        x = torch.randn(16, 32, device=device)
        y = torch.randint(0, 5, (16,), device=device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        v = float(loss.detach().cpu())
        if not _finite(v):
            return _nan_hit(
                "huge_lr_fp32",
                "loss",
                f"non-finite loss at step {step}",
                "LR warmup / smaller LR / grad clip; LR bugs are not AMP-only",
                first_step=step,
            )
        loss.backward()
        opt.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            return _nan_hit(
                "huge_lr_fp32",
                "param",
                f"non-finite params after step {step}",
                "LR warmup / smaller LR / grad clip; LR bugs are not AMP-only",
                first_step=step,
            )
    return _nan_ok("huge_lr_fp32", "unexpectedly stayed finite", "lower LR")


def _recipe_nan_in_inputs(device: torch.device) -> dict:
    """Corrupt shard already contains NaN features."""
    torch.manual_seed(0)
    model = _mlp(32, 5).to(device)
    x = torch.randn(16, 32, device=device)
    x[0, 0] = float("nan")
    y = torch.randint(0, 5, (16,), device=device)
    loss = F.cross_entropy(model(x), y)
    v = float(loss.detach().cpu())
    if not _finite(v) or not torch.isfinite(model(x)).all():
        return _nan_hit(
            "nan_in_inputs",
            "loss",
            f"NaN in batch → loss={v}",
            "assert isfinite(batch) in dataloader; quarantine corrupt shards",
            first_step=0,
        )
    return _nan_ok("nan_in_inputs", "model somehow absorbed NaN inputs", "validate inputs")


def _recipe_fp16_matmul_overflow(device: torch.device) -> dict:
    """FP16 GEMM overflows to Inf (common when activations grow under AMP)."""
    torch.manual_seed(0)
    a = torch.randn(128, 128, device=device, dtype=torch.float16) * 60
    b = torch.randn(128, 128, device=device, dtype=torch.float16) * 60
    c = a @ b
    c32 = a.float() @ b.float()
    if not torch.isfinite(c).all():
        return _nan_hit(
            "fp16_matmul_overflow",
            "activation",
            f"fp16 GEMM nonfinite={(~torch.isfinite(c)).sum().item()} "
            f"max={float(c.float().abs().max())}; fp32 max={float(c32.abs().max()):.3g}",
            "loss scaling / lower activation scale / BF16 where available",
            first_step=0,
            fp32_finite=bool(torch.isfinite(c32).all()),
        )
    return _nan_ok("fp16_matmul_overflow", "fp16 GEMM stayed finite", "GradScaler / BF16")


def _recipe_unscaled_attention(device: torch.device) -> dict:
    """Two attention NaNs: fully-masked softmax row, and fp16 score overflow."""
    # (1) All positions masked → softmax(-inf,...,-inf) = NaN.
    masked = torch.full((2, 8), float("-inf"), device=device)
    w_masked = torch.softmax(masked, dim=-1)
    # (2) FP16 scores overflow to Inf; stable softmax does Inf-Inf → NaN.
    torch.manual_seed(0)
    d = 64
    q = torch.randn(2, 16, d, device=device, dtype=torch.float16) * 40
    k = torch.randn(2, 16, d, device=device, dtype=torch.float16) * 40
    scores16 = q @ k.transpose(-1, -2)
    w16 = torch.softmax(scores16, dim=-1)
    scores_ok = (q.float() @ k.float().transpose(-1, -2)) / (d**0.5)
    w_ok = torch.softmax(scores_ok, dim=-1)
    triggered = (not torch.isfinite(w_masked).all()) or (not torch.isfinite(w16).all())
    if triggered:
        return _nan_hit(
            "attention_softmax_nan",
            "softmax",
            f"all_masked_finite={bool(torch.isfinite(w_masked).all())} "
            f"fp16_scores_finite={bool(torch.isfinite(scores16).all())} "
            f"fp16_softmax_finite={bool(torch.isfinite(w16).all())}",
            "scale 1/sqrt(d); never fully-mask a row; use SDPA / fused attn",
            first_step=0,
            fixed_scaled_fp32_finite=bool(torch.isfinite(w_ok).all()),
        )
    return _nan_ok("attention_softmax_nan", "attention recipes stayed finite", "scale + valid masks")


def _recipe_masked_nll_zero_times_neginf(device: torch.device) -> dict:
    """Classic mask bug: (mask * log_softmax).sum() with 0 * (-inf) → NaN."""
    torch.manual_seed(0)
    logits = torch.tensor([[100.0, -100.0], [100.0, -100.0]], device=device)
    # Position 1 is fully masked (pad); broken code still indexes it.
    log_probs = F.log_softmax(logits, dim=-1)
    # Force a -inf entry like a hard mask on logits before log_softmax.
    hard = logits.clone()
    hard[1, :] = float("-inf")  # all-masked row
    log_probs_bad = F.log_softmax(hard, dim=-1)
    mask = torch.tensor([1.0, 0.0], device=device)
    # Broken reduction used in many toy CE implementations:
    broken = (mask[:, None] * log_probs_bad).sum()
    # Also the 0 * -inf pattern directly:
    direct = torch.tensor(0.0, device=device) * torch.tensor(float("-inf"), device=device)
    # Fixed: masked_fill / ignore_index / nll_loss
    fixed = F.cross_entropy(
        logits,
        torch.tensor([0, 0], device=device),
        reduction="none",
    )
    fixed = (fixed * mask).sum() / mask.sum()
    triggered = (not torch.isfinite(log_probs_bad).all()) or (not torch.isfinite(broken)) or (
        not torch.isfinite(direct)
    )
    if triggered:
        return _nan_hit(
            "masked_nll_zero_neg_inf",
            "loss",
            f"0*(-inf)={float(direct)} broken_sum finite={bool(torch.isfinite(broken))} "
            f"allmasked_log_softmax finite={bool(torch.isfinite(log_probs_bad).all())}",
            "use ignore_index / masked_fill before softmax / sum only valid positions",
            first_step=0,
            fixed_finite=bool(torch.isfinite(fixed)),
            log_probs_finite=bool(torch.isfinite(log_probs).all()),
        )
    return _nan_ok("masked_nll_zero_neg_inf", "mask recipe did not NaN", "ignore_index")


def _recipe_div_by_zero_normalize(device: torch.device) -> dict:
    """Normalize losses / embeddings by a count that can be zero (empty batch shard)."""
    weights = torch.zeros(8, device=device)  # all padding
    values = torch.randn(8, device=device)
    broken = values.sum() / weights.sum()  # 0/0 → NaN
    fixed = values.sum() / weights.sum().clamp(min=1.0)
    if not torch.isfinite(broken):
        return _nan_hit(
            "div_by_zero_normalize",
            "loss",
            f"sum/count with count=0 → {float(broken)}",
            "clamp denominators; skip empty ranks; use mean over valid only",
            first_step=0,
            fixed_finite=bool(torch.isfinite(fixed)),
        )
    return _nan_ok("div_by_zero_normalize", "unexpectedly finite", "clamp denominator")


def _recipe_log_of_zero_prob(device: torch.device) -> dict:
    """BCE / custom NLL takes log(prob) when prob underflows to 0."""
    probs = torch.tensor([1e-45, 1.0], device=device).float()
    # Underflow to 0 in float32 for very small values depending on platform;
    # force an exact zero.
    probs = torch.tensor([0.0, 1.0], device=device)
    broken = torch.log(probs).mean()
    fixed = torch.log(probs.clamp(min=1e-8)).mean()
    if not torch.isfinite(broken):
        return _nan_hit(
            "log_of_zero",
            "loss",
            f"log(0)={float(torch.log(probs)[0])}",
            "log(clamp(p, min=eps)); prefer logits + BCEWithLogits / cross_entropy",
            first_step=0,
            fixed_finite=bool(torch.isfinite(fixed)),
        )
    return _nan_ok("log_of_zero", "log(0) did not produce NaN/Inf here", "clamp + logits APIs")


def _recipe_soft_label_ce(device: torch.device) -> dict:
    """Soft CE with a zero probability class: target * log(softmax) → 0*(-inf) risk."""
    torch.manual_seed(0)
    logits = torch.tensor([[50.0, -50.0]], device=device)
    # Soft label puts mass only on class 0; class 1 has target 0 against log≈-inf.
    target = torch.tensor([[1.0, 0.0]], device=device)
    log_p = F.log_softmax(logits, dim=-1)
    broken = -(target * log_p).sum()  # usually fine when target is exact 0 * large_neg
    # Make it fail: target has tiny epsilon then underflows, or use hard -inf logits
    logits2 = torch.tensor([[0.0, float("-inf")]], device=device)
    log_p2 = F.log_softmax(logits2, dim=-1)
    target2 = torch.tensor([[0.5, 0.5]], device=device)  # illegal mass on -inf class
    broken2 = -(target2 * log_p2).sum()
    fixed = F.cross_entropy(logits, torch.tensor([0], device=device))
    if not torch.isfinite(log_p2).all() or not torch.isfinite(broken2):
        return _nan_hit(
            "soft_label_vs_neg_inf_logit",
            "loss",
            f"soft label on -inf logit → {float(broken2)}; log_softmax finite="
            f"{bool(torch.isfinite(log_p2).all())}",
            "forbid -inf logits under soft labels; mask classes; use label smoothing carefully",
            first_step=0,
            hard_ce_finite=bool(torch.isfinite(broken) and torch.isfinite(fixed)),
        )
    return _nan_ok("soft_label_vs_neg_inf_logit", "did not NaN", "validate soft targets")


def _recipe_optimizer_nan_moment(device: torch.device) -> dict:
    """One NaN grad poisons Adam moments forever after."""
    torch.manual_seed(0)
    model = nn.Linear(8, 4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(4, 8, device=device)
    y = torch.randint(0, 4, (4,), device=device)
    # Normal step
    opt.zero_grad(set_to_none=True)
    F.cross_entropy(model(x), y).backward()
    opt.step()
    # Inject NaN grad once
    opt.zero_grad(set_to_none=True)
    F.cross_entropy(model(x), y).backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.reshape(-1)[0] = float("nan")
            break
    opt.step()
    moment_nan = False
    for st in opt.state.values():
        for k in ("exp_avg", "exp_avg_sq"):
            if k in st and not torch.isfinite(st[k]).all():
                moment_nan = True
    # Next step even with clean grads keeps NaN params
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    param_nan = any(not torch.isfinite(p).all() for p in model.parameters())
    if moment_nan or param_nan:
        return _nan_hit(
            "adam_moment_poison",
            "optimizer_state",
            f"moment_nan={moment_nan} param_nan={param_nan}",
            "skip optimizer.step on non-finite grads; reset Adam state after incident",
            first_step=1,
        )
    return _nan_ok("adam_moment_poison", "Adam absorbed NaN grad", "skip non-finite steps")


def _recipe_ddp_nan_contagion(rank: int, model: nn.Module, device: torch.device) -> dict:
    """One rank injects NaN grads → allreduce spreads them to every replica."""
    ddp = nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.SGD(ddp.parameters(), lr=0.1)
    x = torch.randn(8, 32, device=device)
    y = torch.randint(0, 5, (8,), device=device)
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(ddp(x), y)
    loss.backward()
    # After DDP's autograd hook allreduced grads, mutate local grad on rank 0 only —
    # too late for this backward. Instead: inject BEFORE backward finishes by
    # registering a hook on a parameter that runs pre-allreduce... Simplest demo:
    # rank0 replaces a param with NaN, then next forward/backward contaminates.
    if rank == 0:
        with torch.no_grad():
            for p in model.parameters():
                p.reshape(-1)[0] = float("nan")
                break
    # Broadcast-free: next backward allreduces grads computed from NaN params on rank0
    # and finite on others — average still NaN.
    opt.zero_grad(set_to_none=True)
    loss2 = F.cross_entropy(ddp(x), y)
    loss2.backward()
    grad_nan_local = any(
        p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
    )
    flag = torch.tensor([1.0 if grad_nan_local else 0.0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    ranks_with_nan = int(flag.item())
    if grad_nan_local:
        return _nan_hit(
            "ddp_nan_contagion",
            "grad",
            f"ranks_with_nan_grad={ranks_with_nan}/{dist.get_world_size()}",
            "detect isfinite per rank before step; quarantine offending rank/batch",
            first_step=1,
            ranks_with_nan_grad=ranks_with_nan,
        )
    return _nan_ok(
        "ddp_nan_contagion",
        f"no local NaN grad (ranks_with_nan={ranks_with_nan})",
        "assert isfinite before allreduce consumers",
    )


def case_nan(rank: int, world_size: int, backend: str, port: int) -> dict:
    """Catalog of common NaN/Inf failure modes (data, numerics, masks, DDP)."""
    _init_pg(rank, world_size, backend, port)
    device = _device(rank, backend)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    recipes = [
        _recipe_fp16_overflow(device),
        _recipe_huge_lr_fp32(device),
        _recipe_nan_in_inputs(device),
        _recipe_fp16_matmul_overflow(device),
        _recipe_unscaled_attention(device),
        _recipe_masked_nll_zero_times_neginf(device),
        _recipe_div_by_zero_normalize(device),
        _recipe_log_of_zero_prob(device),
        _recipe_soft_label_ce(device),
        _recipe_optimizer_nan_moment(device),
    ]

    # DDP contagion needs the process group (already init).
    torch.manual_seed(0)
    contagion_model = _mlp(32, 5).to(device)
    recipes.append(_recipe_ddp_nan_contagion(rank, contagion_model, device))

    # Healthy control: fp32 + mild LR + clip.
    torch.manual_seed(0)
    model_ok = _mlp(32, 5).to(device)
    ddp_ok = nn.parallel.DistributedDataParallel(model_ok)
    opt_ok = torch.optim.SGD(ddp_ok.parameters(), lr=0.05)
    losses_ok: list[float] = []
    for _step in range(20):
        x = torch.randn(16, 32, device=device)
        y = torch.randint(0, 5, (16,), device=device)
        opt_ok.zero_grad(set_to_none=True)
        loss = F.cross_entropy(ddp_ok(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ddp_ok.parameters(), 1.0)
        opt_ok.step()
        losses_ok.append(float(loss.detach().float().cpu()))

    triggered = [r for r in recipes if r["triggered"]]
    # Keep a short fp16 loss curve for the existing figure.
    fp16 = next(r for r in recipes if r["name"] == "fp16_overflow")
    broken_losses = []
    torch.manual_seed(0)
    model = _mlp(32, 5).to(device).half()
    opt = torch.optim.SGD(model.parameters(), lr=50.0)
    for _step in range(12):
        x = torch.randn(16, 32, device=device, dtype=torch.float16)
        y = torch.randint(0, 5, (16,), device=device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).float(), y)
        v = float(loss.detach().cpu())
        broken_losses.append(v)
        if not _finite(v):
            break
        loss.backward()
        opt.step()

    out = {
        "case": "nan",
        "rank": rank,
        "backend": backend,
        "device": str(device),
        "recipes": recipes,
        "n_recipes": len(recipes),
        "n_triggered": len(triggered),
        "triggered_names": [r["name"] for r in triggered],
        "nan_detected": len(triggered) > 0,
        "first_nan_step": fp16.get("first_step"),
        "broken_losses": broken_losses,
        "healthy_losses": losses_ok,
        "healthy_all_finite": all(_finite(v) for v in losses_ok),
        "fix": "catalog: validate inputs, scale attn, clamp denoms, skip non-finite Adam steps",
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
            "n_triggered",
            "triggered_names",
            "n_recipes",
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
            print(
                f"  nan: detected={p.get('nan_detected')} "
                f"{p.get('n_triggered')}/{p.get('n_recipes')} recipes → "
                f"{p.get('triggered_names')}"
            )
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
