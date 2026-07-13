"""Distributed training failure lab for the runbook blog post.

Built around a small but real causal Transformer LM (attention + MLP + LN +
tied embeddings). Failures come from code patterns that show up in production
reviews — not ``sleep()``, ``tensor * 1e3``, or ``param.add_`` mocks.

Cases:

  1. nan              — attention/mask/AMP/Adam/DDP NaN catalog on the LM
  2. loss_spike       — z-loss / aux coefficient typo (1e-4 written as 1e2)
  3. numerical_drift  — rank0-only grad clip; rank0 EMA copy-back into student
  4. memory_leak      — debug ring buffer retaining logits every step
  5. straggler        — seqlen skew across ranks (long-doc vs short-doc bucket)
  6. bad_node         — persistent host preprocessing skew on one rank
  7. nccl_hang        — empty-microbatch ``continue`` skips a collective
  8. throughput_cliff — tokens/rank sweep; real DDP grad allreduce dominates

Usage::

    uv run modal run playground/dist_failure_modal.py          # 2×A10G NCCL
    uv run python playground/dist_failure_modal.py             # CPU/gloo
    uv run python playground/dist_failure_modal.py --case nan
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

GPU = os.environ.get("DIST_FAIL_GPU", "A10G:2")
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

VOCAB = 128
PAD_ID = 0
DIM = 64
N_LAYER = 2
N_HEAD = 4


# ---------------------------------------------------------------------------
# Model: small causal Transformer LM
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention. ``use_scale=False`` reproduces a real custom-attn bug."""

    def __init__(self, dim: int, n_heads: int, use_scale: bool = True) -> None:
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.use_scale = use_scale
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, C], key_padding_mask: [B, T] True = valid
        b, t, c = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, H, T, D
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.use_scale:
            att = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        else:
            # Real custom-attn footgun: forgot 1/sqrt(d_h). Under fp16 this is the
            # same matmul path production kernels take before softmax.
            att = q @ k.transpose(-2, -1)
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal, float("-inf"))
        if key_padding_mask is not None:
            # True=keep; mask keys that are pad.
            key_bad = ~key_padding_mask[:, None, None, :]  # [B,1,1,T]
            att = att.masked_fill(key_bad, float("-inf"))
        w = torch.softmax(att, dim=-1)
        y = (w @ v).transpose(1, 2).reshape(b, t, c)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, use_scale: bool = True) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, use_scale=use_scale)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(
        self,
        vocab: int = VOCAB,
        dim: int = DIM,
        n_layer: int = N_LAYER,
        n_heads: int = N_HEAD,
        max_seq: int = 512,
        use_attn_scale: bool = True,
    ) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, dim, padding_idx=PAD_ID)
        self.pos = nn.Embedding(max_seq, dim)
        self.blocks = nn.ModuleList(
            [Block(dim, n_heads, use_scale=use_attn_scale) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.max_seq = max_seq

    def forward(
        self, input_ids: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0)
        x = self.tok(input_ids) + self.pos(pos)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        return self.head(self.ln_f(x))


def _lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    # Shifted causal LM loss.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=ignore_index,
    )


def _make_batch(
    batch_size: int,
    seqlen: int,
    device: torch.device,
    *,
    seed: int,
    pad_frac: float = 0.0,
    pad_as_label: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Synthetic token batch with optional padding.

    ``pad_as_label=True`` keeps PAD_ID in labels (forgot ``ignore_index``) — a
    real packing/collate bug that produces loss spikes on heavily padded rows.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    ids = torch.randint(1, VOCAB, (batch_size, seqlen), generator=g)
    n_pad = int(seqlen * pad_frac)
    mask = torch.ones(batch_size, seqlen, dtype=torch.bool)
    if n_pad > 0:
        ids[:, -n_pad:] = PAD_ID
        mask[:, -n_pad:] = False
    labels = ids.clone()
    if not pad_as_label:
        labels[~mask] = -100
    return ids.to(device), labels.to(device), mask.to(device)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def _backend(prefer_nccl: bool = False) -> str:
    if prefer_nccl and torch.cuda.is_available() and dist.is_nccl_available():
        return "nccl"
    return "gloo"


def _device(rank: int, backend: str) -> torch.device:
    if backend == "nccl":
        return torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    return torch.device("cpu")


def _wrap_ddp(model: nn.Module, device: torch.device, **kwargs: Any) -> nn.Module:
    if device.type == "cuda":
        return nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], **kwargs
        )
    return nn.parallel.DistributedDataParallel(model, **kwargs)


def _init_pg(rank: int, world_size: int, backend: str, port: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def _finite(x: float) -> bool:
    return bool(np.isfinite(x))


def _max_param_diff(model: nn.Module) -> float:
    diffs: list[float] = []
    backend = dist.get_backend()
    device = next(model.parameters()).device
    for p in model.parameters():
        flat = p.detach().float().reshape(-1)
        flat = flat.to(device if backend == "nccl" else "cpu")
        gathered = [torch.zeros_like(flat) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, flat)
        ref = gathered[0]
        for g in gathered[1:]:
            diffs.append(float((g - ref).abs().max().item()))
    return max(diffs) if diffs else 0.0


def _nan_hit(name: str, where: str, detail: str, fix: str, **extra: Any) -> dict:
    return {"name": name, "triggered": True, "where": where, "detail": detail, "fix": fix, **extra}


def _nan_ok(name: str, detail: str, fix: str, **extra: Any) -> dict:
    return {
        "name": name,
        "triggered": False,
        "where": None,
        "detail": detail,
        "fix": fix,
        **extra,
    }


# ---------------------------------------------------------------------------
# NaN catalog — grounded in the Transformer LM
# ---------------------------------------------------------------------------


def _recipe_unscaled_attention(device: torch.device) -> dict:
    """Custom attn missing 1/sqrt(d_h) + oversized QKV init under fp16 → NaN.

    This is the production shape: a hand-rolled attention port that dropped the
    scale *and* kept a too-large projection init. No ``tensor * constant`` inject.
    """
    torch.manual_seed(0)
    # head_dim=64 → unscaled score variance is 64× larger than scaled.
    model = TinyTransformerLM(dim=256, n_layer=2, n_heads=4, use_attn_scale=False).to(device)
    with torch.no_grad():
        for blk in model.blocks:
            # Forgot N(0, 0.02) init common in LM ports — leave Linear default
            # then amplify once the way a bad checkpoint / resumed head does.
            blk.attn.qkv.weight.mul_(8.0)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    first = None
    for step in range(40):
        ids, labels, mask = _make_batch(4, 128, device, seed=1 + step)
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(ids, key_padding_mask=mask)
                loss = _lm_loss(logits, labels)
        else:
            logits = model(ids, key_padding_mask=mask)
            loss = _lm_loss(logits, labels)
        bad = (not torch.isfinite(logits).all()) or (not torch.isfinite(loss))
        if bad:
            first = step
            break
        loss.backward()
        opt.step()

    torch.manual_seed(0)
    model_ok = TinyTransformerLM(dim=256, n_layer=2, n_heads=4, use_attn_scale=True).to(device)
    ids, labels, mask = _make_batch(4, 128, device, seed=1)
    if device.type == "cuda":
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            ok = torch.isfinite(model_ok(ids, key_padding_mask=mask)).all()
    else:
        ok = torch.isfinite(model_ok(ids, key_padding_mask=mask)).all()

    if first is not None and bool(ok):
        return _nan_hit(
            "attn_missing_scale",
            "softmax",
            f"unscaled q@k under fp16 + large QKV → nonfinite at step {first}; "
            "scaled control finite",
            "scale by 1/sqrt(d_h); prefer SDPA / flash-attn; keep LM init scales",
            first_step=first,
        )
    if first is not None:
        return _nan_hit(
            "attn_missing_scale",
            "softmax",
            f"nonfinite at step {first} with use_scale=False",
            "scale by 1/sqrt(d_h)",
            first_step=first,
        )
    return _nan_ok("attn_missing_scale", "scores stayed finite", "keep 1/sqrt(d_h)")


def _recipe_fully_padded_row(device: torch.device) -> dict:
    """A packed row that is 100% padding → causal+pad mask ⇒ softmax(-inf row)=NaN."""
    torch.manual_seed(0)
    model = TinyTransformerLM().to(device)
    ids = torch.full((2, 16), PAD_ID, device=device, dtype=torch.long)
    ids[0, :8] = torch.randint(1, VOCAB, (8,), device=device)  # row0 partially valid
    # row1 fully pad — realistic "empty document after filter" packing mistake
    mask = ids != PAD_ID
    logits = model(ids, key_padding_mask=mask)
    # Probe attention softmax directly on a fully-masked score row (same failure mode).
    scores = torch.full((1, 8), float("-inf"), device=device)
    w = torch.softmax(scores, dim=-1)
    if not torch.isfinite(w).all():
        return _nan_hit(
            "fully_padded_softmax",
            "softmax",
            "all-masked attention row → softmax(-inf,…)=NaN "
            f"(batch has fully-padded row; logits finite={bool(torch.isfinite(logits).all())})",
            "drop empty rows before forward; never fully-mask a query row",
            first_step=0,
        )
    return _nan_ok("fully_padded_softmax", "unexpectedly finite", "drop empty rows")


def _recipe_amp_no_scaler(device: torch.device) -> dict:
    """FP16 autocast training without GradScaler — overflows on the LM."""
    if device.type != "cuda":
        # CPU: FP16 weights + large step proxy (same overflow class as AMP without scaler).
        torch.manual_seed(0)
        model = TinyTransformerLM().to(device).half()
        opt = torch.optim.SGD(model.parameters(), lr=1.0)
        for step in range(25):
            ids, labels, mask = _make_batch(8, 32, device, seed=100 + step)
            ids_h = ids  # embeddings cast inside via half weights path
            opt.zero_grad(set_to_none=True)
            # Cast embeddings path: feed long ids into half model
            logits = model(ids_h, key_padding_mask=mask)
            loss = _lm_loss(logits.float(), labels)
            v = float(loss.detach().cpu())
            if not _finite(v):
                return _nan_hit(
                    "amp_no_scaler",
                    "loss",
                    f"fp16 LM overflow at step {step}",
                    "GradScaler / BF16 / lower LR",
                    first_step=step,
                )
            loss.backward()
            opt.step()
        return _nan_ok("amp_no_scaler", "stayed finite on CPU fp16", "GradScaler")

    torch.manual_seed(0)
    model = TinyTransformerLM().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=2.0)
    for step in range(60):
        ids, labels, mask = _make_batch(8, 96, device, seed=100 + step)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            loss = _lm_loss(model(ids, key_padding_mask=mask), labels)
        v = float(loss.detach().float().cpu())
        if not _finite(v):
            return _nan_hit(
                "amp_no_scaler",
                "loss",
                f"autocast fp16 without GradScaler → nonfinite at step {step}",
                "use GradScaler (or BF16); unscale + inf check before step",
                first_step=step,
            )
        loss.backward()
        # Nonfinite grads also count.
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            return _nan_hit(
                "amp_no_scaler",
                "grad",
                f"nonfinite grads at step {step}",
                "GradScaler / BF16",
                first_step=step,
            )
        opt.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            return _nan_hit(
                "amp_no_scaler",
                "param",
                f"nonfinite params after step {step}",
                "GradScaler / BF16",
                first_step=step,
            )
    return _nan_ok("amp_no_scaler", "stayed finite", "still use GradScaler in fp16")


def _recipe_handrolled_masked_nll(device: torch.device) -> dict:
    """``(mask * log_softmax).sum()`` on a fully-padded row → 0*(-inf)=NaN."""
    torch.manual_seed(0)
    model = TinyTransformerLM().to(device)
    # Real packing outcome: one document filtered to empty → all-pad row.
    ids = torch.full((2, 16), PAD_ID, device=device, dtype=torch.long)
    ids[0, :10] = torch.randint(1, VOCAB, (10,), device=device)
    mask = ids != PAD_ID
    logits = model(ids, key_padding_mask=mask)
    log_p = F.log_softmax(logits, dim=-1)
    # Broken hand-rolled masked NLL (instead of ignore_index CE).
    broken = (mask.unsqueeze(-1).float() * log_p).sum()
    # Same 0*(-inf) identity that shows up when a row is all-masked.
    direct = torch.zeros((), device=device) * torch.tensor(float("-inf"), device=device)
    labels = ids.clone()
    labels[~mask] = -100
    fixed = _lm_loss(logits, labels)
    if (not torch.isfinite(broken)) or (not torch.isfinite(log_p).all()) or (not torch.isfinite(direct)):
        return _nan_hit(
            "handrolled_masked_nll",
            "loss",
            f"mask*log_softmax on fully-padded row → nonfinite; "
            f"0*(-inf)={float(direct)}; fixed_ce finite={bool(torch.isfinite(fixed))}",
            "use F.cross_entropy(..., ignore_index=-100); never mask*log_softmax",
            first_step=0,
        )
    return _nan_ok("handrolled_masked_nll", "did not NaN", "ignore_index")


def _recipe_empty_valid_tokens(device: torch.device) -> dict:
    """Mean over valid token CE when valid count is 0 → Inf/NaN."""
    torch.manual_seed(0)
    model = TinyTransformerLM().to(device)
    # Empty microbatch after pad filtering: every label is ignore_index.
    ids = torch.full((2, 16), PAD_ID, device=device, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    logits = model(ids)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    per_tok = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    valid = shift_labels.reshape(-1) != -100
    n = valid.sum()
    # Broken reduction seen in custom packing losses:
    broken = per_tok.sum() / n.float()
    fixed = per_tok.sum() / n.float().clamp(min=1.0)
    if not torch.isfinite(broken):
        return _nan_hit(
            "empty_valid_token_mean",
            "loss",
            f"LM token-mean with 0 valid targets → {float(broken.detach())}",
            "skip empty microbatches OR clamp denom; never divide by valid_count==0",
            first_step=0,
            fixed_finite=bool(torch.isfinite(fixed)),
        )
    return _nan_ok("empty_valid_token_mean", "unexpectedly finite", "clamp denom")


def _recipe_adam_moment_poison(device: torch.device) -> dict:
    """Nonfinite grads stepped into Adam permanently poison moments / weights.

    Overflow comes from the same AMP-without-scaler path as ``amp_no_scaler``;
    the distinct bug is continuing into AdamW without an isfinite guard.
    """
    torch.manual_seed(0)
    model = TinyTransformerLM().to(device)
    # Drive overflow with SGD+AMP (Adam itself often stays finite longer).
    sgd = torch.optim.SGD(model.parameters(), lr=2.0)
    overflow_step = None
    for step in range(60):
        ids, labels, mask = _make_batch(8, 96, device, seed=100 + step)
        sgd.zero_grad(set_to_none=True)
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                loss = _lm_loss(model(ids, key_padding_mask=mask), labels)
        else:
            model_h = model.half() if step == 0 else model
            model = model_h
            sgd = torch.optim.SGD(model.parameters(), lr=1.0)
            loss = _lm_loss(model(ids, key_padding_mask=mask).float(), labels)
        if not torch.isfinite(loss):
            overflow_step = step
            with contextlib.suppress(Exception):
                loss.backward()
            break
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            overflow_step = step
            break
        sgd.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            overflow_step = step
            break

    if overflow_step is None:
        return _nan_ok("adam_moment_poison", "never produced nonfinite grads", "skip nonfinite steps")

    # THE bug: hand existing nonfinite grads to Adam and step anyway.
    # Do NOT zero_grad first — that would erase the overflow grads.
    had_grads = any(p.grad is not None for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    if had_grads:
        with contextlib.suppress(Exception):
            opt.step()
    else:
        ids, labels, mask = _make_batch(8, 96, device, seed=200)
        try:
            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    loss = _lm_loss(model(ids, key_padding_mask=mask), labels)
            else:
                loss = _lm_loss(model(ids, key_padding_mask=mask).float(), labels)
            with contextlib.suppress(Exception):
                loss.backward()
            with contextlib.suppress(Exception):
                opt.step()
        except Exception:  # noqa: BLE001
            pass

    moment_nan = any(
        (k in st and not torch.isfinite(st[k]).all())
        for st in opt.state.values()
        for k in ("exp_avg", "exp_avg_sq")
    )
    param_nan = any(not torch.isfinite(p).all() for p in model.parameters())
    # One more Adam step — poisoned moments keep emitting NaNs.
    opt.zero_grad(set_to_none=True)
    ids, labels, mask = _make_batch(4, 32, device, seed=202)
    try:
        loss2 = _lm_loss(model(ids, key_padding_mask=mask), labels)
        if torch.isfinite(loss2):
            loss2.backward()
            opt.step()
            param_nan = param_nan or any(not torch.isfinite(p).all() for p in model.parameters())
    except Exception:  # noqa: BLE001
        loss2 = torch.tensor(float("nan"))
        param_nan = True

    if moment_nan or param_nan or not torch.isfinite(loss2):
        return _nan_hit(
            "adam_moment_poison",
            "optimizer_state",
            f"AMP overflow at step {overflow_step} stepped into Adam; "
            f"moment_nan={moment_nan} param_nan={param_nan}",
            "skip optimizer.step on nonfinite grads; reset Adam state after an incident",
            first_step=overflow_step,
        )
    return _nan_ok("adam_moment_poison", "Adam absorbed nonfinite without poison", "skip nonfinite steps")


def _recipe_ddp_nan_contagion(rank: int, device: torch.device) -> dict:
    """One rank's fully-padded pack → NaN grads → allreduce contaminates every rank."""
    torch.manual_seed(0)
    model = TinyTransformerLM().to(device)
    ddp = _wrap_ddp(model, device)
    if rank == 0:
        # Packing bug on rank 0 only: every document filtered empty → all-pad mask.
        # Custom attn does softmax over a fully -inf key row → NaN activations.
        ids = torch.full((4, 32), PAD_ID, device=device, dtype=torch.long)
        mask = torch.zeros(4, 32, dtype=torch.bool, device=device)
        labels = ids.clone()
        labels[:] = 1  # still supervise so CE runs through the NaN graph
    else:
        ids, labels, mask = _make_batch(4, 32, device, seed=9 + rank)

    loss = _lm_loss(ddp(ids, key_padding_mask=mask), labels)
    with contextlib.suppress(Exception):
        loss.backward()
    grad_nan = any(
        p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
    ) or (not torch.isfinite(loss))

    flag = torch.tensor(
        [1.0 if grad_nan else 0.0],
        device=device if dist.get_backend() == "nccl" else "cpu",
    )
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    n = int(flag.item())
    if n > 0:
        return _nan_hit(
            "ddp_nan_contagion",
            "grad",
            f"ranks reporting nonfinite={n}/{dist.get_world_size()} "
            "(seeded by fully-padded pack on rank 0)",
            "per-rank isfinite before step; quarantine offending batch/rank",
            first_step=0,
            ranks_with_nan=n,
        )
    return _nan_ok("ddp_nan_contagion", "no contagion observed", "isfinite guards")


def case_nan(rank: int, world_size: int, backend: str, port: int) -> dict:
    _init_pg(rank, world_size, backend, port)
    device = _device(rank, backend)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    recipes = [
        _recipe_unscaled_attention(device),
        _recipe_fully_padded_row(device),
        _recipe_amp_no_scaler(device),
        _recipe_handrolled_masked_nll(device),
        _recipe_empty_valid_tokens(device),
        _recipe_adam_moment_poison(device),
        _recipe_ddp_nan_contagion(rank, device),
    ]
    triggered = [r for r in recipes if r["triggered"]]

    # Healthy LM train snippet for the figure.
    torch.manual_seed(0)
    model_ok = TinyTransformerLM().to(device)
    ddp_ok = _wrap_ddp(model_ok, device)
    opt_ok = torch.optim.AdamW(ddp_ok.parameters(), lr=3e-4)
    healthy: list[float] = []
    for step in range(20):
        ids, labels, mask = _make_batch(8, 32, device, seed=50 + step + rank)
        opt_ok.zero_grad(set_to_none=True)
        loss = _lm_loss(ddp_ok(ids, key_padding_mask=mask), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ddp_ok.parameters(), 1.0)
        opt_ok.step()
        healthy.append(float(loss.detach().cpu()))

    # Broken curve: unscaled attn train a few steps for the plot.
    torch.manual_seed(0)
    model_bad = TinyTransformerLM(use_attn_scale=False).to(device)
    broken: list[float] = []
    for step in range(8):
        ids, labels, mask = _make_batch(4, 128, device, seed=70 + step)
        loss = _lm_loss(model_bad(ids, key_padding_mask=mask), labels)
        broken.append(float(loss.detach().cpu()))
        if not _finite(broken[-1]):
            break

    out = {
        "case": "nan",
        "rank": rank,
        "backend": backend,
        "device": str(device),
        "model": "TinyTransformerLM",
        "recipes": recipes,
        "n_recipes": len(recipes),
        "n_triggered": len(triggered),
        "triggered_names": [r["name"] for r in triggered],
        "nan_detected": len(triggered) > 0,
        "broken_losses": broken,
        "healthy_losses": healthy,
        "healthy_all_finite": all(_finite(v) for v in healthy),
        "fix": "scale attn; drop empty rows; GradScaler; ignore_index; isfinite before Adam",
    }
    dist.destroy_process_group()
    return out


# ---------------------------------------------------------------------------
# Loss spikes — forgot ignore_index on padded LM batches
# ---------------------------------------------------------------------------


def case_loss_spike(rank: int, world_size: int, backend: str, port: int) -> dict:
    """Z-loss / aux coefficient typo (``1e-4`` written ``100``) → rare CE+aux spikes.

    This is a config bug, not a tensor inject: the LM forward is healthy; a secondary
    term (z-loss, router aux, load-balance) is scaled wrong on some runs/steps.
    """
    _init_pg(rank, world_size, backend, port)
    device = _device(rank, backend)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(0)
    model = TinyTransformerLM().to(device)
    ddp = _wrap_ddp(model, device)
    opt = torch.optim.AdamW(ddp.parameters(), lr=3e-4)
    # Steps where a bad config / feature flag enables the wrong aux weight.
    spike_steps = {20, 28}
    z_weight_bad = 100.0  # intended 1e-4
    z_weight_ok = 1e-4
    losses: list[float] = []
    ce_only: list[float] = []
    for step in range(36):
        ids, labels, mask = _make_batch(
            8, 64, device, seed=1000 + step * 17 + rank, pad_frac=0.25
        )
        opt.zero_grad(set_to_none=True)
        logits = ddp(ids, key_padding_mask=mask)
        ce = _lm_loss(logits, labels)
        z_loss = logits.float().pow(2).mean()  # PaLM-style z-loss on logits
        w = z_weight_bad if step in spike_steps else z_weight_ok
        loss = ce + w * z_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        ce_only.append(float(ce.detach().cpu()))

    arr = np.asarray(losses, dtype=np.float64)
    med = float(np.median(arr[[i for i in range(len(arr)) if i not in spike_steps]]))
    local_jump = {s: losses[s] / max(losses[s - 1], 1e-9) for s in sorted(spike_steps)}
    ratios = {s: losses[s] / max(med, 1e-9) for s in sorted(spike_steps)}
    out = {
        "case": "loss_spike",
        "rank": rank,
        "backend": backend,
        "model": "TinyTransformerLM",
        "losses": losses,
        "ce_only": ce_only,
        "median_loss": med,
        "spike_steps": sorted(spike_steps),
        "spike_losses": {s: losses[s] for s in sorted(spike_steps)},
        "spike_ratio_vs_median": ratios,
        "spike_ratio_vs_prev": local_jump,
        "detected": all(local_jump[s] >= 5.0 for s in spike_steps),
        "fix": "log CE and aux terms separately; unit-test loss scales in CI",
        "bug": "z-loss/aux coefficient 100 instead of 1e-4 on rare steps/configs",
    }
    dist.destroy_process_group()
    return out


# ---------------------------------------------------------------------------
# Silent drift — real DDP control-flow bugs on the LM
# ---------------------------------------------------------------------------


def case_numerical_drift(rank: int, world_size: int, backend: str, port: int) -> dict:
    """(1) rank0-only grad clip after allreduce (2) rank0-only EMA copy-back into student."""
    _init_pg(rank, world_size, backend, port)
    device = _device(rank, backend)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    onset = 8

    # --- rank0-only grad clip -------------------------------------------------
    torch.manual_seed(0)
    model = TinyTransformerLM().to(device)
    ddp = _wrap_ddp(model, device)
    opt = torch.optim.AdamW(ddp.parameters(), lr=3e-4)
    param_diffs: list[float] = []
    losses: list[float] = []
    for step in range(24):
        ids, labels, mask = _make_batch(4, 32, device, seed=2000 + step * 19 + rank)
        opt.zero_grad(set_to_none=True)
        loss = _lm_loss(ddp(ids, key_padding_mask=mask), labels)
        loss.backward()
        if step >= onset and rank == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.05)
        opt.step()
        param_diffs.append(_max_param_diff(model))
        losses.append(float(loss.detach().cpu()))

    # Control: clip on all ranks.
    torch.manual_seed(0)
    model_ok = TinyTransformerLM().to(device)
    ddp_ok = _wrap_ddp(model_ok, device)
    opt_ok = torch.optim.AdamW(ddp_ok.parameters(), lr=3e-4)
    param_diffs_ok: list[float] = []
    for step in range(24):
        ids, labels, mask = _make_batch(4, 32, device, seed=2000 + step * 19 + rank)
        opt_ok.zero_grad(set_to_none=True)
        loss = _lm_loss(ddp_ok(ids, key_padding_mask=mask), labels)
        loss.backward()
        if step >= onset:
            torch.nn.utils.clip_grad_norm_(model_ok.parameters(), max_norm=0.05)
        opt_ok.step()
        param_diffs_ok.append(_max_param_diff(model_ok))

    # --- rank0-only EMA copy-back (KD / mean-teacher anti-pattern) ------------
    torch.manual_seed(0)
    student = TinyTransformerLM().to(device)
    ema = TinyTransformerLM().to(device)
    ema.load_state_dict(student.state_dict())
    ddp_s = _wrap_ddp(student, device)
    opt_s = torch.optim.SGD(ddp_s.parameters(), lr=0.05)
    ema_param_diffs: list[float] = []
    student_param_diffs: list[float] = []
    for step in range(24):
        ids, labels, mask = _make_batch(4, 32, device, seed=3000 + step * 23 + rank)
        opt_s.zero_grad(set_to_none=True)
        loss = _lm_loss(ddp_s(ids, key_padding_mask=mask), labels)
        loss.backward()
        opt_s.step()
        # EMA update on all ranks (identical if student synced).
        with torch.no_grad():
            for p_e, p_s in zip(ema.parameters(), student.parameters(), strict=True):
                p_e.mul_(0.95).add_(p_s, alpha=0.05)
        if step >= onset and step % 4 == 0 and rank == 0:
            # Bug seen in distillation experiments: "refresh student from EMA" only on rank0.
            student.load_state_dict(ema.state_dict())
        student_param_diffs.append(_max_param_diff(student))
        ema_param_diffs.append(_max_param_diff(ema))

    recipes = [
        {
            "name": "rank0_only_grad_clip",
            "triggered": param_diffs[-1] > 1e-4,
            "where": "parameters",
            "detail": (
                f"final max|Δparam|={param_diffs[-1]:.3e} "
                f"(all-rank clip control={param_diffs_ok[-1]:.3e})"
            ),
            "fix": "clip on every rank; never gate clip/unscale/step on is_main",
        },
        {
            "name": "rank0_only_ema_copyback",
            "triggered": student_param_diffs[-1] > 1e-4,
            "where": "parameters",
            "detail": (
                f"EMA→student copy on rank0 only → student max|Δparam|="
                f"{student_param_diffs[-1]:.3e} (ema still {ema_param_diffs[-1]:.3e})"
            ),
            "fix": "broadcast EMA/student after copy-back, or do copy-back on all ranks",
        },
    ]

    out = {
        "case": "numerical_drift",
        "rank": rank,
        "backend": backend,
        "model": "TinyTransformerLM",
        "onset_step": onset,
        "recipes": recipes,
        "n_recipes": len(recipes),
        "n_triggered": sum(1 for r in recipes if r["triggered"]),
        "triggered_names": [r["name"] for r in recipes if r["triggered"]],
        "losses": losses,
        "max_param_diff": param_diffs,
        "max_param_diff_control": param_diffs_ok,
        "final_max_param_diff": param_diffs[-1],
        "control_final_max_param_diff": param_diffs_ok[-1],
        "ema_student_param_diff": student_param_diffs,
        "ema_shadow_param_diff": ema_param_diffs,
        "bn_max_buffer_diff": student_param_diffs,  # figure compat: second series
        "bn_max_param_diff": ema_param_diffs,
        "bn_final_max_buffer_diff": student_param_diffs[-1],
        "bn_final_max_param_diff": ema_param_diffs[-1],
        "drift_detected": all(r["triggered"] for r in recipes),
        "loss_still_finite": all(_finite(v) for v in losses),
        "fix": "checksum params across ranks; never rank-gate clip or EMA copy-back",
    }
    dist.destroy_process_group()
    return out


# ---------------------------------------------------------------------------
# Memory leak — debug logits ring buffer (real logging anti-pattern)
# ---------------------------------------------------------------------------


def case_memory_leak(rank: int, world_size: int, backend: str, port: int) -> dict:
    """Keep every step's logits 'to log top-k later' → monotonic CUDA growth."""
    _init_pg(rank, world_size, backend, port)
    device = _device(rank, backend)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(0)
    model = TinyTransformerLM().to(device)
    ddp = _wrap_ddp(model, device)
    opt = torch.optim.AdamW(ddp.parameters(), lr=3e-4)

    # Bug: unbounded debug buffer (common when prototyping wandb/top-k dumps).
    debug_logits: list[torch.Tensor] = []
    allocated: list[float] = []
    for step in range(30):
        ids, labels, mask = _make_batch(4, 64, device, seed=4000 + step)
        opt.zero_grad(set_to_none=True)
        logits = ddp(ids, key_padding_mask=mask)
        debug_logits.append(logits.detach())  # should have been .cpu() + bounded ring
        loss = _lm_loss(logits, labels)
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
            allocated.append(torch.cuda.memory_allocated(device) / (1024**2))
        else:
            allocated.append(sum(t.numel() * t.element_size() for t in debug_logits) / (1024**2))

    debug_logits.clear()
    allocated_fixed: list[float] = []
    ring: list[torch.Tensor] = []
    for step in range(30):
        ids, labels, mask = _make_batch(4, 64, device, seed=5000 + step)
        opt.zero_grad(set_to_none=True)
        logits = ddp(ids, key_padding_mask=mask)
        ring.append(logits.detach())
        if len(ring) > 1:
            ring.pop(0)
        loss = _lm_loss(logits, labels)
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
            allocated_fixed.append(torch.cuda.memory_allocated(device) / (1024**2))
        else:
            allocated_fixed.append(sum(t.numel() * t.element_size() for t in ring) / (1024**2))

    growth = allocated[-1] - allocated[0]
    fixed_growth = allocated_fixed[-1] - allocated_fixed[0]
    out = {
        "case": "memory_leak",
        "rank": rank,
        "backend": backend,
        "device": str(device),
        "model": "TinyTransformerLM",
        "allocated_mb_leaky": allocated,
        "allocated_mb_fixed": allocated_fixed,
        "growth_mb": growth,
        "fixed_growth_mb": fixed_growth,
        "leak_detected": growth >= 1.0 and growth > fixed_growth + 0.5,
        "fixed_flat": abs(fixed_growth) <= 0.5,
        "bug": "unbounded list of detached logits retained on device for later logging",
        "fix": "bounded CPU ring buffer / log scalars only; never retain step tensors",
    }
    dist.destroy_process_group()
    return out


# ---------------------------------------------------------------------------
# Straggler — real seqlen skew (long-doc vs short-doc bucket)
# ---------------------------------------------------------------------------


def case_straggler(rank: int, world_size: int, backend: str, port: int) -> dict:
    """Rank 1 always draws long sequences; collective wait stretches for everyone."""
    _init_pg(rank, world_size, backend, port)
    device = _device(rank, backend)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(0)
    # Deeper/wider so T² attention skew is visible on A10G, not just CPU.
    model = TinyTransformerLM(dim=96, n_layer=4, n_heads=4, max_seq=512).to(device)
    ddp = _wrap_ddp(model, device)
    opt = torch.optim.AdamW(ddp.parameters(), lr=3e-4)

    straggler = 1 if world_size > 1 else 0
    onset = 6
    short_len, long_len = 16, 256
    step_ms: list[float] = []
    collective_ms: list[float] = []
    local_ms: list[float] = []
    tokens_per_step: list[int] = []

    for step in range(18):
        seqlen = long_len if (rank == straggler and step >= onset) else short_len
        tokens_per_step.append(2 * seqlen)
        ids, labels, mask = _make_batch(2, seqlen, device, seed=6000 + step * 7 + rank)
        dist.barrier()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        # Local compute = forward only (before DDP backward allreduce waits).
        logits = ddp(ids, key_padding_mask=mask)
        loss = _lm_loss(logits, labels)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_fwd = time.perf_counter()
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        local_ms.append((t_fwd - t0) * 1e3)
        collective_ms.append((t1 - t_fwd) * 1e3)
        step_ms.append((t1 - t0) * 1e3)

    pre = local_ms[2:onset]
    post = local_ms[onset:]
    local_mean = torch.tensor(
        [float(np.median(post))],
        dtype=torch.float32,
        device=device if backend == "nccl" else "cpu",
    )
    gathered = [torch.zeros_like(local_mean) for _ in range(world_size)]
    dist.all_gather(gathered, local_mean)
    means = [float(t.item()) for t in gathered]
    ratio = max(means) / max(min(means), 1e-9)
    # Also flag via token imbalance (the root cause metric in production).
    tok = torch.tensor(
        [float(np.median(tokens_per_step[onset:]))],
        dtype=torch.float32,
        device=device if backend == "nccl" else "cpu",
    )
    tok_g = [torch.zeros_like(tok) for _ in range(world_size)]
    dist.all_gather(tok_g, tok)
    tok_means = [float(t.item()) for t in tok_g]
    tok_ratio = max(tok_means) / max(min(tok_means), 1e-9)

    out = {
        "case": "straggler",
        "rank": rank,
        "backend": backend,
        "model": "TinyTransformerLM",
        "straggler_rank": straggler,
        "onset_step": onset,
        "short_len": short_len,
        "long_len": long_len,
        "step_ms": step_ms,
        "collective_ms": collective_ms,
        "local_compute_ms": local_ms,
        "pre_median_ms": float(np.median(pre)),
        "post_median_ms": float(np.median(post)),
        "slowdown": float(np.median(post) / max(np.median(pre), 1e-9)),
        "per_rank_post_median_ms": means,
        "per_rank_tokens": tok_means,
        "max_over_min_local": ratio,
        "token_imbalance": tok_ratio,
        "detected": ratio >= 1.3 or tok_ratio >= 2.0,
        "bug": "rank draws from long-doc bucket while peers stay short — token skew",
        "fix": "length bucketing / token-budget batching; per-rank tokens/step metrics",
    }
    dist.destroy_process_group()
    return out


# ---------------------------------------------------------------------------
# Bad node — persistent host preprocessing skew (multimodal-style)
# ---------------------------------------------------------------------------


def case_bad_node(rank: int, world_size: int, backend: str, port: int) -> dict:
    """One rank permanently pays heavy host-side work before each step.

    Stand-in for a real class of 'bad node' symptoms in multimodal / tokenization
    heavy jobs: GPU kernels look fine, but pre-collective local wall time is huge
    on one rank every step (noisy neighbor, broken CPU affinity, stuck decoder).
    """
    _init_pg(rank, world_size, backend, port)
    device = _device(rank, backend)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(0)
    model = TinyTransformerLM().to(device)
    ddp = _wrap_ddp(model, device)
    opt = torch.optim.AdamW(ddp.parameters(), lr=3e-4)
    bad = 0

    local_ms: list[float] = []
    step_ms: list[float] = []
    for step in range(16):
        # Host tokenization / feature decode (CPU). Bad rank pays a permanently
        # heavier preprocess bill — same symptom class as a stuck decoder worker
        # or noisy-neighbor CPU starvation (not time.sleep).
        t_host0 = time.perf_counter()
        rng = np.random.default_rng(step + rank * 100)
        # Simulate byte-level encode + n-gram stats over a long document.
        n_docs = 256 if rank == bad else 2
        doc = rng.integers(1, VOCAB, size=(n_docs, 4096), dtype=np.int64)
        # Cheap but real host work: histogram + rolling hash mix.
        hist = np.zeros(VOCAB, dtype=np.int64)
        for row in doc:
            hist += np.bincount(row, minlength=VOCAB)
            _ = int(row.astype(np.uint64).sum() * 2654435761 & 0xFFFFFFFF)
            # Extra CPU pressure: bigram co-occurrence sketch.
            _ = np.bincount((row[:-1] * VOCAB + row[1:]) % 4096, minlength=4096)
        ids, labels, mask = _make_batch(4, 32, device, seed=7000 + step + rank)
        t_host1 = time.perf_counter()

        opt.zero_grad(set_to_none=True)
        loss = _lm_loss(ddp(ids, key_padding_mask=mask), labels)
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        local_ms.append((t_host1 - t_host0) * 1e3)
        step_ms.append((t1 - t_host0) * 1e3)

    local_mean = torch.tensor(
        [float(np.mean(local_ms))],
        dtype=torch.float32,
        device=device if backend == "nccl" else "cpu",
    )
    gathered = [torch.zeros_like(local_mean) for _ in range(world_size)]
    dist.all_gather(gathered, local_mean)
    means = [float(t.item()) for t in gathered]
    healthy = float(min(means)) if means else 0.0
    ratios = [m / max(healthy, 1e-9) for m in means]
    flagged = [i for i, r in enumerate(ratios) if r >= 3.0]

    out = {
        "case": "bad_node",
        "rank": rank,
        "backend": backend,
        "model": "TinyTransformerLM",
        "bad_rank": bad,
        "local_compute_ms": local_ms,
        "step_ms": step_ms,
        "per_rank_local_mean_ms": means,
        "ratio_vs_fastest": ratios,
        "flagged_ranks": flagged,
        "detected": bad in flagged,
        "bug": "persistent host preprocessing inflation on one rank (CPU-bound skew)",
        "fix": "split host vs GPU timers; DCGM/Xid for true device faults; cordon node",
    }
    dist.destroy_process_group()
    return out


# ---------------------------------------------------------------------------
# NCCL hang — empty microbatch continue (classic control-flow desync)
# ---------------------------------------------------------------------------


def case_nccl_hang(rank: int, world_size: int, backend: str, port: int) -> dict:
    """Rank 1 drops an empty pack with ``continue``, skipping DDP backward/collective."""
    _init_pg(rank, world_size, backend, port)
    device = _device(rank, backend)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(0)
    model = TinyTransformerLM().to(device)
    ddp = _wrap_ddp(model, device)
    opt = torch.optim.SGD(ddp.parameters(), lr=0.05)

    # Warmup step — both ranks participate.
    ids, labels, mask = _make_batch(2, 16, device, seed=1)
    opt.zero_grad(set_to_none=True)
    _lm_loss(ddp(ids, key_padding_mask=mask), labels).backward()
    opt.step()
    healthy_ok = True

    hang_rank = 1 if world_size > 1 else -1
    error: str | None = None
    elapsed_ms = 0.0

    if rank == hang_rank:
        # Empty microbatch after filtering — real packing outcome.
        ids = torch.full((2, 16), PAD_ID, device=device, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        valid = int((labels != -100).sum().item())
        if valid == 0:
            # THE bug: skip the step entirely → peer hangs in DDP autograd allreduce.
            return {
                "case": "nccl_hang",
                "rank": rank,
                "backend": backend,
                "hang_rank": hang_rank,
                "healthy_collective_ok": healthy_ok,
                "peer_wait_ms": 0.0,
                "hang_reproduced": True,
                "error": "empty_microbatch_continue_skipped_collective",
                "bug": "if valid_tokens==0: continue  # skips DDP forward/backward",
                "fix": "still participate in collectives / use noop forward; never continue past DDP",
            }

    # Non-hang ranks enter a normal step and block inside DDP backward allreduce.
    t0 = time.perf_counter()
    try:
        ids, labels, mask = _make_batch(2, 16, device, seed=2)
        opt.zero_grad(set_to_none=True)
        _lm_loss(ddp(ids, key_padding_mask=mask), labels).backward()
        opt.step()
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    elapsed_ms = (time.perf_counter() - t0) * 1e3

    out = {
        "case": "nccl_hang",
        "rank": rank,
        "backend": backend,
        "hang_rank": hang_rank,
        "healthy_collective_ok": healthy_ok,
        "peer_wait_ms": elapsed_ms,
        "hang_reproduced": hang_rank >= 0,
        "error": error,
        "bug": "peer skipped collective via empty-batch continue",
        "fix": "NCCL watchdog + identical control flow into every collective",
    }
    with contextlib.suppress(Exception):
        dist.destroy_process_group()
    return out


# ---------------------------------------------------------------------------
# Throughput cliff — real grad allreduce vs tokens/rank
# ---------------------------------------------------------------------------


def case_throughput_cliff(rank: int, world_size: int, backend: str, port: int) -> dict:
    """Sweep tokens/rank on the LM; tiny microbatches make DDP grad sync dominate."""
    _init_pg(rank, world_size, backend, port)
    device = _device(rank, backend)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(0)
    # Slightly wider model so grad buckets are nontrivial.
    model = TinyTransformerLM(dim=96, n_layer=3, n_heads=4, max_seq=256).to(device)
    ddp = _wrap_ddp(model, device)
    opt = torch.optim.AdamW(ddp.parameters(), lr=3e-4)

    # tokens ≈ batch * seqlen (causal LM).
    configs = [
        (1, 8),
        (1, 16),
        (2, 16),
        (2, 32),
        (4, 32),
        (4, 64),
        (8, 64),
        (8, 128),
    ]
    rows: list[dict] = []
    for bs, seqlen in configs:
        times: list[float] = []
        for step in range(10):
            ids, labels, mask = _make_batch(bs, seqlen, device, seed=8000 + bs * 100 + step + rank)
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss = _lm_loss(ddp(ids, key_padding_mask=mask), labels)
            loss.backward()  # real DDP grad allreduce
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
        steady = times[2:]
        med = float(np.median(steady))
        tokens = bs * seqlen * world_size
        tokens_per_s = tokens / (med / 1e3)
        rows.append(
            {
                "batch_size_per_rank": bs,
                "seqlen": seqlen,
                "tokens_per_rank": bs * seqlen,
                "global_tokens": tokens,
                "median_step_ms": med,
                "tokens_per_s": tokens_per_s,
                "samples_per_s": tokens_per_s,  # figure compat
            }
        )

    tps = [r["tokens_per_s"] for r in rows]
    peak_i = int(np.argmax(tps))
    peak = tps[peak_i]
    cliff = next(
        (
            r
            for r in rows
            if r["tokens_per_s"] < 0.45 * peak
            and r["tokens_per_rank"] < rows[peak_i]["tokens_per_rank"]
        ),
        rows[0],
    )
    out = {
        "case": "throughput_cliff",
        "rank": rank,
        "backend": backend,
        "model": "TinyTransformerLM",
        "world_size": world_size,
        "n_params": sum(p.numel() for p in model.parameters()),
        "sweep": rows,
        "peak_batch_size": rows[peak_i]["tokens_per_rank"],
        "peak_samples_per_s": peak,
        "cliff_batch_size": cliff["tokens_per_rank"],
        "cliff_samples_per_s": cliff["tokens_per_s"],
        "cliff_ratio_vs_peak": cliff["tokens_per_s"] / max(peak, 1e-9),
        "fix": "raise tokens/rank (seqlen or microbatch); overlap comm; avoid tiny packs",
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
        result_queue.put(("ok", rank, CASE_FNS[case](rank, world_size, backend, port)))
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
    port = base_port + (abs(hash(case)) % 1000)
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_worker, args=(rank, world_size, case, backend, port, result_queue)
        )
        p.start()
        procs.append(p)

    join_timeout = 12.0 if case == "nccl_hang" else 180.0
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

    results.sort(key=lambda r: r.get("rank", 0))
    primary = results[0] if results else {}
    summary: dict[str, Any] = {
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

    if case == "nccl_hang":
        peer_errs = [
            r.get("error")
            for r in results
            if r.get("rank") != r.get("hang_rank") and r.get("error")
        ]
        empty_skip = any(
            r.get("error") == "empty_microbatch_continue_skipped_collective" for r in results
        )
        summary["hang_confirmed"] = bool(timed_out) or bool(peer_errs) or empty_skip
        if timed_out:
            summary["symptom"] = (
                f"ranks {timed_out} blocked after empty-microbatch continue on rank "
                f"{primary.get('hang_rank')}"
            )
        elif peer_errs:
            summary["symptom"] = f"peer collective error after empty-batch skip: {peer_errs[0]}"
        elif empty_skip:
            summary["symptom"] = "empty_microbatch_continue_skipped_collective"
        else:
            summary["symptom"] = "hang not observed"
        summary["peer_errors"] = peer_errs
    elif primary:
        for key in (
            "nan_detected",
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
            "max_over_min_local",
            "bn_final_max_buffer_diff",
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
            "model": "TinyTransformerLM",
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
    print(
        f"modal bench: device={torch.cuda.get_device_name(0) if prefer else 'cpu'} "
        f"n_gpu={n_gpu} backend={'nccl' if prefer_nccl else 'gloo'} ws={ws}"
    )
    if which == "all":
        results = run_all(world_size=ws, prefer_nccl=prefer_nccl)
    else:
        results = {
            "meta": {
                "torch": torch.__version__,
                "cuda_available": prefer,
                "device_name": torch.cuda.get_device_name(0) if prefer else "cpu",
                "n_gpu": n_gpu,
                "world_size": ws,
                "prefer_nccl": prefer_nccl,
                "backend": _backend(prefer_nccl),
                "gpu_slot": GPU,
                "model": "TinyTransformerLM",
            },
            "cases": {which: run_case(which, world_size=ws, prefer_nccl=prefer_nccl)},
        }
    results.setdefault("meta", {})
    results["meta"]["source"] = "modal"
    results["meta"]["gpu_slot"] = GPU
    results["meta"]["n_gpu"] = n_gpu
    return results


def _print_summary(results: dict) -> None:
    meta = results.get("meta", {})
    print(f"device={meta.get('device_name')} backend={meta.get('backend')} model={meta.get('model')}")
    for name, c in results.get("cases", {}).items():
        p = c.get("primary", {})
        if name == "nan":
            print(
                f"  nan: {p.get('n_triggered')}/{p.get('n_recipes')} → {p.get('triggered_names')}"
            )
        elif name == "loss_spike":
            print(f"  loss_spike: detected={p.get('detected')} ratios={p.get('spike_ratio_vs_median')}")
        elif name == "numerical_drift":
            print(
                f"  drift: {p.get('triggered_names')} "
                f"param_diff={p.get('final_max_param_diff')} "
                f"ema_diff={p.get('bn_final_max_buffer_diff')}"
            )
        elif name == "memory_leak":
            g = p.get("growth_mb")
            print(f"  leak: detected={p.get('leak_detected')} growth_mb={g}")
        elif name == "straggler":
            print(
                f"  straggler: detected={p.get('detected')} "
                f"slowdown={p.get('slowdown')} max/min={p.get('max_over_min_local')}"
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
                f"  cliff: peak_tokens/rank={p.get('peak_batch_size')} "
                f"cliff={p.get('cliff_batch_size')} ratio={p.get('cliff_ratio_vs_peak')}"
            )


@app.local_entrypoint()
def main(case: str = "all") -> None:
    which = str(case).strip().lower()
    if which not in {"all", *CASES}:
        raise SystemExit(f"Unknown --case {case!r}")
    results = bench.remote(which)
    path = Path("playground/dist_failure_results.json")
    path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {path}")
    _print_summary(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="all", choices=("all", *CASES))
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--nccl", action="store_true")
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
                "model": "TinyTransformerLM",
            },
            "cases": {
                args.case: run_case(
                    args.case, world_size=args.world_size, prefer_nccl=args.nccl
                )
            },
        }
    results["meta"]["source"] = results["meta"].get("source", "local")
    path = Path("playground/dist_failure_results.json")
    path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {path}")
    _print_summary(results)
