"""Capacity-planning calculator for LLM / VLM training.

Pure arithmetic — no GPU required. Emits JSON you can plot or paste into a post.

Usage (from repo root)::

    uv run python playground/capacity_plan.py \\
        --out playground/capacity_plan_results.json

Formulas (planning heuristics, not physics):
    tokens ≈ tokens_per_param × params
    steps  = tokens / (global_batch_seqs × seq_len)
    FLOPs  ≈ 6 × params × tokens          # decoder-only forward+backward rule of thumb
    GPU-h  = FLOPs / (eff_tflops × 1e12) / 3600
    GPUs   = GPU-h / target_wall_hours
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlanInputs:
    name: str
    params: float
    tokens_per_param: float
    seq_len: int
    global_batch_seqs: int
    peak_tflops: float
    mfu: float
    target_days: float
    gpu_mem_gb: float
    precision_bytes: float = 2.0  # bf16
    optimizer_bytes_per_param: float = 8.0  # AdamW fp32 moments-ish
    gpu_hour_usd: float = 4.0


@dataclass
class PlanResult:
    name: str
    params_b: float
    tokens_b: float
    steps: float
    flops: float
    eff_tflops: float
    gpu_hours: float
    gpus_needed: float
    gpus_rounded: int
    wall_days_at_rounded: float
    cost_usd: float
    weight_gb: float
    optimizer_gb: float
    static_gb: float
    fits_single_gpu: bool
    suggested_tp: int
    suggested_dp: int
    microbatch_per_replica: float


def _round_up_multiple(x: float, multiple: int) -> int:
    n = int(x)
    if n < multiple:
        return multiple
    return ((n + multiple - 1) // multiple) * multiple


def suggest_tp(static_gb: float, gpu_mem_gb: float, *, headroom: float = 0.55) -> int:
    """Minimal power-of-two TP so static state fits with activation headroom."""
    budget = gpu_mem_gb * headroom
    for tp in (1, 2, 4, 8):
        if static_gb / tp <= budget:
            return tp
    return 8


def suggest_tp_zero1(
    weight_gb: float, opt_gb: float, gpu_mem_gb: float, dp: int, *, headroom: float = 0.55
) -> int:
    """TP with ZeRO-1: optimizer sharded across DP, weights+grads sharded by TP."""
    budget = gpu_mem_gb * headroom
    for tp in (1, 2, 4, 8):
        per = (weight_gb + weight_gb) / tp + opt_gb / max(dp, 1)
        if per <= budget:
            return tp
    return 8


def plan(inp: PlanInputs) -> PlanResult:
    tokens = inp.tokens_per_param * inp.params
    tokens_per_step = inp.global_batch_seqs * inp.seq_len
    steps = tokens / tokens_per_step
    flops = 6.0 * inp.params * tokens
    eff = inp.peak_tflops * inp.mfu
    gpu_seconds = flops / (eff * 1e12)
    gpu_hours = gpu_seconds / 3600.0
    target_hours = inp.target_days * 24.0
    gpus_needed = gpu_hours / target_hours
    gpus_rounded = _round_up_multiple(gpus_needed, 8)

    weight_gb = inp.params * inp.precision_bytes / 1e9
    opt_gb = inp.params * inp.optimizer_bytes_per_param / 1e9
    grad_gb = weight_gb
    static_gb = weight_gb + opt_gb + grad_gb
    tp = suggest_tp(static_gb, inp.gpu_mem_gb)
    dp = max(1, gpus_rounded // tp)
    while tp * dp > gpus_rounded and dp > 1:
        dp -= 1
    gpus_rounded = tp * dp
    # Re-evaluate TP under ZeRO-1 given this DP; may free a tighter NVLink map.
    tp_z = suggest_tp_zero1(weight_gb, opt_gb, inp.gpu_mem_gb, dp)
    if tp_z < tp and gpus_rounded % tp_z == 0:
        tp = tp_z
        dp = gpus_rounded // tp
    wall_days = gpu_hours / gpus_rounded / 24.0 if gpus_rounded else float("inf")
    micro = inp.global_batch_seqs / dp

    return PlanResult(
        name=inp.name,
        params_b=inp.params / 1e9,
        tokens_b=tokens / 1e9,
        steps=steps,
        flops=flops,
        eff_tflops=eff,
        gpu_hours=gpu_hours,
        gpus_needed=gpus_needed,
        gpus_rounded=gpus_rounded,
        wall_days_at_rounded=wall_days,
        cost_usd=gpu_hours * inp.gpu_hour_usd,
        weight_gb=weight_gb,
        optimizer_gb=opt_gb,
        static_gb=static_gb,
        fits_single_gpu=static_gb < inp.gpu_mem_gb * 0.55,
        suggested_tp=tp,
        suggested_dp=dp,
        microbatch_per_replica=micro,
    )


DEFAULT_CASES = [
    PlanInputs(
        name="7B-A100",
        params=7e9,
        tokens_per_param=20.0,
        seq_len=4096,
        global_batch_seqs=512,
        peak_tflops=312.0,
        mfu=0.40,
        target_days=14.0,
        gpu_mem_gb=80.0,
    ),
    PlanInputs(
        name="30B-A100",
        params=30e9,
        tokens_per_param=20.0,
        seq_len=4096,
        global_batch_seqs=1024,
        peak_tflops=312.0,
        mfu=0.42,
        target_days=21.0,
        gpu_mem_gb=80.0,
    ),
    PlanInputs(
        name="30B-H100",
        params=30e9,
        tokens_per_param=20.0,
        seq_len=4096,
        global_batch_seqs=1024,
        peak_tflops=989.0,  # dense BF16-ish ballpark; planning only
        mfu=0.35,
        target_days=14.0,
        gpu_mem_gb=80.0,
        gpu_hour_usd=6.0,
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("playground/capacity_plan_results.json"))
    args = parser.parse_args()

    results = []
    for inp in DEFAULT_CASES:
        r = plan(inp)
        results.append({"inputs": asdict(inp), "result": asdict(r)})
        print(
            f"{r.name}: tokens={r.tokens_b:.0f}B steps={r.steps:,.0f} "
            f"GPU-h={r.gpu_hours:,.0f} GPUs≈{r.gpus_rounded} "
            f"(TP={r.suggested_tp} DP={r.suggested_dp}) "
            f"static={r.static_gb:.0f}GB cost≈${r.cost_usd:,.0f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cases": results}, indent=2) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
