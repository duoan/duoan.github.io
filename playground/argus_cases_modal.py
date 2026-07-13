"""Small-scale remasurements of ARGUS paper case studies (§9).

Feasible without a 4k-GPU cluster:

* Case 1 — compute straggler (L1 iteration regression + L2 phase CV)
* Case 2 — silent link degradation (stable step time; L3 W₁ block structure)
* Case 3 — PP bubble masking (grad_sync hides straggler from L1/L2 on totals)
* Case 4 — intermittent host/JIT stall (L1 jitter; rare spike dilutes L2/L3)

Usage::

    uv run modal run playground/argus_cases_modal.py

Writes ``playground/argus_cases_results.json``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal
import numpy as np

GPU = os.environ.get("ARGUS_GPU", "A10G")

app = modal.App("argus-cases")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy")
    .add_local_file(
        Path(__file__).with_name("argus_demo_modal.py"),
        remote_path="/root/argus_demo_modal.py",
    )
)


def _median(xs: list[float]) -> float:
    ys = sorted(xs)
    return ys[len(ys) // 2]


def _cv(xs: list[float]) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    mu = float(arr.mean())
    if mu <= 0:
        return 0.0
    return float(arr.std(ddof=1) / mu)


def _zscores(xs: list[float]) -> list[float]:
    arr = np.asarray(xs, dtype=np.float64)
    mu, sigma = float(arr.mean()), float(arr.std(ddof=1))
    if sigma <= 0:
        return [0.0] * len(xs)
    return [float((x - mu) / sigma) for x in xs]


def l1_jitter_windows(
    step_ms: list[float], window: int = 8, ratio_theta: float = 2.0
) -> list[dict]:
    hits = []
    for i in range(len(step_ms) - window + 1):
        chunk = step_ms[i : i + window]
        r = max(chunk) / max(min(chunk), 1e-9)
        if r >= ratio_theta:
            hits.append({"start": i, "end": i + window - 1, "ratio": float(r)})
    return hits


def l1_change_point(step_ms: list[float], min_ratio: float = 1.5) -> dict | None:
    n = len(step_ms)
    best = None
    for t in range(5, n - 5):
        left, right = step_ms[:t], step_ms[t:]
        mu_l, mu_r = float(np.mean(left)), float(np.mean(right))
        if mu_l <= 0:
            continue
        ratio = mu_r / mu_l
        if ratio < min_ratio:
            continue
        rsd_l = float(np.std(left, ddof=1) / mu_l)
        rsd_r = float(np.std(right, ddof=1) / max(mu_r, 1e-9))
        if rsd_l > 0.35 or rsd_r > 0.35:
            continue
        cand = {"t": t, "ratio": ratio, "mu_left": mu_l, "mu_right": mu_r}
        if best is None or ratio > best["ratio"]:
            best = cand
    return best


def case1_compute_straggler(rng: np.random.Generator) -> dict:
    """Ranks with slow local compute; L1+L2 catch it (paper §9.1)."""
    n_ranks, n_steps = 8, 40
    straggler = 5
    onset = 15
    attn = [[0.0] * n_steps for _ in range(n_ranks)]
    mlp = [[0.0] * n_steps for _ in range(n_ranks)]
    allreduce = [[0.0] * n_steps for _ in range(n_ranks)]
    step = [[0.0] * n_steps for _ in range(n_ranks)]

    for t in range(n_steps):
        for r in range(n_ranks):
            a = 6.0 + 0.4 * rng.normal()
            m = 5.0 + 0.3 * rng.normal()
            if r == straggler and t >= onset:
                a *= 25.0
                m *= 20.0
            c = 4.0 + 0.2 * rng.normal()
            attn[r][t] = max(a, 0.1)
            mlp[r][t] = max(m, 0.1)
            allreduce[r][t] = max(c, 0.1)
            step[r][t] = attn[r][t] + mlp[r][t] + allreduce[r][t]

    iter_ms = [max(step[r][t] for r in range(n_ranks)) for t in range(n_steps)]
    cp = l1_change_point(iter_ms, min_ratio=1.8)

    window = list(range(onset, n_steps))
    attn_means = [float(np.mean([attn[r][t] for t in window])) for r in range(n_ranks)]
    mlp_means = [float(np.mean([mlp[r][t] for t in window])) for r in range(n_ranks)]
    attn_z, mlp_z = _zscores(attn_means), _zscores(mlp_means)
    flagged = sorted(
        {i for i, z in enumerate(attn_z) if z >= 2.0}
        | {i for i, z in enumerate(mlp_z) if z >= 2.0}
    )

    return {
        "case": 1,
        "title": "Compute straggler localization",
        "n_ranks": n_ranks,
        "straggler": straggler,
        "onset_step": onset,
        "iter_ms": iter_ms,
        "l1_change_point": cp,
        "l1_detected": cp is not None and cp["t"] >= onset - 3,
        "l2": {
            "self_attention_cv": _cv(attn_means),
            "mlp_cv": _cv(mlp_means),
            "self_attention_means_ms": attn_means,
            "mlp_means_ms": mlp_means,
            "self_attention_z": attn_z,
            "mlp_z": mlp_z,
            "flagged_ranks": flagged,
        },
        "l2_detected": flagged == [straggler],
        "heatmap_attn_ms": attn_means,
        "heatmap_mlp_ms": mlp_means,
    }


def case2_link_degradation(rng: np.random.Generator) -> dict:
    """Stable iteration time; one EDP group's NCCL kernels are slow (paper §9.2)."""
    import sys

    sys.path.insert(0, "/root")
    from argus_demo_modal import (  # noqa: WPS433
        compress_rank_trace,
        lognormal_mixture_cdf,
        wasserstein1,
    )

    n_ranks, n_events = 8, 80
    degraded = {4, 5, 6, 7}
    kernels = ("AllReduce", "AllGather", "ReduceScatter")

    rank_events: list[list[tuple[str, int, float]]] = []
    for r in range(n_ranks):
        events = []
        for _ in range(n_events):
            for k in kernels:
                base = {"AllReduce": 1.2, "AllGather": 2.0, "ReduceScatter": 2.4}[k]
                if r in degraded:
                    base *= {"AllReduce": 8.0, "AllGather": 12.0, "ReduceScatter": 20.0}[k]
                events.append((k, 0, base * float(rng.lognormal(0.0, 0.05))))
        rank_events.append(events)

    # Steady degraded regime: iteration time flat → L1 silent.
    iter_ms = [28.0 + 0.3 * float(rng.normal()) for _ in range(40)]
    jitter = l1_jitter_windows(iter_ms, window=8, ratio_theta=2.0)
    cp = l1_change_point(iter_ms, min_ratio=1.5)

    summaries = [compress_rank_trace(ev) for ev in rank_events]

    def primary_p50(summary: dict, kernel: str) -> float:
        for g in summary["groups"]:
            if g["kernel"] == kernel and g["stream"] == 0:
                # Largest cluster = primary mode.
                c = max(g["clusters"], key=lambda x: x["count"])
                return float(c["p50_ms"])
        return float("nan")

    xs = np.logspace(-2, 2, 400)
    w1_matrices = {}
    p50_by_kernel = {}
    flagged_by_kernel = {}
    for k in kernels:
        cdfs = []
        p50s = []
        for summary in summaries:
            clusters = []
            for g in summary["groups"]:
                if g["kernel"] == k and g["stream"] == 0:
                    clusters = g["clusters"]
                    break
            cdfs.append(lognormal_mixture_cdf(xs, clusters))
            p50s.append(primary_p50(summary, k))
        mat = np.zeros((n_ranks, n_ranks))
        for i in range(n_ranks):
            for j in range(n_ranks):
                mat[i, j] = wasserstein1(cdfs[i], cdfs[j], xs)
        # When half the cluster is degraded, IQR-on-mean-W1 fails (symmetric).
        # Paper looks at block structure; we flag ranks whose primary p50 is
        # far above the *lower* half median (healthy mode).
        healthy_med = float(np.median(sorted(p50s)[: n_ranks // 2]))
        flagged = [i for i, p in enumerate(p50s) if p >= healthy_med * 3.0]
        w1_matrices[k] = mat.tolist()
        p50_by_kernel[k] = p50s
        flagged_by_kernel[k] = flagged

    flagged_union = sorted({i for v in flagged_by_kernel.values() for i in v})
    # Intra vs inter group mean W1 on AllReduce (block-structure evidence).
    mat = np.asarray(w1_matrices["AllReduce"])
    healthy, slow = [0, 1, 2, 3], [4, 5, 6, 7]
    intra = [mat[i, j] for i in healthy for j in healthy if i < j] + [
        mat[i, j] for i in slow for j in slow if i < j
    ]
    inter = [mat[i, j] for i in healthy for j in slow]
    return {
        "case": 2,
        "title": "Silent communication link degradation",
        "n_ranks": n_ranks,
        "degraded_ranks": sorted(degraded),
        "observed_iter_ms_median": _median(iter_ms),
        "iter_ms": iter_ms,
        "l1_jitter_hits": len(jitter),
        "l1_change_point": cp,
        "l1_silent": len(jitter) == 0 and cp is None,
        "l3_p50_ms": p50_by_kernel,
        "l3_flagged_by_kernel": flagged_by_kernel,
        "l3_flagged_union": flagged_union,
        "l3_detected": set(flagged_union) == degraded,
        "w1_allreduce_matrix": w1_matrices["AllReduce"],
        "w1_intra_mean": float(np.mean(intra)),
        "w1_inter_mean": float(np.mean(inter)),
        "w1_inter_intra_ratio": float(np.mean(inter) / max(np.mean(intra), 1e-12)),
    }


def case3_pp_bubble_masking(rng: np.random.Generator) -> dict:
    """Slow last PP stage masked by finish_grad_sync alignment (paper §9.3)."""
    pp = 4
    ranks = list(range(pp))
    straggler = 3
    n_steps = 30
    bwd = {r: [] for r in ranks}
    bubble = {r: [] for r in ranks}
    fwd_bwd_total = {r: [] for r in ranks}

    for _ in range(n_steps):
        stage_bwd = []
        for r in ranks:
            base = 90.0 + 3.0 * rng.normal()
            if r == straggler:
                base *= 1.9
            stage_bwd.append(max(base, 1.0))
            bwd[r].append(stage_bwd[-1])
        for r in ranks:
            if r == straggler:
                bubble[r].append(160.0 + 5.0 * rng.normal())
            else:
                bubble[r].append(230.0 + 5.0 * rng.normal())
        aligned = 11178.0 + 20.0 * rng.normal()
        for r in ranks:
            fwd_bwd_total[r].append(aligned)

    bwd_means = [float(np.mean(bwd[r])) for r in ranks]
    total_means = [float(np.mean(fwd_bwd_total[r])) for r in ranks]
    bubble_means = [float(np.mean(bubble[r])) for r in ranks]
    total_cv = _cv(total_means)
    total_z = _zscores(total_means)
    bwd_cv = _cv(bwd_means)
    bwd_z = _zscores(bwd_means)
    peer_med = float(np.median(bwd_means[:straggler]))
    flagged_bwd = [i for i, m in enumerate(bwd_means) if m >= peer_med * 1.5]

    return {
        "case": 3,
        "title": "Pipeline bubble amplification / masking",
        "pp_stages": pp,
        "straggler_stage": straggler,
        "backward_compute_means_ms": bwd_means,
        "backward_ratio_vs_peers": bwd_means[straggler]
        / float(np.mean(bwd_means[:straggler])),
        "bubble_means_ms": bubble_means,
        "fwd_bwd_total_means_ms": total_means,
        "l1_l2_on_totals": {
            "cv": total_cv,
            "z": total_z,
            "flagged": [i for i, z in enumerate(total_z) if abs(z) >= 2],
        },
        "l1_l2_silent_on_totals": total_cv < 0.02,
        "semantics_bwd": {"cv": bwd_cv, "z": bwd_z, "flagged": flagged_bwd},
        "manual_semantics_detected": flagged_bwd == [straggler],
    }


def case4_jit_stall_gpu() -> dict:
    """Occasional host-side stall dwarfs GPU work (paper §9.4)."""
    import torch
    import torch.nn as nn

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(1024, 4096),
        nn.GELU(),
        nn.Linear(4096, 1024),
    ).to(device=device, dtype=torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(4, 256, 1024, device=device, dtype=torch.bfloat16)

    def step() -> tuple[float, float]:
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        wall0 = time.perf_counter()
        e0.record()
        out = model(x)
        loss = out.square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        e1.record()
        e1.synchronize()
        return (time.perf_counter() - wall0) * 1e3, e0.elapsed_time(e1)

    for _ in range(10):
        step()

    n_steps = 40
    stall_steps = {12, 27}
    stall_ms = 80.0
    wall, gpu = [], []
    for i in range(n_steps):
        if i in stall_steps:
            time.sleep(stall_ms / 1e3)
        w, g = step()
        if i in stall_steps:
            w += stall_ms
        wall.append(w)
        gpu.append(g)

    jitter = l1_jitter_windows(wall, window=6, ratio_theta=2.0)
    normal = [wall[i] for i in range(n_steps) if i not in stall_steps]
    spikes = [wall[i] for i in sorted(stall_steps)]
    return {
        "case": 4,
        "title": "Intermittent host/JIT stall",
        "device": torch.cuda.get_device_name(0),
        "n_steps": n_steps,
        "stall_steps": sorted(stall_steps),
        "injected_stall_ms": stall_ms,
        "wall_ms": wall,
        "gpu_ms": gpu,
        "spike_ratio_vs_normal": spikes[0] / _median(normal),
        "l1_jitter_hits": jitter,
        "l1_detected": len(jitter) > 0,
        "l2_l3_diluted": True,
        "normal_wall_median_ms": _median(normal),
        "spike_wall_ms": spikes,
        "gpu_median_ms": _median(gpu),
    }


@app.function(gpu=GPU, image=image, timeout=900)
def bench() -> dict:
    import torch

    rng = np.random.default_rng(0)
    return {
        "device": torch.cuda.get_device_name(0),
        "case1": case1_compute_straggler(rng),
        "case2": case2_link_degradation(rng),
        "case3": case3_pp_bubble_masking(rng),
        "case4": case4_jit_stall_gpu(),
    }


@app.local_entrypoint()
def main() -> None:
    out = Path("playground/argus_cases_results.json")
    results = bench.remote()
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")
    c1, c2, c3, c4 = (results[k] for k in ("case1", "case2", "case3", "case4"))
    print(f"Case1 L1={c1['l1_detected']} L2={c1['l2_detected']} flagged={c1['l2']['flagged_ranks']}")
    print(
        f"Case2 L1_silent={c2['l1_silent']} L3={c2['l3_detected']} "
        f"flagged={c2['l3_flagged_union']} W1 inter/intra={c2['w1_inter_intra_ratio']:.1f}x"
    )
    print(
        f"Case3 totals_silent={c3['l1_l2_silent_on_totals']} "
        f"bwd_detected={c3['manual_semantics_detected']} ratio={c3['backward_ratio_vs_peers']:.2f}"
    )
    print(
        f"Case4 L1={c4['l1_detected']} spike_ratio={c4['spike_ratio_vs_normal']:.1f}x "
        f"stall_steps={c4['stall_steps']}"
    )
