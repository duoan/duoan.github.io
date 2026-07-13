"""Generate figures for the ARGUS paper-reading blog post.

Usage (from repo root)::

    uv run python playground/argus_demo_figures.py \\
        --results playground/argus_demo_results.json \\
        --out content/posts/argus-tracing-at-10000-gpu-scale
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from argus_demo_modal import gaussian_kde_grid, lognormal_mixture_cdf

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#94a3b8",
        "axes.labelcolor": "#1f2933",
        "axes.titlecolor": "#1f2933",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
    }
)


def kde_cluster_figure(durations_ms: list[float], clusters: list[dict], out: Path) -> None:
    arr = np.asarray(durations_ms, dtype=np.float64)
    log_d = np.log(arr)
    lo, hi = float(log_d.min()), float(log_d.max())
    pad = max(0.05 * (hi - lo), 0.05)
    grid = np.linspace(lo - pad, hi + pad, 256)
    density = gaussian_kde_grid(log_d, grid)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    ax.plot(grid, density, color="#2563eb", lw=2)
    ax.set_xlabel("log(duration ms)")
    ax.set_ylabel("KDE density")
    ax.set_title("(a) KDE valley detection")

    ax = axes[1]
    ax.hist(arr, bins=30, color="#94a3b8", alpha=0.55, edgecolor="white", density=True)
    colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2"]
    for i, c in enumerate(clusters):
        lo_ms = math.exp(c["log_lo"])
        hi_ms = math.exp(c["log_hi"])
        ax.axvspan(lo_ms, hi_ms, color=colors[i % len(colors)], alpha=0.25, label=f"cluster {i+1}")
        ax.axvline(c["p50_ms"], color=colors[i % len(colors)], ls="--", lw=1.5)
    ax.set_xlabel("duration (ms)")
    ax.set_title("(b) clusters → (count, p50, p99)")
    ax.legend(fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(out / "kde_compression.svg", bbox_inches="tight")
    plt.close(fig)


def w1_matrix_figure(results: dict, out: Path) -> None:
    groups = results["compression"]["example_rank0"]["groups"]
    target = results["l3_detection"]
    kernel, stream = target["kernel"], target["stream"]

    n = results["n_ranks"]
    xs = np.logspace(-3, 1.5, 400)

    # Build CDFs from rank0 clusters and straggler scaling for visualization.
    base_clusters = next(g for g in groups if g["kernel"] == kernel and g["stream"] == stream)["clusters"]
    straggler = results["simulated_straggler"]
    factor = results["slow_factor"]

    cdfs = []
    for r in range(n):
        clusters = []
        for c in base_clusters:
            cc = dict(c)
            if r == straggler:
                cc["p50_ms"] *= factor
                cc["p99_ms"] *= factor
            clusters.append(cc)
        cdfs.append(lognormal_mixture_cdf(xs, clusters))

    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = float(np.trapezoid(np.abs(cdfs[i] - cdfs[j]), xs))

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(mat, cmap="YlOrRd")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"r{i}" for i in range(n)])
    ax.set_yticklabels([f"r{i}" for i in range(n)])
    ax.set_xlabel("rank")
    ax.set_ylabel("rank")
    ax.set_title(f"W₁ distance matrix ({kernel})")
    for i in range(n):
        for j in range(n):
            if i == straggler or j == straggler:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#1d4ed8", lw=2))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="W₁")
    fig.tight_layout()
    fig.savefig(out / "w1_distance_matrix.svg", bbox_inches="tight")
    plt.close(fig)


def progressive_diagnosis_figure(out: Path) -> None:
    levels = [
        ("L1", "Iteration time", "Anomaly window", "#4C72B0"),
        ("L2", "Phase duration", "Straggler rank + phase", "#55A868"),
        ("L3", "Kernel stats", "Degraded kernel", "#C44E52"),
        ("L4", "Perfetto trace", "Root cause (manual)", "#8172B2"),
        ("L5", "CPU call stack", "Host stall (manual)", "#CCB974"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")

    x = 0.6
    for label, source, purpose, color in levels:
        ax.add_patch(plt.Rectangle((x, 0.55), 1.5, 0.9, color=color, alpha=0.85))
        ax.text(x + 0.75, 1.0, label, ha="center", va="center", color="white", fontweight="bold")
        ax.text(x + 0.75, 0.25, source, ha="center", va="center", fontsize=9, color="#334155")
        ax.text(x + 0.75, 1.65, purpose, ha="center", va="center", fontsize=9, color="#1f2933")
        if x > 0.6:
            ax.annotate("", xy=(x, 1.0), xytext=(x - 0.35, 1.0), arrowprops=dict(arrowstyle="->", color="#64748b"))
        x += 1.85

    ax.text(5, 2.35, "ARGUS progressive diagnosis (10k ranks → few suspects)", ha="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "progressive_diagnosis.svg", bbox_inches="tight")
    plt.close(fig)


def architecture_figure(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        (0.3, "Trace Producer", "py-spy · semantics · CUPTI", "#dbeafe"),
        (2.5, "Processor", "Perfetto + KDE compress", "#dcfce7"),
        (4.7, "Storage", "metrics + object store", "#fef3c7"),
        (6.9, "Analysis", "L1–L3 auto detect", "#fee2e2"),
    ]
    for x, title, sub, color in boxes:
        ax.add_patch(plt.Rectangle((x, 0.8), 1.8, 1.2, facecolor=color, edgecolor="#64748b"))
        ax.text(x + 0.9, 1.55, title, ha="center", va="center", fontweight="bold")
        ax.text(x + 0.9, 1.05, sub, ha="center", va="center", fontsize=9)
    for x in [2.1, 4.3, 6.5]:
        ax.annotate("", xy=(x + 0.35, 1.4), xytext=(x, 1.4), arrowprops=dict(arrowstyle="->", color="#475569"))

    ax.text(5, 2.55, "ARGUS data path (simplified)", ha="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "argus_architecture.svg", bbox_inches="tight")
    plt.close(fig)


def overhead_figures(overhead: dict, out: Path) -> None:
    """Bar chart of step-time overhead + RSS growth from Modal remasurement."""
    rows = [r for r in overhead["summary"] if r.get("ok")]
    labels = {
        "baseline": "baseline",
        "argus_semantics": "ARGUS-style\nsemantics",
        "torch_profiler_cuda": "torch.profiler\n(CUDA only)",
        "torch_profiler_full": "torch.profiler\n(CPU+CUDA+stack)",
        "nsys": "nsys\nalways-on",
    }
    colors = {
        "baseline": "#94a3b8",
        "argus_semantics": "#55A868",
        "torch_profiler_cuda": "#C44E52",
        "torch_profiler_full": "#8B1E1E",
        "nsys": "#4C72B0",
    }
    names = [r["name"] for r in rows]
    overheads = [r["overhead_pct"] for r in rows]
    rss = [r["rss_delta_mb"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))

    ax = axes[0]
    x = np.arange(len(names))
    bars = ax.bar(
        x,
        overheads,
        color=[colors.get(n, "#64748b") for n in names],
        edgecolor="white",
        width=0.72,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(n, n) for n in names], fontsize=9)
    ax.set_ylabel("step-time overhead vs baseline (%)")
    ax.set_title("(a) Always-on step-time tax")
    ax.axhline(0, color="#94a3b8", lw=0.8)
    for bar, val in zip(bars, overheads, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax = axes[1]
    bars = ax.bar(
        x,
        rss,
        color=[colors.get(n, "#64748b") for n in names],
        edgecolor="white",
        width=0.72,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(n, n) for n in names], fontsize=9)
    ax.set_ylabel("RSS growth over 200 steps (MB)")
    ax.set_title("(b) Trace accumulation (RSS Δ)")
    for bar, val in zip(bars, rss, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(rss) * 0.02,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    device = overhead.get("device", "")
    steps = overhead.get("steps", "")
    fig.suptitle(
        f"Modal remasurement on {device} · launch-heavy MLP · {steps} steps",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out / "overhead_comparison.svg", bbox_inches="tight")
    plt.close(fig)

    # Step-time distributions for the main configs.
    series = overhead.get("step_ms_by_config") or {}
    want = ["baseline", "argus_semantics", "torch_profiler_cuda", "torch_profiler_full", "nsys"]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for name in want:
        if name not in series or not series[name]:
            continue
        arr = np.asarray(series[name], dtype=np.float64)
        ax.plot(
            np.arange(1, len(arr) + 1),
            arr,
            label=labels.get(name, name).replace("\n", " "),
            color=colors.get(name, "#64748b"),
            lw=1.4,
            alpha=0.9,
        )
    ax.set_xlabel("step")
    ax.set_ylabel("step time (ms)")
    ax.set_title("Per-step wall time under always-on profiling")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "overhead_step_series.svg", bbox_inches="tight")
    plt.close(fig)


def case_figures(cases: dict, out: Path) -> None:
    """Figures for remasured ARGUS cases 1–4."""
    c1, c2, c3, c4 = cases["case1"], cases["case2"], cases["case3"], cases["case4"]

    # Case 1: iteration time + phase heatmap
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    ax.plot(c1["iter_ms"], color="#1f2933", lw=1.6)
    ax.axvline(c1["onset_step"], color="#C44E52", ls="--", lw=1.2, label="straggler onset")
    if c1.get("l1_change_point"):
        ax.axvline(c1["l1_change_point"]["t"], color="#4C72B0", ls=":", lw=1.4, label="L1 change-point")
    ax.set_xlabel("step")
    ax.set_ylabel("iteration time (ms, max across ranks)")
    ax.set_title("Case 1 · L1 sees regression")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    data = np.vstack([c1["heatmap_attn_ms"], c1["heatmap_mlp_ms"]])
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["self_attention", "mlp"])
    ax.set_xticks(range(c1["n_ranks"]))
    ax.set_xticklabels([f"r{i}" for i in range(c1["n_ranks"])])
    ax.set_title("Case 1 · L2 phase means (post-onset)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ms")
    fig.tight_layout()
    fig.savefig(out / "case1_compute_straggler.svg", bbox_inches="tight")
    plt.close(fig)

    # Case 2: W1 matrix
    mat = np.asarray(c2["w1_allreduce_matrix"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(mat, cmap="YlOrRd")
    n = c2["n_ranks"]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"r{i}" for i in range(n)])
    ax.set_yticklabels([f"r{i}" for i in range(n)])
    ax.set_title(
        f"Case 2 · AllReduce W₁ (inter/intra={c2['w1_inter_intra_ratio']:.0f}×)"
    )
    for i in c2["degraded_ranks"]:
        ax.add_patch(plt.Rectangle((i - 0.5, -0.5), 1, n, fill=False, edgecolor="#1d4ed8", lw=1.5))
        ax.add_patch(plt.Rectangle((-0.5, i - 0.5), n, 1, fill=False, edgecolor="#1d4ed8", lw=1.5))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="W₁")
    fig.tight_layout()
    fig.savefig(out / "case2_link_degradation.svg", bbox_inches="tight")
    plt.close(fig)

    # Case 3: bwd vs aligned totals
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    stages = [f"PP{i}" for i in range(c3["pp_stages"])]
    ax = axes[0]
    ax.bar(stages, c3["backward_compute_means_ms"], color=["#94a3b8"] * 3 + ["#C44E52"])
    ax.set_ylabel("backward-compute mean (ms)")
    ax.set_title(f"Case 3 · bwd ratio={c3['backward_ratio_vs_peers']:.2f}×")
    ax = axes[1]
    ax.bar(stages, c3["fwd_bwd_total_means_ms"], color="#4C72B0")
    ax.set_ylabel("fwd–bwd total after grad_sync (ms)")
    ax.set_title("Case 3 · L1/L2 on totals are silent")
    fig.tight_layout()
    fig.savefig(out / "case3_pp_masking.svg", bbox_inches="tight")
    plt.close(fig)

    # Case 4: wall vs gpu with spikes
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    xs = np.arange(c4["n_steps"])
    ax.plot(xs, c4["wall_ms"], color="#C44E52", lw=1.6, label="wall step time")
    ax.plot(xs, c4["gpu_ms"], color="#4C72B0", lw=1.4, label="GPU event time")
    for s in c4["stall_steps"]:
        ax.axvline(s, color="#94a3b8", ls="--", lw=1.0)
    ax.set_xlabel("step")
    ax.set_ylabel("ms")
    ax.set_title(
        f"Case 4 · host/JIT stall ({c4['spike_ratio_vs_normal']:.0f}× wall spike; GPU flat)"
    )
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "case4_jit_stall.svg", bbox_inches="tight")
    plt.close(fig)


def cupti_figures(cupti: dict, out: Path) -> None:
    """Overhead bars + top kernel counts from the CUPTI Activity demo."""
    oh = cupti["overhead"]
    tops = cupti["cupti"]["top_kernels"][:8]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    ax = axes[0]
    labels = ["baseline", "CUPTI\nActivity", "CUPTI +\nsemantics"]
    vals = [
        0.0,
        oh["cupti_overhead_pct"],
        oh["cupti_plus_semantics_overhead_pct"],
    ]
    colors = ["#94a3b8", "#2563eb", "#55A868"]
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.7)
    ax.set_ylabel("step-time overhead vs baseline (%)")
    ax.set_title("CUPTI Activity API always-on tax")
    for bar, val in zip(bars, vals, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax = axes[1]
    names = []
    for t in tops:
        n = t["name"]
        if len(n) > 28:
            n = n[:14] + "…" + n[-10:]
        names.append(n)
    counts = [t["count"] for t in tops]
    ax.barh(names[::-1], counts[::-1], color="#4C72B0")
    ax.set_xlabel("records in collection window")
    ax.set_title(
        f"Top CUPTI kernels ({cupti['cupti']['records_in_straggler_window']} total)"
    )
    fig.suptitle(
        f"{cupti.get('device', '')} · {cupti['cupti'].get('kernel_struct', '')}",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out / "cupti_activity.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("playground/argus_demo_results.json"))
    parser.add_argument(
        "--overhead",
        type=Path,
        default=Path("playground/argus_overhead_results.json"),
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("playground/argus_cases_results.json"),
    )
    parser.add_argument(
        "--cupti",
        type=Path,
        default=Path("playground/argus_cupti_results.json"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.results.read_text()) if args.results.exists() else None
    overhead = json.loads(args.overhead.read_text()) if args.overhead.exists() else None
    cases = json.loads(args.cases.read_text()) if args.cases.exists() else None
    cupti = json.loads(args.cupti.read_text()) if args.cupti.exists() else None

    architecture_figure(args.out)
    progressive_diagnosis_figure(args.out)

    if results:
        kde = results["kde_demo"]
        kde_cluster_figure(kde["durations_ms"], kde["clusters"], args.out)
        w1_matrix_figure(results, args.out)

    if overhead:
        overhead_figures(overhead, args.out)

    if cases:
        case_figures(cases, args.out)

    if cupti:
        cupti_figures(cupti, args.out)

    print(f"Wrote figures to {args.out}")


if __name__ == "__main__":
    main()
