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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("playground/argus_demo_results.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.results.read_text()) if args.results.exists() else None

    architecture_figure(args.out)
    progressive_diagnosis_figure(args.out)

    if results:
        kde = results["kde_demo"]
        kde_cluster_figure(kde["durations_ms"], kde["clusters"], args.out)
        w1_matrix_figure(results, args.out)
        print(f"Wrote figures to {args.out}")
    else:
        print(f"Wrote conceptual figures to {args.out} (no results at {args.results})")


if __name__ == "__main__":
    main()
