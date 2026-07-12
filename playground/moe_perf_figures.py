"""Generate figures for the Large MoE performance post.

Usage (from repo root)::

    uv run python playground/moe_perf_figures.py \\
        --results playground/moe_perf_results.json \\
        --out content/posts/large-moe-from-sparsity-to-communication
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

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

INK = "#1f2933"
MUTED = "#64748b"
BLUE = "#4C72B0"
GREEN = "#55A868"
RED = "#C44E52"
ORANGE = "#DD8452"
PURPLE = "#8172B3"


def _box(ax, xy, w, h, text, *, fc="#f8fafc", ec=INK, lw=1.4, fontsize=10):
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=INK)


def _arrow(ax, a, b, *, color=MUTED, lw=1.3):
    ax.add_patch(
        FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=11, linewidth=lw, color=color, shrinkA=1, shrinkB=1
        )
    )


def sparsity_trend_figure(out: Path) -> None:
    """Expert pool growth vs activation fraction (illustrative)."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    models = ["Mixtral\n8×7B", "DeepSeek\nMoE", "Qwen3\nMoE", "Kimi\nK2"]
    E = np.array([8, 64, 128, 384])
    k = np.array([2, 6, 8, 8])
    active_frac = k / E * 100

    x = np.arange(len(models))
    bars = ax.bar(x, E, color="#DBE7F5", edgecolor=BLUE, width=0.55, label="# experts (E)")
    ax2 = ax.twinx()
    ax2.plot(x, active_frac, color=RED, marker="o", lw=2, label="k/E activated %")
    ax2.spines["top"].set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Expert count E")
    ax2.set_ylabel("Activated fraction k/E (%)")
    ax.set_title("Expert pools grew; activated fraction kept collapsing")
    ax.set_ylim(0, 420)
    ax2.set_ylim(0, 35)

    for i, e in enumerate(E):
        ax.text(i, e + 8, f"E={e}\nk={k[i]}", ha="center", fontsize=9, color=MUTED)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left", frameon=False)
    fig.savefig(out / "sparsity_trend.svg", bbox_inches="tight")
    plt.close(fig)
    _ = bars


def arithmetic_intensity_figure(out: Path, analytical: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ai = analytical["arithmetic_intensity"]
    nv = analytical["hardware"]["nvlink_h100"]["balance_flop_per_b"]
    ib = analytical["hardware"]["ib_400g_ndr"]["balance_flop_per_b"]

    # Log-scale bar-ish annotation
    ax.axhline(ai, color=INK, lw=2.2, label=f"MoE layer AI ≈ {ai:.0f} FLOP/B")
    ax.axhspan(0, nv, color=GREEN, alpha=0.12)
    ax.axhspan(nv, ib, color=ORANGE, alpha=0.10)
    ax.axhspan(ib, ib * 1.15, color=RED, alpha=0.08)

    ax.scatter([0.3], [nv], color=GREEN, s=60, zorder=3)
    ax.scatter([0.7], [ib], color=RED, s=60, zorder=3)
    ax.scatter([0.5], [ai], color=INK, s=70, zorder=4)

    ax.annotate("NVLink balance ≈ 1100", xy=(0.3, nv), xytext=(0.05, nv * 1.8),
                fontsize=10, color=GREEN,
                arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1})
    ax.annotate("IB 400G balance ≈ 25k", xy=(0.7, ib), xytext=(0.45, ib * 0.45),
                fontsize=10, color=RED,
                arrowprops={"arrowstyle": "->", "color": RED, "lw": 1})
    ax.annotate("DeepSeek-V3-ish MoE", xy=(0.5, ai), xytext=(0.55, ai * 2.2),
                fontsize=10, color=INK,
                arrowprops={"arrowstyle": "->", "color": INK, "lw": 1})

    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_ylabel("Arithmetic intensity (FLOP / Byte, log)")
    ax.set_title("Same MoE layer: compute-bound on NVLink, comm-bound on IB")
    ax.text(0.5, 0.08, "green = below NVLink balance · orange = between · red = above IB balance",
            transform=ax.transAxes, ha="center", fontsize=9, color=MUTED)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(out / "moe_arithmetic_intensity.svg", bbox_inches="tight")
    plt.close(fig)


def moe_pipeline_figure(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("One MoE layer is five stages — optimize each, then overlap them", fontsize=13)

    stages = [
        (0.3, "Router", "#DBE7F5", BLUE),
        (2.5, "Permute", "#EDE7F6", PURPLE),
        (4.7, "Dispatch\nA2A", "#F8D7D9", RED),
        (6.9, "Experts\nGrouped GEMM", "#D9F0DF", GREEN),
        (9.1, "Combine\nA2A", "#F8D7D9", RED),
    ]
    for x, label, fc, ec in stages:
        _box(ax, (x, 1.3), 1.9, 1.5, label, fc=fc, ec=ec, fontsize=10)
    for x0 in [2.2, 4.4, 6.6, 8.8]:
        _arrow(ax, (x0, 2.05), (x0 + 0.3, 2.05), color=INK)

    ax.text(5.8, 0.55, "unpermute sits on the combine path · shared-expert compute can cover dispatch",
            ha="center", fontsize=9, color=MUTED)
    fig.savefig(out / "moe_layer_pipeline.svg", bbox_inches="tight")
    plt.close(fig)


def group_limited_routing_figure(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Group-limited routing caps cross-machine fanout from k to g", fontsize=13)

    # Free routing
    ax.text(2.5, 4.5, "Free top-k (k=8)", ha="center", fontsize=11, color=RED, fontweight="bold")
    for i in range(8):
        ax.add_patch(Rectangle((0.4 + i * 0.5, 3.2), 0.4, 0.7, facecolor="#F8D7D9", edgecolor=RED))
    ax.text(2.5, 2.7, "up to 8 machines", ha="center", fontsize=9, color=RED)

    # Group limited
    ax.text(7.5, 4.5, "Group-limited (g=2, k=8)", ha="center", fontsize=11, color=GREEN, fontweight="bold")
    ax.add_patch(FancyBboxPatch((5.6, 2.9), 1.6, 1.2, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="#D9F0DF", edgecolor=GREEN, lw=1.4))
    ax.add_patch(FancyBboxPatch((7.6, 2.9), 1.6, 1.2, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="#D9F0DF", edgecolor=GREEN, lw=1.4))
    ax.text(6.4, 3.5, "group A\n4 experts", ha="center", fontsize=9)
    ax.text(8.4, 3.5, "group B\n4 experts", ha="center", fontsize=9)
    ax.text(7.5, 2.4, "≤ 2 machines / NVLink domains", ha="center", fontsize=9, color=GREEN)

    ax.text(5.0, 1.2, "Trade a little routing freedom for a lot less all-to-all fanout",
            ha="center", fontsize=10, color=MUTED, style="italic")
    ax.text(5.0, 0.6, "DeepSeek-V3-style: score groups by sum of top-2 experts, pick g groups, then top-k inside",
            ha="center", fontsize=9, color=MUTED)
    fig.savefig(out / "group_limited_routing.svg", bbox_inches="tight")
    plt.close(fig)


def grouped_gemm_figure(out: Path, rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    labels = [r["label"] for r in rows]
    naive = [r["ms_naive"] for r in rows]
    grouped = [r["ms_grouped"] for r in rows]
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, naive, w, color="#F8D7D9", edgecolor=RED, label="naive E× matmul")
    ax.bar(x + w / 2, grouped, w, color="#D9F0DF", edgecolor=GREEN, label="torch._grouped_mm")
    for i, r in enumerate(rows):
        ax.text(i, max(r["ms_naive"], r["ms_grouped"]) * 1.03, f"{r['speedup']:.1f}×",
                ha="center", fontsize=10, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Median kernel time (ms)")
    ax.set_title(f"Grouped GEMM vs per-expert loop ({rows[0].get('device', 'GPU')})")
    ax.legend(frameon=False)
    fig.savefig(out / "grouped_gemm_bench.svg", bbox_inches="tight")
    plt.close(fig)


def permute_figure(out: Path, rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    labels = [f"T={r['T']}\nH={r['H']}" for r in rows]
    multi = [r["ms_multi_step"] for r in rows]
    fused = [r["ms_gather_scale"] for r in rows]
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, multi, w, color="#F8E3D4", edgecolor=ORANGE, label="multi-step intermediates")
    ax.bar(x + w / 2, fused, w, color="#DBE7F5", edgecolor=BLUE, label="gather + scale")
    for i, r in enumerate(rows):
        ax.text(i, max(r["ms_multi_step"], r["ms_gather_scale"]) * 1.03,
                f"{r['intermediate_mb']:.0f}MB", ha="center", fontsize=9, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Median time (ms)")
    ax.set_title("Permute path: intermediate [T·k·H] tensors dominate")
    ax.legend(frameon=False)
    fig.savefig(out / "permute_bench.svg", bbox_inches="tight")
    plt.close(fig)


def load_balance_figure(out: Path, lb: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    for ax, key, title, color in [
        (axes[0], "no_bias", "No bias (skewed router)", RED),
        (axes[1], "aux_loss_free_bias", "Aux-loss-free bias", GREEN),
    ]:
        mean = np.array(lb[key]["mean_tokens"])
        ax.bar(np.arange(len(mean)), mean, color=color, alpha=0.55, width=1.0, edgecolor="none")
        ax.axhline(mean.mean(), color=INK, lw=1.2, ls="--")
        ax.set_title(f"{title}\nCV={lb[key]['cv']:.2f}, max/mean={lb[key]['max_over_mean']:.1f}")
        ax.set_xlabel("Expert id")
        ax.set_xlim(-0.5, len(mean) - 0.5)
    axes[0].set_ylabel("Mean tokens / step (last 50)")
    fig.suptitle("Bias updates selection, not expert weights — load without aux loss", y=1.02)
    fig.savefig(out / "load_balance_bias.svg", bbox_inches="tight")
    plt.close(fig)


def overlap_figure(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Overlap goal: hide A2A under someone else's compute", fontsize=13)

    # Without overlap
    ax.text(0.2, 4.4, "No overlap", fontsize=10, color=RED)
    stages = [("router", BLUE), ("perm", PURPLE), ("A2A", RED), ("GEMM", GREEN), ("A2A", RED)]
    x = 0.2
    for name, c in stages:
        ax.add_patch(Rectangle((x, 3.5), 1.5, 0.7, facecolor=c, alpha=0.35, edgecolor=c))
        ax.text(x + 0.75, 3.85, name, ha="center", va="center", fontsize=8)
        x += 1.55

    # With overlap
    ax.text(0.2, 2.4, "With overlap", fontsize=10, color=GREEN)
    ax.add_patch(Rectangle((0.2, 1.5), 1.5, 0.7, facecolor=BLUE, alpha=0.35, edgecolor=BLUE))
    ax.text(0.95, 1.85, "router", ha="center", fontsize=8)
    ax.add_patch(Rectangle((1.8, 1.5), 1.3, 0.7, facecolor=PURPLE, alpha=0.35, edgecolor=PURPLE))
    ax.text(2.45, 1.85, "perm", ha="center", fontsize=8)
    # overlapped region
    ax.add_patch(Rectangle((3.2, 1.85), 3.0, 0.55, facecolor=RED, alpha=0.35, edgecolor=RED))
    ax.text(4.7, 2.12, "A2A (hidden)", ha="center", fontsize=8)
    ax.add_patch(Rectangle((3.2, 1.2), 3.0, 0.55, facecolor=GREEN, alpha=0.45, edgecolor=GREEN))
    ax.text(4.7, 1.47, "shared / other GEMM", ha="center", fontsize=8)
    ax.add_patch(Rectangle((6.4, 1.5), 2.0, 0.7, facecolor=GREEN, alpha=0.35, edgecolor=GREEN))
    ax.text(7.4, 1.85, "critical GEMM", ha="center", fontsize=8)

    ax.text(6.0, 0.5,
            r"$T \approx T_{router}+T_{perm}+\max(T_{A2A}, T_{overlap\ compute})+T_{critical\ GEMM}$",
            ha="center", fontsize=10, color=MUTED)
    fig.savefig(out / "comm_compute_overlap.svg", bbox_inches="tight")
    plt.close(fig)


def three_walls_figure(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("MoE’s three walls are coupled — fix one, pressure moves", fontsize=13)

    walls = [
        (0.4, "Memory Wall", "all E experts live in RAM\nrecompute · offload · FSDP", "#F8D7D9", RED),
        (4.2, "Communication Wall", "A2A scales with k·H\nNVLink vs IB · overlap", "#F8E3D4", ORANGE),
        (8.0, "Compute Wall", "many small GEMMs\ngrouped_mm · fusion · graphs", "#DBE7F5", BLUE),
    ]
    for x, title, body, fc, ec in walls:
        _box(ax, (x, 2.2), 3.4, 2.6, "", fc=fc, ec=ec, lw=1.6)
        ax.text(x + 1.7, 4.3, title, ha="center", fontsize=12, fontweight="bold", color=INK)
        ax.text(x + 1.7, 3.2, body, ha="center", fontsize=9, color=MUTED)

    # coupling arrows
    _arrow(ax, (3.8, 3.5), (4.2, 3.5), color=INK, lw=1.4)
    _arrow(ax, (7.6, 3.5), (8.0, 3.5), color=INK, lw=1.4)
    ax.text(6.0, 1.4, "↑ batch  → better GEMM, worse memory + bytes", ha="center", fontsize=9, color=MUTED)
    ax.text(6.0, 0.85, "CUDA Graphs → less host overhead, fights dropless dynamic shapes",
            ha="center", fontsize=9, color=MUTED)
    ax.text(6.0, 0.3, "After Megatron-Core MoE (arXiv:2603.07685)", ha="center", fontsize=8, color=MUTED)
    fig.savefig(out / "three_walls.svg", bbox_inches="tight")
    plt.close(fig)


def solutions_map_figure(out: Path) -> None:
    """Mind-map of concrete MegaScale / Megatron recipes per wall."""
    fig, ax = plt.subplots(figsize=(11.2, 7.2))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_title("Solution map: three walls → two production recipes", fontsize=14, pad=8)

    # Root
    _box(ax, (7.6, 12.2), 6.8, 1.3, "Break MoE’s Three Walls", fc="#EEF2FF", ec=PURPLE, lw=1.8, fontsize=12)
    ax.text(11.0, 11.55, "MegaScale-MoE  ·  Megatron-Core MoE", ha="center", fontsize=8, color=MUTED)

    walls = [
        {
            "x": 0.3,
            "title": "Memory",
            "fc": "#F8D7D9",
            "ec": RED,
            "ms": "SAR under independent ops\nMixtral act −45% / −57%\nMFU Δ <0.5%",
            "nv": "Mem-eff. permute ~26 GB\nfine-grain recompute 42 GB\noffload / FSDP+EP",
        },
        {
            "x": 7.6,
            "title": "Communication",
            "fc": "#F8E3D4",
            "ec": ORANGE,
            "ms": "SP+EP (+13%)\ninter-op (+9%) · intra (+6%)\nAG+RS when k large\nDP BF16 compress",
            "nv": "Parallel Folding\nDeepEP / HybridEP\nFWD↔BWD + W/D split\nEP time 30–40% → <5%",
        },
        {
            "x": 14.9,
            "title": "Compute",
            "fc": "#DBE7F5",
            "ec": BLUE,
            "ms": "EP not TP on experts\n+15–33% MFU vs TP+TP\ncustom CUDA scatter\nGroupedGEMM",
            "nv": "TEGroupedMLP\npermute / router fusion\nCUDA Graphs + sync-free\nECHO for hot experts",
        },
    ]

    for w in walls:
        x = w["x"]
        # wall header
        _box(ax, (x, 9.4), 6.8, 1.15, w["title"] + " Wall", fc=w["fc"], ec=w["ec"], lw=1.6, fontsize=11)
        _arrow(ax, (11.0, 12.2), (x + 3.4, 10.55), color=MUTED, lw=1.1)

        # MegaScale leaf
        _box(ax, (x, 5.35), 6.8, 3.5, "", fc="#D9F0DF", ec=GREEN, lw=1.3)
        ax.text(x + 3.4, 8.4, "MegaScale", ha="center", fontsize=10, fontweight="bold", color=GREEN)
        ax.text(x + 3.4, 6.7, w["ms"], ha="center", va="center", fontsize=8.5, color=INK)

        # Megatron leaf
        _box(ax, (x, 1.2), 6.8, 3.5, "", fc="#DBE7F5", ec=BLUE, lw=1.3)
        ax.text(x + 3.4, 4.25, "Megatron-Core", ha="center", fontsize=10, fontweight="bold", color=BLUE)
        ax.text(x + 3.4, 2.55, w["nv"], ha="center", va="center", fontsize=8.5, color=INK)

        _arrow(ax, (x + 3.4, 9.4), (x + 3.4, 8.85), color=GREEN, lw=1.0)
        _arrow(ax, (x + 3.4, 5.35), (x + 3.4, 4.7), color=BLUE, lw=1.0)

    ax.text(
        11.0,
        0.35,
        "Order of attack: geometry → memory headroom → dispatcher BW → hide latency → pack compute → compress",
        ha="center",
        fontsize=9,
        color=MUTED,
    )
    fig.savefig(out / "solutions_map.svg", bbox_inches="tight")
    plt.close(fig)


def parallelism_philosophies_figure(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4))
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")

    ax = axes[0]
    ax.set_title("MegaScale-MoE: keep MoE on NVLink", fontsize=11, color=GREEN)
    ax.add_patch(FancyBboxPatch((0.5, 1.2), 9.0, 3.8, boxstyle="round,pad=0.02,rounding_size=0.1",
                                 facecolor="#D9F0DF", edgecolor=GREEN, lw=1.5))
    ax.text(5.0, 4.4, "One node = one MoE layer’s EP domain", ha="center", fontsize=10, fontweight="bold")
    ax.text(5.0, 3.5, "Attention: sequence parallel (not TP)\nFFN: expert parallel (not TP)",
            ha="center", fontsize=9)
    ax.text(5.0, 2.3, "PP across nodes · SP/EP inside\nA2A→AG+RS when k large\n+ tile-fused intra-op overlap",
            ha="center", fontsize=9, color=MUTED)
    ax.text(5.0, 0.55, "352B · 1440 H100 · 1.88× vs Megatron-LM", ha="center", fontsize=8, color=GREEN)

    ax = axes[1]
    ax.set_title("Megatron-Core: fold across topologies", fontsize=11, color=BLUE)
    ax.add_patch(FancyBboxPatch((0.5, 1.2), 9.0, 3.8, boxstyle="round,pad=0.02,rounding_size=0.1",
                                 facecolor="#DBE7F5", edgecolor=BLUE, lw=1.5))
    ax.text(5.0, 4.4, "Parallel Folding decouples Attn ↔ MoE", ha="center", fontsize=10, fontweight="bold")
    ax.text(5.0, 3.5, "Attention: high TP/CP\nMoE: high EP (ETP=1), independent",
            ha="center", fontsize=9)
    ax.text(5.0, 2.3, "DeepEP / HybridEP for cross-node\nGrouped GEMM · permute fusion\nFP8 / NVFP4 selective precision",
            ha="center", fontsize=9, color=MUTED)
    ax.text(5.0, 0.55, "DeepSeek-V3: 1233/1048 TFLOPS on GB300/GB200", ha="center", fontsize=8, color=BLUE)

    fig.suptitle("Two production answers to the same communication wall", y=1.02, fontsize=13)
    fig.savefig(out / "parallelism_philosophies.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("playground/moe_perf_results.json"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("content/posts/large-moe-from-sparsity-to-communication"),
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sparsity_trend_figure(args.out)
    moe_pipeline_figure(args.out)
    group_limited_routing_figure(args.out)
    overlap_figure(args.out)
    three_walls_figure(args.out)
    solutions_map_figure(args.out)
    parallelism_philosophies_figure(args.out)

    # Analytical figure works even without Modal results.
    if args.results.exists():
        data = json.loads(args.results.read_text())
        analytical = data["analytical"]
        for row in data["grouped_gemm"]:
            row["device"] = data["device"]
        arithmetic_intensity_figure(args.out, analytical)
        grouped_gemm_figure(args.out, data["grouped_gemm"])
        permute_figure(args.out, data["permute"])
        load_balance_figure(args.out, data["load_balance"])
    else:
        # Conceptual AI figure without Modal results.
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "moe_perf_modal", Path("playground/moe_perf_modal.py")
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        arithmetic_intensity_figure(args.out, mod.analytical_ai())
        print(f"No {args.results}; skipped measured bench figures")

    print(f"Wrote figures to {args.out}")


if __name__ == "__main__":
    main()
