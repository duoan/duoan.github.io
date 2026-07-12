"""Generate figures for the torch.compile internals post.

Usage (from repo root)::

    uv run python playground/torch_compile_figures.py \\
        --out content/posts/torch-compile-from-bytecode-to-triton
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

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
LINE = "#94a3b8"


def _box(ax, xy, w, h, text, *, fc="#f8fafc", ec=INK, lw=1.4, fontsize=10, weight="normal"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        mutation_aspect=0.3,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        fontweight=weight,
        wrap=True,
    )
    return patch


def _arrow(ax, start, end, *, color=LINE, style="-|>", lw=1.3):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=12,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def pipeline_figure(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis("off")
    ax.set_title(
        "torch.compile is a specialization pipeline, not a magic flag",
        fontsize=13,
        pad=10,
    )

    layers = [
        (9.2, "Python / nn.Module", "eager call site", "#f1f5f9", MUTED),
        (7.6, "1 · TorchDynamo", "frame hook → FX graph + guards", "#DBE7F5", BLUE),
        (6.0, "2 · AOTAutograd", "functionalize · decompose · joint partition", "#D9F0DF", GREEN),
        (4.4, "3 · TorchInductor", "IR · fusion · memory plan · schedule", "#F8E3D4", ORANGE),
        (2.8, "4 · Codegen", "Triton (GPU)  or  C++/OpenMP (CPU)", "#EDE7F6", PURPLE),
        (1.2, "5 · Runtime", "guard check → cache hit → launch", "#F8D7D9", RED),
    ]

    for y, title, subtitle, fc, ec in layers:
        _box(ax, (1.2, y), 7.6, 1.15, "", fc=fc, ec=ec, lw=1.6)
        ax.text(1.45, y + 0.72, title, fontsize=12, fontweight="bold", color=INK, va="center")
        ax.text(1.45, y + 0.35, subtitle, fontsize=10, color=MUTED, va="center")

    for y0, y1 in [(9.2, 8.75), (7.6, 7.15), (6.0, 5.55), (4.4, 3.95), (2.8, 2.35)]:
        _arrow(ax, (5.0, y0), (5.0, y1), color=INK, lw=1.5)

    # Side note: cache
    _box(
        ax,
        (0.15, 3.5),
        0.9,
        3.2,
        "FxGraph\nCache",
        fc="#fffbeb",
        ec=ORANGE,
        fontsize=9,
    )
    ax.annotate(
        "",
        xy=(1.2, 4.9),
        xytext=(1.05, 4.9),
        arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 1.2},
    )

    ax.text(
        5.0,
        0.35,
        "Compile once under recorded assumptions. Reuse while guards hold.",
        ha="center",
        fontsize=10,
        color=MUTED,
        style="italic",
    )
    fig.savefig(out / "compile_pipeline.svg", bbox_inches="tight")
    plt.close(fig)


def frame_hook_figure(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Dynamo sits under the interpreter, not beside it", fontsize=13, pad=8)

    # Interpreter region
    ax.add_patch(
        FancyBboxPatch(
            (0.4, 0.5),
            5.2,
            4.6,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor="#f8fafc",
            edgecolor=LINE,
            linewidth=1.4,
            linestyle="--",
        )
    )
    ax.text(3.0, 4.8, "CPython", ha="center", fontsize=11, color=MUTED)

    _box(ax, (0.8, 2.7), 2.0, 1.0, "Function\ncall", fc="#ede9fe", ec=PURPLE)
    # diamond
    diamond = Polygon(
        [(4.0, 4.0), (5.0, 3.2), (4.0, 2.4), (3.0, 3.2)],
        closed=True,
        facecolor="#ede9fe",
        edgecolor=PURPLE,
        linewidth=1.4,
    )
    ax.add_patch(diamond)
    ax.text(4.0, 3.2, "eval_frame\nhook?", ha="center", va="center", fontsize=9, color=INK)

    _box(ax, (3.0, 0.8), 2.2, 0.9, "Eager\nbytecode", fc="#f1f5f9", ec=MUTED)
    _arrow(ax, (2.8, 3.2), (3.0, 3.2), color=INK)
    _arrow(ax, (4.0, 2.4), (4.0, 1.7), color=RED)
    ax.text(4.25, 2.0, "no", fontsize=9, color=RED)

    # Dynamo region
    ax.add_patch(
        FancyBboxPatch(
            (6.4, 0.5),
            5.1,
            4.6,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor="#eff6ff",
            edgecolor=BLUE,
            linewidth=1.4,
            linestyle="--",
        )
    )
    ax.text(8.95, 4.8, "Registered Dynamo path", ha="center", fontsize=11, color=BLUE)

    _box(ax, (7.0, 3.5), 3.8, 0.85, "Symbolic bytecode → FX", fc="#DBE7F5", ec=BLUE)
    _box(ax, (7.0, 2.2), 3.8, 0.85, "Backend compile (Inductor…)", fc="#DBE7F5", ec=BLUE)
    _box(ax, (7.0, 0.9), 3.8, 0.85, "Cached callable + guards", fc="#DBE7F5", ec=BLUE)

    _arrow(ax, (5.0, 3.2), (7.0, 3.9), color=GREEN)
    ax.text(5.7, 3.75, "yes", fontsize=9, color=GREEN)
    _arrow(ax, (8.9, 3.5), (8.9, 3.05), color=INK)
    _arrow(ax, (8.9, 2.2), (8.9, 1.75), color=INK)

    fig.savefig(out / "frame_hook.svg", bbox_inches="tight")
    plt.close(fig)


def guard_runtime_figure(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title("Every call: check assumptions before paying for compile", fontsize=13, pad=8)

    _box(ax, (0.5, 2.5), 2.2, 1.2, "Incoming\nframe", fc="#f1f5f9", ec=MUTED, weight="bold")
    _box(ax, (3.5, 2.5), 2.4, 1.2, "Guard check\n(C fast path)", fc="#DBE7F5", ec=BLUE, weight="bold")
    _box(ax, (7.0, 4.2), 2.6, 1.2, "Cache hit\nrun compiled code", fc="#D9F0DF", ec=GREEN, weight="bold")
    _box(ax, (7.0, 2.5), 2.6, 1.2, "Try next\ncache entry", fc="#fff7ed", ec=ORANGE)
    _box(ax, (7.0, 0.6), 2.6, 1.2, "Miss → recompile\nnew specialization", fc="#F8D7D9", ec=RED, weight="bold")
    _box(ax, (10.2, 2.5), 1.5, 1.2, "LRU\nlist", fc="#f8fafc", ec=LINE, fontsize=9)

    _arrow(ax, (2.7, 3.1), (3.5, 3.1), color=INK)
    _arrow(ax, (5.9, 3.5), (7.0, 4.6), color=GREEN)
    ax.text(6.2, 4.2, "pass", fontsize=9, color=GREEN)
    _arrow(ax, (5.9, 2.9), (7.0, 3.1), color=ORANGE)
    ax.text(6.15, 2.55, "fail", fontsize=9, color=ORANGE)
    _arrow(ax, (8.3, 2.5), (8.3, 1.8), color=RED)
    ax.text(8.55, 2.1, "all fail", fontsize=9, color=RED)
    _arrow(ax, (9.6, 3.1), (10.2, 3.1), color=LINE, style="<|-|>")

    ax.text(
        6.0,
        0.2,
        "dynamic=True widens one guard set — it does not emit many specializations in one compile.",
        ha="center",
        fontsize=10,
        color=MUTED,
        style="italic",
    )
    fig.savefig(out / "guard_runtime.svg", bbox_inches="tight")
    plt.close(fig)


def graph_break_figure(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4)
        ax.axis("off")

    axes[0].set_title("One compiled region", fontsize=12, color=GREEN)
    axes[0].add_patch(
        Rectangle((0.5, 1.3), 9.0, 1.4, facecolor="#D9F0DF", edgecolor=GREEN, lw=1.6)
    )
    axes[0].text(
        5.0,
        2.0,
        "fused Triton kernels · low launch overhead",
        ha="center",
        va="center",
        fontsize=10,
    )
    axes[0].text(5.0, 0.5, "fullgraph-friendly Python", ha="center", fontsize=9, color=MUTED)

    axes[1].set_title("Graph breaks = eager islands", fontsize=12, color=RED)
    segments = [
        (1.6, "#D9F0DF", GREEN, "compile"),
        (1.3, "#f1f5f9", MUTED, "eager"),
        (1.8, "#D9F0DF", GREEN, "compile"),
        (1.3, "#f1f5f9", MUTED, "eager"),
        (2.0, "#D9F0DF", GREEN, "compile"),
    ]
    x = 0.6
    for w, fc, ec, label in segments:
        axes[1].add_patch(Rectangle((x, 1.3), w, 1.4, facecolor=fc, edgecolor=ec, lw=1.4))
        axes[1].text(x + w / 2, 2.0, label, ha="center", va="center", fontsize=9)
        x += w + 0.15
    axes[1].text(
        5.0,
        0.45,
        "Python crossings + lost fusion across islands",
        ha="center",
        fontsize=9,
        color=MUTED,
    )

    fig.suptitle("Graph breaks destroy the compile bargain", fontsize=13, y=1.02)
    fig.savefig(out / "graph_breaks.svg", bbox_inches="tight")
    plt.close(fig)


def fusion_bandwidth_figure(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))

    # Left: unfused
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Eager / unfused", fontsize=12)

    # HBM bar
    ax.add_patch(Rectangle((0.5, 0.4), 9.0, 1.1, facecolor="#fee2e2", edgecolor=RED, lw=1.3))
    ax.text(5.0, 0.95, "HBM  ·  each op round-trips intermediates", ha="center", va="center", fontsize=9)

    for name, y in [("add", 7.8), ("relu", 5.4), ("dropout", 3.0)]:
        _box(ax, (3.2, y), 3.6, 1.1, f"kernel: {name}", fc="#DBE7F5", ec=BLUE, fontsize=10)
    # temps materialize to HBM between kernels
    for y0, y1 in [(7.8, 6.5), (5.4, 4.1), (3.0, 1.5)]:
        _arrow(ax, (5.0, y0), (5.0, y1), color=RED, lw=1.3)
    ax.text(6.9, 5.95, "temp→HBM", fontsize=8, color=RED)
    ax.text(6.9, 3.55, "temp→HBM", fontsize=8, color=RED)
    ax.text(5.0, 9.4, "3 launches · 3 HBM stores of temps", ha="center", fontsize=9, color=RED)

    # Right: fused
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Inductor fused", fontsize=12)

    ax.add_patch(Rectangle((0.5, 0.4), 9.0, 1.1, facecolor="#dcfce7", edgecolor=GREEN, lw=1.3))
    ax.text(5.0, 0.95, "HBM  ·  load inputs once, store result once", ha="center", va="center", fontsize=9)

    _box(
        ax,
        (1.8, 3.2),
        6.4,
        4.8,
        "",
        fc="#D9F0DF",
        ec=GREEN,
        lw=1.6,
    )
    ax.text(5.0, 7.5, "triton_poi_fused_add_relu_dropout", ha="center", fontsize=10, fontweight="bold")
    for i, step in enumerate(["tl.load ×2", "x = x + b", "relu", "dropout", "tl.store"]):
        ax.text(5.0, 6.6 - i * 0.7, step, ha="center", fontsize=10, color=INK)
    ax.text(5.0, 2.4, "registers / SRAM hold temps", ha="center", fontsize=9, color=GREEN)
    _arrow(ax, (5.0, 1.5), (5.0, 3.2), color=GREEN, lw=1.4)
    _arrow(ax, (5.0, 3.2), (5.0, 1.5), color=GREEN, lw=1.4)
    ax.text(5.0, 9.4, "1 launch · temps never hit DRAM", ha="center", fontsize=9, color=GREEN)

    fig.suptitle("Fusion is mostly about traffic, not FLOPs", fontsize=13, y=1.02)
    fig.savefig(out / "fusion_bandwidth.svg", bbox_inches="tight")
    plt.close(fig)


def debug_order_figure(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("When compile disappoints: ask questions in this order", fontsize=13, pad=8)

    steps = [
        (8.2, "1. Graph breaks?", "fullgraph=True · TORCH_LOGS=graph_breaks", BLUE),
        (6.4, "2. Recompile thrash?", "TORCH_LOGS=recompiles · mark_dynamic / bucket", ORANGE),
        (4.6, "3. Weak fusion / launch gaps?", "profiler: nested regions · aten:: · idle gaps", PURPLE),
        (2.8, "4. Wrong mode for the job?", "dynamic · reduce-overhead · max-autotune", GREEN),
    ]
    for y, title, detail, color in steps:
        _box(ax, (1.5, y), 7.0, 1.3, "", fc="#f8fafc", ec=color, lw=1.6)
        ax.text(1.8, y + 0.85, title, fontsize=12, fontweight="bold", color=INK, va="center")
        ax.text(1.8, y + 0.4, detail, fontsize=10, color=MUTED, va="center")

    for y0, y1 in [(8.2, 7.7), (6.4, 5.9), (4.6, 4.1)]:
        _arrow(ax, (5.0, y0), (5.0, y1), color=INK, lw=1.5)

    ax.text(
        5.0,
        1.7,
        "Do not start in Nsight Compute. Start at the compile contract.",
        ha="center",
        fontsize=10,
        color=MUTED,
        style="italic",
    )
    ax.text(
        5.0,
        1.1,
        "Optional: torch._dynamo.explain(fn)(*args) to count graphs / breaks / guards",
        ha="center",
        fontsize=9,
        color=MUTED,
    )
    fig.savefig(out / "debug_order.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("content/posts/torch-compile-from-bytecode-to-triton"),
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pipeline_figure(args.out)
    frame_hook_figure(args.out)
    guard_runtime_figure(args.out)
    graph_break_figure(args.out)
    fusion_bandwidth_figure(args.out)
    debug_order_figure(args.out)
    print(f"Wrote figures to {args.out}")


if __name__ == "__main__":
    main()
