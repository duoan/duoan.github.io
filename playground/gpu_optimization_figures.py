"""Generate diagrams for the GPU optimization playbook post.

These are hand-authored schematic figures (no measured data) recreated in
English for the blog. Run from the repo root::

    uv run python playground/gpu_optimization_figures.py \\
        --out content/posts/gpu-optimization-playbook
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
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

FILL = "#f8fafc"
BLUE_FILL = "#e8eef7"
GREEN_FILL = "#e7f1ea"
RED_FILL = "#f7e9e8"
ORANGE_FILL = "#faefe6"
PURPLE_FILL = "#efecf5"


def _box(ax, xy, w, h, text, *, fc=FILL, ec=INK, lw=1.4, fontsize=10, weight="normal", tc=INK):
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.06",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            mutation_aspect=1.0,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=tc,
        fontweight=weight,
    )


def _arrow(ax, start, end, *, color=LINE, style="-|>", lw=1.4):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=13,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def _clean(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def architecture_figure(out: Path) -> None:
    """Host <-> device, and what lives inside the GPU chip."""
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    _clean(ax, (0, 12), (0, 9))

    # GPU outer box
    _box(ax, (0.3, 0.4), 7.6, 8.2, "", fc="#ffffff", ec=BLUE, lw=1.8)
    ax.text(0.55, 8.25, "GPU", fontsize=11, color=BLUE, fontweight="bold")

    _box(ax, (0.7, 7.2), 6.8, 0.9, "Device memory (HBM / GDDR)  —  high capacity, ~1–8 TB/s",
         fc=RED_FILL, ec=RED, fontsize=9.5)

    # Chip box
    _box(ax, (0.7, 0.8), 6.8, 6.0, "", fc="#fbfcfe", ec=INK, lw=1.3)
    ax.text(0.95, 6.45, "Chip", fontsize=10.5, color=INK, fontweight="bold")
    _box(ax, (1.0, 5.5), 6.2, 0.75, "L2 cache (shared across all SMs)",
         fc=PURPLE_FILL, ec=PURPLE, fontsize=9.5)

    # SM array
    for i in range(3):
        x = 1.0 + i * 2.15
        _box(ax, (x, 1.2), 1.9, 3.9, "", fc=ORANGE_FILL, ec=ORANGE, lw=1.3)
        ax.text(x + 0.95, 4.85, f"SM {i}", fontsize=9.5, color=ORANGE, ha="center",
                fontweight="bold")
        _box(ax, (x + 0.12, 3.9), 1.66, 0.7, "CUDA cores\nFP32 / INT32", fc="#ffffff",
             ec=ORANGE, fontsize=7.6)
        _box(ax, (x + 0.12, 3.05), 1.66, 0.7, "Tensor Cores", fc="#ffffff", ec=GREEN,
             fontsize=7.8)
        _box(ax, (x + 0.12, 2.2), 1.66, 0.7, "Register file", fc="#ffffff", ec=BLUE,
             fontsize=7.8)
        _box(ax, (x + 0.12, 1.35), 1.66, 0.7, "L1 / shared mem", fc="#ffffff", ec=PURPLE,
             fontsize=7.6)
    ax.text(3.95, 0.95, "…  tens of SMs per GPU", fontsize=8.5, color=MUTED, ha="center")

    # Host box
    _box(ax, (8.7, 3.2), 3.0, 2.6, "", fc="#ffffff", ec=GREEN, lw=1.8)
    ax.text(8.95, 5.45, "Host", fontsize=11, color=GREEN, fontweight="bold")
    _box(ax, (9.0, 4.5), 2.4, 0.75, "CPU", fc=GREEN_FILL, ec=GREEN, fontsize=9.5)
    _box(ax, (9.0, 3.5), 2.4, 0.75, "Host memory (DRAM)", fc=GREEN_FILL, ec=GREEN,
         fontsize=8.6)

    # PCIe link
    _arrow(ax, (7.9, 4.5), (8.7, 4.5), color=INK, style="<|-|>", lw=1.6)
    ax.text(8.3, 4.75, "PCIe", fontsize=8.5, color=INK, ha="center")
    ax.text(8.3, 4.2, "~tens of GB/s", fontsize=7.5, color=MUTED, ha="center")

    # memory bus arrow
    _arrow(ax, (4.1, 7.2), (4.1, 6.3), color=RED, style="<|-|>", lw=1.6)
    ax.text(4.35, 6.72, "memory bus", fontsize=8, color=RED, ha="left")

    ax.set_title("The GPU is a throughput coprocessor hanging off the host over PCIe",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out / "gpu_architecture.svg")
    plt.close(fig)


def sm_figure(out: Path) -> None:
    """Inside one streaming multiprocessor."""
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    _clean(ax, (0, 10), (0, 10))

    _box(ax, (0.4, 0.4), 9.2, 9.2, "", fc="#fbfcfe", ec=ORANGE, lw=1.8)
    ax.text(0.7, 9.15, "Streaming Multiprocessor (SM)", fontsize=11.5, color=ORANGE,
            fontweight="bold")

    _box(ax, (0.8, 8.1), 8.4, 0.75, "L1 instruction cache", fc=BLUE_FILL, ec=BLUE,
         fontsize=9.5)

    # Four processing partitions
    for i in range(4):
        x = 0.8 + i * 2.12
        _box(ax, (x, 2.6), 1.95, 5.1, "", fc="#ffffff", ec=MUTED, lw=1.1)
        _box(ax, (x + 0.1, 6.9), 1.75, 0.6, "Warp scheduler", fc=ORANGE_FILL, ec=ORANGE,
             fontsize=7.8)
        _box(ax, (x + 0.1, 6.2), 1.75, 0.55, "Dispatch unit", fc=ORANGE_FILL, ec=ORANGE,
             fontsize=7.8)
        _box(ax, (x + 0.1, 5.35), 1.75, 0.7, "Register file\n(16K × 32-bit)", fc=BLUE_FILL,
             ec=BLUE, fontsize=7.2)
        _box(ax, (x + 0.1, 4.55), 1.75, 0.65, "INT32 / FP32", fc="#ffffff", ec=INK,
             fontsize=7.8)
        _box(ax, (x + 0.1, 3.85), 1.75, 0.6, "FP64", fc="#ffffff", ec=INK, fontsize=7.8)
        _box(ax, (x + 0.1, 3.15), 1.75, 0.6, "Tensor Core", fc=GREEN_FILL, ec=GREEN,
             fontsize=7.8)
        _box(ax, (x + 0.1, 2.68), 1.75, 0.38, "LD/ST · SFU", fc="#ffffff", ec=MUTED,
             fontsize=7.2)

    _box(ax, (0.8, 1.55), 8.4, 0.85, "256 KB L1 data cache / shared memory (software-managed)",
         fc=PURPLE_FILL, ec=PURPLE, fontsize=9.5)
    _box(ax, (0.8, 0.7), 8.4, 0.65, "Tensor Memory Accelerator (TMA)  ·  Texture units",
         fc=FILL, ec=MUTED, fontsize=9)

    ax.set_title("One SM = 4 warp schedulers issuing to lanes over a shared register file",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out / "sm_internals.svg")
    plt.close(fig)


def execution_figure(out: Path) -> None:
    """Grid -> block -> warp -> thread, and blocks mapped onto SMs."""
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    _clean(ax, (0, 13), (0, 8))

    # Software side: grid of blocks
    _box(ax, (0.3, 0.5), 5.8, 7.0, "", fc="#ffffff", ec=BLUE, lw=1.6)
    ax.text(0.55, 7.15, "Software: Grid", fontsize=11, color=BLUE, fontweight="bold")
    for r in range(2):
        for c in range(3):
            x = 0.7 + c * 1.75
            y = 4.4 - r * 1.5
            _box(ax, (x, y), 1.55, 1.2, f"Block\n({c},{r})", fc=BLUE_FILL, ec=BLUE,
                 fontsize=8.5)
    # zoom into one block -> threads/warp
    _box(ax, (0.7, 0.9), 5.1, 1.9, "", fc="#fbfcfe", ec=INK, lw=1.1)
    ax.text(0.9, 2.55, "Block = threads grouped into warps of 32", fontsize=8.5, color=INK)
    for w in range(2):
        for t in range(8):
            x = 0.95 + t * 0.6
            y = 1.65 - w * 0.5
            fc = GREEN_FILL if w == 0 else ORANGE_FILL
            ec = GREEN if w == 0 else ORANGE
            _box(ax, (x, y), 0.5, 0.4, "", fc=fc, ec=ec, lw=0.9)
    ax.text(5.15, 1.85, "warp 0", fontsize=7.6, color=GREEN, ha="left")
    ax.text(5.15, 1.35, "warp 1", fontsize=7.6, color=ORANGE, ha="left")

    # Hardware side: SMs
    _box(ax, (7.0, 0.5), 5.7, 7.0, "", fc="#ffffff", ec=ORANGE, lw=1.6)
    ax.text(7.25, 7.15, "Hardware: SMs", fontsize=11, color=ORANGE, fontweight="bold")
    for i in range(3):
        y = 5.2 - i * 2.1
        _box(ax, (7.4, y), 4.9, 1.7, "", fc=ORANGE_FILL, ec=ORANGE, lw=1.2)
        ax.text(7.6, y + 1.4, f"SM {i}", fontsize=8.5, color=ORANGE, fontweight="bold")
        ax.text(7.6, y + 0.55,
                "resident blocks share\nregisters + shared memory\n→ occupancy limit",
                fontsize=7.6, color=MUTED, va="center")

    # scheduler arrow
    _arrow(ax, (6.1, 4.0), (7.0, 4.0), color=INK, lw=1.7)
    ax.text(6.55, 4.3, "block\nscheduler", fontsize=8, color=INK, ha="center")

    ax.set_title("The programmer writes a grid of blocks; hardware schedules blocks onto SMs",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out / "execution_model.svg")
    plt.close(fig)


def memory_hierarchy_figure(out: Path) -> None:
    """Scope / latency / bandwidth of each memory space."""
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    _clean(ax, (0, 12), (0, 8))

    rows = [
        ("Registers", "per-thread", "~1 cycle", "highest", BLUE, BLUE_FILL, 10.4),
        ("Shared memory / L1", "per-block", "~30 cycles", "very high", PURPLE, PURPLE_FILL, 8.8),
        ("L2 cache", "per-GPU", "~200 cycles", "high", GREEN, GREEN_FILL, 7.2),
        ("Global memory (HBM)", "per-GPU", "~400+ cycles", "1–8 TB/s", RED, RED_FILL, 5.6),
        ("Host memory (over PCIe)", "system", "µs-scale", "tens of GB/s", ORANGE,
         ORANGE_FILL, 4.0),
    ]
    # header
    ax.text(3.0, 7.5, "Memory space", fontsize=9.5, color=INK, fontweight="bold", ha="center")
    ax.text(6.4, 7.5, "Scope", fontsize=9.5, color=INK, fontweight="bold", ha="center")
    ax.text(8.4, 7.5, "Latency", fontsize=9.5, color=INK, fontweight="bold", ha="center")
    ax.text(10.4, 7.5, "Bandwidth", fontsize=9.5, color=INK, fontweight="bold", ha="center")

    y = 6.6
    for name, scope, lat, bw, ec, fc, _w in rows:
        _box(ax, (0.5, y), 5.0, 0.85, name, fc=fc, ec=ec, fontsize=9.5, tc=INK)
        ax.text(6.4, y + 0.42, scope, fontsize=9, color=MUTED, ha="center")
        ax.text(8.4, y + 0.42, lat, fontsize=9, color=MUTED, ha="center")
        ax.text(10.4, y + 0.42, bw, fontsize=9, color=MUTED, ha="center")
        y -= 1.15

    ax.annotate("", xy=(0.25, 0.9), xytext=(0.25, 7.2),
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.6})
    ax.text(0.05, 4.0, "farther from the ALU → slower, more shared",
            fontsize=8.5, color=INK, rotation=90, va="center", ha="center")

    ax.set_title("Every optimization is a fight to keep data close to the ALUs",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out / "memory_hierarchy.svg")
    plt.close(fig)


def coalescing_figure(out: Path) -> None:
    """Coalesced vs strided global access for a warp."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4))

    def draw(ax, title, coalesced):
        _clean(ax, (0, 10), (0, 8))
        ax.set_title(title, fontsize=11, color=INK)
        # threads
        for t in range(8):
            _box(ax, (0.4 + t * 1.15, 6.4), 0.95, 0.8, f"t{t}", fc=BLUE_FILL, ec=BLUE,
                 fontsize=8)
        ax.text(5.0, 7.5, "one warp (8 of 32 lanes shown)", fontsize=8.5, color=MUTED,
                ha="center")
        # memory cells
        for m in range(8):
            _box(ax, (0.4 + m * 1.15, 1.2), 0.95, 0.8, f"{m}", fc=FILL, ec=MUTED,
                 fontsize=8)
        ax.text(5.0, 0.6, "global memory addresses", fontsize=8.5, color=MUTED, ha="center")

        if coalesced:
            for t in range(8):
                x = 0.4 + t * 1.15 + 0.47
                _arrow(ax, (x, 6.4), (x, 2.0), color=GREEN, lw=1.3)
            _box(ax, (0.4, 3.4), 9.0, 1.0,
                 "contiguous addresses → 1 memory transaction", fc=GREEN_FILL, ec=GREEN,
                 fontsize=9.5)
        else:
            order = [0, 3, 6, 1, 4, 7, 2, 5]
            for t in range(8):
                x0 = 0.4 + t * 1.15 + 0.47
                x1 = 0.4 + order[t] * 1.15 + 0.47
                _arrow(ax, (x0, 6.4), (x1, 2.0), color=RED, lw=1.1)
            _box(ax, (0.4, 3.4), 9.0, 1.0,
                 "scattered addresses → many transactions, wasted bandwidth",
                 fc=RED_FILL, ec=RED, fontsize=9.5)

    draw(ax1, "Coalesced access (good)", True)
    draw(ax2, "Strided / scattered access (bad)", False)
    fig.suptitle("Adjacent threads should touch adjacent addresses", fontsize=12.5,
                 color=INK)
    fig.tight_layout()
    fig.savefig(out / "coalescing.svg")
    plt.close(fig)


def divergence_figure(out: Path) -> None:
    """Warp divergence serializes the two sides of a branch."""
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    _clean(ax, (0, 12), (0, 8))

    _box(ax, (0.5, 6.4), 3.2, 1.0, "warp: 32 lanes\nin lock-step", fc=BLUE_FILL, ec=BLUE,
         fontsize=9)
    _box(ax, (4.6, 6.5), 2.6, 0.8, "if (cond)", fc=FILL, ec=INK, fontsize=9.5)

    # path A
    _box(ax, (8.0, 6.9), 3.5, 0.85, "then-branch\nlanes with cond=1 active", fc=GREEN_FILL,
         ec=GREEN, fontsize=8.2)
    # path B
    _box(ax, (8.0, 5.2), 3.5, 0.85, "else-branch\nlanes with cond=0 active", fc=RED_FILL,
         ec=RED, fontsize=8.2)
    _arrow(ax, (7.2, 6.9), (8.0, 7.3), color=GREEN, lw=1.4)
    _arrow(ax, (7.2, 6.9), (8.0, 5.6), color=RED, lw=1.4)
    _arrow(ax, (3.7, 6.9), (4.6, 6.9), color=INK, lw=1.4)

    # serialized timeline
    ax.text(0.5, 4.1, "What the SM actually does (one program counter per warp):",
            fontsize=9.5, color=INK)
    _box(ax, (0.5, 2.7), 4.8, 1.0, "run then-branch\n(else lanes masked OFF, idle)",
         fc=GREEN_FILL, ec=GREEN, fontsize=8.5)
    _box(ax, (5.7, 2.7), 4.8, 1.0, "run else-branch\n(then lanes masked OFF, idle)",
         fc=RED_FILL, ec=RED, fontsize=8.5)
    _arrow(ax, (5.3, 3.2), (5.7, 3.2), color=INK, lw=1.4)
    _box(ax, (2.9, 1.2), 5.0, 0.85, "reconverge → both halves paid for",
         fc=FILL, ec=INK, fontsize=9)
    _arrow(ax, (2.9, 2.7), (4.0, 2.05), color=MUTED, lw=1.2)
    _arrow(ax, (8.1, 2.7), (6.8, 2.05), color=MUTED, lw=1.2)

    ax.set_title("Warp divergence: a data-dependent branch serializes both paths",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out / "warp_divergence.svg")
    plt.close(fig)


def taxonomy_figure(out: Path) -> None:
    """The three-pillar optimization map."""
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    _clean(ax, (0, 12), (0, 9))

    _box(ax, (4.2, 7.9), 3.6, 0.9, "GPU kernel optimization", fc=INK, ec=INK, fontsize=11,
         tc="white", weight="bold")

    pillars = [
        ("Memory access", "keep data close,\nmove it in wide bursts", BLUE, BLUE_FILL, 0.4,
         ["Coalesced global access", "Shared-memory tiling", "Warp shuffle / registers",
          "Kernel fusion", "Constant / texture cache", "Software prefetch (double buffer)"]),
        ("Irregularity", "make divergent work\nlook regular to a warp", RED, RED_FILL, 4.3,
         ["Reduce warp divergence", "Loop unrolling", "Predication / LUT",
          "Thread & data remapping", "Sparse formats (CSR/ELL)", "Balance load across SMs"]),
        ("Balance", "match work to the\nresources you actually have", GREEN, GREEN_FILL, 8.2,
         ["Vectorize (float4)", "Fast-math intrinsics", "Thread / block coarsening",
          "Register blocking", "Auto-tuning", "Fewer atomics / syncs"]),
    ]

    for name, sub, ec, fc, x, items in pillars:
        _box(ax, (x, 6.0), 3.5, 1.2, f"{name}\n{sub}", fc=fc, ec=ec, fontsize=9.5,
             weight="bold")
        _arrow(ax, (6.0, 7.9), (x + 1.75, 7.2), color=ec, lw=1.4)
        y = 5.2
        for it in items:
            _box(ax, (x, y), 3.5, 0.62, it, fc="#ffffff", ec=ec, fontsize=8.2)
            y -= 0.75

    ax.set_title("Three questions to ask any hot kernel", fontsize=12.5, color=INK)
    fig.tight_layout()
    fig.savefig(out / "optimization_map.svg")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    architecture_figure(args.out)
    sm_figure(args.out)
    execution_figure(args.out)
    memory_hierarchy_figure(args.out)
    coalescing_figure(args.out)
    divergence_figure(args.out)
    taxonomy_figure(args.out)
    print(f"wrote figures to {args.out}")


if __name__ == "__main__":
    main()
