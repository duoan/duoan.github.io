"""Figures for the SiQ-VL curriculum / compute-constrained VLM post.

Usage (from repo root)::

    uv run python playground/siq_curriculum_figures.py \\
        --out content/posts/siq-vl-curriculum-under-compute-constraints
"""

from __future__ import annotations

import argparse
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


def _box(ax, xy, w, h, text, *, fc="#f8fafc", ec=INK, lw=1.4, fontsize=9.5):
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


def architecture_figure(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 3.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("SiQ-VL: freeze the towers, train the connector first", fontsize=13)

    blocks = [
        (0.3, "Image\n512x512", "#F1F5F9", MUTED),
        (2.5, "SigLIP-2\nFROZEN", "#F8D7D9", RED),
        (4.9, "Patches\n1024 x D", "#F8E3D4", ORANGE),
        (7.3, "Pixel shuffle\nx4 tokens", "#DBE7F5", BLUE),
        (9.7, "Linear\nprojector", "#D9F0DF", GREEN),
        (11.9, "Qwen2.5\n0.5B / 1.5B", "#EEF2FF", PURPLE),
    ]
    for x, text, fc, ec in blocks:
        _box(ax, (x, 1.8), 2.0, 2.0, text, fc=fc, ec=ec, lw=1.5, fontsize=9)
    for x0, x1 in [(2.3, 2.5), (4.5, 4.9), (6.9, 7.3), (9.3, 9.7), (11.7, 11.9)]:
        _arrow(ax, (x0, 2.8), (x1, 2.8), color=MUTED, lw=1.2)

    ax.text(3.5, 1.15, "never unfrozen", ha="center", fontsize=8, color=RED)
    ax.text(8.3, 1.15, "1024 -> 64 tokens", ha="center", fontsize=8, color=BLUE)
    ax.text(12.9, 1.15, "frozen in Stage 1\nLoRA/full in Stage 2+", ha="center", fontsize=8, color=PURPLE)
    ax.text(7.0, 0.35, "Trainable surface starts as projector-only — one failure mode at a time",
            ha="center", fontsize=9, color=MUTED)
    fig.savefig(out / "siq_architecture.svg", bbox_inches="tight")
    plt.close(fig)


def curriculum_figure(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Curriculum: one capability per stage", fontsize=13)

    stages = [
        (0.4, "Stage 1\nProjector alignment", "Vision frozen\nLLM frozen\nProjector only",
         "FineVision-class\nalignment data", "#D9F0DF", GREEN, "done"),
        (4.9, "Stage 2\nInstruction / VQA", "Vision frozen\nLLM LoRA/full\n+ projector",
         "VQAv2 / GQA /\nTextVQA mix", "#DBE7F5", BLUE, "done"),
        (9.4, "Stage 3\nOffline CoT", "Vision frozen\nLLM continues\n+ rationales",
         "Multi-teacher\n(image,Q,R,A)", "#F8E3D4", ORANGE, "in progress"),
    ]
    for x, title, train, data, fc, ec, status in stages:
        _box(ax, (x, 2.6), 4.0, 3.6, "", fc=fc, ec=ec, lw=1.6)
        ax.text(x + 2.0, 5.7, title, ha="center", fontsize=11, fontweight="bold", color=INK)
        ax.text(x + 2.0, 4.55, train, ha="center", fontsize=9, color=INK)
        ax.text(x + 2.0, 3.2, data, ha="center", fontsize=8.5, color=MUTED)
        badge_fc = "#D9F0DF" if status == "done" else "#FEF3C7"
        badge_ec = GREEN if status == "done" else ORANGE
        ax.add_patch(Rectangle((x + 1.15, 2.75), 1.7, 0.35, facecolor=badge_fc, edgecolor=badge_ec, lw=1.0))
        ax.text(x + 2.0, 2.92, status, ha="center", va="center", fontsize=8, color=INK)

    _arrow(ax, (4.4, 4.4), (4.9, 4.4), color=INK, lw=1.4)
    _arrow(ax, (8.9, 4.4), (9.4, 4.4), color=INK, lw=1.4)
    ax.text(7.0, 1.6, "Do not merge stages. Alignment failure and reasoning failure look identical in one loss.",
            ha="center", fontsize=9, color=MUTED)
    ax.text(7.0, 0.9, "Stage 4 (RL) is optional and last — distill first while GPU-poor",
            ha="center", fontsize=9, color=MUTED)
    ax.text(7.0, 0.3, "After SiQ-VL curriculum (github.com/duoan/SiQ_VL)", ha="center", fontsize=8, color=MUTED)
    fig.savefig(out / "siq_curriculum.svg", bbox_inches="tight")
    plt.close(fig)


def token_memory_figure(out: Path) -> None:
    """Analytical memory vs pixel-shuffle factor (from SiQ MEMORY_ANALYSIS)."""
    # Numbers from MEMORY_ANALYSIS.md for siglip2-so400m-patch16-512 + Qwen2.5-1.5B, B=4, L_text=2048
    factors = np.array([1, 2, 4, 8, 16])
    patches = np.array([1024, 256, 64, 16, 4])
    total_gb = np.array([10.82, 8.68, 8.33, 8.65, 10.31])
    attn_gb = np.array([4.00, 2.25, 1.89, 1.81, 1.80])

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))

    ax = axes[0]
    ax.plot(factors, patches, "o-", color=BLUE, lw=2, markersize=7)
    for f, p in zip(factors, patches, strict=True):
        ax.text(f, p * 1.08, str(p), ha="center", fontsize=8, color=MUTED)
    ax.set_xscale("log", base=2)
    ax.set_xticks(factors)
    ax.set_xticklabels([str(f) for f in factors])
    ax.set_xlabel("Pixel-shuffle factor")
    ax.set_ylabel("Vision tokens after shuffle")
    ax.set_title("Token compression (512 px SigLIP-2)")
    ax.set_ylim(0, 1200)

    ax = axes[1]
    ax.bar(np.arange(len(factors)) - 0.18, total_gb, 0.36, color="#DBE7F5", edgecolor=BLUE, label="total est. GB")
    ax.bar(np.arange(len(factors)) + 0.18, attn_gb, 0.36, color="#F8E3D4", edgecolor=ORANGE, label="attention GB")
    ax.axhline(10.82, color=RED, ls="--", lw=1.0, alpha=0.7)
    ax.text(4.05, 10.95, "no shuffle", fontsize=8, color=RED, ha="right")
    ax.set_xticks(np.arange(len(factors)))
    ax.set_xticklabels([str(f) for f in factors])
    ax.set_xlabel("Pixel-shuffle factor")
    ax.set_ylabel("Estimated training memory (GB)")
    ax.set_title("B=4, text 2048, Qwen2.5-1.5B")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, 12.5)

    fig.suptitle("First systems knob: cut vision tokens before they hit the LLM", y=1.02, fontsize=13)
    fig.savefig(out / "pixel_shuffle_memory.svg", bbox_inches="tight")
    plt.close(fig)


def offline_cot_figure(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Offline CoT: pay teacher cost once, train the student many times", fontsize=13)

    teachers = [
        (0.4, "Qwen3-VL\nThinking", "structured\nmath / steps", GREEN),
        (3.6, "InternVL", "charts /\nvisual analytics", BLUE),
        (6.8, "HunyuanOCR", "OCR-heavy\ntext-in-image", ORANGE),
    ]
    for x, name, bias, ec in teachers:
        _box(ax, (x, 3.4), 2.8, 2.0, f"{name}\n{bias}", fc="#f8fafc", ec=ec, lw=1.4, fontsize=9)
        _arrow(ax, (x + 1.4, 3.4), (8.2, 2.4), color=MUTED, lw=1.0)

    _box(ax, (7.0, 1.3), 3.2, 1.5, "Offline traces\n(image, Q, R, A)", fc="#FEF3C7", ec=ORANGE, lw=1.5, fontsize=10)
    _arrow(ax, (10.2, 2.05), (11.0, 2.05), color=INK, lw=1.3)
    _box(ax, (11.0, 1.3), 2.6, 1.5, "Student\nSiQ-VL", fc="#D9F0DF", ec=GREEN, lw=1.5, fontsize=10)

    ax.text(7.0, 0.55, "No teacher on the training GPU. Diversity of teachers > one giant teacher under a memory wall.",
            ha="center", fontsize=9, color=MUTED)
    fig.savefig(out / "offline_cot_teachers.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("content/posts/siq-vl-curriculum-under-compute-constraints"),
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    architecture_figure(args.out)
    curriculum_figure(args.out)
    token_memory_figure(args.out)
    offline_cot_figure(args.out)
    print(f"Wrote figures to {args.out}")


if __name__ == "__main__":
    main()
