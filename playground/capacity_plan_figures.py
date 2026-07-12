"""Figures for the large-model capacity planning post.

Usage::

    uv run python playground/capacity_plan.py
    uv run python playground/capacity_plan_figures.py \\
        --results playground/capacity_plan_results.json \\
        --out content/posts/large-model-capacity-plan
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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


def _box(ax, xy, w, h, text, *, fc="#f8fafc", ec=INK, lw=1.4, fontsize=9):
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


def _arrow(ax, a, b, *, color=MUTED, lw=1.2):
    ax.add_patch(
        FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=11, linewidth=lw, color=color, shrinkA=1, shrinkB=1
        )
    )


def workflow_figure(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Capacity planning is a pipeline, not a vibe", fontsize=13)
    steps = [
        (0.2, "Intent\nbase vs SFT", "#F1F5F9", MUTED),
        (2.7, "Tokens\n~20x params", "#DBE7F5", BLUE),
        (5.2, "Seq len L\nlocks B, FLOPs", "#F8E3D4", ORANGE),
        (7.7, "FLOPs\n6PT", "#EEF2FF", PURPLE),
        (10.2, "GPU-hours\n/ MFU", "#D9F0DF", GREEN),
        (12.7, "TP/DP/PP\nfrom memory", "#F8D7D9", RED),
    ]
    for x, text, fc, ec in steps:
        _box(ax, (x, 1.1), 2.2, 1.8, text, fc=fc, ec=ec, lw=1.4)
    for x0 in (2.4, 4.9, 7.4, 9.9, 12.4):
        _arrow(ax, (x0, 2.0), (x0 + 0.3, 2.0))
    ax.text(8.0, 0.4, "If any stage is off by 10x, stop — do not 'just launch and see'",
            ha="center", fontsize=9, color=MUTED)
    fig.savefig(out / "capacity_workflow.svg", bbox_inches="tight")
    plt.close(fig)


def gpu_hours_figure(out: Path, cases: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    names = [c["result"]["name"] for c in cases]
    hours = [c["result"]["gpu_hours"] / 1000 for c in cases]
    colors = [BLUE, ORANGE, GREEN][: len(names)]
    bars = ax.bar(names, hours, color=colors, alpha=0.75, edgecolor=INK, width=0.55)
    for b, c in zip(bars, cases, strict=True):
        r = c["result"]
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() * 1.03,
            f"{r['gpu_hours']/1e3:.0f}k h\n~{r['gpus_rounded']} GPUs",
            ha="center",
            fontsize=9,
            color=INK,
        )
    ax.set_ylabel("GPU-hours (thousands)")
    ax.set_title("Same Chinchilla heuristic, different hardware / schedule")
    ax.set_ylim(0, max(hours) * 1.35)
    fig.savefig(out / "gpu_hours_cases.svg", bbox_inches="tight")
    plt.close(fig)


def memory_force_figure(out: Path, case: dict) -> None:
    r = case["result"]
    inp = case["inputs"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    labels = ["weights\n(bf16)", "grads\n(bf16)", "optimizer\n(~fp32 Adam)", "static total"]
    vals = [r["weight_gb"], r["weight_gb"], r["optimizer_gb"], r["static_gb"]]
    colors = [BLUE, PURPLE, ORANGE, RED]
    ax.bar(labels, vals, color=colors, alpha=0.75, edgecolor=INK)
    ax.axhline(inp["gpu_mem_gb"], color=MUTED, ls="--", lw=1.2)
    ax.text(3.35, inp["gpu_mem_gb"] + 5, f"HBM {inp['gpu_mem_gb']:.0f} GB", fontsize=9, color=MUTED)
    ax.set_ylabel("GB")
    ax.set_title(f"{r['name']}: static state forces sharding (TP>={r['suggested_tp']})")
    for i, v in enumerate(vals):
        ax.text(i, v + 4, f"{v:.0f}", ha="center", fontsize=9, color=INK)
    fig.savefig(out / "memory_forces_parallelism.svg", bbox_inches="tight")
    plt.close(fig)


def parallelism_tree_figure(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Parallelism is forced by fit, then by fabric", fontsize=13)

    _box(ax, (4.5, 6.5), 5.0, 1.1, "Does static state fit one GPU\nwith activation headroom?", fc="#EEF2FF", ec=PURPLE, lw=1.6)
    _arrow(ax, (5.5, 6.5), (2.5, 5.3))
    _arrow(ax, (8.5, 6.5), (11.5, 5.3))
    _box(ax, (0.5, 4.2), 4.0, 1.2, "Yes → DP (+ ZeRO)\nscale throughput", fc="#D9F0DF", ec=GREEN, lw=1.4)
    _box(ax, (9.5, 4.2), 4.0, 1.2, "No → raise TP\n(NVLink domain first)", fc="#F8D7D9", ec=RED, lw=1.4)

    _arrow(ax, (11.5, 4.2), (11.5, 3.2))
    _box(ax, (9.5, 2.0), 4.0, 1.2, "Still OOM on acts?\nSP / CP next", fc="#F8E3D4", ec=ORANGE, lw=1.4)
    _arrow(ax, (11.5, 2.0), (7.5, 1.2))
    _box(ax, (4.5, 0.3), 5.0, 1.0, "Only then PP — bubbles are a tax, not a flex", fc="#F1F5F9", ec=MUTED, lw=1.4)

    _arrow(ax, (2.5, 4.2), (2.5, 1.5))
    _box(ax, (0.5, 0.3), 4.0, 1.0, "Long context?\nadd SP/CP even if fit", fc="#DBE7F5", ec=BLUE, lw=1.4)
    fig.savefig(out / "parallelism_decision.svg", bbox_inches="tight")
    plt.close(fig)


def tokens_steps_figure(out: Path, case: dict) -> None:
    """Show how seq length trades steps vs tokens/step for fixed token budget."""
    r = case["result"]
    inp = case["inputs"]
    tokens = inp["tokens_per_param"] * inp["params"]
    gbs = inp["global_batch_seqs"]
    seqs = np.array([2048, 4096, 8192, 16384])
    steps = tokens / (gbs * seqs)
    tok_per_step = gbs * seqs / 1e6

    fig, ax1 = plt.subplots(figsize=(8.5, 4.2))
    ax2 = ax1.twinx()
    ax1.plot(seqs, steps / 1000, "o-", color=BLUE, lw=2, label="steps (k)")
    ax2.plot(seqs, tok_per_step, "s--", color=ORANGE, lw=2, label="M tokens / step")
    ax1.set_xlabel("Sequence length L")
    ax1.set_ylabel("Training steps (thousands)", color=BLUE)
    ax2.set_ylabel("Tokens per step (millions)", color=ORANGE)
    ax1.set_title(
        f"{r['name']}: fixed {r['tokens_b']:.0f}B tokens, GBS={gbs} seqs — L moves everything"
    )
    ax1.set_xticks(seqs)
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, frameon=False, loc="center right")
    fig.savefig(out / "seqlen_trades_steps.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("playground/capacity_plan_results.json"))
    parser.add_argument("--out", type=Path, default=Path("content/posts/large-model-capacity-plan"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.results.exists():
        raise SystemExit(f"Missing {args.results}; run playground/capacity_plan.py first")
    data = json.loads(args.results.read_text())
    cases = data["cases"]
    case_30 = next(c for c in cases if c["result"]["name"] == "30B-A100")

    workflow_figure(args.out)
    gpu_hours_figure(args.out, cases)
    memory_force_figure(args.out, case_30)
    parallelism_tree_figure(args.out)
    tokens_steps_figure(args.out, case_30)
    # also drop results into page bundle for the post
    (args.out / "capacity_plan_results.json").write_text(json.dumps(data, indent=2) + "\n")
    print(f"Wrote figures to {args.out}")


if __name__ == "__main__":
    main()
