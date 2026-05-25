"""Generate the figures for the SiQ-VL training efficiency blog post.

Reads benchmark sweep JSONs from a SiQ-VL ``docs/traces`` directory and emits
SVG figures + a consolidated CSV into the target post bundle.

Each input JSON is one configuration run with the schema::

    {
      "model": "small" | "large",
      "model_label": "...",
      "stage": 1 | 2,
      "config": {batch_size, use_bucketing, use_packing, use_tilegym,
                 use_liger, no_fused_ce, use_compile, force_fp32, ...},
      "results": {real_tok_per_sec, vram_gb, pad_pct, avg_step_ms, ...}
    }

Usage::

    uv run python playground/vlm_efficiency_figures.py \\
        --traces /path/to/SiQ_VL/docs/traces \\
        --out content/posts/optimizing-vlm-training-on-one-gpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ─── Style ──────────────────────────────────────────────────────────────────
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

# Stable color per technique category. Values picked to be distinguishable
# both in color and in print-friendly grayscale.
CATEGORY_COLORS: dict[str, str] = {
    "vanilla_fp32": "#94a3b8",
    "vanilla_bf16": "#4C72B0",
    "liger_no_ce": "#a08bc4",
    "liger": "#8172B3",
    "compile": "#DD8452",
    "tilegym": "#55A868",
    "packing": "#E69138",
    "pack_tilegym": "#C44E52",
}

CATEGORY_MARKERS: dict[str, str] = {
    "vanilla_fp32": "x",
    "vanilla_bf16": "o",
    "liger_no_ce": "s",
    "liger": "s",
    "compile": "^",
    "tilegym": "D",
    "packing": "P",
    "pack_tilegym": "*",
}

CATEGORY_LABELS: dict[str, str] = {
    "vanilla_fp32": "Vanilla (FP32)",
    "vanilla_bf16": "Vanilla (BF16)",
    "liger_no_ce": "Liger (no FusedCE)",
    "liger": "Liger (+FusedCE)",
    "compile": "torch.compile",
    "tilegym": "TileGym",
    "packing": "Packing",
    "pack_tilegym": "Packing + TileGym",
}

# Order used for legend / iteration. Matches the natural optimization journey.
CATEGORY_ORDER = [
    "vanilla_fp32",
    "vanilla_bf16",
    "liger_no_ce",
    "liger",
    "compile",
    "tilegym",
    "packing",
    "pack_tilegym",
]


def categorize(row: pd.Series) -> str:
    if row["force_fp32"]:
        return "vanilla_fp32"
    if row["use_packing"] and row["use_tilegym"]:
        return "pack_tilegym"
    if row["use_packing"]:
        return "packing"
    if row["use_tilegym"]:
        return "tilegym"
    if row["use_compile"]:
        return "compile"
    if row["use_liger"] and row.get("no_fused_ce", False):
        return "liger_no_ce"
    if row["use_liger"]:
        return "liger"
    return "vanilla_bf16"


def load_runs(traces: Path) -> pd.DataFrame:
    """Load every benchmark JSON across the small + large sweeps."""
    sweep_dirs = [
        traces / "benchmark_v3_20260522_001447",  # small model sweep (s1 + s2)
        traces / "benchmark_v3_large",  # large model sweep (s1 + s2)
    ]
    rows: list[dict] = []
    for d in sweep_dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"sweep dir missing: {d}")
        for f in sorted(d.glob("s*.json")):
            j = json.loads(f.read_text())
            row = {
                "file": f.name,
                "sweep": d.name,
                "model": j["model"],
                "model_label": j["model_label"],
                "stage": j["stage"],
                **j["config"],
                **j["results"],
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    df["category"] = df.apply(categorize, axis=1)
    return df


# ─── Figure 1: cumulative speedup by layer ──────────────────────────────────
def fig_cumulative_layers(df: pd.DataFrame, out: Path) -> None:
    """Bar chart showing how each optimization layer compounds on the previous.

    Uses the small-model Stage-1 sweep, which has the cleanest progression
    from FP32 baseline to the Packing + TileGym peak.
    """
    s = df[(df["model"] == "small") & (df["stage"] == 1)]

    fp32 = s[s["force_fp32"] & (s["batch_size"] == 4)]["real_tok_per_sec"].iloc[0]
    bf16 = s[(s["category"] == "vanilla_bf16") & (s["batch_size"] == 4) & ~s["use_bucketing"]][
        "real_tok_per_sec"
    ].iloc[0]
    bucket = s[(s["category"] == "vanilla_bf16") & s["use_bucketing"]]["real_tok_per_sec"].max()
    kernel = s[s["category"].isin(["compile", "tilegym", "liger"])]["real_tok_per_sec"].max()
    full = s[s["category"] == "pack_tilegym"]["real_tok_per_sec"].max()

    layers = [
        "FP32 baseline\n(B=4)",
        "+ BF16 fix\n(B=4)",
        "+ Batching\n+ Bucketing",
        "+ Kernel\nfusion",
        "+ Sequence\npacking",
    ]
    values = [fp32, bf16, bucket, kernel, full]
    speedups = [v / fp32 for v in values]
    colors = ["#94a3b8", "#4C72B0", "#7B9DCC", "#DD8452", "#C44E52"]

    fig, ax = plt.subplots(figsize=(9, 4.4))
    bars = ax.bar(layers, speedups, color=colors, edgecolor="white", linewidth=1.2)

    for bar, sp, raw in zip(bars, speedups, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.12,
            f"{sp:.2f}×\n{raw / 1000:.1f}K tok/s",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#1f2933",
        )

    ax.set_ylabel("Cumulative speedup vs FP32 baseline")
    ax.set_title(
        "Each optimization layer compounds on the previous one\n"
        "(Stage 1 — Frozen LLM, 0.5B model, single Blackwell GPU)",
        fontsize=12,
        fontweight="bold",
        pad=14,
        loc="left",
    )
    ax.set_ylim(0, max(speedups) * 1.25)
    ax.tick_params(axis="x", labelsize=10)
    fig.tight_layout()
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)


# ─── Figure 2 & 3: Speed–VRAM Pareto ────────────────────────────────────────
def fig_pareto(df: pd.DataFrame, out: Path, model: str, stage: int, title: str) -> None:
    s = df[(df["model"] == model) & (df["stage"] == stage)].copy()

    fig, ax = plt.subplots(figsize=(8.8, 4.8))

    seen: set[str] = set()
    for cat in CATEGORY_ORDER:
        sub = s[s["category"] == cat]
        if sub.empty:
            continue
        size = 160 if cat == "pack_tilegym" else 80
        ax.scatter(
            sub["vram_gb"],
            sub["real_tok_per_sec"],
            color=CATEGORY_COLORS[cat],
            marker=CATEGORY_MARKERS[cat],
            s=size,
            label=CATEGORY_LABELS[cat] if cat not in seen else None,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        seen.add(cat)
        for _, row in sub.iterrows():
            ax.annotate(
                f"B={int(row['batch_size'])}",
                (row["vram_gb"], row["real_tok_per_sec"]),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=8,
                color="#475569",
            )

    # Highlight the winner (highest tok/s in this slice).
    winner = s.loc[s["real_tok_per_sec"].idxmax()]
    ax.annotate(
        f"peak: {winner['real_tok_per_sec'] / 1000:.1f}K tok/s\n"
        f"{winner['vram_gb']:.1f} GB VRAM",
        xy=(winner["vram_gb"], winner["real_tok_per_sec"]),
        xytext=(20, -36),
        textcoords="offset points",
        fontsize=9,
        color="#1f2933",
        arrowprops={
            "arrowstyle": "->",
            "color": "#475569",
            "lw": 0.8,
            "shrinkA": 4,
            "shrinkB": 4,
        },
    )

    ax.set_xlabel("Peak VRAM (GB)")
    ax.set_ylabel("Real tokens / sec  (non-padding)")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12, loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)


# ─── Figure 4: FusedCE paradox ──────────────────────────────────────────────
def fig_fusedce_paradox(df: pd.DataFrame, out: Path) -> None:
    """Grouped bar chart comparing Liger no-CE vs +FusedCE per stage / scale."""
    cases = [
        ("small", 1, "Stage 1\n0.5B model"),
        ("small", 2, "Stage 2\n0.5B model"),
        ("large", 1, "Stage 1\n1.5B model"),
        ("large", 2, "Stage 2\n1.5B model"),
    ]
    no_ce_vals: list[float] = []
    with_ce_vals: list[float] = []
    labels: list[str] = []
    for model, stage, label in cases:
        s = df[(df["model"] == model) & (df["stage"] == stage)]
        no_ce = s[s["category"] == "liger_no_ce"]["real_tok_per_sec"].max()
        with_ce = s[s["category"] == "liger"]["real_tok_per_sec"].max()
        no_ce_vals.append(no_ce)
        with_ce_vals.append(with_ce)
        labels.append(label)

    x = list(range(len(cases)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.5))
    b1 = ax.bar(
        [i - width / 2 for i in x],
        no_ce_vals,
        width,
        color=CATEGORY_COLORS["liger_no_ce"],
        edgecolor="white",
        linewidth=1.0,
        label="Liger (no FusedCE)",
    )
    b2 = ax.bar(
        [i + width / 2 for i in x],
        with_ce_vals,
        width,
        color=CATEGORY_COLORS["liger"],
        edgecolor="white",
        linewidth=1.0,
        label="Liger (+FusedCE)",
    )

    for bar, v in zip(b1, no_ce_vals, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(no_ce_vals + with_ce_vals) * 0.015,
            f"{v / 1000:.1f}K",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1f2933",
        )
    for bar, v, no_ce in zip(b2, with_ce_vals, no_ce_vals, strict=True):
        ratio = v / no_ce if no_ce else 0
        color = "#1f2933" if ratio >= 1.0 else "#C44E52"
        annotation = f"{v / 1000:.1f}K\n({ratio:.2f}×)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(no_ce_vals + with_ce_vals) * 0.015,
            annotation,
            ha="center",
            va="bottom",
            fontsize=9,
            color=color,
            fontweight="bold" if ratio < 1.0 else "normal",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Real tokens / sec  (non-padding)")
    ax.set_title(
        "The FusedCE paradox: same kernel, opposite effect across stages",
        fontsize=12,
        fontweight="bold",
        pad=12,
        loc="left",
    )
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.set_ylim(0, max(no_ce_vals + with_ce_vals) * 1.22)
    fig.tight_layout()
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)


# ─── CSV dump ───────────────────────────────────────────────────────────────
def dump_csv(df: pd.DataFrame, out: Path) -> None:
    cols = [
        "model",
        "stage",
        "category",
        "batch_size",
        "use_bucketing",
        "use_packing",
        "pad_to_multiple_of",
        "real_tok_per_sec",
        "vram_gb",
        "pad_pct",
        "avg_step_ms",
        "file",
    ]
    df[cols].sort_values(["model", "stage", "real_tok_per_sec"]).to_csv(out, index=False)


# ─── Entry point ────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--traces",
        type=Path,
        required=True,
        help="Path to SiQ_VL/docs/traces (must contain benchmark_v3_* sweep dirs)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory (the post bundle, e.g. content/posts/<slug>)",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_runs(args.traces)
    print(f"Loaded {len(df)} runs across {df['sweep'].nunique()} sweeps")

    fig_cumulative_layers(df, args.out / "cumulative_speedup.svg")
    fig_pareto(
        df,
        args.out / "pareto_stage1_small.svg",
        "small",
        1,
        "Stage 1 — Speed vs VRAM Pareto (0.5B model, frozen LLM)",
    )
    fig_pareto(
        df,
        args.out / "pareto_stage2_small.svg",
        "small",
        2,
        "Stage 2 — Speed vs VRAM Pareto (0.5B model, full fine-tune)",
    )
    fig_fusedce_paradox(df, args.out / "fusedce_paradox.svg")
    dump_csv(df, args.out / "benchmarks.csv")

    print(f"Wrote 4 SVGs and benchmarks.csv to {args.out}")


if __name__ == "__main__":
    main()
