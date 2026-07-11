"""Generate figures for the roofline blog post from roofline_modal.py output.

Usage (from repo root)::

    uv run python playground/roofline_figures.py \\
        --results playground/roofline_a10g_results.json \\
        --out content/posts/roofline-first-step-of-performance-optimization
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


def concept_figure(out: Path) -> None:
    """Annotated conceptual roofline (no data, generic axes)."""
    peak_flops = 100.0  # arbitrary units
    peak_bw = 1.0
    i_peak = peak_flops / peak_bw

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    xs = np.logspace(-1, 4, 400)
    roof = np.minimum(xs * peak_bw, peak_flops)
    ax.plot(xs, roof, color="#1f2933", lw=2.5, zorder=3)

    ax.fill_between(xs, roof, 1e-2, where=xs <= i_peak, color="#C44E52", alpha=0.06)
    ax.fill_between(xs, roof, 1e-2, where=xs >= i_peak, color="#4C72B0", alpha=0.06)

    ax.annotate(
        "memory-bound\n(slope = peak bandwidth)",
        xy=(2.2, 0.55),
        fontsize=11,
        color="#C44E52",
        ha="center",
        rotation=0,
    )
    ax.annotate(
        "compute-bound\n(plateau = peak FLOP/s)",
        xy=(1500, 6),
        fontsize=11,
        color="#4C72B0",
        ha="center",
    )
    ax.scatter([i_peak], [peak_flops], color="#1f2933", zorder=4, s=45)
    ax.annotate(
        "ridge point\nI_peak = PeakFLOPS / PeakBW",
        xy=(i_peak, peak_flops),
        xytext=(8, -30),
        textcoords="offset points",
        fontsize=10,
        ha="left",
        va="top",
        color="#1f2933",
    )

    # Example operator: ceiling vs achieved.
    i_op = 8.0
    achieved = 3.0
    ax.plot([i_op, i_op], [1e-2, i_op * peak_bw], color="#94a3b8", ls=":", lw=1.2)
    ax.scatter([i_op], [i_op * peak_bw], color="#55A868", zorder=4, s=45)
    ax.scatter([i_op], [achieved], color="#55A868", zorder=4, s=45, facecolor="white", lw=1.6)
    ax.annotate(
        "theoretical ceiling for this op",
        xy=(i_op, i_op * peak_bw),
        xytext=(10, 4),
        textcoords="offset points",
        fontsize=10,
        color="#55A868",
    )
    ax.annotate(
        "measured\n→ utilization = measured / ceiling",
        xy=(i_op, achieved),
        xytext=(10, -12),
        textcoords="offset points",
        fontsize=10,
        color="#55A868",
    )
    ax.annotate(
        "",
        xy=(i_op, i_op * peak_bw * 0.92),
        xytext=(i_op, achieved * 1.25),
        arrowprops={"arrowstyle": "<->", "color": "#55A868", "lw": 1.2},
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(0.08, 400)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Arithmetic intensity I (FLOP / Byte, log scale)")
    ax.set_ylabel("Achievable performance (FLOP/s, log scale)")
    ax.set_title("The Roofline model: two ceilings, one ridge point")
    fig.tight_layout()
    fig.savefig(out / "roofline_concept.svg")
    plt.close(fig)


# Datasheet values: dense BF16 TFLOPS, bandwidth GB/s.
GPUS = [
    ("V100", 2017, 125, 900),
    ("A100", 2020, 312, 2039),
    ("H100", 2022, 989, 3350),
    ("H200", 2024, 989, 4800),
    ("B200", 2025, 2250, 8000),
]


def evolution_figure(out: Path) -> None:
    """Compute vs bandwidth growth and the resulting ridge-point drift."""
    names = [g[0] for g in GPUS]
    flops = np.array([g[2] for g in GPUS], dtype=float)
    bw = np.array([g[3] for g in GPUS], dtype=float)
    ridge = flops * 1e3 / bw

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4))

    xs = np.arange(len(names))
    ax1.plot(xs, flops / flops[0], marker="o", lw=2, color="#4C72B0", label="Peak BF16 FLOPS")
    ax1.plot(xs, bw / bw[0], marker="s", lw=2, color="#C44E52", label="Memory bandwidth")
    ax1.set_xticks(xs, names)
    ax1.set_yscale("log")
    ax1.set_yticks([1, 2, 4, 8, 18], ["1×", "2×", "4×", "8×", "18×"])
    ax1.set_ylabel("Growth vs V100")
    ax1.set_title("Compute grows faster than bandwidth")
    ax1.legend(fontsize=9, frameon=False, loc="upper left")
    for x, f, b in zip(xs, flops / flops[0], bw / bw[0], strict=True):
        ax1.annotate(
            f"{f:.1f}×",
            (x, f),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color="#4C72B0",
        )
        ax1.annotate(
            f"{b:.1f}×",
            (x, b),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color="#C44E52",
        )

    bars = ax2.bar(xs, ridge, color="#8172B2", width=0.55)
    ax2.set_xticks(xs, names)
    ax2.set_ylabel("Ridge point I_peak (FLOP/B, BF16)")
    ax2.set_title("The ridge point keeps moving right")
    ax2.bar_label(bars, fmt="%.0f", fontsize=9, color="#1f2933")
    ax2.annotate(
        "more ops become\nmemory-bound →",
        xy=(0.04, 0.86),
        xycoords="axes fraction",
        fontsize=10,
        color="#475569",
    )

    fig.tight_layout()
    fig.savefig(out / "gpu_evolution.svg")
    plt.close(fig)


def a100_vs_h100_figure(out: Path) -> None:
    """The upgrade incident: the same op lands on different sides of the ridge."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    xs = np.logspace(0, 4, 400)

    specs = {"A100": (312.0, 2039.0, "#4C72B0"), "H100": (989.0, 3350.0, "#C44E52")}
    for name, (pf, pbw, color) in specs.items():
        roof = np.minimum(xs * pbw / 1e3, pf)
        i_peak = pf * 1e3 / pbw
        ax.plot(xs, roof, color=color, lw=2, label=f"{name} (ridge ≈ {i_peak:.0f})")
        ax.axvline(i_peak, color=color, ls=":", lw=1)

    # An operator with I = 200 sits right of the A100 ridge, left of the H100 ridge.
    i_op = 200.0
    ax.axvline(i_op, color="#1f2933", ls="--", lw=1.2)
    ax.scatter([i_op], [312.0], color="#4C72B0", zorder=4, s=50)
    ax.scatter([i_op], [200 * 3.35], color="#C44E52", zorder=4, s=50)
    ax.annotate(
        "on A100: compute-bound,\nsits on the plateau",
        xy=(i_op, 312.0),
        xytext=(26, -26),
        textcoords="offset points",
        ha="left",
        fontsize=10,
        color="#4C72B0",
    )
    ax.annotate(
        "same op on H100: memory-bound,\nceiling = I × BW, not peak FLOPS",
        xy=(i_op, 200 * 3.35),
        xytext=(26, -110),
        textcoords="offset points",
        fontsize=10,
        color="#C44E52",
    )
    ax.annotate(
        "an operator with I = 200",
        xy=(i_op, 2.2),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=10,
        color="#1f2933",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1e4)
    ax.set_ylim(2, 4000)
    ax.set_xlabel("Arithmetic intensity I (FLOP / Byte)")
    ax.set_ylabel("Achievable performance (TFLOP/s, BF16)")
    ax.set_title("Why the H100 upgrade disappointed: the ridge moved")
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "a100_vs_h100.svg")
    plt.close(fig)


def roofline_figure(data: dict, out: Path) -> None:
    peak_flops = data["peak_bf16_tflops"]  # TFLOP/s
    peak_bw = data["peak_mem_tbs"] * 1000  # GB/s
    i_peak = peak_flops * 1e3 / peak_bw  # FLOP/B at the ridge

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    xs = np.logspace(-1.2, 3.8, 400)
    roof = np.minimum(xs * peak_bw / 1e3, peak_flops)
    ax.plot(xs, roof, color="#1f2933", lw=2, zorder=3, label="Datasheet roof")

    meas_bw = max(o["gbs"] for o in data["ops"] if o["kind"] == "ceiling_bw")
    meas_fl = max(o["tflops"] for o in data["ops"] if o["kind"] in ("ceiling_flops", "gemm_sweep"))
    roof_m = np.minimum(xs * meas_bw / 1e3, meas_fl)
    ax.plot(xs, roof_m, color="#64748b", lw=1.5, ls="--", zorder=3, label="Measured roof")

    ax.axvline(i_peak, color="#94a3b8", ls=":", lw=1)
    ax.annotate(
        f"ridge point\nI = {i_peak:.0f} FLOP/B",
        xy=(i_peak, 0.06),
        xycoords=("data", "axes fraction"),
        ha="left",
        fontsize=9,
        color="#475569",
        xytext=(5, 0),
        textcoords="offset points",
    )

    label_offsets = {
        "Elementwise add": (0, -14),
        "GELU": (6, 2),
        "RMSNorm 8192x8192": (6, -12),
        "Softmax 8192x8192": (-4, 6),
        "GEMM M=1 K=N=8192 (decode)": (6, 8),
        "Attention vanilla S=4096": (6, -4),
        "FlashAttention S=4096": (-6, -18),
        "GEMM 8192^3": (-8, 8),
    }
    for op in data["ops"]:
        if op["kind"] != "op" and op["name"] != "GEMM 8192^3":
            continue
        color = "#C44E52" if op["intensity"] < i_peak else "#4C72B0"
        ax.scatter(op["intensity"], op["tflops"], s=42, color=color, zorder=4)
        dx, dy = label_offsets.get(op["name"], (6, 4))
        ax.annotate(
            op["name"],
            (op["intensity"], op["tflops"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5,
            ha="left" if dx >= 0 else "right",
            color="#1f2933",
        )

    ax.scatter([], [], s=42, color="#C44E52", label="memory-bound side")
    ax.scatter([], [], s=42, color="#4C72B0", label="compute-bound side")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10**-1.2, 10**3.8)
    ax.set_ylim(0.03, peak_flops * 4)
    ax.set_xlabel("Arithmetic intensity I (FLOP / Byte)")
    ax.set_ylabel("Achieved performance (TFLOP/s, BF16)")
    ax.set_title(f"Roofline — {data['device']} (measured on Modal)")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "roofline_measured.svg")
    plt.close(fig)


def gemm_sweep_figure(data: dict, out: Path) -> None:
    peak_flops = data["peak_bf16_tflops"]
    sweep = [o for o in data["ops"] if o["kind"] in ("gemm_sweep",)]
    sweep.sort(key=lambda o: o["intensity"])
    sizes = [o["name"].split()[1].split("^")[0] for o in sweep]
    tflops = [o["tflops"] for o in sweep]

    decode = next(o for o in data["ops"] if "decode" in o["name"])

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(range(len(sizes)), tflops, marker="o", color="#4C72B0", lw=2, label="Square GEMM N×N×N")
    ax.axhline(peak_flops, color="#1f2933", ls="--", lw=1.2)
    ax.annotate(
        f"datasheet peak {peak_flops:.0f} TFLOP/s",
        xy=(0.02, peak_flops),
        xycoords=("axes fraction", "data"),
        va="bottom",
        fontsize=9,
        color="#475569",
    )
    ax.scatter([0], [decode["tflops"]], color="#C44E52", zorder=4, s=48)
    ax.annotate(
        f"decode GEMV M=1, K=N=8192\n{decode['tflops']:.2f} TFLOP/s (I ≈ 1)",
        (0, decode["tflops"]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        color="#C44E52",
    )

    for i, o in enumerate(sweep):
        ax.annotate(
            f"I={o['intensity']:.0f}",
            (i, o["tflops"]),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#475569",
        )

    ax.set_xticks(range(len(sizes)), [f"{s}³" for s in sizes])
    ax.set_xlabel("GEMM size (N×N×N, BF16)")
    ax.set_ylabel("Achieved TFLOP/s")
    ax.set_yscale("log")
    ax.set_title(f"Same op, different shape — GEMM on {data['device']}")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "gemm_shape_sweep.svg")
    plt.close(fig)


def decode_sweep_figure(data: dict, out: Path) -> None:
    """Model-level roofline in batch space: decode tokens/s vs batch size."""
    peak_flops = data["peak_bf16_tflops"] * 1e12
    peak_bw = data["peak_mem_tbs"] * 1e12
    weight_bytes = 2 * data["n_params"]
    flops_per_token = 2 * data["n_params"]

    mem_ceiling = peak_bw / weight_bytes  # tok/s per unit batch
    comp_ceiling = peak_flops / flops_per_token  # tok/s plateau
    ridge_batch = comp_ceiling / mem_ceiling  # = peak_flops / peak_bw

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bs = np.logspace(0, np.log10(1024), 300)
    roof = np.minimum(bs * mem_ceiling, comp_ceiling)
    ax.plot(bs, roof, color="#1f2933", lw=2, label="Roofline (datasheet)", zorder=3)

    batches = [p["batch"] for p in data["points"]]
    toks = [p["tokens_per_s"] for p in data["points"]]
    ax.plot(
        batches, toks, marker="o", lw=1.8, color="#55A868", label="Measured decode step", zorder=4
    )

    ax.axvline(ridge_batch, color="#94a3b8", ls=":", lw=1.2)
    ax.annotate(
        f"critical batch ≈ {ridge_batch:.0f}\n(= I_peak of this GPU)",
        xy=(ridge_batch, 0.04),
        xycoords=("data", "axes fraction"),
        xytext=(-8, 0),
        textcoords="offset points",
        ha="right",
        fontsize=9.5,
        color="#475569",
    )
    ax.annotate(
        "memory-bound:\nbatching is (nearly) free",
        xy=(0.16, 0.62),
        xycoords="axes fraction",
        fontsize=10.5,
        color="#C44E52",
        ha="center",
    )
    ax.annotate(
        "compute-bound:\nlatency now scales with batch",
        xy=(0.84, 0.5),
        xycoords="axes fraction",
        fontsize=10.5,
        color="#4C72B0",
        ha="center",
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(batches, [str(b) for b in batches])
    ax.set_xlabel("Decode batch size")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title(
        f"Model-level roofline — {data['n_params'] / 1e9:.1f}B-param MLP decode on {data['device']}"
    )
    ax.legend(loc="upper left", fontsize=9.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "decode_batch_sweep.svg")
    plt.close(fig)


def train_sweep_figure(data: dict, out: Path) -> None:
    """Model-level roofline in token space: fwd+bwd throughput vs tokens/step."""
    peak_flops = data["peak_bf16_tflops"] * 1e12
    peak_bw = data["peak_mem_tbs"] * 1e12
    # fwd read + bwd read + grad write ≈ 3 weight-sized traversals
    bytes_per_step = 6 * data["n_params"]
    flops_per_token = 6 * data["n_params"]

    mem_slope = peak_bw / bytes_per_step  # tok/s gained per extra token (slope)
    comp_ceiling = peak_flops / flops_per_token
    ridge_tokens = comp_ceiling / mem_slope  # = peak_flops / peak_bw

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ts = np.logspace(0, np.log10(8192), 300)
    roof = np.minimum(ts * mem_slope, comp_ceiling)
    ax.plot(ts, roof, color="#1f2933", lw=2, label="Roofline (datasheet)", zorder=3)

    tokens = [p["tokens"] for p in data["points"]]
    toks = [p["tokens_per_s"] for p in data["points"]]
    ax.plot(
        tokens,
        toks,
        marker="o",
        lw=1.8,
        color="#4C72B0",
        label="Measured fwd+bwd step",
        zorder=4,
    )

    ax.axvline(ridge_tokens, color="#94a3b8", ls=":", lw=1.2)
    ax.annotate(
        f"critical tokens/step ≈ {ridge_tokens:.0f}\n(= I_peak of this GPU)",
        xy=(ridge_tokens, 0.04),
        xycoords=("data", "axes fraction"),
        xytext=(-8, 0),
        textcoords="offset points",
        ha="right",
        fontsize=9.5,
        color="#475569",
    )
    ax.annotate(
        "memory-bound:\nmore tokens/step is (nearly) free",
        xy=(0.14, 0.62),
        xycoords="axes fraction",
        fontsize=10.5,
        color="#C44E52",
        ha="center",
    )
    ax.annotate(
        "compute-bound:\nstep time now scales with T",
        xy=(0.84, 0.5),
        xycoords="axes fraction",
        fontsize=10.5,
        color="#4C72B0",
        ha="center",
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(tokens, [str(t) for t in tokens])
    ax.set_xlabel("Tokens per step (T)")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title(
        f"Model-level roofline — {data['n_params'] / 1e9:.1f}B-param MLP training on {data['device']}"
    )
    ax.legend(loc="upper left", fontsize=9.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "train_tokens_sweep.svg")
    plt.close(fig)


def cv_mfu_mbu_figure(data: dict, out: Path) -> None:
    """MFU and MBU vs batch for ResNet-50 / ViT-B/16 (instrumented, not 6PT)."""
    peak_f = data["peak_bf16_tflops"]
    peak_b = data["peak_mem_tbs"] * 1000  # GB/s
    i_peak = peak_f * 1e3 / peak_b

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=False)
    colors = {"resnet50": "#4C72B0", "vit_b_16": "#C44E52"}
    markers = {"infer": "o", "train": "s"}
    labels = {"resnet50": "ResNet-50", "vit_b_16": "ViT-B/16"}

    for ax, metric, ylabel, title in [
        (
            axes[0],
            "mfu",
            "MFU (%)",
            f"MFU vs batch — {data['device']}",
        ),
        (
            axes[1],
            "mbu",
            "MBU (%)",
            f"MBU vs batch — {data['device']}",
        ),
    ]:
        for m in data["models"]:
            color = colors[m["name"]]
            for phase in ("infer", "train"):
                batches = [p["batch"] for p in m[phase]]
                if metric == "mfu":
                    ys = [100 * p["tflops"] / peak_f for p in m[phase]]
                else:
                    ys = [100 * p["gbs"] / peak_b for p in m[phase]]
                ax.plot(
                    batches,
                    ys,
                    marker=markers[phase],
                    lw=1.8,
                    color=color,
                    ls="-" if phase == "infer" else "--",
                    label=f"{labels[m['name']]} {phase}",
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"])
        ax.set_xlabel("Batch size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, frameon=False, loc="lower right")

    fig.suptitle(
        f"Instrumented MFU/MBU (FlopCounterMode + module-IO bytes), I_peak≈{i_peak:.0f} FLOP/B",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "cv_mfu_mbu.svg")
    plt.close(fig)

    # Companion: intensity vs batch, with ridge line
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for m in data["models"]:
        color = colors[m["name"]]
        for phase in ("infer", "train"):
            batches = [p["batch"] for p in m[phase]]
            intens = [p["intensity"] for p in m[phase]]
            ax.plot(
                batches,
                intens,
                marker=markers[phase],
                lw=1.8,
                color=color,
                ls="-" if phase == "infer" else "--",
                label=f"{labels[m['name']]} {phase}",
            )
    ax.axhline(i_peak, color="#1f2933", ls=":", lw=1.2)
    ax.annotate(
        f"I_peak ≈ {i_peak:.0f}",
        xy=(0.02, i_peak),
        xycoords=("axes fraction", "data"),
        ha="left",
        va="top",
        fontsize=9,
        color="#475569",
        xytext=(0, -4),
        textcoords="offset points",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Arithmetic intensity I (FLOP/B)")
    ax.set_title(f"CV model intensity vs batch — {data['device']}")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "cv_intensity.svg")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--decode-results", type=Path, default=None)
    ap.add_argument("--train-results", type=Path, default=None)
    ap.add_argument("--cv-results", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = json.loads(args.results.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    concept_figure(args.out)
    evolution_figure(args.out)
    a100_vs_h100_figure(args.out)
    roofline_figure(data, args.out)
    gemm_sweep_figure(data, args.out)
    if args.decode_results:
        decode_sweep_figure(json.loads(args.decode_results.read_text()), args.out)
    if args.train_results:
        train_sweep_figure(json.loads(args.train_results.read_text()), args.out)
    if args.cv_results:
        cv_mfu_mbu_figure(json.loads(args.cv_results.read_text()), args.out)
    print(f"wrote figures to {args.out}")


if __name__ == "__main__":
    main()
