"""Generate figures for the distributed training failure runbook post.

Usage (from repo root)::

    uv run python playground/dist_failure_figures.py \\
        --results playground/dist_failure_results.json \\
        --out content/posts/distributed-training-failure-runbook
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

C_BLUE = "#2563eb"
C_RED = "#dc2626"
C_GREEN = "#16a34a"
C_AMBER = "#d97706"
C_SLATE = "#64748b"
C_PURPLE = "#7c3aed"


def _primary(results: dict, case: str) -> dict:
    return results["cases"][case]["primary"]


def taxonomy_figure(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    groups = [
        (
            "Numerics",
            0.3,
            4.2,
            [
                ("Silent drift", "rank0 clip / EMA copy-back"),
                ("Loss spikes", "z-loss / aux weight typo"),
                ("NaN / Inf", "attn/mask/AMP/Adam/DDP"),
            ],
            "#dbeafe",
            C_BLUE,
        ),
        (
            "Resources",
            3.5,
            4.2,
            [
                ("Memory leak", "retained tensors / graphs"),
                ("Throughput cliff", "comm ≫ local compute"),
            ],
            "#ffedd5",
            C_AMBER,
        ),
        (
            "Distributed systems",
            6.7,
            4.2,
            [
                ("Straggler", "one slow rank paces all"),
                ("Bad node", "persistent local slowdown"),
                ("NCCL hang", "missing collective peer"),
            ],
            "#fee2e2",
            C_RED,
        ),
    ]

    ax.text(5, 5.6, "Distributed training failure taxonomy", ha="center", fontsize=14, color="#0f172a")
    for title, x, y, items, face, edge in groups:
        h = 0.7 + 0.85 * len(items)
        ax.add_patch(
            plt.Rectangle((x, y - h + 0.5), 2.9, h, facecolor=face, edgecolor=edge, lw=1.5)
        )
        ax.text(x + 1.45, y + 0.15, title, ha="center", va="bottom", fontsize=12, color=edge, fontweight="bold")
        for i, (name, desc) in enumerate(items):
            yy = y - 0.55 - i * 0.85
            ax.text(x + 0.15, yy, name, fontsize=11, color="#0f172a", fontweight="bold")
            ax.text(x + 0.15, yy - 0.32, desc, fontsize=9, color=C_SLATE)

    fig.tight_layout()
    fig.savefig(out / "failure_taxonomy.svg", bbox_inches="tight")
    plt.close(fig)


def nan_catalog_figure(results: dict, out: Path) -> None:
    nan = _primary(results, "nan")
    recipes = nan["recipes"]
    # Stable display order.
    labels = [r["name"] for r in recipes]
    where_colors = {
        "loss": C_RED,
        "grad": C_AMBER,
        "activation": C_PURPLE,
        "softmax": "#0891b2",
        "optimizer_state": "#be185d",
        "param": C_SLATE,
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), gridspec_kw={"width_ratios": [1.35, 1.0]})

    ax = axes[0]
    y = np.arange(len(recipes))
    colors = [where_colors.get(r.get("where") or "", C_SLATE) for r in recipes]
    ax.barh(y, [1 if r["triggered"] else 0 for r in recipes], color=colors, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["miss", "triggered"])
    ax.invert_yaxis()
    ax.set_title(f"(a) NaN catalog — {nan['n_triggered']}/{nan['n_recipes']} triggered")
    for i, r in enumerate(recipes):
        tag = r.get("where") or "—"
        ax.text(1.02, i, tag, va="center", fontsize=8, color=colors[i])

    # Legend for where
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=10, label=k)
        for k, c in where_colors.items()
        if any(r.get("where") == k for r in recipes)
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right", title="where")

    ax = axes[1]
    broken = np.asarray(nan["broken_losses"], dtype=np.float64)
    healthy = np.asarray(nan["healthy_losses"], dtype=np.float64)
    b_x = np.arange(len(broken))
    finite_mask = np.isfinite(broken)
    ax.plot(b_x[finite_mask], broken[finite_mask], "o-", color=C_RED, lw=2, label="fp16_overflow")
    if (~finite_mask).any() and finite_mask.any():
        i = int(np.argmax(~finite_mask))
        last = float(broken[finite_mask][-1])
        ax.axvline(i, color=C_RED, ls="--", alpha=0.7)
        ax.scatter([i], [last], color=C_RED, s=80, zorder=5)
        ax.annotate("NaN", (i, last), textcoords="offset points", xytext=(8, 8), color=C_RED, fontsize=10)
    ax.plot(np.arange(len(healthy)), healthy, "-", color=C_GREEN, lw=2, label="healthy control")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("(b) Example: FP16 overflow curve")
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out / "nan_catalog.svg", bbox_inches="tight")
    plt.close(fig)


def nan_and_spike_figure(results: dict, out: Path) -> None:
    spike = _primary(results, "loss_spike")
    nan = _primary(results, "nan")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax = axes[0]
    # Compact recipe trigger strip.
    recipes = nan["recipes"]
    triggered = [1 if r["triggered"] else 0 for r in recipes]
    ax.bar(range(len(recipes)), triggered, color=C_RED, width=0.8)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["no", "yes"])
    ax.set_xticks(range(len(recipes)))
    ax.set_xticklabels([r["name"] for r in recipes], rotation=75, ha="right", fontsize=7)
    ax.set_title(f"(a) NaN recipes triggered ({nan['n_triggered']}/{nan['n_recipes']})")

    ax = axes[1]
    losses = np.asarray(spike["losses"], dtype=np.float64)
    ax.plot(np.arange(len(losses)), losses, "-", color=C_BLUE, lw=1.8)
    for s in spike["spike_steps"]:
        ax.scatter([s], [losses[s]], color=C_RED, s=70, zorder=5)
        ax.annotate(
            f"{spike['spike_ratio_vs_median'][str(s)]:.0f}×",
            (s, losses[s]),
            textcoords="offset points",
            xytext=(6, 8),
            color=C_RED,
            fontsize=9,
        )
    ax.axhline(spike["median_loss"], color=C_SLATE, ls=":", lw=1)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("(b) Loss spikes from z-loss weight typo")

    fig.tight_layout()
    fig.savefig(out / "nan_and_loss_spike.svg", bbox_inches="tight")
    plt.close(fig)


def drift_and_leak_figure(results: dict, out: Path) -> None:
    drift = _primary(results, "numerical_drift")
    leak = _primary(results, "memory_leak")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    steps = np.arange(len(drift["max_param_diff"]))
    ax.semilogy(
        steps,
        np.maximum(np.asarray(drift["max_param_diff"], dtype=np.float64), 1e-12),
        color=C_RED,
        lw=2,
        label="rank0-only grad clip",
    )
    if "max_param_diff_control" in drift:
        ax.semilogy(
            steps,
            np.maximum(np.asarray(drift["max_param_diff_control"], dtype=np.float64), 1e-12),
            color=C_GREEN,
            lw=1.8,
            label="clip on all ranks",
        )
    if "ema_student_param_diff" in drift:
        ax.semilogy(
            np.arange(len(drift["ema_student_param_diff"])),
            np.maximum(np.asarray(drift["ema_student_param_diff"], dtype=np.float64), 1e-12),
            color=C_AMBER,
            lw=2,
            ls="--",
            label="EMA→student copy on rank0",
        )
    elif "bn_max_buffer_diff" in drift:
        ax.semilogy(
            np.arange(len(drift["bn_max_buffer_diff"])),
            np.maximum(np.asarray(drift["bn_max_buffer_diff"], dtype=np.float64), 1e-12),
            color=C_AMBER,
            lw=2,
            ls="--",
            label="second drift series",
        )
    ax.axvline(drift["onset_step"], color=C_PURPLE, ls=":", label="onset")
    ax.set_xlabel("step")
    ax.set_ylabel("max |Δparam| across ranks")
    ax.set_title("(a) Real silent drift on TinyTransformerLM")
    ax.legend(frameon=False, fontsize=7.5, loc="best")

    ax = axes[1]
    leaky = np.asarray(leak["allocated_mb_leaky"], dtype=np.float64)
    fixed = np.asarray(leak["allocated_mb_fixed"], dtype=np.float64)
    ax.plot(np.arange(len(leaky)), leaky, color=C_RED, lw=2, label="retain every activation")
    ax.plot(np.arange(len(fixed)), fixed, color=C_GREEN, lw=2, label="clear debug buffer")
    ax.set_xlabel("step")
    ax.set_ylabel("retained / allocated (MB)")
    ax.set_title("(b) Memory leak from debug retention")
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out / "drift_and_memory_leak.svg", bbox_inches="tight")
    plt.close(fig)


def straggler_bad_node_figure(results: dict, out: Path) -> None:
    st = _primary(results, "straggler")
    bn = _primary(results, "bad_node")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax = axes[0]
    # Per-rank forward time (local compute) — seqlen skew shows up here.
    for r in results["cases"]["straggler"]["ranks"]:
        key = "local_compute_ms" if "local_compute_ms" in r else "collective_ms"
        ys = np.asarray(r[key], dtype=np.float64)
        color = C_RED if r["rank"] == st["straggler_rank"] else C_BLUE
        label = f"rank {r['rank']}" + (" (long-doc)" if r["rank"] == st["straggler_rank"] else " (short)")
        ax.plot(np.arange(len(ys)), ys, "-o", color=color, ms=4, lw=1.8, label=label)
    ax.axvline(st["onset_step"], color=C_AMBER, ls="--")
    ax.set_xlabel("step")
    ax.set_ylabel("forward time (ms)")
    ax.set_title("(a) Straggler from seqlen skew")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    means = bn["per_rank_local_mean_ms"]
    ranks = np.arange(len(means))
    colors = [C_RED if i in bn["flagged_ranks"] else C_BLUE for i in ranks]
    ax.bar(ranks, np.maximum(means, 1e-6), color=colors, width=0.55, log=True)
    ax.set_xticks(ranks)
    ax.set_xticklabels([f"rank {i}" for i in ranks])
    ax.set_ylabel("local compute before collective (ms)")
    ax.set_title("(b) Host-skew 'bad node' via local timers")
    for i, m in enumerate(means):
        ax.text(
            i,
            max(m, 1e-6) * 1.8,
            f"{bn['ratio_vs_fastest'][i]:.0f}×",
            ha="center",
            fontsize=9,
            color=colors[i],
        )

    fig.tight_layout()
    fig.savefig(out / "straggler_and_bad_node.svg", bbox_inches="tight")
    plt.close(fig)


def hang_and_cliff_figure(results: dict, out: Path) -> None:
    hang = results["cases"]["nccl_hang"]
    cliff = _primary(results, "throughput_cliff")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("(a) Collective hang timeline", loc="left")

    # Simple swimlane timeline.
    ax.plot([0.5, 9], [4.2, 4.2], color="#cbd5e1", lw=6, solid_capstyle="round")
    ax.plot([0.5, 9], [2.2, 2.2], color="#cbd5e1", lw=6, solid_capstyle="round")
    ax.text(0.2, 4.2, "rank 0", va="center", ha="right", fontsize=10)
    ax.text(0.2, 2.2, "rank 1", va="center", ha="right", fontsize=10)

    # Healthy collective
    ax.plot([1.0, 2.2], [4.2, 4.2], color=C_GREEN, lw=8, solid_capstyle="butt")
    ax.plot([1.0, 2.2], [2.2, 2.2], color=C_GREEN, lw=8, solid_capstyle="butt")
    ax.text(1.6, 4.85, "allreduce OK", ha="center", fontsize=9, color=C_GREEN)

    # Rank 1 exits
    ax.scatter([4.0], [2.2], color=C_RED, s=120, zorder=5, marker="x", linewidths=3)
    ax.text(4.0, 1.4, "empty pack\ncontinue", ha="center", fontsize=9, color=C_RED)

    # Rank 0 waits then errors
    ax.plot([4.0, 7.2], [4.2, 4.2], color=C_AMBER, lw=8, solid_capstyle="butt")
    ax.scatter([7.2], [4.2], color=C_RED, s=90, zorder=5)
    ax.text(5.6, 4.85, "blocked in DDP/NCCL", ha="center", fontsize=9, color=C_AMBER)

    symptom = hang.get("symptom", "")
    short = symptom if len(symptom) < 70 else symptom[:67] + "…"
    ax.text(5, 0.4, short, ha="center", fontsize=8, color=C_SLATE, style="italic")

    ax = axes[1]
    sweep = cliff["sweep"]
    if "tokens_per_rank" in sweep[0]:
        xs = [r["tokens_per_rank"] for r in sweep]
        ys = [r.get("tokens_per_s", r.get("samples_per_s")) for r in sweep]
        xlabel = "tokens / rank"
        ylabel = "tokens / s"
    else:
        xs = [r["batch_size_per_rank"] for r in sweep]
        ys = [r["samples_per_s"] for r in sweep]
        xlabel = "batch size / rank"
        ylabel = "samples / s"
    ax.plot(xs, ys, "-o", color=C_BLUE, lw=2, ms=6)
    ax.axvline(cliff["cliff_batch_size"], color=C_RED, ls="--", label="cliff (tiny microbatch)")
    ax.axvline(cliff["peak_batch_size"], color=C_GREEN, ls=":", label="peak throughput")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("(b) Throughput cliff vs tokens/rank")
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out / "hang_and_throughput_cliff.svg", bbox_inches="tight")
    plt.close(fig)


def runbook_flowchart(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(6, 9.5, "Distributed failure triage", ha="center", fontsize=14, color="#0f172a")

    def box(x, y, w, h, text, face, edge):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, color="#0f172a", wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color=C_SLATE, lw=1.4))

    box(4.0, 8.2, 4.0, 0.9, "Job unhealthy?\n(loss / step time / hang)", "#f1f5f9", C_SLATE)
    arrow(6, 8.2, 6, 7.7)

    box(3.5, 6.7, 5.0, 0.9, "Hard fail or soft degrade?", "#e2e8f0", C_SLATE)
    # left hard
    arrow(4.2, 6.7, 2.0, 6.0)
    box(0.3, 4.9, 3.4, 1.1, "HARD\nNaN / NCCL hang / OOM", "#fee2e2", C_RED)
    arrow(2.0, 4.9, 2.0, 4.3)
    box(0.3, 3.1, 3.4, 1.1, "isfinite checks\nNCCL_DEBUG + heartbeats\nmemory snapshots", "#fff1f2", C_RED)
    arrow(2.0, 3.1, 2.0, 2.5)
    box(0.3, 1.3, 3.4, 1.1, "Fix numerics / replace node\nrestart with async error handling", "#fecaca", C_RED)

    # right soft
    arrow(7.8, 6.7, 10.0, 6.0)
    box(8.3, 4.9, 3.4, 1.1, "SOFT\nspikes / drift / slow", "#dbeafe", C_BLUE)
    arrow(10.0, 4.9, 10.0, 4.3)
    box(8.3, 3.1, 3.4, 1.1, "per-rank loss + timers\nweight checksums\nRSS / allocated curve", "#eff6ff", C_BLUE)
    arrow(10.0, 3.1, 10.0, 2.5)
    box(8.3, 1.3, 3.4, 1.1, "quarantine batch / rank\nraise microbatch / overlap", "#bfdbfe", C_BLUE)

    # center bridge
    box(4.0, 3.6, 4.0, 1.2, "Always log:\nrank, step, loss, step_ms,\nmem, collective_ms", "#f8fafc", C_PURPLE)

    fig.tight_layout()
    fig.savefig(out / "runbook_flowchart.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("playground/dist_failure_results.json"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("content/posts/distributed-training-failure-runbook"),
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.results.read_text())

    taxonomy_figure(args.out)
    nan_catalog_figure(results, args.out)
    nan_and_spike_figure(results, args.out)
    drift_and_leak_figure(results, args.out)
    straggler_bad_node_figure(results, args.out)
    hang_and_cliff_figure(results, args.out)
    runbook_flowchart(args.out)

    # Copy results into the page bundle for the post.
    (args.out / "dist_failure_results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote figures + results to {args.out}")


if __name__ == "__main__":
    main()
