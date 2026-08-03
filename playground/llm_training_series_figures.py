"""Generate schematic figures for the LLM training series posts.

Run from the repository root:

    uv run python playground/llm_training_series_figures.py

The diagrams are hand-authored explanatory SVGs, not measured results.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt

from llm_training_figlib import (
    BLUE,
    BLUE_FILL,
    FILL,
    GRAY_FILL,
    GREEN,
    GREEN_FILL,
    INK,
    LINE,
    MUTED,
    ORANGE,
    ORANGE_FILL,
    PURPLE,
    PURPLE_FILL,
    RED,
    RED_FILL,
    YELLOW_FILL,
    arrow,
    box,
    clean,
    rect,
    save,
)

POSTS = {
    "pipeline-parallelism-gpipe": [
        "naive_model_parallel.svg",
        "gpipe_microbatch.svg",
        "rematerialization.svg",
        "bubble_vs_m.svg",
    ],
    "data-parallelism-ddp-ring-allreduce": [
        "parameter_server.svg",
        "async_sgd_staleness.svg",
        "ring_allreduce_reduce_scatter.svg",
        "ring_allreduce_allgather.svg",
        "ring_bandwidth.svg",
    ],
    "zero-redundancy-optimizer": [
        "memory_breakdown.svg",
        "mixed_precision_memory.svg",
        "zero_stages.svg",
        "zero_vs_model_parallel.svg",
        "zero_offload.svg",
    ],
    "tensor-parallelism-megatron": [
        "row_vs_column_split.svg",
        "mlp_tp.svg",
        "attention_tp.svg",
        "embedding_vocab_parallel.svg",
        "tp_dp_hybrid.svg",
    ],
    "llm-training-series": [
        "series_map.svg",
    ],
    "megatron-distributed-init": [
        "process_group_mesh.svg",
        "dp_tp_pp_ranks.svg",
        "init_flow.svg",
    ],
    "megatron-model-parallel-internals": [
        "column_parallel_linear.svg",
        "row_parallel_linear.svg",
        "parallel_attention_block.svg",
        "vocab_parallel_embedding.svg",
        "parallel_cross_entropy.svg",
    ],
    "megatron-mixed-precision-training": [
        "precision_memory_table.svg",
        "amp_flow.svg",
        "dynamic_loss_scale.svg",
        "grad_clip_with_mp.svg",
    ],
    "moe-expert-parallelism-principles": [
        "gshard_moe_layer.svg",
        "gate_top2_capacity.svg",
        "ep_dp_layout.svg",
        "all_to_all_dispatch.svg",
        "ep_dp_tp.svg",
    ],
    "moe-deepspeed-megatron-internals": [
        "moe_init_flow.svg",
        "moe_layer_structure.svg",
        "ep_group_vs_dp.svg",
    ],
    "sequence-parallelism-megatron-sp": [
        "tp_activation_hotspots.svg",
        "megatron_sp_layernorm.svg",
        "tp_sp_mlp.svg",
        "selective_recompute.svg",
    ],
    "sequence-parallelism-ulysses": [
        "ulysses_a2a.svg",
        "megatron_vs_ulysses_comm.svg",
        "ulysses_zero3.svg",
    ],
    "ring-attention": [
        "online_softmax_blocks.svg",
        "ring_kv_passing.svg",
        "chunk_size_tradeoff.svg",
    ],
    "megatron-context-parallel": [
        "cp_init_groups.svg",
        "naive_vs_balanced_ring.svg",
        "cp_comm_overlap.svg",
    ],
    "megatron-tp-comm-overlap": [
        "naive_ag_vs_overlap.svg",
        "p2p_ag_overlap.svg",
        "rs_overlap_p2p.svg",
        "bulk_ag_rs.svg",
    ],
    "zero3-intra-layer-partitioning": [
        "marketing_inter_layer.svg",
        "actual_intra_layer.svg",
        "zero3_collectives.svg",
    ],
}

DEFAULT_POSTS = list(POSTS.keys())


def _rank_row(ax, y: float, label: str, chunks: list[str], fills: list[str] | None = None) -> None:
    ax.text(0.4, y + 0.25, label, ha="right", va="center", fontsize=9, color=INK)
    fills = fills or [BLUE_FILL, GREEN_FILL, ORANGE_FILL, PURPLE_FILL]
    for i, chunk in enumerate(chunks):
        rect(ax, (0.7 + i * 1.05, y), 0.85, 0.5, fc=fills[i % len(fills)], ec=INK, lw=0.9)
        ax.text(1.125 + i * 1.05, y + 0.25, chunk, ha="center", va="center", fontsize=8)


def _mini_timeline(ax, x: float, y: float, stages: list[tuple[str, str]], title: str) -> None:
    ax.text(x, y + 1.05, title, fontsize=9.5, color=INK, fontweight="bold")
    cursor = x
    colors = {"compute": BLUE_FILL, "comm": ORANGE_FILL, "wait": RED_FILL, "free": GREEN_FILL}
    edges = {"compute": BLUE, "comm": ORANGE, "wait": RED, "free": GREEN}
    for name, kind in stages:
        rect(ax, (cursor, y), 1.05, 0.55, fc=colors[kind], ec=edges[kind], lw=1.0)
        ax.text(cursor + 0.525, y + 0.275, name, ha="center", va="center", fontsize=7.5)
        cursor += 1.1


def naive_model_parallel(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    clean(ax, (0, 10), (0, 4.7))
    ax.set_title("Naive layer-wise model parallelism leaves large bubbles")
    for stage in range(4):
        y = 3.6 - 0.8 * stage
        ax.text(0.55, y + 0.18, f"GPU {stage}", ha="right", va="center", fontsize=9, color=MUTED)
        for t in range(8):
            rect(ax, (0.9 + t, y), 0.78, 0.36, fc=GRAY_FILL, ec="white", alpha=0.75)
        rect(ax, (0.9 + stage, y), 0.78, 0.36, fc=BLUE_FILL, ec=BLUE)
        rect(ax, (0.9 + 7 - stage, y), 0.78, 0.36, fc=ORANGE_FILL, ec=ORANGE)
        ax.text(1.29 + stage, y + 0.18, "F", ha="center", va="center", fontsize=9, color=BLUE)
        ax.text(1.29 + 7 - stage, y + 0.18, "B", ha="center", va="center", fontsize=9, color=ORANGE)
    ax.text(5, 0.35, "Bubble fraction = (K - 1) / K", ha="center", fontsize=10, color=MUTED)
    save(fig, out / "naive_model_parallel.svg")


def gpipe_microbatch(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.0))
    clean(ax, (0, 12), (0, 5.1))
    ax.set_title("GPipe fills the pipeline with micro-batches")
    for stage in range(4):
        y = 4.0 - 0.85 * stage
        ax.text(0.65, y + 0.18, f"stage {stage}", ha="right", va="center", fontsize=9, color=MUTED)
        for t in range(11):
            rect(ax, (1.1 + 0.82 * t, y), 0.62, 0.35, fc=GRAY_FILL, ec="white", alpha=0.35)
    for mb in range(6):
        for stage in range(4):
            x = 1.1 + 0.82 * (mb + stage)
            y = 4.0 - 0.85 * stage
            rect(ax, (x, y), 0.62, 0.35, fc=BLUE_FILL, ec=BLUE)
            ax.text(x + 0.31, y + 0.175, f"F{mb}", ha="center", va="center", fontsize=7.5, color=BLUE)
    for mb in range(6):
        for stage in range(4):
            x = 1.1 + 0.82 * (6 + 3 - mb + 3 - stage)
            if x > 9.7:
                continue
            y = 4.0 - 0.85 * stage
            rect(ax, (x, y), 0.62, 0.35, fc=ORANGE_FILL, ec=ORANGE)
            ax.text(x + 0.31, y + 0.175, f"B{mb}", ha="center", va="center", fontsize=7.5, color=ORANGE)
    ax.text(6, 0.35, "More micro-batches amortize fixed ramp-up and ramp-down bubbles.", ha="center", fontsize=10, color=MUTED)
    save(fig, out / "gpipe_microbatch.svg")


def rematerialization(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    clean(ax, (0, 9), (0, 4.7))
    ax.set_title("Activation checkpointing keeps boundaries and recomputes interiors")
    for i in range(4):
        box(ax, (0.8 + 1.4 * i, 3.05), 1.0, 0.55, f"L{i}", fc=BLUE_FILL, ec=BLUE)
        box(ax, (0.8 + 1.4 * i, 1.35), 1.0, 0.55, f"L{i}", fc=YELLOW_FILL, ec=ORANGE)
        if i < 3:
            arrow(ax, (1.8 + 1.4 * i, 3.32), (2.2 + 1.4 * i, 3.32), color=BLUE)
            arrow(ax, (2.2 + 1.4 * i, 1.62), (1.8 + 1.4 * i, 1.62), color=ORANGE)
    box(ax, (6.7, 3.05), 1.2, 0.55, "loss", fc=ORANGE_FILL, ec=ORANGE)
    ax.text(4.5, 2.45, "save partition inputs, not every activation", ha="center", fontsize=10, color=MUTED)
    ax.text(4.5, 0.6, "Peak memory: O(N + (N/M) * (L/K) * d)", ha="center", fontsize=10, color=MUTED)
    save(fig, out / "rematerialization.svg")


def bubble_vs_m(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for k, color in [(2, BLUE), (4, GREEN), (8, ORANGE), (16, PURPLE)]:
        m = list(range(1, 65))
        ax.plot(m, [(k - 1) / (k + x - 1) for x in m], label=f"K={k}", color=color, linewidth=2)
        ax.axvline(4 * k, color=color, linestyle=":", alpha=0.25)
    ax.set_title("GPipe bubble fraction falls as micro-batches increase")
    ax.set_xlabel("micro-batches M")
    ax.set_ylabel("(K - 1) / (K + M - 1)")
    ax.set_ylim(0, 1)
    ax.grid(True, color="#e5e7eb")
    ax.legend(frameon=False)
    save(fig, out / "bubble_vs_m.svg")


def parameter_server(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    clean(ax, (0, 9), (0, 4.7))
    ax.set_title("Parameter server: the server link becomes the hot spot")
    box(ax, (3.55, 2.0), 1.9, 1.0, "Parameter\nserver", fc=ORANGE_FILL, ec=ORANGE)
    for i, (x, y) in enumerate([(0.7, 3.45), (0.7, 2.0), (0.7, 0.55), (6.7, 3.45), (6.7, 2.0), (6.7, 0.55)]):
        box(ax, (x, y), 1.2, 0.6, f"worker {i}", fc=BLUE_FILL, ec=BLUE, fontsize=9)
        arrow(ax, (x + 1.2, y + 0.3), (3.55, 2.5), color=BLUE)
        arrow(ax, (5.45, 2.5), (x, y + 0.3), color=ORANGE)
    ax.text(4.5, 0.25, "Workers push gradients and pull parameters; traffic concentrates at the server.", ha="center", fontsize=9.5, color=MUTED)
    save(fig, out / "parameter_server.svg")


def async_sgd_staleness(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    clean(ax, (0, 9), (0, 4.6))
    ax.set_title("Asynchronous SGD hides waiting by tolerating stale weights")
    for i in range(5):
        x = 1.0 + 1.35 * i
        box(ax, (x, 3.15), 1.0, 0.45, f"step {i}", fc=BLUE_FILL, ec=BLUE, fontsize=8)
        box(ax, (x, 1.35), 1.0, 0.45, f"W{i}", fc=ORANGE_FILL, ec=ORANGE, fontsize=8)
        arrow(ax, (x + 0.5, 3.15), (x + 0.5, 1.8), color=LINE)
    arrow(ax, (2.85, 1.58), (4.2, 3.15), color=RED)
    arrow(ax, (4.2, 1.58), (5.55, 3.15), color=RED)
    ax.text(4.5, 0.55, "Workers keep computing while updates arrive late.", ha="center", fontsize=10, color=MUTED)
    save(fig, out / "async_sgd_staleness.svg")


def ring_allreduce_reduce_scatter(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    clean(ax, (0, 9), (0, 5))
    ax.set_title("Ring All-Reduce phase 1: reduce-scatter")
    nodes = [(2, 3.7), (6.8, 3.7), (6.8, 1.2), (2, 1.2)]
    for i, (x, y) in enumerate(nodes):
        box(ax, (x - 0.45, y - 0.25), 0.9, 0.5, f"GPU{i}", fc=BLUE_FILL, ec=BLUE, fontsize=8)
        _rank_row(ax, y - 0.85, "", ["A", "B", "C", "D"])
    for i in range(4):
        arrow(ax, nodes[i], nodes[(i + 1) % 4], color=LINE)
    ax.text(4.5, 0.35, "After N - 1 hops, each rank owns one fully reduced shard.", ha="center", fontsize=10, color=MUTED)
    save(fig, out / "ring_allreduce_reduce_scatter.svg")


def ring_allreduce_allgather(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    clean(ax, (0, 9), (0, 5))
    ax.set_title("Ring All-Reduce phase 2: all-gather")
    nodes = [(2, 3.7), (6.8, 3.7), (6.8, 1.2), (2, 1.2)]
    fills = [BLUE_FILL, GREEN_FILL, ORANGE_FILL, PURPLE_FILL]
    for i, (x, y) in enumerate(nodes):
        box(ax, (x - 0.45, y - 0.25), 0.9, 0.5, f"GPU{i}", fc=fills[i], ec=INK, fontsize=8)
        ax.text(x, y - 0.75, f"owns shard {i}", ha="center", fontsize=8, color=MUTED)
    for i in range(4):
        arrow(ax, nodes[i], nodes[(i + 1) % 4], color=LINE)
    ax.text(4.5, 0.35, "Reduced shards circulate until every rank reconstructs the full tensor.", ha="center", fontsize=10, color=MUTED)
    save(fig, out / "ring_allreduce_allgather.svg")


def ring_bandwidth(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    clean(ax, (0, 8.5), (0, 4.5))
    ax.set_title("Ring traffic per rank")
    box(ax, (0.7, 2.7), 1.8, 0.65, "tensor\nsize M", fc=BLUE_FILL, ec=BLUE)
    box(ax, (3.3, 2.7), 1.8, 0.65, "N chunks\nsize M/N", fc=GREEN_FILL, ec=GREEN)
    box(ax, (5.9, 2.7), 1.8, 0.65, "two phases", fc=PURPLE_FILL, ec=PURPLE)
    arrow(ax, (2.5, 3.03), (3.3, 3.03), color=LINE)
    arrow(ax, (5.1, 3.03), (5.9, 3.03), color=LINE)
    box(ax, (1.0, 1.3), 2.5, 0.65, "reduce-scatter\n(N - 1)M/N", fc=ORANGE_FILL, ec=ORANGE, fontsize=9)
    box(ax, (5.0, 1.3), 2.5, 0.65, "all-gather\n(N - 1)M/N", fc=ORANGE_FILL, ec=ORANGE, fontsize=9)
    ax.text(4.25, 0.55, "total = 2 * (N - 1) / N * M", ha="center", fontsize=12, color=INK, fontweight="bold")
    save(fig, out / "ring_bandwidth.svg")


def memory_breakdown(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    clean(ax, (0, 9), (0, 4.6))
    ax.set_title("Training memory is not just model weights")
    blocks = [("params", 1.2, BLUE_FILL, BLUE), ("grads", 1.2, GREEN_FILL, GREEN), ("optimizer\nstates", 1.8, ORANGE_FILL, ORANGE), ("activations", 1.5, PURPLE_FILL, PURPLE), ("buffers +\nfragments", 1.6, GRAY_FILL, MUTED)]
    x = 0.7
    for name, w, fc, ec in blocks:
        rect(ax, (x, 2.0), w, 0.9, fc=fc, ec=ec)
        ax.text(x + w / 2, 2.45, name, ha="center", va="center", fontsize=9)
        x += w
    ax.text(4.5, 1.1, "ZeRO targets model states first, then residual states.", ha="center", fontsize=10, color=MUTED)
    save(fig, out / "memory_breakdown.svg")


def mixed_precision_memory(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    clean(ax, (0, 9), (0, 4.6))
    ax.set_title("Mixed precision still keeps fp32 state for Adam")
    parts = [("fp16 params", 2, BLUE_FILL), ("fp16 grads", 2, GREEN_FILL), ("fp32 master", 4, PURPLE_FILL), ("Adam m", 4, ORANGE_FILL), ("Adam v", 4, ORANGE_FILL)]
    total = sum(p[1] for p in parts)
    x = 0.8
    for name, value, fc in parts:
        w = 7.4 * value / total
        rect(ax, (x, 2.15), w, 0.8, fc=fc, ec=INK)
        ax.text(x + w / 2, 2.55, f"{value}B", ha="center", va="center", fontsize=8)
        ax.text(x + w / 2, 1.8, name, ha="center", va="top", fontsize=8, color=MUTED)
        x += w
    ax.text(4.5, 0.75, "2 + 2 + 4 + 4 + 4 = 16 bytes per parameter", ha="center", fontsize=11, color=INK)
    save(fig, out / "mixed_precision_memory.svg")


def zero_stages(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    clean(ax, (0, 10), (0, 5.2))
    ax.set_title("ZeRO progressively removes replicated model states")
    cols = [("DDP", 0.8), ("ZeRO-1", 3.0), ("ZeRO-2", 5.2), ("ZeRO-3", 7.4)]
    rows = [("params", 3.55, BLUE_FILL), ("grads", 2.45, GREEN_FILL), ("optimizer", 1.35, ORANGE_FILL)]
    for title, x in cols:
        ax.text(x + 0.75, 4.55, title, ha="center", fontsize=11, fontweight="bold")
        for row, y, fc in rows:
            sharded = title == "ZeRO-3" or (title == "ZeRO-2" and row != "params") or (title == "ZeRO-1" and row == "optimizer")
            if sharded:
                for i in range(3):
                    rect(ax, (x + 0.5 * i, y), 0.5, 0.5, fc=fc, ec=INK)
            else:
                box(ax, (x, y), 1.5, 0.5, "full", fc=fc, ec=INK, fontsize=8)
    save(fig, out / "zero_stages.svg")


def zero_vs_model_parallel(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    clean(ax, (0, 10), (0, 4.6))
    ax.set_title("ZeRO has model-parallel shape, but data-parallel semantics")
    headers = [("Question", 0.5), ("ZeRO-3", 3.5), ("Tensor/model parallel", 6.7)]
    for text, x in headers:
        box(ax, (x, 3.55), 2.7, 0.5, text, fc=GRAY_FILL, ec=MUTED, fontsize=9)
    rows = [("What is split?", "state storage", "layer compute"), ("Input per rank", "different data", "same layer path"), ("Full params?", "gather just in time", "not needed"), ("Trade", "memory for comm", "compute for comm")]
    for r, row in enumerate(rows):
        y = 2.85 - 0.65 * r
        for text, x in zip(row, [0.5, 3.5, 6.7], strict=True):
            box(ax, (x, y), 2.7, 0.45, text, fc=FILL, ec=LINE, fontsize=8)
    save(fig, out / "zero_vs_model_parallel.svg")


def zero_offload(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    clean(ax, (0, 9), (0, 4.6))
    ax.set_title("ZeRO-Offload and Infinity extend the memory hierarchy")
    box(ax, (0.8, 2.75), 1.7, 0.75, "GPU HBM\ncompute", fc=BLUE_FILL, ec=BLUE)
    box(ax, (3.55, 2.75), 1.7, 0.75, "CPU DRAM\noptimizer", fc=ORANGE_FILL, ec=ORANGE)
    box(ax, (6.3, 2.75), 1.7, 0.75, "NVMe\ncold shards", fc=PURPLE_FILL, ec=PURPLE)
    arrow(ax, (2.5, 3.15), (3.55, 3.15), color=ORANGE)
    arrow(ax, (5.25, 3.15), (6.3, 3.15), color=PURPLE)
    arrow(ax, (6.3, 2.9), (5.25, 2.9), color=PURPLE)
    arrow(ax, (3.55, 2.9), (2.5, 2.9), color=ORANGE)
    ax.text(4.5, 1.3, "Capacity comes from staging; throughput depends on overlap and prefetch.", ha="center", fontsize=10, color=MUTED)
    save(fig, out / "zero_offload.svg")


def row_vs_column_split(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    clean(ax, (0, 9), (0, 4.7))
    ax.set_title("Two ways to split Y = X A")
    box(ax, (0.8, 3.0), 1.1, 0.5, "X", fc=GRAY_FILL, ec=MUTED)
    box(ax, (2.4, 2.7), 1.4, 1.1, "A split\ncolumns", fc=BLUE_FILL, ec=BLUE)
    box(ax, (4.4, 3.0), 1.5, 0.5, "[Y1 | Y2]", fc=GREEN_FILL, ec=GREEN)
    box(ax, (0.8, 1.0), 1.1, 0.5, "X split", fc=GRAY_FILL, ec=MUTED)
    box(ax, (2.4, 0.7), 1.4, 1.1, "A split\nrows", fc=ORANGE_FILL, ec=ORANGE)
    box(ax, (4.4, 1.0), 1.5, 0.5, "sum Yi", fc=PURPLE_FILL, ec=PURPLE)
    for y in [3.25, 1.25]:
        arrow(ax, (1.9, y), (2.4, y), color=LINE)
        arrow(ax, (3.8, y), (4.4, y), color=LINE)
    ax.text(7.1, 3.2, "Column split:\nconcatenate outputs", ha="center", fontsize=10)
    ax.text(7.1, 1.2, "Row split:\nall-reduce partial sums", ha="center", fontsize=10)
    save(fig, out / "row_vs_column_split.svg")


def mlp_tp(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    clean(ax, (0, 10), (0, 5))
    ax.set_title("Megatron MLP: column-parallel then row-parallel")
    box(ax, (0.7, 2.25), 1.0, 0.55, "X", fc=GRAY_FILL, ec=MUTED)
    for y in [3.05, 1.55]:
        box(ax, (2.2, y), 1.2, 0.5, "A shard", fc=BLUE_FILL, ec=BLUE, fontsize=8)
        box(ax, (4.0, y), 1.2, 0.5, "GeLU", fc=GREEN_FILL, ec=GREEN, fontsize=8)
        box(ax, (5.8, y), 1.2, 0.5, "B shard", fc=ORANGE_FILL, ec=ORANGE, fontsize=8)
        arrow(ax, (1.7, 2.52), (2.2, y + 0.25), color=LINE)
        arrow(ax, (3.4, y + 0.25), (4.0, y + 0.25), color=LINE)
        arrow(ax, (5.2, y + 0.25), (5.8, y + 0.25), color=LINE)
        arrow(ax, (7.0, y + 0.25), (8.0, 2.52), color=LINE)
    box(ax, (8.0, 2.25), 1.2, 0.55, "AllReduce", fc=PURPLE_FILL, ec=PURPLE, fontsize=8)
    save(fig, out / "mlp_tp.svg")


def attention_tp(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    clean(ax, (0, 10), (0, 5))
    ax.set_title("Attention is naturally head-parallel")
    box(ax, (0.7, 2.25), 1.0, 0.55, "X", fc=GRAY_FILL, ec=MUTED)
    for i, y in enumerate([3.35, 2.25, 1.15]):
        box(ax, (2.2, y), 1.35, 0.5, f"QKV\nheads {i}", fc=BLUE_FILL, ec=BLUE, fontsize=8)
        box(ax, (4.2, y), 1.35, 0.5, "local\nattention", fc=GREEN_FILL, ec=GREEN, fontsize=8)
        box(ax, (6.2, y), 1.35, 0.5, "output\nshard", fc=ORANGE_FILL, ec=ORANGE, fontsize=8)
        arrow(ax, (1.7, 2.52), (2.2, y + 0.25), color=LINE)
        arrow(ax, (3.55, y + 0.25), (4.2, y + 0.25), color=LINE)
        arrow(ax, (5.55, y + 0.25), (6.2, y + 0.25), color=LINE)
        arrow(ax, (7.55, y + 0.25), (8.2, 2.52), color=LINE)
    box(ax, (8.2, 2.25), 1.2, 0.55, "AllReduce", fc=PURPLE_FILL, ec=PURPLE, fontsize=8)
    save(fig, out / "attention_tp.svg")


def embedding_vocab_parallel(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    clean(ax, (0, 10), (0, 5))
    ax.set_title("Vocab-parallel embedding and loss avoid gathering huge logits")
    box(ax, (0.7, 2.25), 1.25, 0.55, "token ids", fc=GRAY_FILL, ec=MUTED)
    for i, y in enumerate([3.3, 2.25, 1.2]):
        box(ax, (2.7, y), 1.5, 0.5, f"vocab\nshard {i}", fc=BLUE_FILL, ec=BLUE, fontsize=8)
        box(ax, (5.0, y), 1.5, 0.5, "lookup\nor logits", fc=GREEN_FILL, ec=GREEN, fontsize=8)
        arrow(ax, (1.95, 2.52), (2.7, y + 0.25), color=LINE)
        arrow(ax, (4.2, y + 0.25), (5.0, y + 0.25), color=LINE)
        arrow(ax, (6.5, y + 0.25), (7.5, 2.52), color=LINE)
    box(ax, (7.5, 2.25), 1.5, 0.55, "AllReduce\nsmall result", fc=PURPLE_FILL, ec=PURPLE, fontsize=8)
    save(fig, out / "embedding_vocab_parallel.svg")


def tp_dp_hybrid(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    clean(ax, (0, 10), (0, 5))
    ax.set_title("Tensor parallel inside nodes, data parallel across nodes")
    for node in range(2):
        x0 = 0.9 + 5.0 * node
        box(ax, (x0, 3.55), 3.6, 0.35, f"node {node}", fc=GRAY_FILL, ec=MUTED, fontsize=8)
        for i in range(4):
            x = x0 + 1.75 * (i % 2)
            y = 2.55 - 1.0 * (i // 2)
            box(ax, (x, y), 1.25, 0.55, f"GPU {node*4+i}\nTP {i%2}", fc=BLUE_FILL if i < 2 else GREEN_FILL, ec=INK, fontsize=8)
        ax.text(x0 + 1.8, 0.8, "fast TP group", ha="center", fontsize=9, color=MUTED)
    arrow(ax, (4.55, 2.25), (5.35, 2.25), color=ORANGE)
    arrow(ax, (5.35, 2.0), (4.55, 2.0), color=ORANGE)
    ax.text(5, 1.25, "DP gradient sync", ha="center", fontsize=10, color=ORANGE)
    save(fig, out / "tp_dp_hybrid.svg")


def series_map(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    clean(ax, (0, 10), (0, 5))
    ax.set_title("LLM Training parallelism map")
    nodes = [
        ("Data\nParallel", 0.8, 2.3, BLUE_FILL, BLUE),
        ("Tensor\nParallel", 2.8, 3.3, GREEN_FILL, GREEN),
        ("Pipeline\nParallel", 2.8, 1.3, ORANGE_FILL, ORANGE),
        ("ZeRO", 4.8, 2.3, PURPLE_FILL, PURPLE),
        ("Sequence /\nContext", 6.8, 3.3, YELLOW_FILL, ORANGE),
        ("MoE Expert\nParallel", 6.8, 1.3, RED_FILL, RED),
        ("Hybrid\n3D+", 8.6, 2.3, GRAY_FILL, MUTED),
    ]
    for text, x, y, fc, ec in nodes:
        box(ax, (x, y), 1.25, 0.65, text, fc=fc, ec=ec, fontsize=8.5)
    for start, end in [((2.05, 2.62), (2.8, 3.62)), ((2.05, 2.62), (2.8, 1.62)), ((4.05, 3.62), (4.8, 2.62)), ((4.05, 1.62), (4.8, 2.62)), ((6.05, 2.62), (6.8, 3.62)), ((6.05, 2.62), (6.8, 1.62)), ((8.05, 3.62), (8.6, 2.62)), ((8.05, 1.62), (8.6, 2.62))]:
        arrow(ax, start, end, color=LINE)
    ax.text(5, 0.55, "Modern LLM training composes multiple parallel dimensions.", ha="center", fontsize=10, color=MUTED)
    save(fig, out / "series_map.svg")


def tp_activation_hotspots(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    clean(ax, (0, 12), (0, 7))
    ax.set_title("Tensor parallelism shards GEMMs but leaves normalization and dropout replicated")

    for x, label, fc, ec in [
        (0.5, "Input\nLayerNorm", RED_FILL, RED),
        (2.5, "Column\nparallel\nQKV / FC1", BLUE_FILL, BLUE),
        (4.7, "Attention\nor GELU", BLUE_FILL, BLUE),
        (6.7, "Row\nparallel\nProj / FC2", BLUE_FILL, BLUE),
        (8.9, "Dropout\n+ residual", RED_FILL, RED),
    ]:
        box(ax, (x, 3.9), 1.5, 1.2, label, fc=fc, ec=ec, fontsize=8.6)
    for x in [2.0, 4.0, 6.2, 8.2]:
        arrow(ax, (x, 4.5), (x + 0.45, 4.5), color=LINE)
    ax.text(5.9, 5.55, "TP shards compute-heavy GEMMs", ha="center", fontsize=9.5, color=BLUE)

    for i, y in enumerate([2.4, 1.55, 0.7]):
        _rank_row(ax, y, f"TP rank {i}", ["LN", "GEMM shard", "middle shard", "GEMM shard", "dropout"])
    ax.text(
        7.2,
        2.9,
        "The red regions are repeated on every TP rank.\nThey dominate when sequence length grows.",
        fontsize=9,
        color=RED,
        ha="center",
    )
    arrow(ax, (1.3, 3.9), (1.3, 2.9), color=RED)
    arrow(ax, (9.65, 3.9), (9.65, 2.9), color=RED)
    save(fig, out / "tp_activation_hotspots.svg")


def megatron_sp_layernorm(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    clean(ax, (0, 12), (0, 7))
    ax.set_title("Megatron SP shards sequence-local activations and gathers only for TP regions")

    for i, y in enumerate([4.7, 3.75, 2.8, 1.85]):
        box(ax, (0.7, y), 1.4, 0.55, f"rank {i}", fc=GRAY_FILL, ec=MUTED, fontsize=8)
        rect(ax, (2.4, y), 2.4, 0.55, fc=GREEN_FILL, ec=GREEN)
        ax.text(3.6, y + 0.27, f"sequence shard S{i}", ha="center", va="center", fontsize=8.5)
        rect(ax, (5.4, y), 1.7, 0.55, fc=RED_FILL, ec=RED)
        ax.text(6.25, y + 0.27, "LayerNorm\nlocal", ha="center", va="center", fontsize=7.4)
        rect(ax, (8.0, y), 2.0, 0.55, fc=BLUE_FILL, ec=BLUE)
        ax.text(9.0, y + 0.27, "full sequence\nfor TP GEMM", ha="center", va="center", fontsize=7.4)
    ax.text(3.6, 5.65, "SP resident shape: [B, S/t, H]", ha="center", fontsize=9, color=GREEN)
    ax.text(9.0, 5.65, "AllGather before column-parallel compute", ha="center", fontsize=9, color=BLUE)
    arrow(ax, (7.15, 5.0), (7.95, 5.0), color=BLUE)
    arrow(ax, (7.15, 4.05), (7.95, 4.05), color=BLUE)
    arrow(ax, (7.15, 3.1), (7.95, 3.1), color=BLUE)
    arrow(ax, (7.15, 2.15), (7.95, 2.15), color=BLUE)
    box(ax, (3.2, 0.55), 5.8, 0.75, "ReduceScatter after row-parallel output returns to sequence shards", fc=ORANGE_FILL, ec=ORANGE)
    save(fig, out / "megatron_sp_layernorm.svg")


def tp_sp_mlp(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    clean(ax, (0, 13), (0, 7))
    ax.set_title("TP + SP replaces replicated activations with AG/RS around tensor-parallel MLP")

    steps = [
        ("SP shard\n[B,S/t,H]", GREEN_FILL, GREEN),
        ("AllGather\nsequence", ORANGE_FILL, ORANGE),
        ("FC1 column\nparallel", BLUE_FILL, BLUE),
        ("GELU\nsharded H", BLUE_FILL, BLUE),
        ("FC2 row\nparallel", BLUE_FILL, BLUE),
        ("ReduceScatter\nsequence", ORANGE_FILL, ORANGE),
        ("SP shard\nfor dropout", GREEN_FILL, GREEN),
    ]
    x = 0.35
    for label, fc, ec in steps:
        box(ax, (x, 4.0), 1.45, 1.0, label, fc=fc, ec=ec, fontsize=8.2)
        x += 1.8
    for x in [1.82, 3.62, 5.42, 7.22, 9.02, 10.82]:
        arrow(ax, (x, 4.5), (x + 0.28, 4.5), color=LINE)

    ax.text(2.7, 3.25, "same payload class as TP AllReduce,\nbut activations stay sequence-sharded", fontsize=9, ha="center", color=MUTED)
    for i, y in enumerate([2.35, 1.55, 0.75]):
        _rank_row(ax, y, f"rank {i}", ["S shard", "full S", "H shard", "H shard", "S shard"])
    save(fig, out / "tp_sp_mlp.svg")


def selective_recompute(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    clean(ax, (0, 11), (0, 6.2))
    ax.set_title("Selective recomputation keeps expensive activations and replays cheap, bulky ones")

    rows = [
        ("Full save", [("LN / MLP inputs", "free"), ("QKV", "free"), ("scores + softmax", "free"), ("dropout", "free")]),
        ("Full recompute", [("block input", "free"), ("replay whole layer", "wait"), ("no middle saves", "comm"), ("lowest memory", "free")]),
        ("SP + selective", [("LN / MLP inputs", "free"), ("QKV saved", "free"), ("scores recomputed", "wait"), ("best tradeoff", "free")]),
    ]
    for idx, (label, stages) in enumerate(rows):
        y = 4.6 - idx * 1.65
        ax.text(0.6, y + 0.27, label, ha="left", va="center", fontsize=9.5, fontweight="bold")
        _mini_timeline(ax, 2.4, y, stages, "")
    ax.text(8.8, 1.45, "Attention score tensors scale with S^2.\nThey are large, but cheaper to replay\nthan GEMMs are.", fontsize=9, color=RED, ha="center")
    save(fig, out / "selective_recompute.svg")


def ulysses_a2a(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    clean(ax, (0, 12), (0, 7))
    ax.set_title("Ulysses converts sequence shards into head shards with All-to-All")

    for i, y in enumerate([5.0, 4.2, 3.4, 2.6]):
        _rank_row(ax, y, f"rank {i}", [f"S{i}H0", f"S{i}H1", f"S{i}H2", f"S{i}H3"])
    ax.text(2.8, 6.1, "Before: each rank owns one sequence slice and all heads", ha="center", fontsize=9.5)
    box(ax, (5.6, 3.35), 1.2, 1.2, "All-to-All\ntranspose", fc=ORANGE_FILL, ec=ORANGE, fontsize=8.5)
    arrow(ax, (4.9, 4.1), (5.55, 3.95), color=ORANGE)
    arrow(ax, (6.85, 3.95), (7.45, 4.1), color=ORANGE)
    for i, y in enumerate([5.0, 4.2, 3.4, 2.6]):
        _rank_row(ax, y, f"rank {i}", [f"S0H{i}", f"S1H{i}", f"S2H{i}", f"S3H{i}"])
    ax.text(9.6, 6.1, "After: each rank owns all sequence positions for one head", ha="center", fontsize=9.5)
    box(ax, (8.2, 1.0), 2.8, 0.7, "Local attention per head", fc=BLUE_FILL, ec=BLUE)
    save(fig, out / "ulysses_a2a.svg")


def megatron_vs_ulysses_comm(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    clean(ax, (0, 10), (0, 6))
    ax.set_title("Communication scaling: Megatron SP vs Ulysses")
    labels = ["Megatron TP+SP", "Ulysses"]
    values = [8.0, 2.0]
    colors = [BLUE, GREEN]
    fills = [BLUE_FILL, GREEN_FILL]
    for i, (label, value) in enumerate(zip(labels, values, strict=True)):
        y = 3.9 - i * 1.8
        ax.text(0.8, y + 0.25, label, fontsize=10, ha="left", va="center", fontweight="bold")
        rect(ax, (3.1, y), value * 0.65, 0.55, fc=fills[i], ec=colors[i])
        ax.text(3.1 + value * 0.65 + 0.25, y + 0.27, "8Nd" if i == 0 else "8Nd / P", fontsize=10, va="center")
    ax.text(3.1, 1.0, "For Ulysses, P is bounded by the number of attention heads.\nThe scaling story is attractive, but not unbounded.", fontsize=9, color=MUTED)
    save(fig, out / "megatron_vs_ulysses_comm.svg")


def ulysses_zero3(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.7))
    clean(ax, (0, 12), (0, 7))
    ax.set_title("Ulysses can sit inside ZeRO-3: gather weights, sequence-shard compute")
    for x, label in [(0.8, "DP group 0"), (6.6, "DP group 1")]:
        box(ax, (x, 0.75), 4.8, 5.5, "", fc="#ffffff", ec=LINE)
        ax.text(x + 0.25, 5.9, label, fontsize=10, fontweight="bold")
        for i in range(2):
            box(ax, (x + 0.45 + i * 2.1, 4.55), 1.55, 0.7, f"ZeRO shard\nW{i}", fc=PURPLE_FILL, ec=PURPLE, fontsize=8)
            box(ax, (x + 0.45 + i * 2.1, 3.3), 1.55, 0.7, "full W\njust in time", fc=ORANGE_FILL, ec=ORANGE, fontsize=8)
            box(ax, (x + 0.45 + i * 2.1, 1.75), 1.55, 0.9, f"Ulysses\nseq shard {i}", fc=GREEN_FILL, ec=GREEN, fontsize=8)
            arrow(ax, (x + 1.2 + i * 2.1, 4.55), (x + 1.2 + i * 2.1, 4.02), color=ORANGE)
            arrow(ax, (x + 1.2 + i * 2.1, 3.3), (x + 1.2 + i * 2.1, 2.65), color=GREEN)
    arrow(ax, (5.7, 3.9), (6.45, 3.9), color=PURPLE, style="<|-|>")
    ax.text(6.05, 4.2, "grad RS / optimizer shards", fontsize=8, color=PURPLE, ha="center")
    save(fig, out / "ulysses_zero3.svg")


def online_softmax_blocks(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    clean(ax, (0, 11), (0, 6.5))
    ax.set_title("Online softmax lets attention consume score blocks without materializing S x S")
    box(ax, (0.7, 3.8), 1.6, 0.9, "Q block", fc=GREEN_FILL, ec=GREEN)
    for i in range(4):
        box(ax, (3.0 + i * 1.55, 4.05), 1.0, 0.55, f"K{i}", fc=BLUE_FILL, ec=BLUE, fontsize=8)
        box(ax, (3.0 + i * 1.55, 3.25), 1.0, 0.55, f"V{i}", fc=PURPLE_FILL, ec=PURPLE, fontsize=8)
        arrow(ax, (2.3, 4.2), (3.0 + i * 1.55, 4.35), color=LINE)
    box(ax, (3.8, 1.6), 3.8, 1.0, "running row max m\nrunning denominator l\nrunning output O", fc=YELLOW_FILL, ec=ORANGE)
    arrow(ax, (5.5, 3.2), (5.5, 2.65), color=ORANGE)
    ax.text(8.1, 2.0, "Each block updates (m, l, O).\nFinal O matches full softmax.", fontsize=9, color=MUTED)
    save(fig, out / "online_softmax_blocks.svg")


def ring_kv_passing(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    clean(ax, (0, 10), (0, 8))
    ax.set_title("Ring Attention keeps Q fixed and rotates KV blocks")
    coords = [(2, 5.5), (7, 5.5), (7, 1.7), (2, 1.7)]
    for i, (x, y) in enumerate(coords):
        box(ax, (x - 0.85, y - 0.55), 1.7, 1.1, f"GPU {i}\nQ{i} fixed\nK{i},V{i}", fc=GREEN_FILL, ec=GREEN, fontsize=8.3)
    ring = [(2.9, 5.5, 6.1, 5.5), (7, 4.95, 7, 2.25), (6.1, 1.7, 2.9, 1.7), (2, 2.25, 2, 4.95)]
    for x0, y0, x1, y1 in ring:
        arrow(ax, (x0, y0), (x1, y1), color=ORANGE, lw=1.8)
    box(ax, (3.4, 3.25), 3.2, 0.9, "compute current KV\nwhile next KV transfers", fc=BLUE_FILL, ec=BLUE)
    save(fig, out / "ring_kv_passing.svg")


def chunk_size_tradeoff(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    clean(ax, (0, 10), (0, 6))
    ax.set_title("Chunk size must make attention compute at least as long as KV transfer")
    ax.plot([1, 8.5], [1.0, 5.0], color=BLUE, linewidth=2.0)
    ax.plot([1, 8.5], [2.0, 2.0], color=ORANGE, linewidth=2.0)
    ax.text(8.6, 5.0, "compute time ~ c^2", color=BLUE, fontsize=9, va="center")
    ax.text(8.6, 2.0, "KV transfer time ~ c", color=ORANGE, fontsize=9, va="center")
    ax.axvline(3.0, color=RED, linestyle="--", linewidth=1.4)
    ax.text(3.08, 4.6, "c >= F / B", color=RED, fontsize=10)
    ax.text(0.9, 0.45, "small chunks: communication visible", fontsize=8.8, color=MUTED)
    ax.text(5.4, 0.45, "large chunks: more memory, better overlap", fontsize=8.8, color=MUTED)
    ax.set_xlabel("chunk size c")
    ax.set_ylabel("time")
    ax.set_xticks([])
    ax.set_yticks([])
    save(fig, out / "chunk_size_tradeoff.svg")


def cp_init_groups(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    clean(ax, (0, 12), (0, 7))
    ax.set_title("Context parallel groups are formed at matching TP/DP/PP coordinates")
    for r in range(2):
        for c in range(4):
            x = 1.0 + c * 2.4
            y = 4.7 - r * 2.0
            idx = r * 4 + c
            box(ax, (x, y), 1.45, 0.9, f"rank {idx}\nTP{c % 2} CP{c // 2}", fc=BLUE_FILL if c % 2 == 0 else GREEN_FILL, ec=INK, fontsize=8)
    for c in range(2):
        x0 = 1.72 + c * 2.4
        x1 = x0 + 4.8
        arrow(ax, (x0, 5.15), (x1, 5.15), color=ORANGE, style="<|-|>")
        arrow(ax, (x0, 3.15), (x1, 3.15), color=ORANGE, style="<|-|>")
    ax.text(5.9, 1.0, "Same TP/DP/PP position + different sequence shard = one CP group", ha="center", fontsize=10, color=ORANGE)
    save(fig, out / "cp_init_groups.svg")


def naive_vs_balanced_ring(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    clean(ax, (0, 12), (0, 7))
    ax.set_title("Causal masks make naive rings imbalanced; zigzag chunks equalize work")
    box(ax, (0.6, 5.35), 4.8, 0.55, "Naive: rank i owns contiguous chunk i", fc=RED_FILL, ec=RED)
    for i, h in enumerate([0.8, 1.6, 2.4, 3.2]):
        rect(ax, (1.0 + i, 1.2), 0.65, h, fc=RED_FILL, ec=RED)
        ax.text(1.33 + i, 0.85, f"GPU{i}", ha="center", fontsize=8)
    ax.text(3.0, 4.85, "later-token ranks do more causal attention", ha="center", fontsize=8.8, color=RED)

    box(ax, (6.4, 5.35), 4.8, 0.55, "Balanced: pair early and late chunks", fc=GREEN_FILL, ec=GREEN)
    pairs = ["0+7", "1+6", "2+5", "3+4"]
    for i in range(4):
        rect(ax, (6.8 + i, 1.2), 0.65, 2.4, fc=GREEN_FILL, ec=GREEN)
        ax.text(7.13 + i, 2.45, pairs[i], ha="center", fontsize=8)
        ax.text(7.13 + i, 0.85, f"GPU{i}", ha="center", fontsize=8)
    ax.text(8.8, 4.85, "each rank receives one early and one late slice", ha="center", fontsize=8.8, color=GREEN)
    save(fig, out / "naive_vs_balanced_ring.svg")


def cp_comm_overlap(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    clean(ax, (0, 12), (0, 6.3))
    ax.set_title("Megatron CP overlaps KV exchange, attention, and softmax state updates")
    _mini_timeline(ax, 1.0, 4.45, [("send/recv KV", "comm"), ("attention", "compute"), ("lse", "free"), ("idle", "wait")], "NCCL stream")
    _mini_timeline(ax, 1.0, 3.1, [("wait KV", "wait"), ("attention i", "compute"), ("send next", "comm"), ("attention i+2", "compute")], "compute stream 0")
    _mini_timeline(ax, 1.0, 1.75, [("lse update", "free"), ("wait", "wait"), ("attention i+1", "compute"), ("final lse", "free")], "compute stream 1")
    ax.text(7.3, 2.6, "The goal is not zero communication.\nIt is to put communication on a different\ncritical path from GEMM-heavy attention.", fontsize=9.5, color=MUTED)
    save(fig, out / "cp_comm_overlap.svg")


def naive_ag_vs_overlap(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    clean(ax, (0, 11), (0, 6))
    ax.set_title("Naive all-gather serializes before GEMM; overlap starts GEMM on local shards")
    _mini_timeline(ax, 1.0, 3.9, [("AllGather", "comm"), ("GEMM", "compute"), ("GEMM", "compute"), ("done", "free")], "naive")
    _mini_timeline(ax, 1.0, 2.2, [("send shard", "comm"), ("local GEMM", "compute"), ("remote GEMM", "compute"), ("done", "free")], "overlapped")
    ax.text(6.6, 2.7, "Break the collective into exchangeable chunks,\nthen consume each chunk as it arrives.", fontsize=9.2, color=MUTED)
    save(fig, out / "naive_ag_vs_overlap.svg")


def p2p_ag_overlap(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    clean(ax, (0, 12), (0, 7))
    ax.set_title("P2P all-gather overlap rotates input shards through neighboring ranks")
    for i, y in enumerate([5.1, 4.1, 3.1, 2.1]):
        box(ax, (0.8, y), 1.2, 0.55, f"rank {i}", fc=GRAY_FILL, ec=MUTED, fontsize=8)
        for j in range(4):
            fc = GREEN_FILL if j == i else BLUE_FILL
            rect(ax, (2.6 + j * 1.0, y), 0.75, 0.55, fc=fc, ec=INK, lw=0.8)
            ax.text(2.975 + j, y + 0.27, f"D{(i + j) % 4}", ha="center", va="center", fontsize=8)
        arrow(ax, (7.2, y + 0.27), (8.2, y + 0.27), color=ORANGE)
        box(ax, (8.35, y), 1.65, 0.55, "GEMM chunk", fc=BLUE_FILL, ec=BLUE, fontsize=8)
    ax.text(4.5, 6.15, "one new shard per iteration", ha="center", fontsize=9, color=ORANGE)
    save(fig, out / "p2p_ag_overlap.svg")


def rs_overlap_p2p(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    clean(ax, (0, 12), (0, 7))
    ax.set_title("Reduce-scatter overlap sends the output bucket around the ring")
    for i, y in enumerate([5.1, 4.1, 3.1, 2.1]):
        box(ax, (0.8, y), 1.2, 0.55, f"rank {i}", fc=GRAY_FILL, ec=MUTED, fontsize=8)
        box(ax, (2.7, y), 1.5, 0.55, f"bucket C{(i + 1) % 4}", fc=ORANGE_FILL, ec=ORANGE, fontsize=8)
        arrow(ax, (4.35, y + 0.27), (5.2, y + 0.27), color=ORANGE)
        box(ax, (5.35, y), 1.85, 0.55, "add local\npartial", fc=BLUE_FILL, ec=BLUE, fontsize=8)
        arrow(ax, (7.35, y + 0.27), (8.2, y + 0.27), color=ORANGE)
        box(ax, (8.35, y), 1.5, 0.55, f"bucket C{i}", fc=GREEN_FILL, ec=GREEN, fontsize=8)
    ax.text(5.85, 6.15, "compute contribution while the bucket is in hand", ha="center", fontsize=9, color=MUTED)
    save(fig, out / "rs_overlap_p2p.svg")


def bulk_ag_rs(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    clean(ax, (0, 12), (0, 6.2))
    ax.set_title("Bulk AG/RS uses separate streams and user buffers to hide independent collectives")
    _mini_timeline(ax, 1.0, 4.25, [("event", "free"), ("AG/RS in ubuf", "comm"), ("AG/RS", "comm"), ("finish", "free")], "communication stream")
    _mini_timeline(ax, 1.0, 2.65, [("dgrad GEMM", "compute"), ("wgrad GEMM", "compute"), ("GEMM", "compute"), ("finish", "free")], "main compute stream")
    ax.text(7.0, 3.0, "Used when the dependency graph allows\nGEMM and AG/RS to proceed together.\nThe hard part is buffer ownership.", fontsize=9.3, color=MUTED)
    save(fig, out / "bulk_ag_rs.svg")


def marketing_inter_layer(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    clean(ax, (0, 12), (0, 6.5))
    ax.set_title("The memorable but misleading picture: inter-layer shards and broadcasts")
    for i in range(4):
        box(ax, (0.9 + i * 2.55, 4.5), 1.55, 0.75, f"GPU {i}\nM{i}", fc=RED_FILL, ec=RED, fontsize=8.5)
        box(ax, (0.9 + i * 2.55, 2.0), 1.55, 1.4, f"layers\n{i*4}-{i*4+3}", fc=GRAY_FILL, ec=MUTED, fontsize=8.5)
        arrow(ax, (1.68 + i * 2.55, 4.5), (1.68 + i * 2.55, 3.45), color=RED)
    ax.text(6.0, 1.0, "This resembles pipeline-style layer ownership.\nIt is not how current DeepSpeed ZeRO-3 partitions parameters.", ha="center", fontsize=9.3, color=RED)
    save(fig, out / "marketing_inter_layer.svg")


def actual_intra_layer(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    clean(ax, (0, 12), (0, 6.5))
    ax.set_title("Current ZeRO-3 partitions each parameter as a flattened 1D tensor")
    box(ax, (1.0, 4.6), 10.0, 0.75, "one Linear weight or parameter tensor, flattened and padded", fc=BLUE_FILL, ec=BLUE)
    colors = [GREEN_FILL, ORANGE_FILL, PURPLE_FILL, RED_FILL]
    edges = [GREEN, ORANGE, PURPLE, RED]
    for i in range(4):
        rect(ax, (1.0 + i * 2.5, 3.25), 2.5, 0.65, fc=colors[i], ec=edges[i])
        ax.text(2.25 + i * 2.5, 3.58, f"rank {i} slice", ha="center", va="center", fontsize=8.5)
        box(ax, (1.35 + i * 2.5, 1.55), 1.8, 0.75, f"GPU {i}\nkeeps slice", fc=colors[i], ec=edges[i], fontsize=8.5)
        arrow(ax, (2.25 + i * 2.5, 3.25), (2.25 + i * 2.5, 2.32), color=edges[i])
    save(fig, out / "actual_intra_layer.svg")


def zero3_collectives(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    clean(ax, (0, 12), (0, 6.3))
    ax.set_title("The collectives to expect from ZeRO-3: AllGather weights, ReduceScatter gradients")
    steps = [
        ("param shards\nresident", GREEN_FILL, GREEN),
        ("AllGather\nbefore fwd", ORANGE_FILL, ORANGE),
        ("forward\nfull param", BLUE_FILL, BLUE),
        ("AllGather\nbefore bwd", ORANGE_FILL, ORANGE),
        ("backward\nfull param", BLUE_FILL, BLUE),
        ("ReduceScatter\ngrads", PURPLE_FILL, PURPLE),
        ("optimizer\nlocal shard", GREEN_FILL, GREEN),
    ]
    x = 0.35
    for label, fc, ec in steps:
        box(ax, (x, 3.7), 1.35, 0.9, label, fc=fc, ec=ec, fontsize=7.8)
        x += 1.65
    for x in [1.72, 3.37, 5.02, 6.67, 8.32, 9.97]:
        arrow(ax, (x, 4.15), (x + 0.25, 4.15), color=LINE)
    ax.text(6.0, 2.0, "Broadcast is not the steady-state ZeRO-3 training primitive.\nThe important pattern is gather for compute, scatter for ownership.", ha="center", fontsize=9.4, color=MUTED)
    save(fig, out / "zero3_collectives.svg")


def _series_flow(out: Path, filename: str, title: str, nodes: list[tuple[str, str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    clean(ax, (0, 12), (0, 6.2))
    ax.set_title(title)
    step = 10.5 / max(len(nodes), 1)
    for i, (label, fill, edge) in enumerate(nodes):
        x = 0.6 + i * step
        box(ax, (x, 3.0), min(1.55, step - 0.25), 1.05, label, fc=fill, ec=edge, fontsize=8.0)
        if i > 0:
            arrow(ax, (x - 0.35, 3.52), (x, 3.52), color=LINE)
    save(fig, out / filename)


def _rank_grid(out: Path, filename: str, title: str, subtitle: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    clean(ax, (0, 11), (0, 6.5))
    ax.set_title(title)
    fills = [BLUE_FILL, GREEN_FILL]
    edges = [BLUE, GREEN]
    for dp in range(2):
        ax.text(0.75, 4.8 - dp * 2.1, f"DP {dp}", fontsize=10, color=edges[dp], fontweight="bold")
        for pp in range(4):
            for tp in range(2):
                rank = dp * 2 + pp * 4 + tp
                x = 1.7 + pp * 2.0 + tp * 0.85
                y = 4.35 - dp * 2.1 + (1 - tp) * 0.55
                box(ax, (x, y), 0.72, 0.42, f"r{rank}", fc=fills[dp], ec=edges[dp], fontsize=8)
            ax.text(2.15 + pp * 2.0, 3.95 - dp * 2.1, f"PP{pp}", ha="center", fontsize=8, color=MUTED)
    ax.text(5.5, 0.85, subtitle, ha="center", fontsize=9.5, color=MUTED)
    save(fig, out / filename)


def process_group_mesh(out: Path) -> None:
    _rank_grid(
        out,
        "process_group_mesh.svg",
        "Megatron carves DP / TP / PP groups from one rank mesh",
        "TP varies fastest inside a stage; PP walks layers; DP links identical shards.",
    )


def dp_tp_pp_ranks(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    clean(ax, (0, 11), (0, 6.4))
    ax.set_title("The same ranks sliced three ways")
    sections = [
        ("TP groups", BLUE, BLUE_FILL, ["0,1", "4,5", "8,9", "12,13"]),
        ("PP groups", PURPLE, PURPLE_FILL, ["0,4,8,12", "1,5,9,13", "2,6,10,14", "3,7,11,15"]),
        ("DP groups", GREEN, GREEN_FILL, ["0,2", "1,3", "4,6", "5,7"]),
    ]
    for i, (title, edge, fill, groups) in enumerate(sections):
        x0 = 0.75 + i * 3.45
        ax.text(x0 + 1.25, 5.0, title, ha="center", color=edge, fontweight="bold")
        for j, group in enumerate(groups):
            box(ax, (x0 + (j % 2) * 1.35, 4.0 - (j // 2) * 0.75), 1.15, 0.5, group, fc=fill, ec=edge, fontsize=8)
    box(ax, (1.6, 1.0), 7.8, 0.75, "Every rank belongs to one group on each axis; collectives use those subgroup handles.", fc=YELLOW_FILL, ec=ORANGE)
    save(fig, out / "dp_tp_pp_ranks.svg")


def init_flow(out: Path) -> None:
    _series_flow(
        out,
        "init_flow.svg",
        "Distributed initialization builds the communication control plane",
        [
            ("launcher\nenv vars", BLUE_FILL, BLUE),
            ("set CUDA\ndevice", GREEN_FILL, GREEN),
            ("init global\nprocess group", PURPLE_FILL, PURPLE),
            ("create TP\nPP DP groups", ORANGE_FILL, ORANGE),
            ("seed RNG\ntrackers", YELLOW_FILL, ORANGE),
            ("attach ZeRO-R\nhooks", RED_FILL, RED),
        ],
    )


def column_parallel_linear(out: Path) -> None:
    _series_flow(
        out,
        "column_parallel_linear.svg",
        "ColumnParallelLinear splits output features",
        [
            ("X\nfull H", FILL, INK),
            ("W0\nH x 4H/p", BLUE_FILL, BLUE),
            ("W1\nH x 4H/p", GREEN_FILL, GREEN),
            ("Y shards\nstay split", BLUE_FILL, BLUE),
            ("optional\ngather", PURPLE_FILL, PURPLE),
            ("backward\nreduce dX", ORANGE_FILL, ORANGE),
        ],
    )


def row_parallel_linear(out: Path) -> None:
    _series_flow(
        out,
        "row_parallel_linear.svg",
        "RowParallelLinear splits input features and sums partial outputs",
        [
            ("X shards\nH/p", BLUE_FILL, BLUE),
            ("W shards\nH/p x H", GREEN_FILL, GREEN),
            ("partial Y0", BLUE_FILL, BLUE),
            ("partial Y1", GREEN_FILL, GREEN),
            ("all-reduce\nsum", PURPLE_FILL, PURPLE),
            ("Y\nfull H", FILL, INK),
        ],
    )


def parallel_attention_block(out: Path) -> None:
    _series_flow(
        out,
        "parallel_attention_block.svg",
        "Parallel self-attention: local heads, synchronized output projection",
        [
            ("hidden\ninput", FILL, INK),
            ("QKV\ncolumn TP", BLUE_FILL, BLUE),
            ("local\nheads", GREEN_FILL, GREEN),
            ("attention\nper shard", GREEN_FILL, GREEN),
            ("output\nrow TP", PURPLE_FILL, PURPLE),
            ("all-reduce\nresidual", ORANGE_FILL, ORANGE),
        ],
    )


def vocab_parallel_embedding(out: Path) -> None:
    _series_flow(
        out,
        "vocab_parallel_embedding.svg",
        "VocabParallelEmbedding shards rows by token id range",
        [
            ("token ids", FILL, INK),
            ("rank 0\nids 0..V/p", BLUE_FILL, BLUE),
            ("rank 1\nids V/p..", GREEN_FILL, GREEN),
            ("local lookup\nelse zero", ORANGE_FILL, ORANGE),
            ("all-reduce\nsum", PURPLE_FILL, PURPLE),
            ("embedding", FILL, INK),
        ],
    )


def parallel_cross_entropy(out: Path) -> None:
    _series_flow(
        out,
        "parallel_cross_entropy.svg",
        "Vocab-parallel cross entropy reduces scalars instead of gathering logits",
        [
            ("logit\nshards", BLUE_FILL, BLUE),
            ("local max", GREEN_FILL, GREEN),
            ("global max", PURPLE_FILL, PURPLE),
            ("local exp\nsum", GREEN_FILL, GREEN),
            ("global sum", PURPLE_FILL, PURPLE),
            ("target logit\nand loss", ORANGE_FILL, ORANGE),
        ],
    )


def precision_memory_table(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    clean(ax, (0, 11), (0, 6.5))
    ax.set_title("Who occupies memory in mixed precision training")
    rows = [
        ("fp16/bf16 params", "2 B", "fast forward/backward", BLUE_FILL, BLUE),
        ("fp16/bf16 grads", "2 B", "temporary gradient storage", GREEN_FILL, GREEN),
        ("fp32 master params", "4 B", "stable optimizer update", PURPLE_FILL, PURPLE),
        ("fp32 Adam m/v", "8 B", "optimizer state", ORANGE_FILL, ORANGE),
        ("activations", "shape dependent", "often dominant", RED_FILL, RED),
    ]
    for i, (name, size, purpose, fill, edge) in enumerate(rows):
        y = 4.7 - i * 0.75
        box(ax, (0.8, y), 2.4, 0.5, name, fc=fill, ec=edge, fontsize=8)
        box(ax, (3.6, y), 1.25, 0.5, size, fc=fill, ec=edge, fontsize=8)
        box(ax, (5.25, y), 4.5, 0.5, purpose, fc=fill, ec=edge, fontsize=8)
    save(fig, out / "precision_memory_table.svg")


def amp_flow(out: Path) -> None:
    _series_flow(
        out,
        "amp_flow.svg",
        "Mixed precision flow with fp32 master weights",
        [
            ("master\nfp32 W", PURPLE_FILL, PURPLE),
            ("cast to\nmodel fp16", BLUE_FILL, BLUE),
            ("forward\nloss", GREEN_FILL, GREEN),
            ("scale\nloss", YELLOW_FILL, ORANGE),
            ("backward\nfp16 grad", BLUE_FILL, BLUE),
            ("unscale\nfp32 grad", ORANGE_FILL, ORANGE),
            ("optimizer\nstep", PURPLE_FILL, PURPLE),
        ],
    )


def dynamic_loss_scale(out: Path) -> None:
    _series_flow(
        out,
        "dynamic_loss_scale.svg",
        "Dynamic loss scaling raises scale on good steps and backs off on overflow",
        [
            ("current\nscale S", YELLOW_FILL, ORANGE),
            ("scaled\nbackward", BLUE_FILL, BLUE),
            ("finite grad\ncheck", FILL, INK),
            ("yes:\ncount step", GREEN_FILL, GREEN),
            ("interval:\nS grows", GREEN_FILL, GREEN),
            ("overflow:\nskip + shrink", RED_FILL, RED),
        ],
    )


def grad_clip_with_mp(out: Path) -> None:
    _series_flow(
        out,
        "grad_clip_with_mp.svg",
        "Gradient clipping uses a distributed norm over unscaled fp32 grads",
        [
            ("fp16 grads", BLUE_FILL, BLUE),
            ("copy to\nfp32", PURPLE_FILL, PURPLE),
            ("unscale", ORANGE_FILL, ORANGE),
            ("local\nnorm^2", GREEN_FILL, GREEN),
            ("all-reduce\nnorm", PURPLE_FILL, PURPLE),
            ("clip local\ngrads", RED_FILL, RED),
        ],
    )


def gshard_moe_layer(out: Path) -> None:
    _series_flow(
        out,
        "gshard_moe_layer.svg",
        "GShard MoE replaces dense FFN with routed experts",
        [
            ("tokens\nafter attn", FILL, INK),
            ("gate\nsoftmax", YELLOW_FILL, ORANGE),
            ("top-k\nexperts", ORANGE_FILL, ORANGE),
            ("expert\nFFNs", BLUE_FILL, BLUE),
            ("weighted\ncombine", PURPLE_FILL, PURPLE),
            ("output", FILL, INK),
        ],
    )


def gate_top2_capacity(out: Path) -> None:
    _series_flow(
        out,
        "gate_top2_capacity.svg",
        "Top-2 routing is constrained by expert capacity",
        [
            ("gate\nscores", YELLOW_FILL, ORANGE),
            ("1st\nexpert", BLUE_FILL, BLUE),
            ("2nd\nexpert", GREEN_FILL, GREEN),
            ("capacity\nbuffer", PURPLE_FILL, PURPLE),
            ("overflow:\ndrop/residual", RED_FILL, RED),
            ("aux loss\nbalances", ORANGE_FILL, ORANGE),
        ],
    )


def ep_dp_layout(out: Path) -> None:
    _rank_grid(
        out,
        "ep_dp_layout.svg",
        "Expert parallel groups route tokens; data parallel groups sync matching experts",
        "EP moves activations across different experts; DP reduces gradients for replicated expert shards.",
    )


def all_to_all_dispatch(out: Path) -> None:
    _series_flow(
        out,
        "all_to_all_dispatch.svg",
        "All-to-All dispatch sends tokens to expert-owning ranks and returns outputs",
        [
            ("source\nrank tokens", FILL, INK),
            ("bucket by\ndestination", ORANGE_FILL, ORANGE),
            ("dispatch\nA2A", PURPLE_FILL, PURPLE),
            ("local\nexperts", BLUE_FILL, BLUE),
            ("combine\nA2A", PURPLE_FILL, PURPLE),
            ("restore\ntoken order", GREEN_FILL, GREEN),
        ],
    )


def ep_dp_tp(out: Path) -> None:
    _series_flow(
        out,
        "ep_dp_tp.svg",
        "EP, DP, TP, and PP are orthogonal axes around the MoE layer",
        [
            ("EP:\nwhich expert", ORANGE_FILL, ORANGE),
            ("TP:\nsplit expert GEMM", BLUE_FILL, BLUE),
            ("DP:\nreplicate layout", GREEN_FILL, GREEN),
            ("PP:\nsplit layers", PURPLE_FILL, PURPLE),
            ("MoE layer\ncomposition", FILL, INK),
        ],
    )


def moe_init_flow(out: Path) -> None:
    _series_flow(
        out,
        "moe_init_flow.svg",
        "DeepSpeed-Megatron attaches EP groups during model wrapping",
        [
            ("Megatron\nTP/PP/DP init", BLUE_FILL, BLUE),
            ("build model\nwith MoE", GREEN_FILL, GREEN),
            ("deepspeed\ninitialize", PURPLE_FILL, PURPLE),
            ("walk modules", ORANGE_FILL, ORANGE),
            ("set DS\nparallelism", ORANGE_FILL, ORANGE),
            ("create EP\nand EDP groups", RED_FILL, RED),
        ],
    )


def moe_layer_structure(out: Path) -> None:
    _series_flow(
        out,
        "moe_layer_structure.svg",
        "DeepSpeed MoELayer contains gate, dispatch, local experts, and combine",
        [
            ("hidden", FILL, INK),
            ("TopKGate", YELLOW_FILL, ORANGE),
            ("dispatch\nmask", ORANGE_FILL, ORANGE),
            ("A2A\ndispatch", PURPLE_FILL, PURPLE),
            ("ParallelMLP\nexperts", BLUE_FILL, BLUE),
            ("A2A\ncombine", PURPLE_FILL, PURPLE),
            ("output", FILL, INK),
        ],
    )


def ep_group_vs_dp(out: Path) -> None:
    _series_flow(
        out,
        "ep_group_vs_dp.svg",
        "EP groups and expert-DP groups move different tensors",
        [
            ("EP group:\ndifferent experts", ORANGE_FILL, ORANGE),
            ("moves\nactivations", ORANGE_FILL, ORANGE),
            ("expert-DP:\nsame expert shard", GREEN_FILL, GREEN),
            ("moves\ngradients", GREEN_FILL, GREEN),
            ("same rank\njoins both", PURPLE_FILL, PURPLE),
        ],
    )


FIGURES: dict[str, Callable[[Path], None]] = {
    "naive_model_parallel.svg": naive_model_parallel,
    "gpipe_microbatch.svg": gpipe_microbatch,
    "rematerialization.svg": rematerialization,
    "bubble_vs_m.svg": bubble_vs_m,
    "parameter_server.svg": parameter_server,
    "async_sgd_staleness.svg": async_sgd_staleness,
    "ring_allreduce_reduce_scatter.svg": ring_allreduce_reduce_scatter,
    "ring_allreduce_allgather.svg": ring_allreduce_allgather,
    "ring_bandwidth.svg": ring_bandwidth,
    "memory_breakdown.svg": memory_breakdown,
    "mixed_precision_memory.svg": mixed_precision_memory,
    "zero_stages.svg": zero_stages,
    "zero_vs_model_parallel.svg": zero_vs_model_parallel,
    "zero_offload.svg": zero_offload,
    "row_vs_column_split.svg": row_vs_column_split,
    "mlp_tp.svg": mlp_tp,
    "attention_tp.svg": attention_tp,
    "embedding_vocab_parallel.svg": embedding_vocab_parallel,
    "tp_dp_hybrid.svg": tp_dp_hybrid,
    "series_map.svg": series_map,
    "process_group_mesh.svg": process_group_mesh,
    "dp_tp_pp_ranks.svg": dp_tp_pp_ranks,
    "init_flow.svg": init_flow,
    "column_parallel_linear.svg": column_parallel_linear,
    "row_parallel_linear.svg": row_parallel_linear,
    "parallel_attention_block.svg": parallel_attention_block,
    "vocab_parallel_embedding.svg": vocab_parallel_embedding,
    "parallel_cross_entropy.svg": parallel_cross_entropy,
    "precision_memory_table.svg": precision_memory_table,
    "amp_flow.svg": amp_flow,
    "dynamic_loss_scale.svg": dynamic_loss_scale,
    "grad_clip_with_mp.svg": grad_clip_with_mp,
    "gshard_moe_layer.svg": gshard_moe_layer,
    "gate_top2_capacity.svg": gate_top2_capacity,
    "ep_dp_layout.svg": ep_dp_layout,
    "all_to_all_dispatch.svg": all_to_all_dispatch,
    "ep_dp_tp.svg": ep_dp_tp,
    "moe_init_flow.svg": moe_init_flow,
    "moe_layer_structure.svg": moe_layer_structure,
    "ep_group_vs_dp.svg": ep_group_vs_dp,
    "tp_activation_hotspots.svg": tp_activation_hotspots,
    "megatron_sp_layernorm.svg": megatron_sp_layernorm,
    "tp_sp_mlp.svg": tp_sp_mlp,
    "selective_recompute.svg": selective_recompute,
    "ulysses_a2a.svg": ulysses_a2a,
    "megatron_vs_ulysses_comm.svg": megatron_vs_ulysses_comm,
    "ulysses_zero3.svg": ulysses_zero3,
    "online_softmax_blocks.svg": online_softmax_blocks,
    "ring_kv_passing.svg": ring_kv_passing,
    "chunk_size_tradeoff.svg": chunk_size_tradeoff,
    "cp_init_groups.svg": cp_init_groups,
    "naive_vs_balanced_ring.svg": naive_vs_balanced_ring,
    "cp_comm_overlap.svg": cp_comm_overlap,
    "naive_ag_vs_overlap.svg": naive_ag_vs_overlap,
    "p2p_ag_overlap.svg": p2p_ag_overlap,
    "rs_overlap_p2p.svg": rs_overlap_p2p,
    "bulk_ag_rs.svg": bulk_ag_rs,
    "marketing_inter_layer.svg": marketing_inter_layer,
    "actual_intra_layer.svg": actual_intra_layer,
    "zero3_collectives.svg": zero3_collectives,
}


def generate_all(root: Path) -> None:
    for post in DEFAULT_POSTS:
        filenames = POSTS[post]
        out = root / post
        out.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            FIGURES[filename](out)


def generate_target(root: Path, target: str) -> None:
    if target == "all":
        generate_all(root)
        return

    if target in POSTS:
        out = root / target
        out.mkdir(parents=True, exist_ok=True)
        for filename in POSTS[target]:
            FIGURES[filename](out)
        return

    for post, filenames in POSTS.items():
        if target in filenames:
            out = root / post
            out.mkdir(parents=True, exist_ok=True)
            FIGURES[target](out)
            return

    raise ValueError(f"unknown figure target: {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("content/posts"))
    parser.add_argument("--post", choices=["all", *POSTS.keys()])
    parser.add_argument("targets", nargs="*", choices=["all", *POSTS.keys(), *FIGURES.keys()])
    args = parser.parse_args()
    if args.post and args.targets:
        parser.error("use either --post or positional targets, not both")
    targets = [args.post] if args.post else (args.targets or ["all"])
    for target in targets:
        generate_target(args.out, target)


if __name__ == "__main__":
    main()