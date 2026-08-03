---
title: "Sequence Parallelism IV: Megatron Context Parallel and Load-Balanced Rings"
date: 2025-06-02
tags: ["LLM", "Training", "Parallelism", "Megatron", "Context Parallelism"]
categories: ["Engineering"]
draft: false
series: ["LLM Training"]
cover:
  image: naive_vs_balanced_ring.svg
  alt: "Megatron Context Parallel balances causal ring attention by pairing early and late sequence chunks"
  relative: true
---

# Sequence Parallelism IV: Megatron Context Parallel and Load-Balanced Rings

The misconception is that Megatron Context Parallel is just Megatron SP with a larger name.
It is not.
SP shards token-local activations around tensor-parallel linears; CP shards the context for attention itself and exchanges K/V across a context-parallel group.

Megatron Core describes context parallelism as sequence-length parallelism for network inputs and activations, with attention requiring extra K/V communication across GPUs ([Megatron Core CP docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)).
The production detail is load balance.
A naive ring is imbalanced for causal attention, so CP pairs early and late chunks to smooth useful work across ranks.

This post closes the sequence-parallelism set.
It builds on [Megatron SP](../sequence-parallelism-megatron-sp/), [DeepSpeed Ulysses](../sequence-parallelism-ulysses/), and [Ring Attention](../ring-attention/), and it connects back to [Megatron tensor parallelism](../tensor-parallelism-megatron/) because CP lives inside Megatron's process-group structure.

## TL;DR

- Context parallelism shards sequence length across a CP group.
- A CP group is formed from ranks with matching tensor, pipeline, and data-parallel coordinates but different context coordinates.
- Attention inside a CP group resembles Ring Attention: local Q chunks stay put while K/V chunks move.
- Causal masks make a contiguous naive ring imbalanced because early query chunks have less valid work than late query chunks.
- Megatron CP uses balanced or zigzag chunk placement, often pairing early and late sequence slices on the same rank.
- CP can use P2P ring exchange, AllGather, All-to-All, or hierarchical A2A+P2P communication depending on configuration.
- The performance target is the same as Ring Attention: hide K/V exchange under useful attention compute.
- Reproducible figures for this post: [`playground/llm_training_series_figures.py`](https://github.com/duoan/duoan.github.io/blob/main/playground/llm_training_series_figures.py).

## 1. Why Megatron Needed a Context Axis

Megatron SP is useful, but it is not full context parallelism.
SP reduces activation memory around tensor-parallel regions by sharding token-local activations.
It does not make a single attention head over a very long sequence fit by itself.

Ring Attention solves the long-context attention residency problem more directly.
But a production Megatron training stack already has tensor parallelism, pipeline parallelism, data parallelism, activation recomputation, mixed precision, and process groups chosen for topology.
CP is the integration point.

It says:

```text
keep Megatron's hybrid-parallel structure,
add a context dimension,
run distributed attention inside that dimension.
```

The statement is simple.
The group layout, causal masks, and overlap schedule are where the engineering lives.

## 2. Process Groups

Assume a training job with tensor parallel size `tp`, context parallel size `cp`, data parallel size `dp`, and pipeline parallel size `pp`.
The world size is:

```text
tp * cp * dp * pp
```

For a CP group, the useful rule is:

```text
same TP coordinate
same DP coordinate
same PP coordinate
different context coordinate
```

Those ranks own compatible model shards and pipeline stages for the same data replica, but different sequence chunks.

![Context parallel groups are formed at matching TP/DP/PP coordinates](cp_init_groups.svg)

This is why CP is more specific than "run Ring Attention across any ranks."
The attention result has to line up with tensor-parallel head ownership, pipeline-stage ownership, and data-parallel synchronization.
If the CP group is wrong, shapes may line up while semantics are wrong.

Topology matters too.
Tensor-parallel and context-parallel communication is frequent and bandwidth-hungry.
Pipeline and data-parallel groups can tolerate different tradeoffs depending on batch size and schedule.

## 3. What Moves in CP

As in Ring Attention, Q stays logically local.
K/V must be visible to query chunks that need it.
Megatron Core documentation explains CP as storing only local sequence chunks while attention communicates K/V across the CP group, with communication implemented by collectives or P2P ring exchange under the hood ([CP docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)).

A rank updates its local output using arriving K/V chunks.
Online softmax state keeps the result exact across block updates.
The local output does not need to become a full-sequence tensor during forward.

That is the same residency idea as [Ring Attention](../ring-attention/).
The difference is integration.
CP is built to coexist with Megatron's tensor, pipeline, and data-parallel axes.

## 4. Naive Ring Attention Is Imbalanced Under Causal Masks

In bidirectional attention, every query block attends to every key block.
If each rank owns one contiguous query block, useful work is roughly even.

Causal language-model attention is triangular.
Early query blocks attend to fewer key blocks.
Late query blocks attend to many key blocks.
If rank `0` owns the earliest tokens and rank `3` owns the latest tokens, rank `3` has much more useful work.

In a ring, ranks move in lockstep.
If one rank has little work and another rank has a lot of work, the idle rank still waits.
Average work does not determine step time.
Balanced per-iteration work does.

![Causal masks make naive rings imbalanced; zigzag chunks equalize work](naive_vs_balanced_ring.svg)

A common fix is to split the sequence into `2 * cp_size` chunks and pair early and late chunks.
For `cp_size = 4`, the assignment can look like:

```text
rank 0: chunks 0 and 7
rank 1: chunks 1 and 6
rank 2: chunks 2 and 5
rank 3: chunks 3 and 4
```

Each rank receives one early chunk and one late chunk.
The causal work is not perfectly uniform at every micro-detail, but the gross imbalance is much smaller.

## 5. Load Balance Is Part of the Algorithm

The ring schedule alone is not enough.
The chunk assignment must respect the mask.
That is why Megatron CP documentation calls out performance improvements from avoiding unnecessary lower-triangle causal-mask work and keeping load balanced across GPUs ([Megatron Core CP docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)).

Balanced placement adds bookkeeping.
A rank may own two non-contiguous chunks.
The attention mask must know which local query positions and remote key positions are valid.
Some sub-blocks can be skipped because the causal mask makes them future tokens.

The implementation goal is to skip useless work without creating rank-level imbalance.
Chunk layout and mask logic are therefore inseparable.

## 6. Communication Modes

Megatron's configuration exposes CP communication choices.
The transformer config lists `cp_comm_type` options such as `p2p`, `all_gather`, `a2a`, and `a2a+p2p` in current code.
The modes represent different tradeoffs.

- `p2p` exchanges K/V chunks in a ring and can overlap communication with attention compute.
- `all_gather` assembles full K/V before attention and is easier to reason about but harder to hide.
- `a2a` behaves more like Ulysses by scattering attention heads across the CP group.
- `a2a+p2p` is a hierarchical design, often useful when fast intra-node links and slower inter-node links should be treated differently.

The right mode depends on model shape, topology, and kernel support.
The ownership story remains the same: context is sharded, attention needs remote K/V, and gradients must return to owners.

## 7. Overlap: More Than One Stream

The ring only performs well if communication overlaps with compute.
For each iteration, a rank wants to:

1. compute attention using the current K/V block,
2. send a K/V block to a neighbor,
3. receive the next K/V block from a neighbor,
4. update online softmax state for partial outputs.

![Megatron CP overlaps KV exchange, attention, and softmax state updates](cp_comm_overlap.svg)

The exact stream choreography is implementation-specific.
The principle is stable.
K/V transfer should not sit on the same critical path as the attention GEMM if the hardware can run them concurrently.
Softmax correction should also be arranged so it does not serialize every ring step.

This is similar in spirit to [Hiding Tensor-Parallel Collectives](../megatron-tp-comm-overlap/).
The tensors differ, but the performance question is the same: which dependencies are real, and which are artifacts of a naive schedule?

## 8. CP vs SP

Megatron SP and Megatron CP both shard sequence, but they do not solve the same bottleneck.

SP is an activation-memory optimization around tensor-parallel regions.
It shards token-local saved activations and uses AllGather/ReduceScatter at TP boundaries.
It does not distribute the attention context for one head.

CP is an attention-context optimization.
It shards the sequence for attention itself and communicates K/V so long contexts can be processed without one rank holding the full K/V context.

In practice, they can coexist.
SP can reduce memory around MLP and residual regions.
CP can handle the long-context attention middle.
The important thing is to distinguish which collectives belong to which ownership transition.

## 9. CP vs Ulysses

Ulysses performs an All-to-All transpose from sequence ownership to head ownership.
After the transpose, a rank has all tokens for a head slice.

CP keeps context ownership and streams or gathers K/V blocks for attention.
A rank does not need to become the owner of all tokens for a head at once when using the ring-style path.

That difference changes scaling.
Ulysses is naturally bounded by head count unless extended.
CP is tied to context chunking, K/V exchange, and mask-aware load balance.
Both approaches can appear in hybrid designs, but they pressure different parts of the network.

## 10. Debugging Mental Model

When CP is enabled, ask four questions.

First, are CP groups formed from ranks with compatible TP, DP, and PP coordinates?
If not, tensor shapes may line up while ownership is wrong.

Second, are sequence chunks assigned contiguously or in a balanced pattern?
For causal language models, contiguous assignment should raise suspicion.

Third, does the trace show K/V exchange in the attention region rather than an unrelated global collective?
CP should look like context movement, not parameter movement.

Fourth, is communication overlapped with attention compute?
If every ring step waits for receive, computes, then sends, the algorithm may be correct but slow.

These questions catch many mistakes before diving into kernel details.

## Code

- Megatron Core context-parallel docs: [`docs/user-guide/features/context_parallel.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/context_parallel.md).
- Megatron transformer config for CP communication modes: [`megatron/core/transformer/transformer_config.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_config.py), especially `cp_comm_type`.
- Megatron-LM repository for current CP implementation and configuration: [`NVIDIA/Megatron-LM`](https://github.com/NVIDIA/Megatron-LM).

## References

- NVIDIA, [Megatron Core Context Parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html).
- Liu, Zaharia, Abbeel, [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889), 2023.
- Korthikanti et al., [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198), 2022.
