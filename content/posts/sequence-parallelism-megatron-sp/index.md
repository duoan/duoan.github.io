---
title: "Sequence Parallelism I: Megatron SP Cuts Activation Memory Along the Sequence"
date: 2025-05-12
tags: ["LLM", "Training", "Parallelism", "Megatron", "Sequence Parallelism"]
categories: ["Engineering"]
draft: false
series: ["LLM Training"]
cover:
  image: tp_activation_hotspots.svg
  alt: "Tensor parallelism leaves LayerNorm and Dropout activations replicated while Megatron SP shards them along sequence"
  relative: true
---

# Sequence Parallelism I: Megatron SP Cuts Activation Memory Along the Sequence

Tensor parallelism is usually introduced as a way to make matrix multiplications fit.
That is true, but it hides a second problem.
After the weights are split, many activations are still replicated on every tensor-parallel rank.
For short contexts this is tolerable.
For long contexts it becomes one of the reasons training throughput collapses into activation checkpointing.

Megatron sequence parallelism, usually shortened to Megatron SP, is a targeted fix.
It does not replace tensor parallelism.
It keeps Megatron's column-parallel and row-parallel linear layers, then shards the sequence-local regions that tensor parallelism had left replicated.
The trick is small enough to miss and important enough to change the memory budget of a whole Transformer block.

This post is the first piece in a four-part sequence-parallelism set.
It connects directly to [Megatron tensor parallelism](../tensor-parallelism-megatron/) and sets up [DeepSpeed Ulysses](../sequence-parallelism-ulysses/), [Ring Attention](../ring-attention/), and [Megatron Context Parallel](../megatron-context-parallel/).

## TL;DR

- Megatron tensor parallelism shards the heavy GEMMs in attention and MLP, but LayerNorm inputs, LayerNorm outputs, residual paths, and dropout masks can remain replicated.
- Megatron SP shards those sequence-local activations along the sequence dimension: each rank keeps `[batch, sequence / tp_size, hidden]`.
- Before a tensor-parallel region needs the full sequence, SP issues an **AllGather**.
- After a row-parallel region produces an output that can return to sequence shards, SP issues a **ReduceScatter**.
- The communication pattern is comparable to the original TP collectives, but the saved activations are much smaller per rank.
- SP is most useful when activation memory, not parameter memory, is what forces aggressive recomputation.
- Selective activation recomputation remains useful: save the expensive-to-recompute activations and replay the cheap, bulky attention-score state.
- Reproducible figures for this post: [`playground/llm_training_series_figures.py`](https://github.com/duoan/duoan.github.io/blob/main/playground/llm_training_series_figures.py).

## 1. What Tensor Parallelism Did Not Fix

Start with the Transformer block under Megatron tensor parallelism.
The MLP uses a column-parallel first projection and a row-parallel second projection.
Attention does the analogous split across QKV heads and the output projection.
That is the right place to spend model-parallel effort because the GEMMs dominate compute.

But activation memory is not only GEMM intermediates.
Backward needs inputs to LayerNorm, inputs to linear layers, some nonlinear intermediates, residual values, and dropout masks.
Some of those tensors naturally become sharded because the computation that created them is sharded.
Others are sequence-local elementwise operations that every tensor-parallel rank sees in full.

![Tensor parallelism shards GEMMs but leaves normalization and dropout replicated](tp_activation_hotspots.svg)

The red regions are the uncomfortable ones.
They are not expensive enough to justify tensor-parallel math, but they can be large enough to dominate saved activation memory.
The shape is usually proportional to `batch * sequence * hidden`.
If all TP ranks save the same tensor, increasing TP degree does not reduce that part of the activation footprint.

This explains why tensor parallelism alone does not make recomputation disappear.
It reduces parameter and GEMM intermediate pressure.
It does not automatically reduce every activation needed by the backward pass.

## 2. The Observation Behind Megatron SP

LayerNorm, dropout, residual addition, and many pointwise operations do not mix tokens.
For those regions, token `i` does not need token `j`.
That means the sequence dimension is a valid partition axis.

Megatron SP uses the tensor-parallel group as the sequence-parallel group.
If the TP size is `t`, each rank keeps roughly one `1/t` slice of the sequence for the SP regions.
The resident activation shape changes from:

```text
[batch, sequence, hidden]
```

to:

```text
[batch, sequence / t, hidden]
```

The important phrase is "for the SP regions."
Tensor-parallel GEMMs still have their own preferred layouts.
SP therefore has to gather before those regions and scatter after them.

![Megatron SP shards sequence-local activations and gathers only for TP regions](megatron_sp_layernorm.svg)

In the figure, LayerNorm is local because it normalizes over hidden features for each token.
Each rank can normalize its own sequence slice.
When the next column-parallel linear needs the activation layout expected by Megatron TP, the ranks AllGather the sequence shards.
After the row-parallel output has been reduced, ReduceScatter sends each rank back to a sequence shard.

That is the core mechanism.
SP is not a new attention algorithm.
It is not Ulysses-style head reassignment.
It is an activation layout discipline around the existing Megatron TP block.

## 3. MLP: Where the AG and RS Land

The MLP is the cleanest place to see the pattern.
Megatron's MLP has two large linear layers:

1. `FC1`: column-parallel, producing the expanded hidden dimension.
2. `FC2`: row-parallel, reducing back to the model hidden dimension.

Without SP, the activation entering the MLP is replicated across TP ranks.
With SP, the rank starts with only its sequence slice.
Before `FC1`, it AllGathers the sequence dimension so each tensor-parallel rank sees the required input.
After `FC2`, it ReduceScatters so the post-MLP output and following dropout/residual state return to sequence-sharded form.

![TP + SP replaces replicated activations with AG/RS around tensor-parallel MLP](tp_sp_mlp.svg)

This is why Megatron SP is often described as replacing AllReduce with AllGather plus ReduceScatter.
For a ring implementation, the communication volume is in the same class.
The behavioral difference is the saved state between collectives.
The model no longer keeps the redundant LayerNorm and dropout-adjacent activations on every rank.

There is one subtle point worth keeping straight.
SP does not mean every tensor in the block is split along sequence all the time.
The layout changes at boundaries.
The design goal is not "sequence shards everywhere."
The design goal is "sequence shards wherever the operation permits it, with collectives at the minimal places that restore TP correctness."

## 4. Attention: Same Pattern, More Expensive Middle

Attention follows the same high-level flow.
LayerNorm and residual-adjacent activations can be sequence-sharded.
QKV projection and output projection still use Megatron tensor parallelism.
The attention middle is heavier because score tensors scale with `sequence^2`.

This distinction matters for memory accounting.
The `batch * sequence * hidden` tensors are exactly what SP divides by the TP/SP degree.
The score and softmax-related state has a different shape and often a different recomputation policy.
Modern kernels such as FlashAttention already avoid materializing the full score matrix in the naive way.
Distributed long-context variants go further, as discussed in [Ring Attention](../ring-attention/).

For Megatron SP itself, the conceptual answer remains simple.
Gather before the TP computation that needs the full sequence layout.
Scatter after the row-parallel result can return to a local sequence slice.

## 5. Why This Helps More Than It Looks

Activation memory is paid per layer and per microbatch.
A small replicated tensor in one diagram becomes a large persistent footprint across dozens or hundreds of layers.
When context length grows, that footprint grows linearly with sequence length.
When attention score state is materialized, other pieces grow quadratically.

SP attacks the linear replicated portion.
That sounds modest until you compare it with full activation recomputation.
Full recomputation saves memory by refusing to save most intermediates, then replaying forward work during backward.
It is robust but expensive.
SP saves memory by changing ownership of activations that were already cheap to compute locally.
That makes it a better first move when the model is close to fitting.

The trade is not free.
AllGather and ReduceScatter must appear on the critical path unless overlapped.
The next post in this thread on [tensor-parallel collective overlap](../megatron-tp-comm-overlap/) is about that exact pressure.

## 6. Selective Activation Recomputation

Even after SP, a long-context model may not fit if it saves every activation.
The right answer is usually not "save everything" or "recompute everything."
The right answer is selective recomputation.

![Selective recomputation keeps expensive activations and replays cheap, bulky ones](selective_recompute.svg)

The policy is guided by two questions:

1. How much memory does this activation consume?
2. How expensive is it to recreate when backward reaches it?

Attention score and softmax state is often a good recomputation candidate.
It can be large, especially as sequence length grows, but it may be cheaper to replay than a full stack of GEMMs.
Linear inputs and other GEMM-adjacent activations are often worth saving because replaying the associated matrix multiplication is expensive.

This is the practical framing:

- TP reduces the memory and compute burden of large linear layers by splitting weights and GEMMs.
- SP reduces the replicated activation burden around sequence-local operations.
- Selective recomputation handles the remaining high-memory intermediates whose replay cost is acceptable.

Together, they turn a blunt memory trade into a set of localized choices.

## 7. How to Recognize Megatron SP in a Trace

In a profiler, Megatron SP should show collectives around the tensor-parallel linear regions.
You should expect AllGather before column-parallel work and ReduceScatter after row-parallel work.
You should also expect the activation shapes retained between those regions to be sequence-sharded.

If you see full `[B, S, H]` LayerNorm or dropout-adjacent activations retained on every TP rank, SP is not buying what you think it is buying.
If you see AG/RS collectives but no memory relief, check whether the framework is saving a full tensor for a downstream residual, a debug hook, or a fused kernel boundary.
SP is an ownership contract; one stray full-size save can erase much of the benefit.

## 8. Where SP Fits in the Larger Map

Megatron SP is the conservative sequence-parallelism design.
It assumes you are already using Megatron tensor parallelism and want activation memory relief without changing the semantics of attention.
It is therefore very different from Ulysses and Ring Attention.

[DeepSpeed Ulysses](../sequence-parallelism-ulysses/) sequence-shards first, then uses All-to-All to turn sequence shards into head shards for attention.
[Ring Attention](../ring-attention/) keeps local query blocks and circulates key/value blocks so contexts can exceed single-device memory.
[Megatron Context Parallel](../megatron-context-parallel/) brings a ring-attention-like idea back into the Megatron ecosystem and adds load balancing for causal masks.

All four methods cut along the sequence dimension somewhere.
They do it for different bottlenecks.
Megatron SP's bottleneck is replicated activation memory around a TP block.

## References

- Korthikanti et al., [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198), 2022.
- Narayanan et al., [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473), 2021.
- NVIDIA Megatron-LM and Megatron Core documentation for tensor parallel and sequence parallel training.
