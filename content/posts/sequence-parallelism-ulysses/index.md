---
title: "Sequence Parallelism II: DeepSpeed Ulysses and All-to-All Attention"
date: 2025-05-19
tags: ["LLM", "Training", "Parallelism", "DeepSpeed", "Ulysses"]
categories: ["Engineering"]
draft: false
series: ["LLM Training"]
cover:
  image: ulysses_a2a.svg
  alt: "Ulysses sequence-shards activations and uses All-to-All to make each rank own all tokens for one attention head"
  relative: true
---

# Sequence Parallelism II: DeepSpeed Ulysses and All-to-All Attention

Megatron SP is a careful memory optimization around an existing tensor-parallel block.
DeepSpeed Ulysses starts from a different question.
What if each device owns a sequence slice most of the time, but attention temporarily wants each device to own a head slice instead?

The answer is an All-to-All transpose.
Before attention, every rank has all heads for a subset of tokens.
After All-to-All, every rank has all tokens for a subset of heads.
That one layout change lets local attention run per head while the rest of the layer can remain sequence-sharded.

This is the second post in the sequence-parallelism set.
It builds on [Megatron SP](../sequence-parallelism-megatron-sp/) and points toward [Ring Attention](../ring-attention/) and [Megatron Context Parallel](../megatron-context-parallel/).
It also connects to [ZeRO](../zero-redundancy-optimizer/) because Ulysses is often discussed as a natural companion to ZeRO-3.

## TL;DR

- Ulysses partitions activations along the sequence dimension across a sequence-parallel group.
- Each rank computes Q, K, and V for its local sequence slice.
- An **All-to-All** then transposes the layout from sequence shards to head shards.
- Attention runs locally because each rank now has the full sequence for its assigned head or heads.
- A second All-to-All reverses the transpose so the output returns to sequence-sharded layout.
- Compared with Megatron SP, the attractive communication story is that single-rank All-to-All traffic scales roughly with `1 / P`, where `P` is the sequence-parallel degree.
- The catch is that plain Ulysses is bounded by the number of attention heads.
- With ZeRO-3, weights may be gathered just in time while Ulysses handles activation layout.
- Reproducible figures for this post: [`playground/llm_training_series_figures.py`](https://github.com/duoan/duoan.github.io/blob/main/playground/llm_training_series_figures.py).

## 1. The Layout Problem

Consider a single attention layer with sequence length `N`, hidden size `d`, and `P` ranks.
Ulysses begins with a simple sequence partition.
Rank `i` owns `N / P` tokens and the full hidden dimension.
The local activation shape is:

```text
[N / P, d]
```

This is a good layout for LayerNorm, residual operations, and MLP work.
It is not enough for attention.
For one attention head, a query token may attend to keys across the whole sequence.
If a rank only has local sequence positions, it cannot compute full attention for that head without seeing remote K and V.

Ulysses changes the layout exactly at this point.
Each local Q/K/V tensor is split across heads.
Then All-to-All sends the head slice of every local sequence shard to the rank responsible for that head.

![Ulysses converts sequence shards into head shards with All-to-All](ulysses_a2a.svg)

Before the collective, rank `0` might hold:

```text
S0H0, S0H1, S0H2, S0H3
```

After the collective, rank `0` holds:

```text
S0H0, S1H0, S2H0, S3H0
```

That is the whole trick.
It is a distributed transpose from sequence-major ownership to head-major ownership.

## 2. Forward Pass in Slow Motion

The forward pass has five conceptual steps.

First, split the input activation by sequence.
Each rank receives a chunk of tokens and keeps the full hidden dimension.

Second, run the QKV projections for the local sequence chunk.
In plain Ulysses, the rank has the full projection weights.
If ZeRO-3 is also enabled, those weights may be gathered just in time, but the logical projection is still local over the sequence slice.

Third, All-to-All the Q, K, and V tensors.
This changes the ownership from "all heads for my tokens" to "all tokens for my heads."

Fourth, run attention locally.
The rank now has the full sequence for its assigned head slice, so the attention math for those heads is local.

Fifth, All-to-All the attention output back.
The result returns from head-major ownership to sequence-major ownership, and the rest of the block can continue on local sequence chunks.

The important part is that Ulysses avoids a full attention matrix on every rank.
It also avoids asking every rank to compute attention for all heads.
Instead, each rank temporarily becomes the owner of one or more heads across the entire sequence.

## 3. Why All-to-All Is the Right Collective

AllGather would duplicate data.
ReduceScatter would sum data.
Attention needs neither duplication nor summation at this boundary.
It needs a permutation of ownership.

All-to-All is exactly that operation.
Every rank splits its local tensor into `P` pieces.
It keeps the piece addressed to itself and sends the other pieces to their owning ranks.
After the collective, each rank has one piece from every rank.

For Q/K/V, those pieces are head slices.
For the attention output, the inverse pieces are sequence slices.

This is why an All-to-All arrow in a diagram is easy to underestimate.
It is not just "communication happens here."
It is a semantic transpose of the distributed tensor layout.

## 4. Communication Volume Compared with Megatron SP

A simplified comparison is useful.
Ignore batch size and constants, and write the activation size as `N * d`.
In Megatron TP+SP, an attention+MLP block uses AllGather and ReduceScatter around tensor-parallel regions.
A rough per-rank communication count is often summarized as a constant number of payloads shaped like `N * d`.

Ulysses performs All-to-All collectives.
For a single All-to-All, each rank starts with `N * d / P` data and sends all but its own `1 / P` destination slice.
The per-rank send volume is approximately:

```text
N * d / P
```

Across forward and backward, the number of collectives is higher, but each collective is smaller per rank.

![Communication scaling: Megatron SP vs Ulysses](megatron_vs_ulysses_comm.svg)

This is the selling point.
As `N` grows, increasing `P` can keep the per-rank communication from growing at the same rate.
That is especially appealing for long contexts.

There is a catch.
Plain Ulysses assigns head slices to ranks, so `P` is bounded by the number of attention heads and divisibility constraints.
If the model has 32 heads, a pure head-sharded Ulysses group cannot scale to 256 ranks without additional hierarchy or another parallel dimension.

That limitation is one reason long-context systems often combine methods instead of picking a single method forever.

## 5. Ulysses and ZeRO-3

Ulysses is often paired conceptually with ZeRO-3.
The reason is that they optimize different things.
ZeRO-3 partitions model states across data-parallel ranks.
Ulysses partitions activations across sequence-parallel ranks.

During actual compute, ZeRO-3 gathers the parameter needed by the current module.
Ulysses then runs the module over a local sequence shard, with All-to-All only around attention.

![Ulysses can sit inside ZeRO-3: gather weights, sequence-shard compute](ulysses_zero3.svg)

This combination is easiest to reason about if you separate storage ownership from compute layout:

- ZeRO-3 decides which rank stores which parameter shard between compute regions.
- Ulysses decides which rank owns which activation tokens or heads during compute.
- The ZeRO-3 parameter AllGather is about weights.
- The Ulysses All-to-All is about Q/K/V and attention outputs.

Do not collapse these into one mental model.
They are different tensors, different ownership contracts, and different collectives.

For a deeper correction on ZeRO-3 ownership, see [The ZeRO-3 Diagram Most People Remember Is Wrong](../zero3-intra-layer-partitioning/).

## 6. Backward Pass Intuition

Backward reverses the same layout changes.
When gradients reach the attention output, the gradient must be All-to-Alled into the head-major layout used by attention.
When gradients for Q, K, and V are produced, they must be All-to-Alled back to the sequence-major layout.

The weight gradients are a separate issue.
If each rank computed a gradient for a full local projection weight, those gradients must be synchronized according to the data-parallel or ZeRO group semantics.
That synchronization can often overlap with continued backward work, but it is not the same collective as the Ulysses layout transpose.

A useful debugging rule is this:
if the tensor is an activation being rearranged between sequence and head ownership, expect All-to-All.
If the tensor is a replicated parameter gradient, expect a data-parallel reduction or ZeRO-style reduce-scatter.

## 7. Where Ulysses Shines

Ulysses is most appealing when three conditions hold.

First, the sequence length is large enough that sequence-sharding activations materially changes memory.
Second, the number of attention heads is large enough to support the desired sequence-parallel degree.
Third, the hardware network handles All-to-All efficiently enough that the transpose does not dominate runtime.

It is also attractive from an integration perspective.
The conceptual module boundary is clear.
Replace the attention module with one that performs the layout transpose, local attention, and inverse transpose.
The rest of the Transformer can remain close to a sequence-sharded data-parallel implementation.

That simplicity is different from Megatron SP.
Megatron SP is tightly coupled to tensor-parallel linear layers.
Ulysses can be easier to graft onto systems that already rely on ZeRO-style model-state sharding rather than Megatron tensor parallelism.

## 8. Where Ulysses Does Not Solve Everything

Ulysses is not a universal long-context answer.
The head-count bound matters.
All-to-All performance can be sensitive to topology and message sizes.
For causal attention, local attention still has the usual triangular work pattern inside each head.
And if a single head over the full sequence is too large for one device, Ulysses alone does not fix that.

That last point motivates [Ring Attention](../ring-attention/).
Ring Attention changes the problem again.
Instead of assigning one full-sequence head to a rank, it keeps local query blocks and circulates key/value blocks.
That lets the context exceed what one rank can hold for a full head.

In the sequence-parallelism family, Ulysses is the All-to-All transpose design.
It is compact, understandable, and useful.
It is also one piece of a broader toolkit.

## References

- Jacobs et al., [DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://arxiv.org/abs/2309.14509), 2023.
- Rajbhandari et al., [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), 2020.
- DeepSpeed sequence parallelism and ZeRO documentation.
