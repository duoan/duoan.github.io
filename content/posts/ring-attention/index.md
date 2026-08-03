---
title: "Sequence Parallelism III: Ring Attention for Context That Does Not Fit"
date: 2025-05-26
tags: ["LLM", "Training", "Parallelism", "Attention", "Long Context"]
categories: ["Engineering"]
draft: false
series: ["LLM Training"]
cover:
  image: ring_kv_passing.svg
  alt: "Ring Attention keeps query blocks fixed on each GPU while key/value blocks move around a communication ring"
  relative: true
---

# Sequence Parallelism III: Ring Attention for Context That Does Not Fit

The usual misconception is that long-context attention needs every rank to assemble the full context for its head.
Ring Attention avoids that.
Each rank keeps a local query block fixed, streams key/value blocks around a ring, and updates the same exact attention result block by block.

The key is online softmax.
Liu, Zaharia, and Abbeel use blockwise attention plus ring communication so no rank has to materialize the full score matrix or hold the full K/V context at once ([arXiv:2310.01889](https://arxiv.org/abs/2310.01889)).
That makes Ring Attention different from [DeepSpeed Ulysses](../sequence-parallelism-ulysses/), which uses All-to-All to assign full-sequence head shards.
It also sets up [Megatron Context Parallel](../megatron-context-parallel/), which adapts the ring idea inside Megatron.

## TL;DR

- Ring Attention splits Q, K, and V into sequence blocks.
- Each rank keeps its own Q block fixed.
- K/V blocks rotate around the ranks in a ring.
- Each rank updates its local output block as each K/V block arrives.
- Online softmax state makes the blockwise updates mathematically equivalent to full attention.
- Communication can be hidden when K/V transfer for the next chunk is no slower than attention compute on the current chunk.
- Causal masks create load imbalance unless chunks are assigned carefully.
- Reproducible figures for this post: [`playground/llm_training_series_figures.py`](https://github.com/duoan/duoan.github.io/blob/main/playground/llm_training_series_figures.py).

## 1. Start With the Single-GPU Problem

Naive attention forms:

```text
S = Q K^T
P = softmax(S)
O = P V
```

The score matrix `S` has shape `[sequence, sequence]`.
For long contexts, that matrix is too large to materialize as an activation.
FlashAttention solves the local version of this problem by processing tiles and carrying enough state to compute the exact softmax result.

Ring Attention uses the same numerical idea and changes where the K/V tiles come from.
If a query block can consume local K/V blocks one at a time, it can also consume K/V blocks that arrive from other ranks one at a time.
The distributed part is the source of the next K/V block.
The math is still exact blockwise attention.

## 2. Online Softmax Is the Contract

Softmax is not additive in the naive way.
If you compute attention over `K0,V0` and then over `K1,V1`, you cannot just add two independent outputs.
The normalization denominator must include both score blocks.
The row maximum used for numerical stability must also be global across all processed keys.

Online softmax keeps three pieces of state per query row:

1. the running maximum `m`,
2. the running denominator `l`,
3. the running output vector `O`.

When a new score block arrives, the algorithm updates all three.
The previous output is rescaled if the running maximum changes.
The new block contribution is scaled into the same denominator.
After all K/V blocks have been processed, the output matches full attention.

![Online softmax lets attention consume score blocks without materializing S x S](online_softmax_blocks.svg)

This state is small compared with the full score matrix.
It is also local to the query block.
That locality is why a rank can keep Q fixed while K/V moves.

## 3. The Ring Schedule

Assume four ranks.
Rank `0` owns `Q0`, `K0`, and `V0`.
Rank `1` owns `Q1`, `K1`, and `V1`.
The pattern continues.

At iteration `0`, each rank computes attention between its local Q block and its local K/V block.
At the same time, it sends its K/V block to the next rank and receives a K/V block from the previous rank.
At iteration `1`, it computes against the received K/V block while the next exchange is in flight.
After enough iterations, every Q block has seen every K/V block it needs.

![Ring Attention keeps Q fixed and rotates KV blocks](ring_kv_passing.svg)

The output block does not move during forward.
Only K/V blocks rotate.
That is the key difference from Ulysses, where All-to-All reassigns Q/K/V ownership by head.

The original Ring Attention paper describes this with JAX `ppermute` for neighbor exchange and blockwise attention kernels for the local computation ([arXiv:2310.01889](https://arxiv.org/abs/2310.01889)).
The exact transport can vary by framework.
The ownership rule should not.

## 4. Communication-Compute Overlap

The performance target is straightforward:

```text
time_to_transfer_next_KV <= time_to_compute_attention_on_current_KV
```

If this inequality holds, communication hides under compute.
If it does not, each ring step exposes a wait.

Let `c` be the sequence chunk size and `d` the head dimension.
For one query chunk against one K/V chunk, the two dominant matmuls are:

```text
Q K^T  : roughly 2 * d * c^2 FLOPs
P V    : roughly 2 * d * c^2 FLOPs
```

The total is roughly:

```text
4 * d * c^2 FLOPs
```

For fp16 or bf16, transferring K and V is roughly:

```text
4 * d * c bytes
```

If `F` is sustained FLOP/s and `B` is sustained Byte/s for the ring link, hiding transfer requires:

```text
4dc / B <= 4dc^2 / F
```

which simplifies to:

```text
c >= F / B
```

![Chunk size must make attention compute at least as long as KV transfer](chunk_size_tradeoff.svg)

The constants move in real kernels.
Softmax work, masks, launch overhead, topology, dtype, and compute efficiency all matter.
The direction does not change.
Tiny chunks expose communication and scheduling overhead.
Huge chunks improve overlap but raise memory pressure and can reduce parallelism.

## 5. Causal Masks Change Work Distribution

For bidirectional attention, every query block attends to every key block.
If each rank owns one contiguous query block, useful work is evenly distributed.

Causal language-model attention is triangular.
Early query blocks attend to fewer key blocks.
Late query blocks attend to many key blocks.
If rank `0` owns the earliest tokens and rank `3` owns the latest tokens, rank `3` has much more useful work.

In a ring, ranks move in lockstep.
One rank with little work can still wait at synchronization points.
One rank with heavy work can determine the iteration time.
Average FLOPs per rank is not enough.
Balanced useful work per step matters.

This is why [Megatron Context Parallel](../megatron-context-parallel/) pairs early and late sequence chunks for causal models.
The ring schedule solves residency.
Chunk placement solves load balance.

## 6. Ring Attention vs Ulysses

Ulysses and Ring Attention are easy to conflate because both cut along sequence.
They solve different layout problems.

Ulysses starts with sequence shards and uses All-to-All to assemble full-sequence head shards.
After that transpose, a rank computes attention for one or more heads.
The degree is naturally limited by head count unless hierarchy is added.

Ring Attention starts with sequence blocks and keeps each query block local.
It streams K/V blocks through the ring.
The per-rank output is a local sequence block, not a full-sequence head.

In one sentence:

```text
Ulysses moves Q/K/V ownership once; Ring Attention moves K/V ownership repeatedly.
```

That distinction affects memory, collectives, topology, and overlap.

## 7. Backward Pass Intuition

Backward follows the same blockwise ownership.
Gradients for Q remain tied to the local query block.
Gradients for K and V must accumulate contributions from the query blocks that consumed each K/V block.
A mirrored ring schedule can move and accumulate those contributions back to their owners.

The implementation is more involved than the forward explanation.
The invariant is simpler:

```text
every local gradient contribution must be reduced to the owner of the shard that will use it
```

For Ring Attention, K/V-related gradient ownership is coupled to the same block circulation that made forward possible.
When debugging, do not look for one giant attention collective.
Look for neighbor exchanges, local block attention kernels, and reductions to shard owners.

## 8. Practical Failure Modes

Ring Attention is exact, but it is not magic.
Several things can erase the expected win.

First, chunk size can be wrong.
Too small and communication dominates.
Too large and memory pressure returns.

Second, overlap can fail.
If send/recv ordering, CUDA streams, or buffer dependencies serialize transfer and compute, the ring becomes a communication loop with compute between waits.

Third, causal imbalance can dominate.
If early chunks skip most work while late chunks do all work, the slowest useful rank sets the pace.

Fourth, online softmax must be implemented correctly.
The running max and denominator are not optional bookkeeping.
They are what make blockwise attention equal to full attention.

## 9. Where Ring Attention Fits

Ring Attention belongs in the "context does not fit" part of the toolbox.
[Megatron SP](../sequence-parallelism-megatron-sp/) is about replicated activation memory around tensor-parallel regions.
[Ulysses](../sequence-parallelism-ulysses/) is about All-to-All head ownership.
Ring Attention is about streaming K/V so no rank needs the whole context resident for a head at once.

Those methods can compose.
A large training job might use tensor parallelism for GEMMs, ZeRO for model states, sequence or context parallelism for long contexts, and pipeline parallelism for layers.
The hard part is maintaining a precise ownership story for each tensor as it crosses module boundaries.

Ring Attention is a clean example of that discipline.
Keep Q fixed.
Rotate K/V.
Update online softmax state.
Choose chunks so communication hides.

## Code

- Author Ring Attention implementation: [`haoliuhl/ringattention`](https://github.com/haoliuhl/ringattention).
- Ring Attention paper appendix points to complete code in [`lhao499/llm_large_context`](https://github.com/lhao499/llm_large_context).
- The paper's JAX implementation uses blockwise attention plus `jax.lax.ppermute` for ring exchange ([arXiv HTML appendix](https://arxiv.org/html/2310.01889)).

## References

- Liu, Zaharia, Abbeel, [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889), 2023.
- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), 2022.
- Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691), 2023.
