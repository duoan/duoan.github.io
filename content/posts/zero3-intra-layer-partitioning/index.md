---
title: "The ZeRO-3 Diagram Most People Remember Is Wrong"
date: 2025-06-16
tags: ["LLM", "Training", "DeepSpeed", "ZeRO", "Distributed Systems"]
categories: ["Engineering"]
draft: false
series: ["LLM Training"]
cover:
  image: actual_intra_layer.svg
  alt: "Current DeepSpeed ZeRO-3 partitions each parameter as flattened intra-layer slices across ranks"
  relative: true
---

# The ZeRO-3 Diagram Most People Remember Is Wrong

Many engineers remember ZeRO-3 as an inter-layer ownership animation: one GPU owns early layers, another owns later layers, and active layers are broadcast when needed.
That picture is memorable.
It is also the wrong operational model for current DeepSpeed ZeRO-3 training.

The current steady-state model is intra-layer flattening and partitioning.
Parameters are flattened into 1D buffers, padded when needed, split across data-parallel ranks, gathered just in time for module compute, and partitioned again after use.
That is consistent with the ZeRO paper's goal of partitioning model states ([arXiv:1910.02054](https://arxiv.org/abs/1910.02054)) and with DeepSpeed's Stage 3 code in `deepspeed/runtime/zero/stage3.py`.

For background on ZeRO stages, see [Zero Redundancy Optimizer](../zero-redundancy-optimizer/).
For contrast with computation-splitting model parallelism, see [Megatron tensor parallelism](../tensor-parallelism-megatron/).
For activation layout rather than model-state layout, see [Ulysses](../sequence-parallelism-ulysses/) and [Megatron Context Parallel](../megatron-context-parallel/).

## TL;DR

- The popular ZeRO-3 animation suggests inter-layer partitioning: rank `0` owns some layers, rank `1` owns other layers, and layers are broadcast when needed.
- Current DeepSpeed ZeRO-3 partitions parameters intra-layer as flattened 1D tensor shards across ranks.
- Forward compute AllGathers the full parameter for the module that is about to run.
- Backward compute may AllGather the full parameter again when gradient computation needs it.
- Parameter gradients are ReduceScattered or partitioned so each rank receives the gradient shard matching its parameter shard.
- ZeRO-3 has model-state sharding but mostly data-parallel compute semantics.
- The code path to read is `deepspeed/runtime/zero/stage3.py`, not a layer-owner diagram.
- Reproducible figures for this post: [`playground/llm_training_series_figures.py`](https://github.com/duoan/duoan.github.io/blob/main/playground/llm_training_series_figures.py).

## 1. The Diagram People Remember

The memorable diagram looks roughly like this.
Suppose a model has sixteen layers and four GPUs.
GPU `0` owns layers `0-3`.
GPU `1` owns layers `4-7`.
GPU `2` owns layers `8-11`.
GPU `3` owns layers `12-15`.
When the forward pass reaches a layer group, the owner broadcasts that group to the other ranks.

![The memorable but misleading picture: inter-layer shards and broadcasts](marketing_inter_layer.svg)

That picture resembles pipeline parallelism without the pipeline schedule.
It is easy to remember because layers are intuitive objects.
Humans think in layers.
Animations also look cleaner when each GPU owns a neat stack of layers.

But this model creates bad expectations.
You would expect communication to look like layer-group broadcast.
You might expect ownership to align with module boundaries.
You might expect memory pressure to depend heavily on which layers a rank owns.
Those expectations do not match current DeepSpeed ZeRO-3 behavior.

## 2. The Actual Parameter Partition

Current ZeRO-3 treats each parameter tensor as storage that can be flattened and partitioned.
For a linear layer weight, the process is conceptually:

1. take the parameter tensor,
2. flatten it into a 1D buffer,
3. pad it if the element count does not divide evenly across the data-parallel group,
4. assign each rank a contiguous slice,
5. keep only the owned partition resident between compute regions.

![Current ZeRO-3 partitions each parameter as a flattened 1D tensor](actual_intra_layer.svg)

This is intra-layer partitioning.
Every rank owns a slice of many parameters, not an exclusive stack of layers.
The slice may cut across rows, columns, or arbitrary positions in the flattened tensor.
It is not tensor parallelism because the computation is not permanently rewritten to operate on those slices.
It is state partitioning.

That distinction is the heart of ZeRO-3.
Storage is sharded.
Compute usually runs as if each rank temporarily has the full parameter.

## 3. The Collectives You Should Expect

For each parameter needed by a module, ZeRO-3 gathers the full parameter before compute.
During forward, that AllGather makes the module's parameter available.
During backward, the parameter may be gathered again because gradient computation needs the same full tensor.
After gradients are computed, ZeRO-3 partitions gradients back to the owning ranks.

![The collectives to expect from ZeRO-3: AllGather weights, ReduceScatter gradients](zero3_collectives.svg)

The steady-state training loop is closer to:

```text
resident: parameter shards
forward:  AllGather parameter -> compute -> release or keep briefly
backward: AllGather parameter -> compute gradients -> ReduceScatter gradients
update:   optimizer updates local shard
```

It is not:

```text
rank owns layers -> broadcast layer group -> compute everywhere
```

Broadcast can appear in initialization or auxiliary paths.
The important steady-state training pattern is gather for full-parameter compute and scatter for shard ownership.
DeepSpeed's documentation also describes ZeRO-3 as collecting and partitioning parameters during forward and backward, rather than assigning whole layers to ranks ([DeepSpeed ZeRO-3 docs](https://deepspeed.readthedocs.io/en/latest/zero3.html)).

## 4. What the Code Shows

The code path makes the operational model concrete.
DeepSpeed Stage 3 uses flatten and unflatten helpers, tracks fp16 partitioned groups, and creates flat partition buffers in `deepspeed/runtime/zero/stage3.py`.
The code names vary over time, but the pattern is stable: parameters are grouped, flattened, partitioned, gathered for compute, and partitioned again.

The important objects are not "layer owner" objects.
They are partitioned parameter tensors, flat buffers, padding sizes, bucket metadata, and status flags describing whether a parameter is available, partitioned, or in flight.

That is why a trace from real ZeRO-3 training shows many parameter gathers around module execution.
It is not a broken implementation of the layer animation.
It is the real design.

## 5. Why This Is Still ZeRO, Not Tensor Parallelism

ZeRO-3 often confuses people because it has a model-parallel storage shape.
At rest, no rank has the full model state.
That looks like model parallelism.

But the training semantics are data-parallel.
Each rank processes a different data shard.
When a module runs, it reconstructs the parameter needed for that module.
The module computation is not split the way Megatron tensor parallelism splits a matrix multiplication.

Compare the two:

- Tensor parallelism changes the computation graph so each rank computes a slice of a layer.
- ZeRO-3 changes state residency so each rank stores only a slice between compute regions.
- Tensor parallelism requires parallel-aware layers or model code.
- ZeRO-3 aims to wrap ordinary modules by gathering parameters just in time.

This is why ZeRO-3 pairs naturally with other parallel dimensions.
You can use ZeRO-3 for model-state memory while using [Ulysses](../sequence-parallelism-ulysses/) or [Megatron Context Parallel](../megatron-context-parallel/) for long-sequence activations.
The ownership questions are different.

## 6. Why the Wrong Mental Model Spread

There are three reasons the older picture persisted.

First, early ZeRO material emphasized partitioning model states, and layer-wise diagrams were easy to understand.
They communicated "the whole model is no longer replicated" effectively.

Second, some public explanations were produced while implementations and terminology were still evolving.
The details that matter today were not always reflected in early diagrams.

Third, later readers reused the memorable picture because it made intuitive sense.
Layer ownership is easier to draw than flattened parameter shards, gather lifetimes, release hooks, and partition caches.

The lesson is not that diagrams are bad.
The lesson is that storage ownership and compute ownership must be shown separately.
When a system shards storage but reconstructs tensors for compute, a layer-placement diagram hides the important behavior.

## 7. How to Read ZeRO-3 Traces

If your mental model is inter-layer broadcast, a real ZeRO-3 trace looks surprising.
You may see many parameter AllGathers around module execution.
You may see parameter prefetching for upcoming modules.
You may see parameters released after use.
You may see gradient ReduceScatter or partitioning during backward.

Those are expected.
The trace is showing intra-layer state sharding with just-in-time full-parameter compute.

When debugging memory, ask:

- Which parameter shards are resident at rest?
- Which full parameters are currently gathered for compute?
- Are prefetched parameters overlapping with the current module?
- Are old full parameters released soon enough?
- Are gradients partitioned back to shards instead of accumulating full copies?
- Are offload buffers or persistence thresholds changing residency?

These questions are more useful than asking which rank owns which layer group.

## 8. Memory Implications

ZeRO-3 targets model states: parameters, gradients, and optimizer states.
It reduces persistent memory by partitioning those states across ranks.
With Adam and mixed precision, optimizer state is often the largest static piece.
Partitioning it matters.

But ZeRO-3 does not automatically make every activation fit.
If a single layer's activation memory exceeds GPU capacity, adding ZeRO-3 ranks may not fix the problem.
That is where tensor parallelism, sequence parallelism, context parallelism, activation checkpointing, or attention kernels enter.

For long-context training, pair the mental models:

- ZeRO-3: shard model-state storage.
- Megatron SP: shard token-local activation storage around TP regions.
- Ulysses: transpose sequence shards into head shards for attention.
- Ring Attention or CP: stream K/V so a full context does not sit on one rank.

Each one attacks a different memory term.

## 9. Communication Implications

The corrected model changes communication expectations.
AllGather cost scales with parameter size and the timing of module execution.
ReduceScatter or partitioning cost scales with gradient size.
Prefetch and overlap determine how much of that cost is visible.

If you expected broadcasts of layer groups, you may tune the wrong thing.
For current ZeRO-3, focus on:

- parameter bucket sizes,
- prefetch distance,
- persistence thresholds,
- overlap of communication with forward or backward compute,
- release timing for gathered parameters,
- interaction with activation checkpointing,
- offload and CPU/NVMe buffer behavior when enabled.

The communication is not free.
ZeRO-3 buys memory by increasing parameter movement.
That trade can be excellent when the alternative is not fitting the model at all.
It can be poor for small models or slow interconnects.

## 10. Practical Takeaway

If you only need a slogan, use this one:

```text
ZeRO-3 shards storage inside layers; it does not assign whole layers to ranks.
```

That slogan prevents three common mistakes.
It keeps ZeRO-3 distinct from pipeline parallelism.
It keeps ZeRO-3 distinct from tensor parallelism.
It makes AllGather and ReduceScatter feel expected rather than surprising.

Once that is clear, ZeRO-3 becomes easier to combine with the rest of the LLM training toolbox.
Use it to reduce persistent model-state memory.
Use tensor, sequence, or context parallelism when compute tensors themselves need to be partitioned.
Use profiling to decide whether the extra gathers and scatters are hidden well enough by real compute.

The memorable layer-broadcast diagram did its job as an introduction.
For engineering work, replace it with intra-layer 1D partitioning and the AG/RS timeline.

## Code

- DeepSpeed Stage 3 optimizer implementation: [`deepspeed/runtime/zero/stage3.py`](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/runtime/zero/stage3.py), including flat partition groups and partitioned parameter buffers.
- DeepSpeed partitioned parameter logic: [`deepspeed/runtime/zero/partition_parameters.py`](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/runtime/zero/partition_parameters.py).
- DeepSpeed ZeRO-3 documentation: [`zero3.html`](https://deepspeed.readthedocs.io/en/latest/zero3.html), which describes parameter collection and partitioning during forward and backward.

## References

- Rajbhandari et al., [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), 2020.
- DeepSpeed, [ZeRO-3 documentation](https://deepspeed.readthedocs.io/en/latest/zero3.html).
