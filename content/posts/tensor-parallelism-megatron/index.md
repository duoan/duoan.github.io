---
title: "Tensor Parallelism in Megatron-LM: Splitting Layers, Not Stacks"
date: 2025-05-18
tags: ["LLM", "Distributed Training", "Tensor Parallelism", "Megatron-LM", "Transformer", "NCCL"]
categories: ["Engineering"]
draft: false
series: ["LLM Training"]
cover:
  image: mlp_tp.svg
  alt: "Megatron-LM tensor parallel MLP with column and row splits"
  relative: true
---

# Tensor Parallelism in Megatron-LM: Splitting Layers, Not Stacks

Pipeline parallelism splits a Transformer by depth: different GPUs own different layers. Tensor parallelism splits a Transformer inside each layer: different GPUs own different slices of the same matrix multiply, attention head set, or vocabulary table.

Megatron-LM made this practical for large Transformers by choosing splits that preserve local compute and place collectives at a small number of predictable points. The rule is simple: split where the math lets ranks work independently, then use all-reduce only when partial results must be combined.

## TL;DR

- Tensor parallelism splits **layers**, not layer stacks. It is usually placed inside a node where NVLink/NVSwitch bandwidth is high.
- For `Y = X A`, a column split of `A` produces output shards that can be concatenated. A row split of `A` produces partial sums that must be reduced.
- Megatron's MLP uses column-parallel first projection, local GeLU, then row-parallel second projection. This avoids communicating before the nonlinearity.
- Attention is naturally parallel over heads. Split Q, K, and V by head groups, compute attention locally, then reduce after the output projection.
- Vocab-parallel embeddings shard the large token table. Vocab-parallel cross-entropy avoids gathering full logits when vocabulary size is large.
- A common production layout is TP within a node and DP across nodes, often combined with ZeRO for optimizer-state memory.
- Reproducible figures for this post: [`playground/llm_training_series_figures.py`](https://github.com/duoan/duoan.github.io/blob/main/playground/llm_training_series_figures.py).

## 1. Why Tensor Parallelism Exists

Data parallelism scales throughput when a full model replica fits on each rank. ZeRO reduces redundant state, but each layer still has to execute. Pipeline parallelism splits layers across depth, but stage balance and bubbles become new problems.

Tensor parallelism attacks a different limit: **one layer can be too large or too expensive for one GPU**.

Large Transformers contain repeated dense operations:

- MLP projections from `h` to `4h` and back.
- Q, K, V projections for attention.
- Attention output projections.
- Vocabulary embeddings and output logits.

These are matrix operations with structure. If we split the matrices carefully, each GPU can perform a slice of the same layer, and the ranks communicate only at mathematically necessary boundaries.

Megatron-LM's tensor parallelism is sometimes called intra-layer model parallelism. That name is accurate: it partitions the tensor algebra inside a Transformer block.

## 2. Two Splits for `Y = X A`

Let `X` have shape `[b, s, h]`, where:

- `b` is local batch size.
- `s` is sequence length.
- `h` is hidden size.

Let a linear layer weight `A` have shape `[h, h']`. The output is:

```text
Y = X A
```

There are two basic ways to split `A`.

![Column and row splits have different communication needs](row_vs_column_split.svg)

### Column split

Split `A` along its output dimension:

```text
A = [A_1 | A_2 | ... | A_p]
Y_i = X A_i
Y = [Y_1 | Y_2 | ... | Y_p]
```

Each rank receives the same `X` and computes a shard of the output features. No reduction is needed to compute local `Y_i`. A later operation may need the shards concatenated, but the linear operation itself is embarrassingly parallel.

### Row split

Split `A` along its input dimension:

```text
X = [X_1 | X_2 | ... | X_p]
A = [A_1; A_2; ...; A_p]
Y = sum_i X_i A_i
```

Each rank computes a partial output. The final result requires summing partials across ranks, usually with an all-reduce.

Megatron builds Transformer tensor parallelism from these two primitives.

## 3. The MLP Block

A standard Transformer MLP is:

```text
Z = GeLU(X A)
Y = Z B
```

where `A` maps `h -> 4h`, and `B` maps `4h -> h`.

Megatron chooses:

- `A` is **column-parallel**.
- GeLU is computed locally on each output shard.
- `B` is **row-parallel**.
- The row-parallel output is all-reduced.

![Megatron MLP uses column-parallel GeLU then row-parallel output](mlp_tp.svg)

This choice avoids a communication point before GeLU. That matters because GeLU is nonlinear:

```text
GeLU(a + b) != GeLU(a) + GeLU(b)
```

If the first projection were row-parallel, ranks would produce partial sums and need an all-reduce before GeLU. By making it column-parallel, each rank owns complete output channels for its shard and can apply GeLU independently.

The second projection then consumes sharded `4h` features and produces partial `h` outputs. Those partials are summed by an all-reduce. In Megatron notation, the pair of communication operators is often described as:

- `f`: identity in forward, all-reduce in backward.
- `g`: all-reduce in forward, identity in backward.

The exact autograd implementation can vary, but the invariant is stable: one collective at the MLP output in forward, and the corresponding collective on the gradient path.

## 4. Attention: Split the Heads

Multi-head attention already decomposes hidden channels into heads. That makes it a natural fit for tensor parallelism.

![Attention splits QKV and attention heads across tensor-parallel ranks](attention_tp.svg)

For `H` attention heads and tensor parallel size `p`, assign roughly `H/p` heads to each rank. Each rank holds the Q, K, and V projection shards for its heads:

```text
Q_i = X W^Q_i
K_i = X W^K_i
V_i = X W^V_i
```

Then each rank computes attention for its local heads:

```text
O_i = softmax(Q_i K_i^T / sqrt(d_head)) V_i
```

No rank needs another rank's heads to compute its own attention outputs. After the heads are produced, the output projection is row-parallel, just like the second MLP projection. Partial outputs are summed.

This is why tensor-parallel configurations often require the number of attention heads to be divisible by TP size. Some systems support uneven or grouped layouts, but clean divisibility reduces edge cases and load imbalance.

## 5. Communication Per Transformer Block

The simplified Megatron block has two major all-reduce points in forward:

1. MLP output after the row-parallel projection.
2. Attention output after the row-parallel projection.

Backward has the matching reductions for input gradients. If the communicated activation tensor has size:

```text
Phi_TP = b * s * h
```

then the communication volume per block is often summarized as proportional to several all-reduces over `Phi_TP`.

This explains a critical placement rule: tensor parallelism wants fast local links. TP communication happens every layer and moves activation-sized tensors. Put TP ranks on GPUs connected by NVLink or NVSwitch when possible. Use data parallelism across slower cross-node links, where communication is gradient-sized and happens once per step or per bucket.

## 6. Vocab-Parallel Embedding

The token embedding table has shape `[vocab, h]`. For large vocabularies, it can be a significant memory block. Megatron shards it across the vocabulary dimension.

![Vocab-parallel embedding shards token ranges and reduces sparse results](embedding_vocab_parallel.svg)

Each TP rank owns a range of token IDs:

```text
rank 0: tokens [0, v/p)
rank 1: tokens [v/p, 2v/p)
...
```

During embedding lookup:

1. Each rank checks which input token IDs fall into its vocabulary range.
2. It returns embeddings for tokens it owns and zeros for tokens it does not own.
3. Ranks all-reduce the embedding outputs.

Because each token belongs to exactly one shard, summing the local results reconstructs the full embedding output.

The output embedding or language-model head is often tied to the input embedding. Weight tying across pipeline stages or tensor-parallel groups requires care: gradients from input and output uses must be accumulated into the same sharded parameter.

## 7. Vocab-Parallel Cross-Entropy

The naive way to compute language-model loss is:

1. Gather all vocab-sharded logits into a full `[b, s, vocab]` tensor.
2. Apply softmax.
3. Compute cross-entropy against target tokens.

That all-gather is expensive because `vocab` is large.

Vocab-parallel cross-entropy avoids materializing full logits. Each rank computes local logits for its vocabulary shard. The softmax denominator can be formed with reductions over per-token local sums. The target logit is selected from the rank that owns the target token, then reduced or broadcast as a small tensor.

The communication changes from something proportional to:

```text
b * s * vocab
```

to reductions closer to:

```text
b * s
```

plus small scalar or vector exchanges. For large vocabularies, this is the difference between a practical output layer and a communication wall.

## 8. TP + DP Hybrid Layout

Megatron's classic large-model recipe combines tensor parallelism and data parallelism.

![Tensor parallelism is usually intra-node; data parallelism spans replicas](tp_dp_hybrid.svg)

A common placement is:

- TP group inside a node, using fast GPU-to-GPU links.
- DP group across nodes, using network links for gradient synchronization.
- ZeRO or distributed optimizer across DP ranks to reduce optimizer-state memory.

This placement follows the communication pattern:

- TP communicates activation tensors every layer, so it wants the fastest links.
- DP communicates gradient buckets during backward, so it can tolerate larger, less frequent cross-node collectives.
- Pipeline parallelism, when used, adds another process-group dimension across layer stages.

The result is often called 3D parallelism: data, tensor, and pipeline. Modern training stacks add sequence/context parallelism and expert parallelism for long-context and MoE models, but TP remains the core intra-layer primitive.

## 9. Experimental Lessons from Megatron-LM

The Megatron-LM paper, *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* (Shoeybi et al., 2019), showed that carefully chosen tensor parallelism can scale Transformer training to multi-billion-parameter models with high hardware utilization.

The important experimental lessons are durable:

- Splitting by Transformer structure is better than generic matrix sharding. MLP and attention have different natural split points.
- Communication must be placed around large GEMMs, not between tiny operations.
- TP size is limited by communication and head divisibility. Larger TP is not automatically better.
- Combining TP with DP gives a better scaling surface than using either alone.
- Kernel efficiency matters. If tensor shards become too small, GEMMs lose efficiency and communication dominates.

These lessons still apply even though modern models are larger, GPUs are faster, and Megatron-Core has evolved far beyond the original code.

## 10. Debugging Tensor Parallel Training

Tensor-parallel bugs are often shape or group bugs. A short checklist:

- Verify hidden size, number of heads, and MLP intermediate size are divisible by TP size.
- Confirm TP process groups are local to the intended fast-link domain.
- Check that row-parallel layers reduce outputs exactly once.
- Check that column-parallel layers do not accidentally gather too early.
- Validate vocab-parallel target handling for tokens on every shard.
- Ensure tied embeddings receive gradients from all uses before the optimizer step.
- Profile all-reduce overlap and GEMM sizes; small shards can look correct but run slowly.

Correct tensor parallelism is not just "split the matrix." It is split, compute locally, communicate at the mathematical boundary, and keep that boundary aligned with hardware.

## 11. The Core Pattern

Megatron-LM tensor parallelism can be reduced to one pattern:

```text
Choose a split that preserves local nonlinear work.
Delay communication until partial linear results must be combined.
Place the TP group on the fastest links available.
```

For the MLP, that means column split before GeLU and row split after. For attention, it means split heads. For embeddings and loss, it means shard the vocabulary and reduce only the small quantities needed to reconstruct the result.

The reward is a Transformer block that can be wider than one GPU while still looking like one layer to the rest of the training stack.

## References

- Mohammad Shoeybi et al., [*Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*](https://arxiv.org/abs/1909.08053), 2019.
- Deepak Narayanan et al., [*Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*](https://arxiv.org/abs/2104.04473), SC 2021.
- Ashish Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), NeurIPS 2017.
- Samyam Rajbhandari et al., [*ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*](https://arxiv.org/abs/1910.02054), SC 2020.
