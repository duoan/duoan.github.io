---
title: "Megatron Internals II: Column/Row Parallel Linear and Vocab Parallel Embedding"
date: 2025-04-19
tags: ["LLM", "Megatron", "Tensor Parallel", "Model Parallel", "Embeddings", "Attention"]
categories: ["Engineering"]
draft: false
series: ["LLM Training"]
cover:
  image: column_parallel_linear.svg
  alt: "Column-parallel linear layer splitting output features across tensor-parallel ranks"
  relative: true
---

# Megatron Internals II: Column/Row Parallel Linear and Vocab Parallel Embedding

Tensor parallelism is not "split every tensor somehow." In Megatron, it is a small set of layer contracts: which dimension is local, which collective completes the dense math, and which gradient path communicates.

The original [Megatron-LM paper](https://arxiv.org/abs/1909.08053) is still the cleanest starting point: split transformer matrix multiplies so each GPU does useful dense GEMM, then communicate only where the algebra requires it.
This post walks the implementation-level contracts behind `ColumnParallelLinear`, `RowParallelLinear`, `VocabParallelEmbedding`, and parallel cross entropy.

## TL;DR

- `ColumnParallelLinear` splits output features. Forward can leave output shards local; backward all-reduces input gradients.
- `RowParallelLinear` splits input features. Forward all-reduces partial outputs; backward can keep input-gradient shards local.
- The common `f` and `g` operators are identity in one direction and all-reduce in the other direction.
- Attention uses column-parallel QKV, local heads, and a row-parallel output projection.
- The MLP uses column-parallel expansion, local activation, and row-parallel contraction.
- `VocabParallelEmbedding` shards vocabulary rows and all-reduces embeddings, so no rank stores the full table.
- Vocab-parallel cross entropy reduces max, denominator, and target logit instead of all-gathering full logits.
- For the rank groups underneath these layers, start with [Megatron Internals I](../megatron-distributed-init/). For the broader tensor-parallel overview, see [Tensor Parallelism in Megatron](../tensor-parallelism-megatron/).

## 1. Why tensor parallelism lives inside layers

A transformer block is mostly matrix multiplication, but different matrix dimensions have different meanings. For a linear layer:

```text
Y = X W
```

you can split `W` two obvious ways:

- split columns: each rank owns different output features;
- split rows: each rank owns different input features and computes a partial sum.

Both are useful. Neither works as a generic wrapper around arbitrary modules because the next layer must know whether it receives a full tensor or a shard. That is why Megatron implements tensor parallelism in custom modules rather than trying to shard an already-built dense model.

## 2. The two autograd operators

Megatron papers describe two conceptual operators:

```text
f: forward identity, backward all-reduce
g: forward all-reduce, backward identity
```

They are not magic math. They are custom autograd communication placements whose job is to avoid gather-scatter noise between adjacent layers. If a tensor is already sharded in the layout the next operation wants, keep it sharded. If the algebra needs a sum across shards, all-reduce exactly there.

## 3. ColumnParallelLinear

Column parallelism splits the output dimension:

```text
W = [W_0, W_1, ..., W_{p-1}]
Y_i = X W_i
Y = concat(Y_i)
```

Every TP rank receives the full input `X`.
Each rank owns one column shard of `W`.
Each rank computes one output-feature shard.

![ColumnParallelLinear: split output features](column_parallel_linear.svg)

A minimal sketch:

```python
class ColumnParallelLinear(nn.Module):
    def forward(self, x):
        y_local = x @ self.weight_local
        if self.gather_output:
            return all_gather_last_dim(y_local, tp_group)
        return y_local
```

The forward mode depends on the consumer.
If the next operation can consume sharded activations, Megatron leaves `Y_i` local.
If a non-parallel consumer needs the full hidden dimension, it all-gathers.

Backward needs the sum:

```text
dX = sum_i dY_i W_i^T
```

Each rank can compute one partial `dX`. The full input gradient is the all-reduce of those partials. That is the backward side of `f`.

## 4. RowParallelLinear

Row parallelism splits the input dimension:

```text
X = [X_0, X_1, ..., X_{p-1}]
W = [W_0; W_1; ...; W_{p-1}]
Y = sum_i X_i W_i
```

Each rank receives or creates one input-feature shard.
It computes a partial output with the full output dimension.
Then the TP group all-reduces those partial outputs.

![RowParallelLinear: split input features](row_parallel_linear.svg)

A minimal sketch:

```python
class RowParallelLinear(nn.Module):
    def forward(self, x):
        x_local = x if self.input_is_parallel else scatter_last_dim(x, tp_group)
        y_partial = x_local @ self.weight_local
        return all_reduce(y_partial, tp_group)
```

The communication moved from backward to forward. That is `g`: forward all-reduce, backward identity.

## 5. Why the column/row pair is efficient

The transformer MLP has an expansion and a contraction:

```text
Z = GELU(X W_up)
Y = Z W_down
```

Megatron computes the dense-equivalent result as:

```text
Z_i = GELU(X W_up_i)
Y_i = Z_i W_down_i
Y = all_reduce_i(Y_i)
```

No approximation is introduced.
The intermediate `Z_i` stays sharded.
That matters because `Z` is usually several times wider than the hidden state.
Gathering it between the two MLP projections would burn memory and bandwidth for no algebraic reason.

This is the pattern you should look for in tensor-parallel code:

1. split where independent work exists;
2. keep the large intermediate local;
3. reduce when the math becomes a sum.

## 6. Parallel self-attention

Attention uses the same contracts.
A typical attention block computes:

```text
Q, K, V = X W_qkv
O       = attention(Q, K, V) W_o
```

Megatron makes the QKV projection column-parallel.
Each rank owns a subset of attention heads.
Those heads can run attention locally because heads are independent once Q, K, and V are formed.
The output projection is row-parallel and all-reduces the head contributions back into the hidden dimension.

![Parallel self-attention alternates split and sync points](parallel_attention_block.svg)

The sequence is:

1. full hidden state enters the block;
2. column-parallel QKV creates local head shards;
3. attention runs locally on those heads;
4. row-parallel output projection all-reduces partial hidden states.

This is why head counts must divide cleanly by TP size.
Grouped-query attention changes how K/V heads are shared, but the same question remains: which heads are local, and where does the hidden dimension need a sum?

## 7. VocabParallelEmbedding

The embedding table can be one of the largest tensors in a language model:

```text
E: vocab_size x hidden_size
```

Megatron shards it by vocabulary rows.
Each TP rank owns a contiguous token-id range.
Every rank receives the token ids, masks out ids outside its range, looks up local rows, zeros the non-owned positions, and all-reduces the result.

![VocabParallelEmbedding: shard rows by token id range](vocab_parallel_embedding.svg)

Sketch:

```python
def forward(input_ids):
    mask = (input_ids < vocab_start) | (input_ids >= vocab_end)
    local_ids = input_ids - vocab_start
    local_ids = local_ids.masked_fill(mask, 0)
    out = embedding_local(local_ids)
    out = out.masked_fill(mask[..., None], 0)
    return all_reduce(out, tp_group)
```

Only the rank that owns a token contributes a non-zero vector.
The all-reduce is cheaper than keeping the full embedding table and optimizer state on every TP rank.

## 8. Vocab-parallel cross entropy

The output projection mirrors the input embedding.
Each rank produces logits for its vocabulary shard.
The naive move is to all-gather logits and run ordinary cross entropy.
That is usually the wrong move.

Cross entropy needs only three global quantities per token:

- the global maximum logit for numerical stability;
- the global sum of exponentials;
- the target logit for the true token id.

![Vocab-parallel cross entropy: reduce scalars, not logits](parallel_cross_entropy.svg)

The stable algorithm is:

1. compute local max over local vocab logits;
2. all-reduce max across TP ranks;
3. subtract global max and exponentiate local logits;
4. all-reduce the denominator;
5. pick the target logit only on the rank that owns the target token;
6. reduce that target logit;
7. compute loss and local vocab gradients.

This avoids a `batch * sequence * vocab` all-gather.
For large vocabularies, that is the difference between a normal output layer and a memory wall.

## 9. Backward-pass audit table

When reviewing a tensor-parallel module, write the forward and backward communication down explicitly:

| Module | Forward communication | Backward communication |
|---|---|---|
| `ColumnParallelLinear` | Optional all-gather | All-reduce `dX` |
| `RowParallelLinear` | All-reduce output | Usually none for sharded `dX` |
| `VocabParallelEmbedding` | All-reduce embeddings | Gradients stay with owned vocab rows |
| `VocabParallelCrossEntropy` | Max, denominator, target-logit reductions | Local vocab gradients plus small reductions |

This table also explains why activation checkpointing with TP must restore both RNG state and tensor layout.
The recomputed forward pass must produce the same shard shapes as the original forward pass.
Otherwise backward collectives run on the wrong tensors.

## 10. The useful mental model

Megatron tensor parallelism is dense math with local shards. It works because the split dimensions match the algebra:

- column split means independent output features;
- row split means partial sums over input features;
- vocab split means independent token-id rows;
- parallel cross entropy means reducing the few global scalars the loss actually needs.

The code looks complicated because it must handle sequence parallelism, async communication, fused kernels, and initialization. The core idea is still small: keep tensors sharded while the next operation can consume the shard, and communicate only when the dense equation requires a concat or a sum.

## Code

- Megatron tensor-parallel layers: [`megatron/core/tensor_parallel/layers.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py).
- Tensor-parallel communication mappings and custom autograd functions: [`megatron/core/tensor_parallel/mappings.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/mappings.py).
- Vocab-parallel cross entropy: [`megatron/core/tensor_parallel/cross_entropy.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/cross_entropy.py).
- Full tensor-parallel package: [`megatron/core/tensor_parallel/`](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/tensor_parallel).
- Megatron process groups used by these layers: [`megatron/core/parallel_state.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py).

## References

- Shoeybi et al., [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053), 2019.
- Narayanan et al., [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473), 2021.
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), NeurIPS 2017.
- Korthikanti et al., [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198), 2022.
