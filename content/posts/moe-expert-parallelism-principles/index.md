---
title: "MoE Parallelism Principles: GShard, Expert Parallel, and All-to-All"
date: 2025-05-10
tags: ["LLM", "MoE", "Expert Parallel", "GShard", "All-to-All", "Distributed Training"]
categories: ["Engineering"]
draft: false
series: ["LLM Training"]
cover:
  image: gshard_moe_layer.svg
  alt: "GShard-style MoE layer with gate, experts, and top-k combine"
  relative: true
---

# MoE Parallelism Principles: GShard, Expert Parallel, and All-to-All

Mixture-of-Experts is sparse compute wrapped in a communication problem. The model idea is simple: replace a dense FFN with many expert FFNs and route each token to a few of them. The distributed-training problem starts when those experts live on different GPUs.

Shazeer et al. introduced the sparsely-gated MoE layer in [Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538).
GShard made the pattern practical at scale with top-2 routing, capacity, load-balancing loss, and all-to-all dispatch in [GShard](https://arxiv.org/abs/2006.16668).
Switch simplified the router to top-1 in [Switch Transformers](https://arxiv.org/abs/2101.03961).
This post explains the shared systems model underneath them.

For the later production bottlenecks after this foundation, see [Large MoE Performance: The Three Walls After Sparsity](../large-moe-from-sparsity-to-communication/).

## TL;DR

- A transformer MoE layer usually replaces the dense FFN with a gate, expert FFNs, dispatch, and combine.
- The gate chooses top-k experts per token; top-2 is GShard-like, top-1 is Switch-like.
- Expert Parallelism (EP) shards experts across ranks. Data Parallelism (DP) replicates that sparse layout.
- All-to-All dispatch sends tokens from source ranks to expert-owning ranks; a second All-to-All sends outputs back.
- Capacity turns irregular routing into fixed tensor shapes by padding, dropping, or otherwise bounding per-expert tokens.
- Load-balancing loss is a systems feature as much as a modeling feature: it keeps experts and network traffic from collapsing.
- EP can compose with TP inside experts and PP across layers, but each extra axis adds a communication schedule.

## 1. Dense FFN versus MoE FFN

A dense transformer FFN applies the same parameters to every token:

```text
Dense FFN: token -> shared FFN
```

An MoE FFN routes tokens:

```text
MoE FFN: token -> gate -> selected experts -> weighted combine
```

If there are `E` experts and each token chooses `k`, the model can have `E` expert parameter sets while each token activates only `k` of them.
That is the attractive scaling law.
Total parameters grow with `E`.
Per-token expert compute grows with `k`.

The catch is ownership. If experts are split across GPUs, routing is no longer just an index operation; it is a communication plan.

## 2. The GShard-style layer

GShard places the MoE layer where the dense FFN would normally sit. Attention remains dense. The MoE block contains:

- a **gate** that maps hidden states to expert scores;
- **experts**, usually FFNs with the same input and output hidden size;
- a **dispatch** path that packs tokens by target expert;
- a **combine** path that restores original token order and weights outputs.

![GShard MoE replaces the dense FFN with routed experts](gshard_moe_layer.svg)

Minimal pseudocode:

```python
logits = gate(hidden)                 # [tokens, experts]
scores = softmax(logits, dim=-1)
expert_ids, weights = topk(scores, k)
expert_inputs = dispatch(hidden, expert_ids, capacity)
expert_outputs = experts(expert_inputs)
hidden_out = combine(expert_outputs, expert_ids, weights)
```

The hard part is hidden in `dispatch`. Selected experts may be remote, token counts per expert are uneven, and kernels and collectives still prefer regular tensors.

## 3. Top-2 and top-1 routing

GShard uses top-2 routing. Each token sends to its two highest-scoring experts and combines the outputs with normalized gate weights. The second expert gives the model another path when the first expert is overloaded or specialized badly, but it also doubles the routed-token traffic relative to top-1.

Switch uses top-1 routing: one token, one expert. This is simpler, cheaper, and easier to scale, but it puts more pressure on the router to avoid collapse.

For hidden size `H`, dtype size `b`, `T` tokens, and top-k `k`, a rough dispatch plus combine byte count is:

```text
bytes ~= 2 * T * k * H * b
```

The leading 2 is dispatch and return. This back-of-the-envelope estimate is crude, but it explains why MoE systems often become communication-bound even though compute is sparse.

## 4. Capacity turns routing into shapes

The gate can send many tokens to one expert. Hardware does not like arbitrary ragged tensors. GShard introduces an expert capacity:

```text
capacity = max(tokens_per_group / num_experts * k * capacity_factor, min_capacity)
```

Each expert gets a fixed-size buffer. If too few tokens arrive, the buffer is padded. If too many arrive, the implementation drops, reroutes, or otherwise handles overflow.

![Top-2 routing: probability, capacity, overflow](gate_top2_capacity.svg)

This is the first important systems tradeoff:

- higher capacity drops fewer tokens but sends more padding;
- lower capacity uses memory and bandwidth better but risks lost expert assignments;
- better load balance improves both quality and throughput.

Capacity is not just a training hyperparameter. It is the contract that turns irregular routing into tensors the runtime can move.

## 5. Load balance is part of correctness

The router can collapse.
If many tokens choose a few experts, three things happen:

1. overloaded experts hit capacity and drop or reroute tokens;
2. underused experts waste parameters and compute slots;
3. All-to-All traffic becomes imbalanced.

Auxiliary load-balancing losses push the gate toward more even expert use.
The exact formulas differ across Shazeer MoE, GShard, Switch, and later systems, but the engineering goal is the same.
You want the router to learn specialization without turning the cluster into a hot-spot machine.

That is why MoE quality and MoE throughput are tied together.
Bad balance is not only a modeling issue.
It becomes padding, dropped tokens, and uneven network traffic.

## 6. Expert Parallelism

Expert Parallelism means different ranks own different experts.
If there are 64 experts and `ep_size = 8`, each EP rank may own 8 experts.
Tokens start on data ranks, not expert ranks.
The MoE layer must move tokens to the ranks that own their selected experts.

![Expert Parallel plus Data Parallel](ep_dp_layout.svg)

EP is not DP with a new name.

- DP asks: which ranks own equivalent parameters and should average gradients?
- EP asks: which ranks own different experts and should exchange tokens?

In a dense DP layer, every rank runs the same parameters on different data.
In an EP group, ranks own different expert parameters for the same sparse layer replica.
That difference is why EP introduces All-to-All.

## 7. EP plus DP

Large jobs usually need both axes.
One EP group hosts one full set of experts.
Multiple DP replicas of that EP group process different batches.

```text
world_size = dp_size * ep_size
experts_per_ep_rank = num_experts / ep_size
```

Two communication patterns appear:

- token dispatch and combine inside the EP group;
- gradient synchronization across DP replicas of the same expert shard.

The first moves activations during the MoE forward and backward path.
The second moves gradients during the optimizer path.
Do not confuse them.
They are different tensors, different groups, and different points in the step.

## 8. All-to-All dispatch

All-to-All is the natural EP collective.
Every source rank may have tokens for every destination rank.
Every destination rank may need tokens from every source rank.

![All-to-All dispatch and combine](all_to_all_dispatch.svg)

A standard forward path is:

1. gate computes expert ids and combine weights for local tokens;
2. tokens are permuted into destination buckets;
3. All-to-All sends buckets to expert-owning ranks;
4. local experts run FFNs on received tokens;
5. a second All-to-All returns expert outputs to source ranks;
6. outputs are unpermuted and combined by gate weights.

This double All-to-All is the defining cost of distributed MoE.
Modern systems spend a lot of engineering effort on faster token permutation, fused dispatch, overlap, and topology-aware expert placement.
Those are the next layer after the foundation in [Large MoE Performance](../large-moe-from-sparsity-to-communication/).

## 9. Why All-to-All is hard

All-to-All is not just "send bytes."
It creates several practical problems:

- small messages when few tokens target a destination;
- load imbalance when a few experts receive many tokens;
- permutation overhead before and after communication;
- padding overhead from fixed capacity;
- cross-node latency when EP spans slow network tiers;
- memory pressure from top-k copies and dispatch buffers.

Capacity regularizes message sizes, but padding sends zeros.
Dropless routing preserves tokens, but shapes become dynamic.
Top-2 can improve quality, but it doubles expert traffic relative to top-1.
There is no free setting.
The right choice depends on model scale, token count per device, network topology, and quality target.

## 10. EP plus TP

Experts are FFNs.
An expert can itself be tensor-parallel.
Then the job has EP plus TP:

```text
EP: which rank owns which expert?
TP: how is each expert's matrix split?
```

![EP + DP + TP: dense and sparse dimensions coexist](ep_dp_tp.svg)

This is useful when individual experts are large or when the dense transformer already uses TP.
It also nests communication.
Around the expert, EP moves tokens with All-to-All.
Inside the expert, TP uses the column-parallel and row-parallel collectives described in [Megatron Internals II](../megatron-model-parallel-internals/).

The schedule matters.
If EP All-to-All and TP all-reduce fight over the same cross-node links, sparse compute will not save you.

## 11. Where pipeline parallelism sits

Pipeline parallelism splits the layer stack.
MoE does not remove that axis.
A pipeline stage may own dense layers and MoE layers.
Inside the stage, MoE layers use EP and maybe TP.
Between stages, PP sends activations forward and activation gradients backward.

PP can help memory by splitting layers.
It can also help topology by keeping EP groups local to a node or rack.
But it adds pipeline bubbles and scheduling constraints.
In MoE training, PP is often as much a placement lever as a memory lever.

## 12. The foundational mental model

MoE parallelism is sparse compute plus dense communication:

```text
gate -> capacity -> token permutation -> All-to-All
     -> local experts -> All-to-All -> combine
```

When reading a paper or codebase, ask five questions:

1. Is routing top-1, top-2, or something else?
2. What happens when an expert exceeds capacity?
3. Which ranks form the EP group?
4. Which ranks synchronize gradients for the same expert?
5. Does expert compute also use TP, and where does that communication land?

Those questions make the implementation legible.
The next post applies them to DeepSpeed-Megatron's MoE path and contrasts that design with Megatron SwitchMLP-style integration.

## Code

- DeepSpeed MoE layer wrapper: [`deepspeed/moe/layer.py`](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/moe/layer.py).
- DeepSpeed sharded MoE routing and dispatch code: [`deepspeed/moe/sharded_moe.py`](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/moe/sharded_moe.py).
- DeepSpeed MoE communication mappings: [`deepspeed/moe/mappings.py`](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/moe/mappings.py).
- DeepSpeed MoE package: [`deepspeed/moe/`](https://github.com/deepspeedai/DeepSpeed/tree/master/deepspeed/moe).
- Megatron-Core MoE modules: [`megatron/core/transformer/moe/`](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe).
- Megatron-Core token dispatchers: [`megatron/core/transformer/moe/token_dispatcher.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/token_dispatcher.py).

## References

- Shazeer et al., [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538), ICLR 2017.
- Lepikhin et al., [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668), ICLR 2021.
- Fedus, Zoph, Shazeer, [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961), JMLR 2022.
- Du et al., [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905), ICML 2022.
- Rajbhandari et al., [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://arxiv.org/abs/2201.05596), ICML 2022.
