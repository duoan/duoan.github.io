---
title: "MoE Internals: DeepSpeed-Megatron Expert Parallel Implementation"
date: 2025-05-17
tags: ["LLM", "MoE", "DeepSpeed", "Megatron", "Expert Parallel", "Distributed Training"]
categories: ["Engineering"]
draft: false
series: ["LLM Training"]
cover:
  image: moe_init_flow.svg
  alt: "DeepSpeed-Megatron MoE initialization flow creating expert-parallel groups during model wrapping"
  relative: true
---

# MoE Internals: DeepSpeed-Megatron Expert Parallel Implementation

DeepSpeed-Megatron MoE is interesting because it does not put every MoE decision in Megatron's base distributed initialization.
Megatron builds the usual TP, PP, and DP topology.
The expert-parallel topology is attached later, when `deepspeed.initialize()` walks the model and gives MoE modules their DeepSpeed-specific process groups.
This post explains that implementation shape conceptually: where EP groups come from, what `MoELayer` contains, and how this differs from Megatron's own SwitchMLP style.
For the routing and all-to-all foundation, start with [MoE Parallelism Principles](../moe-expert-parallelism-principles/).

## TL;DR

- DeepSpeed-Megatron reuses Megatron's training skeleton, but expert-parallel setup is triggered inside DeepSpeed model wrapping.
- Base Megatron initialization creates TP, PP, and DP groups; DeepSpeed MoE modules create EP and expert-DP groups when `set_deepspeed_parallelism` is called.
- The MoE layer wraps a gate, dispatch/combine metadata, expert modules, and All-to-All movement.
- In Megatron-DeepSpeed, an expert is often a `ParallelMLP`, so expert compute can still use tensor-parallel layer contracts.
- DeepSpeed's MoE path is GShard-like: TopK gate, capacity, token dropping or padding, All-to-All, local experts, and combine.
- Megatron's SwitchMLP path is a separate design line, with different assumptions about routing and integration.

## 1. The implementation boundary

The base Megatron flow is:

1. initialize distributed process groups;
2. build the model for the current rank;
3. build optimizer and scheduler;
4. build data iterators;
5. run training.

DeepSpeed-Megatron keeps that skeleton.
The change is that model construction may create MoE modules, and DeepSpeed wrapping may attach extra parallel metadata to those modules.
This makes the implementation boundary easy to miss.
If you inspect only Megatron's distributed initialization, you may find TP, PP, and DP groups but no complete EP topology.
That does not mean EP is absent.
It means DeepSpeed defers the MoE-specific part until it has the model modules in hand.

![DeepSpeed-Megatron MoE initialization flow](moe_init_flow.svg)

## 2. Why defer EP setup?

Base TP, PP, and DP groups are global training topology.
Every transformer layer cares about them.
Expert-parallel groups are needed only by MoE modules.
DeepSpeed owns the MoE module implementation, so it can attach EP groups when it wraps the model:

```python
for module in model.modules():
    if hasattr(module, "set_deepspeed_parallelism"):
        module.set_deepspeed_parallelism(use_data_before_expert_parallel)
```

This is a design choice, not a mathematical requirement.
Megatron-Core style systems can build expert groups in the main parallel-state initialization.
DeepSpeed's path keeps MoE behavior close to DeepSpeed's engine and module wrappers.
The advantage is modularity.
The cost is that understanding topology requires following both Megatron and DeepSpeed code paths.

## 3. Script-level knobs

A typical MoE launch config includes:

```text
--tensor-model-parallel-size TP
--pipeline-model-parallel-size PP
--moe-expert-parallel-size EP
--num-experts E
--moe-train-capacity-factor C_train
--moe-eval-capacity-factor C_eval
--moe-min-capacity C_min
--moe-loss-coeff aux_coeff
--topk k
```

`tensor-model-parallel-size` controls dense tensor splits and optionally expert-internal splits.
`moe-expert-parallel-size` controls how many ranks cooperate to host a full expert set.
`num-experts` controls how many experts each MoE layer contains.
Capacity and auxiliary-loss knobs control router regularization and overflow behavior.
These flags describe a geometry.
The implementation turns that geometry into groups and tensor layouts.

## 4. Base Megatron groups still matter

Even when DeepSpeed attaches EP groups later, the base Megatron groups are still active.
The attention block uses tensor-parallel collectives as described in [Megatron Internals II](../megatron-model-parallel-internals/).
Pipeline stages still determine whether a rank owns early, middle, or late transformer layers as described in [Megatron Internals I](../megatron-distributed-init/).
Mixed precision still needs distributed overflow checks and gradient clipping as described in [Megatron Internals III](../megatron-mixed-precision-training/).
MoE adds an axis.
It does not erase the others.
That is why DeepSpeed-Megatron MoE code often feels layered:

```text
Megatron rank topology
  -> transformer layer construction
     -> DeepSpeed MoE module
        -> expert parallel groups
        -> all-to-all routing
```

## 5. Where the MoE layer appears

In a transformer layer, the dense MLP slot can be filled by different implementations:

- a normal dense `ParallelMLP`;
- a DeepSpeed MoE module wrapping expert MLPs;
- a Megatron SwitchMLP-style module in codebases that include it.

The DeepSpeed path is selected when the layer has more than one expert and the configuration requests DeepSpeed MoE.
Conceptually:

```python
if num_experts <= 1:
    mlp = ParallelMLP(config)
elif use_megatron_switch:
    mlp = SwitchMLP(config)
else:
    expert = ParallelMLP(config, moe=True)
    mlp = deepspeed.moe.layer.MoE(
        hidden_size=config.hidden_size,
        expert=expert,
        num_experts=num_experts,
        ep_size=moe_expert_parallel_size,
        k=topk,
        capacity_factor=train_capacity_factor,
        eval_capacity_factor=eval_capacity_factor,
        min_capacity=min_capacity,
    )
```

The important part is that the expert can itself be a Megatron-style parallel MLP.
So the sparse MoE wrapper and dense TP layer contracts compose.

## 6. Inside `MoELayer`

A GShard-like MoE layer needs four logical pieces:

1. a gate that computes expert scores and top-k assignments;
2. metadata that maps tokens to expert-capacity slots;
3. expert modules that compute local FFNs;
4. dispatch and combine collectives.

![DeepSpeed MoELayer structure](moe_layer_structure.svg)

The gate outputs combine weights and a dispatch mask.
The dispatch mask says which token goes to which expert slot.
The combine weights say how returned expert outputs are weighted back into the original token order.
This is the same conceptual flow as the principles post:

```text
hidden -> gate -> dispatch mask
       -> All-to-All -> local experts
       -> All-to-All -> combine -> output
```

DeepSpeed can optionally use optimized dispatch implementations such as Tutel in some configurations, but the logical contract remains the same.

## 7. EP group versus expert-DP group

Two group types are easy to confuse.
An **EP group** contains ranks that own different experts for one MoE layer replica.
It is used for token dispatch and combine.
An **expert-DP group** contains ranks that own the same expert shard across data replicas.
It is used for gradient synchronization of expert parameters.

![EP groups and DP groups answer different questions](ep_group_vs_dp.svg)

A useful phrasing:

```text
EP group:  route tokens across different experts.
EDP group: synchronize gradients for the same expert.
```

The same physical rank participates in both.
The collectives happen at different times and move different tensors.
Routing moves activations.
Gradient synchronization moves parameter gradients.

## 8. The All-to-All path

The DeepSpeed MoE forward path follows the standard sparse layer sequence.
First, local hidden states are flattened from sequence and batch into a token dimension.
Second, the gate computes top-k expert assignments and capacity positions.
Third, tokens are permuted into expert-major order.
Fourth, an All-to-All sends tokens to the ranks that own their experts.
Fifth, local experts run.
Sixth, another All-to-All sends outputs back.
Seventh, combine weights reconstruct the original token order.
The layout choices around steps three and seven matter.
Bad permutation code can dominate small-expert MoE layers.
Bad capacity choices can send mostly padding.
Bad EP placement can push all traffic over the slowest network tier.
These are the same "three walls" that modern MoE systems attack more aggressively in production: memory, communication, and compute efficiency.
See [Large MoE Performance](../large-moe-from-sparsity-to-communication/) for that later optimization layer.

## 9. Expert tensor parallelism

If `enable_expert_tensor_parallelism` is set, each expert MLP can use tensor-parallel linear layers.
That means an expert is not necessarily fully local to one rank.
Instead, an expert can be spread across a TP subgroup inside or alongside EP.
This composition is powerful but subtle.
There are two levels of "parallel":

- EP chooses which rank group owns which experts and routes tokens there.
- TP splits matrix multiplies inside each expert.

The expert's own forward pass may therefore include the column-parallel and row-parallel collectives from [Megatron Internals II](../megatron-model-parallel-internals/).
The MoE wrapper must deliver expert inputs in the layout those expert modules expect.

## 10. Capacity and auxiliary loss in implementation terms

The router returns more than expert ids.
It returns tensors that make sparse routing executable.
Common outputs include:

- `l_aux`: auxiliary load-balancing loss;
- `combine_weights`: weights for final token reconstruction;
- `dispatch_mask`: boolean or sparse metadata for token-to-slot placement;
- expert counts or capacity metadata for dispatch.

The training loss becomes:

```text
loss_total = loss_lm + moe_loss_coeff * l_aux
```

The auxiliary loss is not an implementation detail.
Without it, the router can collapse onto a few experts.
Collapsed routing causes worse quality and worse systems behavior: overloaded experts, padding elsewhere, and imbalanced All-to-All.
Capacity protects the system from unbounded shape growth.
Auxiliary loss trains the model away from repeatedly hitting that guardrail.

## 11. Contrast with Megatron SwitchMLP

Megatron's SwitchMLP-style path is conceptually closer to Switch Transformer top-1 routing.
The DeepSpeed path is closer to GShard top-k MoE with DeepSpeed-owned dispatch and grouping.
The difference is not just the value of `k`.
It changes the module boundary:

- DeepSpeed MoE wraps an expert module and attaches DeepSpeed parallelism.
- SwitchMLP tends to be integrated more directly into Megatron's model code.
- DeepSpeed emphasizes engine-level wrapping and optional MoE-specific runtime features.
- Megatron-integrated paths can keep more topology in one parallel-state system.

Neither style is universally better.
DeepSpeed's modularity makes it easier to insert MoE into a Megatron training stack.
Megatron-integrated MoE can be easier to reason about once the whole stack is standardized.
For production-scale MoE, modern Megatron-Core designs go much further with specialized dispatchers, overlap, and parallel folding.
That is beyond this internals post and belongs with the performance discussion.

## 12. The compact mental model

DeepSpeed-Megatron MoE is a layered system:

Megatron initializes dense-model parallelism, builds transformer layers, DeepSpeed wraps MoE modules, MoE modules attach EP and expert-DP groups, forward uses gate -> dispatch -> experts -> combine, and backward synchronizes expert gradients with the right replicated peers.
This explains why reading only one repository or one initialization function is misleading.
The topology is split across the base training framework and the MoE module wrapper.
Once that split is clear, the implementation becomes much less mysterious.
The key question for every line of code is: is this about dense transformer ownership, expert ownership, token movement, or replicated-gradient synchronization?

## References

- Rajbhandari et al., [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://arxiv.org/abs/2201.05596), ICML 2022.
- Lepikhin et al., [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668), ICLR 2021.
- Fedus, Zoph, Shazeer, [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961), JMLR 2022.
- Shoeybi et al., [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053), 2019.
- Microsoft, [DeepSpeed MoE documentation](https://www.deepspeed.ai/tutorials/mixture-of-experts/) and [DeepSpeed repository](https://github.com/microsoft/DeepSpeed).
