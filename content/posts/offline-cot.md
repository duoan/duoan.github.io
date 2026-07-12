---
title: "What Offline CoT Distillation Taught Us About Small Vision-Language Models"
date: 2025-12-11
tags: ["VLM", "Reasoning", "Distillation", "CoT", "Multimodal"]
categories: ["Engineering", "Research"]
draft: true
---

> **Superseded.** This note is folded into [SiQ-VL: A Curriculum for Small VLMs When Compute Is the Hard Constraint](/posts/siq-vl-curriculum-under-compute-constraints/). Kept as draft for git history only.

## Reasoning Is Expensive — But It Doesn’t Have to Be

Reasoning is one of the most expensive capabilities to train in Vision-Language Models.

Most recent approaches rely on:

- very large models,
- long context windows,
- online teacher–student setups,
- or reinforcement learning.

All of these assume **abundant compute**.

In the [SiQ-VL](https://github.com/duoan/SiQ_VL/) project, we had none of that.

What we did have was a question:

> *Can a small VLM learn to reason if we only change **how** we train it?*

The answer turned out to be **yes — but only with the right constraints**.

---

## Why Small VLMs Fail at Reasoning by Default

Small Vision-Language Models usually fail at reasoning for three reasons:

1. **Modality alignment dominates early training**  
   The model spends most of its capacity just learning how images map to text.

2. **Instruction tuning ≠ reasoning**  
   Following instructions does not automatically produce multi-step reasoning.

3. **Zero-shot reasoning is too hard**  
   Without explicit supervision, small models collapse to short, heuristic answers.

This means that *reasoning must be injected deliberately*.

---

## Offline CoT Distillation Was the Only Viable Option

Running large teacher models online was simply not feasible for us:

- GPU memory was limited
- training already required aggressive accumulation
- even inference-time teachers would bottleneck experiments

So we made a hard decision early:

> **All reasoning supervision would be generated offline.**

This single decision shaped the rest of the pipeline.

Offline distillation let us:

- pre-generate reasoning traces once,
- train the student cheaply many times,
- and control reasoning length and format explicitly.

---

## Why Multiple Teachers Matter More Than Bigger Ones

We didn’t rely on a single teacher model.

Instead, we distilled from **multiple teachers**, each with a different inductive bias:

- **Qwen3-VL-Thinking** for structured, step-by-step reasoning
- **InternVL** for chart and visual analytics
- **HunyuanOCR** for text-heavy and OCR-centric reasoning

What we observed was subtle but important:

> The student didn’t copy answers — it absorbed *reasoning styles*.

Each teacher shaped different failure and success modes:

- some produced longer but noisier chains,
- others converged faster but reasoned shallowly.

This diversity mattered more than raw teacher size.

---

## Reasoning Emerges Gradually — Explosively Is a Myth

One common intuition is that reasoning “emerges” suddenly.

In practice, we observed the opposite.

During offline CoT distillation:

- loss dropped quickly,
- answers stabilized early,
- but reasoning structure improved *slowly*.

The model first learned **where** to reason,
then **how long** to reason,
and only later **how to structure reasoning steps**.

This gradual behavior suggests:
> reasoning is a curriculum problem, not a switch you flip.

---

## What Improved — and What Didn’t

Some results were expected. Others were not.

### What improved clearly

- Longer, more structured rationales
- Lower hallucination rates
- More stable running accuracy
- Better grounding in visual evidence (when available)

### What barely changed

- Surface metrics like ROUGE-L
- Short-answer accuracy on simple queries

This reinforced an uncomfortable truth:

> **Reasoning quality is poorly captured by common metrics.**

Qualitative inspection mattered far more than leaderboard-style scores.

---

## A Surprising Insight: Visual Reasoning Isn’t Always Visual

One unexpected result came from a text-only student distilled from multimodal teachers.

Even without images:

- the model learned visual priors,
- inferred likely objects,
- and reasoned probabilistically about scenes.

This suggests that:
> a significant portion of “visual commonsense” is encoded linguistically.

CoT distillation transfers more than answers — it transfers *world structure*.

---

## What Offline Distillation Is *Not* Good At

Offline CoT distillation has clear limitations:

- It cannot correct teacher hallucinations
- It inherits teacher biases
- It struggles with fine-grained perception
- It does not replace representation learning

In other words:
> distillation amplifies what’s already there — it does not create new perception.

For small VLMs, this trade-off is acceptable. For large ones, maybe not.

---

## Practical Takeaways

If you are training small or medium VLMs under tight compute budgets:

- **Don’t expect reasoning to emerge naturally**
- **Separate alignment, instruction, and reasoning stages**
- **Generate reasoning data offline**
- **Use multiple teachers with complementary strengths**
- **Evaluate reasoning qualitatively, not just numerically**

Reasoning is not free — but it *is* transferable.

---

## Closing Thoughts

SiQ-VL did not prove that small models can rival large ones.

What it did show is something more practical:

> With the right curriculum and supervision, small models can punch far above their weight.

Offline CoT distillation isn’t glamorous.
But under compute constraints, it might be the most honest tool we have.

Repo: <https://github.com/duoan/SiQ_VL>  
Tech Report: <https://github.com/duoan/SiQ_VL/blob/master/SiQ_VL_Tech_Report.pdf>
