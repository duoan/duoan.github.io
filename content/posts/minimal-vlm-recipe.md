---
title: "A Minimal Recipe for Training VLMs Under Compute Constraints"
date: 2025-12-15
tags: ["VLM", "Multimodal", "Compute Constraints", "Recipe", "Engineering"]
categories: ["Engineering"]
draft: true
---

> **Superseded.** This note is folded into [SiQ-VL: A Curriculum for Small VLMs When Compute Is the Hard Constraint](/posts/siq-vl-curriculum-under-compute-constraints/). Kept as draft for git history only.

## The Problem Nobody Likes to Admit

Most Vision-Language Models assume one thing:

> **You can always add more GPUs.**

But many of us can’t.

When compute is the hard constraint, the question changes from  
*“How do we scale?”* to:

> *“What actually still works?”*

Based on building **SiQ-VL**, here is a **minimal, battle-tested recipe** for training VLMs when GPUs are scarce.

---

## The Minimal Recipe (TL;DR)

If you remember nothing else, remember this:

1. Freeze aggressively  
2. Reduce vision tokens early  
3. Train in stages, not end-to-end  
4. Inject reasoning explicitly  
5. Optimize for stability, not peak metrics  

Everything else is secondary.

---

## 1. Freeze Aggressively (More Than Feels Comfortable)

Freezing is not a compromise — it’s a strategy.

- Freeze the vision encoder entirely
- Freeze the language model at the beginning
- Only unfreeze when you know *why* you’re doing it

Why this works:

- Fewer trainable parameters → fewer failure modes
- Lower memory usage → fewer OOM surprises
- Clearer debugging when things go wrong

If you’re GPU-poor, **freezing is your leverage**.

---

## 2. Reduce Vision Tokens Before They Hit the LLM

Vision encoders are expensive because of **sequence length**, not parameters.

A simple rule of thumb:
> If you feed 700+ visual tokens into an LLM, you will pay for it.

Do token reduction *before* multimodal fusion:

- pixel shuffle
- spatial merging
- lightweight projection

Cutting vision tokens by 4× often matters more than any optimizer tweak.

---

## 3. Never Train Alignment, Instruction, and Reasoning Together

This is the most common failure pattern.

Small VLMs collapse when asked to:

- align modalities,
- follow instructions,
- and reason — all at once.

Instead, **separate concerns**:

- **Stage 1**: modality alignment (projector only)
- **Stage 2**: instruction following (LLM adapts)
- **Stage 3**: reasoning (explicit supervision)

Each stage solves exactly one problem.

---

## 4. Reasoning Will Not “Emerge” — You Must Inject It

Small models do not discover reasoning on their own.

If you want reasoning:

- provide reasoning traces
- control their length
- decide when they appear in training

Offline Chain-of-Thought distillation is often the only affordable option:

- no online teachers
- no RL loops
- no massive context windows

Reasoning is **data**, not magic.

---

## 5. Use Multiple Weak Teachers, Not One Perfect One

A single strong teacher gives you:

- one reasoning style
- one bias
- one failure mode

Multiple teachers give you:

- diversity
- robustness
- better generalization

Even small, imperfect teachers can be valuable if their biases differ.

Think in terms of **ensemble supervision**, not oracle imitation.

---

## 6. Optimize for Stability, Not Peak Scores

Under compute constraints:

- long training runs are risky
- hyperparameter sweeps are expensive
- crashes cost real time

Prioritize:

- stable loss curves
- predictable gradients
- repeatable runs

A slightly worse model that trains reliably beats a fragile one every time.

---

## 7. Evaluate Reasoning With Your Eyes, Not Just Metrics

Many reasoning improvements do **not** show up in:

- ROUGE
- exact match
- short-answer accuracy

But they *do* show up in:

- structured explanations
- reduced hallucinations
- consistent step ordering

For small VLMs, **qualitative inspection matters**.

---

## The Real Takeaway

Training VLMs under compute constraints is not about clever tricks.

It’s about discipline:

- knowing what to freeze,
- knowing when to unfreeze,
- and knowing which problems to solve first.

If you respect those constraints, small models can still surprise you.

---

## Final Checklist

Before you start another VLM experiment, ask yourself:

- Do I really need to unfreeze this?
- Can I reduce tokens earlier?
- What single capability am I training *right now*?
- Where does reasoning supervision actually come from?

If you can answer those, you’re ready to train — even without big GPUs.

---

Built from lessons learned in the SiQ-VL project.  
Repo: <https://github.com/duoan/SiQ_VL>
