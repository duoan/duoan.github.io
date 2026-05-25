# LLM-training experiments

Training-time optimizations on a single node: AMP, activation checkpointing,
fused optimizers, gradient accumulation, attention variants, sequence
packing. Most experiments here use a small synthetic transformer so the
result is the *delta* from each knob, not absolute numbers.

## Index

_(no experiments yet — this directory is a placeholder.)_

## Planned

| Slug                          | Question                                                    |
|-------------------------------|-------------------------------------------------------------|
| `01-amp-fp16-vs-bf16`         | Tokens/s and loss curve for fp32 vs. fp16 vs. bf16 on the same model. |
| `02-activation-checkpointing` | At what model size does checkpointing pay off (memory vs. throughput)? |
| `03-fused-optim-step`         | `torch.optim.AdamW(fused=True)` vs. unfused: how much of step time is the optimizer? |
| `04-seq-packing`              | Tokens/s with and without sequence packing on a varlen distribution. |

## Patterns

- Synthetic data + tiny model is the default. Real datasets belong in
  `../data-infra/`, not here.
- Capture *both* throughput (tokens/s) and a sanity metric (training loss
  at step N) so you notice when an optimization silently breaks the model.
- Lock the seed and the data order. Otherwise your "+8% tokens/s" might be
  noise.
