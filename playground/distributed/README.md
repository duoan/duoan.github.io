# Distributed-training experiments

DDP / FSDP / TP / PP / collective-communication ablations. Most of the
write-ups in `../../content/posts/` titled around "DDP …" or "varlen …" are
backed by experiments here.

## Index

| Slug                          | Question                                                    | Status |
|-------------------------------|-------------------------------------------------------------|--------|
| `01-ddp-allreduce-bench`      | What's the achieved algorithmic bandwidth of `all_reduce` across tensor sizes on this box? | seed   |

## Patterns

- Run via `torchrun --nproc_per_node=<N> -m src.<entry>`. The `<N>` may
  exceed the visible GPU count — extra ranks share device 0. This is
  deliberately useful for sanity-checking comm patterns on a single-GPU
  workstation.
- Always print rank-0 only when summarizing, and write CSVs *only on
  rank 0* to avoid races.
- Time collectives with `torch.cuda.Event` after a `set_device` and a
  cold-cache warmup; NCCL's first call does buffer allocation and gives
  misleadingly slow numbers.
- Compute algorithmic bandwidth as
  `2 * (N-1) / N * bytes / time` for ring all-reduce — this is what NCCL's
  own `all_reduce_perf` reports.
