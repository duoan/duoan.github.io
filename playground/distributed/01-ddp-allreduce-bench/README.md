# 01 - DDP all-reduce bandwidth

## Question

For NCCL `all_reduce` on this box, how does achieved algorithmic bandwidth
scale with message size, and where does it saturate?

## Hypothesis

- Small messages are latency-bound: bandwidth grows linearly with size.
- Once messages cross ~1 MiB, ring all-reduce saturates and we measure a
  flat plateau set by the slowest interconnect on the path.
- On a single-GPU workstation with `nproc_per_node=2` (both ranks sharing
  device 0), the "interconnect" is actually device-local memcpy, so the
  plateau will be much higher than what we'd see on a real multi-GPU box
  with PCIe-only links. That contrast is the point.

## Run

```bash
./scripts/snapshot_env.sh > env.txt
./scripts/run.sh                     # default: nproc_per_node=2
NPROC=4 ./scripts/run.sh             # override world size
```

Outputs (rank 0 only):

- `results/timings_ws<N>.csv` — per-size time + algorithmic bandwidth.
- `results/throughput_ws<N>.svg` — algbw vs. message size.

## Results

_(populated after first run)_

## Follow-ups

- Re-run on a real multi-GPU machine and compare the plateau.
- Add a `reduce_scatter` + `all_gather` curve to confirm equivalence with
  `all_reduce` on ring.
- Vary dtype (fp32 / fp16 / bf16) — does NCCL hit the same algbw for all?
