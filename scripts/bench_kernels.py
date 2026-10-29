"""Time and memory of one attention layer's forward + backward, by context length.

    PYTHONPATH=src python scripts/bench_kernels.py [--seq 2048 4096 8192 16384] [--batch 2]
        [--heads 12] [--head-dim 64] [--ops selective dilution softmax] [--reps 10]

"softmax" is torch's scaled_dot_product_attention (is_causal), the flash path the softmax controls
train with; "dilution" is the Dilution-Attention kernel; "selective" is this repository's kernel.
Inputs are bf16, as under training autocast. Peak memory is the allocator's high-water mark for
the forward + backward of the one layer, inputs included.
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from selective_attention import dilution_kernels
from selective_attention import kernels as selective_kernels

OPS = {
    "softmax": lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
    "dilution": lambda q, k, v: dilution_kernels.dilution_attention(q, k, v, dot_precision="bf16"),
    "selective": lambda q, k, v: selective_kernels.selective_attention(q, k, v, dot_precision="bf16"),
}


def measure(op, batch, heads, seq, head_dim, reps):
    torch.manual_seed(0)
    q, k, v = (torch.randn(batch, heads, seq, head_dim, device="cuda", dtype=torch.bfloat16,
                           requires_grad=True) for _ in range(3))
    weight = torch.randn(batch, heads, seq, head_dim, device="cuda", dtype=torch.bfloat16)

    def step():
        out = OPS[op](q, k, v)
        torch.cuda.synchronize()
        mid = time.perf_counter()
        (out.to(weight.dtype) * weight).sum().backward()
        torch.cuda.synchronize()
        q.grad = k.grad = v.grad = None
        return mid

    for _ in range(3):
        step()
    torch.cuda.reset_peak_memory_stats()
    fwd = bwd = 0.0
    for _ in range(reps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        mid = step()
        end = time.perf_counter()
        fwd += mid - start
        bwd += end - mid
    return dict(op=op, seq=seq, fwd_ms=1e3 * fwd / reps, bwd_ms=1e3 * bwd / reps,
                peak_gib=torch.cuda.max_memory_allocated() / 2**30)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--ops", nargs="+", default=list(OPS), choices=list(OPS))
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--output")
    a = ap.parse_args()
    rows = []
    for seq in a.seq:
        for op in a.ops:
            row = measure(op, a.batch, a.heads, seq, a.head_dim, a.reps)
            rows.append(row)
            print(f"T {seq:6d}  {op:9s}  fwd {row['fwd_ms']:8.2f} ms  bwd {row['bwd_ms']:8.2f} ms  "
                  f"total {row['fwd_ms'] + row['bwd_ms']:8.2f} ms  peak {row['peak_gib']:.2f} GiB", flush=True)
    if a.output:
        meta = dict(batch=a.batch, heads=a.heads, head_dim=a.head_dim, reps=a.reps,
                    gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
        with open(a.output, "w", encoding="utf-8") as fh:
            json.dump(dict(meta=meta, rows=rows), fh, indent=1)


if __name__ == "__main__":
    main()
