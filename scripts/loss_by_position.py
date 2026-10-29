"""Validation loss by position inside the trained window, for several checkpoints on the same slice.

    PYTHONPATH=src python scripts/loss_by_position.py \
        --model selective_s1337=runs/scale_16384/all_selective_nope_seed1337/checkpoint_step_0064000.pt \
        --model dilution_s2357=../dilution-attention-runs/.../latest_1x.pt \
        --data-from runs/scale_16384/all_selective_nope_seed1337 [--batches 64] [--batch-size 2] \
        --output results/position/loss_by_position_16k.json

Every model scores the same sequences: the first `batches x batch-size` windows of the validation
stream that scripts/eval_checkpoint.py uses (its common 2.1M-token slice at the defaults). Losses are
averaged within position bands that double in width; the per-sequence band means are kept so
differences between models can carry a standard error across sequences.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from selective_attention.niah_eval import load_model_for_niah
from selective_attention.runtime import autocast_context, create_token_stream, resolve_runtime

EDGES = [0, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384]


@torch.no_grad()
def token_losses(model, idx: torch.Tensor, targets: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """(B, T) per-token cross-entropy, lm_head applied per chunk (as DilutionLM.eval_chunked)."""
    x = model.transformer.drop(model._embed(idx, 0))
    for block in model.transformer.h:
        x = block(x)
    x = model.transformer.ln_f(x)
    out = []
    for start in range(0, idx.shape[1], chunk):
        logits = model.lm_head(x[:, start:start + chunk])
        out.append(F.cross_entropy(logits.float().transpose(1, 2), targets[:, start:start + chunk],
                                   reduction="none"))
        del logits
    return torch.cat(out, dim=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True, help="name=path/to/checkpoint.pt")
    ap.add_argument("--data-from", required=True, help="a run directory whose resolved_config.yaml names the data")
    ap.add_argument("--batches", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load((Path(a.data_from) / "resolved_config.yaml").read_text(encoding="utf-8"))
    seq = int(cfg["training"]["sequence_length"])
    edges = [e for e in EDGES if e < seq] + [seq]
    runtime = resolve_runtime("auto", "auto")
    result = {"sequence_length": seq, "batches": a.batches, "batch_size": a.batch_size, "bands": edges,
              "data": cfg["data"]["val_path"], "models": {}}
    for spec in a.model:
        name, path = spec.split("=", 1)
        model = load_model_for_niah(Path(path), runtime.device).eval()
        stream = create_token_stream(cfg["data"], "val")
        per_seq = []
        for idx, targets in stream.sequential_batches(batch_size=a.batch_size, sequence_length=seq,
                                                      max_batches=a.batches, device=runtime.device):
            with autocast_context(runtime):
                losses = token_losses(model, idx, targets)
            for row in losses:
                per_seq.append([float(row[lo:hi].mean()) for lo, hi in zip(edges[:-1], edges[1:])])
        means = [sum(col) / len(col) for col in zip(*per_seq)]
        overall = sum(m * (hi - lo) for m, lo, hi in zip(means, edges[:-1], edges[1:])) / seq
        result["models"][name] = {"checkpoint": path, "band_means": means, "overall": overall,
                                  "per_sequence": per_seq}
        print(f"{name:28s} overall {overall:.4f} | " +
              " ".join(f"{lo}-{hi}:{m:.3f}" for m, lo, hi in zip(means, edges[:-1], edges[1:])), flush=True)
        del model
        torch.cuda.empty_cache()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=1), encoding="utf-8")
    names = list(result["models"])
    if len(names) > 1:   # paired differences against the first model, SE across sequences
        base = result["models"][names[0]]["per_sequence"]
        for other in names[1:]:
            rows = result["models"][other]["per_sequence"]
            parts = []
            for b in range(len(edges) - 1):
                d = [r[b] - s[b] for r, s in zip(rows, base)]
                mean = sum(d) / len(d)
                se = math.sqrt(sum((x - mean) ** 2 for x in d) / max(len(d) - 1, 1) / len(d))
                parts.append(f"{mean:+.3f}±{se:.3f}")
            print(f"{other} - {names[0]}: " + " ".join(parts), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
