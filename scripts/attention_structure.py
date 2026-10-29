"""Measure what a trained attention layer actually does: diagonal dominance, sink mass,
column-load concentration and attended-token age.

    PYTHONPATH=src python scripts/attention_structure.py runs/<run> [--seq 1024] [--layers 0 5 11]

Reconstructs the attention matrix per layer on a real validation batch through the eager
path (so it works for any operator/bid/share mode) and reports, averaged over heads:

    diag        mean weight a query puts on itself
    first       mean weight on token 0 (the attention-sink position)
    top1/top8   mean mass in the largest 1 / 8 weights per row (selectivity)
    entropy     mean row entropy in nats (ln T is uniform)
    age         mean (query index - attended index), in tokens, as a fraction of the row's span
    col_gini    Gini of per-key received mass (0 = every key used equally, 1 = one key takes all)

The off_* columns repeat the selectivity and load statistics with the diagonal removed
and each row renormalised. They exist because at layer 0 a query's top weight is usually
itself, so top1/col_gini there partly measure self-attention rather than retrieval:

    off_mass    fraction of each row's mass that is NOT on the diagonal
    off_top1    largest off-diagonal weight, as a fraction of the off-diagonal mass
    off_entropy entropy of the renormalised off-diagonal row
    off_gini    Gini of per-key load excluding self-attention
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import yaml

from selective_attention.config import ModelConfig, build_model
from selective_attention.runtime import create_token_stream, load_checkpoint, resolve_runtime


def attention_rows(model, block, x, seq):
    """(heads, seq, seq) attention for one block, via the eager path."""
    attn_mod = block.attn
    q, k, v = attn_mod._split_heads(attn_mod.ln_in(x) if hasattr(attn_mod, "ln_in") else x)
    head_dim = q.shape[-1]
    aff = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(head_dim)
    causal = torch.ones(seq, seq, device=x.device, dtype=torch.bool).tril()
    if type(attn_mod).__name__ == "CausalSelfAttention":
        return aff.masked_fill(~causal, float("-inf")).softmax(-1)[0]
    if type(attn_mod).__name__ == "SelectiveAttention":  # softmax(L - F); groups of heads share F
        from selective_attention.kernels import selective_logits
        b, h, t, d = q.shape
        g = attn_mod.groups
        logits = selective_logits(q.reshape(b * g, h // g, t, d), k.reshape(b * g, h // g, t, d),
                                  attn_mod.exempt)[0]
        return logits.reshape(b, h, t, t).softmax(-1)[0]
    from selective_attention.kernels_bid import bids_from_affinity
    p = bids_from_affinity(aff, causal, attn_mod.bid, attn_mod.context_length)
    sc = attn_mod.share_cumsum
    if not sc:
        return p[0]
    committed = p.cumsum(dim=-2) - p
    denom = committed if sc == "exclusive" else committed + p
    share = p / (denom + 1e-9)
    return (share / (share.sum(-1, keepdim=True) + 1e-9))[0]


def gini(x):
    x = torch.sort(x)[0]
    n = x.numel()
    idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    return ((2 * idx - n - 1) * x).sum() / (n * x.sum().clamp(min=1e-9))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--layers", type=int, nargs="+")
    ap.add_argument("--output")
    a = ap.parse_args()
    run = Path(a.run)
    cfg = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8"))
    runtime = resolve_runtime("auto", "auto")
    ckpt = load_checkpoint(run / "latest.pt", map_location="cpu")
    model_config = ModelConfig.from_dict(dict(ckpt["model_config"]))
    model = build_model(model_config)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(runtime.device).eval()

    stream = create_token_stream(cfg["data"], "val")
    gen = torch.Generator().manual_seed(1337)   # fixed window, so runs are comparable
    ids, _ = stream.sample_batch(batch_size=1, sequence_length=a.seq, generator=gen,
                                 device=runtime.device)
    layers = a.layers if a.layers else [0, len(model.transformer.h) // 2, len(model.transformer.h) - 1]
    rows = []
    with torch.no_grad():
        x = model.transformer.drop(model._embed(ids, 0))
        for i, block in enumerate(model.transformer.h):
            if i in layers and hasattr(block.attn, "_split_heads"):   # attention layers only
                A = attention_rows(model, block, block.ln_1(x), a.seq).float()
                pos = torch.arange(a.seq, device=A.device, dtype=A.dtype)
                span = pos.clamp(min=1)
                diag = A.diagonal(dim1=-2, dim2=-1).mean().item()
                first = A[:, 1:, 0].mean().item()
                srt = A.sort(dim=-1, descending=True)[0]
                top1, top8 = srt[..., 0].mean().item(), srt[..., :8].sum(-1).mean().item()
                ent = (-(A.clamp(min=1e-12).log() * A).sum(-1)).mean().item()
                age = (((pos[None, :, None] - pos[None, None, :]) * A).sum(-1) / span[None, :]).mean().item()
                col = A.sum(-2).mean(0)
                # same statistics with self-attention removed and rows renormalised
                O = A - torch.diag_embed(A.diagonal(dim1=-2, dim2=-1))
                off_mass = O.sum(-1)
                On = O / off_mass.clamp(min=1e-12).unsqueeze(-1)
                osrt = On.sort(dim=-1, descending=True)[0]
                valid = off_mass > 1e-6                      # row 0 has no off-diagonal
                off_top1 = osrt[..., 0][valid].mean().item()
                off_ent = (-(On.clamp(min=1e-12).log() * On).sum(-1))[valid].mean().item()
                off_gini = gini(O.sum(-2).mean(0)).item()
                rows.append(dict(layer=i, kind=type(block.attn).__name__, diag=diag, first=first,
                                 top1=top1, top8=top8, entropy=ent, age=age, col_gini=gini(col).item(),
                                 off_mass=off_mass[valid].mean().item(), off_top1=off_top1,
                                 off_entropy=off_ent, off_gini=off_gini))
                print(f"L{i:<3d} {rows[-1]['kind']:<20s} diag {diag:.3f}  first {first:.3f}  "
                      f"top1 {top1:.3f}  entropy {ent:.2f}  col_gini {gini(col).item():.3f}  ||  "
                      f"off_top1 {off_top1:.3f}  off_entropy {off_ent:.2f}  off_gini {off_gini:.3f}", flush=True)
            x = block(x)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps(dict(run=str(run), seq=a.seq, layers=rows), indent=2) + "\n",
                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
