"""How large does Selective Attention's history term get past the trained length, and what does
it do to attention? Evaluation only, on PG-19 books.

    PYTHONPATH=src python scripts/investigate/history_probe.py --data data/pg19_gpt2 \
        --model s1337_1x=runs/.../checkpoint_step_0064000.pt --model s1337_2x=runs/.../latest.pt \
        [--books 4] [--max-length 131072] --output results/investigate/history_probe.json

For every selective layer and every doubling position band, over the first `--books` books long
enough for `--max-length` tokens:
  F_mean, F_p99, F_max   the history term F[i, j] that query i subtracts from key j's logit,
                         over keys j < i, at the first query of every 32-query block in the band
  entropy, max_weight    attention of 8 sampled queries per band (per head, then averaged):
                         Shannon entropy in nats, divided by log(i + 1), and the largest weight
Values inside the trained length are the reference for what the model saw in training.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from selective_attention import kernels
from selective_attention.niah_eval import load_model_for_niah


def bands(n: int) -> list[tuple[int, int]]:
    out, lo, hi = [], 0, 2048
    while lo < n:
        out.append((lo, min(hi, n)))
        lo, hi = hi, hi * 2
    return out


@torch.no_grad()
def probe_book(model, ids: torch.Tensor, rs) -> dict[int, list[dict]]:
    x = model.transformer.drop(model._embed(ids[None], 0))
    per_layer = {}
    for li, block in enumerate(model.transformer.h):
        attn = block.attn
        if type(attn).__name__ == "SelectiveAttention":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q, k, v = attn._split_heads(block.ln_1(x))
            _, aux = kernels.selective_attention_forward(q, k, v, dot_precision="bf16", return_aux=True,
                                                        exempt=attn.exempt)
            carry = aux["carry"][0]                                          # (T / 32, T): F at block starts
            stats = []
            T = ids.numel()
            for lo, hi in rs:
                rows = range(max(lo, 32) // 32, (hi + 31) // 32)
                vals = [carry[rb, : rb * 32] for rb in rows if rb * 32 < T]
                flat = torch.cat(vals).float() if vals else torch.zeros(1, device=ids.device)
                pick = np.linspace(max(lo, 32) // 32, (hi - 1) // 32, 8).astype(int)
                ent, mw = [], []
                for rb in pick:
                    i = int(rb) * 32
                    f = carry[rb, : i + 1].float()
                    logit = torch.einsum("hd,htd->ht", q[0, :, i].float(), k[0, :, : i + 1].float()) / math.sqrt(q.shape[-1])
                    a = (logit - f).softmax(-1)
                    ent.append(float((-(a * a.clamp_min(1e-30).log()).sum(-1) / math.log(i + 1)).mean()))
                    mw.append(float(a.max(-1).values.mean()))
                stats.append({"F_mean": float(flat.mean()), "F_p99": float(torch.quantile(flat[:: max(1, flat.numel() // 2_000_000)], 0.99)),
                              "F_max": float(flat.max()), "entropy": float(np.mean(ent)), "max_weight": float(np.mean(mw))})
            per_layer[li] = stats
            del aux, carry
        x = block(x)
    return per_layer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True, help="name=checkpoint.pt")
    ap.add_argument("--data", required=True)
    ap.add_argument("--books", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=131072)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    root = Path(a.data)
    meta = json.loads((root / "books.json").read_text(encoding="utf-8"))
    books = [b for b in (meta["books"] if isinstance(meta, dict) else meta) if b["length"] >= a.max_length][: a.books]
    data = np.memmap(root / "books.bin", dtype=np.uint16, mode="r")
    rs = bands(a.max_length)
    out = {"bands": rs, "books": [b.get("title") for b in books], "models": {}}
    for spec in a.model:
        name, path = spec.split("=", 1)
        model = load_model_for_niah(Path(path), torch.device("cuda")).eval()
        trained = model.config.context_length
        model.config.context_length = a.max_length
        from selective_attention.runtime import load_checkpoint
        resolved = load_checkpoint(path, map_location="cpu").get("resolved_config") or {}
        bos = (resolved.get("data") or {}).get("bos_token")
        acc: dict[int, list[list[dict]]] = {}
        for b in books:
            ids = torch.from_numpy(data[b["offset"]: b["offset"] + a.max_length].astype(np.int64)).cuda()
            if bos is not None:
                ids = torch.cat([ids.new_full((1,), bos), ids[:-1]])
            for li, stats in probe_book(model, ids, rs).items():
                acc.setdefault(li, []).append(stats)
        summary = {}
        for li, per_book in acc.items():
            summary[li] = [{key: float(np.mean([pb[bi][key] for pb in per_book])) for key in per_book[0][bi]}
                           for bi in range(len(rs))]
        out["models"][name] = {"checkpoint": path, "trained_context": trained, "layers": summary}
        # one line per model: the layer-average of each statistic per band
        for key in ("F_p99", "entropy", "max_weight"):
            row = [np.mean([summary[li][bi][key] for li in summary]) for bi in range(len(rs))]
            print(f"{name:14s} {key:10s} " + " ".join(f"{lo // 1024}K-{hi // 1024}K:{v:8.3f}" for (lo, hi), v in zip(rs, row)), flush=True)
        del model
        torch.cuda.empty_cache()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
