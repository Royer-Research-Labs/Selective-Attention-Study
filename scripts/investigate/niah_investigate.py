"""Why do some Selective Attention seeds miss early needles? Per-placement NIAH plus a probe of head 0's
accumulated selection on the needle.

    PYTHONPATH=src python scripts/investigate/niah_investigate.py --device cpu \
        --model s1337=runs/scale_16384/all_selective_nope_seed1337/checkpoint_step_0064000.pt ... \
        [--lengths 256 1920] [--samples 20] --output results/investigate/early_needles.json

Part 1 runs the NIAH harness with record_rows=True (one record per needle x depth placement) and
prints accuracy at every one of the depth placements. Part 2 takes one prompt per (depth, needle)
and, for every Selective Attention layer, measures at the query that predicts the answer:
  F_answer   the selection the answer token has accumulated from all earlier queries (subtracted
             from every head's logit there)
  F_rank     its rank among all keys (0 = the most-suppressed key in the prompt)
  attn_max   the largest weight any head puts on the answer token
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from selective_attention.niah_eval import _ensure_niah_importable, _trained_with_bos, load_model_for_niah


@torch.no_grad()
def probe_prompt(model, ids: list[int], answer_pos: int) -> list[dict]:
    """Per-layer selection and attention on the answer token, for the last query of ids."""
    idx = torch.tensor([ids], device=next(model.parameters()).device)
    x = model.transformer.drop(model._embed(idx, 0))
    out = []
    for li, block in enumerate(model.transformer.h):
        h = block.ln_1(x)
        if type(block.attn).__name__ == "SelectiveAttention":
            q, k, _ = block.attn._split_heads(h)
            d = q.shape[-1]
            l0 = (q[0, 0].float() @ k[0, 0].float().T) / math.sqrt(d)             # (T, T) head 0
            T = l0.shape[0]
            causal = torch.ones(T, T, dtype=torch.bool, device=l0.device).tril()
            s = torch.relu(l0.masked_fill(~causal, 0.0))
            s.fill_diagonal_(0.0)
            s[:, 0] = 0.0                                                       # BOS exempt
            f_last = s[:-1].sum(dim=0)                                          # from queries before the last
            logits = torch.einsum("hd,htd->ht", q[0, :, -1].float(), k[0].float()) / math.sqrt(d)   # (H, T)
            attn = (logits - f_last).softmax(dim=-1)
            order = torch.argsort(f_last, descending=True)
            rank = int((order == answer_pos).nonzero()[0])
            out.append({"layer": li, "F_answer": float(f_last[answer_pos]),
                        "F_median": float(f_last[1:].median()), "F_rank": rank,
                        "attn_max": float(attn[:, answer_pos].max()),
                        "attn_head": int(attn[:, answer_pos].argmax())})
        x = block(x)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True, help="name=checkpoint.pt")
    ap.add_argument("--lengths", type=int, nargs="+", default=[256, 1920])
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--num-needles", type=int, default=8)
    ap.add_argument("--probe-length", type=int, default=1920)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    _ensure_niah_importable()
    from niah.eval import NiahConfig, run_niah
    from niah.spec import build_niah_spec, iter_niah_rows, locate_needle, make_prompt
    from niah.tokenizer import load_tokenizer

    tok = load_tokenizer("gpt2")
    device = torch.device(a.device)
    report = {"lengths": a.lengths, "samples": a.samples, "models": {}}
    for spec_arg in a.model:
        name, path = spec_arg.split("=", 1)
        model = load_model_for_niah(Path(path), device).eval()
        entry = {"checkpoint": path, "niah": {}, "probe": []}
        for length in a.lengths:
            cfg = NiahConfig(context_length=length, num_needles=a.num_needles, placements_per_needle=a.samples,
                             seed=42, use_shared_choices=True, canonical_tokenizer="gpt2", record_rows=True,
                             prepend_bos=_trained_with_bos(path))
            res = run_niah(model, config=cfg, tokenizer=tok, device=str(device))
            rows = res.get("rows", [])
            by_depth: dict[float, list[bool]] = {}
            for r in rows:
                by_depth.setdefault(r["depth"], []).append(r["pass"])
            entry["niah"][length] = {"accuracy": res["accuracy"], "rows": rows,
                                     "by_depth": {f"{d:.2f}": sum(v) / len(v) for d, v in sorted(by_depth.items())}}
            misses = [(r["depth"], r["needle"], round(r["retrieval_gap_ll"], 2), r["answer_token"]) for r in rows if not r["pass"]]
            print(f"{name} ctx {length}: acc {res['accuracy']:.3f}; by depth " +
                  " ".join(f"{d}:{v:.2f}" for d, v in entry["niah"][length]["by_depth"].items() if v < 1.0) +
                  f" | misses {misses}", flush=True)
        # probe: every needle at the first two depths and at mid depth, at the probe length
        spec = build_niah_spec(tokenizer_name="gpt2", tokenizer=tok, use_shared_choices=True,
                               num_samples=a.samples, context_length=a.probe_length, n_choices=a.num_needles, seed=42)
        for row in iter_niah_rows(spec, tokenizer=tok):
            if row["depth"] not in ("0.00", "0.05", "0.53"):
                continue
            ids = tok.encode(make_prompt(row["haystack"], spec["question"]))
            if _trained_with_bos(path):   # the model saw a BOS at position 0 in training and in NIAH
                ids = [tok.eot_token_id] + ids
            loc = locate_needle(ids, row["needle"], spec, tok)
            if loc is None:
                continue
            layers = probe_prompt(model, ids, loc[2])
            entry["probe"].append({"depth": row["depth"], "needle": row["needle"], "answer_token": loc[2],
                                   "prompt_tokens": len(ids), "layers": layers})
        entry["probe_summary"] = {}
        for depth in ("0.00", "0.05", "0.53"):
            probes = [p for p in entry["probe"] if p["depth"] == depth]
            if not probes:
                continue
            n_layers = len(probes[0]["layers"])
            summ = []
            for li in range(n_layers):
                vals = [p["layers"][li] for p in probes]
                summ.append({"layer": vals[0]["layer"],
                             "F_answer": sum(v["F_answer"] for v in vals) / len(vals),
                             "F_median": sum(v["F_median"] for v in vals) / len(vals),
                             "F_rank_mean": sum(v["F_rank"] for v in vals) / len(vals),
                             "attn_max": sum(v["attn_max"] for v in vals) / len(vals)})
            entry["probe_summary"][depth] = summ
            top = sorted(summ, key=lambda r: -r["attn_max"])[:3]
            print(f"{name} probe depth {depth}: answer-token attention (best layers) " +
                  ", ".join(f"L{r['layer']} {r['attn_max']:.2f} (F {r['F_answer']:.1f} vs median {r['F_median']:.1f}, rank {r['F_rank_mean']:.0f})" for r in top),
                  flush=True)
        report["models"][name] = entry
        del model
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
