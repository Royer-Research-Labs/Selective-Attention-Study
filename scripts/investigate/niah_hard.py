"""Niah hard mode (natural filler, keyed distractors; niah.hard in the Niah harness) on harness checkpoints.

    PYTHONPATH=src python scripts/investigate/niah_hard.py --filler data/niah_filler/pg19_train6.txt \
        --model sel_bos_s1337=runs/.../checkpoint_step_0064000.pt [--model ...] \
        [--lengths 4096 16000 64000] [--samples 10] [--distractors 3] --output results/investigate/niah_hard.json

Same scoring as selective_attention.niah_eval (likelihood over 8 choices, control prior, BOS for models
trained with one; past the trained context the context cap is raised, RoPE models with NTK scaling).
Rows are recorded, so per-depth accuracy and distractor picks can be read back.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from selective_attention.extend import ExtendedContext
from selective_attention.niah_eval import (
    _autocast_ctx, _ensure_niah_importable, _trained_with_bos, checkpoint_identity, harness_provenance,
    load_model_for_niah)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=CHECKPOINT")
    ap.add_argument("--filler", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 16000, 64000])
    ap.add_argument("--samples", type=int, default=10, help="placements (depths) per needle")
    ap.add_argument("--num-needles", type=int, default=8)
    ap.add_argument("--distractors", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    _ensure_niah_importable()
    from niah.eval import NiahConfig, run_niah
    from niah.tokenizer import load_tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer("gpt2")
    out = Path(a.output)
    result = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {
        "task": "niah_hard", "harness": harness_provenance(), "argv": sys.argv[1:], "models": {}}
    for spec in a.model:
        name, _, ckpt = spec.partition("=")
        model = load_model_for_niah(ckpt, device)
        has_rope = any(type(m).__name__ == "RopeModule" for m in model.modules())
        entry = result["models"].setdefault(name, {"checkpoint": checkpoint_identity(
            ckpt, getattr(model, "checkpoint_step", None)), "trained_context": model.config.context_length,
            "by_length": {}})
        for length in a.lengths:
            if str(length) in entry["by_length"]:
                continue
            cfg = NiahConfig(context_length=length, num_needles=a.num_needles, placements_per_needle=a.samples,
                             seed=a.seed, use_shared_choices=True, canonical_tokenizer="gpt2", record_rows=True,
                             prepend_bos=_trained_with_bos(ckpt), mode="hard", filler_path=a.filler,
                             num_distractors=a.distractors)
            t0 = time.time()
            # +8: room for the BOS token in front of models trained with one (the input itself is `length`)
            with torch.no_grad(), ExtendedContext(model, length + 8, has_rope):
                metrics = run_niah(model, config=cfg, tokenizer=tok, device=str(device),
                                   ctx=_autocast_ctx(device, "bfloat16"))
            if "error" in metrics:
                print(f"[hard] {name} {length}: ERROR {metrics['error']}", flush=True)
                continue
            metrics["ntk"] = bool(has_rope and length > entry["trained_context"])
            entry["by_length"][str(length)] = metrics
            print(f"[hard] {name} {length:>7}: acc {metrics['accuracy']:.3f}  distractor picks "
                  f"{metrics['distractor_pick_rate']:.3f}  early/mid/late {metrics['early_accuracy']:.2f}/"
                  f"{metrics['mid_accuracy']:.2f}/{metrics['late_accuracy']:.2f}  n={metrics['total']}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
