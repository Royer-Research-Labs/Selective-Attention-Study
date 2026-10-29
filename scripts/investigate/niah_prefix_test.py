"""Depth-0 NIAH placements with and without a few tokens before the prompt. If a short prefix fixes the
misses, the blind spot is "the first tokens of the model's input window", not "early in a document".

    PYTHONPATH=src python scripts/investigate/niah_prefix_test.py --device cuda --model s2357=...         [--length 256] [--output results/investigate/niah_prefix_test.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from selective_attention.niah_eval import _ensure_niah_importable, load_model_for_niah

PREFIXES = {"none": "", "newline": "\n", "space+newline": " \n", "one filler sentence": None, "eot": "<|endoftext|>"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--length", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", default=None, help="JSON file for the results")
    a = ap.parse_args()
    _ensure_niah_importable()
    import niah.eval as ne
    from niah.tokenizer import load_tokenizer

    tok = load_tokenizer("gpt2")
    original = ne._get_prompt
    record = dict(task="niah_prefix_test", argv=sys.argv[1:], length=a.length, num_needles=8,
                  placements_per_needle=20, seed=42, prefixes={k: v for k, v in PREFIXES.items()}, models={})
    for spec_arg in a.model:
        name, path = spec_arg.split("=", 1)
        model = load_model_for_niah(Path(path), torch.device(a.device)).eval()
        record["models"][name] = {"checkpoint": path, "results": {}}
        for label, prefix in PREFIXES.items():
            if prefix is None:
                prefix = "The office was quiet in the morning. "
            ne._get_prompt = lambda h, q, p=prefix: p + original(h, q)
            cfg = ne.NiahConfig(context_length=a.length, num_needles=8, placements_per_needle=20, seed=42,
                                use_shared_choices=True, canonical_tokenizer="gpt2", record_rows=True)
            try:
                res = ne.run_niah(model, config=cfg, tokenizer=tok, device=a.device)
            except Exception as e:  # noqa: BLE001 - e.g. a special token the tokenizer refuses
                print(f"{name} prefix {label!r}: {type(e).__name__}: {e}"[:200], flush=True)
                record["models"][name]["results"][label] = {"error": f"{type(e).__name__}: {e}"[:200]}
                continue
            d0 = [r["pass"] for r in res["rows"] if r["depth"] == 0.0]
            print(f"{name} prefix {label!r}: overall {res['accuracy']:.3f}, depth 0.00: {sum(d0)}/{len(d0)}", flush=True)
            record["models"][name]["results"][label] = {"accuracy": res["accuracy"], "depth0_pass": sum(d0),
                                                        "depth0_total": len(d0)}
        ne._get_prompt = original
        del model
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
