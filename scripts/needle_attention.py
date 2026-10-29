"""Per-layer attention from the answer position onto the needle, on real NIAH prompts.

    PYTHONPATH=src python scripts/needle_attention.py runs/<run> --context 7936 --output results/needle/<name>.json

A thin adapter over the NIAH package's `needle_attention_profile`, which builds the niah-v4
rows the eval scores, locates the needle and answer token, and aggregates attention onto them
per layer and head. This script supplies the model-specific part: the last prompt position's
attention in every layer, reconstructed through the eager path
(scripts/attention_structure.attention_rows), so it works for any operator. The 1,024-token
structure probe (S20) could not separate the seeds that failed to retrieve (S22).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attention_structure import attention_rows  # noqa: E402

from selective_attention.config import ModelConfig, build_model  # noqa: E402
from selective_attention.niah_eval import _ensure_niah_importable  # noqa: E402
from selective_attention.runtime import load_checkpoint, resolve_runtime  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run")
    ap.add_argument("--context", type=int, default=7936)
    ap.add_argument("--needles", type=int, default=4)
    ap.add_argument("--depths", type=int, default=5)
    ap.add_argument("--output")
    ap.add_argument("--label", default="", help="free text (lets queue scripts' pgrep see this job)")
    a = ap.parse_args()
    _ensure_niah_importable()
    from niah.attention import needle_attention_profile

    runtime = resolve_runtime("auto", "auto")
    ckpt = load_checkpoint(Path(a.run) / "latest.pt", map_location="cpu")
    model = build_model(ModelConfig.from_dict(dict(ckpt["model_config"])))
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(runtime.device).eval()
    model.config.context_length = max(model.config.context_length, a.context)
    kinds = [type(b.attn).__name__ for b in model.transformer.h]

    def last_row_attention(ids):
        out = []
        x = model.transformer.drop(model._embed(ids, 0))
        for block in model.transformer.h:
            A = attention_rows(model, block, block.ln_1(x), ids.shape[1])
            out.append(A[:, -1, :].float())
            del A
            x = block(x)
        return out

    prof = needle_attention_profile(last_row_attention, context_length=a.context, num_needles=a.needles,
                                    placements_per_needle=a.depths, device=runtime.device)
    prof["run"] = a.run
    from selective_attention.niah_eval import checkpoint_identity, harness_provenance
    prof["provenance"] = {"harness": harness_provenance(),
                          "checkpoint": checkpoint_identity(Path(a.run) / "latest.pt", ckpt.get("step"))}
    for layer, kind in zip(prof["layers"], kinds):
        layer["kind"] = kind
        print(f"L{layer['layer']:<2d} {kind[:8]:8s} answer mean {layer['answer_mean']:.3f} "
              f"max-head {layer['answer_max_head']:.3f} | needle mean {layer['needle_mean']:.3f} "
              f"max-head {layer['needle_max_head']:.3f} | first {layer['first_mean']:.3f}", flush=True)
    print(f"rows {prof['rows']}, skipped {prof['skipped']}")
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps(prof, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
