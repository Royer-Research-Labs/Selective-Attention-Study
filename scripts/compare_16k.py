"""Table of the 16K results: this repository's Selective Attention runs next to the published
Dilution-Attention runs (same model, data, schedule and evaluation commands).

    python scripts/compare_16k.py [--reference results/reference]

Loss is the common 2.1M-token validation slice (scripts/eval_checkpoint.py). NIAH is accuracy over
7 scored needles x 20 placements (in context) or x 10 placements (past it); chance is 1/7. For the
reference runs the niah-v4-spec re-score (results/niah_v4) is used where it exists, as in that
repository's figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LENGTHS = [3840, 7936, 12288, 16000, 24000, 32000, 48000, 64000, 96000, 128000, 192000, 256000, 384000, 512000]
ARMS = [  # (label, arm name, repository: "here" or "reference")
    ("softmax + RoPE", "all_softmax_rope", "reference"),
    ("softmax, no position", "all_softmax_nope_lr6e-4", "reference"),
    ("dilution, no position", "all_dilution_nope", "reference"),
    ("alt6 (softmax + RoPE / dilution layers)", "alt6_softmax_rope_dilution", "reference"),
    ("selective + BOS, no position (recommended)", "all_selective_nope_bos", "here"),
    ("selective, no position, no BOS", "all_selective_nope", "here"),
]
SEEDS = (1337, 2357, 7331)


def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def niah(results: Path, arm: str, seed: int) -> dict[int, float]:
    acc: dict[int, float] = {}
    names = [f"scale1x__{arm}_seed{seed}"] + [f"{t}__scale_16384_{arm}_seed{seed}"
                                               for t in ("extrap64k", "extrap128k", "extrap256k", "extrap512k")]
    for name in names:
        rows = None
        for sub in ("niah_v4", "niah"):
            rows = load(results / sub / f"{name}.json")
            if rows:
                break
        for row in rows or []:
            acc.setdefault(int(row["context_length"]), float(row["accuracy"]))
    return acc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", default=str(ROOT / "results/reference"),
                    help="the published Dilution-Attention results (copied here, see SOURCE.json; or "
                         "the results/ directory of a checkout of that repository)")
    a = ap.parse_args()
    roots = {"here": ROOT / "results", "reference": Path(a.reference)}
    header = "| model | seed | loss | " + " | ".join(f"{n // 1000}K" if n >= 10000 else str(n) for n in LENGTHS) + " |"
    print(header)
    print("|" + "---|" * (3 + len(LENGTHS)))
    for label, arm, repo in ARMS:
        for seed in SEEDS:
            ev = load(roots[repo] / "eval" / f"scale1x__{arm}_seed{seed}.json")
            acc = niah(roots[repo], arm, seed)
            if ev is None and not acc:
                continue
            loss = f"{ev['loss']:.4f}" if ev else "-"
            cells = [f"{acc[n]:.2f}" if n in acc else "-" for n in LENGTHS]
            print(f"| {label} | {seed} | {loss} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
