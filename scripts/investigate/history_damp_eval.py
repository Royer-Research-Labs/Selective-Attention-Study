"""Evaluation-only test of damping Selective Attention's history term past the trained length.

    PYTHONPATH=src python scripts/investigate/history_damp_eval.py --damp linear pg19 <long_perplexity.py eval args>
    PYTHONPATH=src python scripts/investigate/history_damp_eval.py --damp linear niah <selective-attention-niah args>

Sets `history_damp` on every Selective Attention layer of every model these tools load, then runs
them unchanged. "linear" scales query i's history term by min(1, T_train / (i + 1)), "sqrt" by its
square root; inside the trained length nothing changes. See kernels.selective_attention_forward.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

import selective_attention.niah_eval as niah_eval

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    if len(sys.argv) < 4 or sys.argv[1] != "--damp" or sys.argv[3] not in ("pg19", "niah"):
        print(__doc__)
        return 2
    damp, tool, rest = sys.argv[2], sys.argv[3], sys.argv[4:]
    original = niah_eval.load_model_for_niah

    def load_damped(*args, **kwargs):
        model = original(*args, **kwargs)
        for module in model.modules():
            if type(module).__name__ == "SelectiveAttention":
                module.history_damp = damp
        return model

    niah_eval.load_model_for_niah = load_damped
    if tool == "pg19":
        sys.argv = [str(ROOT / "scripts/long_perplexity.py"), *rest]
        runpy.run_path(sys.argv[0], run_name="__main__")
    else:
        niah_eval.main(rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
