"""Score a checkpoint on a fixed slice of its validation stream, independent of the
training batch size (the in-run eval reads eval_batches x batch_size sequences, so
arms trained at different batch sizes are scored on different windows).

    PYTHONPATH=src python scripts/eval_checkpoint.py runs/scale_16384/all_softmax_rope_seed1337 \
        [--batches 64] [--batch-size 2] [--output results/eval/<name>.json]

Uses the run's resolved_config.yaml for the data paths and sequence length, the
eager model (no compile), bf16 autocast on CUDA, and the chunked evaluation loss.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from selective_attention.niah_eval import load_model_for_niah
from selective_attention.runtime import create_token_stream, evaluate_language_model, resolve_runtime


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run")
    ap.add_argument("--batches", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--output")
    a = ap.parse_args()
    run = Path(a.run)
    cfg = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8"))
    runtime = resolve_runtime("auto", "auto")
    model = load_model_for_niah(run / "latest.pt", runtime.device)
    stream = create_token_stream(cfg["data"], "val")
    seq = int(cfg["training"]["sequence_length"])
    metrics = evaluate_language_model(model, stream, batch_size=a.batch_size, sequence_length=seq,
                                      max_batches=a.batches, runtime=runtime)
    metrics = dict(metrics)
    metrics.update(run=str(run), batches=a.batches, batch_size=a.batch_size, sequence_length=seq,
                   tokens=a.batches * a.batch_size * seq)
    line = f"{run}: val loss {metrics['loss']:.4f} ppl {metrics.get('perplexity', float('nan')):.2f} acc {metrics.get('accuracy', float('nan')):.4f} on {metrics['tokens']:,} tokens"
    print(line, flush=True)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
