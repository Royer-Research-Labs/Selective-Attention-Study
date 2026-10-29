"""Does distant context help? PG-19 loss with the full prefix against a 16K rolling window, on the
same target tokens (a context-use control; the full-prefix test alone shows only that long inputs are
tolerated).

    PYTHONPATH=src python scripts/investigate/rolling_window.py --data data/pg19_gpt2 \
        --model sel_bos_s1337=runs/.../checkpoint_step_0064000.pt [--model ...] \
        [--max-length 131072] [--window 16384] [--stride 4096] --output results/investigate/rolling_window.json

Per book, two passes over the first `max_length` predicted tokens (the same tokens and the same BOS
handling as scripts/long_perplexity.py):
  full     one teacher-forced pass over the whole prefix (RoPE models: NTK scaling to max_length)
  window   windows of `window` input tokens every `stride` tokens; each window scores only its last
           `stride` predictions, so every token past the first window sees between window - stride + 1
           and window tokens of context (the first window scores all of its predictions). Models
           trained with a BOS token get it at the start of every window, as in training.
Reported per doubling position range: both losses, and full minus window with its paired standard
error across books. Negative = the context beyond the window lowered the loss.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from long_perplexity import _bos_token, load_books, position_nll, range_means, ranges  # noqa: E402


def window_nll(model, seq: torch.Tensor, window: int, stride: int, bos) -> torch.Tensor:
    """Per-position NLL of seq[1:] where position i is predicted from at most `window` tokens."""
    n = seq.numel() - 1
    if (n - window) % stride or n < window:
        raise ValueError("max_length - window must be a non-negative multiple of stride")
    out = torch.empty(n, dtype=torch.float32, device=seq.device)
    out[:window] = position_nll(model, seq[: window + 1])
    for s in range(stride, n - window + 1, stride):
        ids = seq[s: s + window + 1]
        if bos is not None:                       # every training window started with the BOS token
            ids = torch.cat([ids.new_full((1,), bos), ids[1:]])
        out[s + window - stride: s + window] = position_nll(model, ids)[window - stride:]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=CHECKPOINT")
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-length", type=int, default=131072)
    ap.add_argument("--window", type=int, default=16384)
    ap.add_argument("--stride", type=int, default=4096)
    ap.add_argument("--books", type=int, default=0)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    from selective_attention.extend import ExtendedContext
    from selective_attention.niah_eval import checkpoint_identity, load_model_for_niah

    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest, books, data = load_books(Path(a.data), a.max_length + 1)
    if a.books:
        books = books[: a.books]
    rs = ranges(a.max_length)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.no_grad()
    result = {"task": "rolling_window", "argv": sys.argv[1:], "data": manifest, "max_length": a.max_length,
              "window": a.window, "stride": a.stride, "ranges": rs,
              "books": [{"title": b["title"], "split": b["split"], "length": b["length"]} for b in books],
              "models": {}}
    for spec in a.model:
        name, _, ckpt = spec.partition("=")
        model = load_model_for_niah(ckpt, device)
        bos = _bos_token(ckpt)
        has_rope = any(type(m).__name__ == "RopeModule" for m in model.modules())
        full_rows, win_rows = [], []
        t0 = time.time()
        for b in books:
            seq = torch.from_numpy(data[b["offset"]:b["offset"] + a.max_length + 1].astype(np.int64)).to(device)
            if bos is not None:
                seq = torch.cat([seq.new_full((1,), bos), seq[:-1]])
            with torch.no_grad(), autocast:
                with ExtendedContext(model, a.max_length, has_rope):
                    full = position_nll(model, seq)
                win = window_nll(model, seq, a.window, a.stride, bos)
            full_rows.append(range_means(full.cpu().numpy(), rs))
            win_rows.append(range_means(win.cpu().numpy(), rs))
        f, w = np.asarray(full_rows), np.asarray(win_rows)
        diff = f - w
        se = (diff.std(0, ddof=1) / math.sqrt(len(books))).tolist() if len(books) > 1 else None
        result["models"][name] = {
            "checkpoint": checkpoint_identity(ckpt, getattr(model, "checkpoint_step", None)),
            "trained_context": model.config.context_length, "rope": has_rope, "bos_token": bos,
            "full": f.mean(0).tolist(), "window": w.mean(0).tolist(), "full_minus_window": diff.mean(0).tolist(),
            "full_minus_window_se": se, "full_per_book": full_rows, "window_per_book": win_rows}
        print(f"[rolling] {name}: {len(books)} books in {time.time() - t0:.0f}s", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    labels = [f"{lo // 1024}K-{hi // 1024}K" if lo >= 1024 else f"{lo}-{hi}" for lo, hi in rs]
    print(f"\nfull prefix minus {a.window}-token window, nats/token (negative: distant context helps)")
    print(f"{'model':24s} " + " ".join(f"{l:>13s}" for l in labels))
    for name, e in result["models"].items():
        cells = [f"{d:+.3f}" + (f"±{s:.3f}" if e["full_minus_window_se"] else "")
                 for d, s in zip(e["full_minus_window"], e["full_minus_window_se"] or [0] * len(rs))]
        print(f"{name:24s} " + " ".join(f"{c:>13s}" for c in cells))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
