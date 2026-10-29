"""Perplexity by position on long books (PG-19): does a model keep predicting real text past the
context it was trained on?

    PYTHONPATH=src python scripts/long_perplexity.py prepare [--out data/pg19_gpt2]
    PYTHONPATH=src python scripts/long_perplexity.py eval --model NAME=CHECKPOINT [...] \
        --max-length 131072 --output results/longppl/<set>.json
    PYTHONPATH=src python scripts/long_perplexity.py plot results/longppl/<set>.json [...] --out <png>

prepare: downloads the PG-19 test and validation splits (Project Gutenberg books published before
1919; parquet mirror emozilla/pg19 at a pinned revision), tokenizes every book with the GPT-2
tokenizer, and writes <out>/books.bin (uint16 tokens, books concatenated), <out>/books.json (title,
split, offset, length) and <out>/manifest.json (source, revision, SHA-256 of books.bin).

eval: every book with at least --max-length + 1 tokens, in file order. Each model reads the first
L + 1 tokens of each book teacher-forced, and the loss at every position is averaged within
doubling position ranges (0-512, 512-1K, 1K-2K, ...). Models without RoPE compute the same loss at
a position whatever the sequence length, so one pass at --max-length gives every position. Models
with RoPE get one pass per length L = trained context x 2^k with NTK base scaling to L (as in the
NIAH ladders), and the curve takes each range (L/2, L] from the pass at L; they also get one pass
at --max-length without scaling. Output per model: the loss per range (mean over books, with the
standard error across books) and every book's per-range loss, so paired comparisons can be made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
PG19 = {"repo": "emozilla/pg19", "revision": "c021754c8e01c5b1cc83a1f549c1f97fbbb756b8",
        "files": {"test": "data/test-00000-of-00001-29a571947c0b5ccc.parquet",
                  "validation": "data/validation-00000-of-00001-0f92e2337f79aeac.parquet"}}


def log(msg: str) -> None:
    print(f"[long_ppl {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- prepare ------------------------------------------------------------------------------

def prepare(out: Path) -> None:
    import pyarrow.parquet as pq
    import tiktoken
    from huggingface_hub import hf_hub_download

    enc = tiktoken.get_encoding("gpt2")
    out.mkdir(parents=True, exist_ok=True)
    books, chunks, offset = [], [], 0
    for split, fname in PG19["files"].items():
        path = hf_hub_download(PG19["repo"], fname, repo_type="dataset", revision=PG19["revision"])
        table = pq.read_table(path)
        for title, text in zip(table.column("short_book_title").to_pylist(), table.column("text").to_pylist()):
            ids = np.asarray(enc.encode_ordinary(text), dtype=np.uint16)
            books.append({"title": title, "split": split, "offset": offset, "length": int(ids.size)})
            chunks.append(ids)
            offset += int(ids.size)
    data = np.concatenate(chunks)
    (out / "books.bin").write_bytes(data.tobytes())
    (out / "books.json").write_text(json.dumps(books, indent=1) + "\n", encoding="utf-8")
    manifest = {"source": PG19, "tokenizer": "gpt2 (tiktoken, no special tokens)", "books": len(books),
                "tokens": int(data.size), "books_bin_sha256": hashlib.sha256(data.tobytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {len(books)} books, {data.size:,} tokens to {out}")


def load_books(root: Path, min_tokens: int):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    books = json.loads((root / "books.json").read_text(encoding="utf-8"))
    data = np.fromfile(root / "books.bin", dtype=np.uint16)
    keep = [b for b in books if b["length"] >= min_tokens]
    return manifest, keep, data


# ---- eval ---------------------------------------------------------------------------------

def ranges(max_length: int) -> list[tuple[int, int]]:
    """Doubling position ranges [lo, hi) covering the predicted positions 0 .. max_length - 1."""
    out, lo, hi = [], 0, 512
    while lo < max_length:
        out.append((lo, min(hi, max_length)))
        lo, hi = hi, hi * 2
    return out


@torch.no_grad()
def position_nll(model, ids: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Per-position NLL (nats) of ids[1:] given ids[:-1] under one teacher-forced pass; the LM head
    runs in chunks so no full-vocabulary logits tensor is held for the whole sequence."""
    x = model.transformer.drop(model._embed(ids[None, :-1], 0))
    for block in model.transformer.h:
        x = block(x)
    x = model.transformer.ln_f(x)[0]
    tgt = ids[1:]
    out = torch.empty(tgt.numel(), dtype=torch.float32, device=ids.device)
    for s in range(0, tgt.numel(), chunk):
        logits = model.lm_head(x[s:s + chunk]).float()
        out[s:s + chunk] = F.cross_entropy(logits, tgt[s:s + chunk], reduction="none")
    return out


def range_means(nll: np.ndarray, rs) -> list[float]:
    return [float(nll[lo:hi].mean()) if hi <= nll.size else float("nan") for lo, hi in rs]


def _bos_token(ckpt):
    """data.bos_token of the run that produced a checkpoint, or None."""
    from selective_attention.runtime import load_checkpoint
    resolved = load_checkpoint(ckpt, map_location="cpu").get("resolved_config") or {}
    return (resolved.get("data") or {}).get("bos_token")


def evaluate(a) -> int:
    from selective_attention.extend import ExtendedContext
    from selective_attention.niah_eval import checkpoint_identity, load_model_for_niah

    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest, books, data = load_books(Path(a.data), a.max_length + 1)
    if not books:
        log(f"no book has {a.max_length + 1} tokens")
        return 2
    if a.books:
        books = books[: a.books]
    log(f"{len(books)} books with at least {a.max_length + 1:,} tokens")
    rs = ranges(a.max_length)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.no_grad()
    result = {"task": "long_perplexity", "argv": sys.argv[1:], "data": manifest,
              "books": [{"title": b["title"], "split": b["split"], "length": b["length"]} for b in books],
              "max_length": a.max_length, "ranges": rs, "models": {}}
    for spec in a.model:
        name, _, ckpt = spec.partition("=")
        model = load_model_for_niah(ckpt, device)
        bos = _bos_token(ckpt)   # models trained with data.bos_token read [bos] + book[:L] (L targets, as without it)
        trained = model.config.context_length
        has_rope = any(type(m).__name__ == "RopeModule" for m in model.modules())
        entry = {"checkpoint": checkpoint_identity(ckpt, getattr(model, "checkpoint_step", None)),
                 "trained_context": trained, "rope": has_rope, "passes": {}}
        lengths = sorted({min(a.max_length, trained * 2 ** k) for k in range(0, 12)
                          if trained * 2 ** (k - 1) < a.max_length})
        passes = [("plain", a.max_length, False)]
        if has_rope:
            passes += [(f"ntk@{L}", L, True) for L in lengths if L > trained]
        t0 = time.time()
        per_pass = {}
        for tag, L, ntk in passes:
            rows = []
            with ExtendedContext(model, L, ntk), autocast:
                for b in books:
                    ids = torch.from_numpy(data[b["offset"]:b["offset"] + L + 1].astype(np.int64)).to(device)
                    if bos is not None:
                        ids = torch.cat([ids.new_full((1,), bos), ids[:-1]])
                    rows.append(range_means(position_nll(model, ids).cpu().numpy(), rs))
            per_pass[tag] = {"length": L, "ntk": ntk, "per_book": rows}
            log(f"{name} {tag}: {len(books)} books in {time.time() - t0:.0f}s")
        # the reported curve: range (L/2, L] from the pass at L (NTK past the trained context);
        # ranges inside the trained context, and every range for models without RoPE, from the
        # plain pass
        curve_books = []
        for bi in range(len(books)):
            row = []
            for ri, (lo, hi) in enumerate(rs):
                tag = "plain"
                if has_rope and hi > trained:
                    L = next(L for L in lengths if L >= hi)
                    tag = f"ntk@{L}" if L > trained else "plain"
                row.append(per_pass[tag]["per_book"][bi][ri])
            curve_books.append(row)
        entry["passes"] = per_pass
        entry["curve_per_book"] = curve_books
        arr = np.asarray(curve_books)
        entry["curve"] = arr.mean(0).tolist()
        entry["curve_se"] = (arr.std(0, ddof=1) / math.sqrt(arr.shape[0])).tolist() if arr.shape[0] > 1 else None
        plain = np.asarray(per_pass["plain"]["per_book"])
        entry["curve_plain"] = plain.mean(0).tolist()
        result["models"][name] = entry
        del model
        torch.cuda.empty_cache() if device == "cuda" else None
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    labels = [f"{lo // 1024}K-{hi // 1024}K" if lo >= 1024 else f"{lo}-{hi}" for lo, hi in rs]
    print("\nloss (nats/token) by position range; curve uses NTK scaling past the trained context for RoPE models")
    print(f"{'model':28s} " + " ".join(f"{l:>9s}" for l in labels))
    for name, e in result["models"].items():
        print(f"{name:28s} " + " ".join(f"{v:9.3f}" for v in e["curve"]))
    log(f"wrote {out}")
    return 0


# ---- plot ---------------------------------------------------------------------------------

def plot(a) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = [json.loads(Path(f).read_text(encoding="utf-8")) for f in a.results]
    fig, axes = plt.subplots(1, len(files), figsize=(6.4 * len(files), 4.6), squeeze=False)
    for ax, res, title in zip(axes[0], files, a.titles or [None] * len(files)):
        rs = res["ranges"]
        mids = [math.sqrt(max(lo, 256) * hi) for lo, hi in rs]
        trained = None
        for name, e in res["models"].items():
            style = "-" if "dilution" in name and "alt6" not in name else "--" if "softmax" in name else "-."
            label = (name.replace("_nope", ", no position encoding").replace("_rope", " + RoPE")
                     .replace("_s", ", seed ").replace("alt6", "alt6 hybrid"))
            # one colour family per arm, the same in every panel; the second seed is lighter
            family = ("alt6" if "alt6" in name else "dilution" if "dilution" in name else
                      "rope" if "rope" in name else "nope")
            shades = {"dilution": ("#1f5fa8", "#6fa8dc"), "rope": ("#e07b00", "#f6b26b"),
                      "nope": ("#b3261e", "#e69188"), "alt6": ("#6a3d9a", "#b4a7d6")}[family]
            color = shades[1] if name.endswith("2357") and not name.endswith("s1337") else shades[0]
            ax.errorbar(mids, e["curve"], yerr=e["curve_se"], fmt=style, marker="o", ms=3, capsize=2,
                        label=label, color=color)
            trained = e["trained_context"]
        if trained:
            ax.axvline(trained, color="#111111", lw=2)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("token position in the book")
        ax.set_ylabel("loss (nats per token)")
        if a.ymin is not None or a.ymax is not None:
            ax.set_ylim(a.ymin, a.ymax)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
        if title:
            ax.set_title(title, fontsize=11, loc="left")
    fig.text(0.01, 0.01, "PG-19 books (test + validation), mean over books with standard errors. Black line: "
             "trained context. RoPE models use NTK scaling past it.", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170)
    log(f"wrote {a.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", default=str(ROOT / "data/pg19_gpt2"))
    e = sub.add_parser("eval")
    e.add_argument("--model", action="append", required=True, metavar="NAME=CHECKPOINT")
    e.add_argument("--data", default=str(ROOT / "data/pg19_gpt2"))
    e.add_argument("--max-length", type=int, default=131072)
    e.add_argument("--books", type=int, default=0, help="use only the first N qualifying books (0 = all)")
    e.add_argument("--output", required=True)
    g = sub.add_parser("plot")
    g.add_argument("results", nargs="+")
    g.add_argument("--titles", nargs="*")
    g.add_argument("--ymin", type=float)
    g.add_argument("--ymax", type=float)
    g.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(Path(a.out))
        return 0
    return evaluate(a) if a.cmd == "eval" else plot(a)


if __name__ == "__main__":
    sys.exit(main())
