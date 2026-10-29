"""Natural-text filler for Niah hard mode: the first `--books` books of a prepared PG-19 set, decoded and
joined by blank lines.

    PYTHONPATH=src python scripts/investigate/make_niah_filler.py --data data/pg19_long --books 6 \
        --output data/niah_filler/pg19_train6.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from long_perplexity import load_books  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--books", type=int, default=6)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    _, books, data = load_books(Path(a.data), 1)
    enc = tiktoken.get_encoding("gpt2")
    texts = [enc.decode(data[b["offset"]:b["offset"] + b["length"]].tolist()) for b in books[: a.books]]
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(texts), encoding="utf-8", newline="\n")
    print(f"{len(texts)} books, {sum(b['length'] for b in books[: a.books]):,} tokens -> {out}")
    for b in books[: a.books]:
        print("  ", b["title"][:70], b["split"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
