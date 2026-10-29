"""A set of very long PG-19 books for long-document perplexity past 128K tokens.

    PYTHONPATH=src python scripts/prepare_pg19_long.py [--min-tokens 524289] [--books 16] \
        [--out data/pg19_long]

The PG-19 test and validation splits (scripts/long_perplexity.py prepare) have 29 books of at
least 131,073 GPT-2 tokens but only 3 of at least 524,289. This reads the train split of the same
parquet mirror at the same pinned revision, shard by shard in file order, tokenizes every book
with the GPT-2 tokenizer, and keeps those with at least --min-tokens tokens until --books are
found. None of our models trained on PG-19 (they trained on FineWeb-Edu), so the split does not
matter for evaluation. Output has the layout long_perplexity.py reads: books.bin (uint16,
concatenated), books.json (title, split, offset, length) and manifest.json (source, revision,
shards read, SHA-256 of books.bin).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPO, REVISION = "emozilla/pg19", "c021754c8e01c5b1cc83a1f549c1f97fbbb756b8"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-tokens", type=int, default=524289)
    ap.add_argument("--books", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "data/pg19_long"))
    a = ap.parse_args()

    import pyarrow.parquet as pq
    import tiktoken
    from huggingface_hub import hf_hub_download, list_repo_files

    enc = tiktoken.get_encoding("gpt2")
    shards = sorted(f for f in list_repo_files(REPO, repo_type="dataset", revision=REVISION)
                    if f.startswith("data/train-") and f.endswith(".parquet"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    books, chunks, offset, read = [], [], 0, []
    for shard in shards:
        path = hf_hub_download(REPO, shard, repo_type="dataset", revision=REVISION)
        read.append(shard)
        table = pq.read_table(path, columns=["short_book_title", "text"])
        for title, text in zip(table.column("short_book_title").to_pylist(), table.column("text").to_pylist()):
            if len(text) < a.min_tokens * 3:          # cheap pre-filter: GPT-2 averages > 3 chars per token
                continue
            ids = enc.encode_ordinary(text)
            if len(ids) < a.min_tokens:
                continue
            arr = np.asarray(ids, dtype=np.uint16)
            chunks.append(arr)
            books.append({"title": title, "split": "train", "offset": offset, "length": int(arr.size)})
            offset += int(arr.size)
            print(f"{len(books):3d}  {arr.size:>9,} tokens  {title[:60]}", flush=True)
            if len(books) >= a.books:
                break
        if len(books) >= a.books:
            break
    data = np.concatenate(chunks)
    data.tofile(out / "books.bin")
    (out / "books.json").write_text(json.dumps(books, indent=1), encoding="utf-8")
    digest = hashlib.sha256((out / "books.bin").read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps({
        "source": REPO, "revision": REVISION, "split": "train", "shards_read": read,
        "min_tokens": a.min_tokens, "tokenizer": "gpt2", "books": len(books), "tokens": int(data.size),
        "books_bin_sha256": digest}, indent=1), encoding="utf-8")
    print(f"wrote {len(books)} books, {data.size:,} tokens, sha256 {digest[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
