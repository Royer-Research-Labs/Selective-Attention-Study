"""Verify the files copied from Dilution-Attention against the recorded commit.

    python scripts/check_vendored.py [path/to/Dilution-Attention checkout]

vendored.json lists each copied file, the SHA-256 of its source and the (old, new) text edits
applied to it; most have none. This checks that every file is as recorded and that undoing its
edits reproduces the source hash, so the copy is the source apart from exactly those edits. With a
checkout of the source repository it also hashes the files there.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    manifest = json.loads((ROOT / "vendored.json").read_text(encoding="utf-8"))
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    bad, edited = [], 0
    for entry in manifest["files"]:
        data = (ROOT / entry["file"]).read_bytes()
        if sha(data) != entry["sha256"]:
            bad.append(f"{entry['file']}: differs from vendored.json")
        for old, new in reversed(entry["edits"]):   # edits were applied in order; undo last first
            data = data.replace(new.encode(), old.encode())
        edited += bool(entry["edits"])
        if sha(data) != entry["source_sha256"]:
            bad.append(f"{entry['file']}: undoing the recorded edits does not give {entry['source']}")
        if source is not None and sha((source / entry["source"]).read_bytes()) != entry["source_sha256"]:
            bad.append(f"{entry['source']}: differs in {source}")
    listed = {entry["file"] for entry in manifest["files"]}
    added = sorted(p.name for p in (ROOT / "src" / "selective_attention").glob("*.py")
                   if f"src/selective_attention/{p.name}" not in listed)
    print(f"{len(listed)} files checked against {manifest['commit'][:7]} "
          f"({len(listed) - edited} identical, {edited} with the recorded edits); new in the package: {', '.join(added)}")
    for line in bad:
        print("MISMATCH", line)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
