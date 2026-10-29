"""Prepare reproducible GPT-2-tokenized Hugging Face dataset streams.

The optional ``datasets`` and ``tiktoken`` dependencies are imported only when
preparation begins, so importing the core package does not require data extras.
"""

from __future__ import annotations

import argparse
from array import array
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, BinaryIO


UINT16_MAX = (1 << 16) - 1
TOKENIZER_NAME = "gpt2"
MANIFEST_VERSION = 1


class DatasetSourceError(RuntimeError):
    """A dataset repository, revision, or config cannot be inspected or loaded."""


def _positive_count(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {parsed}")
    return parsed


def _require_nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class DataPreparationConfig:
    """Complete source identity and output sizes for dataset preparation."""

    dataset: str
    dataset_config: str
    revision: str
    output_dir: Path
    train_tokens: int
    val_tokens: int
    split: str = "train"
    text_column: str = "text"
    overwrite: bool = False

    def __post_init__(self) -> None:
        _require_nonempty(self.dataset, "dataset")
        _require_nonempty(self.dataset_config, "dataset_config")
        _require_nonempty(self.revision, "revision")
        _require_nonempty(self.split, "split")
        _require_nonempty(self.text_column, "text_column")
        if isinstance(self.train_tokens, bool) or not isinstance(self.train_tokens, int):
            raise TypeError("train_tokens must be an integer")
        if isinstance(self.val_tokens, bool) or not isinstance(self.val_tokens, int):
            raise TypeError("val_tokens must be an integer")
        if self.train_tokens <= 0:
            raise ValueError("train_tokens must be positive")
        if self.val_tokens <= 0:
            raise ValueError("val_tokens must be positive")
        if not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not isinstance(self.overwrite, bool):
            raise TypeError("overwrite must be a bool")


@dataclass(frozen=True)
class _PreparedSplit:
    name: str
    path: str
    num_tokens: int
    num_bytes: int
    sha256: str
    documents_consumed: int
    final_document_truncated: bool

    def to_manifest(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "num_tokens": self.num_tokens,
            "num_bytes": self.num_bytes,
            "sha256": self.sha256,
            "documents_consumed": self.documents_consumed,
            "final_document_truncated": self.final_document_truncated,
        }


def _load_optional_dependencies() -> tuple[Any, Any]:
    try:
        import datasets  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "dataset preparation requires Hugging Face datasets; install the "
            "project's data extras with `pip install -e '.[data]'`"
        ) from exc

    try:
        import tiktoken  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "dataset preparation requires tiktoken; install the project's data "
            "extras with `pip install -e '.[data]'`"
        ) from exc
    return datasets, tiktoken


def _package_version(distribution: str, module: Any) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(module, "__version__", "unknown"))


def _uint16_le_bytes(token_ids: Sequence[int]) -> bytes:
    if not token_ids:
        return b""
    values = array("H", token_ids)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def _write_token_ids(
    handle: BinaryIO,
    digest: Any,
    token_ids: Sequence[int],
    *,
    chunk_tokens: int = 1 << 20,
) -> None:
    for start in range(0, len(token_ids), chunk_tokens):
        chunk = token_ids[start : start + chunk_tokens]
        if chunk:
            minimum = min(chunk)
            maximum = max(chunk)
            if minimum < 0 or maximum > UINT16_MAX:
                raise ValueError(
                    f"token IDs must fit uint16, observed range [{minimum}, {maximum}]"
                )
        encoded = _uint16_le_bytes(chunk)
        handle.write(encoded)
        digest.update(encoded)


def _document_text(row: Any, *, text_column: str, document_index: int) -> str:
    if not isinstance(row, Mapping):
        raise TypeError(
            f"dataset row {document_index} is {type(row).__name__}, expected a mapping"
        )
    if text_column not in row:
        columns = ", ".join(sorted(str(column) for column in row))
        raise KeyError(
            f"dataset row {document_index} has no {text_column!r} column; "
            f"available columns: {columns or '<none>'}"
        )
    text = row[text_column]
    if not isinstance(text, str):
        raise TypeError(
            f"dataset row {document_index} column {text_column!r} is "
            f"{type(text).__name__}, expected str"
        )
    return text


def _prepare_split(
    *,
    name: str,
    filename: str,
    temp_path: Path,
    target_tokens: int,
    documents: Iterator[Any],
    tokenizer: Any,
    eos_token_id: int,
    text_column: str,
    source_document_offset: int,
) -> tuple[_PreparedSplit, int]:
    digest = hashlib.sha256()
    written_tokens = 0
    documents_consumed = 0
    final_document_truncated = False

    with temp_path.open("wb") as handle:
        while written_tokens < target_tokens:
            try:
                row = next(documents)
            except StopIteration as exc:
                raise RuntimeError(
                    f"source dataset ended after {source_document_offset + documents_consumed} "
                    f"documents while preparing {name}: wrote {written_tokens} of "
                    f"{target_tokens} requested tokens"
                ) from exc

            document_index = source_document_offset + documents_consumed
            text = _document_text(
                row, text_column=text_column, document_index=document_index
            )
            # encode_ordinary treats text resembling tiktoken special-token syntax
            # as ordinary document text.  The only special ID we add is EOS below.
            document_tokens = tokenizer.encode_ordinary(text)
            documents_consumed += 1

            remaining = target_tokens - written_tokens
            # Keep the split boundary document-disjoint and EOS-terminated even
            # when an exact token budget cuts through the final source document.
            if len(document_tokens) + 1 <= remaining:
                document_tokens.append(eos_token_id)
                emitted = document_tokens
                final_document_truncated = False
            else:
                emitted = document_tokens[: remaining - 1]
                emitted.append(eos_token_id)
                final_document_truncated = True
            _write_token_ids(handle, digest, emitted)
            written_tokens += len(emitted)

    prepared = _PreparedSplit(
        name=name,
        path=filename,
        num_tokens=written_tokens,
        num_bytes=written_tokens * 2,
        sha256=digest.hexdigest(),
        documents_consumed=documents_consumed,
        final_document_truncated=final_document_truncated,
    )
    return prepared, source_document_offset + documents_consumed


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    data = serialized.encode("utf-8")
    with path.open("wb") as handle:
        handle.write(data)
    return hashlib.sha256(data).hexdigest()


def _temporary_path(output_dir: Path, filename: str) -> Path:
    return output_dir / f".{filename}.{os.getpid()}.tmp"


def _hf_stream_shutdown_delay(datasets: Any, source: Any) -> float:
    """Return the upstream Arrow-worker settling delay for a HF stream.

    Hugging Face streaming Parquet readers can leave native Arrow objects queued
    briefly after their Python iterator closes.  If those objects are destroyed
    during interpreter finalization, affected Arrow/Python combinations can abort
    in ``PyGILState_Release``.  The datasets package carries a shutdown delay for
    this exact class of Arrow issue; apply it to real HF iterable datasets while
    leaving custom/test iterators unaffected.
    """

    if not type(source).__module__.startswith("datasets."):
        return 0.0
    datasets_config = getattr(datasets, "config", None)
    delay = getattr(datasets_config, "SLEEP_TIME_ON_THREADS_SHUTDOWN", 5.0)
    try:
        return max(0.0, float(delay))
    except (TypeError, ValueError):
        return 5.0


def _validate_dataset_config(datasets: Any, config: DataPreparationConfig) -> None:
    """Fail early when a pinned revision does not expose the requested config."""

    try:
        available = list(
            datasets.get_dataset_config_names(
                config.dataset,
                revision=config.revision,
            )
        )
    except Exception as exc:
        raise DatasetSourceError(
            f"could not inspect dataset {config.dataset!r} at revision "
            f"{config.revision!r}: {exc}"
        ) from exc

    if config.dataset_config in available:
        return

    preview_limit = 8
    preview = ", ".join(repr(name) for name in available[:preview_limit])
    if len(available) > preview_limit:
        preview = f"{preview}, ..."
    if not preview:
        preview = "<none>"
    raise DatasetSourceError(
        f"dataset config {config.dataset_config!r} is not available for "
        f"{config.dataset!r} at revision {config.revision!r}. Available configs "
        f"({len(available)}): {preview}. Choose an available --dataset-config or "
        "a --revision whose dataset metadata declares the requested config."
    )


def prepare_dataset(config: DataPreparationConfig) -> dict[str, Any]:
    """Stream, tokenize, and write exact-size validation and training splits.

    Validation is taken first.  Each output split begins at a source-document
    boundary; when its token budget ends partway through a document, that
    document's unused suffix is discarded so no document is shared between
    validation and training.
    """

    if not isinstance(config, DataPreparationConfig):
        raise TypeError("config must be a DataPreparationConfig")
    datasets, tiktoken = _load_optional_dependencies()

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = {
        "train": output_dir / "train.bin",
        "val": output_dir / "val.bin",
        "manifest": output_dir / "manifest.json",
        "checksums": output_dir / "SHA256SUMS",
    }
    existing = [path for path in final_paths.values() if path.exists()]
    if existing and not config.overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"refusing to overwrite existing prepared-data files: {paths}; "
            "pass overwrite=True or --overwrite to replace them"
        )

    _validate_dataset_config(datasets, config)

    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
    eos_token_id = int(tokenizer.eot_token)
    if not 0 <= eos_token_id <= UINT16_MAX:
        raise ValueError(
            f"GPT-2 EOS token ID {eos_token_id} does not fit the uint16 format"
        )

    source = datasets.load_dataset(
        config.dataset,
        config.dataset_config,
        split=config.split,
        revision=config.revision,
        streaming=True,
    )
    documents = iter(source)
    stream_shutdown_delay = _hf_stream_shutdown_delay(datasets, source)

    temp_paths = {
        key: _temporary_path(output_dir, path.name)
        for key, path in final_paths.items()
    }
    for temp_path in temp_paths.values():
        if temp_path.exists():
            temp_path.unlink()

    try:
        source_document_offset = 0
        val, source_document_offset = _prepare_split(
            name="validation",
            filename=final_paths["val"].name,
            temp_path=temp_paths["val"],
            target_tokens=config.val_tokens,
            documents=documents,
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            text_column=config.text_column,
            source_document_offset=source_document_offset,
        )
        train, source_document_offset = _prepare_split(
            name="train",
            filename=final_paths["train"].name,
            temp_path=temp_paths["train"],
            target_tokens=config.train_tokens,
            documents=documents,
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            text_column=config.text_column,
            source_document_offset=source_document_offset,
        )

        manifest: dict[str, Any] = {
            "manifest_version": MANIFEST_VERSION,
            "token_format": {
                "dtype": "uint16",
                "byte_order": "little",
                "header_bytes": 0,
                "bytes_per_token": 2,
            },
            "tokenizer": {
                "library": "tiktoken",
                "encoding": TOKENIZER_NAME,
                "eos_token_id": eos_token_id,
                "eos_policy": "append one EOS token after every source document",
            },
            "source": {
                "dataset": config.dataset,
                "config": config.dataset_config,
                "revision": config.revision,
                "split": config.split,
                "text_column": config.text_column,
                "streaming": True,
            },
            "split_policy": {
                "order": ["validation", "train"],
                "exact_token_counts": True,
                "document_disjoint": True,
                "truncated_document_suffix": "discarded",
            },
            "splits": {
                "train": train.to_manifest(),
                "validation": val.to_manifest(),
            },
            "software": {
                "datasets": _package_version("datasets", datasets),
                "tiktoken": _package_version("tiktoken", tiktoken),
            },
        }
        manifest_sha256 = _write_json(temp_paths["manifest"], manifest)
        checksums = (
            f"{train.sha256}  {train.path}\n"
            f"{val.sha256}  {val.path}\n"
            f"{manifest_sha256}  {final_paths['manifest'].name}\n"
        )
        with temp_paths["checksums"].open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(checksums)

        # Commit outputs only after both token files and both manifests are complete.
        for key in ("train", "val", "manifest", "checksums"):
            os.replace(temp_paths[key], final_paths[key])
        return manifest
    finally:
        try:
            # Streaming datasets may own an open HTTP/Arrow reader. We stop well
            # before exhausting the upstream dataset, so close its generator
            # explicitly while Python is still fully initialized instead of
            # leaving native cleanup to interpreter shutdown.
            close_documents = getattr(documents, "close", None)
            if close_documents is not None:
                close_documents()

            if stream_shutdown_delay:
                # Drop the iterator and dataset before interpreter finalization so
                # native Arrow buffers are released with a valid Python thread
                # state.  The short delay mirrors datasets' own Parquet workaround.
                documents = None
                source = None
                gc.collect()
                time.sleep(stream_shutdown_delay)
                gc.collect()
        finally:
            for temp_path in temp_paths.values():
                if temp_path.exists():
                    temp_path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a pinned Hugging Face dataset, tokenize it with tiktoken GPT-2, "
            "and write checksummed little-endian uint16 train/validation streams."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Hugging Face dataset repository ID or loading-script path",
    )
    parser.add_argument(
        "--config",
        "--dataset-config",
        dest="dataset_config",
        required=True,
        help="explicit Hugging Face dataset configuration name",
    )
    parser.add_argument(
        "--revision",
        required=True,
        help="explicit source dataset revision, preferably an immutable commit SHA",
    )
    parser.add_argument(
        "--split", default="train", help="source split to stream (default: train)"
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="source row column containing document text (default: text)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--train-tokens", type=_positive_count, required=True, metavar="N"
    )
    parser.add_argument(
        "--val-tokens",
        "--validation-tokens",
        dest="val_tokens",
        type=_positive_count,
        required=True,
        metavar="N",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace train.bin, val.bin, and manifests in the output directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = DataPreparationConfig(
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        revision=args.revision,
        output_dir=args.output_dir,
        train_tokens=args.train_tokens,
        val_tokens=args.val_tokens,
        split=args.split,
        text_column=args.text_column,
        overwrite=args.overwrite,
    )
    try:
        prepare_dataset(config)
    except DatasetSourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Prepared dataset and SHA256 manifests in {config.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DataPreparationConfig",
    "DatasetSourceError",
    "build_parser",
    "main",
    "prepare_dataset",
]
