"""Deterministic token streams used by training and evaluation.

The on-disk format intentionally has no hidden metadata: it is a contiguous array
of little-endian unsigned 16-bit token IDs.  Dataset provenance and checksums live
next to these files in the manifest.json written by `dilution.prepare_data`.

Training randomness is owned by the caller through an explicit CPU
``torch.Generator``.  Saving and restoring ``generator.get_state()`` therefore
reproduces the next batch exactly.  Evaluation does not use randomness at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from array import array
from collections.abc import Iterator, Sequence
from pathlib import Path
import random
import sys
from typing import BinaryIO, TypeAlias

import torch


Device: TypeAlias = str | torch.device | None
TokenBatch: TypeAlias = tuple[torch.Tensor, torch.Tensor]


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _decode_uint16_le(data: bytes) -> torch.Tensor:
    """Decode raw little-endian uint16 bytes into a CPU int64 tensor."""

    if len(data) % 2:
        raise ValueError("uint16 data must contain an even number of bytes")
    values = array("H")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    if not values:
        return torch.empty(0, dtype=torch.long)
    # frombuffer avoids the per-element Python sequence path; full-file
    # validation decodes the entire corpus at every training start.
    return torch.frombuffer(values, dtype=torch.uint16).to(torch.long)


class TokenStream(ABC):
    """A finite, random-access sequence of integer token IDs.

    Concrete streams only implement :meth:`read_tokens`; batching behavior is
    shared so synthetic and real-data experiments use identical sampling rules.
    """

    _num_tokens: int

    @classmethod
    def from_binary(
        cls,
        path: str | Path,
        *,
        vocab_size: int | None = None,
        validate: bool = True,
    ) -> UInt16TokenStream:
        """Construct a stream from a headerless little-endian uint16 file."""

        return UInt16TokenStream(path, vocab_size=vocab_size, validate=validate)

    @classmethod
    def from_synthetic(
        cls,
        num_tokens: int,
        vocab_size: int,
        *,
        seed: int = 0,
        pattern_length: int = 256,
    ) -> SyntheticTokenStream:
        """Construct an offline deterministic, learnable synthetic stream."""

        return SyntheticTokenStream(
            num_tokens,
            vocab_size,
            seed=seed,
            pattern_length=pattern_length,
        )

    @property
    def num_tokens(self) -> int:
        """Number of token IDs in this stream."""

        return self._num_tokens

    def __len__(self) -> int:
        return self.num_tokens

    @abstractmethod
    def read_tokens(self, start: int, count: int) -> torch.Tensor:
        """Return ``count`` tokens beginning at ``start`` as CPU int64."""

    def _validate_read(self, start: int, count: int) -> None:
        if isinstance(start, bool) or not isinstance(start, int):
            raise TypeError("start must be an integer")
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("count must be an integer")
        if start < 0:
            raise ValueError(f"start must be non-negative, got {start}")
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if start > self.num_tokens or count > self.num_tokens - start:
            raise IndexError(
                f"token range [{start}, {start + count}) exceeds stream length "
                f"{self.num_tokens}"
            )

    def _read_windows(self, starts: Sequence[int], window_length: int) -> torch.Tensor:
        return torch.stack(
            [self.read_tokens(start, window_length) for start in starts], dim=0
        )

    def _make_batch(
        self,
        starts: Sequence[int],
        sequence_length: int,
        device: Device,
    ) -> TokenBatch:
        bos = getattr(self, "bos_token", None)
        if bos is None:
            windows = self._read_windows(starts, sequence_length + 1)
            input_ids = windows[:, :-1].contiguous()
            targets = windows[:, 1:].contiguous()
        else:  # opt-in (data.bos_token): input = [bos] + window[:-1], every window token is a target
            windows = self._read_windows(starts, sequence_length)
            input_ids = torch.cat([torch.full_like(windows[:, :1], bos), windows[:, :-1]], dim=1).contiguous()
            targets = windows.contiguous()
        if device is not None:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
        return input_ids, targets

    def sample_batch(
        self,
        batch_size: int,
        sequence_length: int,
        generator: torch.Generator,
        device: Device = None,
    ) -> TokenBatch:
        """Sample random next-token windows using a caller-owned CPU generator.

        The generator is explicit so its state can be included in a training
        checkpoint.  No module-level or process-global random state is consumed.
        """

        batch_size = _positive_int(batch_size, "batch_size")
        sequence_length = _positive_int(sequence_length, "sequence_length")
        if not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator")
        if generator.device.type != "cpu":
            raise ValueError("sample_batch requires a CPU torch.Generator")
        if self.num_tokens < sequence_length + 1:
            raise ValueError(
                f"stream has {self.num_tokens} tokens, but a sequence length of "
                f"{sequence_length} requires at least {sequence_length + 1}"
            )

        # randint's upper bound is exclusive.  The final valid start is
        # num_tokens - sequence_length - 1.
        high = self.num_tokens - sequence_length
        starts_tensor = torch.randint(
            0,
            high,
            (batch_size,),
            generator=generator,
            device="cpu",
            dtype=torch.long,
        )
        starts = starts_tensor.tolist()
        return self._make_batch(starts, sequence_length, device)

    def iter_eval_batches(
        self,
        batch_size: int,
        sequence_length: int,
        max_batches: int | None = None,
        device: Device = None,
    ) -> Iterator[TokenBatch]:
        """Yield deterministic, sequential next-token evaluation batches.

        Adjacent examples advance by ``sequence_length`` tokens.  Thus target
        tokens are never duplicated, while the boundary token needed as the next
        example's first input is reused.  A final partial batch is yielded when
        necessary.  Calling this method again always restarts at token zero.
        """

        batch_size = _positive_int(batch_size, "batch_size")
        sequence_length = _positive_int(sequence_length, "sequence_length")
        if max_batches is not None:
            if isinstance(max_batches, bool) or not isinstance(max_batches, int):
                raise TypeError("max_batches must be an integer or None")
            if max_batches < 0:
                raise ValueError("max_batches must be non-negative")
        if self.num_tokens < sequence_length + 1:
            raise ValueError(
                f"stream has {self.num_tokens} tokens, but a sequence length of "
                f"{sequence_length} requires at least {sequence_length + 1}"
            )

        num_examples = (self.num_tokens - 1) // sequence_length
        batch_index = 0
        for first_example in range(0, num_examples, batch_size):
            if max_batches is not None and batch_index >= max_batches:
                break
            last_example = min(first_example + batch_size, num_examples)
            starts = [
                example_index * sequence_length
                for example_index in range(first_example, last_example)
            ]
            yield self._make_batch(starts, sequence_length, device)
            batch_index += 1

    def sequential_batches(
        self,
        batch_size: int,
        sequence_length: int,
        max_batches: int | None = None,
        device: Device = None,
    ) -> Iterator[TokenBatch]:
        """Alias for :meth:`iter_eval_batches`."""

        return self.iter_eval_batches(
            batch_size,
            sequence_length,
            max_batches=max_batches,
            device=device,
        )


class SyntheticTokenStream(TokenStream):
    """A deterministic repeated-token pattern for offline smoke tests.

    ``seed`` selects a fixed permutation of unique vocabulary IDs and that short
    pattern is repeated for the requested stream length.  Each token therefore
    has a consistent successor (including at the wraparound), making this a
    deliberately learnable language-modeling signal rather than iid noise.
    """

    def __init__(
        self,
        num_tokens: int,
        vocab_size: int,
        *,
        seed: int = 0,
        pattern_length: int = 256,
    ) -> None:
        self._num_tokens = _positive_int(num_tokens, "num_tokens")
        self.vocab_size = _positive_int(vocab_size, "vocab_size")
        pattern_length = _positive_int(pattern_length, "pattern_length")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")

        self.seed = seed
        self.pattern_length = min(pattern_length, self.vocab_size)
        rng = random.Random(seed)
        pattern = rng.sample(range(self.vocab_size), self.pattern_length)
        self._pattern = torch.tensor(pattern, dtype=torch.long)

    def read_tokens(self, start: int, count: int) -> torch.Tensor:
        self._validate_read(start, count)
        if count == 0:
            return torch.empty(0, dtype=torch.long)
        positions = torch.arange(start, start + count, dtype=torch.long)
        return self._pattern[positions.remainder(self.pattern_length)]

    def _read_windows(self, starts: Sequence[int], window_length: int) -> torch.Tensor:
        start_tensor = torch.tensor(starts, dtype=torch.long).unsqueeze(1)
        offsets = torch.arange(window_length, dtype=torch.long).unsqueeze(0)
        pattern_indices = (start_tensor + offsets).remainder(self.pattern_length)
        return self._pattern[pattern_indices]


class UInt16TokenStream(TokenStream):
    """Random access over a headerless little-endian uint16 token file."""

    bytes_per_token = 2
    max_token_id = (1 << 16) - 1

    def __init__(
        self,
        path: str | Path,
        *,
        vocab_size: int | None = None,
        validate: bool = True,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"token file does not exist: {self.path}")
        if not self.path.is_file():
            raise ValueError(f"token path is not a regular file: {self.path}")

        size_bytes = self.path.stat().st_size
        if size_bytes == 0:
            raise ValueError(f"token file is empty: {self.path}")
        if size_bytes % self.bytes_per_token:
            raise ValueError(
                f"uint16 token file has an odd byte length ({size_bytes}): {self.path}"
            )
        self._num_tokens = size_bytes // self.bytes_per_token

        if vocab_size is not None:
            vocab_size = _positive_int(vocab_size, "vocab_size")
            if vocab_size > self.max_token_id + 1:
                raise ValueError(
                    "vocab_size cannot exceed 65536 for a uint16 token stream"
                )
        if not isinstance(validate, bool):
            raise TypeError("validate must be a bool")
        self.vocab_size = vocab_size
        if validate and vocab_size is not None:
            self.validate_token_ids()

    def validate_token_ids(self, *, chunk_tokens: int = 1 << 20) -> None:
        """Scan the file and reject IDs outside ``[0, vocab_size)``."""

        if self.vocab_size is None:
            raise ValueError("vocab_size is required to validate token ID bounds")
        chunk_tokens = _positive_int(chunk_tokens, "chunk_tokens")

        token_offset = 0
        with self.path.open("rb") as handle:
            while data := handle.read(chunk_tokens * self.bytes_per_token):
                tokens = _decode_uint16_le(data)
                invalid = torch.nonzero(tokens >= self.vocab_size, as_tuple=False)
                if invalid.numel():
                    local_index = int(invalid[0, 0])
                    token_id = int(tokens[local_index])
                    absolute_index = token_offset + local_index
                    raise ValueError(
                        f"token ID {token_id} at index {absolute_index} is outside "
                        f"vocabulary size {self.vocab_size} in {self.path}"
                    )
                token_offset += tokens.numel()

        if token_offset != self.num_tokens:
            raise OSError(
                f"token file changed while validating: expected {self.num_tokens} "
                f"tokens, read {token_offset}"
            )

    @staticmethod
    def _read_exact(handle: BinaryIO, num_bytes: int, path: Path) -> bytes:
        data = handle.read(num_bytes)
        if len(data) != num_bytes:
            raise OSError(
                f"short read from {path}: expected {num_bytes} bytes, got {len(data)}"
            )
        return data

    def read_tokens(self, start: int, count: int) -> torch.Tensor:
        self._validate_read(start, count)
        if count == 0:
            return torch.empty(0, dtype=torch.long)
        with self.path.open("rb") as handle:
            handle.seek(start * self.bytes_per_token)
            data = self._read_exact(
                handle, count * self.bytes_per_token, self.path
            )
        return _decode_uint16_le(data)

    def _read_windows(self, starts: Sequence[int], window_length: int) -> torch.Tensor:
        # Open once per batch instead of once per sampled sequence.
        windows: list[torch.Tensor] = []
        num_bytes = window_length * self.bytes_per_token
        with self.path.open("rb") as handle:
            for start in starts:
                self._validate_read(start, window_length)
                handle.seek(start * self.bytes_per_token)
                data = self._read_exact(handle, num_bytes, self.path)
                windows.append(_decode_uint16_le(data))
        return torch.stack(windows, dim=0)


BinaryTokenStream = UInt16TokenStream


__all__ = [
    "BinaryTokenStream",
    "SyntheticTokenStream",
    "TokenBatch",
    "TokenStream",
    "UInt16TokenStream",
]
