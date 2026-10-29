"""Shared runtime utilities for training, evaluation, and benchmarking.

Shared by training, evaluation, and benchmarking. The model section is
validated by
`dilution.config.ModelConfig`; models follow the ``(logits, loss)`` tuple
interface of `dilution.model.DilutionLM`.

This module deliberately contains infrastructure only. The model and the data
stream remain independently testable, while the command-line entry points share
the same device, precision, checkpoint, and metric behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import warnings
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ContextManager, Iterable, Mapping

import torch
import yaml

from .config import ConfigurationError, ModelConfig
from dataclasses import dataclass


CHECKPOINT_FORMAT_VERSION = 1


DEFAULT_DATA_CONFIG: dict[str, Any] = {
    "train_path": None,
    "val_path": None,
    "synthetic": False,
    "vocab_size": None,
    "train_tokens": 1_000_000,
    "val_tokens": 100_000,
    "seed": 1337,
    "pattern_length": 256,
}


DEFAULT_TRAINING_CONFIG: dict[str, Any] = {
    "output_dir": "runs/default",
    "device": "auto",
    "dtype": "auto",
    "compile": False,
    "batch_size": 8,
    "sequence_length": None,
    "gradient_accumulation_steps": 1,
    "max_steps": 1_000,
    "learning_rate": 3.0e-4,
    "min_lr_ratio": 0.1,
    "warmup_steps": 100,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1.0e-8,
    "grad_clip": 1.0,
    "fused_optimizer": True,
    # Route clean-design dilution layers through the fused Triton kernels
    # (forward + v3 backward). Validated 2026-09-01: gradient parity on every
    # input and a trajectory-identical 300-step screen at 3.3x throughput
    # (S4). Runtime knob: CUDA-only, resume-ignored, no-op for other arms.
    "use_dilution_kernel": True,
    "eval_interval": 100,
    "eval_batches": 20,
    "checkpoint_interval": 0,
    "log_interval": 10,
    "seed": 1337,
    "resume": None,
}


@dataclass(frozen=True)
class RuntimeSpec:
    """The concrete execution device and numeric precision."""

    device: torch.device
    dtype: torch.dtype
    dtype_name: str

    @property
    def uses_autocast(self) -> bool:
        return self.dtype != torch.float32


def _merge_defaults(defaults: Mapping[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(defaults)
    result.update(values)
    return result


def _reject_unknown_keys(
    values: Mapping[str, Any], allowed: Iterable[str], section: str
) -> None:
    allowed_keys = set(allowed)
    unknown = sorted(str(key) for key in values if key not in allowed_keys)
    if unknown:
        joined = ", ".join(f"{section}.{key}" if section else str(key) for key in unknown)
        raise ConfigurationError(f"unknown configuration key(s): {joined}")


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML mapping and give malformed inputs an actionable error."""

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            contents = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ConfigurationError(f"configuration file does not exist: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(contents, dict):
        raise ConfigurationError(f"configuration root must be a mapping: {config_path}")
    return contents


def resolve_experiment_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Apply public defaults and validate cross-section training constraints."""

    _reject_unknown_keys(raw, {"model", "data", "training"}, "")
    model = raw.get("model")
    if not isinstance(model, Mapping):
        raise ConfigurationError("configuration must contain a 'model' mapping")
    data_values = raw.get("data", {})
    training_values = raw.get("training", {})
    if not isinstance(data_values, Mapping):
        raise ConfigurationError("'data' must be a mapping")
    if not isinstance(training_values, Mapping):
        raise ConfigurationError("'training' must be a mapping")

    _reject_unknown_keys(data_values, set(DEFAULT_DATA_CONFIG) | {"bos_token"}, "data")  # bos_token: opt-in
    _reject_unknown_keys(training_values, DEFAULT_TRAINING_CONFIG, "training")

    data = _merge_defaults(DEFAULT_DATA_CONFIG, data_values)
    training = _merge_defaults(DEFAULT_TRAINING_CONFIG, training_values)
    # ModelConfig is the single source of truth for model keys and defaults.
    model = ModelConfig.from_dict(dict(model)).to_dict()

    if data["vocab_size"] is None:
        data["vocab_size"] = model["vocab_size"]
    if int(data["vocab_size"]) != int(model["vocab_size"]):
        raise ConfigurationError("data.vocab_size must equal model.vocab_size")
    if training["sequence_length"] is None:
        training["sequence_length"] = model["context_length"]

    positive_ints = (
        ("training.batch_size", training["batch_size"]),
        ("training.sequence_length", training["sequence_length"]),
        ("training.gradient_accumulation_steps", training["gradient_accumulation_steps"]),
        ("training.max_steps", training["max_steps"]),
        ("training.log_interval", training["log_interval"]),
        ("data.train_tokens", data["train_tokens"]),
        ("data.val_tokens", data["val_tokens"]),
    )
    for name, value in positive_ints:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigurationError(f"{name} must be a positive integer")
    for name in ("warmup_steps", "eval_interval", "eval_batches", "checkpoint_interval"):
        value = training[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ConfigurationError(f"training.{name} must be a non-negative integer")
    if training["eval_interval"] > 0 and training["eval_batches"] == 0:
        raise ConfigurationError(
            "training.eval_batches must be positive when evaluation is enabled"
        )
    if int(training["sequence_length"]) > int(model["context_length"]):
        raise ConfigurationError(
            "training.sequence_length cannot exceed model.context_length "
            f"({training['sequence_length']} > {model['context_length']})"
        )
    if float(training["learning_rate"]) <= 0:
        raise ConfigurationError("training.learning_rate must be positive")
    if not 0.0 <= float(training["min_lr_ratio"]) <= 1.0:
        raise ConfigurationError("training.min_lr_ratio must be in [0, 1]")
    if float(training["weight_decay"]) < 0:
        raise ConfigurationError("training.weight_decay must be non-negative")
    if float(training["grad_clip"]) < 0:
        raise ConfigurationError("training.grad_clip must be non-negative")
    if int(training["warmup_steps"]) > int(training["max_steps"]):
        raise ConfigurationError("training.warmup_steps cannot exceed training.max_steps")

    return {"model": model, "data": data, "training": training}


def load_experiment_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    return resolve_experiment_config(load_yaml(path))


def jsonable(value: Any) -> Any:
    """Convert common runtime values to JSON/YAML-safe builtins."""

    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _atomic_write_text(output_path: Path, content: str) -> None:
    # Downstream tooling treats an existing artifact file as complete, so a
    # torn write must never be left behind under the final name.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_yaml(path: str | os.PathLike[str], values: Mapping[str, Any]) -> None:
    _atomic_write_text(Path(path), yaml.safe_dump(jsonable(values), sort_keys=False))


def save_json(path: str | os.PathLike[str], values: Mapping[str, Any]) -> None:
    content = json.dumps(jsonable(values), indent=2, sort_keys=True) + "\n"
    _atomic_write_text(Path(path), content)


def sha256_file(path: str | os.PathLike[str], *, chunk_bytes: int = 1 << 20) -> str:
    """Hash a file without loading the token corpus into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def verify_token_manifest(
    token_path: str | os.PathLike[str],
    split: str,
    *,
    expected_num_tokens: int | None = None,
) -> dict[str, Any]:
    """Verify a prepared uint16 split against its sibling ``manifest.json``."""

    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    path = Path(token_path)
    if not path.is_file():
        raise FileNotFoundError(f"token file does not exist: {path}")
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ConfigurationError(
            f"prepared-data manifest is missing beside {path}: expected {manifest_path}; "
            "prepared corpora must ship a sibling manifest.json with sha256 provenance"
        )
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(
            f"cannot read prepared-data manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise ConfigurationError(f"manifest root must be a mapping: {manifest_path}")
    if manifest.get("manifest_version") != 1:
        raise ConfigurationError(
            f"unsupported manifest_version in {manifest_path}: "
            f"{manifest.get('manifest_version')!r}"
        )
    token_format = manifest.get("token_format")
    required_format = {
        "dtype": "uint16",
        "byte_order": "little",
        "header_bytes": 0,
        "bytes_per_token": 2,
    }
    if not isinstance(token_format, Mapping) or any(
        token_format.get(key) != value for key, value in required_format.items()
    ):
        raise ConfigurationError(
            f"manifest token_format is not headerless little-endian uint16: {manifest_path}"
        )

    manifest_split = "train" if split == "train" else "validation"
    splits = manifest.get("splits")
    entry = splits.get(manifest_split) if isinstance(splits, Mapping) else None
    if not isinstance(entry, Mapping):
        raise ConfigurationError(
            f"manifest has no {manifest_split!r} split mapping: {manifest_path}"
        )
    recorded_path = entry.get("path")
    if not isinstance(recorded_path, str) or not recorded_path:
        raise ConfigurationError(
            f"manifest split {manifest_split!r} has no valid path: {manifest_path}"
        )
    recorded_absolute = (manifest_path.parent / recorded_path).resolve()
    actual_absolute = path.resolve()
    if recorded_absolute != actual_absolute:
        raise ConfigurationError(
            f"manifest split {manifest_split!r} points to {recorded_absolute}, "
            f"not configured token file {actual_absolute}"
        )

    file_size = path.stat().st_size
    if file_size == 0 or file_size % 2:
        raise ConfigurationError(f"token file has an invalid byte size {file_size}: {path}")
    actual_num_tokens = file_size // 2
    recorded_num_tokens = entry.get("num_tokens")
    recorded_num_bytes = entry.get("num_bytes")
    if recorded_num_tokens != actual_num_tokens or recorded_num_bytes != file_size:
        raise ConfigurationError(
            f"manifest size mismatch for {path}: manifest has "
            f"{recorded_num_tokens!r} tokens/{recorded_num_bytes!r} bytes, file has "
            f"{actual_num_tokens} tokens/{file_size} bytes"
        )
    if expected_num_tokens is not None and actual_num_tokens != expected_num_tokens:
        raise ConfigurationError(
            f"data.{split}_tokens={expected_num_tokens} but verified file has "
            f"{actual_num_tokens} tokens: {path}"
        )
    recorded_sha256 = entry.get("sha256")
    if not isinstance(recorded_sha256, str) or len(recorded_sha256) != 64:
        raise ConfigurationError(
            f"manifest split {manifest_split!r} has no valid SHA256: {manifest_path}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256.lower() != recorded_sha256.lower():
        raise ConfigurationError(
            f"SHA256 mismatch for {path}: manifest={recorded_sha256}, "
            f"actual={actual_sha256}"
        )

    return {
        "kind": "prepared_uint16",
        "split": split,
        "manifest_split": manifest_split,
        "token_path": str(actual_absolute),
        "num_tokens": actual_num_tokens,
        "num_bytes": file_size,
        "sha256": actual_sha256,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
    }


def verify_data_provenance(
    data_config: Mapping[str, Any],
    *,
    splits: Iterable[str],
) -> dict[str, Any]:
    """Verify data identity without writing anything to a run directory."""

    requested_splits = tuple(splits)
    if bool(data_config.get("synthetic", False)):
        return {
            "kind": "synthetic",
            "vocab_size": int(data_config["vocab_size"]),
            "seed": int(data_config.get("seed", 0)),
            "pattern_length": int(data_config.get("pattern_length", 256)),
            "splits": {
                split: {"num_tokens": int(data_config[f"{split}_tokens"])}
                for split in requested_splits
            },
        }

    verified: dict[str, dict[str, Any]] = {}
    for split in requested_splits:
        if split not in {"train", "val"}:
            raise ValueError("provenance split must be 'train' or 'val'")
        token_path = data_config.get(f"{split}_path")
        if not token_path:
            raise ConfigurationError(f"data.{split}_path is required")
        verified[split] = verify_token_manifest(
            token_path,
            split,
            expected_num_tokens=int(data_config[f"{split}_tokens"]),
        )
    return {"kind": "prepared_uint16", "splits": verified}


def record_data_provenance(
    data_config: Mapping[str, Any],
    output_dir: str | os.PathLike[str],
    *,
    splits: Iterable[str],
    verified: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify input data and preserve its manifest/provenance in the run.

    Pass ``verified`` (a :func:`verify_data_provenance` result) when data
    identity was already checked, so validation can happen -- and fail -- before
    the run directory is created or modified.
    """

    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    provenance = (
        verified
        if verified is not None
        else verify_data_provenance(data_config, splits=splits)
    )
    if provenance["kind"] == "synthetic":
        save_json(destination_dir / "data_provenance.json", provenance)
        return provenance

    manifests: dict[Path, list[str]] = {}
    for split, split_provenance in provenance["splits"].items():
        manifest_path = Path(split_provenance["manifest_path"])
        manifests.setdefault(manifest_path, []).append(split)

    for index, (manifest_path, manifest_splits) in enumerate(manifests.items()):
        if len(manifests) == 1:
            copy_name = "data_manifest.json"
        else:
            split_label = "_".join(manifest_splits) or str(index)
            copy_name = f"data_manifest_{split_label}.json"
        copy_path = destination_dir / copy_name
        temporary = copy_path.with_name(f".{copy_path.name}.tmp-{os.getpid()}")
        try:
            shutil.copyfile(manifest_path, temporary)
            os.replace(temporary, copy_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        copied_sha256 = sha256_file(copy_path)
        for split in manifest_splits:
            provenance["splits"][split]["manifest_copy"] = str(copy_path.resolve())
            provenance["splits"][split]["manifest_copy_sha256"] = copied_sha256

    save_json(destination_dir / "data_provenance.json", provenance)
    return provenance


def append_jsonl(path: str | os.PathLike[str], values: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        json.dump(jsonable(values), handle, sort_keys=True)
        handle.write("\n")
        handle.flush()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_runtime(device: str = "auto", dtype: str = "auto") -> RuntimeSpec:
    """Resolve `auto` to CUDA/BF16 when usable, otherwise CPU/FP32."""

    requested_device = str(device).lower()
    if requested_device == "auto":
        concrete_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        concrete_device = torch.device(requested_device)
    if concrete_device.type == "cuda" and not torch.cuda.is_available():
        raise ConfigurationError("CUDA was requested but torch.cuda.is_available() is false")
    if concrete_device.type not in {"cuda", "cpu"}:
        raise ConfigurationError("supported devices are 'auto', 'cuda', and 'cpu'")

    requested_dtype = str(dtype).lower().replace("torch.", "")
    if requested_dtype in {"auto", "bf16", "bfloat16"}:
        can_use_bf16 = concrete_device.type == "cuda" and torch.cuda.is_bf16_supported()
        if requested_dtype == "auto" and can_use_bf16:
            concrete_dtype = torch.bfloat16
        elif requested_dtype in {"bf16", "bfloat16"} and can_use_bf16:
            concrete_dtype = torch.bfloat16
        else:
            concrete_dtype = torch.float32
    elif requested_dtype in {"fp32", "float32", "float"}:
        concrete_dtype = torch.float32
    else:
        raise ConfigurationError("dtype must be one of: auto, bfloat16/bf16, float32/fp32")

    dtype_name = "bfloat16" if concrete_dtype == torch.bfloat16 else "float32"
    return RuntimeSpec(device=concrete_device, dtype=concrete_dtype, dtype_name=dtype_name)


def autocast_context(runtime: RuntimeSpec) -> ContextManager[Any]:
    if not runtime.uses_autocast:
        return nullcontext()
    return torch.autocast(device_type=runtime.device.type, dtype=runtime.dtype)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state(train_generator: torch.Generator | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if train_generator is not None:
        state["train_generator"] = train_generator.get_state()
    return state


def restore_rng_state(
    state: Mapping[str, Any], train_generator: torch.Generator | None = None
) -> None:
    if "python" in state:
        random.setstate(tuple(state["python"]))
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        cuda_states = state["torch_cuda"]
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
        else:
            warnings.warn(
                f"checkpoint stores CUDA RNG state for {len(cuda_states)} "
                f"device(s) but {torch.cuda.device_count()} are visible; "
                "skipping CUDA RNG restore",
                RuntimeWarning,
                stacklevel=2,
            )
    if train_generator is not None and "train_generator" in state:
        train_generator.set_state(state["train_generator"])


def environment_info(runtime: RuntimeSpec) -> dict[str, Any]:
    info: dict[str, Any] = {
        "created_at": utc_now(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(runtime.device),
        "dtype": runtime.dtype_name,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if runtime.device.type == "cuda":
        index = runtime.device.index
        if index is None:
            index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        info.update(
            {
                "cuda_device_index": index,
                "cuda_device_name": properties.name,
                "cuda_device_total_memory_bytes": properties.total_memory,
                "cuda_bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return info


def _with_bos(data_config: Mapping[str, Any], stream):
    if data_config.get("bos_token") is not None:  # opt-in: every window starts with this token
        stream.bos_token = int(data_config["bos_token"])
    return stream


def create_token_stream(data_config: Mapping[str, Any], split: str):
    """Construct a binary or deterministic synthetic token stream."""

    from .data import TokenStream

    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    vocab_size = int(data_config["vocab_size"])
    if bool(data_config.get("synthetic", False)):
        num_tokens = int(data_config[f"{split}_tokens"])
        seed = int(data_config.get("seed", 0)) + (0 if split == "train" else 1)
        pattern_length = int(data_config.get("pattern_length", 256))
        return _with_bos(data_config, TokenStream.from_synthetic(
            num_tokens=num_tokens,
            vocab_size=vocab_size,
            seed=seed,
            pattern_length=pattern_length,
        ))

    path = data_config.get(f"{split}_path")
    if not path:
        raise ConfigurationError(
            f"data.{split}_path is required unless data.synthetic is true"
        )
    stream = TokenStream.from_binary(path, vocab_size=vocab_size)
    if data_config.get("bos_token") is not None:  # opt-in: every window starts with this token
        stream.bos_token = int(data_config["bos_token"])
    return stream


def make_adamw(
    parameters: Iterable[torch.nn.Parameter],
    training_config: Mapping[str, Any],
    runtime: RuntimeSpec,
) -> tuple[torch.optim.AdamW, bool]:
    """Create AdamW, using the fused implementation only on supported CUDA builds."""

    kwargs: dict[str, Any] = {
        "lr": float(training_config["learning_rate"]),
        "betas": (float(training_config["beta1"]), float(training_config["beta2"])),
        "eps": float(training_config["eps"]),
        "weight_decay": float(training_config["weight_decay"]),
    }
    # Materialize the parameter iterable: a failed fused construction below
    # must not leave the fallback constructor an exhausted generator.
    parameters = list(parameters)
    want_fused = (
        bool(training_config.get("fused_optimizer", True))
        and runtime.device.type == "cuda"
    )
    used_fused = False
    if want_fused:
        try:
            optimizer = torch.optim.AdamW(parameters, fused=True, **kwargs)
            used_fused = True
            return optimizer, used_fused
        except (RuntimeError, TypeError):
            pass
    return torch.optim.AdamW(parameters, **kwargs), used_fused


def learning_rate_for_step(step: int, training_config: Mapping[str, Any]) -> float:
    """Warm up linearly, then decay to `min_lr_ratio` with a cosine."""

    base_lr = float(training_config["learning_rate"])
    warmup_steps = int(training_config["warmup_steps"])
    max_steps = int(training_config["max_steps"])
    min_ratio = float(training_config["min_lr_ratio"])
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    decay_steps = max_steps - warmup_steps
    if decay_steps <= 1:
        return base_lr * min_ratio
    progress = min(1.0, max(0.0, (step - warmup_steps) / float(decay_steps - 1)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)


def set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def atomic_torch_save(values: Mapping[str, Any], path: str | os.PathLike[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        torch.save(dict(values), temporary)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(
    path: str | os.PathLike[str],
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:  # `weights_only` was added after older supported PyTorch releases.
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint root must be a mapping: {checkpoint_path}")
    version = int(checkpoint.get("format_version", 0))
    if version > CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format {version} is newer than supported format "
            f"{CHECKPOINT_FORMAT_VERSION}"
        )
    if "model" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError(f"checkpoint is missing model weights or model_config: {checkpoint_path}")
    return checkpoint


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@torch.no_grad()
def evaluate_language_model(
    model: torch.nn.Module,
    stream: Any,
    *,
    batch_size: int,
    sequence_length: int,
    max_batches: int | None,
    runtime: RuntimeSpec,
) -> dict[str, Any]:
    """Compute token-weighted loss, perplexity, and next-token accuracy.

    Models follow DilutionLM's interface: ``logits, loss = model(input_ids, targets)``
    where supplying targets yields full-sequence logits.
    """

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    batches = 0
    try:
        iterator = stream.sequential_batches(
            batch_size=batch_size,
            sequence_length=sequence_length,
            max_batches=max_batches,
            device=runtime.device,
        )
        for input_ids, labels in iterator:
            chunked = getattr(getattr(model, "_orig_mod", model), "eval_chunked", None)
            # Evaluation never needs autograd: without this the whole forward graph
            # (and the dilution kernels' saved statistics) stayed alive per batch.
            with torch.no_grad(), autocast_context(runtime):
                if chunked is not None:
                    # Per-chunk lm_head + loss: no full-sequence logits tensor (S11).
                    predictions, loss, _ = chunked(input_ids, labels)
                    logits = None
                else:
                    logits, loss = model(input_ids, labels)
            if loss is None:
                raise RuntimeError("model did not return a loss when labels were supplied")
            token_count = labels.numel()
            total_loss += float(loss.detach().float()) * token_count
            if logits is not None:
                predictions = logits.detach().argmax(dim=-1)
            total_correct += int((predictions == labels).sum().item())
            total_tokens += token_count
            batches += 1
    finally:
        model.train(was_training)

    if total_tokens == 0:
        raise ValueError("evaluation stream did not yield any target tokens")
    mean_loss = total_loss / total_tokens
    perplexity = math.exp(mean_loss) if mean_loss < 700.0 else math.inf
    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "accuracy": total_correct / total_tokens,
        "tokens": total_tokens,
        "batches": batches,
    }
