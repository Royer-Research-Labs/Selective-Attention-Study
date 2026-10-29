"""Single-GPU training entry point for dilution-attention experiments.

Run-record semantics: append-only metrics.jsonl, refuse-to-clobber,
resume guards on config and data-content identity, and full-schedule screening
via --stop-after-steps.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .config import ConfigurationError, ModelConfig, build_model
from .runtime import (
    CHECKPOINT_FORMAT_VERSION,
    append_jsonl,
    atomic_torch_save,
    autocast_context,
    capture_rng_state,
    create_token_stream,
    environment_info,
    evaluate_language_model,
    jsonable,
    learning_rate_for_step,
    load_checkpoint,
    load_experiment_config,
    make_adamw,
    parameter_count,
    record_data_provenance,
    resolve_runtime,
    restore_rng_state,
    verify_data_provenance,
    save_json,
    save_yaml,
    seed_everything,
    set_optimizer_learning_rate,
    synchronize,
    utc_now,
)


def _save_checkpoint(
    *,
    output_dir: Path,
    step: int,
    model: torch.nn.Module,
    model_config: ModelConfig,
    optimizer: torch.optim.Optimizer,
    resolved_config: Mapping[str, Any],
    train_generator: torch.Generator,
    tokens_seen: int,
    best_val_loss: float | None,
) -> Path:
    values = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "created_at": utc_now(),
        # `step` is the number of completed optimizer steps and therefore the
        # next zero-based step index after resume.
        "step": step,
        "tokens_seen": tokens_seen,
        "model_config": model_config.to_dict(),
        "resolved_config": jsonable(resolved_config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_state": capture_rng_state(train_generator),
        "best_val_loss": best_val_loss,
    }
    # A single rolling file: each interval save atomically overwrites latest.pt
    # (tmp + os.replace, so a crash mid-write cannot corrupt it). There are no
    # numbered per-step checkpoints; the trade-off is no rewind to earlier steps.
    latest_path = output_dir / "latest.pt"
    atomic_torch_save(values, latest_path)
    return latest_path


def _keep_milestone(output_dir: Path, latest_path: Path, step: int) -> Path:
    """Copy the just-written latest.pt to checkpoint_step_<step>.pt (atomic), so a milestone
    such as the 1x-Chinchilla point survives later training on the same schedule."""
    kept = output_dir / f"checkpoint_step_{step:07d}.pt"
    tmp = kept.with_name(kept.name + ".tmp")
    shutil.copyfile(latest_path, tmp)
    os.replace(tmp, kept)
    return kept


def _configs_match(left: ModelConfig, right_values: Mapping[str, Any]) -> bool:
    right = ModelConfig.from_dict(dict(right_values))
    return left.to_dict() == right.to_dict()


def _resume_config_differences(
    checkpoint: Mapping[str, Any], current: Mapping[str, Any]
) -> list[str]:
    """Return scientific config changes that would invalidate continuation."""

    saved = checkpoint.get("resolved_config")
    if not isinstance(saved, Mapping):
        return ["checkpoint.resolved_config is missing"]
    differences: list[str] = []
    for section in ("data", "training"):
        saved_section = saved.get(section)
        current_section = current.get(section)
        if not isinstance(saved_section, Mapping):
            differences.append(f"checkpoint.resolved_config.{section} is missing")
            continue
        if not isinstance(current_section, Mapping):
            differences.append(f"current resolved config {section} is missing")
            continue
        # log_interval only changes console/metric cadence, never the science.
        # device/dtype are runtime spellings ("auto" vs "cuda"/"bf16") — the
        # RESOLVED dtype is compared separately below, because precision is
        # scientific but its spelling is not. compile_prefix changes kernel
        # fusion, not semantics.
        ignored = (
            {"output_dir", "resume", "log_interval", "device", "dtype", "use_dilution_kernel"}
            if section == "training"
            else set()
        )
        keys = (set(saved_section) | set(current_section)) - ignored
        for key in sorted(keys):
            saved_value = saved_section.get(key, "<missing>")
            current_value = current_section.get(key, "<missing>")
            if saved_value != current_value:
                differences.append(
                    f"{section}.{key}: checkpoint={saved_value!r}, "
                    f"current={current_value!r}"
                )
    return differences


def _resume_resolved_dtype_difference(
    checkpoint: Mapping[str, Any], resolved_dtype_name: str
) -> list[str]:
    """Compare the RESOLVED numeric precision, not its config spelling."""

    saved = checkpoint.get("resolved_config")
    saved_runtime = saved.get("runtime") if isinstance(saved, Mapping) else None
    saved_dtype = saved_runtime.get("dtype") if isinstance(saved_runtime, Mapping) else None
    if saved_dtype is None:
        return ["checkpoint records no resolved runtime dtype"]
    if str(saved_dtype) != resolved_dtype_name:
        return [
            f"training.dtype (resolved): checkpoint={saved_dtype!r}, "
            f"current={resolved_dtype_name!r}"
        ]
    return []


def _resume_provenance_differences(
    checkpoint: Mapping[str, Any], current: Mapping[str, Any]
) -> list[str]:
    """Compare content identities, not run-local manifest-copy paths."""

    resolved = checkpoint.get("resolved_config")
    saved = resolved.get("data_provenance") if isinstance(resolved, Mapping) else None
    if not isinstance(saved, Mapping):
        return ["checkpoint has no verified data_provenance"]
    if saved.get("kind") != current.get("kind"):
        return [
            f"data kind: checkpoint={saved.get('kind')!r}, "
            f"current={current.get('kind')!r}"
        ]
    if current.get("kind") == "synthetic":
        return [] if saved == current else ["synthetic data provenance changed"]

    saved_splits = saved.get("splits")
    current_splits = current.get("splits")
    if not isinstance(saved_splits, Mapping) or not isinstance(current_splits, Mapping):
        return ["prepared-data split provenance is missing"]
    differences: list[str] = []
    if set(saved_splits) != set(current_splits):
        differences.append(
            "data_provenance split set: "
            f"checkpoint={sorted(saved_splits)!r}, current={sorted(current_splits)!r}"
        )
    stable_fields = ("sha256", "manifest_sha256", "num_tokens", "num_bytes")
    for split, current_split in current_splits.items():
        saved_split = saved_splits.get(split)
        if not isinstance(saved_split, Mapping) or not isinstance(current_split, Mapping):
            differences.append(f"data_provenance.splits.{split} is missing")
            continue
        for field in stable_fields:
            if saved_split.get(field) != current_split.get(field):
                differences.append(
                    f"data_provenance.splits.{split}.{field}: "
                    f"checkpoint={saved_split.get(field)!r}, "
                    f"current={current_split.get(field)!r}"
                )
    return differences


def _format_compact(value: int | float) -> str:
    magnitude = abs(float(value))
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= scale:
            return f"{float(value) / scale:.2f}{suffix}"
    return f"{float(value):.0f}"


def _format_bytes(value: int | float) -> str:
    amount = float(value)
    for scale, suffix in (
        (1024**3, "GiB"),
        (1024**2, "MiB"),
        (1024, "KiB"),
    ):
        if abs(amount) >= scale:
            return f"{amount / scale:.2f} {suffix}"
    return f"{amount:.0f} B"


def _format_progress(values: Mapping[str, Any], max_steps: int | None) -> str:
    step = int(values.get("step", 0))
    if max_steps is None or max_steps <= 0:
        return f"step {step}"
    width = len(str(max_steps))
    percent = 100.0 * step / max_steps
    return f"step {step:>{width}}/{max_steps} ({percent:5.1f}%)"


def _format_console_event(
    values: Mapping[str, Any], *, max_steps: int | None = None
) -> str:
    """Format one concise terminal line without changing metrics.jsonl."""

    event = str(values.get("event", "event"))
    label = {
        "evaluation": "eval",
        "checkpoint": "checkpoint",
    }.get(event, event)
    progress = _format_progress(values, max_steps)

    if event in {"start", "resume"}:
        compile_mode = "compile on" if values.get("compile") else "compile off"
        fused_mode = "fused AdamW" if values.get("fused_optimizer") else "AdamW"
        return " | ".join(
            (
                f"{label:<10} {progress}",
                str(values.get("device", "unknown")),
                str(values.get("dtype", "unknown")),
                f"{_format_compact(int(values.get('parameters', 0)))} params",
                str(values.get("attention", "unknown")),
                compile_mode,
                fused_mode,
            )
        )

    if event == "train":
        fields = [
            f"{label:<10} {progress}",
            f"tokens {_format_compact(int(values['tokens_seen']))}",
            f"loss {float(values['loss']):.4f}",
            f"ppl {float(values['perplexity']):.2f}",
            f"lr {float(values['learning_rate']):.3e}",
        ]
        if values.get("grad_norm") is not None:
            fields.append(f"grad {float(values['grad_norm']):.3f}")
        fields.append(f"{_format_compact(float(values['tokens_per_second']))} tok/s")
        if values.get("max_memory_allocated_bytes") is not None:
            fields.append(f"mem {_format_bytes(int(values['max_memory_allocated_bytes']))}")
        return " | ".join(fields)

    if event == "evaluation":
        return " | ".join(
            (
                f"{label:<10} {progress}",
                f"tokens {_format_compact(int(values['tokens_seen']))}",
                f"loss {float(values['loss']):.4f}",
                f"ppl {float(values['perplexity']):.2f}",
                f"acc {100.0 * float(values['accuracy']):.2f}%",
                f"eval {_format_compact(int(values['tokens']))} tokens",
            )
        )

    if event == "checkpoint":
        return f"{label:<10} {progress} | {values.get('path', '<unknown path>')}"

    if event == "complete":
        best = values.get("best_val_loss")
        best_text = "n/a" if best is None else f"{float(best):.4f}"
        return " | ".join(
            (
                f"{label:<10} {progress}",
                f"tokens {_format_compact(int(values['tokens_seen']))}",
                f"best val {best_text}",
                str(values.get("checkpoint", "<unknown path>")),
            )
        )

    return json.dumps(jsonable(values), sort_keys=True)


def _print_event(values: Mapping[str, Any], *, max_steps: int | None = None) -> None:
    print(_format_console_event(values, max_steps=max_steps), flush=True)


_RUN_ARTIFACT_PATTERNS = (
    "metrics.jsonl",
    "latest.pt",
    "checkpoint_step_*.pt",
    "resolved_config.yaml",
    "environment.json",
    "data_provenance.json",
    "data_manifest*.json",
)


def _existing_run_artifacts(output_dir: Path) -> list[Path]:
    """Return run records already present in a directory."""

    if not output_dir.is_dir():
        return []
    found: list[Path] = []
    for pattern in _RUN_ARTIFACT_PATTERNS:
        found.extend(sorted(output_dir.glob(pattern)))
    return found


def _has_complete_event(metrics_path: Path) -> bool:
    if not metrics_path.is_file():
        return False
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                if json.loads(line).get("event") == "complete":
                    return True
            except json.JSONDecodeError:
                continue
    return False


def run_training(
    config_path: str | Path,
    *,
    resume_override: str | Path | None = None,
    output_dir_override: str | Path | None = None,
    device_override: str | None = None,
    dtype_override: str | None = None,
    compile_override: bool | None = None,
    seed_override: int | None = None,
    batch_size_override: int | None = None,
    gradient_accumulation_steps_override: int | None = None,
    overwrite_run: bool = False,
    stop_after_steps: int = 0,
    stop_at_step: int = 0,
    keep_checkpoint_at: Sequence[int] | None = None,
) -> dict[str, Any] | None:
    """Train an experiment and return a compact summary for tests/callers.

    ``stop_at_step`` trains on the unchanged full schedule to that step, writes latest.pt
    and a kept ``checkpoint_step_<N>.pt``, emits a ``stopped`` event and returns; resuming
    from latest.pt continues the same schedule. ``keep_checkpoint_at`` keeps such copies at
    the listed steps without stopping. (``stop_after_steps`` is the throwaway screening mode
    and writes no checkpoint at all.)
    """

    resolved_config = load_experiment_config(config_path)
    training = resolved_config["training"]
    data_config = resolved_config["data"]
    if resume_override is not None:
        training["resume"] = str(resume_override)
    # Micro-batch / accumulation overrides trade memory for step count without
    # changing the effective batch, as long as their product (tokens/step) is
    # preserved. Use them to fit a large model on a smaller card.
    if batch_size_override is not None:
        if isinstance(batch_size_override, bool) or batch_size_override <= 0:
            raise ConfigurationError("batch size override must be a positive integer")
        training["batch_size"] = batch_size_override
    if gradient_accumulation_steps_override is not None:
        if (
            isinstance(gradient_accumulation_steps_override, bool)
            or gradient_accumulation_steps_override <= 0
        ):
            raise ConfigurationError(
                "gradient accumulation override must be a positive integer"
            )
        training["gradient_accumulation_steps"] = gradient_accumulation_steps_override
    if output_dir_override is not None:
        training["output_dir"] = str(output_dir_override)
    if device_override is not None:
        training["device"] = device_override
    if dtype_override is not None:
        training["dtype"] = dtype_override
    if compile_override is not None:
        training["compile"] = compile_override
    if seed_override is not None:
        if isinstance(seed_override, bool) or seed_override < 0:
            raise ConfigurationError("seed override must be a non-negative integer")
        training["seed"] = seed_override

    # Screening that would cover the whole schedule is a full run mislabeled as
    # a screen; refuse before any run records are written.
    stop_after = int(stop_after_steps or 0)
    if stop_after > 0 and stop_after >= int(training["max_steps"]):
        raise ConfigurationError(
            f"--stop-after-steps {stop_after} >= max_steps {training['max_steps']}: this "
            "would run the full schedule but be recorded as a screen; run without "
            "--stop-after-steps"
        )
    stop_at = int(stop_at_step or 0)
    milestones = {int(s) for s in (keep_checkpoint_at or ())}
    if stop_at:
        if stop_after:
            raise ConfigurationError("--stop-at-step and --stop-after-steps are mutually exclusive")
        if not 0 < stop_at < int(training["max_steps"]):
            raise ConfigurationError(
                f"--stop-at-step {stop_at} must lie strictly inside the schedule "
                f"(0 < N < max_steps={training['max_steps']})"
            )
        milestones.add(stop_at)
    bad_milestones = sorted(s for s in milestones if not 0 < s <= int(training["max_steps"]))
    if bad_milestones:
        raise ConfigurationError(f"checkpoint milestones outside the schedule: {bad_milestones}")

    runtime = resolve_runtime(str(training["device"]), str(training["dtype"]))
    output_dir = Path(training["output_dir"])
    metrics_path = output_dir / "metrics.jsonl"
    resume_path = training.get("resume")

    # A completed or in-progress run's records are scientific evidence; a fresh
    # invocation must never clobber them implicitly.
    if overwrite_run and resume_path:
        raise ConfigurationError("overwrite_run cannot be combined with resume")
    existing_artifacts = _existing_run_artifacts(output_dir)
    pending_deletion: list[Path] = []
    if not resume_path and existing_artifacts:
        if not overwrite_run:
            names = ", ".join(path.name for path in existing_artifacts[:3])
            if len(existing_artifacts) > 3:
                names += ", ..."
            raise ConfigurationError(
                f"output directory {output_dir} already contains run records "
                f"({names}); pass --resume to continue that run or "
                "--overwrite-run to discard it and start fresh"
            )
        # Deleted only after the data has been verified below, so a run that cannot
        # start (missing corpus, wrong hashes) leaves the old records intact.
        pending_deletion = list(existing_artifacts)

    model_config = ModelConfig.from_dict(dict(resolved_config["model"]))
    start_step = 0
    tokens_seen = 0
    best_val_loss: float | None = None
    checkpoint: dict[str, Any] | None = None
    if resume_path:
        checkpoint = load_checkpoint(resume_path, map_location="cpu")
        if not _configs_match(model_config, checkpoint["model_config"]):
            raise ConfigurationError(
                "checkpoint model_config does not match the requested experiment model"
            )
        resume_differences = _resume_config_differences(checkpoint, resolved_config)
        resume_differences += _resume_resolved_dtype_difference(checkpoint, runtime.dtype_name)
        if resume_differences:
            formatted = "\n  - ".join(resume_differences)
            raise ConfigurationError(
                "resume would change scientific data/training settings:\n  - " + formatted
            )
        if "optimizer" not in checkpoint:
            raise ValueError("training resume checkpoint is missing optimizer state")
        start_step = int(checkpoint.get("step", 0))
        # Reject an impossible stop before anything in the run directory is touched
        # (latest.pt re-anchoring and the resume event come later).
        if stop_at and stop_at <= start_step:
            raise ConfigurationError(
                f"--stop-at-step {stop_at} is not after the resumed step {start_step}"
            )
        tokens_seen = int(checkpoint.get("tokens_seen", 0))
        saved_best = checkpoint.get("best_val_loss")
        best_val_loss = float(saved_best) if saved_best is not None else None
        if start_step < 0 or start_step > int(training["max_steps"]):
            raise ConfigurationError(
                f"checkpoint step {start_step} is outside configured "
                f"max_steps={training['max_steps']}"
            )
        if start_step == int(training["max_steps"]):
            if _has_complete_event(metrics_path):
                raise ConfigurationError(
                    f"checkpoint already completed max_steps={training['max_steps']}; "
                    "resuming would only append duplicate events to metrics.jsonl"
                )
            # The run finished its last step but was killed between the final
            # checkpoint save and the terminal event. Finalize idempotently so
            # suite auto-resume does not dead-end on a run that actually completed.
            complete_event = {
                "event": "complete",
                "time": utc_now(),
                "step": start_step,
                "tokens_seen": tokens_seen,
                "best_val_loss": best_val_loss,
                "checkpoint": str(output_dir / "latest.pt"),
            }
            append_jsonl(metrics_path, complete_event)
            _print_event(complete_event, max_steps=int(training["max_steps"]))
            return complete_event

    provenance_splits = ["train"]
    if int(training["eval_interval"]) > 0:
        provenance_splits.append("val")
    # Verify data identity before writing anything into the run directory so a
    # rejected resume cannot alter an existing run's recorded provenance.
    data_provenance = verify_data_provenance(data_config, splits=provenance_splits)
    if checkpoint is not None:
        provenance_differences = _resume_provenance_differences(
            checkpoint, data_provenance
        )
        if provenance_differences:
            formatted = "\n  - ".join(provenance_differences)
            raise ConfigurationError(
                "resume data content identity changed:\n  - " + formatted
            )
    for artifact in pending_deletion:     # --overwrite-run, now that the data checked out
        artifact.unlink(missing_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_provenance = record_data_provenance(
        data_config, output_dir, splits=provenance_splits, verified=data_provenance
    )

    # A rewind to a numbered checkpoint abandons the timeline latest.pt points
    # at. Re-anchor latest.pt to the resumed state immediately so a crash before
    # the next checkpoint boundary cannot silently auto-resume the old timeline.
    if checkpoint is not None and resume_path:
        latest_path = output_dir / "latest.pt"
        if Path(resume_path).resolve() != latest_path.resolve():
            atomic_torch_save(checkpoint, latest_path)

    seed = int(training["seed"])
    seed_everything(seed)
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(seed + 1)

    model = build_model(model_config).to(runtime.device)
    # CUDA-only: on CPU the first call would pay a long Inductor compile
    # (smoke tests and CI stay fast); semantics are unchanged either way.
    if bool(training.get("use_dilution_kernel", False)) and runtime.device.type == "cuda":
        model.enable_dilution_kernel()
    optimizer, fused_optimizer = make_adamw(model.parameters(), training, runtime)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])

    train_stream = create_token_stream(data_config, "train")
    validation_stream = None
    if int(training["eval_interval"]) > 0:
        validation_stream = create_token_stream(data_config, "val")

    train_model: torch.nn.Module = model
    compile_enabled = bool(training.get("compile", False))
    if compile_enabled:
        if not hasattr(torch, "compile"):
            raise RuntimeError("training.compile requires a PyTorch build with torch.compile")
        train_model = torch.compile(model)

    # Restore only after model/optimizer construction, which consume RNG state.
    if checkpoint is not None and "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"], train_generator)

    run_config = {
        **resolved_config,
        "data_provenance": data_provenance,
        "runtime": {
            "device": str(runtime.device),
            "dtype": runtime.dtype_name,
            "compile": compile_enabled,
            "fused_optimizer": fused_optimizer,
            "parameter_count": parameter_count(model),
        },
    }
    save_yaml(output_dir / "resolved_config.yaml", run_config)
    save_json(output_dir / "environment.json", environment_info(runtime))

    batch_size = int(training["batch_size"])
    sequence_length = int(training["sequence_length"])
    accumulation_steps = int(training["gradient_accumulation_steps"])
    max_steps = int(training["max_steps"])
    tokens_per_step = batch_size * sequence_length * accumulation_steps

    start_event = {
        "event": "start" if start_step == 0 else "resume",
        "time": utc_now(),
        "step": start_step,
        "tokens_seen": tokens_seen,
        "parameters": parameter_count(model),
        "attention": model_config.attention,
        "device": str(runtime.device),
        "dtype": runtime.dtype_name,
        "compile": compile_enabled,
        "fused_optimizer": fused_optimizer,
    }
    append_jsonl(metrics_path, start_event)
    _print_event(start_event, max_steps=max_steps)

    train_model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_steps = 0
    window_tokens = 0
    last_eval_step = -1
    last_saved_step = -1
    synchronize(runtime.device)
    window_start = time.perf_counter()

    # Screening mode: stop early while keeping the FULL-length LR schedule.
    # learning_rate_for_step() reads training["max_steps"], so lowering
    # max_steps would compress warmup and decay and produce numbers that do NOT
    # match the step-N evals of full runs. Halting the loop leaves the schedule
    # untouched.
    loop_end = stop_after if stop_after > 0 else max_steps
    if stop_at > 0:
        # Checkpointed stop: same full-length schedule, the loop just ends early.
        # (stop_at > start_step was checked before any run artifact was modified.)
        loop_end = stop_at

    for step_index in range(start_step, loop_end):
        learning_rate = learning_rate_for_step(step_index, training)
        set_optimizer_learning_rate(optimizer, learning_rate)
        step_loss = 0.0

        for _ in range(accumulation_steps):
            input_ids, labels = train_stream.sample_batch(
                batch_size=batch_size,
                sequence_length=sequence_length,
                generator=train_generator,
                device=runtime.device,
            )
            with autocast_context(runtime):
                _, micro_loss = train_model(input_ids, labels)
                if micro_loss is None:
                    raise RuntimeError("model did not return a loss when labels were supplied")
                scaled_loss = micro_loss / accumulation_steps
            scaled_loss.backward()
            step_loss += float(micro_loss.detach().float()) / accumulation_steps

        grad_clip = float(training["grad_clip"])
        grad_norm: float | None = None
        if grad_clip > 0:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_norm = float(norm.detach().float())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        completed_step = step_index + 1
        tokens_seen += tokens_per_step
        running_loss += step_loss
        running_steps += 1
        window_tokens += tokens_per_step

        if completed_step % int(training["log_interval"]) == 0 or completed_step == max_steps:
            synchronize(runtime.device)
            elapsed = max(time.perf_counter() - window_start, 1.0e-12)
            train_event: dict[str, Any] = {
                "event": "train",
                "time": utc_now(),
                "step": completed_step,
                "tokens_seen": tokens_seen,
                "loss": running_loss / running_steps,
                "perplexity": math.exp(min(running_loss / running_steps, 80.0)),
                "learning_rate": learning_rate,
                "grad_norm": grad_norm,
                "tokens_per_second": window_tokens / elapsed,
            }
            if runtime.device.type == "cuda":
                train_event["max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
                    runtime.device
                )
            append_jsonl(metrics_path, train_event)
            _print_event(train_event, max_steps=max_steps)
            running_loss = 0.0
            running_steps = 0
            window_tokens = 0
            window_start = time.perf_counter()

        eval_interval = int(training["eval_interval"])
        # Milestone steps (including a checkpointed stop) are evaluated here, before the
        # checkpoint below is written, so the kept checkpoint carries its own validation.
        if validation_stream is not None and (
            completed_step % eval_interval == 0 or completed_step in milestones
        ):
            synchronize(runtime.device)
            overhead_started = time.perf_counter()
            metrics = evaluate_language_model(
                model,
                validation_stream,
                batch_size=batch_size,
                sequence_length=sequence_length,
                max_batches=int(training["eval_batches"]),
                runtime=runtime,
            )
            eval_event = {
                "event": "evaluation",
                "time": utc_now(),
                "step": completed_step,
                "tokens_seen": tokens_seen,
                **metrics,
            }
            append_jsonl(metrics_path, eval_event)
            _print_event(eval_event, max_steps=max_steps)
            best_val_loss = (
                metrics["loss"]
                if best_val_loss is None
                else min(best_val_loss, float(metrics["loss"]))
            )
            last_eval_step = completed_step
            train_model.train()
            # Do not charge deterministic evaluation to training throughput.
            window_start += time.perf_counter() - overhead_started

        checkpoint_interval = int(training["checkpoint_interval"])
        # `loop_end == max_steps` is false when screening truncates the run: screens
        # are throwaway and write no checkpoints. A checkpointed stop (stop_at) does
        # write them, and milestone steps are always saved and kept.
        is_milestone = completed_step in milestones
        if (
            (loop_end == max_steps or stop_at > 0)
            and ((checkpoint_interval > 0 and completed_step % checkpoint_interval == 0) or is_milestone)
            and completed_step < max_steps
        ):
            synchronize(runtime.device)
            overhead_started = time.perf_counter()
            saved_path = _save_checkpoint(
                output_dir=output_dir,
                step=completed_step,
                model=model,
                model_config=model_config,
                optimizer=optimizer,
                resolved_config=run_config,
                train_generator=train_generator,
                tokens_seen=tokens_seen,
                best_val_loss=best_val_loss,
            )
            checkpoint_event = {
                "event": "checkpoint",
                "time": utc_now(),
                "step": completed_step,
                "path": str(saved_path),
            }
            if is_milestone:
                checkpoint_event["kept"] = str(_keep_milestone(output_dir, saved_path, completed_step))
            append_jsonl(metrics_path, checkpoint_event)
            _print_event(checkpoint_event, max_steps=max_steps)
            last_saved_step = completed_step
            # Checkpoint serialization may synchronize CUDA and is not model work.
            window_start += time.perf_counter() - overhead_started

    if stop_at > 0:
        # A deliberate, resumable stop on the full schedule (e.g. the 1x-Chinchilla point of
        # a 2x schedule). The loop evaluated at the stop step and then wrote and kept the
        # step-N checkpoint, so both carry the same validation result; record why it ended.
        stopped_event = {
            "event": "stopped",
            "time": utc_now(),
            "step": stop_at,
            "schedule_max_steps": max_steps,
            "tokens_seen": tokens_seen,
            "best_val_loss": best_val_loss,
            "checkpoint": str(output_dir / "latest.pt"),
            "kept": str(output_dir / f"checkpoint_step_{stop_at:07d}.pt"),
        }
        append_jsonl(metrics_path, stopped_event)
        _print_event(stopped_event, max_steps=max_steps)
        return stopped_event

    if stop_after > 0 and loop_end < max_steps:
        # A screen, not a run. Emit a distinct terminal event so nothing
        # downstream mistakes it for a completed run, evaluate once at the stop
        # point, and skip the checkpoint (screens are throwaway).
        if validation_stream is not None and last_eval_step != loop_end:
            metrics = evaluate_language_model(
                model, validation_stream, batch_size=batch_size,
                sequence_length=sequence_length,
                max_batches=int(training["eval_batches"]), runtime=runtime,
            )
            eval_event = {"event": "evaluation", "time": utc_now(),
                          "step": loop_end, "tokens_seen": tokens_seen, **metrics}
            append_jsonl(metrics_path, eval_event)
            _print_event(eval_event, max_steps=max_steps)
            best_val_loss = (
                metrics["loss"] if best_val_loss is None
                else min(best_val_loss, float(metrics["loss"]))
            )
        screen_event = {
            "event": "screen_complete",
            "time": utc_now(),
            "step": loop_end,
            "schedule_max_steps": max_steps,
            "tokens_seen": tokens_seen,
            "best_val_loss": best_val_loss,
        }
        append_jsonl(metrics_path, screen_event)
        _print_event(screen_event, max_steps=max_steps)
        return None

    # Short smoke runs still get a validation result when evaluation is enabled.
    if validation_stream is not None and last_eval_step != max_steps:
        metrics = evaluate_language_model(
            model,
            validation_stream,
            batch_size=batch_size,
            sequence_length=sequence_length,
            max_batches=int(training["eval_batches"]),
            runtime=runtime,
        )
        eval_event = {
            "event": "evaluation",
            "time": utc_now(),
            "step": max_steps,
            "tokens_seen": tokens_seen,
            **metrics,
        }
        append_jsonl(metrics_path, eval_event)
        _print_event(eval_event, max_steps=max_steps)
        best_val_loss = (
            metrics["loss"]
            if best_val_loss is None
            else min(best_val_loss, float(metrics["loss"]))
        )

    if last_saved_step != max_steps:
        saved_path = _save_checkpoint(
            output_dir=output_dir,
            step=max_steps,
            model=model,
            model_config=model_config,
            optimizer=optimizer,
            resolved_config=run_config,
            train_generator=train_generator,
            tokens_seen=tokens_seen,
            best_val_loss=best_val_loss,
        )
        checkpoint_event = {
            "event": "checkpoint",
            "time": utc_now(),
            "step": max_steps,
            "path": str(saved_path),
        }
        if max_steps in milestones:
            checkpoint_event["kept"] = str(_keep_milestone(output_dir, saved_path, max_steps))
        append_jsonl(metrics_path, checkpoint_event)
        _print_event(checkpoint_event, max_steps=max_steps)

    complete_event = {
        "event": "complete",
        "time": utc_now(),
        "step": max_steps,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
        "checkpoint": str(output_dir / "latest.pt"),
    }
    append_jsonl(metrics_path, complete_event)
    _print_event(complete_event, max_steps=max_steps)
    return complete_event


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument("--resume", help="checkpoint to resume (overrides training.resume)")
    parser.add_argument("--output-dir", help="run output directory override")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), help="device override")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "bf16", "float32", "fp32"),
        help="numeric precision override",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable torch.compile",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="training/model/data-sampling seed override",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help=(
            "micro-batch override (fit memory); keep batch_size * "
            "gradient_accumulation_steps constant to preserve the effective batch"
        ),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="gradient accumulation override (pair with --batch-size)",
    )
    parser.add_argument(
        "--stop-after-steps",
        type=int,
        default=0,
        help=(
            "SCREENING: halt after N steps while keeping the FULL-length LR "
            "schedule (lowering max_steps would compress warmup/decay and give "
            "numbers that do not match full-run step-N evals). Emits "
            "'screen_complete', writes no checkpoint."
        ),
    )
    parser.add_argument(
        "--stop-at-step",
        type=int,
        default=0,
        help=(
            "train on the unchanged full schedule to step N, write latest.pt and a kept "
            "checkpoint_step_N.pt, emit 'stopped' and exit; --resume latest.pt later continues "
            "the same schedule (e.g. the 1x-Chinchilla point of a 2x run)"
        ),
    )
    parser.add_argument(
        "--keep-checkpoint-at",
        type=int,
        nargs="+",
        metavar="N",
        help="also keep checkpoint_step_N.pt copies at these steps, without stopping",
    )
    parser.add_argument(
        "--overwrite-run",
        action="store_true",
        help=(
            "discard run records already present in the output directory and "
            "start fresh (fresh runs refuse to clobber them otherwise)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_training(
        args.config,
        resume_override=args.resume,
        output_dir_override=args.output_dir,
        device_override=args.device,
        dtype_override=args.dtype,
        compile_override=args.compile,
        seed_override=args.seed,
        batch_size_override=args.batch_size,
        gradient_accumulation_steps_override=args.gradient_accumulation_steps,
        overwrite_run=args.overwrite_run,
        stop_after_steps=args.stop_after_steps,
        stop_at_step=args.stop_at_step,
        keep_checkpoint_at=args.keep_checkpoint_at,
    )


if __name__ == "__main__":
    main()
