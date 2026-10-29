"""Needle-in-a-haystack retrieval eval for trained DilutionLM checkpoints.

Bridges a `dilution.model.DilutionLM` checkpoint to the external NIAH eval
(https://github.com/Royer-Research-Labs/Niah). The eval is likelihood-based
multiple-choice retrieval — no text generation — and shares this project's
GPT-2 BPE tokenizer.

The NIAH package is not on PyPI. Clone it next to this repository (``../Niah``) or point
``DILUTION_NIAH_ROOT`` at a checkout; an already-importable ``niah`` package also works.
The adapter contract NIAH needs —
``forward(idx, targets=None) -> (logits, loss)`` plus ``.config.context_length``
— is DilutionLM's native interface, so no wrapper module is required.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig, build_model
from .runtime import load_checkpoint, resolve_runtime

# Default location: a sibling checkout next to this repository (<parent>/Niah).
_NIAH_CANDIDATES = (str(Path(__file__).resolve().parents[3] / "Niah"),)


def _resolve_niah_root() -> str:
    env = os.environ.get("DILUTION_NIAH_ROOT")
    for cand in (env, *_NIAH_CANDIDATES):
        if cand and Path(cand).is_dir():
            return cand
    return env or _NIAH_CANDIDATES[0]


def _ensure_niah_importable() -> None:
    root = _resolve_niah_root()
    if root and root not in sys.path and Path(root).is_dir():
        sys.path.insert(0, root)
    try:
        import niah.eval  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guidance
        raise ImportError(
            f"cannot import the NIAH package from {root!r}; set DILUTION_NIAH_ROOT "
            "to a checkout of https://github.com/Royer-Research-Labs/Niah"
        ) from exc


NIAH_REPOSITORY = "https://github.com/Royer-Research-Labs/Niah"


def harness_provenance() -> dict[str, Any]:
    """Which NIAH harness produced a result: repository, commit, dirty flag and spec version.

    The commit comes from the checkout on the import path (git); a pip-installed copy without
    git metadata reports only the spec version. Results are only comparable within one spec
    version: niah-v4 builds prompts to exactly the requested length, v3 ran 12 tokens over.
    """
    info: dict[str, Any] = {"repository": NIAH_REPOSITORY}
    try:
        import niah.spec as spec
        info["spec_version"] = spec.NIAH_SPEC_VERSION
        root = Path(spec.__file__).resolve().parents[1]
    except ImportError:
        return info
    try:
        rev = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if rev.returncode == 0:
            info["commit"] = rev.stdout.strip()
            # content changes only: a checkout shared between Windows and WSL differs in
            # line endings, which `git status` reports as modifications
            dirty = subprocess.run(["git", "-C", str(root), "diff", "--quiet", "--ignore-cr-at-eol", "HEAD", "--"],
                                   capture_output=True, text=True, timeout=10)
            if dirty.returncode in (0, 1):
                info["dirty"] = dirty.returncode == 1
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def checkpoint_identity(checkpoint_path: str | Path, step: int | None = None) -> dict[str, Any]:
    """Path, size, SHA-256 and training step of the evaluated checkpoint."""
    path = Path(checkpoint_path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 2**20), b""):
            digest.update(block)
    return {"path": str(checkpoint_path), "bytes": path.stat().st_size, "sha256": digest.hexdigest(),
            "step": step}


def load_model_for_niah(checkpoint_path: str | Path, device: torch.device | str):
    """Reconstruct a checkpoint for the NIAH adapter (eval mode).

    DilutionLM already follows the nanoGPT calling convention NIAH expects:
    ``model(idx)`` returns last-position logits, ``model(idx, idx)`` returns
    full-sequence logits, and ``model.config.context_length`` exists.
    """

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model_config = ModelConfig.from_dict(dict(checkpoint["model_config"]))
    model = build_model(model_config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.checkpoint_step = checkpoint.get("step")
    model.to(device).eval()
    if torch.device(device).type == "cuda":
        # Route dilution layers through the fused kernels: the eager tensor path
        # materialises a (T x T) attention per head and needs 12 GiB per layer at
        # 16K, 190 GiB at 64K (S11: this OOMed the 24L common eval).
        model.enable_dilution_kernel()
    return model


def _autocast_ctx(device: torch.device, dtype: str):
    if dtype == "float32" or device.type != "cuda":
        return contextlib.nullcontext()
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=torch_dtype)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    context_lengths: list[int],
    num_needles: int = 8,
    samples: int = 10,
    seed: int = 42,
    tokenizer: str = "gpt2",
    device: str = "auto",
    dtype: str = "bfloat16",
    output_json: str | Path | None = None,
    chart_dir: str | Path | None = None,
    context_override: int | None = None,
    ntk: bool = False,
) -> list[dict[str, Any]]:
    """Run a depth x context NIAH sweep over one checkpoint.

    context_override raises the model's context cap for length-extrapolation
    evals; only valid for models without a learned position table
    (positional none / rope / rope_alt). ntk applies dynamic NTK-aware RoPE
    scaling: for a haystack of L tokens past the trained context T, every
    RoPE base becomes theta * (L / T) ** (D / (D - 2))."""

    _ensure_niah_importable()
    from niah.eval import NiahConfig, run_niah
    from niah.tokenizer import load_tokenizer

    runtime = resolve_runtime(device=device, dtype="auto")
    torch_device = runtime.device
    model = load_model_for_niah(checkpoint_path, torch_device)
    trained_ctx = model.config.context_length
    if context_override is not None:
        if getattr(model.config, "positional", "learned") == "learned":
            raise ValueError("context override needs a model without a learned position table")
        model.config.context_length = int(context_override)
    block_size = model.config.context_length
    ropes = [m for m in model.modules() if type(m).__name__ == "RopeModule"]
    base_thetas = [r.theta for r in ropes]
    if ntk and not ropes:
        raise ValueError("--ntk needs a model with RoPE layers")
    tok = load_tokenizer(tokenizer)
    ctx = _autocast_ctx(torch_device, dtype)

    provenance = {
        "harness": harness_provenance(),
        "checkpoint": checkpoint_identity(checkpoint_path, getattr(model, "checkpoint_step", None)),
        "eval": {"num_needles": num_needles, "samples": samples, "seed": seed, "tokenizer": tokenizer,
                 "dtype": dtype, "context_override": context_override, "ntk": ntk,
                 "trained_context": trained_ctx},
    }
    results: list[dict[str, Any]] = []
    with torch.no_grad():
        for ctx_len in context_lengths:
            # NIAH (niah-v4-spec) builds every scored input to exactly ctx_len tokens,
            # question included, so ctx_len == block_size fits the whole window.
            if ctx_len > block_size:
                print(f"[skip] context {ctx_len} exceeds context_length {block_size}")
                continue
            if ntk:
                scale = max(1.0, ctx_len / trained_ctx)
                for r, th in zip(ropes, base_thetas):
                    r.theta = th * scale ** (r.dim / (r.dim - 2))
                print(f"[ctx={ctx_len}] ntk scale {scale:.2f}: theta {base_thetas[0]:.0f} -> {ropes[0].theta:.0f}")
            cfg = NiahConfig(
                context_length=ctx_len,
                num_needles=num_needles,
                placements_per_needle=samples,
                seed=seed,
                use_shared_choices=True,
                canonical_tokenizer=tokenizer,
                prepend_bos=_trained_with_bos(checkpoint_path),
            )
            metrics = run_niah(
                model, config=cfg, tokenizer=tok, device=str(torch_device), ctx=ctx
            )
            if "error" in metrics:
                print(f"[ctx={ctx_len}] ERROR: {metrics['error']}")
                continue
            metrics.setdefault("context_length", ctx_len)
            metrics["provenance"] = provenance
            if ntk:
                metrics["ntk_theta"] = ropes[0].theta
            print(
                f"[ctx={ctx_len}] accuracy={metrics['accuracy']:.3f} "
                f"gap={metrics.get('avg_retrieval_gap_ll', 0.0):.2f} "
                f"early/mid/late={metrics.get('early_accuracy', 0.0):.2f}/"
                f"{metrics.get('mid_accuracy', 0.0):.2f}/"
                f"{metrics.get('late_accuracy', 0.0):.2f} n={metrics.get('total', 0)}"
            )
            results.append(metrics)

    if output_json is not None and results:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {out}")

    if chart_dir is not None and results:
        from niah.charting import save_charts

        paths = save_charts(results, str(chart_dir))
        print("wrote charts: " + ", ".join(str(p) for p in paths))

    return results


def _trained_with_bos(checkpoint_path) -> bool:
    """Models trained with data.bos_token see that token at position 0; NIAH prompts then get it too."""
    resolved = load_checkpoint(checkpoint_path, map_location="cpu").get("resolved_config") or {}
    return (resolved.get("data") or {}).get("bos_token") is not None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="latest.pt from dilution.train")
    parser.add_argument(
        "--context-length", type=int, nargs="+", default=[256, 512, 896],
        help="scored input lengths in tokens, question included (<= context_length)",
    )
    parser.add_argument("--num-needles", type=int, default=8)
    parser.add_argument("--samples", type=int, default=10, help="depth placements per needle")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--output", help="JSON output path")
    parser.add_argument("--context-override", type=int,
                        help="evaluate past the trained context (rope / none models only)")
    parser.add_argument("--ntk", action="store_true",
                        help="dynamic NTK-aware RoPE base scaling for haystacks past the trained context")
    parser.add_argument("--chart-dir", help="directory to render heatmap/accuracy charts into")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    evaluate_checkpoint(
        args.checkpoint,
        context_lengths=args.context_length,
        num_needles=args.num_needles,
        samples=args.samples,
        seed=args.seed,
        tokenizer=args.tokenizer,
        device=args.device,
        dtype=args.dtype,
        output_json=args.output,
        chart_dir=args.chart_dir,
        context_override=args.context_override,
        ntk=args.ntk,
    )


if __name__ == "__main__":
    main()
