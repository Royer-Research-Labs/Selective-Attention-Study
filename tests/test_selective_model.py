"""Selective Attention inside the training harness: construction, decode parity, kernel parity."""

from __future__ import annotations

import pytest
import torch

from selective_attention.config import ATTENTION_TYPES, ModelConfig, build_model
from selective_attention import kernels

needs_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and kernels.HAS_TRITON),
    reason="needs CUDA + triton",
)


def _config(**overrides):
    values = dict(vocab_size=97, n_layers=3, d_model=64, n_heads=2, context_length=96,
                  attention="selective", positional="none")
    values.update(overrides)
    return ModelConfig(**values)


def test_selective_is_a_registered_attention_type():
    assert "selective" in ATTENTION_TYPES
    model = build_model(_config())
    assert [type(b.attn).__name__ for b in model.transformer.h] == ["SelectiveAttention"] * 3


def test_parameter_count_equals_softmax():
    assert (build_model(_config()).num_parameters()
            == build_model(_config(attention="softmax")).num_parameters())


def test_mixed_stack_builds():
    model = build_model(_config(layer_attention=["selective", "dilution", "softmax"]))
    idx = torch.randint(0, 97, (2, 40))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 40, 97) and torch.isfinite(loss)


@pytest.mark.parametrize("positional", ["none", "rope", "learned"])
def test_cached_decode_matches_full_forward(positional):
    torch.manual_seed(0)
    model = build_model(_config(positional=positional)).eval()
    idx = torch.randint(0, 97, (2, 48))
    with torch.no_grad():
        full, _ = model(idx, idx)
        logits, caches = model.prefill(idx[:, :20])
        steps = [logits[:, -1]]
        for t in range(20, 47):
            steps.append(model.decode_step(idx[:, t:t + 1], t, caches)[:, -1])
    torch.testing.assert_close(torch.stack(steps, 1), full[:, 19:47], atol=2e-4, rtol=2e-4)


@needs_gpu
def test_kernel_path_matches_reference_path_in_the_model():
    torch.manual_seed(1)
    model = build_model(_config(d_model=128, n_heads=4, context_length=300)).cuda()
    idx = torch.randint(0, 97, (2, 300), device="cuda")
    results = []
    for use_kernel in (True, False):
        for block in model.transformer.h:
            block.attn.use_kernel = use_kernel
            block.attn.dot_precision = "ieee"   # the default for fp32 inputs is tf32 (~1e-3 on a dot)
        model.zero_grad()
        _, loss = model(idx, idx)
        loss.backward()
        results.append((loss.detach(), torch.cat([p.grad.flatten() for p in model.parameters()])))
    torch.testing.assert_close(results[0][0], results[1][0], atol=1e-5, rtol=1e-5)
    assert (results[0][1] - results[1][1]).norm() / results[1][1].norm() < 2e-3


@needs_gpu
@pytest.mark.parametrize("attention", ["selective", "selective_x16"])
def test_kernel_prefill_then_decode_matches_full_forward(attention):
    torch.manual_seed(2)
    model = build_model(_config(d_model=128, n_heads=4, context_length=200, attention=attention)).cuda().eval()
    for block in model.transformer.h:
        block.attn.dot_precision = "ieee"
    idx = torch.randint(0, 97, (1, 160), device="cuda")
    with torch.no_grad():
        full, _ = model(idx, idx)
        logits, caches = model.prefill(idx[:, :100])
        steps = [logits[:, -1]]
        for t in range(100, 159):
            steps.append(model.decode_step(idx[:, t:t + 1], t, caches)[:, -1])
    torch.testing.assert_close(torch.stack(steps, 1), full[:, 99:159], atol=5e-4, rtol=5e-4)


@needs_gpu
def test_bf16_autocast_training_step_runs_and_is_finite():
    torch.manual_seed(3)
    model = build_model(_config(d_model=128, n_heads=2, context_length=513)).cuda()
    idx = torch.randint(0, 97, (2, 513), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(idx, idx)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_selective_x16_type_exempts_sixteen_positions_and_decodes():
    torch.manual_seed(5)
    model = build_model(_config(attention="selective_x16")).eval()
    assert all(block.attn.exempt == 16 for block in model.transformer.h)
    idx = torch.randint(0, 97, (2, 48))
    with torch.no_grad():
        full, _ = model(idx, idx)
        logits, caches = model.prefill(idx[:, :20])
        steps = [logits[:, -1]]
        for t in range(20, 47):
            steps.append(model.decode_step(idx[:, t:t + 1], t, caches)[:, -1])
    torch.testing.assert_close(torch.stack(steps, 1), full[:, 19:47], atol=2e-4, rtol=2e-4)


def test_bos_token_option_starts_every_window(tmp_path):
    import numpy as np
    from selective_attention.runtime import create_token_stream, load_experiment_config
    tokens = np.arange(1000, dtype=np.uint16) % 97
    path = tmp_path / "toy.bin"
    tokens.tofile(path)
    data = dict(train_path=str(path), val_path=str(path), synthetic=False, vocab_size=97, seed=1)
    plain = create_token_stream(data, "val")
    bos = create_token_stream({**data, "bos_token": 96}, "val")
    x0, y0 = next(iter(plain.sequential_batches(batch_size=2, sequence_length=16, max_batches=1, device=None)))
    x1, y1 = next(iter(bos.sequential_batches(batch_size=2, sequence_length=16, max_batches=1, device=None)))
    assert torch.all(x1[:, 0] == 96)
    torch.testing.assert_close(x1[:, 1:], y1[:, :-1])          # input = [bos] + window[:-1]
    assert x1.shape == x0.shape and y1.shape == y0.shape
    assert not hasattr(plain, "bos_token")                     # off unless the config sets it



def test_bos_token_option_also_applies_to_synthetic_streams():
    from selective_attention.runtime import create_token_stream
    data = dict(synthetic=True, vocab_size=97, train_tokens=4096, val_tokens=1024, seed=1)
    plain = create_token_stream(data, "val")
    bos = create_token_stream({**data, "bos_token": 96}, "val")
    assert not hasattr(plain, "bos_token") and bos.bos_token == 96
    x, y = next(iter(bos.sequential_batches(batch_size=2, sequence_length=16, max_batches=1, device=None)))
    assert torch.all(x[:, 0] == 96)
    torch.testing.assert_close(x[:, 1:], y[:, :-1])


@pytest.mark.parametrize("attention", ["selective", "selective_perhead"])
def test_attention_structure_handles_selective_layers(attention):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from attention_structure import attention_rows
    torch.manual_seed(8)
    model = build_model(_config(attention=attention, n_heads=2, d_model=64)).eval()
    block = model.transformer.h[0]
    x = torch.randn(1, 24, 64)
    with torch.no_grad():
        rows = attention_rows(model, block, x, 24)
        attn = block.attn
        q, k, v = attn._split_heads(x)
        y, _ = attn._attend(q, k, v)
    assert rows.shape == (2, 24, 24)
    torch.testing.assert_close(rows.sum(-1), torch.ones(2, 24))
    torch.testing.assert_close((rows @ v[0].float()), y[0].float(), atol=1e-5, rtol=1e-5)
