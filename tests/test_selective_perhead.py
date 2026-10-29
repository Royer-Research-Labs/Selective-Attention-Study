"""The per-head-history variant (`attention: selective_perhead`): per-head reference, decode, kernel path."""

from __future__ import annotations

import pytest
import torch

from selective_attention import kernels
from selective_attention.config import ModelConfig, build_model
from selective_attention.kernels import selective_attention_reference

needs_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and kernels.HAS_TRITON),
    reason="needs CUDA + triton",
)


def _config(**overrides):
    values = dict(vocab_size=97, n_layers=3, d_model=64, n_heads=2, context_length=96,
                  attention="selective", positional="none")
    values.update(overrides)
    return ModelConfig(**values)


def _per_head_reference(attn, x):
    """Every head run alone through the plain reference: its own selection history."""
    q, k, v = attn._split_heads(x)
    outs = [selective_attention_reference(q[:, h:h + 1], k[:, h:h + 1], v[:, h:h + 1]) for h in range(q.shape[1])]
    return torch.cat(outs, dim=1)


def test_perhead_variant_matches_per_head_reference_and_decodes():
    torch.manual_seed(6)
    model = build_model(_config(attention="selective_perhead", n_heads=4, d_model=64)).eval()
    attn = model.transformer.h[0].attn
    assert attn.groups == 4
    x = torch.randn(2, 30, 64)
    with torch.no_grad():
        y, _ = attn._attend(*attn._split_heads(x))
        torch.testing.assert_close(y.float(), _per_head_reference(attn, x), atol=1e-5, rtol=1e-5)
        idx = torch.randint(0, 97, (2, 48))
        full, _ = model(idx, idx)
        logits, caches = model.prefill(idx[:, :20])
        steps = [logits[:, -1]]
        for t in range(20, 47):
            steps.append(model.decode_step(idx[:, t:t + 1], t, caches)[:, -1])
    torch.testing.assert_close(torch.stack(steps, 1), full[:, 19:47], atol=2e-4, rtol=2e-4)
    shared = build_model(_config(attention="selective", n_heads=4, d_model=64)).eval()
    shared.load_state_dict(model.state_dict())
    with torch.no_grad():
        assert (shared(idx, idx)[0] - full).abs().max() > 1e-4      # a different operator


@needs_gpu
def test_perhead_kernel_path_matches_reference_path():
    torch.manual_seed(7)
    model = build_model(_config(attention="selective_perhead", d_model=128, n_heads=4, context_length=300)).cuda()
    idx = torch.randint(0, 97, (2, 300), device="cuda")
    results = []
    for use_kernel in (True, False):
        for block in model.transformer.h:
            block.attn.use_kernel = use_kernel
            block.attn.dot_precision = "ieee"
        model.zero_grad()
        _, loss = model(idx, idx)
        loss.backward()
        results.append((loss.detach(), torch.cat([p.grad.flatten() for p in model.parameters()])))
    torch.testing.assert_close(results[0][0], results[1][0], atol=1e-5, rtol=1e-5)
    assert (results[0][1] - results[1][1]).norm() / results[1][1].norm() < 2e-3
    model.eval()
    with torch.no_grad():
        full, _ = model(idx[:1, :160], idx[:1, :160])
        logits, caches = model.prefill(idx[:1, :100])
        steps = [logits[:, -1]]
        for t in range(100, 159):
            steps.append(model.decode_step(idx[:1, t:t + 1], t, caches)[:, -1])
    torch.testing.assert_close(torch.stack(steps, 1), full[:, 99:159], atol=5e-4, rtol=5e-4)
