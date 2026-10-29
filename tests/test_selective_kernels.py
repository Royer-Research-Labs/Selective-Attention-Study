"""Parity tests for the fused Triton Selective Attention kernels against the plain reference."""

from __future__ import annotations

import math

import pytest
import torch

from selective_attention import kernels
from selective_attention.kernels import selective_attention_reference, selective_logits

needs_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and kernels.HAS_TRITON),
    reason="needs CUDA + triton",
)


def _qkv(batch, heads, seq, head_dim, seed=0, device="cuda", scale=1.0):
    torch.manual_seed(seed)
    return tuple(scale * torch.randn(batch, heads, seq, head_dim, device=device) for _ in range(3))


def test_reference_matches_a_per_query_loop():
    """The vectorised reference against the definition, one query at a time (CPU)."""
    q, k, v = _qkv(2, 3, 23, 16, device="cpu", scale=1.5)
    got = selective_attention_reference(q, k, v)
    seq, head_dim = 23, 16
    out = torch.zeros_like(got)
    for b in range(2):
        masked = torch.zeros(seq)                       # selection accumulated by earlier queries
        for i in range(seq):
            logits = torch.einsum("hd,hjd->hj", q[b, :, i], k[b, :, : i + 1]) / head_dim ** 0.5
            attn = (logits - masked[: i + 1]).softmax(-1)
            out[b, :, i] = torch.einsum("hj,hjd->hd", attn, v[b, :, : i + 1])
            sel = torch.relu(logits[0]).clone()
            sel[0] = 0.0                                # not <BOS>
            sel[i] = 0.0                                # not self
            masked[: i + 1] += sel
    torch.testing.assert_close(got, out, atol=1e-5, rtol=1e-5)


def test_reference_first_key_and_self_are_never_masked():
    q, k, _ = _qkv(1, 2, 12, 16, device="cpu", scale=3.0)
    logits, total = selective_logits(q, k)
    raw = (q @ k.transpose(-2, -1)) / 4.0
    torch.testing.assert_close(logits[..., 0], raw[..., 0])                       # <BOS> column
    torch.testing.assert_close(logits.diagonal(dim1=-2, dim2=-1), raw.diagonal(dim1=-2, dim2=-1))
    assert total[0, 0] == 0 and total[0, -1] == 0


@needs_gpu
@pytest.mark.parametrize("seq", [1, 2, 31, 32, 33, 64, 65, 130, 257, 513])
@pytest.mark.parametrize("heads", [1, 3])
def test_forward_matches_reference(seq, heads):
    q, k, v = _qkv(2, heads, seq, 32, seed=seq, scale=1.5)
    got, aux = kernels.selective_attention_forward(q, k, v, dot_precision="ieee", return_aux=True)
    torch.testing.assert_close(got, selective_attention_reference(q, k, v), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(aux["selected_total"], selective_logits(q, k)[1], atol=2e-4, rtol=2e-5)


@needs_gpu
@pytest.mark.parametrize("head_dim", [16, 32, 64, 128])
@pytest.mark.parametrize("blocks", [(32, 32), (64, 64), (32, 64), (64, 32), (128, 64)])
def test_forward_block_shapes(head_dim, blocks):
    q, k, v = _qkv(1, 2, 300, head_dim, seed=head_dim, scale=1.5)
    got = kernels.selective_attention_forward(q, k, v, dot_precision="ieee", block_m=blocks[0],
                                              block_n=blocks[1], num_stages=1)
    torch.testing.assert_close(got, selective_attention_reference(q, k, v), atol=2e-5, rtol=2e-5)


@needs_gpu
@pytest.mark.parametrize("seq", [1, 2, 33, 64, 65, 200, 513])
@pytest.mark.parametrize("heads", [1, 4])
def test_backward_matches_autograd_of_reference(seq, heads):
    q, k, v = _qkv(2, heads, seq, 32, seed=seq + 7, scale=1.5)
    weight = torch.randn(2, heads, seq, 32, device="cuda")
    grads = []
    for fn in (lambda a, b, c: kernels.selective_attention(a, b, c, dot_precision="ieee"),
               selective_attention_reference):
        qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
        (fn(qq, kk, vv) * weight).sum().backward()
        grads.append((qq.grad, kk.grad, vv.grad))
    for name, got, want in zip("qkv", *grads):
        torch.testing.assert_close(got, want, atol=5e-5, rtol=5e-4, msg=lambda m, name=name: f"d{name}: {m}")


@needs_gpu
def test_selection_gradient_reaches_head_zero_only_through_the_history_term():
    """Loss on heads >= 1 only still sends gradient to head 0's q and k (through F), and none to
    head 0's v."""
    q, k, v = _qkv(1, 3, 97, 32, seed=3, scale=2.0)
    qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
    kernels.selective_attention(qq, kk, vv, dot_precision="ieee")[:, 1:].sum().backward()
    rq, rk, rv = (t.clone().requires_grad_(True) for t in (q, k, v))
    selective_attention_reference(rq, rk, rv)[:, 1:].sum().backward()
    assert rq.grad[:, 0].abs().max() > 1e-4 and rv.grad[:, 0].abs().max() == 0
    torch.testing.assert_close(qq.grad, rq.grad, atol=5e-5, rtol=5e-4)
    torch.testing.assert_close(kk.grad, rk.grad, atol=5e-5, rtol=5e-4)
    torch.testing.assert_close(vv.grad, rv.grad, atol=5e-5, rtol=5e-4)


@needs_gpu
@pytest.mark.parametrize("precision", ["bf16", "tf32"])
def test_low_precision_paths_stay_close(precision):
    q, k, v = _qkv(2, 4, 400, 64, seed=11)
    if precision == "bf16":
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    want = selective_attention_reference(q, k, v)
    qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
    got = kernels.selective_attention(qq, kk, vv, dot_precision=precision)
    assert (got - want).abs().max() < 3e-2
    got.float().pow(2).sum().backward()
    rq, rk, rv = (t.float().clone().requires_grad_(True) for t in (q, k, v))
    selective_attention_reference(rq, rk, rv).pow(2).sum().backward()
    for got_g, want_g in ((qq.grad, rq.grad), (kk.grad, rk.grad), (vv.grad, rv.grad)):
        assert got_g.dtype == q.dtype
        err = (got_g.float() - want_g).norm() / want_g.norm()
        assert err < 3e-2, float(err)


@needs_gpu
def test_long_sequence_forward_and_backward():
    """Several carry checkpoints, a partial last block, and enough queries for the history term
    to be large."""
    q, k, v = _qkv(1, 2, 2049, 32, seed=5, scale=1.2)
    qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
    got = kernels.selective_attention(qq, kk, vv, dot_precision="ieee")
    rq, rk, rv = (t.clone().requires_grad_(True) for t in (q, k, v))
    want = selective_attention_reference(rq, rk, rv)
    torch.testing.assert_close(got, want, atol=5e-5, rtol=5e-5)
    weight = torch.randn_like(want)
    (got * weight).sum().backward()
    (want * weight).sum().backward()
    for got_g, want_g in ((qq.grad, rq.grad), (kk.grad, rk.grad), (vv.grad, rv.grad)):
        assert (got_g - want_g).norm() / want_g.norm() < 1e-3


# -- head-grouped kernels (the defaults since 2026-10-16) -------------------------------------------

@needs_gpu
@pytest.mark.parametrize("heads,group,bwd_group", [(2, 2, 2), (4, 4, 2), (6, 2, 1), (12, 4, 2), (12, 4, 1), (5, None, None)])
@pytest.mark.parametrize("seq", [1, 33, 257, 513])
def test_grouped_kernels_match_reference(heads, group, bwd_group, seq):
    q, k, v = _qkv(2, heads, seq, 32, seed=seq + heads, scale=1.5)
    weight = torch.randn(2, heads, seq, 32, device="cuda")
    grads = []
    for fn in (lambda a, b, c: kernels.selective_attention(a, b, c, dot_precision="ieee", group=group,
                                                           bwd_group=bwd_group, bwd_block_n=16),
               selective_attention_reference):
        qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
        out = fn(qq, kk, vv)
        (out * weight).sum().backward()
        grads.append((out.detach(), qq.grad, kk.grad, vv.grad))
    for name, got, want in zip(("out", "dq", "dk", "dv"), *grads):
        torch.testing.assert_close(got, want, atol=5e-5, rtol=5e-4, msg=lambda m, name=name: f"{name}: {m}")


@needs_gpu
@pytest.mark.parametrize("head_dim", [16, 32, 64, 128])
@pytest.mark.parametrize("precision", ["bf16", "tf32", "ieee"])
def test_default_launch_works_for_every_head_dim(head_dim, precision):
    q, k, v = _qkv(1, 4, 300, head_dim, seed=head_dim)
    if precision == "bf16":
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    want = selective_attention_reference(q, k, v)
    qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
    got = kernels.selective_attention(qq, kk, vv, dot_precision=precision)
    got.float().sum().backward()
    tol = 3e-2 if precision != "ieee" else 5e-5
    assert (got - want).abs().max() < tol
    assert all(torch.isfinite(g).all() for g in (qq.grad, kk.grad, vv.grad))


@needs_gpu
def test_grouped_defaults_agree_with_the_per_head_kernels():
    """The default (grouped) launch and the one-head-per-program launch (group=1, bwd_group=1)
    compute the same thing in bf16."""
    q, k, v = (t.bfloat16() for t in _qkv(2, 12, 1000, 64, seed=99))
    w = torch.randn(2, 12, 1000, 64, device="cuda")
    res = []
    for kw in ({}, dict(group=1, bwd_group=1)):
        qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
        out = kernels.selective_attention(qq, kk, vv, **kw)
        (out * w).sum().backward()
        res.append((out.detach(), qq.grad.float(), kk.grad.float(), vv.grad.float()))
    for got, want in zip(*res):
        assert (got - want).norm() / want.norm() < 1e-2


# -- exempt prefix (first-tokens blind spot, 2026-10-19) ---------------------------------------------

def test_reference_exempt_positions_are_never_masked():
    q, k, _ = _qkv(1, 2, 20, 16, device="cpu", scale=3.0)
    for exempt in (1, 4):
        logits, total = selective_logits(q, k, exempt)
        raw = (q @ k.transpose(-2, -1)) / 4.0
        causal = torch.ones(20, 20, dtype=torch.bool).tril()
        torch.testing.assert_close(logits[..., :exempt], raw.masked_fill(~causal, float("-inf"))[..., :exempt])
        assert torch.all(total[..., :exempt] == 0) and float(total[..., exempt:].sum()) > 0


@needs_gpu
@pytest.mark.parametrize("exempt", [1, 3, 16, 40])
@pytest.mark.parametrize("groups", [(1, 1), (None, None)])
def test_exempt_kernels_match_reference(exempt, groups):
    q, k, v = _qkv(2, 4, 257, 32, seed=exempt, scale=1.5)
    weight = torch.randn(2, 4, 257, 32, device="cuda")
    grads = []
    for fn in (lambda a, b, c: kernels.selective_attention(a, b, c, dot_precision="ieee", group=groups[0],
                                                           bwd_group=groups[1], bwd_block_n=16, exempt=exempt),
               lambda a, b, c: selective_attention_reference(a, b, c, exempt=exempt)):
        qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
        out = fn(qq, kk, vv)
        (out * weight).sum().backward()
        grads.append((out.detach(), qq.grad, kk.grad, vv.grad))
    for name, got, want in zip(("out", "dq", "dk", "dv"), *grads):
        torch.testing.assert_close(got, want, atol=5e-5, rtol=5e-4, msg=lambda m, name=name: f"{name}: {m}")
    _, aux = kernels.selective_attention_forward(q, k, v, dot_precision="ieee", return_aux=True, exempt=exempt)
    torch.testing.assert_close(aux["selected_total"], selective_logits(q, k, exempt)[1], atol=2e-4, rtol=2e-5)


@needs_gpu
@pytest.mark.parametrize("damp", ["linear", "sqrt"])
@pytest.mark.parametrize("heads,group", [(3, 1), (4, 4), (4, 2)])
def test_history_damping_is_exact_inside_and_scales_outside_the_trained_length(damp, heads, group):
    q, k, v = _qkv(1, heads, 300, 32, seed=17, scale=1.5)
    plain = kernels.selective_attention_forward(q, k, v, dot_precision="ieee", group=group)
    inside = kernels.selective_attention_forward(q, k, v, dot_precision="ieee", group=group, damp=damp,
                                                 train_len=300)
    torch.testing.assert_close(inside, plain)                       # nothing changes within train_len
    got = kernels.selective_attention_forward(q, k, v, dot_precision="ieee", group=group, damp=damp,
                                              train_len=100)
    logits, _ = selective_logits(q, k)
    raw = ((q @ k.transpose(-2, -1)) / math.sqrt(32)).masked_fill(
        ~torch.ones(300, 300, dtype=torch.bool, device="cuda").tril(), float("-inf"))
    hist = torch.nan_to_num(raw - logits, nan=0.0)                  # F (0 above the diagonal)
    scale = (100.0 / torch.arange(1, 301, device="cuda", dtype=torch.float32)).clamp(max=1.0)
    if damp == "sqrt":
        scale = scale.sqrt()
    want = (raw - hist * scale[:, None]).softmax(-1) @ v
    torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)


# -- release checks (2026-10-26) --------------------------------------------------------------------

@needs_gpu
def test_v2_matches_the_v1_kernels_in_bf16():
    """kernels_v1.py trained the first halves of two 16K seeds; v2 (default launch and one head per
    program) must compute the same function at the training precision."""
    from selective_attention import kernels_v1

    q, k, v = (t.bfloat16() for t in _qkv(2, 12, 1000, 64, seed=123))
    w = torch.randn(2, 12, 1000, 64, device="cuda")
    res = []
    for fn in (lambda a, b, c: kernels_v1.selective_attention(a, b, c, dot_precision="bf16"),
               lambda a, b, c: kernels.selective_attention(a, b, c, group=1, bwd_group=1),
               lambda a, b, c: kernels.selective_attention(a, b, c)):
        qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
        out = fn(qq, kk, vv)
        (out * w).sum().backward()
        res.append((out.detach(), qq.grad.float(), kk.grad.float(), vv.grad.float()))
    for other in res[1:]:
        for name, got, want in zip(("out", "dq", "dk", "dv"), other, res[0]):
            assert (got - want).norm() / want.norm() < 1e-2, name


@needs_gpu
def test_default_precision_follows_the_input_dtype():
    q, k, v = _qkv(1, 4, 200, 32, seed=8)
    torch.testing.assert_close(kernels.selective_attention_forward(q, k, v),
                               kernels.selective_attention_forward(q, k, v, dot_precision="tf32"))
    qb, kb, vb = q.bfloat16(), k.bfloat16(), v.bfloat16()
    torch.testing.assert_close(kernels.selective_attention_forward(qb, kb, vb),
                               kernels.selective_attention_forward(qb, kb, vb, dot_precision="bf16"))
    torch.testing.assert_close(kernels.selective_attention(qb, kb, vb),
                               kernels.selective_attention(qb, kb, vb, dot_precision="bf16"))


@needs_gpu
def test_bad_inputs_raise_clear_errors():
    q, k, v = _qkv(1, 2, 64, 32)
    with pytest.raises(ValueError, match="must all be"):
        kernels.selective_attention(q, k[:, :, :32], v)
    with pytest.raises(ValueError, match="CUDA"):
        kernels.selective_attention(q.cpu(), k.cpu(), v.cpu())
    with pytest.raises(ValueError, match="dtype"):
        kernels.selective_attention(q.double(), k.double(), v.double())
    with pytest.raises(ValueError, match="head_dim"):
        kernels.selective_attention(q[..., :24], k[..., :24], v[..., :24])
    with pytest.raises(ValueError, match="dot_precision"):
        kernels.selective_attention(q, k, v, dot_precision="fp8")
    with pytest.raises(ValueError, match="exempt"):
        kernels.selective_attention(q, k, v, exempt=0)
    with pytest.raises(ValueError, match="bwd_group"):
        kernels.selective_attention(q.requires_grad_(True), k, v, bwd_group=4).sum().backward()
    with pytest.raises(ValueError, match="damp"):
        kernels.selective_attention_forward(q, k, v, damp="cubic", train_len=10)
    with pytest.raises(TypeError):
        kernels.selective_attention_forward(q, k, v, None, None, 32, False, 4, None, None, 1, "linear")
    assert kernels.supported(q) and not kernels.supported(q.cpu()) and not kernels.supported(q[..., :24])


def test_reference_rejects_exempt_zero():
    q, k, v = _qkv(1, 1, 8, 16, device="cpu")
    with pytest.raises(ValueError, match="exempt"):
        selective_attention_reference(q, k, v, exempt=0)


def _tail_reference(q, k, v, tail, chunk=1024):
    """Reference outputs for the last `tail` queries of a sequence too long for (T x T) tensors (the
    selection of every earlier query is summed chunk by chunk), and the selection totals."""
    batch, heads, seq, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)
    keys = torch.arange(seq, device=q.device)

    def selection(rows):
        l0 = (q[:, 0, rows].float() @ k[:, 0].float().transpose(-2, -1)) * scale
        return torch.relu(l0) * ((keys[None, :] < rows[:, None]) & (keys >= 1)[None, :])

    start = seq - tail
    earlier = torch.zeros(batch, seq, device=q.device)             # selection by queries < start
    for a in range(0, start, chunk):
        earlier += selection(torch.arange(a, min(a + chunk, start), device=q.device)).sum(dim=1)
    rows = torch.arange(start, seq, device=q.device)
    s = selection(rows)
    hist = earlier[:, None, :] + torch.cumsum(s, dim=1) - s        # F for the tail rows
    logits = (q[:, :, rows].float() @ k.float().transpose(-2, -1)) * scale - hist[:, None]
    logits = logits.masked_fill(keys[None, None, None, :] > rows[None, None, :, None], float("-inf"))
    return rows, logits.softmax(dim=-1) @ v.float(), earlier + s.sum(dim=1)


@needs_gpu
def test_forward_past_the_int32_offset_limit():
    """CARRY is (B, T/32, T): block index x T passes 2^31 from T ~ 262K. A forward at T = 270,000
    (9.1 GB of CARRY) must still match the reference in its last queries."""
    seq = 270_000
    if torch.cuda.get_device_properties(0).total_memory < 24 * 2**30:
        pytest.skip("needs a 24 GB GPU")
    torch.manual_seed(31)
    q, k, v = (torch.randn(1, 2, seq, 16, device="cuda") for _ in range(3))
    got, aux = kernels.selective_attention_forward(q, k, v, dot_precision="ieee", return_aux=True)
    rows, want, total = _tail_reference(q, k, v, tail=1000)
    torch.testing.assert_close(got[:, :, rows], want, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(aux["selected_total"], total, atol=1e-2, rtol=1e-3)
