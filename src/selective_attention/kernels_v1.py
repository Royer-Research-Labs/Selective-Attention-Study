"""Fused Triton kernels for Selective Attention: reference, training forward / backward, prefill.

Selective Attention (Leviathan, Kalman, Matias, "Selective Attention Improves Transformer",
ICLR 2025, arXiv 2410.02703) is causal softmax attention with one extra term on the logits:

    L_h[i, j] = q_h[i] . k_h[j] / sqrt(D)                    # every head h, key j <= query i
    S[i, j]   = relu(L_0[i, j])     for 1 <= j < i, else 0   # head 0 only; not <BOS>, not self
    F[i, j]   = sum over r < i of S[r, j]                    # selection by strictly earlier queries
    A_h       = causal_softmax(L_h - F);   out_h = A_h @ v_h

`selective_attention_reference` is that definition written plainly, following the listing in the
paper, and is what the kernels are tested against (tests/test_selective_kernels.py).

F is the operator's only coupling between queries: a running sum down the query axis, per key,
read from one head and shared by all heads. The kernels never materialise a (T x T) tensor:

    carry   (grid B x key-blocks; head 0 only, query blocks in causal order): keep each key's
            running selection in registers and store it as it stands on entry to every query
            block: CARRY (B, T / CARRY_M, T) fp32, plus the totals over all T queries
    forward (grid B*H x query-blocks): flash-style online softmax; for each key tile rebuild F
            as CARRY + the in-block exclusive cumsum of head 0's selection, subtract it from the
            head's own logits, accumulate exp(.) @ v and the row statistics
    backward (grid B*H x key-blocks; query blocks in REVERSE order): rebuild A from the saved row
            statistics and F from CARRY, and accumulate dv, dk, dq of ordinary softmax attention.
            The gradient reaching F[i, j] is minus the sum over heads of the logit gradient
            dz_h[i, j], so the gradient reaching S[r, j] is minus the sum over heads and over
            LATER queries i > r. Visiting query blocks last to first makes that a running per-key
            suffix sum, and because it is linear in dz_h every head's program adds its own share
            of head 0's extra gradient (through relu'(L_0)) with atomics. One launch; the full
            gradient, nothing detached.

Operands are loaded in their native dtype (bf16 under autocast) and the dot products run with
fp32 accumulation; `dot_precision` is "bf16" (bf16 tensor-core operands, the default for bf16
inputs), "tf32", or "ieee" (parity tests). The selection sum, F and the softmax are fp32.

The scratch is CARRY: (B, T / 32, T) fp32, 64 MiB per sequence at T = 16,384. It is shared by all
heads, so it does not grow with the head count.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments
    HAS_TRITON = False


def selective_logits(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Adjusted logits L - F, (B, H, T, T) fp32 with -inf above the diagonal, and the selection each
    key has received from all T queries, (B, T): the cached-decode accumulator after this prefix."""

    seq, head_dim = q.shape[-2], q.shape[-1]
    logits = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(head_dim)
    causal = torch.ones(seq, seq, dtype=torch.bool, device=q.device).tril()
    logits = logits.masked_fill(~causal, float("-inf"))
    sel = torch.relu(logits[:, 0])                         # head 0, only positive selection
    sel = sel.masked_fill(torch.eye(seq, dtype=torch.bool, device=q.device), 0.0)   # not self
    sel[..., 0] = 0.0                                      # do not mask <BOS>
    total = sel.sum(dim=-2)
    shifted = torch.roll(sel, 1, -2)                       # a query is masked only by earlier ones
    shifted[..., 0, :] = 0.0
    return logits - torch.cumsum(shifted, dim=-2)[:, None], total


def selective_attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Selective Attention written plainly (fp32). q / k / v: (B, H, T, D). Returns (B, H, T, D)."""

    return selective_logits(q, k)[0].softmax(dim=-1) @ v.float()


# Query rows per CARRY checkpoint, and the query-tile height of the backward pass.
CARRY_M = 32

if HAS_TRITON:
    LOG2E = tl.constexpr(1.4426950408889634)

    @triton.jit
    def _dot(a, b, LOWP: tl.constexpr, DOT_PREC: tl.constexpr):
        """Tensor-core dot with fp32 accumulation; LOWP: bf16 operands."""
        if LOWP:
            return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
        else:
            return tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision=DOT_PREC)

    @triton.jit
    def _selection(q0_tile, k0_tile, n_offs, m_offs, T,
                   SCALE: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr):
        """Head 0's selection for one (queries x keys) tile: the gate relu'(L_0) restricted to
        1 <= key < query, and S = relu(L_0) under that gate."""
        l0 = _dot(q0_tile, tl.trans(k0_tile), LOWP, DOT_PREC) * SCALE
        allowed = ((n_offs[None, :] < m_offs[:, None]) & (n_offs >= 1)[None, :]
                   & (n_offs < T)[None, :] & (m_offs < T)[:, None])
        gate = allowed & (l0 > 0.0)
        return gate, tl.where(gate, l0, 0.0)

    @triton.jit
    def _selective_carry(
        Q, K, CARRY, TOTAL,
        T: tl.constexpr, NB: tl.constexpr, HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
        SCALE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        DOT_PREC: tl.constexpr, LOWP: tl.constexpr,
    ):
        """Head 0 only. For each key block, visit the query blocks in causal order with the
        key's running selection in registers; store it on entry to every query block."""
        pid_b = tl.program_id(0)
        pid_kb = tl.program_id(1)
        j_offs = pid_kb * BLOCK_N + tl.arange(0, BLOCK_N)
        j_mask = j_offs < T
        d_offs = tl.arange(0, HEAD_DIM)
        base0 = (pid_b * HEADS).to(tl.int64) * T * HEAD_DIM
        k0_blk = tl.load(K + base0 + j_offs[:, None] * HEAD_DIM + d_offs[None, :],
                         mask=j_mask[:, None], other=0.0)
        carry = tl.zeros((BLOCK_N,), tl.float32)
        carry_base = CARRY + pid_b.to(tl.int64) * NB * T
        for rb in range((pid_kb * BLOCK_N) // BLOCK_M, NB):
            m_offs = rb * BLOCK_M + tl.arange(0, BLOCK_M)
            q0_tile = tl.load(Q + base0 + m_offs[:, None] * HEAD_DIM + d_offs[None, :],
                              mask=(m_offs < T)[:, None], other=0.0)
            _, s0 = _selection(q0_tile, k0_blk, j_offs, m_offs, T, SCALE, DOT_PREC, LOWP)
            tl.store(carry_base + rb.to(tl.int64) * T + j_offs, carry, mask=j_mask)
            carry += tl.sum(s0, axis=0)
        tl.store(TOTAL + pid_b * T + j_offs, carry, mask=j_mask)

    @triton.jit
    def _selective_forward(
        Q, K, V, OUT, RMAX, RDEN, CARRY,
        T: tl.constexpr, NB: tl.constexpr, HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
        SCALE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CARRY_BLOCK: tl.constexpr,
        DOT_PREC: tl.constexpr, LOWP: tl.constexpr,
    ):
        """Online causal softmax of L_h - F for one block of queries of one (batch, head) row.
        BLOCK_M must be a multiple of CARRY_BLOCK (the checkpoint at the tile's first row is used
        and the in-tile cumsum supplies the rest)."""
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_b = pid_n // HEADS
        m_start = pid_m * BLOCK_M
        m_offs = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offs < T
        d_offs = tl.arange(0, HEAD_DIM)
        base = pid_n.to(tl.int64) * T * HEAD_DIM
        base0 = (pid_b * HEADS).to(tl.int64) * T * HEAD_DIM
        row_ptr = m_offs[:, None] * HEAD_DIM + d_offs[None, :]
        q_tile = tl.load(Q + base + row_ptr, mask=m_mask[:, None], other=0.0)
        q0_tile = tl.load(Q + base0 + row_ptr, mask=m_mask[:, None], other=0.0)
        c_base = CARRY + pid_b.to(tl.int64) * NB * T + (m_start // CARRY_BLOCK).to(tl.int64) * T

        rmax = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        rden = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
        for n_start in range(0, m_start + BLOCK_M, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offs < T
            key_ptr = n_offs[:, None] * HEAD_DIM + d_offs[None, :]
            k_tile = tl.load(K + base + key_ptr, mask=n_mask[:, None], other=0.0)
            k0_tile = tl.load(K + base0 + key_ptr, mask=n_mask[:, None], other=0.0)
            v_tile = tl.load(V + base + key_ptr, mask=n_mask[:, None], other=0.0)
            _, s0 = _selection(q0_tile, k0_tile, n_offs, m_offs, T, SCALE, DOT_PREC, LOWP)
            carry = tl.load(c_base + n_offs, mask=n_mask, other=0.0)
            hist = carry[None, :] + tl.cumsum(s0, axis=0) - s0       # F: strictly earlier queries
            aff = _dot(q_tile, tl.trans(k_tile), LOWP, DOT_PREC) * SCALE
            causal = (n_offs[None, :] <= m_offs[:, None]) & n_mask[None, :] & m_mask[:, None]
            z = tl.where(causal, (aff - hist) * LOG2E, float("-inf"))
            new_max = tl.maximum(rmax, tl.max(z, axis=1))
            alpha = tl.exp2(rmax - new_max)
            e = tl.exp2(z - new_max[:, None])
            acc = acc * alpha[:, None] + _dot(e, v_tile, LOWP, DOT_PREC)
            rden = rden * alpha + tl.sum(e, axis=1)
            rmax = new_max
        tl.store(OUT + base + row_ptr, acc / rden[:, None], mask=m_mask[:, None])
        tl.store(RMAX + pid_n * T + m_offs, rmax, mask=m_mask)
        tl.store(RDEN + pid_n * T + m_offs, rden, mask=m_mask)

    @triton.jit
    def _selective_backward(
        Q, K, V, DO, DK, DV, DQ, DK0, DQ0, DBAR, RMAX, RDEN, CARRY,
        T: tl.constexpr, NB: tl.constexpr, HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
        SCALE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        DOT_PREC: tl.constexpr, LOWP: tl.constexpr,
    ):
        """Key-parallel backward for one key block of one (batch, head) row; query blocks last to
        first. With A rebuilt from the row statistics and dz = A (do . v - do . o):
            dv[j]  += sum_i A_ij do_i                        (registers)
            dk[j]  += SCALE sum_i dz_ij q_i                  (registers)
            dq[i]  += SCALE sum_j dz_ij k_j                  (fp32 atomics, relaxed)
        and, with U_rj = sum over later queries i > r of dz_ij (a running suffix) and
        g = -gate * U the gradient this head sends to head 0's logit L_0[r, j]:
            dk0[j] += SCALE sum_r g_rj q0_r                  (registers, one atomic add at the end)
            dq0[r] += SCALE sum_j g_rj k0_j                  (fp32 atomics, relaxed)
        DQ, DK0 and DQ0 must be zeroed. Nothing in this launch reads them, so relaxed ordering
        is exact. BLOCK_M must equal the CARRY checkpoint spacing."""
        pid_n = tl.program_id(0)
        pid_kb = tl.program_id(1)
        pid_b = pid_n // HEADS
        j_offs = pid_kb * BLOCK_N + tl.arange(0, BLOCK_N)
        j_mask = j_offs < T
        d_offs = tl.arange(0, HEAD_DIM)
        base = pid_n.to(tl.int64) * T * HEAD_DIM
        base0 = (pid_b * HEADS).to(tl.int64) * T * HEAD_DIM
        baseb = pid_b.to(tl.int64) * T * HEAD_DIM
        key_ptr = j_offs[:, None] * HEAD_DIM + d_offs[None, :]
        k_blk = tl.load(K + base + key_ptr, mask=j_mask[:, None], other=0.0)
        v_blk = tl.load(V + base + key_ptr, mask=j_mask[:, None], other=0.0)
        k0_blk = tl.load(K + base0 + key_ptr, mask=j_mask[:, None], other=0.0)
        dk_acc = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
        dv_acc = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
        dk0_acc = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
        suffix = tl.zeros((BLOCK_N,), tl.float32)
        carry_base = CARRY + pid_b.to(tl.int64) * NB * T
        first_rb = (pid_kb * BLOCK_N) // BLOCK_M
        for step in range(0, NB - first_rb):
            rb = NB - 1 - step
            m_offs = rb * BLOCK_M + tl.arange(0, BLOCK_M)
            m_mask = m_offs < T
            row_ptr = m_offs[:, None] * HEAD_DIM + d_offs[None, :]
            q_tile = tl.load(Q + base + row_ptr, mask=m_mask[:, None], other=0.0)
            q0_tile = tl.load(Q + base0 + row_ptr, mask=m_mask[:, None], other=0.0)
            do_tile = tl.load(DO + base + row_ptr, mask=m_mask[:, None], other=0.0)
            rmax = tl.load(RMAX + pid_n * T + m_offs, mask=m_mask, other=0.0)
            rden = tl.load(RDEN + pid_n * T + m_offs, mask=m_mask, other=1.0)
            dbar = tl.load(DBAR + pid_n * T + m_offs, mask=m_mask, other=0.0)
            carry = tl.load(carry_base + rb.to(tl.int64) * T + j_offs, mask=j_mask, other=0.0)

            gate, s0 = _selection(q0_tile, k0_blk, j_offs, m_offs, T, SCALE, DOT_PREC, LOWP)
            hist = carry[None, :] + tl.cumsum(s0, axis=0) - s0
            aff = _dot(q_tile, tl.trans(k_blk), LOWP, DOT_PREC) * SCALE
            causal = (j_offs[None, :] <= m_offs[:, None]) & j_mask[None, :] & m_mask[:, None]
            z = tl.where(causal, (aff - hist) * LOG2E, float("-inf"))
            attn = tl.where(causal, tl.exp2(z - rmax[:, None]) * (1.0 / rden)[:, None], 0.0)
            dv_acc += _dot(tl.trans(attn), do_tile, LOWP, DOT_PREC)
            dov = _dot(do_tile, tl.trans(v_blk), LOWP, DOT_PREC)
            dz = tl.where(causal, attn * (dov - dbar[:, None]), 0.0)
            dk_acc += _dot(tl.trans(dz), q_tile, LOWP, DOT_PREC)
            tl.atomic_add(DQ + base + row_ptr, _dot(dz, k_blk, LOWP, DOT_PREC) * SCALE,
                          mask=m_mask[:, None], sem="relaxed")
            colsum = tl.sum(dz, axis=0)
            later = suffix[None, :] + colsum[None, :] - tl.cumsum(dz, axis=0)   # queries i > r
            suffix += colsum
            g = tl.where(gate, -later, 0.0)
            dk0_acc += _dot(tl.trans(g), q0_tile, LOWP, DOT_PREC)
            tl.atomic_add(DQ0 + baseb + row_ptr, _dot(g, k0_blk, LOWP, DOT_PREC) * SCALE,
                          mask=m_mask[:, None], sem="relaxed")
        tl.store(DK + base + key_ptr, dk_acc * SCALE, mask=j_mask[:, None])
        tl.store(DV + base + key_ptr, dv_acc.to(DV.dtype.element_ty), mask=j_mask[:, None])
        tl.atomic_add(DK0 + baseb + key_ptr, dk0_acc * SCALE, mask=j_mask[:, None], sem="relaxed")


def _resolve_precision(dot_precision: str) -> tuple[int, str]:
    """"ieee" / "tf32": fp32 operands; "bf16": bf16 tensor-core operands."""

    if dot_precision == "bf16":
        return 1, "tf32"
    if dot_precision in ("ieee", "tf32"):
        return 0, dot_precision
    raise ValueError("dot_precision must be one of ieee, tf32, bf16")


def _check(q: torch.Tensor, block_m: int, block_n: int) -> None:
    if not HAS_TRITON:
        raise RuntimeError("triton is not available")
    head_dim = q.shape[-1]
    if head_dim < 16 or head_dim & (head_dim - 1):
        raise ValueError("head_dim must be a power of two, at least 16")
    if block_m % CARRY_M or block_n % CARRY_M:
        raise ValueError(f"block_m and block_n must be multiples of {CARRY_M}")


def _fit(block: int, head_dim: int, lowp: int, budget: int) -> int:
    """fp32 operand tiles are twice the size of bf16 ones in shared memory: shrink the tile until
    block * head_dim fits (results do not depend on the tile size)."""
    while not lowp and block > CARRY_M and block * head_dim > budget:
        block //= 2
    return max(block, CARRY_M) if block % CARRY_M == 0 else CARRY_M


def selective_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dot_precision: str = "ieee",
    block_m: int = 64,
    block_n: int = 32,
    return_aux: bool = False,
    num_warps: int = 4,
    num_stages: int = 2,
):
    """Fused forward; returns fp32 outputs (B, H, T, D).

    With return_aux=True also returns the row statistics and CARRY the backward needs, and
    `selected_total` (B, T): the selection every key has received from all T queries, which is the
    cached-decode accumulator after this prefix.
    """

    _check(q, block_m, block_n)
    batch, heads, seq, head_dim = q.shape
    n = batch * heads
    qf = q.reshape(n, seq, head_dim).contiguous()
    kf = k.reshape(n, seq, head_dim).contiguous()
    vf = v.reshape(n, seq, head_dim).contiguous()
    dev = q.device
    nb = (seq + CARRY_M - 1) // CARRY_M
    # Zeroed: entries for keys later than a query block are never written (and never used, the
    # causal mask covers them), but they are loaded.
    carry = torch.zeros(batch, nb, seq, device=dev, dtype=torch.float32)
    total = torch.empty(batch, seq, device=dev, dtype=torch.float32)
    out = torch.empty(n, seq, head_dim, device=dev, dtype=torch.float32)
    rmax = torch.empty(n, seq, device=dev, dtype=torch.float32)
    rden = torch.empty(n, seq, device=dev, dtype=torch.float32)

    scale = 1.0 / math.sqrt(head_dim)
    lowp, prec = _resolve_precision(dot_precision)
    block_m, block_n = _fit(block_m, head_dim, lowp, 4096), _fit(block_n, head_dim, lowp, 4096)
    common = dict(T=seq, NB=nb, HEADS=heads, HEAD_DIM=head_dim, SCALE=scale, DOT_PREC=prec, LOWP=lowp)
    _selective_carry[(batch, triton.cdiv(seq, block_n))](
        qf, kf, carry, total, BLOCK_M=CARRY_M, BLOCK_N=block_n, num_warps=num_warps, num_stages=num_stages,
        **common)
    _selective_forward[(n, triton.cdiv(seq, block_m))](
        qf, kf, vf, out, rmax, rden, carry, BLOCK_M=block_m, BLOCK_N=block_n, CARRY_BLOCK=CARRY_M,
        num_warps=num_warps, num_stages=num_stages, **common)
    result = out.reshape(batch, heads, seq, head_dim)
    if return_aux:
        return result, dict(rmax=rmax, rden=rden, carry=carry, selected_total=total)
    return result


class _SelectiveAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, dot_precision, block_m, block_n, num_warps, bwd_block_n, bwd_num_warps,
                bwd_num_stages):
        out, aux = selective_attention_forward(
            q, k, v, dot_precision=dot_precision, block_m=block_m, block_n=block_n, return_aux=True,
            num_warps=num_warps)
        ctx.save_for_backward(q, k, v, out, aux["rmax"], aux["rden"], aux["carry"])
        ctx.dot_precision = dot_precision
        ctx.bwd = (bwd_block_n, bwd_num_warps, bwd_num_stages)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, rmax, rden, carry = ctx.saved_tensors
        batch, heads, seq, head_dim = q.shape
        n = batch * heads
        dev = q.device
        block_n, num_warps, num_stages = ctx.bwd
        # Native dtypes (bf16 under autocast): tiles are converted in registers.
        qf = q.reshape(n, seq, head_dim).contiguous()
        kf = k.reshape(n, seq, head_dim).contiguous()
        vf = v.reshape(n, seq, head_dim).contiguous()
        dof = dout.reshape(n, seq, head_dim).contiguous()
        dbar = (dof.float() * out.reshape(n, seq, head_dim)).sum(dim=-1).contiguous()

        dq = torch.zeros(n, seq, head_dim, device=dev, dtype=torch.float32)       # atomics
        dk = torch.empty(n, seq, head_dim, device=dev, dtype=torch.float32)
        dv = torch.empty(n, seq, head_dim, device=dev, dtype=v.dtype)
        dq0 = torch.zeros(batch, seq, head_dim, device=dev, dtype=torch.float32)  # atomics, all heads
        dk0 = torch.zeros(batch, seq, head_dim, device=dev, dtype=torch.float32)

        lowp, prec = _resolve_precision(ctx.dot_precision)
        block_n = _fit(block_n, head_dim, lowp, 2048)
        _selective_backward[(n, triton.cdiv(seq, block_n))](
            qf, kf, vf, dof, dk, dv, dq, dk0, dq0, dbar, rmax, rden, carry,
            T=seq, NB=carry.shape[1], HEADS=heads, HEAD_DIM=head_dim, SCALE=1.0 / math.sqrt(head_dim),
            BLOCK_M=CARRY_M, BLOCK_N=block_n, DOT_PREC=prec, LOWP=lowp,
            num_warps=num_warps, num_stages=num_stages)

        shape4 = (batch, heads, seq, head_dim)
        dq, dk = dq.view(shape4), dk.view(shape4)
        dq[:, 0] += dq0                      # the selection reads head 0's logits
        dk[:, 0] += dk0
        return dq.to(q.dtype), dk.to(k.dtype), dv.view(shape4), None, None, None, None, None, None, None


def selective_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dot_precision: str = "bf16",
    block_m: int = 64,
    block_n: int = 32,
    num_warps: int = 4,
    bwd_block_n: int = 32,
    bwd_num_warps: int = 4,
    bwd_num_stages: int = 1,
) -> torch.Tensor:
    """Differentiable fused Selective Attention (forward + backward kernels), fp32 output.

    dot_precision "bf16" (default) uses bf16 tensor-core operands for the dot products with fp32
    accumulation; "tf32" / "ieee" keep fp32 operands (parity tests use "ieee").

    The tile sizes are the measured optimum at head_dim 64 on an RTX PRO 6000 (64 x 32 forward,
    32 x 32 backward): at 16K tokens they are 2.3x / 1.5x faster than 64 x 64 tiles.
    """

    return _SelectiveAttentionFn.apply(q, k, v, dot_precision, block_m, block_n, num_warps,
                                       bwd_block_n, bwd_num_warps, bwd_num_stages)
