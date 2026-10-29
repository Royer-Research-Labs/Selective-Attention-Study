"""Fused Triton kernels for Selective Attention: reference, training forward / backward, prefill.

Selective Attention (Leviathan, Kalman, Matias, "Selective Attention Improves Transformer",
ICLR 2025, arXiv 2410.02703) is causal softmax attention with one extra term on the logits:

    L_h[i, j] = q_h[i] . k_h[j] / sqrt(D)                    # every head h, key j <= query i
    S[i, j]   = relu(L_0[i, j])     for E <= j < i, else 0   # head 0 only; not self, not the first
                                                             # E = `exempt` positions (1: <BOS>)
    F[i, j]   = sum over r < i of S[r, j]                    # selection by strictly earlier queries
    A_h       = causal_softmax(L_h - F);   out_h = A_h @ v_h

`selective_attention_reference` is that definition written plainly, following the listing in the
paper, and is what the kernels are tested against (tests/test_selective_kernels.py).

F is the operator's only coupling between queries: a running sum down the query axis, per key,
read from one head and shared by all heads. The kernels never materialise a (T x T) tensor:

    carry   (grid B x key-blocks; head 0 only, query blocks in causal order): keep each key's
            running selection in registers and store it as it stands on entry to every query
            block: CARRY (B, T / CARRY_M, T) fp32, plus the totals over all T queries
    forward (grid B*H/G x query-blocks; G = 4 heads per program by default): flash-style online softmax; for each key tile rebuild F
            as CARRY + the in-block exclusive cumsum of head 0's selection, subtract it from the
            head's own logits, accumulate exp(.) @ v and the row statistics
    backward (grid B*H/G x key-blocks, G = 2 by default; query blocks in REVERSE order): rebuild A from the saved row
            statistics and F from CARRY, and accumulate dv, dk, dq of ordinary softmax attention.
            The gradient reaching F[i, j] is minus the sum over heads of the logit gradient
            dz_h[i, j], so the gradient reaching S[r, j] is minus the sum over heads and over
            LATER queries i > r. Visiting query blocks last to first makes that a running per-key
            suffix sum, and because it is linear in dz_h every head's program adds its own share
            of head 0's extra gradient (through relu'(L_0)) with atomics. One launch; the full
            gradient, nothing detached.

Operands are loaded in their native dtype (bf16 under autocast) and the dot products run with
fp32 accumulation; `dot_precision` is "bf16" (bf16 tensor-core operands, the default for bf16
inputs), "tf32" (the default otherwise), or "ieee" (parity tests). The selection sum, F and the
softmax are fp32.

The scratch is CARRY: (B, T / 32, T) fp32, i.e. T^2 / 8 bytes per sequence: 32 MiB at T = 16,384,
2 GiB at 131,072, 32 GiB at 524,288 (the longest forward run so far). It is shared by all heads, so
it does not grow with the head count, but its quadratic size is what bounds T in practice.

T is a compile-time constant: each new sequence length compiles its own kernels.
"""

from __future__ import annotations

import math

import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments
    HAS_TRITON = False


def selective_logits(q: torch.Tensor, k: torch.Tensor, exempt: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Adjusted logits L - F, (B, H, T, T) fp32 with -inf above the diagonal, and the selection each
    key has received from all T queries, (B, T): the cached-decode accumulator after this prefix."""

    exempt = _check_exempt(exempt)
    seq, head_dim = q.shape[-2], q.shape[-1]
    logits =(q.float() @ k.float().transpose(-2, -1)) / math.sqrt(head_dim)
    causal = torch.ones(seq, seq, dtype=torch.bool, device=q.device).tril()
    logits = logits.masked_fill(~causal, float("-inf"))
    sel = torch.relu(logits[:, 0])                         # head 0, only positive selection
    sel = sel.masked_fill(torch.eye(seq, dtype=torch.bool, device=q.device), 0.0)   # not self
    sel[..., :exempt] = 0.0                                # do not mask <BOS> (the first `exempt` positions)
    total = sel.sum(dim=-2)
    shifted = torch.roll(sel, 1, -2)                       # a query is masked only by earlier ones
    shifted[..., 0, :] = 0.0
    return logits - torch.cumsum(shifted, dim=-2)[:, None], total


def selective_attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, exempt: int = 1) -> torch.Tensor:
    """Selective Attention written plainly (fp32). q / k / v: (B, H, T, D). Returns (B, H, T, D).
    exempt: positions [0, exempt) can never be selected; 1 is the paper (<BOS> only)."""

    return selective_logits(q, k, exempt)[0].softmax(dim=-1) @ v.float()


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
                   SCALE: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr, EXEMPT: tl.constexpr = 1):
        """Head 0's selection for one (queries x keys) tile: the gate relu'(L_0) restricted to
        1 <= key < query, and S = relu(L_0) under that gate."""
        l0 = _dot(q0_tile, tl.trans(k0_tile), LOWP, DOT_PREC) * SCALE
        allowed = ((n_offs[None, :] < m_offs[:, None]) & (n_offs >= EXEMPT)[None, :]
                   & (n_offs < T)[None, :] & (m_offs < T)[:, None])
        gate = allowed & (l0 > 0.0)
        return gate, tl.where(gate, l0, 0.0)

    @triton.jit
    def _selective_carry(
        Q, K, CARRY, TOTAL,
        T: tl.constexpr, NB: tl.constexpr, HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
        SCALE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        DOT_PREC: tl.constexpr, LOWP: tl.constexpr,
        EXEMPT: tl.constexpr = 1,
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
            _, s0 = _selection(q0_tile, k0_blk, j_offs, m_offs, T, SCALE, DOT_PREC, LOWP, EXEMPT)
            tl.store(carry_base + rb.to(tl.int64) * T + j_offs, carry, mask=j_mask)
            carry += tl.sum(s0, axis=0)
        tl.store(TOTAL + pid_b.to(tl.int64) * T + j_offs, carry, mask=j_mask)

    @triton.jit
    def _selective_forward(
        Q, K, V, OUT, RMAX, RDEN, CARRY,
        T: tl.constexpr, NB: tl.constexpr, HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
        SCALE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CARRY_BLOCK: tl.constexpr,
        DOT_PREC: tl.constexpr, LOWP: tl.constexpr,
        EXEMPT: tl.constexpr = 1,
        DAMP: tl.constexpr = 0, T_TRAIN: tl.constexpr = 1,
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
            _, s0 = _selection(q0_tile, k0_tile, n_offs, m_offs, T, SCALE, DOT_PREC, LOWP, EXEMPT)
            carry = tl.load(c_base + n_offs, mask=n_mask, other=0.0)
            hist = carry[None, :] + tl.cumsum(s0, axis=0) - s0       # F: strictly earlier queries
            if DAMP == 1:    # evaluation only: F * min(1, T_train / (i + 1)) past the trained length
                hist = hist * tl.minimum(1.0, T_TRAIN / (m_offs + 1.0))[:, None]
            elif DAMP == 2:  # F * min(1, sqrt(T_train / (i + 1)))
                hist = hist * tl.minimum(1.0, tl.sqrt(T_TRAIN / (m_offs + 1.0)))[:, None]
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
        tl.store(RMAX + pid_n.to(tl.int64) * T + m_offs, rmax, mask=m_mask)
        tl.store(RDEN + pid_n.to(tl.int64) * T + m_offs, rden, mask=m_mask)

    @triton.jit
    def _selective_backward(
        Q, K, V, DO, DK, DV, DQ, DK0, DQ0, DBAR, RMAX, RDEN, CARRY,
        T: tl.constexpr, NB: tl.constexpr, HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
        SCALE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        DOT_PREC: tl.constexpr, LOWP: tl.constexpr,
        EXEMPT: tl.constexpr = 1,
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
            rmax = tl.load(RMAX + pid_n.to(tl.int64) * T + m_offs, mask=m_mask, other=0.0)
            rden = tl.load(RDEN + pid_n.to(tl.int64) * T + m_offs, mask=m_mask, other=1.0)
            dbar = tl.load(DBAR + pid_n.to(tl.int64) * T + m_offs, mask=m_mask, other=0.0)
            carry = tl.load(carry_base + rb.to(tl.int64) * T + j_offs, mask=j_mask, other=0.0)

            gate, s0 = _selection(q0_tile, k0_blk, j_offs, m_offs, T, SCALE, DOT_PREC, LOWP, EXEMPT)
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


    @triton.jit
    def _head_grads(q_tile, do_tile, k_blk, v_blk, rmax, rden, dbar, hist, causal, DQ_ROWS, m_mask,
                    SCALE: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr):
        """One head's tile of the ordinary attention backward: returns this tile's dv and dk
        contributions and dz, and adds its dq contribution atomically."""
        aff = _dot(q_tile, tl.trans(k_blk), LOWP, DOT_PREC) * SCALE
        z = tl.where(causal, (aff - hist) * LOG2E, float("-inf"))
        attn = tl.where(causal, tl.exp2(z - rmax[:, None]) * (1.0 / rden)[:, None], 0.0)
        dv_part = _dot(tl.trans(attn), do_tile, LOWP, DOT_PREC)
        dov = _dot(do_tile, tl.trans(v_blk), LOWP, DOT_PREC)
        dz = tl.where(causal, attn * (dov - dbar[:, None]), 0.0)
        dk_part = _dot(tl.trans(dz), q_tile, LOWP, DOT_PREC)
        tl.atomic_add(DQ_ROWS, _dot(dz, k_blk, LOWP, DOT_PREC) * SCALE, mask=m_mask[:, None], sem="relaxed")
        return dv_part, dk_part, dz

    @triton.jit
    def _selective_backward_pair(
        Q, K, V, DO, DK, DV, DQ, DK0, DQ0, DBAR, RMAX, RDEN, CARRY,
        T: tl.constexpr, NB: tl.constexpr, HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
        SCALE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        DOT_PREC: tl.constexpr, LOWP: tl.constexpr,
        EXEMPT: tl.constexpr = 1,
    ):
        """_selective_backward for two heads of one sequence per program (HEADS even). The selection
        tile, F and the two head-0 gradient products are computed once per pair: the gradient sent
        to head 0's logits is linear in dz, so the pair's dz tiles are summed first."""
        pid_p = tl.program_id(0)
        pid_kb = tl.program_id(1)
        pid_b = pid_p // (HEADS // 2)
        ra = pid_b * HEADS + (pid_p % (HEADS // 2)) * 2            # rows of (B*H): ra, ra + 1
        base_a = ra.to(tl.int64) * T * HEAD_DIM
        base_b = base_a + T * HEAD_DIM
        base0 = (pid_b * HEADS).to(tl.int64) * T * HEAD_DIM
        baseb = pid_b.to(tl.int64) * T * HEAD_DIM
        j_offs = pid_kb * BLOCK_N + tl.arange(0, BLOCK_N)
        j_mask = j_offs < T
        d_offs = tl.arange(0, HEAD_DIM)
        key_ptr = j_offs[:, None] * HEAD_DIM + d_offs[None, :]
        k_a = tl.load(K + base_a + key_ptr, mask=j_mask[:, None], other=0.0)
        v_a = tl.load(V + base_a + key_ptr, mask=j_mask[:, None], other=0.0)
        k_b = tl.load(K + base_b + key_ptr, mask=j_mask[:, None], other=0.0)
        v_b = tl.load(V + base_b + key_ptr, mask=j_mask[:, None], other=0.0)
        k0_blk = tl.load(K + base0 + key_ptr, mask=j_mask[:, None], other=0.0)
        dk_a = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
        dv_a = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
        dk_b = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
        dv_b = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
        dk0_acc = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
        suffix = tl.zeros((BLOCK_N,), tl.float32)
        carry_base = CARRY + pid_b.to(tl.int64) * NB * T
        first_rb = (pid_kb * BLOCK_N) // BLOCK_M
        for step in range(0, NB - first_rb):
            rb = NB - 1 - step
            m_offs = rb * BLOCK_M + tl.arange(0, BLOCK_M)
            m_mask = m_offs < T
            row_ptr = m_offs[:, None] * HEAD_DIM + d_offs[None, :]
            q0_tile = tl.load(Q + base0 + row_ptr, mask=m_mask[:, None], other=0.0)
            carry = tl.load(carry_base + rb.to(tl.int64) * T + j_offs, mask=j_mask, other=0.0)
            gate, s0 = _selection(q0_tile, k0_blk, j_offs, m_offs, T, SCALE, DOT_PREC, LOWP, EXEMPT)
            hist = carry[None, :] + tl.cumsum(s0, axis=0) - s0
            causal = (j_offs[None, :] <= m_offs[:, None]) & j_mask[None, :] & m_mask[:, None]

            sa = ra.to(tl.int64) * T + m_offs
            dv_p, dk_p, dz_a = _head_grads(
                tl.load(Q + base_a + row_ptr, mask=m_mask[:, None], other=0.0),
                tl.load(DO + base_a + row_ptr, mask=m_mask[:, None], other=0.0), k_a, v_a,
                tl.load(RMAX + sa, mask=m_mask, other=0.0), tl.load(RDEN + sa, mask=m_mask, other=1.0),
                tl.load(DBAR + sa, mask=m_mask, other=0.0), hist, causal, DQ + base_a + row_ptr, m_mask,
                SCALE, DOT_PREC, LOWP)
            dv_a += dv_p
            dk_a += dk_p
            sb = sa + T
            dv_p, dk_p, dz_b = _head_grads(
                tl.load(Q + base_b + row_ptr, mask=m_mask[:, None], other=0.0),
                tl.load(DO + base_b + row_ptr, mask=m_mask[:, None], other=0.0), k_b, v_b,
                tl.load(RMAX + sb, mask=m_mask, other=0.0), tl.load(RDEN + sb, mask=m_mask, other=1.0),
                tl.load(DBAR + sb, mask=m_mask, other=0.0), hist, causal, DQ + base_b + row_ptr, m_mask,
                SCALE, DOT_PREC, LOWP)
            dv_b += dv_p
            dk_b += dk_p

            dz = dz_a + dz_b
            colsum = tl.sum(dz, axis=0)
            later = suffix[None, :] + colsum[None, :] - tl.cumsum(dz, axis=0)
            suffix += colsum
            g = tl.where(gate, -later, 0.0)
            dk0_acc += _dot(tl.trans(g), q0_tile, LOWP, DOT_PREC)
            tl.atomic_add(DQ0 + baseb + row_ptr, _dot(g, k0_blk, LOWP, DOT_PREC) * SCALE,
                          mask=m_mask[:, None], sem="relaxed")
        tl.store(DK + base_a + key_ptr, dk_a * SCALE, mask=j_mask[:, None])
        tl.store(DV + base_a + key_ptr, dv_a.to(DV.dtype.element_ty), mask=j_mask[:, None])
        tl.store(DK + base_b + key_ptr, dk_b * SCALE, mask=j_mask[:, None])
        tl.store(DV + base_b + key_ptr, dv_b.to(DV.dtype.element_ty), mask=j_mask[:, None])
        tl.atomic_add(DK0 + baseb + key_ptr, dk0_acc * SCALE, mask=j_mask[:, None], sem="relaxed")

    @triton.jit
    def _selective_forward_grouped(
        Q, K, V, OUT, RMAX, RDEN, CARRY,
        T: tl.constexpr, NB: tl.constexpr, HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
        SCALE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CARRY_BLOCK: tl.constexpr,
        GROUP: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr,
        EXEMPT: tl.constexpr = 1,
        DAMP: tl.constexpr = 0, T_TRAIN: tl.constexpr = 1,
    ):
        """_selective_forward for GROUP heads of one sequence at once: head 0's selection tile and F
        are computed once per tile and shared by the group (batched (G, M, N) products)."""
        pid_g = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_b = pid_g // (HEADS // GROUP)
        h0 = (pid_g % (HEADS // GROUP)) * GROUP
        rows = pid_b * HEADS + h0 + tl.arange(0, GROUP)                       # (G,) row of (B*H)
        base = rows.to(tl.int64) * T * HEAD_DIM
        base0 = (pid_b * HEADS).to(tl.int64) * T * HEAD_DIM
        m_start = pid_m * BLOCK_M
        m_offs = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offs < T
        d_offs = tl.arange(0, HEAD_DIM)
        row_ptr = m_offs[:, None] * HEAD_DIM + d_offs[None, :]
        row3 = base[:, None, None] + row_ptr[None, :, :]
        q_tile = tl.load(Q + row3, mask=m_mask[None, :, None], other=0.0)      # (G, M, D)
        q0_tile = tl.load(Q + base0 + row_ptr, mask=m_mask[:, None], other=0.0)
        c_base = CARRY + pid_b.to(tl.int64) * NB * T + (m_start // CARRY_BLOCK).to(tl.int64) * T

        rmax = tl.full((GROUP, BLOCK_M), float("-inf"), tl.float32)
        rden = tl.zeros((GROUP, BLOCK_M), tl.float32)
        acc = tl.zeros((GROUP, BLOCK_M, HEAD_DIM), tl.float32)
        for n_start in range(0, m_start + BLOCK_M, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offs < T
            key_ptr = n_offs[:, None] * HEAD_DIM + d_offs[None, :]
            key3 = base[:, None, None] + key_ptr[None, :, :]
            k_tile = tl.load(K + key3, mask=n_mask[None, :, None], other=0.0)  # (G, N, D)
            v_tile = tl.load(V + key3, mask=n_mask[None, :, None], other=0.0)
            k0_tile = tl.load(K + base0 + key_ptr, mask=n_mask[:, None], other=0.0)
            _, s0 = _selection(q0_tile, k0_tile, n_offs, m_offs, T, SCALE, DOT_PREC, LOWP, EXEMPT)
            carry = tl.load(c_base + n_offs, mask=n_mask, other=0.0)
            hist = carry[None, :] + tl.cumsum(s0, axis=0) - s0                 # (M, N), shared
            if DAMP == 1:    # evaluation only: F * min(1, T_train / (i + 1)) past the trained length
                hist = hist * tl.minimum(1.0, T_TRAIN / (m_offs + 1.0))[:, None]
            elif DAMP == 2:  # F * min(1, sqrt(T_train / (i + 1)))
                hist = hist * tl.minimum(1.0, tl.sqrt(T_TRAIN / (m_offs + 1.0)))[:, None]
            aff = _dot(q_tile, tl.permute(k_tile, (0, 2, 1)), LOWP, DOT_PREC) * SCALE
            causal = (n_offs[None, :] <= m_offs[:, None]) & n_mask[None, :] & m_mask[:, None]
            z = tl.where(causal[None, :, :], (aff - hist[None, :, :]) * LOG2E, float("-inf"))
            new_max = tl.maximum(rmax, tl.max(z, axis=2))
            alpha = tl.exp2(rmax - new_max)
            e = tl.exp2(z - new_max[:, :, None])
            acc = acc * alpha[:, :, None] + _dot(e, v_tile, LOWP, DOT_PREC)
            rden = rden * alpha + tl.sum(e, axis=2)
            rmax = new_max
        tl.store(OUT + row3, acc / rden[:, :, None], mask=m_mask[None, :, None])
        stat = rows.to(tl.int64)[:, None] * T + m_offs[None, :]
        tl.store(RMAX + stat, rmax, mask=m_mask[None, :])
        tl.store(RDEN + stat, rden, mask=m_mask[None, :])


def _resolve_precision(dot_precision: str | None, dtype: torch.dtype) -> tuple[int, str]:
    """None: "bf16" for bf16 inputs, else "tf32". "bf16": bf16 tensor-core operands;
    "tf32" / "ieee": fp32 operands. Returns (LOWP, the Triton input_precision for fp32 operands)."""

    if dot_precision is None:
        dot_precision = "bf16" if dtype == torch.bfloat16 else "tf32"
    if dot_precision == "bf16":
        return 1, "tf32"
    if dot_precision in ("ieee", "tf32"):
        return 0, dot_precision
    raise ValueError(f"dot_precision must be None, 'bf16', 'tf32' or 'ieee', not {dot_precision!r}")


HEAD_DIMS = (16, 32, 64, 128)
DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def check_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    """Raise ValueError / RuntimeError unless the kernels can run on these inputs."""

    if not HAS_TRITON:
        raise RuntimeError("triton is not available")
    if q.dim() != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q, k, v must all be (B, H, T, D); got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")
    if not (q.is_cuda and k.device == q.device and v.device == q.device):
        raise ValueError("q, k, v must be CUDA tensors on one device")
    if any(t.dtype not in DTYPES for t in (q, k, v)):
        raise ValueError(f"dtype must be float16, bfloat16 or float32; got {q.dtype}, {k.dtype}, {v.dtype}")
    if q.shape[-1] not in HEAD_DIMS:
        raise ValueError(f"head_dim must be one of {HEAD_DIMS}; got {q.shape[-1]}")
    if q.shape[-2] < 1:
        raise ValueError("the sequence is empty")


def supported(q: torch.Tensor, k: torch.Tensor | None = None, v: torch.Tensor | None = None) -> bool:
    """True when check_inputs would pass (k and v default to q's shape and device)."""

    try:
        check_inputs(q, q if k is None else k, q if v is None else v)
    except (ValueError, RuntimeError):
        return False
    return True


def _check_exempt(exempt: int) -> int:
    if int(exempt) != exempt or exempt < 1:
        raise ValueError(f"exempt must be an integer >= 1 (position 0, <BOS>, is always exempt); got {exempt}")
    return int(exempt)


def _check_blocks(block_m: int, block_n: int) -> None:
    if block_m % CARRY_M or block_n % CARRY_M:
        raise ValueError(f"block_m and block_n must be multiples of {CARRY_M}")


def _group(heads: int, group: int) -> int:
    """Largest power of two <= group that divides the head count (1 = per-head kernels)."""
    group = max(1, int(group))
    while group > 1 and (heads % group or group & (group - 1)):
        group -= 1
    return group


def _auto_forward(heads: int, head_dim: int, group, block_m, num_stages, lowp: int = 1) -> tuple[int, int, int]:
    """Forward launch defaults, chosen at T = 16K, H = 12, D = 64 on an RTX PRO 6000: four heads per
    program on 32 x 32 tiles, one stage (results/bench/layer_fwd_bwd_v2.json: 6.5 ms per layer
    forward at B = 2, against 7.4 ms for one head per program in layer_fwd_bwd_v1.json). Wider
    heads, and fp32 operands (twice the shared memory), take smaller groups."""
    if group is None:
        group = max(1, min(4, (256 if lowp else 64) // head_dim))
    group = _group(heads, group)
    if block_m is None:
        block_m = 32 if group > 1 else 64
    if num_stages is None:   # fp32 operand tiles at head_dim 128 do not fit two stages
        num_stages = 1 if group > 1 or (not lowp and head_dim >= 128) else 2
    return group, block_m, num_stages


def _auto_backward(heads: int, head_dim: int, group, lowp: int = 1) -> int:
    """Backward heads per program, 1 or 2. Default: two (_selective_backward_pair) when the head
    count is even and head_dim <= 64 (<= 32 for fp32 operands); 21.2 ms per layer backward at
    T = 16K, H = 12, D = 64, B = 2 (layer_fwd_bwd_v2.json) against 26.0 ms per head (v1).
    Measured and dropped: three heads per program (spills registers, slower) and batched
    (G, M, N) products in the backward (2-3x slower)."""
    if group is None:
        group = 2 if head_dim <= (64 if lowp else 32) else 1
    if group not in (1, 2):
        raise ValueError(f"bwd_group must be None, 1 or 2; got {group}")
    return 2 if group == 2 and heads % 2 == 0 else 1


def _fit(block: int, head_dim: int, lowp: int, budget: int, floor: int = CARRY_M) -> int:
    """fp32 operand tiles are twice the size of bf16 ones in shared memory: shrink the tile until
    block * head_dim fits (results do not depend on the tile size)."""
    while not lowp and block > floor and block * head_dim > budget:
        block //= 2
    return max(block, floor) if block % floor == 0 else floor


def selective_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dot_precision: str | None = None,
    block_m: int | None = None,
    block_n: int = 32,
    return_aux: bool = False,
    num_warps: int = 4,
    num_stages: int | None = None,
    group: int | None = None,
    exempt: int = 1,
    *,
    damp: str = "none",
    train_len: int = 0,
):
    """Fused forward, no autograd; returns fp32 outputs (B, H, T, D). For inference and prefill.

    dot_precision: None picks "bf16" for bf16 inputs and "tf32" otherwise; "ieee" is exact fp32.
    exempt: positions [0, exempt) are never selected (1: <BOS> only, the paper's rule).

    With return_aux=True also returns the row statistics and CARRY the backward needs, and
    `selected_total` (B, T): the selection every key has received from all T queries, which is the
    cached-decode accumulator after this prefix.

    damp ("none" | "linear" | "sqrt") with train_len is an evaluation-only experiment
    (scripts/investigate/history_damp_eval.py): past the trained length the history term of query i
    is scaled by min(1, train_len / (i + 1)) (or its square root). It made results worse; the
    backward and cached decode do not implement it.
    """

    check_inputs(q, k, v)
    exempt = _check_exempt(exempt)
    batch, heads, seq, head_dim = q.shape
    lowp, prec = _resolve_precision(dot_precision, q.dtype)
    group, block_m, num_stages = _auto_forward(heads, head_dim, group, block_m, num_stages, lowp)
    _check_blocks(block_m, block_n)
    damp_mode = {"none": 0, "linear": 1, "sqrt": 2}.get(damp)
    if damp_mode is None:
        raise ValueError(f"damp must be 'none', 'linear' or 'sqrt'; got {damp!r}")
    if damp_mode and train_len < 1:
        raise ValueError("damp needs train_len (the trained context length)")
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
    block_m, block_n = _fit(block_m, head_dim * group, lowp, 4096), _fit(block_n, head_dim * group, lowp, 4096)
    if not lowp and group > 1:
        num_stages = 1
    common = dict(T=seq, NB=nb, HEADS=heads, HEAD_DIM=head_dim, SCALE=scale, DOT_PREC=prec, LOWP=lowp,
                  EXEMPT=exempt)
    with torch.cuda.device(dev):
        _selective_carry[(batch, triton.cdiv(seq, block_n))](
            qf, kf, carry, total, BLOCK_M=CARRY_M, BLOCK_N=block_n, num_warps=num_warps, num_stages=num_stages,
            **common)
        if group > 1:
            _selective_forward_grouped[(n // group, triton.cdiv(seq, block_m))](
                qf, kf, vf, out, rmax, rden, carry, BLOCK_M=block_m, BLOCK_N=block_n, CARRY_BLOCK=CARRY_M,
                GROUP=group, num_warps=num_warps, num_stages=num_stages, DAMP=damp_mode,
                T_TRAIN=max(int(train_len), 1), **common)
        else:
            _selective_forward[(n, triton.cdiv(seq, block_m))](
                qf, kf, vf, out, rmax, rden, carry, BLOCK_M=block_m, BLOCK_N=block_n, CARRY_BLOCK=CARRY_M,
                num_warps=num_warps, num_stages=num_stages, DAMP=damp_mode, T_TRAIN=max(int(train_len), 1),
                **common)
    result = out.reshape(batch, heads, seq, head_dim)
    if return_aux:
        return result, dict(rmax=rmax, rden=rden, carry=carry, selected_total=total)
    return result


class _SelectiveAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, dot_precision, block_m, block_n, num_warps, bwd_block_n, bwd_num_warps,
                bwd_num_stages, group, bwd_group, num_stages, exempt):
        out, aux = selective_attention_forward(
            q, k, v, dot_precision=dot_precision, block_m=block_m, block_n=block_n, return_aux=True,
            num_warps=num_warps, num_stages=num_stages, group=group, exempt=exempt)
        ctx.exempt = int(exempt)
        ctx.save_for_backward(q, k, v, out, aux["rmax"], aux["rden"], aux["carry"])
        ctx.dot_precision = dot_precision
        ctx.bwd = (bwd_block_n, bwd_num_warps, bwd_num_stages, bwd_group)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        q, k, v, out, rmax, rden, carry = ctx.saved_tensors
        batch, heads, seq, head_dim = q.shape
        n = batch * heads
        dev = q.device
        block_n, num_warps, num_stages, group = ctx.bwd
        lowp, prec = _resolve_precision(ctx.dot_precision, q.dtype)
        group = _auto_backward(heads, head_dim, group, lowp)
        # Native dtypes (bf16 under autocast): tiles are converted in registers.
        qf = q.reshape(n, seq, head_dim).contiguous()
        kf = k.reshape(n, seq, head_dim).contiguous()
        vf = v.reshape(n, seq, head_dim).contiguous()
        dof = dout.reshape(n, seq, head_dim).contiguous()
        dbar = (dof.float() * out.reshape(n, seq, head_dim)).sum(dim=-1).contiguous()

        dq = torch.zeros(n, seq, head_dim, device=dev, dtype=torch.float32)       # atomics
        dk = torch.empty(n, seq, head_dim, device=dev, dtype=torch.float32)
        dv = torch.empty(n, seq, head_dim, device=dev, dtype=v.dtype)
        dq0 = torch.zeros(batch, seq, head_dim, device=dev, dtype=torch.float32)  # atomics
        dk0 = torch.zeros(batch, seq, head_dim, device=dev, dtype=torch.float32)

        if not lowp and prec == "tf32" and head_dim >= 128:
            # TF32 operand layouts at head_dim 128 do not fit shared memory in this backward; IEEE fp32
            # operands do, so the backward silently runs at full fp32 there (v1 does the same).
            prec = "ieee"
        block_n = _fit(block_n, head_dim * group, lowp, 2048, floor=16)   # key tile: any power of two >= 16
        if not lowp and group > 1:
            num_stages = 1
        common = dict(T=seq, NB=carry.shape[1], HEADS=heads, HEAD_DIM=head_dim, SCALE=1.0 / math.sqrt(head_dim),
                      BLOCK_M=CARRY_M, BLOCK_N=block_n, DOT_PREC=prec, LOWP=lowp, EXEMPT=ctx.exempt,
                      num_warps=num_warps, num_stages=num_stages)
        with torch.cuda.device(dev):
            if group == 2:
                _selective_backward_pair[(n // 2, triton.cdiv(seq, block_n))](
                    qf, kf, vf, dof, dk, dv, dq, dk0, dq0, dbar, rmax, rden, carry, **common)
            else:
                _selective_backward[(n, triton.cdiv(seq, block_n))](
                    qf, kf, vf, dof, dk, dv, dq, dk0, dq0, dbar, rmax, rden, carry, **common)

        shape4 = (batch, heads, seq, head_dim)
        dq, dk = dq.view(shape4), dk.view(shape4)
        dq[:, 0] += dq0                      # the selection reads head 0's logits
        dk[:, 0] += dk0
        return (dq.to(q.dtype), dk.to(k.dtype), dv.view(shape4)) + (None,) * 11


def selective_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dot_precision: str | None = None,
    block_m: int | None = None,
    block_n: int = 32,
    num_warps: int = 4,
    bwd_block_n: int = 32,
    bwd_num_warps: int = 4,
    bwd_num_stages: int = 1,
    group: int | None = None,
    bwd_group: int | None = None,
    num_stages: int | None = None,
    exempt: int = 1,
) -> torch.Tensor:
    """Differentiable fused Selective Attention (forward + backward kernels). q / k / v: (B, H, T, D)
    CUDA tensors (fp16, bf16 or fp32; D in 16, 32, 64, 128). Returns fp32 outputs (B, H, T, D).

    dot_precision: None picks "bf16" (bf16 tensor-core operands, fp32 accumulation) for bf16 inputs
    and "tf32" otherwise; "ieee" is exact fp32 (the parity tests).
    exempt: positions [0, exempt) are never selected; 1 (<BOS> only) is the paper's rule.

    Defaults (None) were tuned at head_dim 64 on an RTX PRO 6000: the forward runs four heads per
    program on 32 x 32 tiles and the backward two heads per program on 32 x 32 tiles, sharing head
    0's selection work within the group (see _auto_forward / _auto_backward). group=1, bwd_group=1
    run one head per program: the same computation as kernels_v1.py (README, "Kernel versions").

    The sequence length is a compile-time constant of the kernels: every new T compiles (and
    caches) a new set, which takes a few seconds. Pad or bucket lengths when they vary a lot.
    """

    return _SelectiveAttentionFn.apply(q, k, v, dot_precision, block_m, block_n, num_warps,
                                       bwd_block_n, bwd_num_warps, bwd_num_stages, group, bwd_group,
                                       num_stages, exempt)
