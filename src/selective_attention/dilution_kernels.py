"""Fused Triton dilution attention: reference, training, prefill, and decode.

`dilution_attention_reference` is the authoritative PyTorch spec of the
operator (docs/operator.md): one causal softmax, one exclusive column-cumsum,
one renormalise, no parameters. The Triton path must match it;
tests/test_kernels.py pins parity.

Long-context training uses two score sweeps: online softmax row statistics and p @ k,
then a key-parallel pass visiting query blocks in causal order. The second
pass carries cumulative demand in registers, saves block checkpoints for
backward, and atomically accumulates output rows before normalization.

Forward-only / prefill uses the row-parallel scan decomposition:
    K1 (grid N x nB): causal-softmax row stats + per-block key-sums S
        carry = exclusive block-cumsum(S)               # Triton, (N, nB, T)
    K2 (grid N x nB): rebuild p, committed = carry + in-block exclusive cumsum,
        share = p / (committed + p + eps) -> row-normalize -> @ V

Nothing (T x T)-shaped ever reaches DRAM. fp32 accumulators throughout; dot precision
is a knob ("ieee" for parity tests, "tf32" for production fp32 inputs, "bf16"
for bf16 tensor-core operands). The backward kernels below make the operator
fully differentiable; the decode kernel serves one cached token per launch.
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


def dilution_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """The dilution-attention operator, written plainly (fp32).

    q/k/v: (B, H, T, D). Returns (B, H, T, D) fp32.

        p         = causal_softmax(q k^T / sqrt(D))   # bids
        committed = exclusive_cumsum_cols(p)          # taken by earlier queries
        share     = p / (committed + p + eps)         # bid vs cumulative demand
        A         = share / rowsum(share);  out = A @ v

    One causal softmax, one exclusive cumulative sum down the query axis, one
    renormalise. There are no learnable scalars and only one softmax: the order
    asymmetry falls out of the running sum alone.
    """

    seq, head_dim = q.shape[-2], q.shape[-1]
    aff = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(head_dim)
    mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=q.device))
    aff = aff.masked_fill(~mask, torch.finfo(torch.float32).min)

    p = aff.softmax(dim=-1)
    committed = p.cumsum(dim=-2) - p
    share = p / (committed + p + 1e-9)
    attn = share / (share.sum(dim=-1, keepdim=True) + 1e-9)
    return attn @ v.float()


# Training sequences longer than this take the key-parallel forward (KF) and 32-row backward
# tiles; shorter ones and forward-only calls take the row-parallel K1 -> scan -> K2 path. Was
# 1024, which put ctx-1024 training on the slower path (S26). With the current backward the
# key-parallel path is 0.86x / 0.95x / 1.00x / 1.01x the row-parallel time at 1024 / 512 /
# 256 / 128 tokens (scripts/bench_crossover.py), so it now starts above 256.
KEY_FORWARD_MIN_SEQ = 256

if HAS_TRITON:
    LOG2E = tl.constexpr(1.4426950408889634)

    @triton.jit
    def _dot(a, b, LOWP: tl.constexpr, DOT_PREC: tl.constexpr):
        """Tensor-core dot with fp32 accumulation.

        LOWP: operands in bf16 (the standard flash-attention choice for q.k and
        p.v products; the operator's cumsum/share math stays fp32 regardless).
        Otherwise fp32 operands with DOT_PREC ("ieee" for parity tests, "tf32").
        """
        if LOWP:
            return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
        else:
            return tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision=DOT_PREC)

    @triton.jit
    def _p_from_k(q_tile, k_tile, n_offs, m_offs, T, rmax, rinv,
                  SCALE: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr):
        """p for one (rows x keys) tile from an already-loaded key tile and row stats."""
        aff = _dot(q_tile, tl.trans(k_tile), LOWP, DOT_PREC) * (SCALE * LOG2E)
        causal = (n_offs[None, :] <= m_offs[:, None]) & (n_offs < T)[None, :] & (m_offs < T)[:, None]
        aff_m = tl.where(causal, aff, float("-inf"))
        p = tl.exp2(aff_m - rmax[:, None]) * rinv[:, None]
        return causal, tl.where(causal, p, 0.0)

    @triton.jit
    def _p_tile(q_tile, k_base, n_offs, d_offs, stride_qt, stride_qd,
                m_offs, T, rmax, rinv, SCALE: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr):
        """Load one key tile and recompute p from stored row stats."""
        k_tile = tl.load(
            k_base + n_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
            mask=(n_offs < T)[:, None], other=0.0,
        )
        return _p_from_k(q_tile, k_tile, n_offs, m_offs, T, rmax, rinv, SCALE, DOT_PREC, LOWP)

    @triton.jit
    def _share_from_k(q_tile, k_tile, n_offs, m_offs, T, rmax, rinv, c_base, stride_st,
                      SCALE: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr):
        """p, its exclusive column-cumsum and 1/(committed+p+eps) from a loaded key tile."""
        causal, p = _p_from_k(q_tile, k_tile, n_offs, m_offs, T, rmax, rinv, SCALE, DOT_PREC, LOWP)
        carry = tl.load(c_base + n_offs * stride_st, mask=n_offs < T, other=0.0)
        committed = carry[None, :] + tl.cumsum(p, axis=0) - p
        inv = 1.0 / (committed + p + 1e-9)  # one reciprocal per element, reused
        return causal, p, committed, inv

    @triton.jit
    def _share_tile(q_tile, k_base, n_offs, d_offs, stride_qt, stride_qd,
                    m_offs, T, rmax, rinv, c_base, stride_st,
                    SCALE: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr):
        """Load one key tile, then _share_from_k."""
        k_tile = tl.load(
            k_base + n_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
            mask=(n_offs < T)[:, None], other=0.0,
        )
        return _share_from_k(q_tile, k_tile, n_offs, m_offs, T, rmax, rinv, c_base, stride_st,
                             SCALE, DOT_PREC, LOWP)

    @triton.jit
    def _dilution_k1(
        Q, K, RMAX, RDEN, S,
        stride_qn, stride_qt, stride_qd,
        stride_sn, stride_sb, stride_st,
        T, HEAD_DIM: tl.constexpr, SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr,
        PK=None, STATS_ONLY: tl.constexpr = False,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        m_start = pid_m * BLOCK_M
        m_offs = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offs < T
        d_offs = tl.arange(0, HEAD_DIM)
        q_base = Q + pid_n * stride_qn
        k_base = K + pid_n * stride_qn
        q_tile = tl.load(
            q_base + m_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
            mask=m_mask[:, None], other=0.0,
        )

        # Sweep 1: online row max/denominator of the causal softmax.
        rmax = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        rden = tl.zeros((BLOCK_M,), tl.float32)
        if PK is not None:
            # Online softmax with K as the value operand computes p @ k in
            # the statistics sweep. Rescale it whenever the running max rises.
            pk_acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
        for n_start in range(0, m_start + BLOCK_M, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offs < T
            k_tile = tl.load(
                k_base + n_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                mask=n_mask[:, None], other=0.0,
            )
            aff = _dot(q_tile, tl.trans(k_tile), LOWP, DOT_PREC) * (SCALE * LOG2E)
            causal = (n_offs[None, :] <= m_offs[:, None]) & n_mask[None, :] & m_mask[:, None]
            aff = tl.where(causal, aff, float("-inf"))
            tile_max = tl.max(aff, axis=1)
            new_max = tl.maximum(rmax, tile_max)
            alpha = tl.exp2(rmax - new_max)
            exp_aff = tl.exp2(aff - new_max[:, None])
            if PK is not None:
                pk_acc = pk_acc * alpha[:, None] + _dot(exp_aff, k_tile, LOWP, DOT_PREC)
            rden = rden * alpha + tl.sum(exp_aff, axis=1)
            rmax = new_max
        tl.store(RMAX + pid_n * T + m_offs, rmax, mask=m_mask)
        tl.store(RDEN + pid_n * T + m_offs, rden, mask=m_mask)
        rinv = 1.0 / rden
        if PK is not None:
            tl.store(PK + pid_n * T * HEAD_DIM + m_offs[:, None] * HEAD_DIM + d_offs[None, :],
                     pk_acc * rinv[:, None] * SCALE, mask=m_mask[:, None])

        if not STATS_ONLY:
            # Sweep 2: per-block key-sums of p (this block's rows only).
            s_base = S + pid_n.to(tl.int64) * stride_sn + pid_m.to(tl.int64) * stride_sb
            for n_start in range(0, m_start + BLOCK_M, BLOCK_N):
                n_offs = n_start + tl.arange(0, BLOCK_N)
                k_tile = tl.load(k_base + n_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                                 mask=(n_offs < T)[:, None], other=0)
                _, p = _p_from_k(q_tile, k_tile, n_offs, m_offs, T, rmax, rinv, SCALE, DOT_PREC, LOWP)
                tl.store(s_base + n_offs * stride_st, tl.sum(p, axis=0), mask=n_offs < T)

    @triton.jit
    def _dilution_key_forward(
        Q, K, V, OUT, SSUM, RMAX, RDEN, CARRY, TOTAL,
        T: tl.constexpr, NB: tl.constexpr, HEAD_DIM: tl.constexpr, SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr,
    ):
        """Visit queries in order for each key block, carrying its demand locally.

        This combines K1's key-sum sweep, the block scan, and K2. Each bid
        tile is computed once here; only carry checkpoints reach DRAM.
        OUT and SSUM must be zeroed. Their consumers run after this launch,
        so the row contributions need only relaxed atomic accumulation.
        """
        pid_n = tl.program_id(0)
        pid_kb = tl.program_id(1)
        j_offs = pid_kb * BLOCK_N + tl.arange(0, BLOCK_N)
        d_offs = tl.arange(0, HEAD_DIM)
        key_ptr = pid_n * T * HEAD_DIM + j_offs[:, None] * HEAD_DIM + d_offs[None, :]
        k_tile = tl.load(K + key_ptr, mask=(j_offs < T)[:, None], other=0.0)
        v_tile = tl.load(V + key_ptr, mask=(j_offs < T)[:, None], other=0.0)
        carry = tl.zeros((BLOCK_N,), tl.float32)
        carry_base = CARRY + pid_n.to(tl.int64) * NB * T   # (n, NB, T) exceeds int32 at long context
        for rb in range((pid_kb * BLOCK_N) // BLOCK_M, NB):
            m_offs = rb * BLOCK_M + tl.arange(0, BLOCK_M)
            row_ptr = pid_n * T * HEAD_DIM + m_offs[:, None] * HEAD_DIM + d_offs[None, :]
            q_tile = tl.load(Q + row_ptr, mask=(m_offs < T)[:, None], other=0.0)
            rmax = tl.load(RMAX + pid_n * T + m_offs, mask=m_offs < T, other=0.0)
            rden = tl.load(RDEN + pid_n * T + m_offs, mask=m_offs < T, other=1.0)
            causal, p = _p_from_k(q_tile, k_tile, j_offs, m_offs, T, rmax, 1.0 / rden,
                                  SCALE, DOT_PREC, LOWP)
            tl.store(carry_base + rb * T + j_offs, carry, mask=j_offs < T)
            inclusive = carry[None, :] + tl.cumsum(p, axis=0)
            carry += tl.sum(p, axis=0)
            share = tl.where(causal, p / (inclusive + 1e-9), 0.0)
            tl.atomic_add(SSUM + pid_n * T + m_offs, tl.sum(share, axis=1),
                          mask=m_offs < T, sem="relaxed")
            part = _dot(share, v_tile, LOWP, DOT_PREC)
            tl.atomic_add(OUT + row_ptr, part, mask=(m_offs < T)[:, None], sem="relaxed")
        tl.store(TOTAL + pid_n * T + j_offs, carry, mask=j_offs < T)

    @triton.jit
    def _dilution_normalize(OUT, SSUM, T: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK: tl.constexpr):
        pid_n = tl.program_id(0)
        m_offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        d_offs = tl.arange(0, HEAD_DIM)
        ptr = OUT + pid_n * T * HEAD_DIM + m_offs[:, None] * HEAD_DIM + d_offs[None, :]
        out = tl.load(ptr, mask=(m_offs < T)[:, None], other=0.0)
        den = tl.load(SSUM + pid_n * T + m_offs, mask=m_offs < T, other=1.0)
        tl.store(ptr, out / (den[:, None] + 1e-9), mask=(m_offs < T)[:, None])

    @triton.jit
    def _dilution_scan(S, CARRY, TOTAL, T, NB: tl.constexpr,
                       BLOCK_M: tl.constexpr, SCAN_B: tl.constexpr, KEYS: tl.constexpr):
        # Key tiles on axis 0: CUDA caps grid dim 1 at 65535 and T/KEYS exceeds it at 256K.
        keys = tl.program_id(0) * KEYS + tl.arange(0, KEYS)
        n = tl.program_id(1)
        blocks = tl.arange(0, SCAN_B)
        offsets = n.to(tl.int64) * NB * T + blocks[:, None].to(tl.int64) * T + keys[None, :]
        valid = (blocks < NB)[:, None] & (keys < T)[None, :]
        # K1 only writes the causal part of each block's key sums.
        sums = tl.load(S + offsets,
                       mask=valid & (keys[None, :] < (blocks[:, None] + 1) * BLOCK_M), other=0)
        inclusive = tl.cumsum(sums, axis=0)
        tl.store(CARRY + offsets, inclusive - sums, mask=valid)
        tl.store(TOTAL + n * T + keys, tl.sum(sums, axis=0), mask=keys < T)

    @triton.jit
    def _dilution_k2(
        Q, K, V, OUT, SSUM, RMAX, RDEN, CARRY,
        stride_qn, stride_qt, stride_qd,
        stride_on, stride_ot, stride_od,
        stride_sn, stride_sb, stride_st,
        T, HEAD_DIM: tl.constexpr, SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        m_start = pid_m * BLOCK_M
        m_offs = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offs < T
        d_offs = tl.arange(0, HEAD_DIM)
        q_base = Q + pid_n * stride_qn
        k_base = K + pid_n * stride_qn
        v_base = V + pid_n * stride_qn
        q_tile = tl.load(
            q_base + m_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
            mask=m_mask[:, None], other=0.0,
        )
        rmax = tl.load(RMAX + pid_n * T + m_offs, mask=m_mask, other=0.0)
        rden = tl.load(RDEN + pid_n * T + m_offs, mask=m_mask, other=1.0)
        rinv = 1.0 / rden
        c_base = CARRY + pid_n.to(tl.int64) * stride_sn + pid_m.to(tl.int64) * stride_sb

        acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
        share_sum = tl.zeros((BLOCK_M,), tl.float32)
        for n_start in range(0, m_start + BLOCK_M, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offs < T
            causal, p, committed, inv = _share_tile(
                q_tile, k_base, n_offs, d_offs, stride_qt, stride_qd, m_offs, T,
                rmax, rinv, c_base, stride_st, SCALE, DOT_PREC, LOWP)
            share = tl.where(causal, p * inv, 0.0)
            share_sum += tl.sum(share, axis=1)

            v_tile = tl.load(
                v_base + n_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                mask=n_mask[:, None], other=0.0,
            )
            acc += _dot(share, v_tile, LOWP, DOT_PREC)

        out = acc / (share_sum[:, None] + 1e-9)
        tl.store(
            OUT + pid_n * stride_on + m_offs[:, None] * stride_ot + d_offs[None, :] * stride_od,
            out, mask=m_mask[:, None],
        )
        tl.store(SSUM + pid_n * T + m_offs, share_sum, mask=m_mask)


def _resolve_precision(dot_precision: str) -> tuple[int, str]:
    """"ieee" / "tf32": fp32 operands; "bf16": bf16 tensor-core operands."""

    if dot_precision == "bf16":
        return 1, "tf32"
    if dot_precision in ("ieee", "tf32"):
        return 0, dot_precision
    raise ValueError("dot_precision must be one of ieee, tf32, bf16")


def dilution_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dot_precision: str = "ieee",
    block_m: int = 64,
    block_n: int = 64,
    return_aux: bool = False,
    num_warps: int = 4,
    num_stages: int = 2,
    _save_pk: bool = False,
    scratch_budget_gib: float = 16.0,
):
    """Fused forward; returns fp32 outputs.

    With return_aux=True also returns the statistics the backward kernels
    need (row stats, block carry, share row-sums) and the inclusive column
    totals of p, which are the cached-decode accumulator after this prefix.
    Without return_aux the (batch, head) rows are processed in chunks sized to
    scratch_budget_gib, since the row-parallel scratch is (n, T/64, T) fp32 twice
    over and therefore quadratic in T (96 GiB at H=12, T=256K). Rows are
    independent, so this changes nothing numerically.

    Internally, training sets _save_pk to save the query Jacobian correction
    and select the key-parallel forward above KEY_FORWARD_MIN_SEQ tokens; that path stores carry
    checkpoints every aux["bwd_block_m"] (32) rows. Carry entries for future keys
    in that path are unwritten and must not be read by the causal backward.
    """

    if not HAS_TRITON:
        raise RuntimeError("triton is not available")
    batch, heads, seq, head_dim = q.shape
    n = batch * heads
    n_blocks = (seq + block_m - 1) // block_m
    qf = q.reshape(n, seq, head_dim).contiguous()
    kf = k.reshape(n, seq, head_dim).contiguous()
    vf = v.reshape(n, seq, head_dim).contiguous()
    dev = q.device
    # The key-parallel pass pays at longer contexts. Keep the row-parallel
    # path for short sequences and forward-only / prefill calls.
    key_forward = _save_pk and seq > KEY_FORWARD_MIN_SEQ
    alloc = torch.zeros if key_forward else torch.empty
    out = alloc(n, seq, head_dim, device=dev, dtype=torch.float32)
    pk = torch.empty(n, seq, head_dim, device=dev, dtype=torch.float32) if _save_pk else None
    rmax = torch.empty(n, seq, device=dev, dtype=torch.float32)
    rden = torch.empty(n, seq, device=dev, dtype=torch.float32)
    ssum = alloc(n, seq, device=dev, dtype=torch.float32)
    # Training (key-parallel) path: forward row tiles and carry checkpoints at
    # bwd_block_m = 32 rows, the tile height the register-bound backward pass
    # BA needs. The key-parallel forward is neutral at 32 rows.
    bwd_block_m = 32 if key_forward else block_m
    nb_bwd = (seq + bwd_block_m - 1) // bwd_block_m
    # Scratch is (rows, nb_bwd, seq) fp32, twice over on the row-parallel path.
    per_row = nb_bwd * seq * 4 * (1 if key_forward else 2)
    rows = n
    if not return_aux and not key_forward:
        rows = max(1, min(n, int(scratch_budget_gib * 2**30) // max(per_row, 1)))
    carry = torch.empty(rows, nb_bwd, seq, device=dev, dtype=torch.float32)
    s = None if key_forward else torch.empty_like(carry)

    scale = 1.0 / math.sqrt(head_dim)
    grid = (n, n_blocks)
    lowp, prec = _resolve_precision(dot_precision)
    common = dict(BLOCK_M=block_m, BLOCK_N=block_n, DOT_PREC=prec, LOWP=lowp,
                  num_warps=num_warps, num_stages=num_stages)

    committed_total = torch.empty(n, seq, device=dev, dtype=torch.float32)
    if rows < n:
        # Row-chunked forward-only path: same kernels, scratch reused per chunk.
        for a in range(0, n, rows):
            b = min(a + rows, n)
            m = b - a
            _dilution_k1[(m, n_blocks)](
                qf[a:b], kf[a:b], rmax[a:b], rden[a:b], s[:m],
                qf.stride(0), qf.stride(1), qf.stride(2),
                carry.stride(0), carry.stride(1), carry.stride(2),
                seq, head_dim, scale, PK=None, STATS_ONLY=False, **common,
            )
            scan_b = triton.next_power_of_2(n_blocks)
            keys = 32 if scan_b <= 512 else (16 if scan_b <= 1024 else (8 if scan_b <= 2048 else 4))
            _dilution_scan[(triton.cdiv(seq, keys), m)](
                s[:m], carry[:m], committed_total[a:b], seq, n_blocks, BLOCK_M=block_m,
                SCAN_B=scan_b, KEYS=keys, num_warps=4)
            _dilution_k2[(m, n_blocks)](
                qf[a:b], kf[a:b], vf[a:b], out[a:b], ssum[a:b], rmax[a:b], rden[a:b], carry[:m],
                qf.stride(0), qf.stride(1), qf.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                carry.stride(0), carry.stride(1), carry.stride(2),
                seq, head_dim, scale, **common,
            )
        return out.reshape(batch, heads, seq, head_dim)

    _dilution_k1[grid](
        qf, kf, rmax, rden, s,
        qf.stride(0), qf.stride(1), qf.stride(2),
        carry.stride(0), carry.stride(1), carry.stride(2),
        seq, head_dim, scale, PK=pk, STATS_ONLY=key_forward, **common,
    )
    if key_forward:
        _dilution_key_forward[(n, triton.cdiv(seq, block_n))](
            qf, kf, vf, out, ssum, rmax, rden, carry, committed_total,
            seq, nb_bwd, head_dim, scale, **dict(common, BLOCK_M=bwd_block_m))
        _dilution_normalize[(n, triton.cdiv(seq, 128))](out, ssum, seq, head_dim, BLOCK=128)
    else:
        # The scan tile is SCAN_B x KEYS fp32 in shared memory; keep it under the
        # ~100 KB limit as the block count grows (S12: 48K+ contexts OOMed at KEYS=32).
        scan_b = triton.next_power_of_2(n_blocks)
        keys = 32 if scan_b <= 512 else (16 if scan_b <= 1024 else (8 if scan_b <= 2048 else 4))
        _dilution_scan[(triton.cdiv(seq, keys), n)](
            s, carry, committed_total, seq, n_blocks, BLOCK_M=block_m,
            SCAN_B=scan_b, KEYS=keys, num_warps=4)
        _dilution_k2[grid](
            qf, kf, vf, out, ssum, rmax, rden, carry,
            qf.stride(0), qf.stride(1), qf.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            carry.stride(0), carry.stride(1), carry.stride(2),
            seq, head_dim, scale, **common,
        )
    result = out.reshape(batch, heads, seq, head_dim)
    if return_aux:
        aux = dict(pk=pk, rmax=rmax, rden=rden, carry=carry, ssum=ssum, bwd_block_m=bwd_block_m,
                   block_m=block_m, block_n=block_n, dot_precision=dot_precision,
                   num_warps=num_warps,
                   # Inclusive column totals over all rows: the decode
                   # accumulator (committed) after this prefix.
                   committed_total=committed_total.reshape(batch, heads, seq))
        return result, aux
    return result


if HAS_TRITON:

    @triton.jit
    def _ds_from_v(do_tile, v_tile, dbar, sinv, causal, DOT_PREC: tl.constexpr, LOWP: tl.constexpr):
        """ds = (do . v_j - do . o_i) / (S_i + eps), masked to the causal tile."""
        dov = _dot(do_tile, tl.trans(v_tile), LOWP, DOT_PREC)
        ds = (dov - dbar[:, None]) * sinv[:, None]
        return tl.where(causal, ds, 0.0)

    @triton.jit
    def _ds_tile(do_tile, v_base, n_offs, d_offs, stride_qt, stride_qd, T,
                 dbar, sinv, causal, DOT_PREC: tl.constexpr, LOWP: tl.constexpr):
        """Load one value tile, then _ds_from_v."""
        v_tile = tl.load(
            v_base + n_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
            mask=(n_offs < T)[:, None], other=0.0,
        )
        return _ds_from_v(do_tile, v_tile, dbar, sinv, causal, DOT_PREC, LOWP)

    @triton.jit
    def _dilution_b1(
        Q, K, V, DO, DBAR, SSUM, RMAX, RDEN, CARRY, SK, GPART,
        stride_qn, stride_qt, stride_qd,
        stride_sn, stride_sb, stride_st,
        T, HEAD_DIM: tl.constexpr, SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr,
    ):
        """Per-block key-sums of dk_share = -ds * p / (committed+p+eps)^2, plus the
        part of ghat_i = sum_j dp_ij p_ij that this row block can see on its own.

        dp_ij = ds_ij (committed_ij+eps) inv_ij^2 + suffix_ij, where suffix_ij is
        the sum of dk_share over every later query on key j. Its within-block
        part is computable here; only the cross-block part (the reverse block
        scan of SK) is left for B2, which then needs nothing but p.
        """
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        m_start = pid_m * BLOCK_M
        m_offs = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offs < T
        d_offs = tl.arange(0, HEAD_DIM)
        q_base = Q + pid_n * stride_qn
        k_base = K + pid_n * stride_qn
        v_base = V + pid_n * stride_qn
        do_base = DO + pid_n * stride_qn
        q_tile = tl.load(q_base + m_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                         mask=m_mask[:, None], other=0.0)
        do_tile = tl.load(do_base + m_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                          mask=m_mask[:, None], other=0.0)
        rmax = tl.load(RMAX + pid_n * T + m_offs, mask=m_mask, other=0.0)
        rden = tl.load(RDEN + pid_n * T + m_offs, mask=m_mask, other=1.0)
        rinv = 1.0 / rden
        dbar = tl.load(DBAR + pid_n * T + m_offs, mask=m_mask, other=0.0)
        ssum = tl.load(SSUM + pid_n * T + m_offs, mask=m_mask, other=1.0)
        sinv = 1.0 / (ssum + 1e-9)
        c_base = CARRY + pid_n.to(tl.int64) * stride_sn + pid_m * stride_sb
        sk_base = SK + pid_n.to(tl.int64) * stride_sn + pid_m * stride_sb

        gpart = tl.zeros((BLOCK_M,), tl.float32)
        for n_start in range(0, m_start + BLOCK_M, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            causal, p, committed, inv = _share_tile(
                q_tile, k_base, n_offs, d_offs, stride_qt, stride_qd, m_offs, T,
                rmax, rinv, c_base, stride_st, SCALE, DOT_PREC, LOWP)
            ds = _ds_tile(do_tile, v_base, n_offs, d_offs, stride_qt, stride_qd, T,
                          dbar, sinv, causal, DOT_PREC, LOWP)
            inv2 = inv * inv
            dk_share = -ds * p * inv2
            colsum = tl.sum(dk_share, axis=0)
            tl.store(sk_base + n_offs * stride_st, colsum, mask=n_offs < T)
            inblock = colsum[None, :] - tl.cumsum(dk_share, axis=0)  # later rows in this block
            gpart += tl.sum(p * (ds * (committed + 1e-9) * inv2 + inblock), axis=1)
        tl.store(GPART + pid_n * T + m_offs, gpart, mask=m_mask)

    @triton.jit
    def _dilution_b2(
        Q, K, RMAX, RDEN, SUFK, GPART, GHAT,
        stride_qn, stride_qt, stride_qd,
        stride_sn, stride_sb, stride_st,
        T, HEAD_DIM: tl.constexpr, SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr,
    ):
        """ghat_i = gpart_i + sum_j p_ij * sufk_j: the cross-block suffix term.

        Only p is recomputed here (one q.k product per tile); the share chain,
        ds and the in-block scan were already folded into GPART by B1.
        """
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        m_start = pid_m * BLOCK_M
        m_offs = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offs < T
        d_offs = tl.arange(0, HEAD_DIM)
        q_base = Q + pid_n * stride_qn
        k_base = K + pid_n * stride_qn
        q_tile = tl.load(q_base + m_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                         mask=m_mask[:, None], other=0.0)
        rmax = tl.load(RMAX + pid_n * T + m_offs, mask=m_mask, other=0.0)
        rden = tl.load(RDEN + pid_n * T + m_offs, mask=m_mask, other=1.0)
        rinv = 1.0 / rden
        sufk_base = SUFK + pid_n.to(tl.int64) * stride_sn + pid_m * stride_sb

        ghat = tl.load(GPART + pid_n * T + m_offs, mask=m_mask, other=0.0)
        for n_start in range(0, m_start + BLOCK_M, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            _, p = _p_tile(q_tile, k_base, n_offs, d_offs, stride_qt, stride_qd,
                           m_offs, T, rmax, rinv, SCALE, DOT_PREC, LOWP)
            sufk = tl.load(sufk_base + n_offs * stride_st, mask=n_offs < T, other=0.0)
            ghat += tl.sum(p * sufk[None, :], axis=1)
        tl.store(GHAT + pid_n * T + m_offs, ghat, mask=m_mask)

    @triton.jit
    def _dilution_b3(
        Q, K, DO, DV, SSUM, RMAX, RDEN, CARRY,
        stride_qn, stride_qt, stride_qd,
        stride_sn, stride_sb, stride_st,
        T, NB, HEAD_DIM: tl.constexpr, SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr,
    ):
        """Key-parallel dv = attn^T do. Kept separate from dk/dq on purpose: with
        only p, inv and the accumulator live per tile it runs at low register
        pressure, and the recompute it costs (one q.k product and one scan per
        tile) is cheaper than what the fused kernel lost to spilling."""
        pid_n = tl.program_id(0)
        pid_kb = tl.program_id(1)  # key block
        j_offs = pid_kb * BLOCK_N + tl.arange(0, BLOCK_N)
        j_mask = j_offs < T
        d_offs = tl.arange(0, HEAD_DIM)
        q_base = Q + pid_n * stride_qn
        k_base = K + pid_n * stride_qn
        do_base = DO + pid_n * stride_qn
        k_blk = tl.load(k_base + j_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                        mask=j_mask[:, None], other=0.0)
        dv_acc = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
        carry_base = CARRY + pid_n.to(tl.int64) * stride_sn
        first_rb = (pid_kb * BLOCK_N) // BLOCK_M
        for rb in range(first_rb, NB):
            m_offs = rb * BLOCK_M + tl.arange(0, BLOCK_M)
            m_mask = m_offs < T
            q_tile = tl.load(q_base + m_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                             mask=m_mask[:, None], other=0.0)
            do_tile = tl.load(do_base + m_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                              mask=m_mask[:, None], other=0.0)
            rmax = tl.load(RMAX + pid_n * T + m_offs, mask=m_mask, other=0.0)
            rden = tl.load(RDEN + pid_n * T + m_offs, mask=m_mask, other=1.0)
            ssum = tl.load(SSUM + pid_n * T + m_offs, mask=m_mask, other=1.0)
            c_base = carry_base + rb * stride_sb
            causal, p, committed, inv = _share_from_k(
                q_tile, k_blk, j_offs, m_offs, T, rmax, 1.0 / rden, c_base, stride_st,
                SCALE, DOT_PREC, LOWP)
            attn = tl.where(causal, p * inv * (1.0 / (ssum + 1e-9))[:, None], 0.0)
            dv_acc += _dot(tl.trans(attn), do_tile, LOWP, DOT_PREC)
        tl.store(DV + pid_n * stride_qn + j_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                 dv_acc, mask=j_mask[:, None])

    @triton.jit
    def _dilution_b4(
        Q, K, V, DO, DK, DQ, DBAR, SSUM, RMAX, RDEN, CARRY, SUFK, GHAT,
        stride_qn, stride_qt, stride_qd,
        stride_sn, stride_sb, stride_st,
        T, NB, HEAD_DIM: tl.constexpr, SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr,
    ):
        """Key-parallel dk = da^T q; dq accumulated with fp32 atomics from the same tiles."""
        pid_n = tl.program_id(0)
        pid_kb = tl.program_id(1)  # key block
        j_offs = pid_kb * BLOCK_N + tl.arange(0, BLOCK_N)
        j_mask = j_offs < T
        d_offs = tl.arange(0, HEAD_DIM)
        q_base = Q + pid_n * stride_qn
        k_base = K + pid_n * stride_qn
        v_base = V + pid_n * stride_qn
        do_base = DO + pid_n * stride_qn
        # K and V for this key block are constant across the row-block loop.
        k_blk = tl.load(k_base + j_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                        mask=j_mask[:, None], other=0.0)
        v_blk = tl.load(v_base + j_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                        mask=j_mask[:, None], other=0.0)
        dk_acc = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
        carry_base = CARRY + pid_n.to(tl.int64) * stride_sn
        sufk_base_n = SUFK + pid_n.to(tl.int64) * stride_sn
        first_rb = (pid_kb * BLOCK_N) // BLOCK_M
        for rb in range(first_rb, NB):
            m_offs = rb * BLOCK_M + tl.arange(0, BLOCK_M)
            m_mask = m_offs < T
            q_tile = tl.load(q_base + m_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                             mask=m_mask[:, None], other=0.0)
            do_tile = tl.load(do_base + m_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                              mask=m_mask[:, None], other=0.0)
            rmax = tl.load(RMAX + pid_n * T + m_offs, mask=m_mask, other=0.0)
            rden = tl.load(RDEN + pid_n * T + m_offs, mask=m_mask, other=1.0)
            dbar = tl.load(DBAR + pid_n * T + m_offs, mask=m_mask, other=0.0)
            ssum = tl.load(SSUM + pid_n * T + m_offs, mask=m_mask, other=1.0)
            ghat = tl.load(GHAT + pid_n * T + m_offs, mask=m_mask, other=0.0)
            c_base = carry_base + rb * stride_sb
            sufk_base = sufk_base_n + rb * stride_sb

            causal, p, committed, inv = _share_from_k(
                q_tile, k_blk, j_offs, m_offs, T, rmax, 1.0 / rden, c_base, stride_st,
                SCALE, DOT_PREC, LOWP)
            ds = _ds_from_v(do_tile, v_blk, dbar, 1.0 / (ssum + 1e-9), causal, DOT_PREC, LOWP)
            inv2 = inv * inv
            dk_share = -ds * p * inv2
            sufk = tl.load(sufk_base + j_offs * stride_st, mask=j_mask, other=0.0)
            suffix = sufk[None, :] + tl.sum(dk_share, axis=0)[None, :] - tl.cumsum(dk_share, axis=0)
            da = tl.where(causal, p * (ds * (committed + 1e-9) * inv2 + suffix - ghat[:, None]), 0.0)
            dk_acc += _dot(tl.trans(da), q_tile, LOWP, DOT_PREC)
            # dq is row-indexed; accumulate this key block's contribution atomically
            # (fp32), which removes an entire row-parallel recompute sweep.
            dq_part = _dot(da, k_blk, LOWP, DOT_PREC) * SCALE
            # Relaxed ordering: nothing in-kernel reads DQ and the launch boundary is a
            # full sync; the default acq_rel fences cost 2.4 ms of B4's 5.7 (S26).
            tl.atomic_add(DQ + pid_n * stride_qn + m_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                          dq_part, mask=m_mask[:, None], sem="relaxed")

        tl.store(DK + pid_n * stride_qn + j_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                 dk_acc * SCALE, mask=j_mask[:, None])


    @triton.jit
    def _dilution_ba(
        Q, K, V, DO, DK, DQ, DV, GHAT, DBAR, SSUM, RMAX, RDEN, CARRY,
        stride_qn, stride_qt, stride_qd,
        stride_sn, stride_sb, stride_st,
        T, NB, HEAD_DIM: tl.constexpr, SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr,
    ):
        """Key-parallel heavy pass of the backward (docs/kernels.md). Iterates row blocks from
        the last to the first, so the per-key suffix of dk_share over later rows
        (sufk) is a running sum -- no B1 key-sums, no reverse block scan. With the
        exact dp = ds (committed+eps) inv^2 + suffix in hand it accumulates the
        ghat-free parts of the gradients:
            dk_main[j] += SCALE * sum_i (p dp)_ij q_i          (local)
            dv[j]      += sum_i attn_ij do_i                   (local)
            dq_main[i] += SCALE * sum_j (p dp)_ij k_j          (relaxed fp32 atomics)
            ghat[i]    += sum_j (p dp)_ij                      (relaxed 64-wide atomics)
        The softmax Jacobian's -ghat term is applied afterwards: dk -= p^T (ghat q)
        in _dilution_bc and dq -= ghat * pk in the epilogue. K1 saves
        pk = SCALE * p @ k during its statistics sweep, avoiding a product and
        atomic accumulation in this register-heavy pass. Relaxed ordering
        is exact here: nothing in-kernel reads DQ / GHAT.

        dv is accumulated here from the same attn tile (formed before ds so
        the two are never live together) and this pass runs on 32-row tiles
        (BLOCK_M = the forward's checkpoint granularity), which is what keeps
        the extra product from spilling. dv is stored in DV's dtype.

        The next row block's q / do tiles are loaded one iteration ahead, and
        the pass is launched with num_stages=2 so the remaining per-row loads are
        pipelined too. Neither helps alone; together they take BA from 3.14 to
        2.82 ms at B8 H8 T4096. Prefetching the row stats and carry as well is
        slower (more spills).
        """
        pid_n = tl.program_id(0)
        pid_kb = tl.program_id(1)
        j_offs = pid_kb * BLOCK_N + tl.arange(0, BLOCK_N)
        j_mask = j_offs < T
        d_offs = tl.arange(0, HEAD_DIM)
        q_base = Q + pid_n * stride_qn
        k_base = K + pid_n * stride_qn
        v_base = V + pid_n * stride_qn
        do_base = DO + pid_n * stride_qn
        k_blk = tl.load(k_base + j_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                        mask=j_mask[:, None], other=0.0)
        v_blk = tl.load(v_base + j_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                        mask=j_mask[:, None], other=0.0)
        dk_acc = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
        dv_acc = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
        carry_base = CARRY + pid_n.to(tl.int64) * stride_sn
        sufk = tl.zeros((BLOCK_N,), tl.float32)
        first_rb = (pid_kb * BLOCK_N) // BLOCK_M
        # q / do for the first (last-in-sequence) row block; each iteration then
        # issues the next block's loads before computing on the current one.
        m_nx = (NB - 1) * BLOCK_M + tl.arange(0, BLOCK_M)
        q_nx = tl.load(q_base + m_nx[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                       mask=(m_nx < T)[:, None], other=0.0)
        do_nx = tl.load(do_base + m_nx[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                        mask=(m_nx < T)[:, None], other=0.0)
        for step in range(0, NB - first_rb):
            rb = NB - 1 - step
            m_offs = rb * BLOCK_M + tl.arange(0, BLOCK_M)
            m_mask = m_offs < T
            q_tile = q_nx
            do_tile = do_nx
            m_nx = m_offs - BLOCK_M
            nx_mask = ((m_nx >= 0) & (m_nx < T))[:, None]
            q_nx = tl.load(q_base + m_nx[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                           mask=nx_mask, other=0.0)
            do_nx = tl.load(do_base + m_nx[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                            mask=nx_mask, other=0.0)
            rmax = tl.load(RMAX + pid_n * T + m_offs, mask=m_mask, other=0.0)
            rden = tl.load(RDEN + pid_n * T + m_offs, mask=m_mask, other=1.0)
            dbar = tl.load(DBAR + pid_n * T + m_offs, mask=m_mask, other=0.0)
            ssum = tl.load(SSUM + pid_n * T + m_offs, mask=m_mask, other=1.0)
            c_base = carry_base + rb * stride_sb
            causal, p, committed, inv = _share_from_k(
                q_tile, k_blk, j_offs, m_offs, T, rmax, 1.0 / rden, c_base, stride_st,
                SCALE, DOT_PREC, LOWP)
            sinv = 1.0 / (ssum + 1e-9)
            attn = tl.where(causal, p * inv * sinv[:, None], 0.0)
            dv_acc += _dot(tl.trans(attn), do_tile, LOWP, DOT_PREC)
            ds = _ds_from_v(do_tile, v_blk, dbar, sinv, causal, DOT_PREC, LOWP)
            inv2 = inv * inv
            dk_share = -ds * p * inv2
            colsum = tl.sum(dk_share, axis=0)
            suffix = sufk[None, :] + colsum[None, :] - tl.cumsum(dk_share, axis=0)
            sufk += colsum
            pdp = tl.where(causal, p * (ds * (committed + 1e-9) * inv2 + suffix), 0.0)
            tl.atomic_add(GHAT + pid_n * T + m_offs, tl.sum(pdp, axis=1), mask=m_mask, sem="relaxed")
            dk_acc += _dot(tl.trans(pdp), q_tile, LOWP, DOT_PREC)
            dq_part = _dot(pdp, k_blk, LOWP, DOT_PREC) * SCALE
            tl.atomic_add(DQ + pid_n * stride_qn + m_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                          dq_part, mask=m_mask[:, None], sem="relaxed")

        tl.store(DK + pid_n * stride_qn + j_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                 dk_acc * SCALE, mask=j_mask[:, None])
        tl.store(DV + pid_n * stride_qn + j_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                 dv_acc.to(DV.dtype.element_ty), mask=j_mask[:, None])

    @triton.jit
    def _dilution_bc(
        Q, K, DKM, DK, GHAT, RMAX, RDEN,
        stride_qn, stride_qt, stride_qd,
        T, NB, HEAD_DIM: tl.constexpr, SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr,
    ):
        """Softmax-Jacobian correction for dk: DK = DKM - SCALE * p^T (ghat q),
        stored in DK's dtype. Needs only p (row stats), not the share chain: no
        carry, no cumsum, no do tile. This is what B3C became once dv moved to BA."""
        pid_n = tl.program_id(0)
        pid_kb = tl.program_id(1)
        j_offs = pid_kb * BLOCK_N + tl.arange(0, BLOCK_N)
        j_mask = j_offs < T
        d_offs = tl.arange(0, HEAD_DIM)
        q_base = Q + pid_n * stride_qn
        k_blk = tl.load(K + pid_n * stride_qn + j_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                        mask=j_mask[:, None], other=0.0)
        dkc_acc = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
        first_rb = (pid_kb * BLOCK_N) // BLOCK_M
        for rb in range(first_rb, NB):
            m_offs = rb * BLOCK_M + tl.arange(0, BLOCK_M)
            m_mask = m_offs < T
            q_tile = tl.load(q_base + m_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                             mask=m_mask[:, None], other=0.0)
            rmax = tl.load(RMAX + pid_n * T + m_offs, mask=m_mask, other=0.0)
            rden = tl.load(RDEN + pid_n * T + m_offs, mask=m_mask, other=1.0)
            ghat = tl.load(GHAT + pid_n * T + m_offs, mask=m_mask, other=0.0)
            causal, p = _p_from_k(q_tile, k_blk, j_offs, m_offs, T, rmax, 1.0 / rden, SCALE, DOT_PREC, LOWP)
            dkc_acc += _dot(tl.trans(p * ghat[:, None]), q_tile, LOWP, DOT_PREC)
        off = pid_n * stride_qn + j_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd
        dk_main = tl.load(DKM + off, mask=j_mask[:, None], other=0.0)
        tl.store(DK + off, (dk_main - dkc_acc * SCALE).to(DK.dtype.element_ty), mask=j_mask[:, None])


if HAS_TRITON:

    @triton.jit
    def _dilution_dq_epilogue(DQM, GHAT, PK, DQ, T, HEAD_DIM: tl.constexpr, BLOCK: tl.constexpr):
        """dq = dq_main - ghat[:, None] * pk, stored in DQ's dtype."""
        pid_n = tl.program_id(0)
        pid_t = tl.program_id(1)
        t_offs = pid_t * BLOCK + tl.arange(0, BLOCK)
        d_offs = tl.arange(0, HEAD_DIM)
        mask = (t_offs < T)[:, None]
        base = pid_n * T * HEAD_DIM + t_offs[:, None] * HEAD_DIM + d_offs[None, :]
        dqm = tl.load(DQM + base, mask=mask, other=0.0)
        pk = tl.load(PK + base, mask=mask, other=0.0)
        g = tl.load(GHAT + pid_n * T + t_offs, mask=t_offs < T, other=0.0)
        tl.store(DQ + base, (dqm - g[:, None] * pk).to(DQ.dtype.element_ty), mask=mask)


class _DilutionAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, dot_precision, block_m, block_n,
                num_warps, bwd_num_warps, bwd_num_stages, b3_block_n, b3_num_warps, b4_num_warps):
        out, aux = dilution_attention_forward(
            q, k, v,
            dot_precision=dot_precision, block_m=block_m, block_n=block_n,
            return_aux=True, num_warps=num_warps, _save_pk=any(ctx.needs_input_grad[:3]),
        )
        ctx.bwd_num_warps = bwd_num_warps
        ctx.bwd_num_stages = bwd_num_stages
        ctx.b3_block_n = b3_block_n
        ctx.b3_num_warps = b3_num_warps
        ctx.b4_num_warps = b4_num_warps
        ctx.save_for_backward(q, k, v, out, aux["rmax"], aux["rden"], aux["carry"], aux["ssum"], aux["pk"])
        ctx.dot_precision = dot_precision
        ctx.bwd_block_m = aux["bwd_block_m"]
        ctx.block_m = block_m
        ctx.block_n = block_n
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, rmax, rden, carry, ssum, pk = ctx.saved_tensors
        batch, heads, seq, head_dim = q.shape
        n = batch * heads
        block_m, block_n = ctx.block_m, ctx.block_n
        n_blocks = (seq + block_m - 1) // block_m
        dev = q.device

        # Native dtypes (bf16 under autocast): tiles are loaded as-is and
        # converted in registers, so no fp32 copies of q/k/v/do are made.
        qf = q.reshape(n, seq, head_dim).contiguous()
        kf = k.reshape(n, seq, head_dim).contiguous()
        vf = v.reshape(n, seq, head_dim).contiguous()
        dof = dout.reshape(n, seq, head_dim).contiguous()
        of = out.reshape(n, seq, head_dim)
        dbar = (dof.float() * of).sum(dim=-1).contiguous()

        # Backward: BA (reverse-order key-parallel pass, 32-row tiles)
        # -> BC (dk Jacobian correction) -> dq epilogue.
        dq = torch.zeros(n, seq, head_dim, device=dev, dtype=torch.float32)   # dq_main (atomics)
        ghat = torch.zeros(n, seq, device=dev, dtype=torch.float32)
        dk = torch.empty(n, seq, head_dim, device=dev, dtype=torch.float32)   # dk_main
        dk_out = torch.empty(n, seq, head_dim, device=dev, dtype=k.dtype)
        dv_out = torch.empty(n, seq, head_dim, device=dev, dtype=v.dtype)
        dq_out = torch.empty(n, seq, head_dim, device=dev, dtype=q.dtype)

        scale = 1.0 / math.sqrt(head_dim)
        lowp, prec = _resolve_precision(ctx.dot_precision)
        common = dict(BLOCK_M=block_m, BLOCK_N=block_n, DOT_PREC=prec, LOWP=lowp,
                      num_warps=ctx.bwd_num_warps, num_stages=ctx.bwd_num_stages)
        s_strides = (carry.stride(0), carry.stride(1), carry.stride(2))
        q_strides = (qf.stride(0), qf.stride(1), qf.stride(2))
        kb_bn = ctx.b3_block_n or block_n
        n_kblocks = (seq + kb_bn - 1) // kb_bn
        bwd_block_m = ctx.bwd_block_m
        nb_bwd = (seq + bwd_block_m - 1) // bwd_block_m
        # BA pipelines its per-row loads; fp32 operand tiles keep one stage, since
        # two can exceed shared memory at larger head dims.
        ba_common = dict(common, BLOCK_M=bwd_block_m, BLOCK_N=kb_bn, num_warps=ctx.b4_num_warps or 4,
                         num_stages=2 if lowp else ctx.bwd_num_stages)
        b3_common = dict(common, BLOCK_N=kb_bn, num_warps=ctx.b3_num_warps or ctx.bwd_num_warps)
        _dilution_ba[(n, n_kblocks)](
            qf, kf, vf, dof, dk, dq, dv_out, ghat, dbar, ssum, rmax, rden, carry,
            *q_strides, *s_strides, seq, nb_bwd, head_dim, scale, **ba_common)
        _dilution_bc[(n, n_kblocks)](
            qf, kf, dk, dk_out, ghat, rmax, rden,
            *q_strides, seq, n_blocks, head_dim, scale, **b3_common)
        _dilution_dq_epilogue[(n, (seq + 127) // 128)](
            dq, ghat, pk, dq_out, seq, HEAD_DIM=head_dim, BLOCK=128, num_warps=4)

        shape4 = (batch, heads, seq, head_dim)
        return (dq_out.reshape(shape4), dk_out.reshape(shape4), dv_out.reshape(shape4),
                None, None, None, None, None, None, None, None, None)


def dilution_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dot_precision: str = "bf16",
    block_m: int = 64,
    block_n: int = 64,
    num_warps: int = 4,
    bwd_num_warps: int = 4,
    bwd_num_stages: int = 1,
    b3_block_n: int | None = None,
    b3_num_warps: int | None = None,
    b4_num_warps: int | None = 4,
) -> torch.Tensor:
    """Differentiable fused dilution attention (forward + backward kernels).

    dot_precision "bf16" (default) uses bf16 tensor-core operands for the
    q.k / p.v products with fp32 accumulation and fp32 cumsum/share math;
    "tf32"/"ieee" keep fp32 operands (parity tests use "ieee").
    """

    return _DilutionAttentionFn.apply(
        q, k, v, dot_precision, block_m, block_n,
        num_warps, bwd_num_warps, bwd_num_stages, b3_block_n, b3_num_warps, b4_num_warps,
    )


# --- cached decode: one fused program per (batch, head) row -------------------
if HAS_TRITON:

    @triton.jit
    def _dilution_decode(
        Q, K, V, AFF, COMMITTED, OUT,
        stride_kn, stride_kt, stride_kd, stride_an,
        T, SCALE: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """One new query row against T cached keys, the whole operator in one launch.

        Two online-softmax sweeps over the key blocks: (1) affinity row stats,
        written to an O(T) fp32 scratch (AFF) so K is streamed once; (2) p from
        the stats, share = p / (committed + p + eps), row-normalized output, and
        the accumulator advanced in place (committed += p). V is streamed once.
        Everything is fp32; the row is length 1 so there is no tensor-core
        product, only broadcast multiply-reduce.
        """
        pid = tl.program_id(0)
        d_offs = tl.arange(0, HEAD_DIM)
        q = tl.load(Q + pid * HEAD_DIM + d_offs).to(tl.float32) * (SCALE * LOG2E)
        k_base = K + pid * stride_kn
        v_base = V + pid * stride_kn
        c_base = COMMITTED + pid * stride_an
        f_base = AFF + pid * stride_an
        neg_inf = float("-inf")

        # Sweep 1: affinity max / sum (log2 domain).
        rmax = tl.full((1,), neg_inf, tl.float32)
        rden = tl.zeros((1,), tl.float32)
        for start in range(0, T, BLOCK_N):
            n_offs = start + tl.arange(0, BLOCK_N)
            n_mask = n_offs < T
            k_tile = tl.load(k_base + n_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd,
                             mask=n_mask[:, None], other=0.0).to(tl.float32)
            aff = tl.where(n_mask, tl.sum(k_tile * q[None, :], axis=1), neg_inf)
            tl.store(f_base + n_offs, aff, mask=n_mask)
            m_new = tl.maximum(rmax, tl.max(aff, axis=0))
            rden = rden * tl.exp2(rmax - m_new) + tl.sum(tl.exp2(aff - m_new), axis=0)
            rmax = m_new
        rinv = 1.0 / rden

        # Sweep 2: shares, output, accumulator update.
        acc = tl.zeros((HEAD_DIM,), tl.float32)
        ssum = tl.zeros((1,), tl.float32)
        for start in range(0, T, BLOCK_N):
            n_offs = start + tl.arange(0, BLOCK_N)
            n_mask = n_offs < T
            aff = tl.load(f_base + n_offs, mask=n_mask, other=neg_inf)
            p = tl.exp2(aff - rmax) * rinv
            committed = tl.load(c_base + n_offs, mask=n_mask, other=0.0)
            share = tl.where(n_mask, p / (committed + p + 1e-9), 0.0)
            ssum += tl.sum(share, axis=0)
            v_tile = tl.load(v_base + n_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd,
                             mask=n_mask[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(share[:, None] * v_tile, axis=0)
            tl.store(c_base + n_offs, committed + p, mask=n_mask)
        out = acc / (ssum + 1e-9)
        tl.store(OUT + pid * HEAD_DIM + d_offs, out)


def dilution_decode_step(
    q1: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    committed: torch.Tensor,
    length: int,
    block_n: int = 512,
    num_warps: int = 8,
) -> torch.Tensor:
    """Fused decode step. Returns the new row's output, (B, H, 1, D) in q1.dtype.

    q1: (B, H, 1, D). k_cache / v_cache: contiguous (B, H, capacity, D) buffers
    whose first `length` positions are valid, the new token already written at
    position length-1. committed: contiguous (B, H, capacity) fp32 accumulator
    (column totals of p over all previous rows; position length-1 must be
    zero). It is advanced in place to include this row.
    """

    if not HAS_TRITON:
        raise RuntimeError("triton is not available")
    batch, heads, one, head_dim = q1.shape
    if one != 1:
        raise ValueError("decode step takes exactly one query row per (batch, head)")
    if head_dim & (head_dim - 1):
        raise ValueError("head_dim must be a power of two")
    capacity = k_cache.shape[2]
    if not 1 <= length <= capacity:
        raise ValueError(f"length {length} outside cache capacity {capacity}")
    n = batch * heads
    qf = q1.reshape(n, head_dim)
    if not qf.is_contiguous():
        qf = qf.contiguous()
    kf = k_cache.view(n, capacity, head_dim)
    vf = v_cache.view(n, capacity, head_dim)
    cf = committed.view(n, capacity)
    out = torch.empty(n, head_dim, device=q1.device, dtype=q1.dtype)  # fp32 math, native store
    aff = torch.empty(n, capacity, device=q1.device, dtype=torch.float32)

    _dilution_decode[(n,)](
        qf, kf, vf, aff, cf, out,
        kf.stride(0), kf.stride(1), kf.stride(2), cf.stride(0),
        length, SCALE=1.0 / math.sqrt(head_dim),
        HEAD_DIM=head_dim, BLOCK_N=block_n, num_warps=num_warps,
    )
    return out.view(batch, heads, 1, head_dim)
