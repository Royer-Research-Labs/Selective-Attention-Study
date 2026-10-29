"""Dilution attention with selectable bid function and share step (derived from
kernels.py by the scratch generator make_kernels_bid.py; do not edit by hand).

BID 0: p = causal_softmax(a)                       (kernels.py, the production operator)
BID 1: p = relu(a) / rowsum                        ("percent of share" ablation)
BID 2: p = relu(a)^2 / rowsum
BID 3: p = softplus(a) / rowsum
BID 4: p = (a - rmin_i) / rowsum,  rmin_i = min over the causal window, held
       CONSTANT in the backward (jac = 1/R; the argmin is not differentiated;
       the reference detaches rmin to match)
BID 5: p = sigmoid(a): each bid in (0, 1) on its own, NO row sum in the bid
       step. jac = p (1 - p) carries the factor of p that keeps the share
       step's 1/p gradient bounded (as softmax's does; relu/softplus/minshift
       do not). No row statistics (K1's first sweep is skipped) and no ghat
       term in the backward.
BID 6: p = exp(a) / SHIFT: softmax's exponential WITHOUT the row sum. Row sums
       start near 1 (for SHIFT = context length and small affinities) but are
       not enforced, so the dilution scale is explicit rather than tied to a
       per-row normalisation. jac = p. No row statistics, no ghat. Unbounded
       above; SHIFT is what keeps committed mass sane. Overflows fp32 once
       affinities pass ~88 (measured: NaN at step 400), so prefer BID 7.
BID 7: p = exp(a) / (exp(a) + SHIFT) = sigmoid(a - log SHIFT): expshift below
       the cap, saturating at 1. jac = p (1 - p). Computed as 1/(1 + SHIFT e^-a);
       cannot overflow. Elementwise: no row statistics, no ghat.
with a = q.k / sqrt(d). Keys with f = 0 bid exactly zero (a per-element floor
would be amplified by the share step, whose per-key ratio is scale-free: every
rejected key would collect O(1/i) share). A row with NO mass at all
(all-negative affinities under relu, the first token under minshift) falls back
to uniform attention over its causal window, with zero Jacobian. Row statistics
reuse the (rmax, rden) buffers: rmax holds rmin for BID 4 and 0 otherwise; rden
holds the row sum of f (0 for a mass-less row; callers use 1/rden = inf as the
uniform-row flag).

SHARE 1: share = p / (committed + p + eps), row-normalised (the dilution step).
SHARE 2: share = p / (committed + eps) -- EXCLUSIVE history, i.e. the
       Sankaran/Paulus temporal-attention denominator on row-normalised bids.
       The first row needs no special case: committed is 0 there, so the row
       normalisation cancels eps and reproduces their published first step.
SHARE 0: attn = p / rowsum(p), no cumsum -- the control (identity on the
         row-normalised bids; row-normalised sigmoid attention for BID 5). The
         backward drops the committed/suffix chain (dp = ds) and B1's key-sums
         are zero.

Original docstring follows.

Fused Triton dilution attention: reference + 2-pass scan-decomposition kernels.

`dilution_attention_reference` is the authoritative PyTorch spec of the
operator (docs/operator.md): one causal softmax, one exclusive column-cumsum,
one renormalise, no parameters. The Triton path must match it;
tests/test_kernels.py pins parity.

Kernel design (docs/kernels.md): the exclusive column-cumsum is the only
sequential coupling, so it is decomposed into per-row-block partial key-sums
(computed in parallel) plus one tiny host-side scan over blocks:

    K1 (grid N x nB): causal-softmax row stats + per-block key-sums S
        carry = exclusive block-cumsum(S)               # torch, (N, nB, T)
    K2 (grid N x nB): rebuild p, committed = carry + in-block exclusive cumsum,
        share = p / (committed + p + eps) -> row-normalize -> @ V

Every kernel is fully parallel over (batch*heads) x row-blocks; nothing
(T x T)-shaped ever reaches DRAM. fp32 accumulators throughout; dot precision
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


BID_TYPES = ("softmax", "relu", "relu2", "softplus", "minshift", "sigmoid", "expshift", "sigshift")
# A row with no affinity mass falls back to uniform attention over its window.


def _bid_code(bid: str) -> int:
    if bid not in BID_TYPES:
        raise ValueError(f"bid must be one of {BID_TYPES}")
    return BID_TYPES.index(bid)


def bids_from_affinity(aff: torch.Tensor, causal: torch.Tensor, bid: str, shift: float = 1.0) -> torch.Tensor:
    """p from scaled affinities (fp32) under the selected bid function.
    `shift` is the expshift divisor (the context length in the model)."""
    if bid == "softmax":
        return aff.masked_fill(~causal, torch.finfo(torch.float32).min).softmax(dim=-1)
    if bid == "sigmoid":
        return torch.sigmoid(aff).masked_fill(~causal, 0.0)
    if bid == "expshift":
        return (torch.exp(aff) / float(shift)).masked_fill(~causal, 0.0)
    if bid == "sigshift":
        return torch.sigmoid(aff - math.log(float(shift))).masked_fill(~causal, 0.0)
    if bid == "relu":
        f = aff.clamp_min(0.0)
    elif bid == "relu2":
        f = aff.clamp_min(0.0) ** 2
    elif bid == "softplus":
        f = torch.nn.functional.softplus(aff)
    elif bid == "minshift":
        rmin = aff.masked_fill(~causal, float("inf")).amin(dim=-1, keepdim=True).detach()
        f = aff - rmin
    else:
        raise ValueError(f"bid must be one of {BID_TYPES}")
    f = f.masked_fill(~causal, 0.0)
    rsum = f.sum(dim=-1, keepdim=True)
    nwin = causal.sum(dim=-1, keepdim=True).to(f.dtype)
    uniform = causal.to(f.dtype) / nwin
    return torch.where(rsum > 0, f / rsum.clamp_min(1e-30), uniform)


def _share_code(share_cumsum):
    """True/1 -> inclusive dilution, "exclusive" -> temporal-attention history, False/0 -> none."""
    if share_cumsum == "exclusive":
        return 2
    return int(bool(share_cumsum))


def dilution_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bid: str = "softmax",
    share_cumsum: bool = True,
    shift: float = 1.0,
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
    p = bids_from_affinity(aff, mask, bid, shift)
    if not share_cumsum:
        return (p / p.sum(dim=-1, keepdim=True)) @ v.float()
    committed = p.cumsum(dim=-2) - p
    if share_cumsum == "exclusive":
        share = p / (committed + 1e-9)
    else:
        share = p / (committed + p + 1e-9)
    attn = share / (share.sum(dim=-1, keepdim=True) + 1e-9)
    return attn @ v.float()


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
    def _p_jac_from_k(q_tile, k_tile, n_offs, m_offs, T, rmax, rinv,
                      SCALE: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr, BID: tl.constexpr,
                      SHIFT: tl.constexpr):
        """p for one (rows x keys) tile and jac, the diagonal factor of dp/da, so that
        da = jac * (dp - ghat). softmax: p. relu: 1[a>0]/R. relu2: 2 relu(a)/R.
        softplus: sigmoid(a)/R. minshift: 1/R (rmin constant)."""
        aff = _dot(q_tile, tl.trans(k_tile), LOWP, DOT_PREC) * SCALE
        causal = (n_offs[None, :] <= m_offs[:, None]) & (n_offs < T)[None, :] & (m_offs < T)[:, None]
        if BID == 0:
            aff_m = tl.where(causal, aff * LOG2E, float("-inf"))
            p = tl.exp2(aff_m - rmax[:, None]) * rinv[:, None]
            jac = p
        elif BID == 5:
            p = 1.0 / (1.0 + tl.exp(-aff))
            jac = p * (1.0 - p)
        elif BID == 6:
            p = tl.exp(aff) * rinv[:, None]   # rden buffer holds SHIFT, so rinv = 1/SHIFT
            jac = p
        elif BID == 7:
            p = 1.0 / (1.0 + SHIFT * tl.exp(-aff))
            jac = p * (1.0 - p)
        else:
            # rinv = 1/rden is +inf on a mass-less row: uniform over the window, zero Jacobian
            empty = (rinv > 1e30)[:, None] & causal
            r = tl.where(rinv > 1e30, 0.0, rinv)[:, None] + 0.0 * aff
            unif = 1.0 / (m_offs + 1).to(tl.float32)[:, None] + 0.0 * aff
            if BID == 1:
                f = tl.maximum(aff, 0.0)
                p = f * r
                jac = tl.where(aff > 0.0, r, 0.0)
            elif BID == 2:
                f = tl.maximum(aff, 0.0)
                p = f * f * r
                jac = 2.0 * f * r
            elif BID == 3:
                f = tl.maximum(aff, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(aff)))  # stable softplus
                p = f * r
                jac = r / (1.0 + tl.exp(-aff))                                   # sigmoid(a) / R
            else:
                p = (aff - rmax[:, None]) * r                                    # rmax buffer holds rmin
                jac = r
            p = tl.where(empty, unif, p)
        return causal, tl.where(causal, p, 0.0), tl.where(causal, jac, 0.0)

    @triton.jit
    def _p_from_k(q_tile, k_tile, n_offs, m_offs, T, rmax, rinv,
                  SCALE: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr, BID: tl.constexpr,
                  SHIFT: tl.constexpr):
        """p for one (rows x keys) tile from an already-loaded key tile and row stats."""
        causal, p, _ = _p_jac_from_k(q_tile, k_tile, n_offs, m_offs, T, rmax, rinv, SCALE, DOT_PREC, LOWP, BID, SHIFT)
        return causal, p

    @triton.jit
    def _p_tile(q_tile, k_base, n_offs, d_offs, stride_qt, stride_qd,
                m_offs, T, rmax, rinv, SCALE: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr, BID: tl.constexpr,
                SHIFT: tl.constexpr):
        """Load one key tile and recompute p from stored row stats."""
        k_tile = tl.load(
            k_base + n_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
            mask=(n_offs < T)[:, None], other=0.0,
        )
        return _p_from_k(q_tile, k_tile, n_offs, m_offs, T, rmax, rinv, SCALE, DOT_PREC, LOWP, BID, SHIFT)

    @triton.jit
    def _share_from_k(q_tile, k_tile, n_offs, m_offs, T, rmax, rinv, c_base, stride_st,
                      SCALE: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr,
                      BID: tl.constexpr, SHARE: tl.constexpr, SHIFT: tl.constexpr):
        """p, its exclusive column-cumsum, 1/(committed+p+eps) and jac from a loaded key
        tile. With SHARE == 0 the share step is the identity: committed = 0, inv = 1."""
        causal, p, jac = _p_jac_from_k(q_tile, k_tile, n_offs, m_offs, T, rmax, rinv, SCALE, DOT_PREC, LOWP, BID, SHIFT)
        if SHARE:
            carry = tl.load(c_base + n_offs * stride_st, mask=n_offs < T, other=0.0)
            committed = carry[None, :] + tl.cumsum(p, axis=0) - p
            if SHARE == 2:
                inv = 1.0 / (committed + 1e-9)          # exclusive history
            else:
                inv = 1.0 / (committed + p + 1e-9)      # inclusive (dilution)
        else:
            committed = 0.0 * p
            inv = 1.0 + 0.0 * p
        return causal, p, committed, inv, jac

    @triton.jit
    def _share_tile(q_tile, k_base, n_offs, d_offs, stride_qt, stride_qd,
                    m_offs, T, rmax, rinv, c_base, stride_st,
                    SCALE: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr,
                    BID: tl.constexpr, SHARE: tl.constexpr, SHIFT: tl.constexpr):
        """Load one key tile, then _share_from_k."""
        k_tile = tl.load(
            k_base + n_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
            mask=(n_offs < T)[:, None], other=0.0,
        )
        return _share_from_k(q_tile, k_tile, n_offs, m_offs, T, rmax, rinv, c_base, stride_st,
                             SCALE, DOT_PREC, LOWP, BID, SHARE, SHIFT)

    @triton.jit
    def _dilution_k1(
        Q, K, RMAX, RDEN, S,
        stride_qn, stride_qt, stride_qd,
        stride_sn, stride_sb, stride_st,
        T, HEAD_DIM: tl.constexpr, SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr,
        BID: tl.constexpr, SHARE: tl.constexpr, SHIFT: tl.constexpr,
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
        if BID == 4:
            rmax = tl.full((BLOCK_M,), float("inf"), tl.float32)   # running row MIN for minshift
        else:
            rmax = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        rden = tl.zeros((BLOCK_M,), tl.float32)
        if BID == 5 or BID == 6 or BID == 7:
            n_stats = 0  # elementwise bids: no row statistics
        else:
            n_stats = m_start + BLOCK_M
        for n_start in range(0, n_stats, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offs < T
            k_tile = tl.load(
                k_base + n_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd,
                mask=n_mask[:, None], other=0.0,
            )
            aff = _dot(q_tile, tl.trans(k_tile), LOWP, DOT_PREC) * SCALE
            causal = (n_offs[None, :] <= m_offs[:, None]) & n_mask[None, :] & m_mask[:, None]
            if BID == 0:
                aff = tl.where(causal, aff * LOG2E, float("-inf"))
                tile_max = tl.max(aff, axis=1)
                new_max = tl.maximum(rmax, tile_max)
                rden = rden * tl.exp2(rmax - new_max) + tl.sum(tl.exp2(aff - new_max[:, None]), axis=1)
                rmax = new_max
            elif BID == 4:
                rmax = tl.minimum(rmax, tl.min(tl.where(causal, aff, float("inf")), axis=1))
                rden += tl.sum(tl.where(causal, aff, 0.0), axis=1)
            elif BID == 3:
                a = tl.where(causal, aff, 0.0)
                f = tl.maximum(a, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(a)))
                rden += tl.sum(tl.where(causal, f, 0.0), axis=1)
            else:
                f = tl.maximum(tl.where(causal, aff, 0.0), 0.0)
                if BID == 2:
                    f = f * f
                rden += tl.sum(f, axis=1)
        if BID == 4:
            nwin = (m_offs + 1).to(tl.float32)
            rden = tl.maximum(rden - nwin * rmax, 0.0)  # sum(a - rmin); rmax buffer := rmin
        elif BID == 5 or BID == 7:
            rmax = tl.zeros((BLOCK_M,), tl.float32)
            rden = rden + 1.0  # unused by sigmoid/sigshift; keep 1/rden finite
        elif BID == 6:
            rmax = tl.zeros((BLOCK_M,), tl.float32)
            rden = rden + SHIFT  # expshift divisor, read back as rinv = 1/SHIFT
        elif BID != 0:
            rmax = tl.zeros((BLOCK_M,), tl.float32)
        tl.store(RMAX + pid_n * T + m_offs, rmax, mask=m_mask)
        tl.store(RDEN + pid_n * T + m_offs, rden, mask=m_mask)
        rinv = 1.0 / rden

        # Sweep 2: per-block key-sums of p (this block's rows only).
        s_base = S + pid_n * stride_sn + pid_m * stride_sb
        for n_start in range(0, m_start + BLOCK_M, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            _, p = _p_tile(q_tile, k_base, n_offs, d_offs, stride_qt, stride_qd,
                           m_offs, T, rmax, rinv, SCALE, DOT_PREC, LOWP, BID, SHIFT)
            tl.store(s_base + n_offs * stride_st, tl.sum(p, axis=0), mask=n_offs < T)

    @triton.jit
    def _dilution_k2(
        Q, K, V, OUT, SSUM, RMAX, RDEN, CARRY,
        stride_qn, stride_qt, stride_qd,
        stride_on, stride_ot, stride_od,
        stride_sn, stride_sb, stride_st,
        T, HEAD_DIM: tl.constexpr, SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr,
        BID: tl.constexpr, SHARE: tl.constexpr, SHIFT: tl.constexpr,
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
        c_base = CARRY + pid_n * stride_sn + pid_m * stride_sb

        acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
        share_sum = tl.zeros((BLOCK_M,), tl.float32)
        for n_start in range(0, m_start + BLOCK_M, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offs < T
            causal, p, committed, inv, _ = _share_tile(
                q_tile, k_base, n_offs, d_offs, stride_qt, stride_qd, m_offs, T,
                rmax, rinv, c_base, stride_st, SCALE, DOT_PREC, LOWP, BID, SHARE, SHIFT)
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
    bid: str = "softmax",
    share_cumsum: bool = True,
    shift: float = 1.0,
):
    """2-pass fused forward: parallel over (batch*heads) x row-blocks. Returns fp32.

    With return_aux=True also returns the O(T) statistics the backward kernels
    need (row stats, block carry, share row-sums) and the inclusive column
    totals of p, which are the cached-decode accumulator after this prefix.
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
    out = torch.empty(n, seq, head_dim, device=dev, dtype=torch.float32)
    rmax = torch.empty(n, seq, device=dev, dtype=torch.float32)
    rden = torch.empty(n, seq, device=dev, dtype=torch.float32)
    ssum = torch.empty(n, seq, device=dev, dtype=torch.float32)
    s = torch.zeros(n, n_blocks, seq, device=dev, dtype=torch.float32)

    scale = 1.0 / math.sqrt(head_dim)
    grid = (n, n_blocks)
    lowp, prec = _resolve_precision(dot_precision)
    common = dict(BLOCK_M=block_m, BLOCK_N=block_n, DOT_PREC=prec, LOWP=lowp,
                  BID=_bid_code(bid), SHARE=_share_code(share_cumsum), SHIFT=float(shift),
                  num_warps=num_warps, num_stages=num_stages)

    _dilution_k1[grid](
        qf, kf, rmax, rden, s,
        qf.stride(0), qf.stride(1), qf.stride(2),
        s.stride(0), s.stride(1), s.stride(2),
        seq, head_dim, scale, **common,
    )
    carry = s.cumsum(dim=1) - s  # exclusive over row-blocks
    _dilution_k2[grid](
        qf, kf, vf, out, ssum, rmax, rden, carry,
        qf.stride(0), qf.stride(1), qf.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        s.stride(0), s.stride(1), s.stride(2),
        seq, head_dim, scale, **common,
    )
    result = out.reshape(batch, heads, seq, head_dim)
    if return_aux:
        aux = dict(rmax=rmax, rden=rden, carry=carry, ssum=ssum,
                   block_m=block_m, block_n=block_n, dot_precision=dot_precision,
                   num_warps=num_warps, bid=bid, share_cumsum=share_cumsum, shift=shift,
                   # Inclusive column totals over all rows: the decode
                   # accumulator (committed) after this prefix.
                   committed_total=s.sum(dim=1).reshape(batch, heads, seq))
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
        BID: tl.constexpr, SHARE: tl.constexpr, SHIFT: tl.constexpr,
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
        c_base = CARRY + pid_n * stride_sn + pid_m * stride_sb
        sk_base = SK + pid_n * stride_sn + pid_m * stride_sb

        gpart = tl.zeros((BLOCK_M,), tl.float32)
        for n_start in range(0, m_start + BLOCK_M, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            causal, p, committed, inv, _ = _share_tile(
                q_tile, k_base, n_offs, d_offs, stride_qt, stride_qd, m_offs, T,
                rmax, rinv, c_base, stride_st, SCALE, DOT_PREC, LOWP, BID, SHARE, SHIFT)
            ds = _ds_tile(do_tile, v_base, n_offs, d_offs, stride_qt, stride_qd, T,
                          dbar, sinv, causal, DOT_PREC, LOWP)
            inv2 = inv * inv
            if SHARE:
                dk_share = -ds * p * inv2
                if SHARE == 2:
                    dp_direct = ds * inv                       # d/dp of p/(c+eps)
                else:
                    dp_direct = ds * (committed + 1e-9) * inv2
            else:
                dk_share = 0.0 * ds
                dp_direct = ds
            colsum = tl.sum(dk_share, axis=0)
            tl.store(sk_base + n_offs * stride_st, colsum, mask=n_offs < T)
            inblock = colsum[None, :] - tl.cumsum(dk_share, axis=0)  # later rows in this block
            gpart += tl.sum(p * (dp_direct + inblock), axis=1)
        tl.store(GPART + pid_n * T + m_offs, gpart, mask=m_mask)

    @triton.jit
    def _dilution_b2(
        Q, K, RMAX, RDEN, SUFK, GPART, GHAT,
        stride_qn, stride_qt, stride_qd,
        stride_sn, stride_sb, stride_st,
        T, HEAD_DIM: tl.constexpr, SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DOT_PREC: tl.constexpr, LOWP: tl.constexpr,
        BID: tl.constexpr, SHARE: tl.constexpr, SHIFT: tl.constexpr,
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
        sufk_base = SUFK + pid_n * stride_sn + pid_m * stride_sb

        ghat = tl.load(GPART + pid_n * T + m_offs, mask=m_mask, other=0.0)
        for n_start in range(0, m_start + BLOCK_M, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            _, p = _p_tile(q_tile, k_base, n_offs, d_offs, stride_qt, stride_qd,
                           m_offs, T, rmax, rinv, SCALE, DOT_PREC, LOWP, BID, SHIFT)
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
        BID: tl.constexpr, SHARE: tl.constexpr, SHIFT: tl.constexpr,
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
            c_base = CARRY + pid_n * stride_sn + rb * stride_sb
            causal, p, committed, inv, _ = _share_from_k(
                q_tile, k_blk, j_offs, m_offs, T, rmax, 1.0 / rden, c_base, stride_st,
                SCALE, DOT_PREC, LOWP, BID, SHARE, SHIFT)
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
        BID: tl.constexpr, SHARE: tl.constexpr, SHIFT: tl.constexpr,
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
            c_base = CARRY + pid_n * stride_sn + rb * stride_sb
            sufk_base = SUFK + pid_n * stride_sn + rb * stride_sb

            causal, p, committed, inv, jac = _share_from_k(
                q_tile, k_blk, j_offs, m_offs, T, rmax, 1.0 / rden, c_base, stride_st,
                SCALE, DOT_PREC, LOWP, BID, SHARE, SHIFT)
            ds = _ds_from_v(do_tile, v_blk, dbar, 1.0 / (ssum + 1e-9), causal, DOT_PREC, LOWP)
            inv2 = inv * inv
            if SHARE:
                dk_share = -ds * p * inv2
                if SHARE == 2:
                    dp_direct = ds * inv                       # d/dp of p/(c+eps)
                else:
                    dp_direct = ds * (committed + 1e-9) * inv2
            else:
                dk_share = 0.0 * ds
                dp_direct = ds
            sufk = tl.load(sufk_base + j_offs * stride_st, mask=j_mask, other=0.0)
            suffix = sufk[None, :] + tl.sum(dk_share, axis=0)[None, :] - tl.cumsum(dk_share, axis=0)
            if BID == 5 or BID == 6 or BID == 7:
                da = tl.where(causal, jac * (dp_direct + suffix), 0.0)  # no row normalisation of p
            else:
                da = tl.where(causal, jac * (dp_direct + suffix - ghat[:, None]), 0.0)
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

class _DilutionAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, dot_precision, block_m, block_n,
                num_warps, bwd_num_warps, bwd_num_stages, b3_block_n, b3_num_warps, b4_num_warps,
                bid, share_cumsum, shift):
        out, aux = dilution_attention_forward(
            q, k, v,
            dot_precision=dot_precision, block_m=block_m, block_n=block_n,
            return_aux=True, num_warps=num_warps, bid=bid, share_cumsum=share_cumsum, shift=shift,
        )
        ctx.bid = bid
        ctx.share_cumsum = share_cumsum
        ctx.shift = shift
        ctx.bwd_num_warps = bwd_num_warps
        ctx.bwd_num_stages = bwd_num_stages
        ctx.b3_block_n = b3_block_n
        ctx.b3_num_warps = b3_num_warps
        ctx.b4_num_warps = b4_num_warps
        ctx.save_for_backward(q, k, v, out, aux["rmax"], aux["rden"], aux["carry"], aux["ssum"])
        ctx.dot_precision = dot_precision
        ctx.block_m = block_m
        ctx.block_n = block_n
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, rmax, rden, carry, ssum = ctx.saved_tensors
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

        sk = torch.zeros(n, n_blocks, seq, device=dev, dtype=torch.float32)
        gpart = torch.empty(n, seq, device=dev, dtype=torch.float32)
        ghat = torch.empty(n, seq, device=dev, dtype=torch.float32)
        dq = torch.zeros(n, seq, head_dim, device=dev, dtype=torch.float32)
        dk = torch.empty(n, seq, head_dim, device=dev, dtype=torch.float32)
        dv = torch.empty(n, seq, head_dim, device=dev, dtype=torch.float32)

        scale = 1.0 / math.sqrt(head_dim)
        grid = (n, n_blocks)
        # Backward tiles keep many tensors live; pipelining depth is tunable
        # but limited by shared memory (1 was required with fp32 tiles).
        lowp, prec = _resolve_precision(ctx.dot_precision)
        common = dict(BLOCK_M=block_m, BLOCK_N=block_n, DOT_PREC=prec, LOWP=lowp,
                      BID=_bid_code(ctx.bid), SHARE=_share_code(ctx.share_cumsum), SHIFT=float(ctx.shift),
                      num_warps=ctx.bwd_num_warps, num_stages=ctx.bwd_num_stages)
        s_strides = (sk.stride(0), sk.stride(1), sk.stride(2))
        q_strides = (qf.stride(0), qf.stride(1), qf.stride(2))

        _dilution_b1[grid](
            qf, kf, vf, dof, dbar, ssum, rmax, rden, carry, sk, gpart,
            *q_strides, *s_strides, seq, head_dim, scale, **common)
        sufk = sk.flip(1).cumsum(dim=1).flip(1) - sk  # reverse-exclusive over blocks
        _dilution_b2[grid](
            qf, kf, rmax, rden, sufk, gpart, ghat,
            *q_strides, *s_strides, seq, head_dim, scale, **common)
        kb_bn = ctx.b3_block_n or block_n
        n_kblocks = (seq + kb_bn - 1) // kb_bn
        b3_common = dict(common, BLOCK_N=kb_bn, num_warps=ctx.b3_num_warps or ctx.bwd_num_warps)
        # 8 warps wins for B4 in isolation (warm L2) but loses end to end: the
        # doubled register footprint halves occupancy once the preceding kernels
        # have evicted the cache. 4 warps measured 9.45 vs 11.40 ms fwd+bwd.
        b4_common = dict(common, BLOCK_N=kb_bn, num_warps=ctx.b4_num_warps or 4)
        _dilution_b3[(n, n_kblocks)](
            qf, kf, dof, dv, ssum, rmax, rden, carry,
            *q_strides, *s_strides, seq, n_blocks, head_dim, scale, **b3_common)
        _dilution_b4[(n, n_kblocks)](
            qf, kf, vf, dof, dk, dq, dbar, ssum, rmax, rden, carry, sufk, ghat,
            *q_strides, *s_strides, seq, n_blocks, head_dim, scale, **b4_common)

        shape4 = (batch, heads, seq, head_dim)
        return (dq.reshape(shape4).to(q.dtype), dk.reshape(shape4).to(k.dtype),
                dv.reshape(shape4).to(v.dtype), None, None, None, None, None, None, None, None, None,
                None, None, None)


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
    bid: str = "softmax",
    share_cumsum: bool = True,
    shift: float = 1.0,
) -> torch.Tensor:
    """Differentiable fused dilution attention (forward + backward kernels).

    dot_precision "bf16" (default) uses bf16 tensor-core operands for the
    q.k / p.v products with fp32 accumulation and fp32 cumsum/share math;
    "tf32"/"ieee" keep fp32 operands (parity tests use "ieee").
    """

    return _DilutionAttentionFn.apply(
        q, k, v, dot_precision, block_m, block_n,
        num_warps, bwd_num_warps, bwd_num_stages, b3_block_n, b3_num_warps, b4_num_warps,
        bid, share_cumsum, shift,
    )


# --- cached decode: one fused program per (batch, head) row -------------------
if HAS_TRITON:

    @triton.jit
    def _dilution_decode(
        Q, K, V, AFF, COMMITTED, OUT,
        stride_kn, stride_kt, stride_kd, stride_an,
        T, SCALE: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
        BID: tl.constexpr, SHARE: tl.constexpr, SHIFT: tl.constexpr,
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
        q = tl.load(Q + pid * HEAD_DIM + d_offs).to(tl.float32) * SCALE
        if BID == 0:
            q = q * LOG2E
        k_base = K + pid * stride_kn
        v_base = V + pid * stride_kn
        c_base = COMMITTED + pid * stride_an
        f_base = AFF + pid * stride_an
        neg_inf = float("-inf")

        # Sweep 1: affinity max / sum (log2 domain).
        if BID == 4:
            rmax = tl.full((1,), float("inf"), tl.float32)
        else:
            rmax = tl.full((1,), neg_inf, tl.float32)
        rden = tl.zeros((1,), tl.float32)
        for start in range(0, T, BLOCK_N):
            n_offs = start + tl.arange(0, BLOCK_N)
            n_mask = n_offs < T
            k_tile = tl.load(k_base + n_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd,
                             mask=n_mask[:, None], other=0.0).to(tl.float32)
            aff = tl.where(n_mask, tl.sum(k_tile * q[None, :], axis=1), neg_inf)
            tl.store(f_base + n_offs, aff, mask=n_mask)
            if BID == 5 or BID == 6 or BID == 7:
                rden = rden + 0.0
            elif BID == 0:
                m_new = tl.maximum(rmax, tl.max(aff, axis=0))
                rden = rden * tl.exp2(rmax - m_new) + tl.sum(tl.exp2(aff - m_new), axis=0)
                rmax = m_new
            elif BID == 4:
                rmax = tl.minimum(rmax, tl.min(tl.where(n_mask, aff, float("inf")), axis=0))
                rden += tl.sum(tl.where(n_mask, aff, 0.0), axis=0)
            elif BID == 3:
                a = tl.where(n_mask, aff, 0.0)
                rden += tl.sum(tl.where(n_mask, tl.maximum(a, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(a))), 0.0), axis=0)
            else:
                f = tl.maximum(tl.where(n_mask, aff, 0.0), 0.0)
                if BID == 2:
                    f = f * f
                rden += tl.sum(f, axis=0)
        if BID == 4:
            rden = tl.maximum(rden - T * rmax, 0.0)
        elif BID == 5 or BID == 7:
            rden = rden + 1.0
        elif BID == 6:
            rden = rden + SHIFT
        rinv = 1.0 / rden  # +inf on a mass-less row -> uniform below

        # Sweep 2: shares, output, accumulator update.
        acc = tl.zeros((HEAD_DIM,), tl.float32)
        ssum = tl.zeros((1,), tl.float32)
        for start in range(0, T, BLOCK_N):
            n_offs = start + tl.arange(0, BLOCK_N)
            n_mask = n_offs < T
            aff = tl.load(f_base + n_offs, mask=n_mask, other=neg_inf)
            if BID == 0:
                p = tl.exp2(aff - rmax) * rinv
            elif BID == 5:
                p = tl.where(n_mask, 1.0 / (1.0 + tl.exp(-aff)), 0.0)
            elif BID == 6:
                p = tl.where(n_mask, tl.exp(aff) * rinv, 0.0)
            elif BID == 7:
                p = tl.where(n_mask, 1.0 / (1.0 + SHIFT * tl.exp(-aff)), 0.0)
            else:
                empty = rinv > 1e30
                r = tl.where(empty, 0.0, rinv)
                if BID == 1:
                    p = tl.maximum(aff, 0.0) * r
                elif BID == 2:
                    f = tl.maximum(aff, 0.0)
                    p = f * f * r
                elif BID == 3:
                    a = tl.where(n_mask, aff, 0.0)
                    p = (tl.maximum(a, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(a)))) * r
                else:
                    p = (aff - rmax) * r
                p = tl.where(empty, 1.0 / T + 0.0 * p, p)
            committed = tl.load(c_base + n_offs, mask=n_mask, other=0.0)
            if SHARE == 2:      # exclusive history: earlier queries' demand only
                share = tl.where(n_mask, p / (committed + 1e-9), 0.0)
            elif SHARE:
                share = tl.where(n_mask, p / (committed + p + 1e-9), 0.0)
            else:
                share = tl.where(n_mask, p, 0.0)
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
    bid: str = "softmax",
    share_cumsum: bool = True,
    shift: float = 1.0,
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
        HEAD_DIM=head_dim, BLOCK_N=block_n, BID=_bid_code(bid), SHARE=_share_code(share_cumsum),
        SHIFT=float(shift), num_warps=num_warps,
    )
    return out.view(batch, heads, 1, head_dim)
