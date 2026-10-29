# How the kernels work

`src/selective_attention/kernels.py`. The operator is defined in the README; `F[i, j]`, the
selection key `j` has received from queries before `i`, is its only coupling between queries. It is a
running sum down the query axis, per key, read from head 0 and shared by every head of the
sequence. Everything below follows from keeping that sum in registers and never writing a
(T x T) tensor to memory.

## Forward (two launches)

    carry    grid B x key-blocks. Head 0 only. For one block of keys, visit the query blocks in
             causal order, 32 queries at a time: compute S = relu(L_0) for the tile, store the
             running per-key sum as it stands on entry to the tile, then add the tile's column
             sums. Output: CARRY (B, T / 32, T) fp32, and the totals over all T queries.
    forward  grid B*H/4 x query-blocks. For one block of 32 queries of four heads (two when the
             head count is not a multiple of four; one head and 64 queries when it is odd), sweep
             the key tiles
             with an online softmax (running maximum, denominator and the exp(.) @ v accumulator,
             as in flash attention). For each tile, F is CARRY at the tile's first query plus the
             exclusive cumulative sum of S inside the tile, so the tile's logits are
             L_h - F. Output: the attention output and the row statistics (max, denominator).

The forward pass makes three tensor-core products per tile (`q_h k_h`, `q_0 k_0` for the selection,
`exp(.) v`) where plain softmax attention makes two. The carry launch is one head's worth of one
product.

CARRY is the only scratch. It belongs to the sequence, not to a head: (T / 32) x T fp32, i.e.
T^2 / 8 bytes per sequence. That is 32 MiB at T = 16,384, 2 GiB at 131,072 and 32 GiB at 524,288,
the longest sequence evaluated here (README, "Long documents"). Its quadratic size, not time, is
what limits the sequence length on one GPU.

## Backward (one launch)

Write `z = L_h - F` for the adjusted logit. The softmax and the value product give, as in ordinary
attention,

    dz_h[i, j] = A_h[i, j] (do_h[i] . v_h[j]  -  do_h[i] . out_h[i])
    dv_h[j] = sum_i A_h[i, j] do_h[i]      dk_h[j] = s sum_i dz_h[i, j] q_h[i]      dq_h[i] = s sum_j dz_h[i, j] k_h[j]

with `s = 1 / sqrt(d)`. The history term adds one path. `F[i, j]` enters every head's logit with a
minus sign, and `S[r, j]` enters `F[i, j]` for every later query `i > r`, so

    dS[r, j] = - sum over heads h, sum over i > r of dz_h[i, j]
    dL_0[r, j] += gate[r, j] dS[r, j]          gate = (L_0 > 0) and E <= j < r   (E = exempt, 1: <BOS>)

and that extra gradient on head 0's logits reaches `q_0` and `k_0` the usual way.

    backward grid B*H x key-blocks (B*H/2 with the head pairs below). For one block of keys of one head, visit the query blocks LAST
             TO FIRST, 32 queries at a time. Rebuild A from the saved row statistics and F from
             CARRY; accumulate dv and dk in registers and dq with relaxed fp32 atomics. Keep a
             running per-key suffix sum of dz over the query blocks already visited, so
             U[r, j] = sum over i > r of dz_h[i, j] is exact for every query in the tile. This
             head's share of head 0's extra gradient is g = -gate * U:
                 dk_0[j] += s sum_r g[r, j] q_0[r]      (registers, one atomic add at the end)
                 dq_0[r] += s sum_j g[r, j] k_0[j]      (relaxed fp32 atomics)

Two facts make it a single launch. Visiting the queries in reverse turns the sum over later queries
into a running sum. And the extra gradient is linear in `dz_h`, so each head's program can add its
own share without ever seeing another head's tile; the sum over heads is the atomic accumulation.
Nothing is detached and nothing is approximated: `tests/test_selective_kernels.py` compares all
three gradients with autograd through the reference.

The backward makes eight products per tile (`q_h k_h`, `q_0 k_0`, `do v`, and one each for dv, dk,
dq, dk_0, dq_0) where flash attention's backward makes five.

## Head groups (version 2, 2026-10-16)

The selection tile, F and the two head-0 gradient products are the same for every head of a
sequence, and the gradient sent to head 0's logits is linear in each head's dz. So one program can
serve several heads: it builds the selection and F once, runs the ordinary attention backward for
each of its heads, sums their dz tiles, and makes the head-0 products once for the sum.

    forward   four heads per program (batched (4, 32, 32) products, one stage)
    backward  two heads per program (_selective_backward_pair, ordinary 2D products):
              6.5 products per head instead of 8

Per layer at T = 16K, B = 2, H = 12, D = 64 (`results/bench/layer_fwd_bwd_v1.json`, `_v2.json`;
`scripts/bench_kernels.py`):

| | forward | backward | total |
|---|---|---|---|
| version 1 (one head per program) | 7.4 ms | 26.0 ms | 33.4 ms |
| version 2 (head groups) | 6.5 ms | 21.2 ms | 27.7 ms |
| softmax (flash, for scale) | 2.8 ms | 6.7 ms | 9.4 ms |
| Dilution-Attention kernel | 10.6 ms | 19.9 ms | 30.5 ms |

The remaining numbers in this section are development timings (not saved under `results/`).
Two things did not pay, and were removed from the code. Batched (G, M, N) products in the backward
were 2-3x slower than one head per program, and three heads per program in 2D spills registers
(27-39 ms). Also measured and not used: computing the in-tile running sums as tensor-core products with a triangular matrix
(3xTF32 for fp32 accuracy; 2 ms slower than `tl.cumsum`), and giving every head its own buffer for
head 0's query gradient (no faster; contention was not the cost).

Where the backward's time goes, from timing-only copies with one part removed at a time (version 1,
21.0 ms of kernel time): the head-0 gradient products and their atomic 7.3 ms, the query-gradient
atomics 3.9 ms, the two in-tile running sums 3.7 ms, the selection product 2.6 ms, and 6.5 ms for
the work every attention backward does. Head groups cut the first, third and fourth in proportion
to the group size.

`group=1, bwd_group=1` select the one-head-per-program kernels, the computation of version 1
(`kernels_v1.py`, kept frozen because it trained the first halves of two 16K seeds). The versions
compute the same function and differ only in the order of floating-point operations;
`test_v2_matches_the_v1_kernels_in_bf16` checks this at the training precision.

## Precision

Tiles are loaded in their native dtype. Under autocast the products run on bf16 tensor cores with
fp32 accumulation (`dot_precision="bf16"`); fp32 inputs use TF32 (`"tf32"`), and `"ieee"` is the
exact fp32 path the parity tests use; `dot_precision=None` (the default) picks bf16 for bf16 inputs
and TF32 otherwise. At head dimension 128 the backward runs `"tf32"` as `"ieee"`: TF32 operand
layouts do not fit shared memory there. The selection sum, F, the softmax statistics and every
accumulator are fp32. fp32 operand tiles take twice the shared memory of bf16 ones, so those paths
shrink their key tile automatically; results do not depend on tile sizes.

## Tile sizes

Measured at head dimension 64 on an RTX PRO 6000 Blackwell. Version 1: 64 queries x 32 keys for
the forward, 32 x 32 for the backward, four warps; at 16K tokens 2.3x (forward) and 1.5x (backward)
faster than 64 x 64 tiles. Version 2: 32 x 32 tiles in both passes, four warps, one stage, with the
head groups above; the defaults choose smaller groups for wider heads and for fp32 operands, which
need twice the shared memory.

## Cached decoding

A decode step is one query against the cached keys. The cache holds K, V and one fp32 vector per
sequence, the selection every key has received so far; the step subtracts it from every head's
logits and then adds the new query's own selection. It is written in PyTorch
(`src/selective_attention/mixers.py`); `selective_attention_forward(..., return_aux=True)` returns the vector
after a prefill.

## Limits

- Inputs: (B, H, T, D) CUDA tensors in fp16, bf16 or fp32, with D in {16, 32, 64, 128}; anything
  else raises a ValueError (`kernels.check_inputs`). The model layer falls back to the PyTorch
  reference when `kernels.supported` is false.
- T is a compile-time constant: each new sequence length compiles its own kernels (a few seconds,
  cached afterwards).
- The kernels have been tested to T = 270,000 (forward, past the int32 offset limit) and used to
  T = 524,288 (forward only). Training used T = 16,384.
- `exempt` must be at least 1: position 0 (<BOS>) is never selected, as in the paper.
- `selective_attention_forward(damp=..., train_len=...)` is an evaluation-only experiment that
  scales the history term past the trained length. It made long-document loss worse (README); the
  backward and cached decoding do not implement it.
