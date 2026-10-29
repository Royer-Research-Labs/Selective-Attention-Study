"""Selective Attention as an attention-slot type of the training harness.

The harness modules in this package (config, model, trainer, data, evaluation, NIAH bridge) are
copied from https://github.com/Royer-Research-Labs/Dilution-Attention (vendored.json lists them).
That harness looks for an optional `mixers` module beside it and registers whatever it exports, so
`attention: selective` (or a `layer_attention` entry) builds the layer below and every other code
path is the one that produced the published dilution and softmax results.

The operator is `selective_attention.kernels`; the plain reference there is the fallback on CPU.
"""

from __future__ import annotations

import math

# name: (exempt, per_head).
# "selective_x16": the first 16 positions of the window can never be selected (the paper exempts only
# <BOS>). Tests the first-tokens blind spot found 2026-10-18 (scripts/investigate/niah_investigate.py).
# "selective_perhead": every head subtracts its own selection history instead of head 0's (not the
# paper's operator; the same kernels run each head as its own one-head sequence).
VARIANTS = {"selective": (1, False), "selective_x16": (16, False), "selective_perhead": (1, True)}
MIXER_TYPES = tuple(VARIANTS)


def __getattr__(name: str):
    # Built on first use: `config` imports MIXER_TYPES from here before `model` exists, and the
    # layer class needs `model._AttentionBase`.
    if name == "MIXER_CLASSES":
        cls = _selective_class()
        return {name: (lambda config, n=n, h=h: cls(config, exempt=n, per_head=h))
                for name, (n, h) in VARIANTS.items()}
    raise AttributeError(name)


_CLASS = None


def _selective_class():
    global _CLASS
    if _CLASS is not None:
        return _CLASS

    import torch
    from torch import Tensor

    from . import kernels

    from .model import _AttentionBase, _autocast_off

    class SelectiveAttention(_AttentionBase):
        """Selective Attention (Leviathan, Kalman, Matias; ICLR 2025, arXiv 2410.02703):
        softmax(L - F), F[i, j] the sum over earlier queries of head 0's relu'd logit on key j.
        No parameters beyond the q / k / v projections every attention layer has.

        `use_kernel` (default on) routes CUDA calls through the fused Triton forward / backward;
        otherwise the (T x T) reference runs. Cached decode carries KV plus one O(T) vector per
        sequence and selector: the selection every key has received so far.

        per_head=True gives every head its own history (its own relu'd logits), by running each head
        through the kernels as a separate one-head sequence.
        """

        def __init__(self, config, exempt: int = 1, per_head: bool = False):
            super().__init__(config)
            self.exempt = kernels._check_exempt(exempt)
            self.groups = self.n_heads if per_head else 1   # selectors per layer (heads in contiguous groups)
            self.use_kernel = True
            # Runtime, evaluation only (scripts/investigate/history_damp_eval.py): applied in the
            # kernel forward (prefill); the reference fallback and cached decode ignore it.
            self.history_damp = "none"
            self.dot_precision = None   # runtime knob; None: "bf16" for bf16 inputs, else "tf32"

        def _kernel_ok(self, q: Tensor) -> bool:
            return self.use_kernel and kernels.supported(q) and (not self.training or self.dropout == 0.0)

        def _attend(self, q: Tensor, k: Tensor, v: Tensor, want_total: bool = False):
            """(B, H, T, D) outputs in v's dtype and, on request, the (B, G, T) selection totals."""
            batch, heads, seq, head_dim = q.shape
            groups = self.groups
            if groups > 1:   # each group of heads becomes its own sequence, so its first head selects
                q, k, v = (t.reshape(batch * groups, heads // groups, seq, head_dim) for t in (q, k, v))
            y, total = self._attend_shared(q, k, v, want_total)
            y = y.reshape(batch, heads, seq, head_dim)
            return y, (None if total is None else total.reshape(batch, groups, seq))

        def _attend_shared(self, q: Tensor, k: Tensor, v: Tensor, want_total: bool):
            if self._kernel_ok(q):
                prec = self.dot_precision   # None: the kernel's rule (bf16 for bf16 inputs, else tf32)
                if want_total or not torch.is_grad_enabled():
                    y, aux = kernels.selective_attention_forward(q, k, v, dot_precision=prec, return_aux=True,
                                                                 exempt=self.exempt, damp=self.history_damp,
                                                                 train_len=self.context_length)
                    return y.to(v.dtype), aux["selected_total"]
                return kernels.selective_attention(q, k, v, dot_precision=prec, exempt=self.exempt).to(v.dtype), None
            with _autocast_off(q.device):
                logits, total = kernels.selective_logits(q, k, self.exempt)
                y = (logits.softmax(dim=-1) @ v.float()).to(v.dtype)
            return y, total

        def forward(self, x: Tensor) -> Tensor:
            batch, seq, _ = x.shape
            q, k, v = self._split_heads(x)
            return self._merge_heads(self._attend(q, k, v)[0], batch, seq)

        def begin_cache(self, x: Tensor, capacity: int | None = None):
            batch, seq, _ = x.shape
            q, k, v = self._split_heads(x)
            y, total = self._attend(q, k, v, want_total=True)
            cache = self._alloc_cache(k, v, capacity)
            selected = torch.zeros(batch, self.groups, cache["k"].shape[2], device=x.device, dtype=torch.float32)
            selected[:, :, :seq] = total.float()
            cache.update(selected=selected)
            return self._merge_heads(y, batch, seq), cache

        def step_cache(self, x1: Tensor, cache: dict) -> Tensor:
            batch, _, _ = x1.shape
            q1, k1, v1 = self._split_heads(x1, offset=cache["len"])
            t = self._append_kv(cache, k1, v1)
            k, v = cache["k"][:, :, :t], cache["v"][:, :, :t]
            selected = cache["selected"][:, :, :t]                   # (B, G, t)
            per = self.n_heads // self.groups
            with _autocast_off(x1.device):
                raw = (q1.float() @ k.float().transpose(-2, -1)) / math.sqrt(self.d_model // self.n_heads)
                hist = selected.repeat_interleave(per, dim=1)[:, :, None, :]
                y = ((raw - hist).softmax(dim=-1) @ v.float()).to(x1.dtype)
                sel = torch.relu(raw[:, ::per, 0]).clone()   # this query's own selection, first head of each group
                sel[..., : self.exempt] = 0.0                # not <BOS> (the first `exempt` positions)
                sel[..., t - 1] = 0.0                        # not self
                selected += sel
            return self._merge_heads(y, batch, 1)

    _CLASS = SelectiveAttention
    return _CLASS
