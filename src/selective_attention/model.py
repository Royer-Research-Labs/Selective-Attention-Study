"""DilutionLM: a decoder-only LM whose global mixing is dilution attention.

Dilution attention is defined in docs/operator.md; `dilution.kernels.dilution_attention_reference` is the
authoritative math. `model.attention` / `model.layer_attention` choose dilution or softmax
attention per layer, so all-dilution, all-softmax and alternating stacks (alt6) share one model.

Every attention slot supports cached decode: softmax carries a KV cache, and dilution carries
KV plus one O(T) accumulator vector (committed column totals of p).
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig
from .rope import RopeModule


def _autocast_off(device: torch.device):
    if device.type in ("cuda", "cpu"):
        return torch.autocast(device.type, enabled=False)
    return contextlib.nullcontext()


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class _AttentionBase(nn.Module):
    """Shared q/k/v projection plumbing for the two attention types."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.dropout = config.dropout
        self.context_length = config.context_length
        # Set per layer by DilutionLM under positional "rope" / "rope_alt";
        # rotates q and k in _split_heads (offset = absolute position of row 0).
        self.rope: RopeModule | None = None

    def _split_heads(self, x: Tensor, offset: int = 0) -> tuple[Tensor, Tensor, Tensor]:
        batch, seq, _ = x.shape
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        shape = (batch, seq, self.n_heads, self.d_model // self.n_heads)
        q, k, v = (q.view(shape).transpose(1, 2), k.view(shape).transpose(1, 2),
                   v.view(shape).transpose(1, 2))
        if self.rope is not None:
            q, k = self.rope(q, offset), self.rope(k, offset)
        return q, k, v

    def _merge_heads(self, y: Tensor, batch: int, seq: int) -> Tensor:
        y = y.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        return self.resid_dropout(self.c_proj(y))

    # Decode caches are preallocated (context_length, or the caller's capacity when
    # generating past it) and filled by slice assignment, so a token step never
    # reallocates or copies the cache.
    def _alloc_cache(self, k: Tensor, v: Tensor, capacity: int | None = None) -> dict:
        batch, heads, seq, head_dim = k.shape
        capacity = max(capacity if capacity is not None else self.context_length, seq)
        k_buf = k.new_zeros(batch, heads, capacity, head_dim)
        v_buf = v.new_zeros(batch, heads, capacity, head_dim)
        k_buf[:, :, :seq] = k
        v_buf[:, :, :seq] = v
        return {"k": k_buf, "v": v_buf, "len": seq}

    @staticmethod
    def _append_kv(cache: dict, k1: Tensor, v1: Tensor) -> int:
        t = cache["len"]
        if t >= cache["k"].shape[2]:
            raise ValueError("decode step past the cache capacity (context_length)")
        cache["k"][:, :, t] = k1[:, :, 0]
        cache["v"][:, :, t] = v1[:, :, 0]
        cache["len"] = t + 1
        return t + 1


class CausalSelfAttention(_AttentionBase):
    """Row-wise softmax attention (SDPA): the parameter-matched control."""

    def forward(self, x: Tensor) -> Tensor:
        batch, seq, _ = x.shape
        q, k, v = self._split_heads(x)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        return self._merge_heads(y, batch, seq)

    def begin_cache(self, x: Tensor, capacity: int | None = None):
        y = self(x)
        q, k, v = self._split_heads(x)
        return y, self._alloc_cache(k, v, capacity)

    def step_cache(self, x1: Tensor, cache: dict) -> Tensor:
        batch, _, _ = x1.shape
        q1, k1, v1 = self._split_heads(x1, offset=cache["len"])
        t = self._append_kv(cache, k1, v1)
        y = F.scaled_dot_product_attention(q1, cache["k"][:, :, :t], cache["v"][:, :, :t])
        return self._merge_heads(y, batch, 1)


class DilutionAttention(_AttentionBase):
    """Dilution attention: a bid divided by the demand already on that key.

    See kernels.dilution_attention_reference for the math. The operator has no
    learnable scalars -- the order asymmetry comes from the cumulative sum
    alone. `use_kernel` (runtime knob, see DilutionLM.enable_dilution_kernel)
    routes through the fused Triton forward/backward; the tensor path below is
    the fallback and CPU reference.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.use_kernel = False
        self.bid = config.bid
        self.share_cumsum = config.share_cumsum

    def _bids(self, aff: Tensor, seq: int) -> Tensor:
        """p from scaled affinities under the configured bid function (tensor path)."""
        from .kernels_bid import bids_from_affinity
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=aff.device))
        return bids_from_affinity(aff, mask, self.bid, self.context_length)

    def _standard(self) -> bool:
        return self.bid == "softmax" and self.share_cumsum is True

    def _kernels(self):
        if self._standard():
            from . import dilution_kernels as kernels  # production module: no bid / share arguments
            return kernels
        from . import kernels_bid
        return kernels_bid

    def _bid_kw(self) -> dict:
        if self._standard():
            return {}
        # expshift divides by the model's context length (the shift is fixed, not learned)
        return {"bid": self.bid, "share_cumsum": self.share_cumsum, "shift": float(self.context_length)}

    def _kernel_ok(self, x: Tensor) -> bool:
        return (
            self.use_kernel
            and x.is_cuda
            and self.d_model // self.n_heads >= 16  # Triton dot minimum
            and (self.d_model // self.n_heads) & (self.d_model // self.n_heads - 1) == 0
            and (not self.training or self.dropout == 0.0)
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, seq, _ = x.shape
        q, k, v = self._split_heads(x)

        if self._kernel_ok(x):
            prec = "bf16" if q.dtype == torch.bfloat16 else "tf32"
            if not torch.is_grad_enabled():
                # Inference: the autograd Function would still build the backward's
                # O(n T^2 / 64) carry. The plain forward chunks its scratch instead.
                y = self._kernels().dilution_attention_forward(
                    q, k, v, dot_precision=prec, **self._bid_kw()).to(x.dtype)
            else:
                y = self._kernels().dilution_attention(q, k, v, dot_precision=prec, **self._bid_kw()).to(x.dtype)
            return self._merge_heads(y, batch, seq)

        head_dim = self.d_model // self.n_heads
        with _autocast_off(x.device):
            aff = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(head_dim)
            p = self._bids(aff, seq)
            if self.share_cumsum:
                committed = p.cumsum(dim=-2) - p
                # "exclusive": earlier queries only (temporal attention). The first row
                # has committed == 0, so the row normalise below cancels eps exactly.
                denom = committed if self.share_cumsum == "exclusive" else committed + p
                share = p / (denom + 1e-9)
                attn = share / (share.sum(dim=-1, keepdim=True) + 1e-9)
            else:
                attn = p
        y = (attn.to(v.dtype) @ v)
        return self._merge_heads(y, batch, seq)

    def begin_cache(self, x: Tensor, capacity: int | None = None):
        batch, seq, _ = x.shape
        q, k, v = self._split_heads(x)

        if self._kernel_ok(x):
            prec = "bf16" if q.dtype == torch.bfloat16 else "tf32"
            y, aux = self._kernels().dilution_attention_forward(
                q, k, v, dot_precision=prec, return_aux=True, **self._bid_kw())
            committed = aux["committed_total"]
            y = y.to(x.dtype)
        else:
            head_dim = self.d_model // self.n_heads
            with _autocast_off(x.device):
                aff = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(head_dim)
                p = self._bids(aff, seq)
                if self.share_cumsum:
                    co = p.cumsum(dim=-2) - p
                    denom = co if self.share_cumsum == "exclusive" else co + p
                    share = p / (denom + 1e-9)
                    attn = share / (share.sum(dim=-1, keepdim=True) + 1e-9)
                else:
                    attn = p
                y = attn @ v.float()
                committed = p.sum(dim=-2)
            y = y.to(x.dtype)

        cache = self._alloc_cache(k, v, capacity)
        capacity = cache["k"].shape[2]
        committed_buf = torch.zeros(batch, self.n_heads, capacity, device=x.device, dtype=torch.float32)
        committed_buf[:, :, :seq] = committed.float()
        cache.update(committed=committed_buf)
        return self._merge_heads(y, batch, seq), cache

    def step_cache(self, x1: Tensor, cache: dict) -> Tensor:
        batch, _, _ = x1.shape
        q1, k1, v1 = self._split_heads(x1, offset=cache["len"])
        t = self._append_kv(cache, k1, v1)

        if self._kernel_ok(x1):
            y = self._kernels().dilution_decode_step(
                q1, cache["k"], cache["v"], cache["committed"], t, **self._bid_kw(),
            ).to(q1.dtype)  # kernel stores q1.dtype: no cast launch on the bf16 path
            return self._merge_heads(y, batch, 1)

        head_dim = self.d_model // self.n_heads
        k, v = cache["k"][:, :, :t], cache["v"][:, :, :t]
        committed = cache["committed"][:, :, :t]
        with _autocast_off(x1.device):
            aff = (q1.float() @ k.float().transpose(-2, -1)).squeeze(2) / math.sqrt(head_dim)
            from .kernels_bid import bids_from_affinity
            p = bids_from_affinity(aff, torch.ones_like(aff, dtype=torch.bool), self.bid, self.context_length)
            if self.share_cumsum:
                denom = committed if self.share_cumsum == "exclusive" else committed + p
                share = p / (denom + 1e-9)
                attn = share / (share.sum(dim=-1, keepdim=True) + 1e-9)
            else:
                attn = p
            y = (attn.unsqueeze(2) @ v.float()).to(x1.dtype)
        committed += p
        return self._merge_heads(y, batch, 1)


class SwiGLUMLP(nn.Module):
    """SwiGLU FFN; default hidden 8C/3 parameter-matches the 4C GELU MLP."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        n = config.d_model
        hidden = config.ffn_hidden if config.ffn_hidden is not None else (8 * n) // 3
        self.gate_proj = nn.Linear(n, hidden, bias=config.bias)
        self.up_proj = nn.Linear(n, hidden, bias=config.bias)
        self.down_proj = nn.Linear(hidden, n, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class GELUMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        n = config.d_model
        hidden = config.ffn_hidden if config.ffn_hidden is not None else 4 * n
        self.c_fc = nn.Linear(n, hidden, bias=config.bias)
        self.c_proj = nn.Linear(hidden, n, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


def filter_logits(logits: Tensor, top_k: int | None = None, top_p: float | None = None) -> Tensor:
    """Top-k then nucleus truncation of (..., vocab) logits: excluded entries become -inf.
    Nucleus keeps the smallest set of highest-probability tokens whose mass reaches top_p (the
    most likely token is always kept)."""

    if top_k is not None and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, order = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        drop = probs.cumsum(dim=-1) - probs >= top_p      # mass before this token already >= top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)
    return logits


_ATTENTION_CLASSES = {
    "dilution": DilutionAttention,
    "softmax": CausalSelfAttention,
}
try:  # optional token-local mixers kept for internal ablations (see config.py)
    from .mixers import MIXER_CLASSES
    _ATTENTION_CLASSES.update(MIXER_CLASSES)
except ImportError:
    pass


class Block(nn.Module):
    def __init__(self, config: ModelConfig, kind: str):
        super().__init__()
        self.ln_1 = LayerNorm(config.d_model, config.bias)
        self.attn = _ATTENTION_CLASSES[kind](config)
        self.ln_2 = LayerNorm(config.d_model, config.bias)
        self.mlp = SwiGLUMLP(config) if config.mlp == "swiglu" else GELUMLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def begin_cache(self, x: Tensor, capacity: int | None = None):
        y, cache = self.attn.begin_cache(self.ln_1(x), capacity)
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x, cache

    def step_cache(self, x1: Tensor, cache) -> Tensor:
        x1 = x1 + self.attn.step_cache(self.ln_1(x1), cache)
        x1 = x1 + self.mlp(self.ln_2(x1))
        return x1


class DilutionLM(nn.Module):
    """Decoder-only LM over configurable attention slots, tied embeddings.

    forward(idx, targets=None) -> (logits, loss): with targets, full-sequence
    logits and mean cross-entropy (ignore_index -100); without, last-position
    logits only.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.d_model),
                "wpe": (nn.Embedding(config.context_length, config.d_model)
                        if config.positional == "learned" else nn.Identity()),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList(
                    [Block(config, kind) for kind in config.layer_kinds()]
                ),
                "ln_f": LayerNorm(config.d_model, config.bias),
            }
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        if config.positional in ("rope", "rope_alt"):
            # RoPE lives in the attention layers (softmax and dilution alike;
            # layers without a q.k product are skipped). "rope_alt"
            # rotates alternate layers starting with the first (layers 1, 3,
            # 5, ... counted from 1), so the first layer always has position
            # data and the others see only causal order, as under "none".
            head_dim = config.d_model // config.n_heads
            for i, block in enumerate(self.transformer.h):
                if isinstance(block.attn, _AttentionBase) and (config.positional == "rope" or i % 2 == 0):
                    block.attn.rope = RopeModule(head_dim, theta=float(config.rope_theta))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed(self, idx: Tensor, start: int) -> Tensor:
        """Token embedding plus the learned position signal; "none"/"rope"/"rope_alt" add nothing here."""
        x = self.transformer.wte(idx)
        if self.config.positional == "learned":
            pos = torch.arange(start, start + idx.shape[1], dtype=torch.long, device=idx.device)
            x = x + self.transformer.wpe(pos)
        return x

    def enable_dilution_kernel(self, enabled: bool = True) -> None:
        """Route dilution layers through the fused Triton kernels (runtime knob)."""

        for block in self.transformer.h:
            if isinstance(block.attn, DilutionAttention):
                block.attn.use_kernel = enabled

    def forward(self, idx: Tensor, targets: Tensor | None = None):
        _, seq = idx.shape
        if seq > self.config.context_length:
            raise ValueError(
                f"sequence length {seq} exceeds context_length {self.config.context_length}"
            )
        x = self.transformer.drop(self._embed(idx, 0))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x if targets is not None else x[:, [-1], :])
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def eval_chunked(self, idx: Tensor, targets: Tensor, chunk: int = 2048):
        """Evaluation-only forward: (predictions, mean loss, token count) with the
        lm_head and cross-entropy applied per `chunk` positions, so no full-vocab
        logits tensor for the whole sequence is ever allocated. Same numbers as forward(idx, targets) up to
        fp32 summation order."""
        _, seq = idx.shape
        if seq > self.config.context_length:
            raise ValueError(
                f"sequence length {seq} exceeds context_length {self.config.context_length}"
            )
        x = self.transformer.drop(self._embed(idx, 0))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        loss_sum = torch.zeros((), device=x.device, dtype=torch.float32)
        count = torch.zeros((), device=x.device, dtype=torch.float32)
        preds = []
        for start in range(0, seq, chunk):
            logits = self.lm_head(x[:, start:start + chunk, :])
            tgt = targets[:, start:start + chunk]
            loss_sum = loss_sum + F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100, reduction="sum"
            ).float()
            count = count + (tgt != -100).sum().float()
            preds.append(logits.argmax(dim=-1))
            del logits
        return torch.cat(preds, dim=1), loss_sum / count.clamp(min=1.0), count

    # --- cached decode (docs/kernels.md, v4) --------------------------------

    def prefill(self, idx: Tensor, capacity: int | None = None):
        """Run the prompt once; returns last-position logits and per-layer caches.
        `capacity` sizes the decode caches (default: the attention layers' context_length);
        pass prompt + new tokens when generating past the trained context."""

        x = self.transformer.drop(self._embed(idx, 0))
        caches = []
        for block in self.transformer.h:
            x, cache = block.begin_cache(x, capacity)
            caches.append(cache)
        x = self.transformer.ln_f(x)
        return self.lm_head(x[:, [-1], :]), caches

    def decode_step(self, token: Tensor, position: int, caches) -> Tensor:
        """One cached step: (B, 1) token ids -> (B, 1, vocab) logits."""

        x = self.transformer.drop(self._embed(token, position))
        for block, cache in zip(self.transformer.h, caches):
            x = block.step_cache(x, cache)
        return self.lm_head(self.transformer.ln_f(x))

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
        vocab_limit: int | None = None,
    ) -> Tensor:
        """Cached greedy/sampled generation. temperature 0 = greedy; otherwise sample from
        softmax(logits / temperature) after optional top-k and nucleus (top-p) truncation.
        `generator` makes sampling reproducible; `vocab_limit` restricts sampling to ids below
        it (the GPT-2 tokenizer has 50,257 of the 50,304 embedding rows)."""

        if idx.shape[1] + max_new_tokens > self.config.context_length:
            raise ValueError(
                f"prompt ({idx.shape[1]}) + max_new_tokens ({max_new_tokens}) exceeds "
                f"context_length {self.config.context_length}"
            )
        was_training = self.training
        self.eval()
        try:
            logits, caches = self.prefill(idx, capacity=idx.shape[1] + max_new_tokens)
            tokens = idx
            for step in range(max_new_tokens):
                step_logits = logits[:, -1, :].float()
                if vocab_limit is not None:
                    step_logits[:, vocab_limit:] = float("-inf")
                if temperature and temperature > 0.0:
                    step_logits = filter_logits(step_logits / temperature, top_k=top_k, top_p=top_p)
                    next_token = torch.multinomial(step_logits.softmax(dim=-1), 1, generator=generator)
                else:
                    next_token = step_logits.argmax(dim=-1, keepdim=True)
                tokens = torch.cat([tokens, next_token], dim=1)
                if step + 1 < max_new_tokens:
                    logits = self.decode_step(next_token, tokens.shape[1] - 1, caches)
        finally:
            self.train(was_training)
        return tokens

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
