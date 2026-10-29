"""Rotary position embedding (RoPE) for [B, H, T, D] attention tensors.

Rotary pairs are adjacent elements
(0, 1), (2, 3), ...; the frequency table is built in fp32 on every call so a
bf16 module never carries a downcast table, and cos/sin are cast to the
input dtype only at the end. `offset` is the absolute position of the first
row, which cached decode uses.
"""

from __future__ import annotations

import torch
from torch import nn


class RopeModule(nn.Module):
    def __init__(self, dim: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be a positive even integer, got {dim}")
        if theta <= 0.0:
            raise ValueError(f"RoPE theta must be positive, got {theta}")
        self.dim = dim
        self.theta = theta

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        if x.size(-1) != self.dim:
            raise ValueError(f"RoPE expected last dimension {self.dim}, got {x.size(-1)}")
        if offset < 0:
            raise ValueError(f"RoPE offset must be non-negative, got {offset!r}")
        seq_len = x.size(-2)
        positions = torch.arange(offset, offset + seq_len, device=x.device, dtype=torch.float32)
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, device=x.device, dtype=torch.float32) / self.dim)
        )
        freqs = torch.outer(positions, inv_freq)
        cos = freqs.cos().to(dtype=x.dtype)
        sin = freqs.sin().to(dtype=x.dtype)
        broadcast = [1] * (x.ndim - 2) + [seq_len, self.dim]
        cos = cos.repeat_interleave(2, dim=-1).view(broadcast)
        sin = sin.repeat_interleave(2, dim=-1).view(broadcast)
        pairs = x.reshape(*x.shape[:-1], self.dim // 2, 2)
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        return x * cos + rotated * sin
