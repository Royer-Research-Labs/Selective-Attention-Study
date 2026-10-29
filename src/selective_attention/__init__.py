"""Fused Triton kernels for Selective Attention (Leviathan, Kalman, Matias; ICLR 2025)."""

from .kernels import (
    HAS_TRITON,
    selective_attention,
    selective_attention_forward,
    selective_attention_reference,
    selective_logits,
)

__all__ = [
    "HAS_TRITON",
    "selective_attention",
    "selective_attention_forward",
    "selective_attention_reference",
    "selective_logits",
]
