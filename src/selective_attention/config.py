"""Model configuration: the single source of truth for model keys.

One dataclass serves both the experiment harness (YAML `model:` section) and
model construction: exhaustive validation in ``__post_init__``, ``from_dict``
that rejects unknown keys, ``to_dict`` for self-contained checkpoints.

Only the evidence-frozen clean design is configurable (see
docs/design-rationale.md): the dilution operator itself has no knobs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping

try:  # optional token-local mixers kept for internal ablations; absent from the public release
    from .mixers import MIXER_TYPES
except ImportError:
    MIXER_TYPES = ()
ATTENTION_TYPES = ("dilution", "softmax") + MIXER_TYPES
MLP_TYPES = ("swiglu", "gelu")
POSITIONAL_TYPES = ("learned", "none", "rope", "rope_alt")  # rope_alt: RoPE on layers 1,3,5,... (1-indexed), none on the rest
BID_TYPES = ("softmax", "relu", "relu2", "softplus", "minshift", "sigmoid", "expshift", "sigshift")


class ConfigurationError(ValueError):
    """Raised when an experiment configuration is internally inconsistent."""


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive, got {value}")
    return value


@dataclass
class ModelConfig:
    """Configuration for `dilution.model.DilutionLM`.

    `attention` sets the default attention-slot type; `layer_attention`
    overrides it per layer. "dilution" is the committed-claim competition
    operator (`dilution.kernels.dilution_attention_reference`); "softmax"
    the parameter-matched SDPA control.
    """

    vocab_size: int = 50304
    n_layers: int = 12
    d_model: int = 512
    n_heads: int = 8
    context_length: int = 1024
    dropout: float = 0.0
    bias: bool = False
    attention: str = "dilution"
    layer_attention: list | None = None
    # FFN family: "swiglu" (8C/3 hidden by default, parameter-matched to the
    # conventional 4C GELU MLP) or "gelu"; override width with ffn_hidden.
    mlp: str = "swiglu"
    ffn_hidden: int | None = None
    # Position signal: "learned" absolute embeddings (default) or "none" (NoPE:
    # the only order information is causal masking and, for dilution, the
    # operator's own cumulative asymmetry).
    positional: str = "learned"
    rope_theta: float = 10000.0  # RoPE base; larger flattens the distance decay (Llama 3 uses 5e5 at 8K)
    # Bid function feeding the dilution step: "softmax" (the operator), or the
    # "percent of share" ablations "relu" (relu(a)/rowsum) and "relu2" (relu(a)^2/rowsum).
    bid: str = "softmax"
    # False: attn = p directly (no cumsum / committed division): the "linear share" control.
    # True = inclusive demand (dilution); "exclusive" = temporal-attention history
    # (Sankaran 2016 / Paulus 2018) on row-normalised bids; False = no share step.
    share_cumsum: bool | str = True

    def __post_init__(self) -> None:
        _require_positive_int(self.vocab_size, "model.vocab_size")
        _require_positive_int(self.n_layers, "model.n_layers")
        _require_positive_int(self.d_model, "model.d_model")
        _require_positive_int(self.n_heads, "model.n_heads")
        _require_positive_int(self.context_length, "model.context_length")
        if self.d_model % self.n_heads != 0:
            raise ConfigurationError(
                f"model.d_model ({self.d_model}) must be divisible by model.n_heads ({self.n_heads})"
            )
        if not isinstance(self.dropout, (int, float)) or isinstance(self.dropout, bool):
            raise ConfigurationError("model.dropout must be a number")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ConfigurationError(f"model.dropout must be in [0, 1), got {self.dropout}")
        if not isinstance(self.bias, bool):
            raise ConfigurationError("model.bias must be a boolean")
        if self.attention not in ATTENTION_TYPES:
            raise ConfigurationError(
                f"model.attention must be one of {', '.join(ATTENTION_TYPES)}; got {self.attention!r}"
            )
        if self.layer_attention is not None:
            if not isinstance(self.layer_attention, list) or len(self.layer_attention) != self.n_layers:
                raise ConfigurationError(
                    f"model.layer_attention must be a list of {self.n_layers} entries"
                )
            invalid = sorted({str(k) for k in self.layer_attention if k not in ATTENTION_TYPES})
            if invalid:
                raise ConfigurationError(
                    f"model.layer_attention entries must be in {', '.join(ATTENTION_TYPES)}; "
                    f"got {', '.join(invalid)}"
                )
        if self.mlp not in MLP_TYPES:
            raise ConfigurationError(
                f"model.mlp must be one of {', '.join(MLP_TYPES)}; got {self.mlp!r}"
            )
        if self.ffn_hidden is not None:
            _require_positive_int(self.ffn_hidden, "model.ffn_hidden")
        if not (isinstance(self.share_cumsum, bool) or self.share_cumsum == "exclusive"):
            raise ConfigurationError('model.share_cumsum must be a boolean or "exclusive"')
        if self.bid not in BID_TYPES:
            raise ConfigurationError(
                f"model.bid must be one of {', '.join(BID_TYPES)}; got {self.bid!r}"
            )
        if self.positional not in POSITIONAL_TYPES:
            raise ConfigurationError(
                f"model.positional must be one of {', '.join(POSITIONAL_TYPES)}; got {self.positional!r}"
            )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(str(key) for key in values if key not in known)
        if unknown:
            joined = ", ".join(f"model.{key}" for key in unknown)
            raise ConfigurationError(f"unknown configuration key(s): {joined}")
        return cls(**dict(values))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def layer_kinds(self) -> list[str]:
        if self.layer_attention is not None:
            return list(self.layer_attention)
        return [self.attention] * self.n_layers


def build_model(config: ModelConfig):
    """Materialize the DilutionLM described by a ModelConfig."""

    from .model import DilutionLM

    return DilutionLM(config)
