"""Running a trained model past its trained context, for evaluation.

`ExtendedContext(model, need, ntk)` temporarily raises `model.config.context_length` to `need`
tokens and, with `ntk`, applies dynamic NTK-aware RoPE scaling: every RoPE base becomes
theta * (need / trained) ** (D / (D - 2)), as in the NIAH extrapolation ladders. Models with a
learned position table cannot be extended. Everything is restored on exit.
"""
from __future__ import annotations


class ExtendedContext:
    def __init__(self, model, need: int, ntk: bool = True):
        self.model, self.need, self.ntk = model, int(need), ntk
        self.trained = model.config.context_length
        self.ropes = [m for m in model.modules() if type(m).__name__ == "RopeModule"]
        self.thetas = [r.theta for r in self.ropes]

    @property
    def scale(self) -> float:
        return max(1.0, self.need / self.trained)

    def __enter__(self):
        if self.need > self.trained:
            if getattr(self.model.config, "positional", "learned") == "learned":
                raise ValueError("cannot extend a model with a learned position table")
            self.model.config.context_length = self.need
            if self.ntk:
                for r, th in zip(self.ropes, self.thetas):
                    r.theta = th * self.scale ** (r.dim / (r.dim - 2))
        return self.model

    def __exit__(self, *exc):
        self.model.config.context_length = self.trained
        for r, th in zip(self.ropes, self.thetas):
            r.theta = th
