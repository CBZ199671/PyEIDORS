"""Inexact GN forcing-term controller."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InexactController:
    """Track and adapt forcing term ``eta_k`` for inexact inner solves."""

    mode: str = "eisenstat-walker"
    eta0: float = 0.2
    eta_min: float = 1e-3
    eta_max: float = 0.5
    _eta: float = field(init=False)
    history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.mode = str(self.mode).strip().lower()
        if self.mode not in {"fixed", "eisenstat-walker"}:
            self.mode = "eisenstat-walker"
        self.eta_min = float(max(1e-8, self.eta_min))
        self.eta_max = float(max(self.eta_min, self.eta_max))
        self._eta = self._clip(float(self.eta0))
        self.history.append(float(self._eta))

    def _clip(self, value: float) -> float:
        return float(min(max(value, self.eta_min), self.eta_max))

    @property
    def eta(self) -> float:
        return float(self._eta)

    def suggest_eta(self) -> float:
        return float(self._eta)

    def update(
        self,
        *,
        outer_prev: float | None,
        outer_curr: float | None,
        linear_residual_ratio: float | None,
        step_rejected: bool,
        stalled: bool,
    ) -> float:
        """Update forcing term based on outer progress and inner residual ratio."""
        if self.mode == "fixed":
            if step_rejected or stalled:
                self._eta = self._clip(self._eta * 0.5)
            self.history.append(float(self._eta))
            return float(self._eta)

        ratio = 1.0
        if isinstance(outer_prev, (int, float)) and isinstance(
            outer_curr, (int, float)
        ):
            denom = max(abs(float(outer_prev)), 1e-12)
            ratio = abs(float(outer_curr)) / denom

        eta_candidate = 0.9 * (ratio**2)
        if (
            isinstance(linear_residual_ratio, (int, float))
            and linear_residual_ratio > 0
        ):
            eta_candidate = max(float(linear_residual_ratio), eta_candidate)

        if step_rejected:
            eta_candidate *= 0.5
        if stalled:
            eta_candidate *= 0.7

        self._eta = self._clip(float(eta_candidate))
        self.history.append(float(self._eta))
        return float(self._eta)
