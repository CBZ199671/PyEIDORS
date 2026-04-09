"""Typed contracts shared by inverse solvers and workflows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SolverOutput:
    """Structured solver output consumed by workflow wrappers."""

    conductivity: Any
    residual_history: Sequence[float] | None = None
    sigma_change_history: Sequence[float] | None = None
    iterations: int = 0
    converged: bool = False
    final_residual: float = float("nan")
    final_relative_change: float = float("nan")
    jacobian_method: str | None = None
    regularization_type: str | None = None
    iteration_logs: Sequence[dict[str, Any]] | None = None
    conductivity_history: Sequence[np.ndarray] | None = None
    baseline_measurement: np.ndarray | None = None
    measurement_weight: np.ndarray | None = None
    simulated_measurement: np.ndarray | None = None
    likelihood_noise_std: float | None = None
    prior_scale: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
