"""Typed contracts shared by inverse solvers and workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np


@dataclass
class SolverOutput:
    """Structured solver output consumed by workflow wrappers."""

    conductivity: Any
    residual_history: Optional[Sequence[float]] = None
    sigma_change_history: Optional[Sequence[float]] = None
    iterations: int = 0
    converged: bool = False
    final_residual: float = float("nan")
    final_relative_change: float = float("nan")
    jacobian_method: Optional[str] = None
    regularization_type: Optional[str] = None
    iteration_logs: Optional[Sequence[Dict[str, Any]]] = None
    conductivity_history: Optional[Sequence[np.ndarray]] = None
    baseline_measurement: Optional[np.ndarray] = None
    measurement_weight: Optional[np.ndarray] = None
    simulated_measurement: Optional[np.ndarray] = None
    likelihood_noise_std: Optional[float] = None
    prior_scale: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
