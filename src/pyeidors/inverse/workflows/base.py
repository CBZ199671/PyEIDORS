"""Imaging workflow common utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...data.structures import EITImage
from ...femx import function_get_array
from ..contracts import SolverOutput


@dataclass
class ReconstructionResult:
    """Unified output encapsulation for difference/absolute imaging."""

    mode: str
    conductivity: np.ndarray
    conductivity_image: EITImage
    measured: np.ndarray
    simulated: np.ndarray
    residual: np.ndarray
    residual_history: Sequence[float] | None = None
    sigma_change_history: Sequence[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def l2_error(self) -> float:
        return float(np.linalg.norm(self.residual))

    @property
    def relative_error(self) -> float:
        numerator = np.linalg.norm(self.residual)
        denominator = np.linalg.norm(self.measured) + 1e-12
        return float(numerator / denominator)

    @property
    def mse(self) -> float:
        return float(np.mean(self.residual**2))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for script-facing serialization."""

        data = {
            "mode": self.mode,
            "conductivity_values": self.conductivity,
            "measured_vector": self.measured,
            "simulated_vector": self.simulated,
            "residual_vector": self.residual,
            "l2_error": self.l2_error,
            "rel_error": self.relative_error,
            "mse": self.mse,
            "residual_history": self.residual_history,
            "sigma_change": self.sigma_change_history,
        }
        data.update(self.metadata)
        return data


def resolve_reconstruction_output(
    reconstruction: SolverOutput,
    fwd_model: Any,
) -> tuple[EITImage, np.ndarray, Sequence[float] | None, Sequence[float] | None]:
    """Extract conductivity image and history from typed solver output."""
    if not isinstance(reconstruction, SolverOutput):
        raise TypeError(
            "Expected SolverOutput from inverse solver. "
            f"Received {type(reconstruction).__name__}."
        )

    conductivity_field = reconstruction.conductivity
    if hasattr(conductivity_field, "x") and hasattr(conductivity_field.x, "array"):
        conductivity_values = function_get_array(conductivity_field).copy()
    elif isinstance(conductivity_field, np.ndarray):
        conductivity_values = conductivity_field.copy()
    else:
        raise TypeError(
            "SolverOutput.conductivity must be a DOLFINx Function or numpy array."
        )

    conductivity_image = EITImage(elem_data=conductivity_values, fwd_model=fwd_model)
    return (
        conductivity_image,
        conductivity_values,
        reconstruction.residual_history,
        reconstruction.sigma_change_history,
    )


def compute_residuals(
    measured_vector: np.ndarray,
    simulated_vector: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """Compute residual vector and basic metrics."""

    residual_vector = simulated_vector - measured_vector
    l2_error = float(np.linalg.norm(residual_vector))
    rel_error = float(l2_error / (np.linalg.norm(measured_vector) + 1e-12))
    mse = float(np.mean(residual_vector**2))
    return residual_vector, l2_error, rel_error, mse


ResidualComputer = Callable[
    [np.ndarray, np.ndarray], tuple[np.ndarray, float, float, float]
]


def merge_workflow_metadata(*parts: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge workflow metadata maps with later maps taking precedence."""

    merged: dict[str, Any] = {}
    for part in parts:
        if part:
            merged.update(part)
    return merged


def build_reconstruction_result(
    *,
    mode: str,
    conductivity_values: np.ndarray,
    conductivity_image: EITImage,
    measured_vector: np.ndarray,
    simulated_vector: np.ndarray,
    residual_history: Sequence[float] | None,
    sigma_change_history: Sequence[float] | None,
    metadata: Mapping[str, Any] | None = None,
    residual_fn: ResidualComputer = compute_residuals,
) -> ReconstructionResult:
    """Build a workflow result from already-resolved vectors and metadata."""

    residual_vector, _, _, _ = residual_fn(measured_vector, simulated_vector)
    return ReconstructionResult(
        mode=mode,
        conductivity=conductivity_values,
        conductivity_image=conductivity_image,
        measured=measured_vector,
        simulated=simulated_vector,
        residual=residual_vector,
        residual_history=residual_history,
        sigma_change_history=sigma_change_history,
        metadata=dict(metadata or {}),
    )
