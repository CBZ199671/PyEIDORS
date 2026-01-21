"""Imaging workflow common utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Dict, Any

import numpy as np

from ...data.structures import EITImage


@dataclass
class ReconstructionResult:
    """Unified output encapsulation for difference/absolute imaging."""

    mode: str
    conductivity: np.ndarray
    conductivity_image: EITImage
    measured: np.ndarray
    simulated: np.ndarray
    residual: np.ndarray
    residual_history: Optional[Sequence[float]] = None
    sigma_change_history: Optional[Sequence[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

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
        return float(np.mean(self.residual ** 2))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility with legacy scripts."""

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
    reconstruction: Any,
    fwd_model,
) -> Tuple[EITImage, np.ndarray, Optional[Sequence[float]], Optional[Sequence[float]]]:
    """Extract conductivity image and history from solver output."""

    if isinstance(reconstruction, dict):
        conductivity_field = reconstruction.get("conductivity")
        residual_history = reconstruction.get("residual_history")
        sigma_history = reconstruction.get("sigma_change_history")
    else:
        conductivity_field = getattr(reconstruction, "elem_data", reconstruction)
        residual_history = None
        sigma_history = None

    if hasattr(conductivity_field, "vector"):
        conductivity_values = conductivity_field.vector()[:]
        conductivity_image = EITImage(elem_data=conductivity_values, fwd_model=fwd_model)
    elif isinstance(conductivity_field, np.ndarray):
        conductivity_values = conductivity_field
        conductivity_image = EITImage(elem_data=conductivity_values, fwd_model=fwd_model)
    else:
        raise TypeError("Unrecognized reconstruction result type: expected FEniCS Function or numpy array")

    return conductivity_image, conductivity_values, residual_history, sigma_history


def compute_residuals(
    measured_vector: np.ndarray,
    simulated_vector: np.ndarray,
) -> Tuple[np.ndarray, float, float, float]:
    """Compute residual vector and basic metrics."""

    residual_vector = simulated_vector - measured_vector
    l2_error = float(np.linalg.norm(residual_vector))
    rel_error = float(l2_error / (np.linalg.norm(measured_vector) + 1e-12))
    mse = float(np.mean(residual_vector ** 2))
    return residual_vector, l2_error, rel_error, mse
