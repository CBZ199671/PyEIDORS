"""End-to-end voltage and conductivity digit metrics.

The default path uses a deterministic linear EIT surrogate so the precision
pipeline can be tested without invoking heavy FEM solves. Real PyEIDORS
forward/inverse hooks can replace the surrogate in the same data flow.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .adc_quantization import (
    ADCInjectionConfig,
    effective_digits_from_rmse,
    ideal_decimal_digits,
    inject_adc_measurement,
    rmse,
)


@dataclass(frozen=True)
class EITDigitSummary:
    """Summary metrics for one ADC bit in an end-to-end EIT run."""

    bit: int
    ideal_decimal_digits: float
    voltage_rmse: float
    voltage_effective_digits: float
    sigma_rmse: float
    sigma_effective_digits: float
    hypothesis_delta_digits: float

    def as_csv_row(self) -> dict[str, float | int]:
        return {
            "bit": self.bit,
            "ideal_decimal_digits": self.ideal_decimal_digits,
            "voltage_rmse": self.voltage_rmse,
            "voltage_effective_digits": self.voltage_effective_digits,
            "sigma_rmse": self.sigma_rmse,
            "sigma_effective_digits": self.sigma_effective_digits,
            "hypothesis_delta_digits": self.hypothesis_delta_digits,
        }


def _as_float_vector(values: Iterable[float] | np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _as_float_matrix(
    values: Iterable[Iterable[float]] | np.ndarray, *, name: str
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def default_sigma_true(n_parameters: int = 8) -> np.ndarray:
    """Return a deterministic positive conductivity vector."""

    if n_parameters <= 0:
        raise ValueError("n_parameters must be positive")
    sigma = np.linspace(0.85, 1.15, int(n_parameters), dtype=float)
    if n_parameters >= 4:
        sigma[1] += 0.12
        sigma[-2] -= 0.08
    return sigma


def build_surrogate_sensitivity(
    *,
    n_measurements: int = 16,
    n_parameters: int = 8,
    seed: int = 20260422,
) -> np.ndarray:
    """Build a deterministic full-rank linearized EIT sensitivity matrix."""

    if n_measurements <= 0:
        raise ValueError("n_measurements must be positive")
    if n_parameters <= 0:
        raise ValueError("n_parameters must be positive")
    if n_measurements < n_parameters:
        raise ValueError("n_measurements must be >= n_parameters")

    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(int(n_measurements), int(n_parameters)))
    q_matrix, _ = np.linalg.qr(matrix)
    sensitivity = q_matrix[:, : int(n_parameters)].copy()
    column_scale = np.linspace(0.8, 1.2, int(n_parameters), dtype=float)
    return sensitivity * column_scale


def forward_surrogate(
    sigma: Iterable[float] | np.ndarray,
    sensitivity: Iterable[Iterable[float]] | np.ndarray,
) -> np.ndarray:
    """Compute boundary voltages from a linear surrogate forward model."""

    sigma_vec = _as_float_vector(sigma, name="sigma")
    sens = _as_float_matrix(sensitivity, name="sensitivity")
    if sens.shape[1] != sigma_vec.size:
        raise ValueError("sensitivity column count must match sigma size")
    return sens @ sigma_vec


def inverse_surrogate(
    voltages: Iterable[float] | np.ndarray,
    sensitivity: Iterable[Iterable[float]] | np.ndarray,
    *,
    ridge: float = 1e-8,
) -> np.ndarray:
    """Reconstruct conductivity with ridge-regularized least squares."""

    voltage_vec = _as_float_vector(voltages, name="voltages")
    sens = _as_float_matrix(sensitivity, name="sensitivity")
    if sens.shape[0] != voltage_vec.size:
        raise ValueError("sensitivity row count must match voltage size")
    ridge_value = float(ridge)
    if not math.isfinite(ridge_value) or ridge_value < 0.0:
        raise ValueError("ridge must be non-negative and finite")

    normal = sens.T @ sens
    rhs = sens.T @ voltage_vec
    if ridge_value > 0.0:
        normal = normal + ridge_value * np.eye(normal.shape[0], dtype=float)
    return np.linalg.solve(normal, rhs)


def _hypothesis_delta(sigma_digits: float, voltage_digits: float) -> float:
    if math.isfinite(sigma_digits) and math.isfinite(voltage_digits):
        return sigma_digits - voltage_digits
    return math.nan


def summarize_eit_digit_run(
    *,
    bit: int,
    sigma_true: Iterable[float] | np.ndarray,
    sensitivity: Iterable[Iterable[float]] | np.ndarray,
    full_scale_range: float,
    enob: float | None = None,
    noise_std: float = 0.0,
    noise_relative: float = 0.0,
    seed: int | None = 0,
    ridge: float = 1e-8,
) -> EITDigitSummary:
    """Run one end-to-end precision case and return digit metrics."""

    sigma_vec = _as_float_vector(sigma_true, name="sigma_true")
    sens = _as_float_matrix(sensitivity, name="sensitivity")
    v_true = forward_surrogate(sigma_vec, sens)
    v_adc = inject_adc_measurement(
        v_true,
        ADCInjectionConfig(
            bit=bit,
            full_scale_range=full_scale_range,
            enob=enob,
            noise_std=noise_std,
            noise_relative=noise_relative,
            seed=seed,
        ),
    )
    sigma_recon = inverse_surrogate(v_adc, sens, ridge=ridge)
    voltage_digits = effective_digits_from_rmse(v_true, v_adc)
    sigma_digits = effective_digits_from_rmse(sigma_vec, sigma_recon)
    return EITDigitSummary(
        bit=int(bit),
        ideal_decimal_digits=ideal_decimal_digits(bit),
        voltage_rmse=rmse(v_true, v_adc),
        voltage_effective_digits=voltage_digits,
        sigma_rmse=rmse(sigma_vec, sigma_recon),
        sigma_effective_digits=sigma_digits,
        hypothesis_delta_digits=_hypothesis_delta(sigma_digits, voltage_digits),
    )


def summarize_eit_digit_sweep(
    *,
    bits: Iterable[int],
    full_scale_range: float,
    enob: float | None = None,
    noise_std: float = 0.0,
    noise_relative: float = 0.0,
    seed: int | None = 0,
    ridge: float = 1e-8,
    n_measurements: int = 16,
    n_parameters: int = 8,
    model_seed: int = 20260422,
    sigma_true: Iterable[float] | np.ndarray | None = None,
    sensitivity: Iterable[Iterable[float]] | np.ndarray | None = None,
) -> list[EITDigitSummary]:
    """Run an ADC bit sweep through the surrogate EIT end-to-end pipeline."""

    if sigma_true is None:
        sigma_vec = default_sigma_true(n_parameters)
    else:
        sigma_vec = _as_float_vector(sigma_true, name="sigma_true")
    if sensitivity is None:
        sens = build_surrogate_sensitivity(
            n_measurements=n_measurements,
            n_parameters=sigma_vec.size,
            seed=model_seed,
        )
    else:
        sens = _as_float_matrix(sensitivity, name="sensitivity")

    return [
        summarize_eit_digit_run(
            bit=int(bit),
            sigma_true=sigma_vec,
            sensitivity=sens,
            full_scale_range=full_scale_range,
            enob=enob,
            noise_std=noise_std,
            noise_relative=noise_relative,
            seed=seed,
            ridge=ridge,
        )
        for bit in bits
    ]
