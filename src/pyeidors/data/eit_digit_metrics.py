"""End-to-end voltage and conductivity digit metrics.

The default path uses a deterministic linear EIT surrogate so the precision
pipeline can be tested quickly. T12 also exposes a small PyEIDORS FEM forward
model path and the production reconstruction-matrix inverse helpers.
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


ADJACENT_PATTERN = "{ad}"


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


@dataclass(frozen=True)
class EITLinearizedModel:
    """Linearized EIT model inputs for the ADC precision pipeline."""

    sigma_true: np.ndarray
    sigma_reference: np.ndarray
    voltage_true: np.ndarray
    voltage_reference: np.ndarray
    sensitivity: np.ndarray
    label: str
    n_elec: int | None = None
    stim_pattern: str = ""
    meas_pattern: str = ""
    n_measurements: int = 0


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


def adjacent_measurement_count(n_elec: int) -> int:
    """Expected `{ad}`/`{ad}` frame length for adjacent EIT patterns."""

    count = int(n_elec)
    if count < 4:
        raise ValueError("n_elec must be >= 4 for adjacent measurement count")
    return count * (count - 3)


def build_surrogate_linearized_model(
    *,
    n_measurements: int = 16,
    n_parameters: int = 8,
    seed: int = 20260422,
    sigma_true: Iterable[float] | np.ndarray | None = None,
    sensitivity: Iterable[Iterable[float]] | np.ndarray | None = None,
) -> EITLinearizedModel:
    """Build the deterministic linear-surrogate model used by quick smokes."""

    if sigma_true is None:
        sigma_vec = default_sigma_true(n_parameters)
    else:
        sigma_vec = _as_float_vector(sigma_true, name="sigma_true")
    if sensitivity is None:
        sens = build_surrogate_sensitivity(
            n_measurements=n_measurements,
            n_parameters=sigma_vec.size,
            seed=seed,
        )
    else:
        sens = _as_float_matrix(sensitivity, name="sensitivity")
    if sens.shape[1] != sigma_vec.size:
        raise ValueError("sensitivity column count must match sigma_true size")

    return EITLinearizedModel(
        sigma_true=sigma_vec,
        sigma_reference=np.zeros_like(sigma_vec),
        voltage_true=forward_surrogate(sigma_vec, sens),
        voltage_reference=np.zeros(sens.shape[0], dtype=float),
        sensitivity=sens,
        label="linear-surrogate",
        n_measurements=int(sens.shape[0]),
    )


def _square_edge_parameter(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    eps = 1e-10
    t = np.zeros_like(x)
    left = np.isclose(x, 0.0, atol=eps)
    top = (~left) & np.isclose(y, 1.0, atol=eps)
    right = (~left) & (~top) & np.isclose(x, 1.0, atol=eps)
    bottom = (~left) & (~top) & (~right) & np.isclose(y, 0.0, atol=eps)
    t[left] = y[left]
    t[top] = 1.0 + x[top]
    t[right] = 2.0 + (1.0 - y[right])
    t[bottom] = 3.0 + (1.0 - x[bottom])
    return np.clip(t, 0.0, 4.0 - eps)


def _create_pyeidors_square_mesh(*, n_elec: int, grid: int):
    if n_elec <= 0:
        raise ValueError("n_elec must be positive")
    if grid <= 0:
        raise ValueError("grid must be positive")
    if n_elec > 4 * grid:
        raise ValueError("n_elec must be <= 4 * grid for unit-square tagging")

    from dolfinx import mesh as dmesh
    from mpi4py import MPI

    from pyeidors.femx import build_eit_mesh

    square_mesh = dmesh.create_unit_square(MPI.COMM_WORLD, int(grid), int(grid))
    fdim = square_mesh.topology.dim - 1
    boundary_facets = dmesh.locate_entities_boundary(
        square_mesh,
        fdim,
        lambda x: np.full(x.shape[1], True, dtype=bool),
    ).astype(np.int32)
    square_mesh.topology.create_connectivity(fdim, 0)
    f2v = square_mesh.topology.connectivity(fdim, 0)
    if f2v is None:
        raise RuntimeError("failed to build facet-to-vertex connectivity")

    coords = square_mesh.geometry.x[:, :2]
    centroids = np.zeros((boundary_facets.size, 2), dtype=float)
    for idx, facet in enumerate(boundary_facets):
        centroids[idx, :] = coords[f2v.links(int(facet))].mean(axis=0)

    tags = (
        np.floor(_square_edge_parameter(centroids) / (4.0 / float(n_elec))).astype(
            np.int32
        )
        + 2
    ).astype(np.int32)
    order = np.argsort(boundary_facets)
    facet_tags = dmesh.meshtags(
        square_mesh,
        fdim,
        boundary_facets[order],
        tags[order],
    )
    association = {f"electrode_{idx + 1}": idx + 2 for idx in range(n_elec)}
    return build_eit_mesh(
        square_mesh,
        facet_tags=facet_tags,
        association_table=association,
        radius=1.0,
    )


def _pyeidors_forward_vector(fwd_model, sigma: np.ndarray) -> np.ndarray:
    from pyeidors.data.structures import EITImage

    data, _ = fwd_model.fwd_solve(EITImage(elem_data=sigma, fwd_model=fwd_model))
    return _as_float_vector(data.meas, name="fem voltages")


def build_pyeidors_fem_linearized_model(
    *,
    n_elec: int = 8,
    grid: int = 2,
    expected_measurements: int | None = None,
) -> EITLinearizedModel:
    """Build a small real PyEIDORS FEM forward/Jacobian model for T12 smokes."""

    from dolfinx import fem

    from pyeidors.data.structures import PatternConfig
    from pyeidors.forward.eit_forward_model import EITForwardModel
    from pyeidors.inverse.jacobian.adjoint_jacobian import EidorsStyleAdjointJacobian

    mesh = _create_pyeidors_square_mesh(n_elec=int(n_elec), grid=int(grid))
    pattern = PatternConfig(
        n_elec=int(n_elec),
        stim_pattern=ADJACENT_PATTERN,
        meas_pattern=ADJACENT_PATTERN,
        drive_mode="total_current",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
    )
    fwd_model = EITForwardModel(
        n_elec=int(n_elec),
        pattern_config=pattern,
        z=np.full(int(n_elec), 1e-5, dtype=float),
        mesh=mesh,
        linear_backend="scipy",
    )

    sigma_ref_fun = fem.Function(fwd_model.V_sigma)
    sigma_reference = np.ones(sigma_ref_fun.x.array.size, dtype=float)
    sigma_ref_fun.x.array[:] = sigma_reference
    sigma_true = default_sigma_true(sigma_reference.size)

    voltage_reference = _pyeidors_forward_vector(fwd_model, sigma_reference)
    voltage_true = _pyeidors_forward_vector(fwd_model, sigma_true)
    actual_measurements = int(voltage_reference.size)
    if expected_measurements is not None and actual_measurements != int(
        expected_measurements
    ):
        raise RuntimeError(
            "PyEIDORS FEM measurement count mismatch: "
            f"expected {int(expected_measurements)}, got {actual_measurements}"
        )
    sensitivity = np.asarray(
        EidorsStyleAdjointJacobian(fwd_model).calculate(sigma_ref_fun),
        dtype=float,
    )
    if sensitivity.shape != (voltage_reference.size, sigma_reference.size):
        raise RuntimeError(
            "PyEIDORS FEM sensitivity shape does not match voltage/sigma sizes"
        )
    return EITLinearizedModel(
        sigma_true=sigma_true,
        sigma_reference=sigma_reference,
        voltage_true=voltage_true,
        voltage_reference=voltage_reference,
        sensitivity=sensitivity,
        label="pyeidors-fem",
        n_elec=int(n_elec),
        stim_pattern=ADJACENT_PATTERN,
        meas_pattern=ADJACENT_PATTERN,
        n_measurements=actual_measurements,
    )


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


def inverse_pyeidors_rm(
    voltages: Iterable[float] | np.ndarray,
    sensitivity: Iterable[Iterable[float]] | np.ndarray,
    *,
    lambda_: float = 1e-8,
    mode: str = "tikhonov",
    form: str = "param",
) -> np.ndarray:
    """Reconstruct conductivity with PyEIDORS one-step RM helpers."""

    voltage_vec = _as_float_vector(voltages, name="voltages")
    sens = _as_float_matrix(sensitivity, name="sensitivity")
    if sens.shape[0] != voltage_vec.size:
        raise ValueError("sensitivity row count must match voltage size")

    # Lazy import keeps pyeidors.data import light; this is the T12 integration
    # point with the production RM inverse module.
    from pyeidors.inverse.reconstruction_matrix import (
        build_one_step_rm,
        reconstruct_difference,
    )

    rm = build_one_step_rm(
        sens,
        lambda_=float(lambda_),
        mode=mode,
        form=form,
    )
    return np.asarray(
        reconstruct_difference(rm, voltage_vec, normalize=False), dtype=float
    )


def _inverse(
    voltages: np.ndarray,
    sensitivity: np.ndarray,
    *,
    inverse_backend: str,
    ridge: float,
    rm_mode: str,
    rm_form: str,
) -> np.ndarray:
    backend = str(inverse_backend).strip().lower()
    if backend in {"pyeidors-rm", "rm"}:
        return inverse_pyeidors_rm(
            voltages,
            sensitivity,
            lambda_=ridge,
            mode=rm_mode,
            form=rm_form,
        )
    if backend in {"least-squares", "surrogate"}:
        return inverse_surrogate(voltages, sensitivity, ridge=ridge)
    raise ValueError("inverse_backend must be one of: pyeidors-rm, least-squares")


def reconstruct_linearized_sigma(
    *,
    model: EITLinearizedModel,
    voltages: Iterable[float] | np.ndarray,
    ridge: float = 1e-8,
    inverse_backend: str = "pyeidors-rm",
    rm_mode: str = "tikhonov",
    rm_form: str = "param",
) -> np.ndarray:
    """Reconstruct absolute conductivity from a linearized model voltage vector."""

    sigma_ref = _as_float_vector(model.sigma_reference, name="model.sigma_reference")
    voltage_vec = _as_float_vector(voltages, name="voltages")
    v_ref = _as_float_vector(model.voltage_reference, name="model.voltage_reference")
    sens = _as_float_matrix(model.sensitivity, name="model.sensitivity")
    if voltage_vec.size != v_ref.size:
        raise ValueError("voltages size must match model voltage_reference size")
    if sens.shape != (v_ref.size, sigma_ref.size):
        raise ValueError("model sensitivity shape must match voltage/sigma sizes")

    sigma_delta = _inverse(
        voltage_vec - v_ref,
        sens,
        inverse_backend=inverse_backend,
        ridge=ridge,
        rm_mode=rm_mode,
        rm_form=rm_form,
    )
    return sigma_ref + sigma_delta


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
    inverse_backend: str = "pyeidors-rm",
    rm_mode: str = "tikhonov",
    rm_form: str = "param",
) -> EITDigitSummary:
    """Run one end-to-end precision case and return digit metrics."""

    model = build_surrogate_linearized_model(
        sigma_true=sigma_true,
        sensitivity=sensitivity,
    )
    return _summarize_linearized_digit_run(
        bit=bit,
        model=model,
        full_scale_range=full_scale_range,
        enob=enob,
        noise_std=noise_std,
        noise_relative=noise_relative,
        seed=seed,
        ridge=ridge,
        inverse_backend=inverse_backend,
        rm_mode=rm_mode,
        rm_form=rm_form,
    )


def _summarize_linearized_digit_run(
    *,
    bit: int,
    model: EITLinearizedModel,
    full_scale_range: float,
    enob: float | None = None,
    noise_std: float = 0.0,
    noise_relative: float = 0.0,
    seed: int | None = 0,
    ridge: float = 1e-8,
    inverse_backend: str = "pyeidors-rm",
    rm_mode: str = "tikhonov",
    rm_form: str = "param",
) -> EITDigitSummary:
    sigma_vec = _as_float_vector(model.sigma_true, name="model.sigma_true")
    sigma_ref = _as_float_vector(model.sigma_reference, name="model.sigma_reference")
    v_true = _as_float_vector(model.voltage_true, name="model.voltage_true")
    v_ref = _as_float_vector(model.voltage_reference, name="model.voltage_reference")
    sens = _as_float_matrix(model.sensitivity, name="model.sensitivity")
    if sigma_ref.size != sigma_vec.size:
        raise ValueError("model sigma_reference size must match sigma_true size")
    if v_ref.size != v_true.size:
        raise ValueError("model voltage_reference size must match voltage_true size")
    if sens.shape != (v_true.size, sigma_vec.size):
        raise ValueError("model sensitivity shape must match voltage/sigma sizes")

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
    voltage_delta = v_adc - v_ref
    sigma_recon = _inverse(
        voltage_delta,
        sens,
        inverse_backend=inverse_backend,
        ridge=ridge,
        rm_mode=rm_mode,
        rm_form=rm_form,
    )
    sigma_recon = sigma_ref + sigma_recon
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
    inverse_backend: str = "pyeidors-rm",
    rm_mode: str = "tikhonov",
    rm_form: str = "param",
    n_measurements: int = 16,
    n_parameters: int = 8,
    model_seed: int = 20260422,
    sigma_true: Iterable[float] | np.ndarray | None = None,
    sensitivity: Iterable[Iterable[float]] | np.ndarray | None = None,
    forward_backend: str = "surrogate",
    fem_n_elec: int = 8,
    fem_grid: int = 2,
    expected_fem_measurements: int | None = None,
) -> list[EITDigitSummary]:
    """Run an ADC bit sweep through the EIT digit pipeline."""

    backend = str(forward_backend).strip().lower()
    if backend in {"surrogate", "linear-surrogate"}:
        model = build_surrogate_linearized_model(
            n_measurements=n_measurements,
            n_parameters=n_parameters,
            seed=model_seed,
            sigma_true=sigma_true,
            sensitivity=sensitivity,
        )
    elif backend in {"pyeidors-fem", "fem"}:
        if sigma_true is not None or sensitivity is not None:
            raise ValueError(
                "sigma_true and sensitivity overrides are only supported "
                "for forward_backend='surrogate'"
            )
        expected_measurements = (
            adjacent_measurement_count(fem_n_elec)
            if expected_fem_measurements is None
            else int(expected_fem_measurements)
        )
        model = build_pyeidors_fem_linearized_model(
            n_elec=fem_n_elec,
            grid=fem_grid,
            expected_measurements=expected_measurements,
        )
    else:
        raise ValueError("forward_backend must be one of: surrogate, pyeidors-fem")

    return [
        _summarize_linearized_digit_run(
            bit=int(bit),
            model=model,
            full_scale_range=full_scale_range,
            enob=enob,
            noise_std=noise_std,
            noise_relative=noise_relative,
            seed=seed,
            ridge=ridge,
            inverse_backend=inverse_backend,
            rm_mode=rm_mode,
            rm_form=rm_form,
        )
        for bit in bits
    ]
