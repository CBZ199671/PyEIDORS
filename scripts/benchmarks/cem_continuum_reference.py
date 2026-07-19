#!/usr/bin/env python3
"""Independent continuum CEM reference for a homogeneous circular domain.

The implementation uses the analytic Neumann-to-Dirichlet map of a disk.
It therefore never assembles or consumes an interior FEM matrix.  Electrode
flux is resolved on a periodic midpoint grid, and four doubled resolutions
provide an empirical-order Richardson extrapolation and uncertainty estimate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


DEFAULT_N_THETA_LEVELS = (5120, 10240, 20480, 40960)
DEFAULT_MAX_EXTRAPOLATION_DISAGREEMENT = 5e-6


@dataclass(frozen=True)
class ContinuumGeometry:
    """Physical geometry shared by the FEM sequence and continuum reference."""

    radius: float = 1.0
    n_electrodes: int = 16
    electrode_coverage: float = 0.7
    first_electrode_center: float = math.pi / 2.0

    @property
    def electrode_angle(self) -> float:
        return 2.0 * math.pi * self.electrode_coverage / self.n_electrodes

    @property
    def electrode_arc_length(self) -> float:
        return self.radius * self.electrode_angle

    @property
    def centers(self) -> np.ndarray:
        return self.first_electrode_center + np.arange(
            self.n_electrodes, dtype=np.float64
        ) * (2.0 * math.pi / self.n_electrodes)


@dataclass(frozen=True)
class ContinuumLevelResult:
    """One boundary-grid solution and its directly evaluated CEM residuals."""

    n_theta: int
    voltages: np.ndarray
    linear_relative_residual: float
    current_relative_residual: float
    robin_relative_residual: float
    gauge_relative_residual: float
    gmres_info: int
    gmres_iterations: int
    active_flux_dofs: int

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("voltages")
        return payload


def continuum_current_patterns(
    *,
    n_electrodes: int = 16,
    drive_skip: int = 1,
) -> np.ndarray:
    """Return one rotational family of integer, zero-sum current patterns."""

    count = int(n_electrodes)
    skip = int(drive_skip)
    if count < 2 or skip <= 0 or skip >= count:
        raise ValueError("drive_skip must be between 1 and n_electrodes - 1")
    currents = np.zeros((count, count), dtype=np.float64)
    columns = np.arange(count, dtype=np.int64)
    currents[columns, columns] = 1.0
    currents[(columns + skip) % count, columns] = -1.0
    if not np.array_equal(np.sum(currents, axis=0), np.zeros(count)):
        raise RuntimeError("continuum current patterns are not exactly zero-sum")
    return currents


def disk_ntd_apply(
    boundary_flux: np.ndarray,
    *,
    conductivity: float,
    radius: float,
) -> np.ndarray:
    r"""Apply the zero-mean disk Neumann-to-Dirichlet map.

    If ``q_n`` is the Fourier coefficient of outward current density, the
    boundary voltage coefficient is ``R*q_n/(sigma*abs(n))`` for ``n != 0``.
    """

    flux = np.asarray(boundary_flux, dtype=np.float64)
    if flux.ndim != 1 or flux.size < 4:
        raise ValueError("boundary_flux must be a one-dimensional periodic grid")
    sigma = float(conductivity)
    disk_radius = float(radius)
    if not (math.isfinite(sigma) and sigma > 0.0):
        raise ValueError("conductivity must be finite and positive")
    if not (math.isfinite(disk_radius) and disk_radius > 0.0):
        raise ValueError("radius must be finite and positive")
    frequencies = np.fft.fftfreq(flux.size, d=1.0 / flux.size)
    multiplier = np.zeros(flux.size, dtype=np.float64)
    nonzero = frequencies != 0.0
    multiplier[nonzero] = disk_radius / (sigma * np.abs(frequencies[nonzero]))
    potential = np.fft.ifft(np.fft.fft(flux) * multiplier)
    return np.asarray(potential.real, dtype=np.float64)


def _electrode_groups(
    n_theta: int,
    geometry: ContinuumGeometry,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    count = int(n_theta)
    if count <= 0 or count % (2 * geometry.n_electrodes) != 0:
        raise ValueError("n_theta must be positive and divisible by 2*n_electrodes")
    theta = (np.arange(count, dtype=np.float64) + 0.5) * (2.0 * math.pi / count)
    groups: list[np.ndarray] = []
    expected = int(round(count * geometry.electrode_coverage / geometry.n_electrodes))
    for center in geometry.centers:
        distance = (theta - center + math.pi) % (2.0 * math.pi) - math.pi
        indices = np.flatnonzero(np.abs(distance) < geometry.electrode_angle / 2.0)
        if indices.size != expected:
            raise ValueError(
                "n_theta does not align electrode endpoints with midpoint cells: "
                f"got {indices.size}, expected {expected}"
            )
        groups.append(indices)
    active = np.concatenate(groups)
    if np.unique(active).size != active.size:
        raise RuntimeError("continuum electrode midpoint groups overlap")
    return active, tuple(groups)


def _solve_base_drive(
    *,
    conductivity: float,
    contact_impedance: float,
    drive_skip: int,
    n_theta: int,
    geometry: ContinuumGeometry,
    gmres_rtol: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    sigma = float(conductivity)
    impedance = float(contact_impedance)
    if not (math.isfinite(impedance) and impedance > 0.0):
        raise ValueError("contact_impedance must be finite and positive")
    active, groups = _electrode_groups(int(n_theta), geometry)
    active_count = int(active.size)
    robin_scale = impedance

    def operator_action(active_flux: np.ndarray) -> np.ndarray:
        full_flux = np.zeros(int(n_theta), dtype=np.float64)
        full_flux[active] = np.asarray(active_flux, dtype=np.float64)
        boundary_voltage = disk_ntd_apply(
            full_flux,
            conductivity=sigma,
            radius=geometry.radius,
        )
        robin_trace = boundary_voltage + impedance * full_flux
        output = np.empty(active_count, dtype=np.float64)
        offset = 0
        for indices in groups:
            width = int(indices.size)
            electrode_flux = full_flux[indices]
            electrode_trace = robin_trace[indices]
            output[offset] = float(np.mean(electrode_flux))
            output[offset + 1 : offset + width] = (
                electrode_trace[1:] - electrode_trace[0]
            ) / robin_scale
            offset += width
        return output

    current = continuum_current_patterns(
        n_electrodes=geometry.n_electrodes,
        drive_skip=drive_skip,
    )[:, 0]
    rhs = np.zeros(active_count, dtype=np.float64)
    offset = 0
    for electrode, indices in enumerate(groups):
        rhs[offset] = current[electrode] / geometry.electrode_arc_length
        offset += int(indices.size)
    operator = LinearOperator(
        (active_count, active_count),
        matvec=operator_action,
        dtype=np.float64,
    )
    iteration_count = 0

    def count_iteration(_residual: float) -> None:
        nonlocal iteration_count
        iteration_count += 1

    active_flux, info = gmres(
        operator,
        rhs,
        rtol=float(gmres_rtol),
        atol=0.0,
        restart=200,
        maxiter=100,
        callback=count_iteration,
        callback_type="pr_norm",
    )
    full_flux = np.zeros(int(n_theta), dtype=np.float64)
    full_flux[active] = active_flux
    zero_mean_boundary_voltage = disk_ntd_apply(
        full_flux,
        conductivity=sigma,
        radius=geometry.radius,
    )
    robin_trace = zero_mean_boundary_voltage + impedance * full_flux
    raw_electrode_voltage = np.asarray(
        [np.mean(robin_trace[indices]) for indices in groups],
        dtype=np.float64,
    )
    electrode_voltage = raw_electrode_voltage - np.mean(raw_electrode_voltage)

    linear_residual = operator_action(active_flux) - rhs
    linear_relative_residual = float(
        np.linalg.norm(linear_residual)
        / max(np.linalg.norm(rhs), np.finfo(np.float64).eps)
    )
    recovered_current = np.asarray(
        [
            geometry.electrode_arc_length * np.mean(full_flux[indices])
            for indices in groups
        ]
    )
    current_relative_residual = float(
        np.linalg.norm(recovered_current - current)
        / max(np.linalg.norm(current), np.finfo(np.float64).eps)
    )
    robin_absolute_residual = max(
        float(np.max(np.abs(robin_trace[indices] - np.mean(robin_trace[indices]))))
        for indices in groups
    )
    robin_denominator = max(
        float(np.max(np.abs(raw_electrode_voltage))),
        impedance * float(np.max(np.abs(full_flux))),
        np.finfo(np.float64).eps,
    )
    robin_relative_residual = robin_absolute_residual / robin_denominator
    gauge_relative_residual = float(
        abs(np.sum(electrode_voltage))
        / max(np.linalg.norm(electrode_voltage), np.finfo(np.float64).eps)
    )
    diagnostics = {
        "linear_relative_residual": linear_relative_residual,
        "current_relative_residual": current_relative_residual,
        "robin_relative_residual": robin_relative_residual,
        "gauge_relative_residual": gauge_relative_residual,
        "gmres_info": int(info),
        "gmres_iterations": int(iteration_count),
        "active_flux_dofs": active_count,
    }
    return electrode_voltage, diagnostics


def solve_continuum_level(
    *,
    conductivity: float,
    contact_impedance: float,
    drive_skip: int,
    n_theta: int,
    geometry: ContinuumGeometry | None = None,
    gmres_rtol: float = 2e-12,
) -> ContinuumLevelResult:
    """Solve one continuum boundary level and recover all rotational drives."""

    domain = geometry or ContinuumGeometry()
    base_voltage, diagnostics = _solve_base_drive(
        conductivity=conductivity,
        contact_impedance=contact_impedance,
        drive_skip=drive_skip,
        n_theta=int(n_theta),
        geometry=domain,
        gmres_rtol=float(gmres_rtol),
    )
    voltage_matrix = np.column_stack(
        [np.roll(base_voltage, column) for column in range(domain.n_electrodes)]
    )
    return ContinuumLevelResult(
        n_theta=int(n_theta),
        voltages=np.asarray(voltage_matrix, dtype=np.float64),
        **diagnostics,
    )


def _positive_order(large: float, small: float) -> float:
    if not (math.isfinite(large) and math.isfinite(small) and large > small > 0.0):
        return math.nan
    return float(math.log(large / small, 2.0))


def _richardson(fine: np.ndarray, coarse: np.ndarray, order: float) -> np.ndarray:
    denominator = 2.0 ** float(order) - 1.0
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("Richardson order must be finite and positive")
    return np.asarray(fine + (fine - coarse) / denominator, dtype=np.float64)


def certify_continuum_reference(
    *,
    conductivity: float,
    contact_impedance: float,
    drive_skip: int,
    n_theta_levels: Iterable[int] = DEFAULT_N_THETA_LEVELS,
    geometry: ContinuumGeometry | None = None,
    residual_tolerance: float = 1e-10,
    max_extrapolation_disagreement: float = DEFAULT_MAX_EXTRAPOLATION_DISAGREEMENT,
) -> dict[str, Any]:
    """Build and certify a four-level Richardson continuum reference."""

    levels = tuple(int(value) for value in n_theta_levels)
    if len(levels) != 4 or any(
        fine != 2 * coarse for coarse, fine in zip(levels[:-1], levels[1:], strict=True)
    ):
        raise ValueError("continuum certification requires four doubled n_theta levels")
    domain = geometry or ContinuumGeometry()
    solutions = [
        solve_continuum_level(
            conductivity=conductivity,
            contact_impedance=contact_impedance,
            drive_skip=drive_skip,
            n_theta=level,
            geometry=domain,
        )
        for level in levels
    ]
    voltages = [result.voltages for result in solutions]
    differences = [
        float(np.linalg.norm(fine - coarse))
        for coarse, fine in zip(voltages[:-1], voltages[1:], strict=True)
    ]
    order_previous = _positive_order(differences[0], differences[1])
    order_last = _positive_order(differences[1], differences[2])
    extrapolated_previous = _richardson(voltages[2], voltages[1], order_previous)
    extrapolated_last = _richardson(voltages[3], voltages[2], order_last)
    disagreement_absolute = float(
        np.linalg.norm(extrapolated_last - extrapolated_previous)
    )
    reference_norm = max(
        float(np.linalg.norm(extrapolated_last)), np.finfo(np.float64).eps
    )
    disagreement_relative = disagreement_absolute / reference_norm
    residual_fields = (
        "linear_relative_residual",
        "current_relative_residual",
        "robin_relative_residual",
        "gauge_relative_residual",
    )
    max_residual = max(
        float(getattr(result, field))
        for result in solutions
        for field in residual_fields
    )
    certified = bool(
        all(result.gmres_info == 0 for result in solutions)
        and max_residual <= float(residual_tolerance)
        and math.isfinite(order_previous)
        and math.isfinite(order_last)
        and order_previous > 0.0
        and order_last > 0.0
        and disagreement_relative <= float(max_extrapolation_disagreement)
    )
    return {
        "schema": "cem-continuum-reference-v1",
        "certified": certified,
        "method": "analytic disk NtD + midpoint Fourier-Nystrom + empirical-order Richardson",
        "uses_interior_fem_mesh": False,
        "uses_candidate_solver_matrix": False,
        "geometry": asdict(domain),
        "conductivity": float(conductivity),
        "contact_impedance": float(contact_impedance),
        "drive_skip": int(drive_skip),
        "n_theta_levels": list(levels),
        "level_diagnostics": [result.metadata() for result in solutions],
        "successive_voltage_differences": differences,
        "observed_order_previous": order_previous,
        "observed_order_last": order_last,
        "relative_extrapolation_disagreement": disagreement_relative,
        "absolute_extrapolation_disagreement": disagreement_absolute,
        "reference_relative_uncertainty": disagreement_relative,
        "reference_absolute_uncertainty": disagreement_absolute,
        "max_constraint_relative_residual": max_residual,
        "residual_tolerance": float(residual_tolerance),
        "max_extrapolation_disagreement": float(max_extrapolation_disagreement),
        "reference_voltages": extrapolated_last.tolist(),
        "previous_extrapolated_voltages": extrapolated_previous.tolist(),
        "finest_raw_voltages": voltages[-1].tolist(),
    }


__all__ = [
    "DEFAULT_MAX_EXTRAPOLATION_DISAGREEMENT",
    "DEFAULT_N_THETA_LEVELS",
    "ContinuumGeometry",
    "ContinuumLevelResult",
    "certify_continuum_reference",
    "continuum_current_patterns",
    "disk_ntd_apply",
    "solve_continuum_level",
]
