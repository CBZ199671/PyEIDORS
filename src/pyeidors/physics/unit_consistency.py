"""Unit consistency guards for 2D EIT forward setups."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

import numpy as np

from pyeidors.utils.numeric_ops import all_finite_values

if TYPE_CHECKING:
    from ..forward.eit_forward_model import EITForwardModel


class UnitCheckLevel(str, Enum):
    """Severity level for unit consistency checks."""

    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


@dataclass(frozen=True)
class UnitCheckItem:
    """Single unit consistency check result."""

    name: str
    level: UnitCheckLevel
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnitCheckReport:
    """Aggregate report for unit consistency checks."""

    items: list[UnitCheckItem] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(item.level == UnitCheckLevel.ERROR for item in self.items)

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        for item in self.items:
            status = "PASS" if item.passed else "FAIL"
            lines.append(f"[{item.level}] {status} {item.name}: {item.message}")
        return lines


def _geometry_size_m(mesh, geometry_scale_to_m: float) -> tuple[np.ndarray, float]:
    coords = mesh.geometry.x[:, : mesh.geometry.dim]
    if coords.size == 0:
        raise ValueError("Mesh has no coordinates.")
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    extents_mesh = bbox_max - bbox_min
    extents_m = extents_mesh * float(geometry_scale_to_m)
    return extents_m, float(np.max(extents_m))


def run_unit_consistency_checks(
    fwd_model: "EITForwardModel",
    *,
    expected_domain_size_m: float | None = None,
    geometry_tolerance: float = 0.05,
    density_rel_tol: float = 1e-8,
) -> UnitCheckReport:
    """Run five unit consistency checks on a prepared forward model."""
    from .current_drive import validate_drive_config

    report = UnitCheckReport()
    cfg = fwd_model.pattern_manager.config
    mesh_tdim = fwd_model.mesh.topology.dim
    n_elec = int(fwd_model.n_elec)

    # 1) Drive config legality
    try:
        mode = validate_drive_config(
            drive_mode=cfg.drive_mode,
            drive_value=float(cfg.drive_value),
            geometry_scale_to_m=float(cfg.geometry_scale_to_m),
            mesh_tdim=mesh_tdim,
        )
        report.items.append(
            UnitCheckItem(
                name="drive_config_validity",
                level=UnitCheckLevel.INFO,
                passed=True,
                message="Drive config is physically valid.",
                details={"drive_mode": mode, "mesh_tdim": mesh_tdim},
            )
        )
    except Exception as exc:
        report.items.append(
            UnitCheckItem(
                name="drive_config_validity",
                level=UnitCheckLevel.ERROR,
                passed=False,
                message=str(exc),
            )
        )
        return report

    # 2) Geometry scale sanity
    try:
        extents_m, max_size_m = _geometry_size_m(
            fwd_model.mesh, float(cfg.geometry_scale_to_m)
        )
        finite_and_positive = bool(
            all_finite_values(extents_m)
            and float(np.min(extents_m, initial=np.inf)) > 0.0
        )
        if not finite_and_positive:
            report.items.append(
                UnitCheckItem(
                    name="geometry_scale_consistency",
                    level=UnitCheckLevel.ERROR,
                    passed=False,
                    message="Physical extents are non-finite or non-positive.",
                    details={"extents_m": extents_m.tolist()},
                )
            )
        elif expected_domain_size_m is not None:
            expected = float(expected_domain_size_m)
            rel = abs(max_size_m - expected) / max(expected, np.finfo(float).eps)
            if rel > geometry_tolerance:
                report.items.append(
                    UnitCheckItem(
                        name="geometry_scale_consistency",
                        level=UnitCheckLevel.ERROR,
                        passed=False,
                        message=(
                            "Physical domain size deviates from expected value "
                            f"(rel_err={rel:.3e} > {geometry_tolerance:.3e})."
                        ),
                        details={
                            "max_size_m": max_size_m,
                            "expected_domain_size_m": expected,
                        },
                    )
                )
            else:
                report.items.append(
                    UnitCheckItem(
                        name="geometry_scale_consistency",
                        level=UnitCheckLevel.INFO,
                        passed=True,
                        message="Geometry scale matches expected physical size.",
                        details={
                            "max_size_m": max_size_m,
                            "expected_domain_size_m": expected,
                        },
                    )
                )
        else:
            report.items.append(
                UnitCheckItem(
                    name="geometry_scale_consistency",
                    level=UnitCheckLevel.INFO,
                    passed=True,
                    message="Geometry scale produced valid physical extents.",
                    details={"extents_m": extents_m.tolist()},
                )
            )
    except Exception as exc:
        report.items.append(
            UnitCheckItem(
                name="geometry_scale_consistency",
                level=UnitCheckLevel.ERROR,
                passed=False,
                message=str(exc),
            )
        )

    # 3) Electrode length consistency
    lengths = np.asarray(fwd_model.electrode_lengths_m, dtype=float).reshape(-1)
    lengths_ok = bool(
        lengths.size == n_elec
        and all_finite_values(lengths)
        and float(np.min(lengths, initial=np.inf)) > 0.0
    )
    if lengths_ok:
        report.items.append(
            UnitCheckItem(
                name="electrode_length_physical_consistency",
                level=UnitCheckLevel.INFO,
                passed=True,
                message="Electrode physical lengths are positive and complete.",
                details={
                    "min_length_m": float(lengths.min()),
                    "max_length_m": float(lengths.max()),
                },
            )
        )
    else:
        report.items.append(
            UnitCheckItem(
                name="electrode_length_physical_consistency",
                level=UnitCheckLevel.ERROR,
                passed=False,
                message="Electrode physical lengths are invalid or incomplete.",
                details={"n_lengths": int(lengths.size), "n_elec": n_elec},
            )
        )

    # 4) Current conservation
    stim = np.asarray(fwd_model.pattern_manager.stim_matrix, dtype=float)
    row_sums = stim.sum(axis=1)
    row_scales = np.maximum(1.0, np.max(np.abs(stim), axis=1))
    conservation_tol = 1e-12 * row_scales
    bad_rows = np.nonzero(np.abs(row_sums) > conservation_tol)[0]
    if bad_rows.size == 0:
        report.items.append(
            UnitCheckItem(
                name="current_conservation",
                level=UnitCheckLevel.INFO,
                passed=True,
                message="All stimulation patterns conserve net current.",
                details={
                    "max_abs_row_sum": float(np.max(np.abs(row_sums), initial=0.0))
                },
            )
        )
    else:
        worst = int(bad_rows[np.argmax(np.abs(row_sums[bad_rows]))])
        report.items.append(
            UnitCheckItem(
                name="current_conservation",
                level=UnitCheckLevel.ERROR,
                passed=False,
                message=(
                    f"Current conservation failed in {bad_rows.size} pattern(s), "
                    f"worst pattern={worst}."
                ),
                details={"worst_row_sum": float(row_sums[worst])},
            )
        )

    # 5) Current density closure
    mode = str(cfg.drive_mode).strip().lower()
    if mode != "line_current_density":
        report.items.append(
            UnitCheckItem(
                name="current_density_closure",
                level=UnitCheckLevel.INFO,
                passed=True,
                message="Skipped: drive_mode is not line_current_density.",
            )
        )
        return report

    lengths_full = np.asarray(
        fwd_model.pattern_manager._electrode_lengths_m, dtype=float
    )  # noqa: SLF001
    n_elec_single_ring = int(cfg.n_elec)
    max_rel_err = 0.0
    for stim_idx in range(fwd_model.pattern_manager.n_stim):
        ring = stim_idx // n_elec_single_ring
        elec = stim_idx % n_elec_single_ring
        for inj_i, inj_elec in enumerate(fwd_model.pattern_manager.inj_electrodes):
            idx = (
                inj_elec + fwd_model.pattern_manager.stim_direction * elec
            ) % n_elec_single_ring + ring * n_elec_single_ring
            current = stim[stim_idx, idx]
            density = current / lengths_full[idx]
            expected = float(cfg.drive_value) * float(
                fwd_model.pattern_manager.inj_weights[inj_i]
            )
            rel = abs(density - expected) / max(abs(expected), np.finfo(float).eps)
            max_rel_err = max(max_rel_err, float(rel))

    if max_rel_err <= density_rel_tol:
        report.items.append(
            UnitCheckItem(
                name="current_density_closure",
                level=UnitCheckLevel.INFO,
                passed=True,
                message="Current density closure check passed.",
                details={"max_rel_err": max_rel_err},
            )
        )
    elif max_rel_err <= density_rel_tol * 100.0:
        report.items.append(
            UnitCheckItem(
                name="current_density_closure",
                level=UnitCheckLevel.WARN,
                passed=True,
                message="Current density closure near tolerance boundary.",
                details={
                    "max_rel_err": max_rel_err,
                    "density_rel_tol": density_rel_tol,
                },
            )
        )
    else:
        report.items.append(
            UnitCheckItem(
                name="current_density_closure",
                level=UnitCheckLevel.ERROR,
                passed=False,
                message="Current density closure failed.",
                details={
                    "max_rel_err": max_rel_err,
                    "density_rel_tol": density_rel_tol,
                },
            )
        )

    return report
