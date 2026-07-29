"""Tests for physics.unit_consistency checks and system precheck entrypoint."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from pyeidors.core_system import EITSystem
from pyeidors.data.structures import PatternConfig
from pyeidors.forward.eit_forward_model import EITForwardModel
import pyeidors.physics.unit_consistency as unit_module
from pyeidors.physics import UnitCheckLevel, run_unit_consistency_checks


def _build_forward_model(
    eit_mesh, *, drive_mode: str = "line_current_density"
) -> EITForwardModel:
    config = PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode=drive_mode,  # type: ignore[arg-type]
        drive_value=5e-5 if drive_mode != "normalized" else 1.0,
        geometry_scale_to_m=1.0,
        use_meas_current=False,
        rotate_meas=True,
    )
    z = np.full(16, 1e-5, dtype=float)
    return EITForwardModel(n_elec=16, pattern_config=config, z=z, mesh=eit_mesh)


def test_v498_unit_consistency_uses_bounded_finite_scan() -> None:
    source = inspect.getsource(unit_module.run_unit_consistency_checks)

    assert "all_finite_values(extents_m)" in source
    assert "all_finite_values(lengths)" in source
    assert "np.all(np.isfinite(extents_m))" not in source
    assert "np.all(np.isfinite(lengths))" not in source


def test_unit_consistency_checks_happy_path(eit_mesh):
    model = _build_forward_model(eit_mesh)
    report = run_unit_consistency_checks(model, expected_domain_size_m=1.0)
    assert len(report.items) == 5
    assert report.has_errors is False
    assert all(
        item.passed for item in report.items if item.level != UnitCheckLevel.WARN
    )


def test_geometry_size_mismatch_reports_error(eit_mesh):
    model = _build_forward_model(eit_mesh)
    report = run_unit_consistency_checks(model, expected_domain_size_m=0.2)
    geom_item = next(
        item for item in report.items if item.name == "geometry_scale_consistency"
    )
    assert geom_item.level == UnitCheckLevel.ERROR
    assert report.has_errors is True


def test_current_conservation_violation_is_detected(eit_mesh):
    model = _build_forward_model(eit_mesh)
    model.pattern_manager.stim_matrix[0, 0] += 1e-3
    report = run_unit_consistency_checks(model)
    item = next(item for item in report.items if item.name == "current_conservation")
    assert item.level == UnitCheckLevel.ERROR
    assert item.passed is False


def test_density_closure_warns_near_tolerance(eit_mesh):
    model = _build_forward_model(eit_mesh)
    row = model.pattern_manager.stim_matrix[0]
    nz = np.nonzero(np.abs(row) > 0.0)[0]
    model.pattern_manager.stim_matrix[0, nz] = row[nz] * (1.0 + 5e-8)
    report = run_unit_consistency_checks(model, density_rel_tol=1e-8)
    item = next(item for item in report.items if item.name == "current_density_closure")
    assert item.level == UnitCheckLevel.WARN
    assert item.passed is True


def test_v749_pem_unit_check_does_not_treat_placeholder_lengths_as_physical(
    eit_mesh,
):
    model = _build_forward_model(eit_mesh, drive_mode="total_current")
    model.electrode_model = "pem"
    model.electrode_lengths_m = np.full(model.n_elec, np.nan)

    report = run_unit_consistency_checks(model)

    item = next(
        item
        for item in report.items
        if item.name == "electrode_length_physical_consistency"
    )
    assert item.passed is True
    assert item.details == {"electrode_model": "pem"}
    assert "Not applicable" in item.message


def test_system_run_unit_precheck_respects_strict_flag(eit_mesh):
    pattern = PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="line_current_density",
        drive_value=5e-5,
        geometry_scale_to_m=1.0,
        use_meas_current=False,
        rotate_meas=True,
    )
    system = EITSystem(
        n_elec=16,
        pattern_config=pattern,
        contact_impedance=np.full(16, 1e-5, dtype=float),
    )
    system.setup(mesh=eit_mesh)

    report = system.run_unit_precheck(expected_domain_size_m=0.2, strict=False)
    assert report.has_errors is True
    with pytest.raises(ValueError, match="Unit precheck failed"):
        system.run_unit_precheck(expected_domain_size_m=0.2, strict=True)
