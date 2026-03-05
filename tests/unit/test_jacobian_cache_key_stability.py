"""Tests for Jacobian cache key stability across system instances."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from dolfinx import fem

from pyeidors.core_system import EITSystem
from pyeidors.data.structures import PatternConfig
from pyeidors.inverse.jacobian.direct_jacobian import DirectJacobianCalculator


def _build_system(eit_mesh, cache_dir: Path) -> EITSystem:
    pattern = PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="normalized",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
        use_meas_current=False,
        rotate_meas=True,
    )
    system = EITSystem(
        n_elec=16,
        pattern_config=pattern,
        contact_impedance=np.full(16, 1e-5, dtype=float),
        regularization_type="noser",
        regularization_alpha=1.0,
        cache_scope="both",
        cache_dir=str(cache_dir),
    )
    system.setup(mesh=eit_mesh)
    return system


def test_jacobian_cache_hits_disk_without_object_identity(eit_mesh, tmp_path: Path):
    cache_dir = tmp_path / "jac-cache"

    system_a = _build_system(eit_mesh, cache_dir)
    calc_a = DirectJacobianCalculator(system_a.fwd_model)
    sigma_a = fem.Function(system_a.fwd_model.V_sigma)
    sigma_a.x.array[:] = 1.0
    _ = calc_a.calculate(sigma_a, method="efficient")
    first_lookup = getattr(calc_a, "_last_cache_lookup", {})
    assert first_lookup.get("hit") is False
    assert isinstance(first_lookup.get("key"), str)

    system_b = _build_system(eit_mesh, cache_dir)
    calc_b = DirectJacobianCalculator(system_b.fwd_model)
    sigma_b = fem.Function(system_b.fwd_model.V_sigma)
    sigma_b.x.array[:] = 1.0
    _ = calc_b.calculate(sigma_b, method="efficient")
    second_lookup = getattr(calc_b, "_last_cache_lookup", {})
    assert second_lookup.get("hit") is True
    assert second_lookup.get("layer") in {"disk", "process"}
    assert second_lookup.get("key") == first_lookup.get("key")
