"""3D cache warm-start speed and hit-layer assertions."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from pyeidors.geometry.mesh3d_generator import GMSH_AVAILABLE
from scripts.common.gn_difference_runner import build_shared_context


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
def test_3d_cache_warmup_speed_and_hits(tmp_path: Path):
    kwargs = dict(
        mesh_dir=str(tmp_path / "meshes"),
        mesh_name="cache_warmup_3d",
        mesh_dim=3,
        mesh_height=0.12,
        electrode_height_ratio=0.2,
        z_center=0.0,
        refinement=1,
        n_elec=8,
        radius=0.12,
        drive_value=1.0,
        contact_impedance=1e-5,
        background_sigma=1.0,
        lam=1e-2,
        cache_scope="both",
        cache_dir=str(tmp_path / "cache"),
        solver_mode="fast",
        linear_solver="auto",
    )

    t0 = time.perf_counter()
    cold = build_shared_context(**kwargs)
    cold_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    warm = build_shared_context(**kwargs)
    warm_elapsed = time.perf_counter() - t1

    assert cold["cache_lookups"]["jacobian"]["hit"] is False
    assert warm["cache_lookups"]["jacobian"]["hit"] is True
    assert warm["cache_lookups"]["operator_A"]["hit"] is True
    assert warm_elapsed < cold_elapsed
