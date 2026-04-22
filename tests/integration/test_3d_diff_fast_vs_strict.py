"""3D difference reconstruction parity between strict and fast modes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyeidors.geometry.mesh3d_generator import GMSH_AVAILABLE
from scripts.common.hdf5_outputs import read_output_bundle
from scripts.common.gn_difference_runner import build_shared_context, process_frames


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
def test_diff_3d_fast_vs_strict_rmse(tmp_path: Path):
    common = dict(
        mesh_dir=str(tmp_path / "meshes"),
        mesh_name="diff_fast_vs_strict",
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
    )
    strict_ctx = build_shared_context(**common, solver_mode="strict", linear_solver="auto")
    fast_ctx = build_shared_context(**common, solver_mode="fast", linear_solver="auto")

    vh = np.asarray(strict_ctx["base_meas"], dtype=float)
    vi = vh + 1e-4
    strict_out = process_frames(
        vh=vh,
        vi=vi,
        output_dir=tmp_path / "strict",
        ctx=strict_ctx,
        step_size_calib=False,
        step_size_min=1e-3,
        step_size_max=1.0,
        step_size_maxiter=20,
        lam=1e-2,
        colormap="viridis",
        colorbar_scientific=False,
        colorbar_format="plain",
        transparent=False,
        write_plots=False,
        measurement_gain=1.0,
    )
    fast_out = process_frames(
        vh=vh,
        vi=vi,
        output_dir=tmp_path / "fast",
        ctx=fast_ctx,
        step_size_calib=False,
        step_size_min=1e-3,
        step_size_max=1.0,
        step_size_maxiter=20,
        lam=1e-2,
        colormap="viridis",
        colorbar_scientific=False,
        colorbar_format="plain",
        transparent=False,
        write_plots=False,
        measurement_gain=1.0,
    )

    strict_delta = read_output_bundle(tmp_path / "strict" / "outputs.h5")["delta_sigma"]
    fast_delta = read_output_bundle(tmp_path / "fast" / "outputs.h5")["delta_sigma"]
    rmse = float(np.sqrt(np.mean((strict_delta - fast_delta) ** 2)))

    assert strict_out["solver_mode"] == "strict"
    assert fast_out["solver_mode"] == "fast"
    assert rmse <= 1e-7
