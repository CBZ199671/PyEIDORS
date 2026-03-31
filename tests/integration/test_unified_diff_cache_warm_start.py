"""Integration test for unified GN-difference warm-start cache behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_PATH = REPO_ROOT / "scripts"
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

from common import gn_difference_runner

OPERATOR_CACHE_KEYS = (
    "operator_jt",
    "operator_noser",
    "operator_A",
    "operator_lu",
)


def _build_ctx(cache_dir: Path) -> dict:
    return gn_difference_runner.build_shared_context(
        mesh_dir=str(REPO_ROOT / "eit_meshes"),
        mesh_name="mesh_16e_r0p025_ref10_cov0p5",
        mesh_dim=2,
        mesh_height=1.0,
        electrode_height_ratio=0.2,
        z_center=0.0,
        refinement=6,
        n_elec=16,
        radius=0.025,
        drive_value=1.0,
        contact_impedance=1e-6,
        background_sigma=1.0,
        lam=0.1,
        cache_scope="both",
        cache_dir=str(cache_dir),
        cache_clear_names=[],
    )


def test_warm_start_cache_and_numerical_consistency(tmp_path: Path):
    cache_dir = tmp_path / "cache"

    ctx1 = _build_ctx(cache_dir)
    vh = np.asarray(ctx1["base_meas"], dtype=float)
    vi = vh * 1.002
    metrics_cold = gn_difference_runner.process_frames(
        vh=vh,
        vi=vi,
        output_dir=tmp_path / "cold",
        ctx=ctx1,
        step_size_calib=False,
        step_size_min=1e-3,
        step_size_max=1.0,
        step_size_maxiter=5,
        lam=0.1,
        colormap="viridis",
        colorbar_scientific=False,
        colorbar_format=None,
        transparent=False,
        write_plots=False,
        measurement_gain=1.0,
    )

    ctx2 = _build_ctx(cache_dir)
    assert ctx2["cache_lookups"]["jacobian"]["hit"] is True
    for key in OPERATOR_CACHE_KEYS:
        assert ctx2["cache_lookups"][key]["hit"] is True

    metrics_warm = gn_difference_runner.process_frames(
        vh=vh,
        vi=vi,
        output_dir=tmp_path / "warm",
        ctx=ctx2,
        step_size_calib=False,
        step_size_min=1e-3,
        step_size_max=1.0,
        step_size_maxiter=5,
        lam=0.1,
        colormap="viridis",
        colorbar_scientific=False,
        colorbar_format=None,
        transparent=False,
        write_plots=False,
        measurement_gain=1.0,
    )

    rmse_diff = abs(float(metrics_cold["rmse_abs"]) - float(metrics_warm["rmse_abs"]))
    assert rmse_diff < 1e-10
