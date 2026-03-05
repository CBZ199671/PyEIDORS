"""Integration test for EIDORS-style diff cache parity in unified pipeline."""

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


def test_unified_diff_cache_eidors_style_warm_start(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cold_ctx = _build_ctx(cache_dir)
    warm_ctx = _build_ctx(cache_dir)

    assert cold_ctx["cache_lookups"]["jacobian"]["hit"] is False
    assert warm_ctx["cache_lookups"]["jacobian"]["hit"] is True
    for key in OPERATOR_CACHE_KEYS:
        assert cold_ctx["cache_lookups"][key]["hit"] is False
        assert warm_ctx["cache_lookups"][key]["hit"] is True

    vh = np.asarray(cold_ctx["base_meas"], dtype=float)
    vi = vh * 1.0015

    cold_metrics = gn_difference_runner.process_frames(
        vh=vh,
        vi=vi,
        output_dir=tmp_path / "cold",
        ctx=cold_ctx,
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
    warm_metrics = gn_difference_runner.process_frames(
        vh=vh,
        vi=vi,
        output_dir=tmp_path / "warm",
        ctx=warm_ctx,
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

    rmse_gap = abs(float(cold_metrics["rmse_abs"]) - float(warm_metrics["rmse_abs"]))
    assert rmse_gap < 1e-10
