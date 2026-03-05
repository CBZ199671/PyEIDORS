"""Tests for GN-difference operator cache warm-start behavior."""

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


def _build_ctx(cache_dir: Path, background_sigma: float) -> dict:
    return gn_difference_runner.build_shared_context(
        mesh_dir=str(REPO_ROOT / "eit_meshes"),
        mesh_name="mesh_16e_r0p025_ref10_cov0p5",
        n_elec=16,
        radius=0.025,
        drive_value=1.0,
        contact_impedance=1e-6,
        background_sigma=background_sigma,
        lam=0.1,
        cache_scope="both",
        cache_dir=str(cache_dir),
        cache_clear_names=[],
    )


def test_gn_difference_context_cache_hits_and_invalidates_with_background(tmp_path: Path):
    cache_dir = tmp_path / "diff-cache"

    cold_ctx = _build_ctx(cache_dir, background_sigma=1.0)
    assert cold_ctx["cache_lookups"]["jacobian"]["hit"] is False
    for key in OPERATOR_CACHE_KEYS:
        assert cold_ctx["cache_lookups"][key]["hit"] is False

    warm_ctx = _build_ctx(cache_dir, background_sigma=1.0)
    assert warm_ctx["cache_lookups"]["jacobian"]["hit"] is True
    assert warm_ctx["cache_lookups"]["jacobian"]["layer"] in {"disk", "process"}
    for key in OPERATOR_CACHE_KEYS:
        assert warm_ctx["cache_lookups"][key]["hit"] is True
        assert warm_ctx["cache_lookups"][key]["layer"] in {"disk", "process"}

    changed_bg_ctx = _build_ctx(cache_dir, background_sigma=1.0005)
    assert changed_bg_ctx["cache_lookups"]["jacobian"]["hit"] is False
    for key in OPERATOR_CACHE_KEYS:
        assert changed_bg_ctx["cache_lookups"][key]["hit"] is False


def test_gn_difference_process_frames_reports_cache_metrics(tmp_path: Path):
    cache_dir = tmp_path / "diff-cache-metrics"
    ctx = _build_ctx(cache_dir, background_sigma=1.0)

    vh = np.asarray(ctx["base_meas"], dtype=float)
    vi = vh * 1.001
    output_dir = tmp_path / "case"
    metrics = gn_difference_runner.process_frames(
        vh=vh,
        vi=vi,
        output_dir=output_dir,
        ctx=ctx,
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

    assert "rmse_abs" in metrics
    assert "cache_lookups" in metrics
    assert "cache_stats" in metrics
    assert "cache_miss_reasons" in metrics
    assert "cache_build_seconds" in metrics
