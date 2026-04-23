"""Tests for low-memory exact strict backend selection in GN difference solver."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_PATH = REPO_ROOT / "scripts"
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

from common import gn_difference_runner
from common.hdf5_outputs import read_output_bundle

GIB = float(1024**3)


def _forced_selector(effective: str):
    def _select(
        *,
        mesh_dim: int,
        n_param: int,
        n_meas: int,
        mem_available_bytes: int | None = None,
    ):
        estimated_peak_bytes = gn_difference_runner._estimate_strict_dense_peak_bytes(
            int(n_param)
        )
        return {
            "requested": gn_difference_runner.STRICT_SOLVER_BACKEND_DENSE,
            "effective": str(effective),
            "strict_memory_guard_triggered": str(effective)
            == gn_difference_runner.STRICT_SOLVER_BACKEND_MEASUREMENT,
            "strict_memory_guard_reason": "forced_for_test",
            "strict_dense_estimated_peak_bytes": float(estimated_peak_bytes),
            "strict_dense_estimated_peak_gib": float(estimated_peak_bytes / GIB),
            "strict_memory_guard_limit_bytes": float(12 * 1024**3),
            "strict_memory_guard_limit_gib": 12.0,
            "strict_mem_available_bytes": int(mem_available_bytes or (16 * 1024**3)),
            "strict_mem_available_gib": float(
                (mem_available_bytes or (16 * 1024**3)) / GIB
            ),
            "strict_mem_available_source": "forced",
            "strict_measurement_system_shape": [int(n_meas), int(n_meas)]
            if str(effective) == gn_difference_runner.STRICT_SOLVER_BACKEND_MEASUREMENT
            else None,
        }

    return _select


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
        solver_mode="strict",
        linear_solver="auto",
    )


def test_select_strict_backend_prefers_dense_for_small_problem():
    info = gn_difference_runner._select_strict_solver_backend(
        mesh_dim=2,
        n_param=256,
        n_meas=128,
        mem_available_bytes=16 * 1024**3,
    )
    assert info["requested"] == "dense-param"
    assert info["effective"] == "dense-param"
    assert info["strict_memory_guard_triggered"] is False
    assert info["strict_memory_guard_reason"] == "dense_allowed_non3d"


def test_select_strict_backend_switches_to_measurement_exact_for_default_3d_scale():
    info = gn_difference_runner._select_strict_solver_backend(
        mesh_dim=3,
        n_param=19629,
        n_meas=208,
        mem_available_bytes=14 * 1024**3,
    )
    assert info["effective"] == "measurement-exact"
    assert info["strict_memory_guard_triggered"] is True
    assert info["strict_memory_guard_reason"] == "dense_estimate_exceeds_guard"
    assert (
        info["strict_dense_estimated_peak_gib"] > info["strict_memory_guard_limit_gib"]
    )
    assert info["strict_measurement_system_shape"] == [208, 208]


def test_measurement_exact_strict_matches_dense_param_on_small_problem(
    tmp_path: Path, monkeypatch
):
    dense_ctx = _build_ctx(tmp_path / "cache-dense")

    monkeypatch.setattr(
        gn_difference_runner,
        "_select_strict_solver_backend",
        _forced_selector(gn_difference_runner.STRICT_SOLVER_BACKEND_MEASUREMENT),
    )
    exact_ctx = _build_ctx(tmp_path / "cache-measurement")

    vh = np.asarray(dense_ctx["base_meas"], dtype=float)
    vi = vh * 1.001

    dense_metrics = gn_difference_runner.process_frames(
        vh=vh,
        vi=vi,
        output_dir=tmp_path / "dense-out",
        ctx=dense_ctx,
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
    exact_metrics = gn_difference_runner.process_frames(
        vh=vh,
        vi=vi,
        output_dir=tmp_path / "measurement-out",
        ctx=exact_ctx,
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

    dense_bundle = read_output_bundle(tmp_path / "dense-out" / "outputs.h5")
    exact_bundle = read_output_bundle(tmp_path / "measurement-out" / "outputs.h5")
    np.testing.assert_allclose(
        dense_bundle["delta_sigma"],
        exact_bundle["delta_sigma"],
        rtol=1e-9,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        dense_bundle["pred_diff"],
        exact_bundle["pred_diff"],
        rtol=1e-9,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        dense_bundle["pred_vi"],
        exact_bundle["pred_vi"],
        rtol=1e-9,
        atol=1e-11,
    )
    assert dense_metrics["strict_solver_backend_effective"] == "dense-param"
    assert exact_metrics["strict_solver_backend_effective"] == "measurement-exact"


def test_strict_backend_cache_keys_do_not_mix_between_dense_and_measurement_exact(
    tmp_path: Path, monkeypatch
):
    cache_dir = tmp_path / "shared-cache"
    dense_ctx = _build_ctx(cache_dir)

    dense_operator_key = dense_ctx["cache_lookups"]["operator_A"]["key"]
    dense_factor_key = dense_ctx["cache_lookups"]["operator_lu"]["key"]
    dense_shape = tuple(np.asarray(dense_ctx["operator_bundle"]["A"]).shape)

    monkeypatch.setattr(
        gn_difference_runner,
        "_select_strict_solver_backend",
        _forced_selector(gn_difference_runner.STRICT_SOLVER_BACKEND_MEASUREMENT),
    )
    measurement_cold = _build_ctx(cache_dir)
    measurement_warm = _build_ctx(cache_dir)

    measurement_operator_key = measurement_cold["cache_lookups"]["operator_A"]["key"]
    measurement_factor_key = measurement_cold["cache_lookups"]["operator_lu"]["key"]
    measurement_shape = tuple(
        np.asarray(measurement_cold["operator_bundle"]["A"]).shape
    )

    assert measurement_cold["strict_backend_info"]["effective"] == "measurement-exact"
    assert measurement_cold["cache_lookups"]["operator_A"]["hit"] is False
    assert measurement_cold["cache_lookups"]["operator_lu"]["hit"] is False
    assert measurement_warm["cache_lookups"]["operator_A"]["hit"] is True
    assert measurement_warm["cache_lookups"]["operator_lu"]["hit"] is True
    assert dense_operator_key != measurement_operator_key
    assert dense_factor_key != measurement_factor_key
    assert measurement_shape[0] < dense_shape[0]
    assert measurement_shape[0] == measurement_shape[1]

    payload = {
        "dense": dense_ctx["strict_backend_info"],
        "measurement": measurement_cold["strict_backend_info"],
    }
    assert (
        json.loads(json.dumps(payload))["measurement"]["effective"]
        == "measurement-exact"
    )
