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
from pyeidors.inverse.jacobian.linearized import JacobianLinearization

OPERATOR_CACHE_KEYS = (
    "operator_jt",
    "operator_noser",
    "operator_A",
    "operator_lu",
)


def _small_linearization() -> JacobianLinearization:
    return JacobianLinearization(
        grad_u_all=(
            np.array(
                [
                    [1.0, 0.5],
                    [0.5, 1.0],
                    [1.5, -0.5],
                ],
                dtype=float,
            ),
        ),
        adjoint_gradients=(
            np.array(
                [
                    [0.25, 0.75],
                    [1.0, 0.5],
                    [0.5, -1.0],
                ],
                dtype=float,
            ),
            np.array(
                [
                    [0.75, -0.25],
                    [0.5, 1.25],
                    [1.0, 0.25],
                ],
                dtype=float,
            ),
        ),
        cell_areas=np.array([1.0, 1.5, 0.75], dtype=float),
        n_meas_per_stim=(2,),
        sign=-1.0,
    )


def _small_linearized_bundle(
    *,
    strategy: str = "auto",
    maxiter: int | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    linearization = _small_linearization()
    projection_weights = np.array([1.0, -0.5], dtype=float)
    reg_diag = gn_difference_runner._build_noser_diag_from_linearization(
        linearization,
        projection_weights=projection_weights,
    )
    lam = 0.2
    rhs = np.array([0.4, -0.15], dtype=float)
    dense_j = np.diag(projection_weights) @ linearization.to_dense()
    bundle = {
        "jacobian_representation": "linearized",
        "linearized_solver_strategy": strategy,
        "linearization": linearization,
        "projection_weights": projection_weights,
        "reg_diag": reg_diag,
        "precond_diag": np.maximum(
            np.sum(dense_j * dense_j, axis=0) + lam * reg_diag,
            1e-12,
        ),
        "lambda": lam,
    }
    if maxiter is not None:
        bundle["linearized_maxiter"] = int(maxiter)
    expected = np.linalg.solve(
        dense_j.T @ dense_j + lam * np.diag(reg_diag),
        dense_j.T @ rhs,
    )
    return bundle, rhs, expected


def _build_ctx(cache_dir: Path, background_sigma: float) -> dict:
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


def test_linearized_delta_solver_matches_dense_regularized_solution():
    bundle, rhs, expected = _small_linearized_bundle()
    actual = gn_difference_runner._solve_linearized_delta(
        operator_bundle=bundle,
        rhs=rhs,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


def test_linearized_solver_cg_only_does_not_fallback_to_lsmr(monkeypatch):
    bundle, rhs, _expected = _small_linearized_bundle(
        strategy="cg_only",
        maxiter=1,
    )
    n_param = int(bundle["linearization"].n_parameters)

    def _fake_cg(*_args, **_kwargs):
        return np.zeros(n_param, dtype=float), 1

    def _unexpected_lsmr(**_kwargs):
        raise AssertionError("cg_only must not invoke LSMR fallback")

    monkeypatch.setattr(gn_difference_runner, "cg", _fake_cg)
    monkeypatch.setattr(gn_difference_runner, "_solve_linearized_lsmr", _unexpected_lsmr)

    actual = gn_difference_runner._solve_linearized_delta(
        operator_bundle=bundle,
        rhs=rhs,
    )

    np.testing.assert_allclose(actual, np.zeros(n_param))
    last = bundle["linearized_last_solve"]
    assert last["strategy"] == "cg-only"
    assert last["method"] == "cg"
    assert last["cg_info"] == 1
    assert last["converged"] is False


def test_linearized_solver_cg_lsmr_falls_back_on_failed_cg(monkeypatch):
    bundle, rhs, _expected = _small_linearized_bundle(
        strategy="cg_lsmr",
        maxiter=1,
    )
    n_param = int(bundle["linearization"].n_parameters)
    fallback = np.linspace(0.1, 0.3, n_param, dtype=float)

    def _fake_cg(*_args, **_kwargs):
        return np.zeros(n_param, dtype=float), 1

    def _fake_lsmr(**_kwargs):
        return fallback

    monkeypatch.setattr(gn_difference_runner, "cg", _fake_cg)
    monkeypatch.setattr(gn_difference_runner, "_solve_linearized_lsmr", _fake_lsmr)

    actual = gn_difference_runner._solve_linearized_delta(
        operator_bundle=bundle,
        rhs=rhs,
    )

    np.testing.assert_allclose(actual, fallback)
    last = bundle["linearized_last_solve"]
    assert last["strategy"] == "cg-lsmr"
    assert last["method"] == "lsmr"


def test_linearized_solver_cgls_matches_dense_regularized_solution():
    bundle, rhs, expected = _small_linearized_bundle(
        strategy="cgls",
        maxiter=40,
    )

    actual = gn_difference_runner._solve_linearized_delta(
        operator_bundle=bundle,
        rhs=rhs,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)
    last = bundle["linearized_last_solve"]
    assert last["strategy"] == "cgls"
    assert last["method"] == "cgls"
