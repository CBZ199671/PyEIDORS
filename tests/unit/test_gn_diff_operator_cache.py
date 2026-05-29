"""Tests for GN-difference operator cache warm-start behavior."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_v256_gn_difference_sigma_hash_uses_streaming_payload_helper() -> None:
    source = inspect.getsource(gn_difference_runner.build_shared_context)

    assert "hash_array_payload" in source
    assert ".tobytes()" not in source
    assert "sigma_hash = hashlib.sha256" not in source


def test_gn_difference_column_scaled_jjt_streams_column_blocks() -> None:
    jacobian = np.array(
        [[1.0, 2.0, -1.0, 0.5], [0.25, -0.5, 3.0, 1.5]],
        dtype=np.float64,
    )
    scale = np.array([0.5, 2.0, 0.25, 1.5], dtype=np.float64)

    actual = gn_difference_runner._column_scaled_jjt(
        jacobian,
        scale,
        chunk_target_bytes=16,
    )
    expected = (jacobian * scale.reshape(1, -1)) @ jacobian.T

    np.testing.assert_allclose(actual, expected)
    source = inspect.getsource(gn_difference_runner)
    assert "jacobian * inv_reg_diag[None, :]" not in source
    assert "_column_scaled_jjt(" in source


def test_gn_difference_reduced_rm_regularization_uses_row_scaled_jtj() -> None:
    jacobian = np.array(
        [
            [0.5, 1.0, -0.25, 0.75],
            [1.5, -0.5, 0.25, 1.25],
            [0.25, 0.75, 1.0, -0.5],
        ],
        dtype=np.float64,
    )
    basis = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
            [0.25, -0.5],
        ],
        dtype=np.float64,
    )
    reg_diag = np.array([0.5, 2.0, 1.5, 0.75], dtype=np.float64)
    lam = 0.2

    u_tru = gn_difference_runner._row_scaled_jtj(
        basis,
        reg_diag,
        chunk_target_bytes=16,
    )
    expected_u_tru = basis.T @ (reg_diag[:, None] * basis)
    np.testing.assert_allclose(u_tru, expected_u_tru)

    actual = gn_difference_runner._build_reduced_rm(
        jacobian=jacobian,
        reg_diag=reg_diag,
        lam=lam,
        basis=basis,
    )
    ju = jacobian @ basis
    expected_h = ju.T @ ju + lam * expected_u_tru
    expected_h = 0.5 * (expected_h + expected_h.T)
    expected_h_inv = np.linalg.inv(expected_h)
    expected_rm = basis @ expected_h_inv @ ju.T

    np.testing.assert_allclose(actual["JU"], ju)
    np.testing.assert_allclose(actual["H"], expected_h)
    np.testing.assert_allclose(actual["H_inv"], expected_h_inv)
    np.testing.assert_allclose(actual["RM_reduced"], expected_rm)

    source = inspect.getsource(gn_difference_runner._build_reduced_rm)
    assert "r_diag[:, None] * b_mat" not in source
    assert "_row_scaled_jtj" in source


def test_v521_gn_difference_system_diagonal_terms_added_in_place(monkeypatch) -> None:
    source = inspect.getsource(gn_difference_runner.build_shared_context)
    assert "float(lam) * np.eye(jacobian.shape[0]" not in source
    assert "float(lam) * np.diag(reg_diag)" not in source
    assert "np.diag(system_matrix)" not in source
    assert "_with_identity_shift(" in source
    assert "_with_diagonal_shift(" in source
    assert "_preconditioner_diagonal(system_matrix)" in source

    def _unexpected_dense_diagonal(*_args, **_kwargs):
        raise AssertionError("dense identity/diagonal helper must not be called")

    monkeypatch.setattr(gn_difference_runner.np, "eye", _unexpected_dense_diagonal)
    monkeypatch.setattr(gn_difference_runner.np, "diag", _unexpected_dense_diagonal)

    base = np.array([[2.0, 0.5], [0.5, 3.0]], dtype=np.float64, order="F")
    shifted_identity = gn_difference_runner._with_identity_shift(base, 0.25)
    np.testing.assert_allclose(
        shifted_identity,
        np.array([[2.25, 0.5], [0.5, 3.25]], dtype=np.float64),
    )
    assert shifted_identity.flags.c_contiguous

    shifted_diagonal = gn_difference_runner._with_diagonal_shift(
        base,
        np.array([4.0, 6.0], dtype=np.float64),
        0.5,
    )
    np.testing.assert_allclose(
        shifted_diagonal,
        np.array([[4.0, 0.5], [0.5, 6.0]], dtype=np.float64),
    )
    assert shifted_diagonal.flags.c_contiguous

    precond = gn_difference_runner._preconditioner_diagonal(
        np.array([[0.0, 1.0], [2.0, -3.0]], dtype=np.float64)
    )
    np.testing.assert_allclose(precond, np.array([1.0e-12, 1.0e-12]))


def test_v545_lsmr_augmented_operator_direct_fills_without_concatenate() -> None:
    source = inspect.getsource(gn_difference_runner._solve_linearized_lsmr)
    assert "np.concatenate" not in source
    assert "np.empty(n_meas + n_param" in source
    assert "np.zeros(n_meas + n_param" in source
    assert "out[:n_meas]" in source
    assert "rhs_aug[:n_meas]" in source


def test_v276_gn_difference_snapshot_columns_direct_fill_matrix() -> None:
    columns = [
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([4.0, 5.0, 6.0], dtype=np.float64),
        np.array([7, 8, 9], dtype=np.int32),
    ]

    actual = gn_difference_runner._stack_columns_direct(columns)

    np.testing.assert_allclose(actual, np.column_stack(columns))
    assert actual.dtype == np.float64
    assert actual.flags.c_contiguous
    assert "np.column_stack" not in inspect.getsource(
        gn_difference_runner._stack_columns_direct
    )


def test_v83_2d_auto_petsc_device_uses_cpu_without_blocking_explicit_cuda():
    assert (
        gn_difference_runner._mesh_compatible_petsc_device("auto", mesh_dim=2) == "cpu"
    )
    assert gn_difference_runner._mesh_compatible_petsc_device(None, mesh_dim=2) == "cpu"
    assert (
        gn_difference_runner._mesh_compatible_petsc_device("auto", mesh_dim=3) == "auto"
    )
    assert (
        gn_difference_runner._mesh_compatible_petsc_device("cuda", mesh_dim=2) == "cuda"
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


def test_eidors_adapter_jacobian_keeps_runtime_projection_sign():
    raw_eidors = np.array([[2.0, -4.0], [6.0, 8.0]], dtype=float)
    reference = np.array([2.0, 4.0], dtype=float)

    runtime_jacobian = gn_difference_runner._runtime_jacobian_from_eidors_adapter(
        raw_eidors
    )
    weights = gn_difference_runner._runtime_projection_weights_from_eidors_adapter(
        reference,
        difference_mode="normalized",
        difference_orientation="target_minus_reference",
    )

    np.testing.assert_allclose(runtime_jacobian, raw_eidors)
    np.testing.assert_allclose(
        weights[:, None] * raw_eidors, raw_eidors / reference[:, None]
    )


def test_single_step_semantic_payload_prefers_math_axes_over_version_only():
    payload = gn_difference_runner._single_step_semantic_payload(
        signature_schema_version=gn_difference_runner.SINGLE_STEP_SIGNATURE_SCHEMA_VERSION,
        jacobian_calculator=gn_difference_runner.SINGLE_STEP_JACOBIAN_CALCULATOR,
        jacobian_math_convention=gn_difference_runner.SINGLE_STEP_JACOBIAN_MATH_CONVENTION,
        projection_math_convention=gn_difference_runner.SINGLE_STEP_PROJECTION_MATH_CONVENTION,
        operator_math_convention=gn_difference_runner.SINGLE_STEP_OPERATOR_MATH_CONVENTION,
        algorithm_version=gn_difference_runner.SINGLE_STEP_ALGORITHM_VERSION,
    )

    assert payload["single_step_jacobian_calculator"] == "EidorsJacobianAdapter"
    assert payload["single_step_jacobian_math_convention"]
    assert payload["single_step_projection_math_convention"]
    assert payload["single_step_operator_math_convention"]
    assert payload["single_step_algorithm_version"]


def _build_ctx(cache_dir: Path, background_sigma: float) -> dict:
    return gn_difference_runner.build_shared_context(
        mesh_dir=str(REPO_ROOT / "eit_meshes"),
        mesh_name="mesh_8e_r1_ref8_cov0p5",
        mesh_dim=2,
        mesh_height=1.0,
        electrode_height_ratio=0.2,
        z_center=0.0,
        refinement=8,
        n_elec=8,
        radius=1.0,
        drive_value=1.0,
        contact_impedance=1e-6,
        background_sigma=background_sigma,
        lam=0.1,
        cache_scope="both",
        cache_dir=str(cache_dir),
        cache_clear_names=[],
    )


def test_step_size_calibration_keeps_candidates_above_sigma_floor():
    sigma_floor = 0.2
    sigma_seen: list[np.ndarray] = []

    class _ForwardModel:
        def fwd_solve(self, image):
            sigma = np.asarray(image.elem_data, dtype=float)
            sigma_seen.append(sigma.copy())
            assert float(np.min(sigma)) > sigma_floor
            return SimpleNamespace(meas=np.array([1.0, 2.0], dtype=float)), None

    alpha = gn_difference_runner._calibrate_step_size(
        fwd_model=_ForwardModel(),
        sigma_bg=np.array([1.0], dtype=float),
        delta_sigma=np.array([-2.0], dtype=float),
        dv=np.array([0.1, 0.2], dtype=float),
        base_meas=np.array([0.9, 1.8], dtype=float),
        step_size_min=0.0,
        step_size_max=1.0,
        step_size_maxiter=8,
        sigma_floor=sigma_floor,
    )

    assert sigma_seen
    assert 0.0 <= alpha < 0.4
    assert all(float(np.min(sigma)) > sigma_floor for sigma in sigma_seen)


def test_gn_difference_context_cache_hits_and_invalidates_with_background(
    tmp_path: Path,
):
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
    monkeypatch.setattr(
        gn_difference_runner, "_solve_linearized_lsmr", _unexpected_lsmr
    )

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
