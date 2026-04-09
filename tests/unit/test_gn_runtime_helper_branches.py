"""Additional branch coverage for Gauss-Newton runtime helper logic."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

import pyeidors.inverse.solvers.gauss_newton_runtime as gn_runtime


class _Lookup:
    def __init__(self, layer: str = "memory"):
        self.layer = layer


class _EnabledCacheManager:
    def __init__(self):
        self.enabled = True
        self.calls: list[str] = []

    def get_or_compute_semantic(self, **kwargs):
        self.calls.append(str(kwargs["artifact"]))
        return kwargs["compute_fn"](), _Lookup("memory")


def _solve_reconstructor(**overrides):
    recon = SimpleNamespace(
        device="cpu",
        _torch_dtype=torch.float64,
        measurement_weight_strategy="baseline",
        use_measurement_weights=True,
        difference_mode="normalized",
        difference_orientation="target_minus_reference",
        _measurement_space_type="real",
        _difference_reference_meas=None,
        _difference_target_meas=None,
        _measured_vector=np.array([1.0, 2.0, 3.0], dtype=float),
        weight_floor=0.25,
        verbose=False,
        R_matrix=sparse.diags([1.0, 2.0, 3.0], 0, format="csr"),
        R_diag=np.array([1.0, 2.0, 3.0], dtype=float),
        use_prior_term=True,
        performance_mode="aggressive",
        linear_solver="auto",
        preconditioner="diag",
        fast_linear_path="auto",
        solver_mode="fast",
        rom_mode="auto",
        inexact_mode="auto",
        lowrank_mode="auto",
        cache_manager=None,
        cholmod_max_n=50000,
        cholmod_max_memory_gib=4.0,
        fwd_model=SimpleNamespace(
            mesh=SimpleNamespace(geometry=SimpleNamespace(dim=3)),
            fwd_solve=lambda _img: (SimpleNamespace(meas=np.array([1.0, 2.0, 3.0], dtype=float)), None),
        ),
    )
    for key, value in overrides.items():
        setattr(recon, key, value)
    return recon


def test_to_runtime_tensor_cached_reuses_buffer_and_resizes():
    recon = _solve_reconstructor()
    first = gn_runtime._to_runtime_tensor_cached(recon, "meas", np.array([1.0, 2.0], dtype=float))
    second = gn_runtime._to_runtime_tensor_cached(recon, "meas", np.array([3.0, 4.0], dtype=float))
    third = gn_runtime._to_runtime_tensor_cached(recon, "meas", np.array([5.0, 6.0, 7.0], dtype=float))

    assert torch.allclose(first, torch.tensor([3.0, 4.0], dtype=torch.float64))
    assert second.data_ptr() == first.data_ptr()
    assert third.shape == (3,)
    assert third.data_ptr() != second.data_ptr()


def test_measurement_weights_cover_disabled_and_verbose_paths(monkeypatch: pytest.MonkeyPatch, capsys):
    recon = _solve_reconstructor(use_measurement_weights=False)
    gn_runtime.ensure_measurement_weights(recon, sigma_function=object())
    assert recon._meas_weight_sqrt is None
    assert recon._baseline_measurement is None

    monkeypatch.setattr(gn_runtime, "function_get_array", lambda _sigma: np.array([0.5, 1.5], dtype=float))
    monkeypatch.setattr(
        gn_runtime,
        "build_weight_reference",
        lambda **_kwargs: np.array([4.0, np.nan, 0.1], dtype=float),
    )
    monkeypatch.setattr(
        gn_runtime,
        "project_measurement_vector",
        lambda meas, **_kwargs: np.asarray(meas, dtype=float) * 2.0,
    )
    recon = _solve_reconstructor(verbose=True)
    gn_runtime.ensure_measurement_weights(recon, sigma_function=object())

    np.testing.assert_allclose(recon._baseline_measurement, np.array([2.0, 4.0, 6.0], dtype=float))
    np.testing.assert_allclose(
        recon._meas_weight_sqrt.detach().cpu().numpy(),
        np.array([8.0, 1.0, 1.0], dtype=float),
    )
    assert "measurement weights" in capsys.readouterr().out


def test_finite_helpers_raise_with_context_and_summary():
    assert gn_runtime._finite_summary(np.array([np.nan, np.inf], dtype=float)) == "finite_count=0"
    summary = gn_runtime._finite_summary(np.array([1.0, np.nan, 3.0], dtype=float))
    assert "finite_count=2" in summary
    assert "l2=" in summary

    with pytest.raises(FloatingPointError, match="iteration=init"):
        gn_runtime._require_finite("demo", torch.tensor([1.0, float("nan")], dtype=torch.float64))

    with pytest.raises(FloatingPointError, match="iteration=3"):
        gn_runtime._require_scalar_finite("scalar", float("nan"), iteration=3)


def test_apply_regularization_np_and_diag_preconditioner_cover_all_matrix_kinds():
    vec = np.array([1.0, -2.0, 0.5], dtype=float)

    with pytest.raises(RuntimeError, match="Regularization matrix is not initialized"):
        gn_runtime._apply_regularization_np(SimpleNamespace(R_matrix=None), vec)

    sparse_rec = SimpleNamespace(R_matrix=sparse.diags([1.0, 2.0, 3.0], 0, format="csr"))
    np.testing.assert_allclose(
        gn_runtime._apply_regularization_np(sparse_rec, vec),
        np.array([1.0, -4.0, 1.5], dtype=float),
    )

    linop = LinearOperator((3, 3), matvec=lambda x: 2.0 * np.asarray(x, dtype=float))
    np.testing.assert_allclose(
        gn_runtime._apply_regularization_np(SimpleNamespace(R_matrix=linop), vec),
        2.0 * vec,
    )

    dense_rec = SimpleNamespace(R_matrix=np.diag([4.0, 5.0, 6.0]))
    np.testing.assert_allclose(
        gn_runtime._apply_regularization_np(dense_rec, vec),
        np.array([4.0, -10.0, 3.0], dtype=float),
    )

    J = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]], dtype=float)
    diag_with_r = gn_runtime._diag_preconditioner(
        SimpleNamespace(R_diag=np.array([1.0, 2.0, 3.0], dtype=float)),
        J,
        0.5,
    )
    diag_without_r = gn_runtime._diag_preconditioner(SimpleNamespace(R_diag=np.array([1.0, 2.0])), J, 0.5)
    np.testing.assert_allclose(diag_with_r, np.array([1.75, 6.0, 14.5], dtype=float))
    np.testing.assert_allclose(diag_without_r, np.array([1.75, 5.5, 13.5], dtype=float))


def test_fast_solver_strict_and_lsmr_direct_paths(monkeypatch: pytest.MonkeyPatch):
    J = np.array([[0.8, -0.2, 0.5], [0.1, 0.7, -0.3]], dtype=float)
    residual = np.array([0.1, -0.04], dtype=float)
    de = np.array([0.2, -0.1, 0.08], dtype=float)

    monkeypatch.setattr(gn_runtime, "detect_performance_capabilities", lambda: {})
    monkeypatch.setattr(gn_runtime, "select_preconditioner", lambda *_args, **_kwargs: "diag")
    monkeypatch.setattr(gn_runtime, "select_fused_strategy", lambda **_kwargs: {"enabled": False})
    monkeypatch.setattr(gn_runtime, "select_fast_linear_path", lambda *_args, **_kwargs: "strict")

    recon = _solve_reconstructor(fast_linear_path="strict")
    with pytest.raises(RuntimeError, match="fast_linear_path_requested_strict"):
        gn_runtime._solve_linear_system_fast(
            recon,
            J_weighted_np=J,
            weighted_residual_np=residual,
            de_current_np=de,
            lambda_eff=0.2,
            iteration=0,
        )
    assert recon._last_fast_linear_meta["path"] == "strict-fallback"

    monkeypatch.setattr(gn_runtime, "select_fast_linear_path", lambda *_args, **_kwargs: "pcg")
    monkeypatch.setattr(
        gn_runtime,
        "lsmr",
        lambda *_args, **_kwargs: (np.array([1.0, 2.0, 3.0], dtype=float), None),
    )
    recon_lsmr = _solve_reconstructor(linear_solver="scipy-lsmr")
    delta, _, _ = gn_runtime._solve_linear_system_fast(
        recon_lsmr,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=0.2,
        iteration=1,
    )
    np.testing.assert_allclose(delta, np.array([1.0, 2.0, 3.0], dtype=float))
    assert recon_lsmr._last_fast_linear_meta["path"] == "lsmr-direct"


def test_fast_solver_pyamg_petsc_gamg_woodbury_and_cholmod_limit_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
):
    J = np.array(
        [
            [0.8, -0.2, 0.5],
            [0.1, 0.7, -0.3],
            [0.6, 0.4, 0.2],
        ],
        dtype=float,
    )
    residual = np.array([0.1, -0.04, 0.03], dtype=float)
    de = np.array([0.2, -0.1, 0.08], dtype=float)

    monkeypatch.setattr(gn_runtime, "detect_performance_capabilities", lambda: {"cholmod": False})
    monkeypatch.setattr(gn_runtime, "select_fast_linear_path", lambda *_args, **_kwargs: "pcg")
    monkeypatch.setattr(gn_runtime, "select_fused_strategy", lambda **_kwargs: {"enabled": False})
    monkeypatch.setattr(
        gn_runtime,
        "cg",
        lambda *_args, **_kwargs: (np.array([0.05, -0.01, 0.02], dtype=float), 0),
    )

    monkeypatch.setattr(gn_runtime, "select_preconditioner", lambda *_args, **_kwargs: "pyamg")
    monkeypatch.setattr(gn_runtime, "pyamg", None)
    recon_pyamg = _solve_reconstructor(linear_solver="pyamg-cg", preconditioner="pyamg")
    gn_runtime._solve_linear_system_fast(
        recon_pyamg,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=0.15,
        iteration=2,
    )
    assert "pyamg_unavailable" in str(recon_pyamg._last_fast_linear_meta["fallback_reason"])

    monkeypatch.setattr(gn_runtime, "select_preconditioner", lambda *_args, **_kwargs: "petsc-gamg")
    recon_gamg = _solve_reconstructor(preconditioner="petsc-gamg")
    gn_runtime._solve_linear_system_fast(
        recon_gamg,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=0.15,
        iteration=3,
    )
    assert "petsc_gamg_not_supported_in_matrix_free" in str(recon_gamg._last_fast_linear_meta["fallback_reason"])

    monkeypatch.setattr(gn_runtime, "select_fast_linear_path", lambda *_args, **_kwargs: "woodbury")
    monkeypatch.setattr(gn_runtime, "select_preconditioner", lambda *_args, **_kwargs: "diag")
    recon_woodbury = _solve_reconstructor(
        R_matrix=LinearOperator((3, 3), matvec=lambda x: np.asarray(x, dtype=float)),
        R_diag=np.array([2.0, 3.0, 4.0], dtype=float),
    )
    gn_runtime._solve_linear_system_fast(
        recon_woodbury,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=0.15,
        iteration=4,
    )
    assert "woodbury_requires_diagonal_regularization" in str(
        recon_woodbury._last_fast_linear_meta["fallback_reason"]
    )

    monkeypatch.setattr(gn_runtime, "select_fast_linear_path", lambda *_args, **_kwargs: "pcg")
    monkeypatch.setattr(gn_runtime, "select_preconditioner", lambda *_args, **_kwargs: "cholmod")
    monkeypatch.setattr(gn_runtime, "cholmod_cholesky", lambda _mat: object())
    recon_cholmod = _solve_reconstructor(cholmod_max_n=1)
    gn_runtime._solve_linear_system_fast(
        recon_cholmod,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=0.15,
        iteration=5,
    )
    assert "cholmod_n_limit" in str(recon_cholmod._last_fast_linear_meta["fallback_reason"])


def test_fast_solver_direct_limit_lsmr_fallback_and_terminal_failure(monkeypatch: pytest.MonkeyPatch):
    J = np.array([[0.8, -0.2, 0.5], [0.1, 0.7, -0.3]], dtype=float)
    residual = np.array([0.1, -0.04], dtype=float)
    de = np.array([0.2, -0.1, 0.08], dtype=float)

    monkeypatch.setattr(gn_runtime, "detect_performance_capabilities", lambda: {"cholmod": False})
    monkeypatch.setattr(gn_runtime, "select_preconditioner", lambda *_args, **_kwargs: "diag")
    monkeypatch.setattr(gn_runtime, "select_fused_strategy", lambda **_kwargs: {"enabled": False})
    monkeypatch.setattr(gn_runtime, "select_fast_linear_path", lambda *_args, **_kwargs: "pcg")
    monkeypatch.setattr(gn_runtime, "cg", lambda *_args, **_kwargs: (None, 1))
    monkeypatch.setattr(
        gn_runtime,
        "lsmr",
        lambda *_args, **_kwargs: (np.array([0.2, 0.1, -0.05], dtype=float), None),
    )

    recon_lsmr = _solve_reconstructor()
    gn_runtime._solve_linear_system_fast(
        recon_lsmr,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=0.12,
        iteration=6,
    )
    assert recon_lsmr._last_fast_linear_meta["path"] == "lsmr-fallback"
    assert "pcg_not_converged" in str(recon_lsmr._last_fast_linear_meta["fallback_reason"])

    monkeypatch.setattr(gn_runtime, "lsmr", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad lsmr")))
    recon_fail = _solve_reconstructor(linear_solver="cholmod", cholmod_max_memory_gib=1e-12)
    with pytest.raises(RuntimeError, match="fast_linear_solver_failed"):
        gn_runtime._solve_linear_system_fast(
            recon_fail,
            J_weighted_np=J,
            weighted_residual_np=residual,
            de_current_np=de,
            lambda_eff=0.12,
            iteration=7,
        )
    assert "pcg_not_converged" in str(recon_fail._last_fast_linear_meta["fallback_reason"])


def test_fast_solver_fused_paths_cover_skip_failure_and_success(monkeypatch: pytest.MonkeyPatch):
    J = np.array(
        [
            [0.9, 0.1, -0.3],
            [-0.5, 0.7, 0.4],
            [0.3, -0.2, 0.6],
            [0.2, 0.8, -0.1],
        ],
        dtype=float,
    )
    residual = np.array([0.06, -0.03, 0.01, 0.05], dtype=float)
    de = np.array([0.05, -0.08, 0.12], dtype=float)
    lam = 0.05

    monkeypatch.setattr(gn_runtime, "detect_performance_capabilities", lambda: {"cholmod": False})
    monkeypatch.setattr(gn_runtime, "select_preconditioner", lambda *_args, **_kwargs: "diag")

    monkeypatch.setattr(
        gn_runtime,
        "select_fused_strategy",
        lambda **_kwargs: {"enabled": True, "lowrank": True, "inexact": True, "reason": "unit-fused"},
    )
    monkeypatch.setattr(gn_runtime, "select_fast_linear_path", lambda *_args, **_kwargs: "woodbury")
    recon_skip = _solve_reconstructor(
        rom_mode="auto",
        inexact_mode="on",
        lowrank_mode="on",
        R_matrix=sparse.diags([1.0, 2.0, 3.0], 0, format="csr"),
        R_diag=np.array([1.0, 2.0, 3.0], dtype=float),
    )
    gn_runtime._solve_linear_system_fast(
        recon_skip,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=8,
    )
    assert "fused_skipped:woodbury_optimal" in str(recon_skip._last_fast_linear_meta["fallback_reason"])

    monkeypatch.setattr(gn_runtime, "select_fast_linear_path", lambda *_args, **_kwargs: "pcg")
    monkeypatch.setattr(
        gn_runtime,
        "cg",
        lambda *_args, **_kwargs: (np.array([0.0, 0.0, 0.0], dtype=float), 0),
    )
    recon_small = _solve_reconstructor(
        rom_mode="auto",
        inexact_mode="on",
        lowrank_mode="on",
        R_matrix=np.array([[2.0, 0.5, 0.0], [0.5, 3.0, 0.1], [0.0, 0.1, 4.0]], dtype=float),
    )
    gn_runtime._solve_linear_system_fast(
        recon_small,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=9,
    )
    assert "fused_failed:problem_too_small" in str(recon_small._last_fast_linear_meta["fallback_reason"])

    def fake_merge(*bases, rank_cap=None):
        arrays = [np.asarray(b, dtype=float) for b in bases if isinstance(b, np.ndarray) and b.size > 0]
        if not arrays:
            return np.zeros((J.shape[1], 0), dtype=float)
        merged = np.concatenate(arrays, axis=1)
        if rank_cap is not None:
            merged = merged[:, :rank_cap]
        return merged

    expected = np.linalg.solve(J.T @ J + lam * np.diag([1.0, 2.0, 3.0]), -(J.T @ residual + lam * np.diag([1.0, 2.0, 3.0]) @ de))
    monkeypatch.setattr(gn_runtime, "select_snapshot_matrix", lambda *_args, **_kwargs: np.eye(J.shape[1], 2))
    monkeypatch.setattr(gn_runtime, "compute_pod_basis", lambda *_args, **_kwargs: np.eye(J.shape[1], 2))
    monkeypatch.setattr(gn_runtime, "build_lowrank_subspace", lambda *_args, **_kwargs: (np.eye(J.shape[1], 2), None))
    monkeypatch.setattr(gn_runtime, "merge_orthonormal_bases", fake_merge)
    monkeypatch.setattr(gn_runtime, "build_reduced_operator", lambda **_kwargs: {"ok": True})
    monkeypatch.setattr(gn_runtime, "model_signature_from_forward_model", lambda _fwd: "model")
    monkeypatch.setattr(gn_runtime, "pattern_signature_from_forward_model", lambda _fwd: "pattern")
    monkeypatch.setattr(gn_runtime, "backend_signature_from_forward_model", lambda _fwd: "backend")
    monkeypatch.setattr(
        gn_runtime,
        "solve_reduced_step",
        lambda **_kwargs: (expected, {"linear_residual_ratio": 1.0}),
    )

    recon_fused = _solve_reconstructor(
        rom_mode="on",
        inexact_mode="on",
        lowrank_mode="on",
        inexact_eta0=0.01,
        cache_manager=_EnabledCacheManager(),
        rom_rank_global=2,
        rom_rank_adaptive=1,
        lowrank_rank=2,
        lowrank_energy=0.9,
        rom_refresh_every=1,
    )
    delta, _, _ = gn_runtime._solve_linear_system_fast(
        recon_fused,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=10,
    )
    np.testing.assert_allclose(delta, expected, atol=1e-10, rtol=1e-10)
    assert recon_fused._last_fast_linear_meta["path"] == "fused-rom+inexact+lowrank"
    assert recon_fused._last_fast_linear_meta["rom_enabled_effective"] is True
    assert recon_fused._last_fast_linear_meta["degrade_stage"] == "rom+inexact+lowrank"
    assert getattr(recon_fused, "_force_jacobian_refresh", False) is True
    assert {
        "rom_snapshot_bank",
        "rom_global_basis",
        "rom_adaptive_basis",
        "rom_reduced_operator_absolute",
    }.issubset(set(recon_fused.cache_manager.calls))


def test_fast_solver_fused_no_regularization_signature_and_invalid_mode_paths(monkeypatch: pytest.MonkeyPatch):
    J = np.array([[1.0, 0.2, 0.0], [0.1, 0.8, 0.3], [0.0, 0.4, 0.9]], dtype=float)
    residual = np.array([0.2, -0.1, 0.05], dtype=float)
    de = np.array([0.0, 0.0, 0.0], dtype=float)
    expected = np.linalg.solve(J.T @ J, -(J.T @ residual))

    class _BrokenGeom:
        @property
        def dim(self):
            raise RuntimeError("boom")

    def fake_merge(*bases, rank_cap=None):
        arrays = [np.asarray(b, dtype=float) for b in bases if isinstance(b, np.ndarray) and b.size > 0]
        merged = np.concatenate(arrays, axis=1) if arrays else np.zeros((J.shape[1], 0), dtype=float)
        return merged[:, :rank_cap] if rank_cap is not None else merged

    monkeypatch.setattr(gn_runtime, "detect_performance_capabilities", lambda: {})
    monkeypatch.setattr(gn_runtime, "select_preconditioner", lambda *_args, **_kwargs: "diag")
    monkeypatch.setattr(gn_runtime, "select_fast_linear_path", lambda *_args, **_kwargs: "pcg")
    monkeypatch.setattr(
        gn_runtime,
        "select_fused_strategy",
        lambda **_kwargs: {"enabled": True, "lowrank": False, "inexact": False, "reason": "unit-fused"},
    )
    monkeypatch.setattr(gn_runtime, "select_snapshot_matrix", lambda *_args, **_kwargs: np.eye(J.shape[1], 1))
    monkeypatch.setattr(gn_runtime, "compute_pod_basis", lambda *_args, **_kwargs: np.eye(J.shape[1], 1))
    monkeypatch.setattr(gn_runtime, "merge_orthonormal_bases", fake_merge)
    monkeypatch.setattr(gn_runtime, "_apply_regularization_np", lambda _recon, _vec: np.zeros_like(de))
    monkeypatch.setattr(gn_runtime, "model_signature_from_forward_model", lambda _fwd: "model")
    monkeypatch.setattr(gn_runtime, "pattern_signature_from_forward_model", lambda _fwd: "pattern")
    monkeypatch.setattr(gn_runtime, "backend_signature_from_forward_model", lambda _fwd: "backend")
    monkeypatch.setattr(gn_runtime, "build_reduced_operator", lambda **_kwargs: {"ok": True})
    monkeypatch.setattr(
        gn_runtime,
        "solve_reduced_step",
        lambda **_kwargs: (expected, {"linear_residual_ratio": 0.0}),
    )

    recon = _solve_reconstructor(
        R_matrix=None,
        rom_mode="on",
        inexact_mode="bad-mode",
        lowrank_mode="invalid",
        fwd_model=SimpleNamespace(mesh=SimpleNamespace(geometry=_BrokenGeom())),
    )
    delta, _, _ = gn_runtime._solve_linear_system_fast(
        recon,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=0.1,
        iteration=11,
    )
    np.testing.assert_allclose(delta, expected, atol=1e-10, rtol=1e-10)
    assert recon._last_fast_linear_meta["path"] == "fused-rom"


def test_fast_solver_fused_empty_snapshot_and_empty_basis_fallback(monkeypatch: pytest.MonkeyPatch):
    J = np.array([[0.8, -0.2, 0.5], [0.1, 0.7, -0.3]], dtype=float)
    residual = np.array([0.1, -0.04], dtype=float)
    de = np.array([0.2, -0.1, 0.08], dtype=float)

    monkeypatch.setattr(gn_runtime, "detect_performance_capabilities", lambda: {})
    monkeypatch.setattr(gn_runtime, "select_preconditioner", lambda *_args, **_kwargs: "diag")
    monkeypatch.setattr(gn_runtime, "select_fast_linear_path", lambda *_args, **_kwargs: "pcg")
    monkeypatch.setattr(
        gn_runtime,
        "select_fused_strategy",
        lambda **_kwargs: {"enabled": True, "lowrank": False, "inexact": False, "reason": "unit-fused"},
    )
    monkeypatch.setattr(
        gn_runtime,
        "select_snapshot_matrix",
        lambda *_args, **_kwargs: np.zeros((J.shape[1], 0), dtype=float),
    )
    monkeypatch.setattr(gn_runtime, "cg", lambda *_args, **_kwargs: (np.array([0.1, 0.2, 0.3], dtype=float), 0))

    recon = _solve_reconstructor(rom_mode="on", inexact_mode="off", lowrank_mode="off")
    delta, _, _ = gn_runtime._solve_linear_system_fast(
        recon,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=0.2,
        iteration=12,
    )
    np.testing.assert_allclose(delta, np.array([0.1, 0.2, 0.3], dtype=float))
    assert "fused_failed:empty_basis" in str(recon._last_fast_linear_meta["fallback_reason"])


def test_fast_solver_fused_dense_compute_paths_cover_stage_errors_and_compute_branches(
    monkeypatch: pytest.MonkeyPatch,
):
    J = np.array(
        [
            [0.9, 0.1, -0.3],
            [-0.5, 0.7, 0.4],
            [0.3, -0.2, 0.6],
            [0.2, 0.8, -0.1],
        ],
        dtype=float,
    )
    residual = np.array([0.06, -0.03, 0.01, 0.05], dtype=float)
    de = np.array([0.05, -0.08, 0.12], dtype=float)
    lam = 0.05
    reg = np.diag([2.0, 3.0, 4.0])
    rhs = -(J.T @ residual + lam * reg @ de)
    expected = np.linalg.solve(J.T @ J + lam * reg, rhs)

    def fake_merge(*bases, rank_cap=None):
        arrays = [np.asarray(b, dtype=float) for b in bases if isinstance(b, np.ndarray) and b.size > 0]
        merged = np.concatenate(arrays, axis=1) if arrays else np.zeros((J.shape[1], 0), dtype=float)
        if rank_cap is not None:
            merged = merged[:, :rank_cap]
        return merged

    solve_calls = iter(
        [
            (np.array([np.nan, 0.0, 0.0], dtype=float), {"linear_residual_ratio": 0.0}),
            (expected, {"linear_residual_ratio": 0.0}),
        ]
    )

    monkeypatch.setattr(gn_runtime, "detect_performance_capabilities", lambda: {})
    monkeypatch.setattr(gn_runtime, "select_preconditioner", lambda *_args, **_kwargs: "diag")
    monkeypatch.setattr(gn_runtime, "select_fast_linear_path", lambda *_args, **_kwargs: "pcg")
    monkeypatch.setattr(
        gn_runtime,
        "select_fused_strategy",
        lambda **_kwargs: {"enabled": True, "lowrank": True, "inexact": False, "reason": "unit-fused"},
    )
    monkeypatch.setattr(gn_runtime, "select_snapshot_matrix", lambda *_args, **_kwargs: np.eye(J.shape[1], 2))
    monkeypatch.setattr(gn_runtime, "compute_pod_basis", lambda *_args, **_kwargs: np.eye(J.shape[1], 1))
    monkeypatch.setattr(gn_runtime, "build_lowrank_subspace", lambda *_args, **_kwargs: (np.eye(J.shape[1], 2), None))
    monkeypatch.setattr(gn_runtime, "merge_orthonormal_bases", fake_merge)
    monkeypatch.setattr(gn_runtime, "model_signature_from_forward_model", lambda _fwd: "model")
    monkeypatch.setattr(gn_runtime, "pattern_signature_from_forward_model", lambda _fwd: "pattern")
    monkeypatch.setattr(gn_runtime, "backend_signature_from_forward_model", lambda _fwd: "backend")
    monkeypatch.setattr(gn_runtime, "build_reduced_operator", lambda **_kwargs: {"ok": True})
    monkeypatch.setattr(gn_runtime, "solve_reduced_step", lambda **_kwargs: next(solve_calls))

    recon = _solve_reconstructor(
        R_matrix=reg,
        rom_mode="on",
        inexact_mode="off",
        lowrank_mode="on",
        rom_rank_global=1,
        rom_rank_adaptive=1,
        lowrank_rank=2,
        cache_manager=None,
    )
    delta, _, _ = gn_runtime._solve_linear_system_fast(
        recon,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=13,
    )
    np.testing.assert_allclose(delta, expected, atol=1e-10, rtol=1e-10)
    assert recon._last_fast_linear_meta["path"] == "fused-rom"


def test_fast_solver_fused_paths_cover_adaptive_disabled_and_residual_high(monkeypatch: pytest.MonkeyPatch):
    J = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    residual = np.array([0.5, -0.25], dtype=float)
    de = np.array([0.1, 0.2, 0.3], dtype=float)
    lam = 0.1

    monkeypatch.setattr(gn_runtime, "detect_performance_capabilities", lambda: {})
    monkeypatch.setattr(gn_runtime, "select_preconditioner", lambda *_args, **_kwargs: "diag")
    monkeypatch.setattr(gn_runtime, "select_fast_linear_path", lambda *_args, **_kwargs: "pcg")
    monkeypatch.setattr(
        gn_runtime,
        "select_fused_strategy",
        lambda **_kwargs: {"enabled": True, "lowrank": True, "inexact": False, "reason": "unit-fused"},
    )
    monkeypatch.setattr(gn_runtime, "select_snapshot_matrix", lambda *_args, **_kwargs: np.eye(J.shape[1], 1))
    monkeypatch.setattr(gn_runtime, "compute_pod_basis", lambda *_args, **_kwargs: np.eye(J.shape[1], 1))
    monkeypatch.setattr(gn_runtime, "merge_orthonormal_bases", lambda *bases, rank_cap=None: np.eye(J.shape[1], 1))
    monkeypatch.setattr(gn_runtime, "model_signature_from_forward_model", lambda _fwd: "model")
    monkeypatch.setattr(gn_runtime, "pattern_signature_from_forward_model", lambda _fwd: "pattern")
    monkeypatch.setattr(gn_runtime, "backend_signature_from_forward_model", lambda _fwd: "backend")
    monkeypatch.setattr(gn_runtime, "build_reduced_operator", lambda **_kwargs: {"ok": True})
    monkeypatch.setattr(
        gn_runtime,
        "solve_reduced_step",
        lambda **_kwargs: (np.zeros(J.shape[1], dtype=float), {"linear_residual_ratio": 0.0}),
    )
    monkeypatch.setattr(gn_runtime, "cg", lambda *_args, **_kwargs: (np.array([0.2, 0.1, 0.0], dtype=float), 0))

    recon = _solve_reconstructor(
        R_matrix=sparse.diags([1.0, 2.0, 3.0], 0, format="csr"),
        rom_mode="on",
        inexact_mode="off",
        lowrank_mode="on",
        rom_rank_adaptive=0,
    )
    delta, _, _ = gn_runtime._solve_linear_system_fast(
        recon,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=14,
    )
    np.testing.assert_allclose(delta, np.array([0.2, 0.1, 0.0], dtype=float))
    assert "fused_failed:" in str(recon._last_fast_linear_meta["fallback_reason"])


def test_fast_solver_fused_stage_basis_empty_then_success(monkeypatch: pytest.MonkeyPatch):
    J = np.array([[0.7, -0.1, 0.2], [0.0, 0.9, -0.4], [0.3, 0.2, 0.8]], dtype=float)
    residual = np.array([0.1, -0.02, 0.04], dtype=float)
    de = np.array([0.02, -0.01, 0.03], dtype=float)
    lam = 0.15
    reg = sparse.diags([1.0, 2.0, 3.0], 0, format="csr")
    rhs = -(J.T @ residual + lam * reg.diagonal() * de)
    expected = np.linalg.solve(J.T @ J + lam * np.diag(reg.diagonal()), rhs)

    merge_calls = {"count": 0}

    def fake_merge(*bases, rank_cap=None):
        merge_calls["count"] += 1
        if merge_calls["count"] == 2:
            return np.zeros((J.shape[1], 0), dtype=float)
        arrays = [np.asarray(b, dtype=float) for b in bases if isinstance(b, np.ndarray) and b.size > 0]
        merged = np.concatenate(arrays, axis=1) if arrays else np.zeros((J.shape[1], 0), dtype=float)
        if rank_cap is not None:
            merged = merged[:, :rank_cap]
        return merged

    solve_calls = iter([(expected, {"linear_residual_ratio": 0.0})])

    monkeypatch.setattr(gn_runtime, "detect_performance_capabilities", lambda: {})
    monkeypatch.setattr(gn_runtime, "select_preconditioner", lambda *_args, **_kwargs: "diag")
    monkeypatch.setattr(gn_runtime, "select_fast_linear_path", lambda *_args, **_kwargs: "pcg")
    monkeypatch.setattr(
        gn_runtime,
        "select_fused_strategy",
        lambda **_kwargs: {"enabled": True, "lowrank": True, "inexact": False, "reason": "unit-fused"},
    )
    monkeypatch.setattr(gn_runtime, "select_snapshot_matrix", lambda *_args, **_kwargs: np.eye(J.shape[1], 1))
    monkeypatch.setattr(gn_runtime, "compute_pod_basis", lambda *_args, **_kwargs: np.eye(J.shape[1], 1))
    monkeypatch.setattr(gn_runtime, "build_lowrank_subspace", lambda *_args, **_kwargs: (np.eye(J.shape[1], 1), None))
    monkeypatch.setattr(gn_runtime, "merge_orthonormal_bases", fake_merge)
    monkeypatch.setattr(gn_runtime, "model_signature_from_forward_model", lambda _fwd: "model")
    monkeypatch.setattr(gn_runtime, "pattern_signature_from_forward_model", lambda _fwd: "pattern")
    monkeypatch.setattr(gn_runtime, "backend_signature_from_forward_model", lambda _fwd: "backend")
    monkeypatch.setattr(gn_runtime, "build_reduced_operator", lambda **_kwargs: {"ok": True})
    monkeypatch.setattr(gn_runtime, "solve_reduced_step", lambda **_kwargs: next(solve_calls))

    recon = _solve_reconstructor(
        R_matrix=reg,
        rom_mode="on",
        inexact_mode="off",
        lowrank_mode="on",
        rom_rank_adaptive=1,
    )
    delta, _, _ = gn_runtime._solve_linear_system_fast(
        recon,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=15,
    )
    np.testing.assert_allclose(delta, expected, atol=1e-10, rtol=1e-10)


def test_fast_solver_cholmod_pyamg_and_auto_reason_paths(monkeypatch: pytest.MonkeyPatch):
    J = np.array([[0.8, -0.2, 0.5], [0.1, 0.7, -0.3]], dtype=float)
    residual = np.array([0.1, -0.04], dtype=float)
    de = np.array([0.2, -0.1, 0.08], dtype=float)

    monkeypatch.setattr(gn_runtime, "select_fast_linear_path", lambda *_args, **_kwargs: "pcg")
    monkeypatch.setattr(gn_runtime, "select_fused_strategy", lambda **_kwargs: {"enabled": False})
    monkeypatch.setattr(gn_runtime, "cg", lambda *_args, **_kwargs: (np.array([0.05, -0.01, 0.02], dtype=float), 0))

    monkeypatch.setattr(gn_runtime, "detect_performance_capabilities", lambda: {"cholmod": False})
    monkeypatch.setattr(gn_runtime, "select_preconditioner", lambda *_args, **_kwargs: "cholmod")
    monkeypatch.setattr(gn_runtime, "cholmod_cholesky", None)
    recon_unavailable = _solve_reconstructor(R_matrix=sparse.diags([1.0, 2.0, 3.0], 0, format="csr"))
    gn_runtime._solve_linear_system_fast(
        recon_unavailable,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=0.12,
        iteration=16,
    )
    assert "cholmod_unavailable" in str(recon_unavailable._last_fast_linear_meta["fallback_reason"])

    monkeypatch.setattr(gn_runtime, "cholmod_cholesky", lambda _mat: object())
    recon_not_sparse = _solve_reconstructor(R_matrix=np.diag([1.0, 2.0, 3.0]))
    gn_runtime._solve_linear_system_fast(
        recon_not_sparse,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=0.12,
        iteration=17,
    )
    assert "regularization_not_sparse" in str(recon_not_sparse._last_fast_linear_meta["fallback_reason"])

    recon_mem = _solve_reconstructor(
        R_matrix=sparse.diags([1.0, 2.0, 3.0], 0, format="csr"),
        cholmod_max_memory_gib=1e-12,
    )
    gn_runtime._solve_linear_system_fast(
        recon_mem,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=0.12,
        iteration=18,
    )
    assert "cholmod_memory_limit" in str(recon_mem._last_fast_linear_meta["fallback_reason"])

    monkeypatch.setattr(gn_runtime, "select_preconditioner", lambda *_args, **_kwargs: "pyamg")
    monkeypatch.setattr(gn_runtime, "pyamg", object())
    recon_pyamg_dense = _solve_reconstructor(R_matrix=np.diag([1.0, 2.0, 3.0]))
    gn_runtime._solve_linear_system_fast(
        recon_pyamg_dense,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=0.12,
        iteration=19,
    )
    assert "pyamg_requires_sparse_regularization" in str(
        recon_pyamg_dense._last_fast_linear_meta["fallback_reason"]
    )

    monkeypatch.setattr(gn_runtime, "detect_performance_capabilities", lambda: {"cholmod": True})
    monkeypatch.setattr(gn_runtime, "select_preconditioner", lambda *_args, **_kwargs: "diag")
    recon_auto_reason = _solve_reconstructor(R_matrix=sparse.diags([1.0, 2.0, 3.0], 0, format="csr"))
    gn_runtime._solve_linear_system_fast(
        recon_auto_reason,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=0.12,
        iteration=20,
    )
    assert recon_auto_reason._last_fast_linear_meta["fast_linear_path_reason"] == "auto:sparse_spd_with_cholmod"


def test_fast_solver_rom_off_cholmod_direct_and_woodbury_failure_paths(monkeypatch: pytest.MonkeyPatch):
    J = np.array([[0.8, -0.2, 0.5], [0.1, 0.7, -0.3], [0.6, 0.4, 0.2]], dtype=float)
    residual = np.array([0.1, -0.04, 0.03], dtype=float)
    de = np.array([0.2, -0.1, 0.08], dtype=float)
    reg = sparse.diags([1.0, 2.0, 3.0], 0, format="csr")

    monkeypatch.setattr(gn_runtime, "detect_performance_capabilities", lambda: {"cholmod": True})
    monkeypatch.setattr(gn_runtime, "select_preconditioner", lambda *_args, **_kwargs: "diag")
    monkeypatch.setattr(gn_runtime, "select_fused_strategy", lambda **_kwargs: {"enabled": True, "lowrank": False, "inexact": False})
    monkeypatch.setattr(gn_runtime, "select_fast_linear_path", lambda *_args, **_kwargs: "pcg")
    monkeypatch.setattr(gn_runtime, "cg", lambda *_args, **_kwargs: (np.array([0.05, -0.01, 0.02], dtype=float), 0))

    recon_rom_off = _solve_reconstructor(R_matrix=reg, rom_mode="off")
    gn_runtime._solve_linear_system_fast(
        recon_rom_off,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=0.15,
        iteration=21,
    )
    assert "fused_failed:rom_off" in str(recon_rom_off._last_fast_linear_meta["fallback_reason"])

    class _Factor:
        def solve_A(self, rhs):
            return np.asarray(rhs, dtype=float)

    monkeypatch.setattr(gn_runtime, "select_fused_strategy", lambda **_kwargs: {"enabled": False})
    monkeypatch.setattr(gn_runtime, "cholmod_cholesky", lambda _mat: _Factor())
    recon_cholmod_direct = _solve_reconstructor(R_matrix=reg, linear_solver="cholmod", solver_mode="fast")
    delta_direct, _, _ = gn_runtime._solve_linear_system_fast(
        recon_cholmod_direct,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=0.15,
        iteration=22,
    )
    assert recon_cholmod_direct._last_fast_linear_meta["path"] == "cholmod-direct"
    assert delta_direct.shape == (3,)

    monkeypatch.setattr(gn_runtime, "select_fast_linear_path", lambda *_args, **_kwargs: "woodbury")
    monkeypatch.setattr(
        gn_runtime,
        "cho_factor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("woodbury failed")),
    )
    recon_woodbury_fail = _solve_reconstructor(R_matrix=reg, linear_solver="auto", solver_mode="fast")
    gn_runtime._solve_linear_system_fast(
        recon_woodbury_fail,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=0.15,
        iteration=23,
    )
    assert "woodbury_failed:RuntimeError" in str(recon_woodbury_fail._last_fast_linear_meta["fallback_reason"])


def test_fast_solver_fused_linear_operator_signature_and_blank_error_detail(monkeypatch: pytest.MonkeyPatch):
    J = np.array([[1.0, 0.2, 0.1], [0.3, 0.8, -0.2], [0.0, 0.4, 0.9]], dtype=float)
    residual = np.array([0.2, -0.05, 0.03], dtype=float)
    de = np.array([0.1, -0.02, 0.04], dtype=float)
    lam = 0.1
    rhs = -(J.T @ residual + lam * de)
    expected = np.linalg.solve(J.T @ J + lam * np.eye(J.shape[1]), rhs)

    class _BlankError(RuntimeError):
        def __str__(self) -> str:
            return ""

    def fake_merge(*bases, rank_cap=None):
        arrays = [np.asarray(b, dtype=float) for b in bases if isinstance(b, np.ndarray) and b.size > 0]
        merged = np.concatenate(arrays, axis=1) if arrays else np.zeros((J.shape[1], 0), dtype=float)
        if rank_cap is not None:
            merged = merged[:, :rank_cap]
        return merged

    solve_calls = {"count": 0}

    def solve_reduced(**_kwargs):
        solve_calls["count"] += 1
        if solve_calls["count"] == 1:
            raise _BlankError()
        return expected, {"linear_residual_ratio": 0.0}

    monkeypatch.setattr(gn_runtime, "detect_performance_capabilities", lambda: {})
    monkeypatch.setattr(gn_runtime, "select_preconditioner", lambda *_args, **_kwargs: "diag")
    monkeypatch.setattr(gn_runtime, "select_fast_linear_path", lambda *_args, **_kwargs: "pcg")
    monkeypatch.setattr(
        gn_runtime,
        "select_fused_strategy",
        lambda **_kwargs: {"enabled": True, "lowrank": True, "inexact": False, "reason": "unit-fused"},
    )
    monkeypatch.setattr(gn_runtime, "select_snapshot_matrix", lambda *_args, **_kwargs: np.eye(J.shape[1], 1))
    monkeypatch.setattr(gn_runtime, "compute_pod_basis", lambda *_args, **_kwargs: np.eye(J.shape[1], 1))
    monkeypatch.setattr(gn_runtime, "build_lowrank_subspace", lambda *_args, **_kwargs: (np.eye(J.shape[1], 1), None))
    monkeypatch.setattr(gn_runtime, "merge_orthonormal_bases", fake_merge)
    monkeypatch.setattr(gn_runtime, "model_signature_from_forward_model", lambda _fwd: "model")
    monkeypatch.setattr(gn_runtime, "pattern_signature_from_forward_model", lambda _fwd: "pattern")
    monkeypatch.setattr(gn_runtime, "backend_signature_from_forward_model", lambda _fwd: "backend")
    monkeypatch.setattr(gn_runtime, "build_reduced_operator", lambda **_kwargs: {"ok": True})
    monkeypatch.setattr(gn_runtime, "solve_reduced_step", solve_reduced)

    recon = _solve_reconstructor(
        R_matrix=LinearOperator((3, 3), matvec=lambda x: np.asarray(x, dtype=float)),
        rom_mode="on",
        inexact_mode="off",
        lowrank_mode="on",
        rom_rank_adaptive=1,
    )
    delta, _, _ = gn_runtime._solve_linear_system_fast(
        recon,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=24,
    )
    np.testing.assert_allclose(delta, expected, atol=1e-10, rtol=1e-10)


def test_fast_solver_cholmod_direct_reason_limits(monkeypatch: pytest.MonkeyPatch):
    J = np.array([[0.8, -0.2, 0.5], [0.1, 0.7, -0.3], [0.6, 0.4, 0.2]], dtype=float)
    residual = np.array([0.1, -0.04, 0.03], dtype=float)
    de = np.array([0.2, -0.1, 0.08], dtype=float)

    monkeypatch.setattr(gn_runtime, "detect_performance_capabilities", lambda: {"cholmod": True})
    monkeypatch.setattr(gn_runtime, "select_preconditioner", lambda *_args, **_kwargs: "diag")
    monkeypatch.setattr(gn_runtime, "select_fused_strategy", lambda **_kwargs: {"enabled": False})
    monkeypatch.setattr(gn_runtime, "select_fast_linear_path", lambda *_args, **_kwargs: "pcg")
    monkeypatch.setattr(gn_runtime, "cg", lambda *_args, **_kwargs: (np.array([0.05, -0.01, 0.02], dtype=float), 0))
    monkeypatch.setattr(gn_runtime, "cholmod_cholesky", lambda _mat: object())

    recon_not_sparse = _solve_reconstructor(R_matrix=np.diag([1.0, 2.0, 3.0]), linear_solver="cholmod")
    gn_runtime._solve_linear_system_fast(
        recon_not_sparse,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=0.12,
        iteration=25,
    )
    assert "regularization_not_sparse" in str(recon_not_sparse._last_fast_linear_meta["fallback_reason"])

    recon_n_limit = _solve_reconstructor(
        R_matrix=sparse.diags([1.0, 2.0, 3.0], 0, format="csr"),
        linear_solver="cholmod",
        cholmod_max_n=1,
    )
    gn_runtime._solve_linear_system_fast(
        recon_n_limit,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=0.12,
        iteration=26,
    )
    assert "cholmod_n_limit" in str(recon_n_limit._last_fast_linear_meta["fallback_reason"])

    recon_mem_limit = _solve_reconstructor(
        R_matrix=sparse.diags([1.0, 2.0, 3.0], 0, format="csr"),
        linear_solver="cholmod",
        cholmod_max_memory_gib=1e-12,
    )
    gn_runtime._solve_linear_system_fast(
        recon_mem_limit,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=0.12,
        iteration=27,
    )
    assert "cholmod_memory_limit" in str(recon_mem_limit._last_fast_linear_meta["fallback_reason"])
