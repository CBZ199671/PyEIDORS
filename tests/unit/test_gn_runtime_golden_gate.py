"""T77 phase 1.5 functional golden gate — behavioural lock for commit #4+.

The string-only contract gate in
:mod:`tests.unit.test_gn_runtime_contract_freeze` only stops a sub-
module split commit from renaming things. This file adds the
behavioural complement: fixed inputs → fixed outputs (delta vector,
diagnostic ``matrix_free_pc_source``, fallback reason, linear iteration
count, startup cache SHA256, iteration log payload). A future ``linear_system.py`` /
``step_size.py`` extract that silently changes algorithm tolerances
or skips a fallback branch will fail these assertions even if every
string literal in the contract gate is preserved.

Three deliberately small fixtures plus an iteration-log passthrough.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import diags

from pyeidors.inverse.solvers.gauss_newton_runtime import (
    _IterationLog,
    _solve_linear_system_fast,
    _startup_cache_payload,
)


def _dense_fixture():
    J = np.array(
        [
            [0.8, -0.2, 0.5],
            [0.1, 0.7, -0.3],
            [0.6, 0.4, 0.2],
            [-0.2, 0.3, 0.9],
        ],
        dtype=np.float64,
    )
    residual = np.array([0.1, -0.04, 0.03, 0.05], dtype=np.float64)
    de = np.array([0.2, -0.1, 0.08], dtype=np.float64)
    lam = 0.15
    return J, residual, de, lam


def _auto_reconstructor() -> SimpleNamespace:
    return SimpleNamespace(
        R_matrix=diags([1.0, 2.0, 3.0], 0, format="csr"),
        R_diag=np.array([1.0, 2.0, 3.0], dtype=float),
        use_prior_term=True,
        performance_mode="aggressive",
        linear_solver="auto",
        preconditioner="diag",
        fast_linear_path="auto",
        cholmod_max_n=12000,
        cholmod_max_memory_gib=4.0,
    )


def _expected_dense_solution(
    J: np.ndarray, residual: np.ndarray, de: np.ndarray, lam: float
) -> np.ndarray:
    R = np.diag([1.0, 2.0, 3.0])
    A = J.T @ J + lam * R
    rhs = -(J.T @ residual + lam * (R @ de))
    return np.linalg.solve(A, rhs)


def test_solve_linear_system_fast_auto_config_returns_locked_delta_and_meta() -> None:
    """Auto fast-PCG path: delta + (delta_norm, jtr_norm) + pc_source pinned."""
    reconstructor = _auto_reconstructor()
    J, residual, de, lam = _dense_fixture()
    delta, delta_norm, jtr_norm = _solve_linear_system_fast(
        reconstructor,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=0,
    )

    expected_delta = _expected_dense_solution(J, residual, de, lam)
    np.testing.assert_allclose(delta, expected_delta, rtol=1e-10, atol=1e-12)

    assert np.isclose(delta_norm, float(np.linalg.norm(expected_delta)), rtol=1e-10)
    expected_jtr_norm = float(np.linalg.norm(J.T @ residual))
    assert np.isclose(jtr_norm, expected_jtr_norm, rtol=1e-10)

    meta = getattr(reconstructor, "_last_fast_linear_meta", None)
    assert isinstance(meta, dict)
    assert meta.get("matrix_free_pc_source") == "dense-sensitivity", meta
    assert meta.get("linear_iterations") == EXPECTED_WOODBURY_LINEAR_ITERATIONS, meta


def test_solve_linear_system_fast_petsc_backend_fallback_reason_literal_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V12: when petsc4py is unavailable, the fallback reason is bytewise stable.

    Mirrors the matrix-free trigger used by
    ``tests.unit.test_gn_fast_linear_solver`` — an explicit
    ``fast_linear_path="pcg"`` plus a ``LinearOperator`` Jacobian
    plus ``matrix_free_ksp_backend="petsc"`` plus
    ``gn_runtime._PETSc`` patched to None. Verifies the literal fallback
    reason string ``petsc_backend_unavailable`` AND that the scipy
    fallback delta still matches the dense reference within tol.
    """
    from scipy.sparse.linalg import LinearOperator

    import pyeidors.inverse.solvers.gauss_newton_runtime as gn_runtime

    monkeypatch.setattr(gn_runtime, "_PETSc", None)

    J, residual, de, lam = _dense_fixture()
    op = LinearOperator(
        J.shape,
        matvec=lambda x: J @ np.asarray(x, dtype=float),
        rmatvec=lambda x: J.T @ np.asarray(x, dtype=float),
        dtype=np.float64,
    )

    reconstructor = _auto_reconstructor()
    reconstructor.fast_linear_path = "pcg"
    reconstructor.matrix_free_ksp_backend = "petsc"

    delta, _, _ = _solve_linear_system_fast(
        reconstructor,
        J_weighted_np=op,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=0,
    )

    meta = getattr(reconstructor, "_last_fast_linear_meta", {})
    assert meta.get("matrix_free_ksp_backend_requested") == "petsc"
    assert meta.get("matrix_free_ksp_backend_effective") == "scipy"
    assert (
        meta.get("matrix_free_ksp_backend_fallback_reason")
        == "petsc_backend_unavailable"
    ), meta
    assert meta.get("linear_iterations") == EXPECTED_FAST_PCG_LINEAR_ITERATIONS, meta

    expected_delta = _expected_dense_solution(J, residual, de, lam)
    np.testing.assert_allclose(delta, expected_delta, rtol=1e-5, atol=1e-7)


def _fake_signature(_fwd_model) -> str:
    return "stub-signature"


EXPECTED_WOODBURY_LINEAR_ITERATIONS = 0
EXPECTED_FAST_PCG_LINEAR_ITERATIONS = 3
EXPECTED_STARTUP_CACHE_PAYLOAD_SHA256 = (
    "b73d9dfd0d440872a6920a0c46c102afc74025e33232c959677ef3ed34d916f5"
)


def test_v594_startup_cache_payload_streams_noncontiguous_sigma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyeidors.inverse.solvers.gauss_newton_startup_cache as startup_cache

    monkeypatch.setattr(
        startup_cache, "model_signature_from_forward_model", _fake_signature
    )
    monkeypatch.setattr(
        startup_cache, "pattern_signature_from_forward_model", _fake_signature
    )
    monkeypatch.setattr(
        startup_cache, "backend_signature_from_forward_model", _fake_signature
    )

    sigma_view = np.arange(24, dtype=np.float64).reshape(8, 3)[:, 1]
    assert not sigma_view.flags.c_contiguous
    expected = startup_cache.hash_array_payload(
        np.ascontiguousarray(sigma_view, dtype=np.float64)
    )
    original_hash = startup_cache.hash_array_payload
    captured: dict[str, np.ndarray] = {}

    def _capture_hash(arr: np.ndarray, *, prefix: bytes = b"") -> str:
        captured["arr"] = arr
        return original_hash(arr, prefix=prefix)

    monkeypatch.setattr(startup_cache, "hash_array_payload", _capture_hash)

    reconstructor = SimpleNamespace(fwd_model=SimpleNamespace(), solver_mode="fast")
    payload = _startup_cache_payload(reconstructor, sigma_view, "efficient")

    assert payload["sigma_hash"] == expected
    assert captured["arr"] is sigma_view
    assert captured["arr"].dtype == np.float64
    assert not captured["arr"].flags.c_contiguous


def test_startup_cache_payload_sha256_locked_for_synthetic_reconstructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V36-style: cache-key payload + canonical JSON encoding byte-stable."""
    import pyeidors.inverse.solvers.gauss_newton_startup_cache as startup_cache

    monkeypatch.setattr(
        startup_cache, "model_signature_from_forward_model", _fake_signature
    )
    monkeypatch.setattr(
        startup_cache, "pattern_signature_from_forward_model", _fake_signature
    )
    monkeypatch.setattr(
        startup_cache, "backend_signature_from_forward_model", _fake_signature
    )

    reconstructor = SimpleNamespace(fwd_model=SimpleNamespace(), solver_mode="fast")
    sigma_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    payload = _startup_cache_payload(reconstructor, sigma_array, "efficient")

    assert set(payload.keys()) == {
        "solver",
        "mode",
        "jacobian_method",
        "sigma_hash",
        "model_signature",
        "pattern_signature",
        "backend_signature",
        "solver_config",
    }
    assert payload["solver"] == "gn_absolute"
    assert payload["mode"] == "fast"
    assert payload["jacobian_method"] == "efficient"
    assert (
        payload["sigma_hash"]
        == hashlib.sha256(
            np.ascontiguousarray(sigma_array, dtype=np.float64).tobytes()
        ).hexdigest()
    )
    complex_sigma = np.array([1.0 + 0.25j, 2.0 - 0.5j], dtype=np.complex64)
    complex_payload = _startup_cache_payload(reconstructor, complex_sigma, "efficient")
    complex_sigma_values = np.ascontiguousarray(complex_sigma, dtype=np.complex64)
    assert (
        complex_payload["sigma_hash"]
        == hashlib.sha256(b"complex64\0" + complex_sigma_values.tobytes()).hexdigest()
    )
    assert ".tobytes(" not in inspect.getsource(_startup_cache_payload)

    expected_solver_config = {
        "linear_solver": "auto",
        "preconditioner": "auto",
        "line_search_mode": "full",
        "jacobian_update_every": 1,
        "jacobian_reuse_tol": 0.0,
        "rom_mode": "off",
        "rom_rank_global": 32,
        "rom_rank_adaptive": 16,
        "rom_refresh_every": 2,
        "rom_snapshot_source": "hybrid",
        "inexact_mode": "off",
        "inexact_forcing": "eisenstat-walker",
        "inexact_eta0": 0.2,
        "inexact_eta_min": 1e-3,
        "inexact_eta_max": 0.5,
        "lowrank_mode": "off",
        "lowrank_rank": 16,
        "lowrank_method": "tsvd",
        "lowrank_energy": 0.995,
    }
    assert payload["solver_config"] == expected_solver_config

    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    assert digest == EXPECTED_STARTUP_CACHE_PAYLOAD_SHA256


def test_iteration_log_to_payload_passes_through_known_input_values() -> None:
    """V73 contract: iteration log dict bytewise stable for known input."""
    log = _IterationLog(
        iteration=3,
        residual=0.5,
        residual_weighted=0.4,
        relative_residual=0.25,
        relative_residual_weighted=0.2,
        residual_max=0.6,
        meas_norm=1.5,
        pred_norm=1.4,
        meas_max=2.0,
        pred_max=1.9,
        jtr_norm=0.7,
        delta_norm=0.05,
        step=0.95,
        lambda_eff=0.01,
        relative_change=0.02,
        res_drop=0.1,
        meas_misfit=0.3,
        prior_misfit=0.05,
        total_objective=0.35,
    )
    assert log.to_payload() == {
        "iteration": 3,
        "residual": 0.5,
        "residual_weighted": 0.4,
        "relative_residual": 0.25,
        "relative_residual_weighted": 0.2,
        "residual_max": 0.6,
        "meas_norm": 1.5,
        "pred_norm": 1.4,
        "meas_max": 2.0,
        "pred_max": 1.9,
        "JTr_norm": 0.7,
        "delta_norm": 0.05,
        "step": 0.95,
        "lambda_eff": 0.01,
        "relative_change": 0.02,
        "res_drop": 0.1,
        "meas_misfit": 0.3,
        "prior_misfit": 0.05,
        "total_objective": 0.35,
    }
