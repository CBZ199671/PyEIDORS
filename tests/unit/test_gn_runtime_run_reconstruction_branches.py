"""Targeted branch tests for run_reconstruction with lightweight stubs."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import pyeidors.inverse.solvers.gauss_newton_runtime as gn_runtime
from pyeidors.inverse.jacobian.linearized import JacobianLinearization


class _Progress:
    def __init__(self):
        self.postfix: list[str] = []
        self.updates: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        _ = (exc_type, exc, tb)
        return False

    def set_postfix_str(self, value: str) -> None:
        self.postfix.append(str(value))

    def update(self, value: int) -> None:
        self.updates.append(int(value))


class _Sigma:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)


def _make_reconstructor(
    *, max_iterations: int, verbose: bool, n_elements: int
) -> tuple[SimpleNamespace, _Progress]:
    progress = _Progress()
    recon = SimpleNamespace(
        n_measurements=2,
        regularization_param=0.1,
        hyperparameter=0.01,
        verbose=bool(verbose),
        max_iterations=int(max_iterations),
        min_iterations=1,
        convergence_tol=-1.0,
        solver_mode="fast",
        linear_solver="auto",
        line_search_mode="full",
        preconditioner="diag",
        rom_mode="off",
        rom_snapshot_source="hybrid",
        inexact_mode="off",
        inexact_forcing="eisenstat-walker",
        lowrank_mode="off",
        lowrank_method="tsvd",
        jacobian_reuse_tol=1e-6,
        jacobian_update_every=4,
        n_elements=int(n_elements),
        negate_jacobian=False,
        clip_values=None,
        difference_mode="normalized",
        difference_orientation="target_minus_reference",
        difference_step_size_mode="off",
        best_homog_mode="off",
        active_preset_name="",
        jacobian_background_conductivity=1.0,
        device_requested="cpu",
        device_effective="cpu",
        device="cpu",
        cache_manager=None,
        regularization=object(),
        _progress=lambda total: progress,
        ensure_regularization_ready=lambda: None,
        _ensure_measurement_weights=lambda _sigma: None,
        fwd_model=SimpleNamespace(
            fwd_solve=lambda _img: (
                SimpleNamespace(meas=np.array([0.5, 0.4], dtype=float)),
                None,
            ),
            linear_backend="petsc",
            _last_cache_lookup={},
            _petsc_backend_info={},
            get_backend_diagnostics=lambda: {},
        ),
        jacobian_calculator=SimpleNamespace(
            calculate=lambda _sigma, method=None: np.eye(2, dtype=float),
            block_tuning_info=lambda: {},
            _last_cache_lookup={},
        ),
    )
    return recon, progress


def _install_common_runtime_stubs(
    monkeypatch: pytest.MonkeyPatch, reconstructor: SimpleNamespace
) -> None:
    monkeypatch.setattr(
        gn_runtime,
        "_extract_measured_vector",
        lambda measured: np.asarray(measured, dtype=float),
    )
    monkeypatch.setattr(
        gn_runtime, "_configure_measurement_space", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        gn_runtime,
        "_to_runtime_tensor",
        lambda _recon, value: torch.as_tensor(
            np.asarray(value, dtype=float), dtype=torch.float64
        ),
    )
    monkeypatch.setattr(
        gn_runtime,
        "_to_runtime_tensor_cached",
        lambda _recon, _name, value: torch.as_tensor(
            np.asarray(value, dtype=float), dtype=torch.float64
        ),
    )
    monkeypatch.setattr(
        gn_runtime,
        "_estimate_best_homogeneous_conductivity",
        lambda *_args, **_kwargs: {"mode": "off", "applied": False, "value": None},
    )
    monkeypatch.setattr(
        gn_runtime,
        "_init_sigma_function",
        lambda _recon, initial: (_Sigma([1.0, 1.0]), float(initial)),
    )
    monkeypatch.setattr(
        gn_runtime,
        "function_get_array",
        lambda sigma: np.asarray(sigma.values, dtype=float),
    )
    monkeypatch.setattr(
        gn_runtime,
        "function_set_array",
        lambda sigma, values: setattr(
            sigma, "values", np.asarray(values, dtype=float).copy()
        ),
    )
    monkeypatch.setattr(
        gn_runtime,
        "_prepare_prior",
        lambda *_args, **_kwargs: torch.zeros(2, dtype=torch.float64),
    )
    monkeypatch.setattr(
        gn_runtime,
        "_project_measurement_jacobian",
        lambda _recon, jac: np.asarray(jac, dtype=float),
    )
    monkeypatch.setattr(
        gn_runtime,
        "_project_simulated_measurements",
        lambda _recon, meas: np.asarray(meas, dtype=float),
    )
    monkeypatch.setattr(
        gn_runtime,
        "_build_linear_system",
        lambda _recon, JTJ, JTr, de_torch, lambda_eff, iteration, RtR_de=None: (
            torch.eye(2, dtype=torch.float64),
            torch.tensor([0.1, 0.05], dtype=torch.float64),
        ),
    )
    monkeypatch.setattr(
        gn_runtime,
        "_solve_linear_system",
        lambda _recon, A, b, JTJ, iteration: (
            torch.tensor([0.05, 0.02], dtype=torch.float64),
            float(np.linalg.norm([0.05, 0.02])),
        ),
    )
    monkeypatch.setattr(
        gn_runtime,
        "_record_iteration_log",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        gn_runtime,
        "_apply_difference_step_size",
        lambda _recon, sigma_final, measured_vector: (
            np.asarray(sigma_final, dtype=float),
            {"mode": "off", "applied": False, "value": 1.0},
        ),
    )

    residual_calls = {"count": 0}

    def fake_compute_residuals(_recon, simulated_measurement, meas_torch, iteration):
        residual_calls["count"] += 1
        base = 1.0 - (0.1 * residual_calls["count"])
        data_sim = torch.tensor([0.5, 0.4], dtype=torch.float64)
        residual = torch.tensor([0.2, -0.1], dtype=torch.float64)
        weighted = residual.clone()
        return data_sim, residual, weighted, float(base), float(base), 0.2

    monkeypatch.setattr(gn_runtime, "_compute_residuals", fake_compute_residuals)
    monkeypatch.setattr(
        gn_runtime,
        "_compute_objective",
        lambda *_args, **_kwargs: (0.1, 0.2, 0.3, None),
    )
    monkeypatch.setattr(
        gn_runtime,
        "_startup_cache_lookup",
        lambda *_args, **_kwargs: (np.eye(2, dtype=float), {"hit": False}),
    )
    reconstructor._meas_weight_sqrt = None


def test_run_reconstruction_covers_large_problem_reuse_and_verbose_reuse_info(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    recon, progress = _make_reconstructor(
        max_iterations=2, verbose=True, n_elements=6000
    )
    _install_common_runtime_stubs(monkeypatch, recon)

    monkeypatch.setattr(
        gn_runtime,
        "_solve_linear_system_fast",
        lambda reconstructor, **_kwargs: (
            reconstructor.__dict__.__setitem__(
                "_last_fast_linear_meta",
                {
                    "path": "pcg-diag-precond",
                    "resolved_preconditioner": "diag",
                    "fallback_reason": "demo-reason",
                    "fast_linear_path_selected": "pcg",
                    "fast_linear_path_reason": "auto:matrix_free_pcg",
                },
            )
            or (
                np.array([0.1, 0.2], dtype=float),
                float(np.linalg.norm([0.1, 0.2])),
                0.5,
            )
        ),
    )
    monkeypatch.setattr(gn_runtime, "_select_step_size", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(
        gn_runtime, "_maybe_rollback", lambda *_args, **_kwargs: (False, False, 0)
    )

    results = gn_runtime.run_reconstruction(
        recon,
        measured_data=np.array([1.0, 0.8], dtype=float),
    )

    output = capsys.readouterr().out
    assert "[INFO] iteration=1: reused Jacobian (fast mode)" in output
    assert results.iterations == 2
    assert results.diagnostics["backend_info"]["fallback_reason"] == "demo-reason"
    assert progress.postfix


def test_run_reconstruction_covers_force_refresh_fast_exception_and_rollback_paths(
    monkeypatch: pytest.MonkeyPatch,
):
    recon, _progress = _make_reconstructor(
        max_iterations=3, verbose=False, n_elements=100
    )
    _install_common_runtime_stubs(monkeypatch, recon)

    jacobian_calls = {"count": 0}

    def calculate(_sigma, method=None):
        _ = method
        jacobian_calls["count"] += 1
        return np.eye(2, dtype=float)

    recon.jacobian_calculator.calculate = calculate

    solve_calls = {"count": 0}

    def fake_fast_solver(reconstructor, **_kwargs):
        solve_calls["count"] += 1
        if solve_calls["count"] == 2:
            raise RuntimeError("boom")
        reconstructor._last_fast_linear_meta = {
            "path": "pcg-diag-precond",
            "resolved_preconditioner": "diag",
            "fast_linear_path_selected": "pcg",
            "fast_linear_path_reason": "auto:matrix_free_pcg",
        }
        return np.array([0.1, 0.0], dtype=float), 0.1, 0.2

    monkeypatch.setattr(gn_runtime, "_solve_linear_system_fast", fake_fast_solver)

    step_calls = {"count": 0}

    def fake_step_size(*_args, **_kwargs):
        step_calls["count"] += 1
        if step_calls["count"] == 1:
            recon._force_jacobian_refresh = True
        return 1.0

    monkeypatch.setattr(gn_runtime, "_select_step_size", fake_step_size)

    rollback_calls = {"count": 0}

    def fake_rollback(*_args, **_kwargs):
        rollback_calls["count"] += 1
        if rollback_calls["count"] == 2:
            return True, False, 1
        if rollback_calls["count"] == 3:
            return True, True, 2
        return False, False, 0

    monkeypatch.setattr(gn_runtime, "_maybe_rollback", fake_rollback)

    results = gn_runtime.run_reconstruction(
        recon,
        measured_data=np.array([1.0, 0.8], dtype=float),
    )

    assert jacobian_calls["count"] >= 1
    assert solve_calls["count"] == 3
    assert recon._force_jacobian_refresh is False
    assert results.iterations >= 1


def test_run_reconstruction_linearized_jacobian_routes_operator_to_fast_solver(
    monkeypatch: pytest.MonkeyPatch,
):
    recon, _progress = _make_reconstructor(
        max_iterations=1, verbose=False, n_elements=2
    )
    _install_common_runtime_stubs(monkeypatch, recon)

    weights = torch.tensor([2.0, 0.5], dtype=torch.float64)
    recon._ensure_measurement_weights = lambda _sigma: setattr(
        recon, "_meas_weight_sqrt", weights
    )

    linearization = JacobianLinearization(
        grad_u_all=(np.ones((2, 1), dtype=float),),
        adjoint_gradients=(
            np.array([[1.0], [0.0]], dtype=float),
            np.array([[0.0], [1.0]], dtype=float),
        ),
        cell_areas=np.ones(2, dtype=float),
        n_meas_per_stim=(2,),
        sign=1.0,
    )
    calls = {"linearize": 0, "calculate": 0}

    def linearize(_sigma, method=None):
        assert method == "efficient"
        calls["linearize"] += 1
        return linearization

    def calculate(_sigma, method=None):
        _ = method
        calls["calculate"] += 1
        raise AssertionError("dense calculate should not run for linearized Jacobian")

    recon.jacobian_calculator = SimpleNamespace(
        calculate=calculate,
        linearize=linearize,
        block_tuning_info=lambda: {},
        _last_cache_lookup={},
    )
    monkeypatch.setattr(
        gn_runtime,
        "_startup_cache_lookup",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("startup dense cache should be skipped")
        ),
    )
    monkeypatch.setattr(
        gn_runtime,
        "_project_measurement_jacobian",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("operator Jacobian should not enter dense projection")
        ),
    )

    seen = {}

    def fake_fast_solver(reconstructor, **kwargs):
        seen.update(kwargs)
        reconstructor._last_fast_linear_meta = {
            "path": "pcg-diag-precond",
            "resolved_preconditioner": "diag",
            "fast_linear_path_selected": "pcg",
            "fast_linear_path_reason": "auto:matrix_free_pcg",
            "jacobian_representation": "jacobian_linearization",
            "jacobian_shape": [2, 2],
            "dense_jacobian_materialized": False,
            "linear_iterations": 1,
        }
        return np.array([0.1, 0.0], dtype=float), 0.1, 0.2

    monkeypatch.setattr(gn_runtime, "_solve_linear_system_fast", fake_fast_solver)
    monkeypatch.setattr(gn_runtime, "_select_step_size", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(
        gn_runtime, "_maybe_rollback", lambda *_args, **_kwargs: (False, False, 0)
    )

    results = gn_runtime.run_reconstruction(
        recon,
        measured_data=np.array([1.0, 0.8], dtype=float),
        jacobian_method="linearized",
    )

    assert calls == {"linearize": 1, "calculate": 0}
    assert seen["J_weighted_np"] is linearization
    np.testing.assert_allclose(seen["measurement_weight_np"], weights.numpy())
    backend = results.diagnostics["backend_info"]
    assert backend["jacobian_representation"] == "jacobian_linearization"
    assert backend["dense_jacobian_materialized"] is False


def test_operator_mode_skips_startup_dense_cache(monkeypatch: pytest.MonkeyPatch):
    """T9: operator Jacobian must never consult the dense startup cache lookup."""
    recon, _progress = _make_reconstructor(
        max_iterations=1, verbose=False, n_elements=2
    )
    _install_common_runtime_stubs(monkeypatch, recon)

    class _FakeArrayHolder:
        def __init__(self, values):
            self.array = np.asarray(values, dtype=float)

    class _FakeSigmaFunction:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=float)
            self.x = _FakeArrayHolder(self.values)

    monkeypatch.setattr(
        gn_runtime,
        "_init_sigma_function",
        lambda _recon, initial: (_FakeSigmaFunction([1.0, 1.0]), float(initial)),
    )

    linearization = JacobianLinearization(
        grad_u_all=(np.ones((2, 1), dtype=float),),
        adjoint_gradients=(
            np.array([[1.0], [0.0]], dtype=float),
            np.array([[0.0], [1.0]], dtype=float),
        ),
        cell_areas=np.ones(2, dtype=float),
        n_meas_per_stim=(2,),
        sign=1.0,
    )
    linearization.sigma_fingerprint = ""

    def linearize(_sigma, method=None):
        assert method == "efficient"
        return linearization

    recon.jacobian_calculator = SimpleNamespace(
        calculate=lambda *_a, **_kw: (_ for _ in ()).throw(
            AssertionError("dense calculate must not run")
        ),
        linearize=linearize,
        block_tuning_info=lambda: {},
        _last_cache_lookup={},
    )
    monkeypatch.setattr(
        gn_runtime,
        "_startup_cache_lookup",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            AssertionError("startup dense cache must not run under operator mode")
        ),
    )
    monkeypatch.setattr(
        gn_runtime,
        "_solve_linear_system_fast",
        lambda reconstructor, **_kw: (
            reconstructor.__dict__.__setitem__(
                "_last_fast_linear_meta",
                {
                    "jacobian_representation": "jacobian_linearization",
                    "dense_jacobian_materialized": False,
                },
            )
            or (np.array([0.01, 0.0], dtype=float), 0.01, 0.02)
        ),
    )
    monkeypatch.setattr(gn_runtime, "_select_step_size", lambda *_a, **_kw: 1.0)
    monkeypatch.setattr(
        gn_runtime, "_maybe_rollback", lambda *_a, **_kw: (False, False, 0)
    )

    results = gn_runtime.run_reconstruction(
        recon,
        measured_data=np.array([1.0, 0.8], dtype=float),
        jacobian_method="linearized",
    )

    backend = results.diagnostics["backend_info"]
    assert backend["jacobian_representation"] == "jacobian_linearization"
    assert backend["dense_jacobian_materialized"] is False


def test_linearize_path_asserts_sigma_fingerprint(monkeypatch: pytest.MonkeyPatch):
    """Operator path calls JacobianLinearization.assert_compatible(current sigma)."""
    recon, _progress = _make_reconstructor(
        max_iterations=1, verbose=False, n_elements=2
    )
    _install_common_runtime_stubs(monkeypatch, recon)

    # Override _init_sigma_function to return a DOLFINx-Function-shaped stub so
    # compute_sigma_fingerprint() can read sigma.x.array and emit a real hash.
    class _FakeArrayHolder:
        def __init__(self, values):
            self.array = np.asarray(values, dtype=float)

    class _FakeSigmaFunction:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=float)
            self.x = _FakeArrayHolder(self.values)

    monkeypatch.setattr(
        gn_runtime,
        "_init_sigma_function",
        lambda _recon, initial: (_FakeSigmaFunction([1.0, 1.0]), float(initial)),
    )

    stale_linearization = JacobianLinearization(
        grad_u_all=(np.ones((2, 1), dtype=float),),
        adjoint_gradients=(
            np.array([[1.0], [0.0]], dtype=float),
            np.array([[0.0], [1.0]], dtype=float),
        ),
        cell_areas=np.ones(2, dtype=float),
        n_meas_per_stim=(2,),
        sign=1.0,
        sigma_fingerprint="stale-fingerprint-not-matching-current",
    )

    def linearize(_sigma, method=None):
        assert method == "efficient"
        return stale_linearization

    recon.jacobian_calculator = SimpleNamespace(
        calculate=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dense calculate must not run")
        ),
        linearize=linearize,
        block_tuning_info=lambda: {},
        _last_cache_lookup={},
    )
    monkeypatch.setattr(
        gn_runtime,
        "_startup_cache_lookup",
        lambda *_args, **_kwargs: (None, {"hit": False}),
    )

    with pytest.raises(ValueError, match="sigma fingerprint mismatch"):
        gn_runtime.run_reconstruction(
            recon,
            measured_data=np.array([1.0, 0.8], dtype=float),
            jacobian_method="linearized",
        )


def test_run_reconstruction_reuses_linearized_jacobian_across_iterations(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Operator path honours jacobian_update_every without re-running linearize()."""
    recon, _progress = _make_reconstructor(max_iterations=4, verbose=True, n_elements=2)
    _install_common_runtime_stubs(monkeypatch, recon)
    recon.jacobian_update_every = 2
    recon.jacobian_reuse_tol = 10.0  # force reuse based on update cadence only

    linearization = JacobianLinearization(
        grad_u_all=(np.ones((2, 1), dtype=float),),
        adjoint_gradients=(
            np.array([[1.0], [0.0]], dtype=float),
            np.array([[0.0], [1.0]], dtype=float),
        ),
        cell_areas=np.ones(2, dtype=float),
        n_meas_per_stim=(2,),
        sign=1.0,
    )
    calls = {"linearize": 0, "calculate": 0}

    def linearize(_sigma, method=None):
        assert method == "efficient"
        calls["linearize"] += 1
        return linearization

    def calculate(_sigma, method=None):
        _ = method
        calls["calculate"] += 1
        raise AssertionError(
            "dense calculate must not run for linearized Jacobian reuse"
        )

    recon.jacobian_calculator = SimpleNamespace(
        calculate=calculate,
        linearize=linearize,
        block_tuning_info=lambda: {},
        _last_cache_lookup={},
    )
    monkeypatch.setattr(
        gn_runtime,
        "_startup_cache_lookup",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("startup dense cache should be skipped")
        ),
    )
    monkeypatch.setattr(
        gn_runtime,
        "_project_measurement_jacobian",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("operator Jacobian must not enter dense projection")
        ),
    )

    observed_reuse = []

    def fake_fast_solver(reconstructor, **kwargs):
        observed_reuse.append(kwargs["J_weighted_np"] is linearization)
        reconstructor._last_fast_linear_meta = {
            "path": "pcg-diag-precond",
            "resolved_preconditioner": "diag",
            "fast_linear_path_selected": "pcg",
            "fast_linear_path_reason": "auto:matrix_free_pcg",
            "jacobian_representation": "jacobian_linearization",
            "jacobian_shape": [2, 2],
            "dense_jacobian_materialized": False,
            "linear_iterations": 1,
        }
        return np.array([0.01, 0.0], dtype=float), 0.01, 0.02

    monkeypatch.setattr(gn_runtime, "_solve_linear_system_fast", fake_fast_solver)
    monkeypatch.setattr(gn_runtime, "_select_step_size", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(
        gn_runtime, "_maybe_rollback", lambda *_args, **_kwargs: (False, False, 0)
    )

    results = gn_runtime.run_reconstruction(
        recon,
        measured_data=np.array([1.0, 0.8], dtype=float),
        jacobian_method="linearized",
    )

    # jacobian_update_every=2 means linearize() runs at iters 0 and 2 only.
    assert calls["linearize"] == 2
    assert calls["calculate"] == 0
    # All four fast-solver calls see the SAME operator object (no rebuild).
    assert observed_reuse == [True, True, True, True]
    assert results.iterations == 4

    # Verbose mode prints the reuse banner on the reused iterations.
    output = capsys.readouterr().out
    assert output.count("reused Jacobian") >= 2
