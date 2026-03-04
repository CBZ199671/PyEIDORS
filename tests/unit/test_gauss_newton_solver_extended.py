"""Extended unit tests for GaussNewtonReconstructor branches."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from dolfinx import fem

from pyeidors.inverse.solvers.gauss_newton import GaussNewtonReconstructor


class _DummyJacobian:
    def __init__(self, jacobian: np.ndarray):
        self._jacobian = np.asarray(jacobian, dtype=float)

    def calculate(self, sigma_function, method: str = "efficient") -> np.ndarray:
        _ = sigma_function
        _ = method
        return self._jacobian.copy()


class _DummyRegularization:
    def __init__(self, matrix: np.ndarray):
        self._matrix = np.asarray(matrix, dtype=float)

    def get_regularization_matrix(self) -> np.ndarray:
        return self._matrix.copy()


class _LinearForwardModel:
    def __init__(self, V_sigma, jacobian: np.ndarray, bias: np.ndarray | None = None):
        self.V_sigma = V_sigma
        self.pattern_manager = SimpleNamespace(n_meas_total=int(jacobian.shape[0]))
        self._jacobian = np.asarray(jacobian, dtype=float)
        self._bias = (
            np.asarray(bias, dtype=float).ravel()
            if bias is not None
            else np.zeros(self._jacobian.shape[0], dtype=float)
        )
        self._seq: list[np.ndarray] | None = None
        self._call_count = 0
        self._raise_after: int | None = None

    def set_sequence(self, sequence: list[np.ndarray] | None) -> None:
        self._seq = [np.asarray(v, dtype=float).ravel() for v in sequence] if sequence else None
        self._call_count = 0

    def set_raise_after(self, call_index: int | None) -> None:
        self._raise_after = call_index
        self._call_count = 0

    def fwd_solve(self, image):
        self._call_count += 1
        if self._raise_after is not None and self._call_count >= self._raise_after:
            raise RuntimeError("forced forward failure")

        if self._seq is not None and self._call_count <= len(self._seq):
            meas = self._seq[self._call_count - 1]
        else:
            sigma = np.asarray(image.elem_data, dtype=float).ravel()
            meas = np.dot(self._jacobian, sigma) + self._bias
        return SimpleNamespace(meas=np.asarray(meas, dtype=float)), None


def _make_solver(eit_system, n_meas: int = 8):
    V_sigma = eit_system.fwd_model.V_sigma
    n_elem = int(fem.Function(V_sigma).x.array.size)
    rng = np.random.default_rng(1234)
    jacobian = rng.normal(scale=0.2, size=(n_meas, n_elem))
    jacobian += np.eye(n_meas, n_elem) * 0.8

    fwd_model = _LinearForwardModel(V_sigma=V_sigma, jacobian=jacobian, bias=np.linspace(0.01, 0.02, n_meas))
    reconstructor = GaussNewtonReconstructor(
        fwd_model=fwd_model,
        jacobian_calculator=_DummyJacobian(jacobian),
        regularization=_DummyRegularization(np.eye(n_elem)),
        max_iterations=4,
        min_iterations=1,
        convergence_tol=1e-8,
        regularization_param=0.05,
        line_search_steps=6,
        device="cpu",
        verbose=False,
        clip_values=(1e-5, 5.0),
        min_step=0.0,
    )
    return reconstructor, fwd_model, jacobian


def _safe_torch_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    A_np = A.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    try:
        x_np = np.linalg.solve(A_np, b_np)
    except np.linalg.LinAlgError:
        x_np = np.linalg.lstsq(A_np, b_np, rcond=None)[0]
    return torch.from_numpy(np.asarray(x_np)).to(device=A.device, dtype=A.dtype)


def test_reconstruct_and_history_recording(eit_system, monkeypatch):
    monkeypatch.setattr(torch.linalg, "solve", _safe_torch_solve)
    reconstructor, fwd_model, jacobian = _make_solver(eit_system, n_meas=10)
    n_elem = jacobian.shape[1]
    sigma_true = np.linspace(0.9, 1.1, n_elem)
    measured = np.dot(jacobian, sigma_true) + 0.01

    result = reconstructor.reconstruct(
        measured_data=SimpleNamespace(meas=measured),
        initial_conductivity=np.full(n_elem, 1.0),
        jacobian_method="efficient",
        record_conductivity_history=True,
        conductivity_history_stride=1,
    )

    assert result.conductivity is not None
    assert result.iterations >= 1
    assert result.residual_history is not None
    assert len(result.residual_history) >= 1
    assert result.conductivity_history is not None
    assert result.conductivity_history[0].shape[0] == n_elem
    assert np.isfinite(result.final_residual)
    assert fwd_model._call_count > 1


def test_measurement_length_mismatch_raises(eit_system):
    reconstructor, _, _ = _make_solver(eit_system, n_meas=6)
    wrong = np.zeros(reconstructor.n_measurements + 1, dtype=float)
    try:
        reconstructor.reconstruct(wrong)
    except ValueError as exc:
        assert "Measurement data length mismatch" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError for measurement length mismatch")


def test_measurement_weight_strategies_and_baseline_storage(eit_system, monkeypatch):
    monkeypatch.setattr(torch.linalg, "solve", _safe_torch_solve)
    for strategy in ("baseline", "scaled_baseline", "difference"):
        reconstructor, _, jacobian = _make_solver(eit_system, n_meas=7)
        reconstructor.measurement_weight_strategy = strategy
        reconstructor.use_measurement_weights = True
        measured = np.dot(jacobian, np.ones(jacobian.shape[1], dtype=float))
        out = reconstructor.reconstruct(
            measured_data=SimpleNamespace(meas=measured),
            initial_conductivity=1.0,
            jacobian_method="efficient",
        )
        assert out.measurement_weight is not None
        assert out.baseline_measurement is not None
        assert out.measurement_weight.shape[0] == reconstructor.n_measurements


def test_scale_and_difference_helpers(eit_system):
    reconstructor, _, _ = _make_solver(eit_system, n_meas=5)
    baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    reconstructor._measured_vector = baseline * 2.0 + 0.3
    scaled = reconstructor._scale_baseline_to_measured(baseline)
    assert scaled.shape == baseline.shape
    assert np.isfinite(scaled).all()

    reconstructor._measured_vector = baseline.copy()
    diff = reconstructor._difference_with_baseline(baseline)
    assert np.all(diff >= reconstructor.weight_floor)

    reconstructor._measured_vector = None
    assert np.allclose(reconstructor._scale_baseline_to_measured(baseline), baseline)
    assert np.allclose(reconstructor._difference_with_baseline(baseline), baseline)


def test_line_search_and_perturb_updates(eit_system):
    reconstructor, _, jacobian = _make_solver(eit_system, n_meas=8)
    n_elem = jacobian.shape[1]
    sigma_current = fem.Function(reconstructor.fwd_model.V_sigma)
    sigma_current.x.array[:] = 1.0
    reconstructor.R_torch = torch.eye(n_elem, dtype=torch.float64)

    target = torch.from_numpy(np.dot(jacobian, np.ones(n_elem, dtype=float))).to(dtype=torch.float64)
    delta = torch.from_numpy(np.full(n_elem, -0.05, dtype=float)).to(dtype=torch.float64)
    prior = torch.from_numpy(np.ones(n_elem, dtype=float)).to(dtype=torch.float64)

    step = reconstructor._line_search_torch(
        sigma_current=sigma_current,
        delta_sigma_torch=delta,
        meas_target_torch=target,
        current_weighted_residual=1.0,
        weight_vector=None,
        prior_torch=prior,
        lambda_eff=0.05,
    )
    assert 0.0 <= step <= 1.0

    perturb = reconstructor._calc_perturb_limits(
        x=np.array([1.0, 2.0, 3.0]),
        dx=np.array([0.1, -0.2, 0.3]),
    )
    assert perturb[0] == 0.0
    assert np.isfinite(perturb).all()

    reconstructor._line_search_perturb = np.array([0.0, 0.1, 0.5, 1.0])
    mlist = np.array([1.0, 0.95, 0.9, 0.85], dtype=float)
    reconstructor._update_perturb_eidors_style(
        chosen_step=0.5,
        perturb=np.array([0.0, 0.1, 0.5, 1.0]),
        mlist=mlist,
        valid_idx=np.array([0, 1, 2, 3], dtype=int),
    )
    assert reconstructor._line_search_perturb[-1] <= 1.0 + 1e-9


def test_line_search_handles_forward_failures(eit_system):
    reconstructor, fwd_model, jacobian = _make_solver(eit_system, n_meas=6)
    n_elem = jacobian.shape[1]
    sigma_current = fem.Function(reconstructor.fwd_model.V_sigma)
    sigma_current.x.array[:] = 1.0
    reconstructor.R_torch = torch.eye(n_elem, dtype=torch.float64)
    fwd_model.set_raise_after(2)

    step = reconstructor._line_search_torch(
        sigma_current=sigma_current,
        delta_sigma_torch=torch.ones(n_elem, dtype=torch.float64) * 0.2,
        meas_target_torch=torch.zeros(reconstructor.n_measurements, dtype=torch.float64),
        current_weighted_residual=1.0,
        weight_vector=None,
        prior_torch=torch.zeros(n_elem, dtype=torch.float64),
        lambda_eff=0.1,
    )
    assert 0.0 <= step <= 1.0


def test_reconstruct_rollback_early_stop(eit_system, monkeypatch):
    monkeypatch.setattr(torch.linalg, "solve", _safe_torch_solve)
    reconstructor, fwd_model, jacobian = _make_solver(eit_system, n_meas=8)
    n_elem = jacobian.shape[1]
    measured = np.dot(jacobian, np.ones(n_elem, dtype=float))

    good = measured.copy()
    bad = measured + 10.0
    fwd_model.set_sequence([good] + [bad] * 10)
    reconstructor.max_iterations = 12
    reconstructor.step_schedule = [0.3] * reconstructor.max_iterations

    result = reconstructor.reconstruct(
        measured_data=SimpleNamespace(meas=measured),
        initial_conductivity=1.0,
    )
    assert result.iterations < reconstructor.max_iterations
    assert np.isfinite(result.final_residual)


def test_setters_invalidate_or_replace_components(eit_system):
    reconstructor, _, jacobian = _make_solver(eit_system, n_meas=5)
    n_elem = jacobian.shape[1]

    reconstructor.R_torch = torch.eye(n_elem, dtype=torch.float64)
    reconstructor.set_regularization(_DummyRegularization(np.eye(n_elem) * 2.0))
    assert reconstructor.R_torch is None

    new_jac = _DummyJacobian(np.eye(reconstructor.n_measurements, n_elem))
    reconstructor.set_jacobian_calculator(new_jac)
    assert reconstructor.jacobian_calculator is new_jac
