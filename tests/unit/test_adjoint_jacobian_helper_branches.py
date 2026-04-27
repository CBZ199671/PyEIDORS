"""Additional branch coverage for adjoint Jacobian helper logic."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import pyeidors.inverse.jacobian._core as core_module
import pyeidors.inverse.jacobian.adjoint_jacobian as adjoint_module
import pyeidors.inverse.jacobian.base_jacobian as base_module
from pyeidors.inverse.jacobian.adjoint_jacobian import EidorsStyleAdjointJacobian


class _FakeCellVec:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=float)

    def assemble(self):
        return None


class _InterpFunction:
    def __init__(self, space):
        self.space = space
        size = 2 if getattr(space, "name", "") == "V" else 4
        self.x = SimpleNamespace(array=np.zeros(size, dtype=float))

    def interpolate(self, expr):
        base = float(expr.grad_value)
        self.x.array[:] = np.array(
            [base, base + 1.0, base + 2.0, base + 3.0], dtype=float
        )


def test_resolve_torch_dtype_and_init_cover_aliases_and_auto_device_paths(
    monkeypatch: pytest.MonkeyPatch,
):
    assert EidorsStyleAdjointJacobian._resolve_torch_dtype(None) == torch.float64
    assert (
        EidorsStyleAdjointJacobian._resolve_torch_dtype(torch.float32) == torch.float32
    )
    assert EidorsStyleAdjointJacobian._resolve_torch_dtype("fp32") == torch.float32
    assert EidorsStyleAdjointJacobian._resolve_torch_dtype("double") == torch.float64
    with pytest.raises(ValueError, match="Unsupported torch dtype"):
        EidorsStyleAdjointJacobian._resolve_torch_dtype("int8")

    monkeypatch.setattr(
        EidorsStyleAdjointJacobian,
        "_setup",
        lambda self: setattr(self, "_setup_called", True),
    )
    monkeypatch.setattr(
        base_module.fem,
        "Function",
        lambda _space: SimpleNamespace(
            x=SimpleNamespace(array=np.zeros(1, dtype=float))
        ),
    )
    fake_fwd_model = SimpleNamespace(
        V_sigma=SimpleNamespace(), pattern_manager=SimpleNamespace(n_meas_total=1)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if getattr(torch.backends, "mps", None) is not None:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        jac_mps = EidorsStyleAdjointJacobian(
            fake_fwd_model, use_torch=False, device=None
        )
        assert jac_mps.torch_device.type == "mps"

    if getattr(torch.backends, "mps", None) is not None:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    jac_cpu = EidorsStyleAdjointJacobian(fake_fwd_model, use_torch=False, device=None)
    assert jac_cpu.torch_device.type == "cpu"

    jac_explicit = EidorsStyleAdjointJacobian(
        fake_fwd_model, use_torch=False, device="cpu"
    )
    assert jac_explicit.torch_device.type == "cpu"


def test_setup_and_compute_field_gradients_cover_torch_and_callable_interpolation_points(
    monkeypatch: pytest.MonkeyPatch,
):
    jac = EidorsStyleAdjointJacobian.__new__(EidorsStyleAdjointJacobian)
    jac.fwd_model = SimpleNamespace(
        mesh=SimpleNamespace(geometry=SimpleNamespace(dim=2)),
        V=SimpleNamespace(name="V"),
        V_sigma=SimpleNamespace(name="V_sigma"),
    )
    jac.use_torch = True
    jac.torch_device = torch.device("cpu")
    jac.torch_dtype = torch.float32

    monkeypatch.setattr(
        core_module.fem,
        "functionspace",
        lambda _mesh, _desc: SimpleNamespace(
            element=SimpleNamespace(interpolation_points=lambda: np.array([[0.0, 0.0]]))
        ),
    )
    monkeypatch.setattr(core_module.ufl, "TestFunction", lambda _space: 1.0)
    monkeypatch.setattr(core_module.ufl, "dx", 1.0)
    monkeypatch.setattr(core_module.fem, "form", lambda expr: expr)
    monkeypatch.setattr(
        core_module.fem_petsc,
        "assemble_vector",
        lambda _form: _FakeCellVec([2.0, 3.0]),
    )
    EidorsStyleAdjointJacobian._setup(jac)
    np.testing.assert_allclose(jac.cell_areas, np.array([2.0, 3.0], dtype=float))
    np.testing.assert_allclose(
        jac.cell_areas_t.cpu().numpy(), np.array([2.0, 3.0], dtype=np.float32)
    )

    jac.V = SimpleNamespace(name="V")
    jac.Q_DG = SimpleNamespace(
        element=SimpleNamespace(interpolation_points=lambda: np.array([[0.0, 0.0]]))
    )
    jac.gdim = 2
    jac._geometry = SimpleNamespace(V=jac.V, Q_DG=jac.Q_DG, gdim=jac.gdim)
    monkeypatch.setattr(core_module.fem, "Function", _InterpFunction)
    monkeypatch.setattr(
        core_module.fem,
        "Expression",
        lambda grad_value, points: SimpleNamespace(
            grad_value=grad_value, points=points
        ),
    )
    monkeypatch.setattr(
        core_module.ufl, "grad", lambda u_fun: float(np.sum(u_fun.x.array))
    )

    grads = EidorsStyleAdjointJacobian._compute_field_gradients(
        jac,
        [np.array([1.0, 2.0], dtype=float), np.array([3.0, 4.0], dtype=float)],
    )
    np.testing.assert_allclose(
        grads[0], np.array([[3.0, 4.0], [5.0, 6.0]], dtype=float)
    )
    np.testing.assert_allclose(
        grads[1], np.array([[7.0, 8.0], [9.0, 10.0]], dtype=float)
    )


def test_measurement_patterns_and_numpy_torch_assembly_cover_remaining_paths(
    monkeypatch: pytest.MonkeyPatch,
):
    jac = EidorsStyleAdjointJacobian.__new__(EidorsStyleAdjointJacobian)
    jac.fwd_model = SimpleNamespace(
        n_elec=2,
        pattern_manager=SimpleNamespace(
            n_meas_total=3,
            n_stim=2,
            n_meas_per_stim=[2, 1],
            meas_matrices=[
                np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
                np.array([[1.0, -1.0]], dtype=float),
            ],
        ),
    )
    jac.cell_areas = np.array([2.0, 3.0], dtype=float)
    jac.cell_areas_t = torch.tensor([2.0, 3.0], dtype=torch.float32)
    jac.torch_device = torch.device("cpu")
    jac.torch_dtype = torch.float32
    jac.torch_batch_all = False

    patterns = EidorsStyleAdjointJacobian._measurement_to_current_patterns(jac)
    np.testing.assert_allclose(
        patterns,
        np.array([[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]], dtype=float),
    )

    grad_u_all = [
        np.array([[1.0, 2.0], [0.5, 1.0]], dtype=float),
        np.array([[2.0, 1.0], [1.0, 0.5]], dtype=float),
    ]
    grad_adj_all = [
        np.array([[0.5, 1.0], [1.0, 1.5]], dtype=float),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
        np.array([[1.5, 0.5], [0.5, 1.5]], dtype=float),
    ]
    assembled_np = EidorsStyleAdjointJacobian._assemble_numpy(
        jac, grad_u_all, grad_adj_all
    )
    expected = np.array(
        [
            [-5.0, -6.0],
            [-2.0, -3.0],
            [-7.0, -3.75],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(assembled_np, expected)

    assembled_torch = EidorsStyleAdjointJacobian._assemble_torch(
        jac, grad_u_all, grad_adj_all
    )
    np.testing.assert_allclose(assembled_torch, expected, atol=1e-6, rtol=1e-6)

    jac.torch_batch_all = True
    monkeypatch.setattr(
        jac, "_assemble_torch_all", lambda gu, ga: np.array([[42.0]], dtype=float)
    )
    np.testing.assert_allclose(
        EidorsStyleAdjointJacobian._assemble_torch(jac, grad_u_all, grad_adj_all),
        np.array([[42.0]]),
    )

    jac.torch_batch_all = False
    assembled_all = EidorsStyleAdjointJacobian._assemble_torch_all(
        jac, grad_u_all, grad_adj_all
    )
    np.testing.assert_allclose(assembled_all, expected, atol=1e-6, rtol=1e-6)


def test_linearize_returns_matrix_free_operator_matching_dense_assembly():
    grad_u_all = [
        np.array([[1.0, 2.0], [0.5, 1.0]], dtype=float),
        np.array([[2.0, 1.0], [1.0, 0.5]], dtype=float),
    ]
    grad_adj_all = [
        np.array([[0.5, 1.0], [1.0, 1.5]], dtype=float),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
        np.array([[1.5, 0.5], [0.5, 1.5]], dtype=float),
    ]
    calls = {"count": 0}

    def fake_forward_solve(_sigma, current_patterns=None):
        calls["count"] += 1
        if current_patterns is None:
            return [np.array([1.0], dtype=float), np.array([2.0], dtype=float)], None
        return [
            np.array([3.0], dtype=float),
            np.array([4.0], dtype=float),
            np.array([5.0], dtype=float),
        ], None

    jac = EidorsStyleAdjointJacobian.__new__(EidorsStyleAdjointJacobian)
    jac.fwd_model = SimpleNamespace(
        forward_solve=fake_forward_solve,
        pattern_manager=SimpleNamespace(
            n_meas_total=3,
            n_stim=2,
            n_meas_per_stim=[2, 1],
        ),
    )
    jac.cell_areas = np.array([2.0, 3.0], dtype=float)
    jac._measurement_to_current_patterns = lambda: np.eye(3, dtype=float)
    jac._compute_field_gradients = lambda _fields: (
        grad_u_all if calls["count"] == 1 else grad_adj_all
    )
    sigma = SimpleNamespace(x=SimpleNamespace(array=np.array([1.0, 1.0], dtype=float)))

    linearization = EidorsStyleAdjointJacobian.linearize(jac, sigma)

    expected = EidorsStyleAdjointJacobian._assemble_numpy(jac, grad_u_all, grad_adj_all)
    np.testing.assert_allclose(linearization.to_dense(), expected)
    np.testing.assert_allclose(
        linearization.matvec(np.array([0.25, 0.5])), expected @ np.array([0.25, 0.5])
    )
    np.testing.assert_allclose(
        linearization.rmatvec(np.array([1.0, -2.0, 0.5])),
        expected.T @ np.array([1.0, -2.0, 0.5]),
    )
    assert linearization.sigma_fingerprint


def test_calculate_dispatches_torch_path(monkeypatch: pytest.MonkeyPatch):
    grad_u_all = [np.array([[1.0, 0.0]], dtype=float)]
    grad_adj_all = [np.array([[0.5, 1.0]], dtype=float)]
    calls = {"count": 0}

    def fake_forward_solve(_sigma, current_patterns=None):
        calls["count"] += 1
        if current_patterns is None:
            return [np.array([1.0, 2.0], dtype=float)], None
        return [np.array([3.0, 4.0], dtype=float)], None

    jac = EidorsStyleAdjointJacobian.__new__(EidorsStyleAdjointJacobian)
    jac.fwd_model = SimpleNamespace(forward_solve=fake_forward_solve)
    jac.use_torch = True
    jac._compute_field_gradients = lambda _fields: (
        grad_u_all if calls["count"] == 1 else grad_adj_all
    )
    jac._measurement_to_current_patterns = lambda: np.array(
        [[1.0], [-1.0]], dtype=float
    )
    jac._assemble_torch = lambda gu, ga: np.array([[7.0]], dtype=float)
    out = EidorsStyleAdjointJacobian.calculate(jac, sigma=SimpleNamespace())
    np.testing.assert_allclose(out, np.array([[7.0]], dtype=float))
