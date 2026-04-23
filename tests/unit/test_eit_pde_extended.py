"""Extended tests for EITPDE adapter helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from dolfinx import fem

from pyeidors.inverse.solvers import eit_pde as eit_pde_module


class _FakeJacobianCalculator:
    def __init__(self, matrix: np.ndarray):
        self.matrix = np.asarray(matrix, dtype=float)
        self.calls = 0

    def calculate(self, sigma_fn):
        _ = sigma_fn
        self.calls += 1
        return self.matrix.copy()


def _build_pde_stub(eit_system):
    V_sigma = eit_system.fwd_model.V_sigma
    n_elem = int(fem.Function(V_sigma).x.array.size)
    n_meas = int(eit_system.fwd_model.pattern_manager.n_meas_total)

    pde = eit_pde_module.EITPDE.__new__(eit_pde_module.EITPDE)
    pde._eit_system = eit_system
    pde._fwd_model = SimpleNamespace(
        V_sigma=V_sigma,
        pattern_manager=SimpleNamespace(n_meas_total=n_meas),
    )
    pde._V_sigma = V_sigma
    pde._sigma_function = fem.Function(V_sigma)
    pde._current_image = None
    pde._cached_jacobian = None
    pde._cached_sigma_vector = None
    pde._jacobian_calculator = _FakeJacobianCalculator(np.eye(n_meas, n_elem))

    def _fake_fwd_solve(image):
        sigma = np.asarray(image.elem_data, dtype=float).ravel()
        meas = np.dot(np.eye(n_meas, n_elem), sigma)
        return SimpleNamespace(meas=meas), {"U": np.zeros((1, 1))}

    pde._fwd_model.fwd_solve = _fake_fwd_solve
    return pde, n_elem, n_meas


def test_constructor_initializes_runtime_state(eit_system):
    pde = eit_pde_module.EITPDE(eit_system)

    assert pde._eit_system is eit_system
    assert pde._fwd_model is eit_system.fwd_model
    assert pde._current_image is None
    assert pde._cached_jacobian is None
    assert pde._cached_sigma_vector is None
    assert isinstance(pde._jacobian_calculator, eit_pde_module.DirectJacobianCalculator)


def test_assemble_and_solve_observe(eit_system):
    pde, n_elem, n_meas = _build_pde_stub(eit_system)

    with pytest.raises(ValueError):
        pde.assemble(np.zeros(n_elem + 1))

    parameter = np.linspace(1.0, 2.0, n_elem)
    pde.assemble(parameter)
    assert np.allclose(pde._sigma_function.x.array, parameter)
    assert pde._current_image is not None

    data, meta = pde.solve()
    observed = pde.observe((data, meta))
    assert observed.shape[0] == n_meas
    assert np.isfinite(observed).all()

    observed_direct = pde.observe(np.arange(n_meas, dtype=float))
    assert observed_direct.shape[0] == n_meas


def test_solve_requires_assemble_first(eit_system):
    pde, _, _ = _build_pde_stub(eit_system)
    with pytest.raises(RuntimeError):
        pde.solve()


def test_gradient_and_jacobian_cache(eit_system):
    pde, n_elem, n_meas = _build_pde_stub(eit_system)
    wrt = np.linspace(0.5, 1.5, n_elem)
    direction = np.linspace(-0.2, 0.2, n_meas)

    grad = pde.gradient_wrt_parameter(direction, wrt)
    assert grad.shape[0] == n_elem

    jac1 = pde.jacobian_wrt_parameter(wrt)
    jac2 = pde.jacobian_wrt_parameter(wrt.copy())
    assert np.allclose(jac1, jac2)
    assert (
        pde._jacobian_calculator.calls == 2
    )  # one for gradient + one for cached jacobian

    with pytest.raises(ValueError):
        pde.gradient_wrt_parameter(direction, np.zeros(n_elem + 1))

    with pytest.raises(ValueError):
        pde.jacobian_wrt_parameter(np.zeros(n_elem + 2))


def test_geometry_info_and_forward(eit_system):
    pde, n_elem, n_meas = _build_pde_stub(eit_system)
    info = pde.geometry_info
    assert info.n_elements == n_elem
    assert info.n_measurements == n_meas

    out = pde.forward(np.ones(n_elem, dtype=float))
    assert out.shape[0] == n_meas


def test_create_pde_model_branches(monkeypatch, eit_system):
    monkeypatch.setattr(eit_pde_module, "PDEModel", None)
    with pytest.raises(ImportError):
        eit_pde_module.create_pde_model(eit_system)

    class _FakeEITPDE:
        def __init__(self, system):
            _ = system
            self.geometry_info = eit_pde_module.EITGeometryInfo(
                n_elements=11, n_measurements=22
            )

    class _FakePDEModel:
        def __init__(self, PDE, range_geometry, domain_geometry):
            self.PDE = PDE
            self.range_geometry = range_geometry
            self.domain_geometry = domain_geometry

    monkeypatch.setattr(eit_pde_module, "EITPDE", _FakeEITPDE)
    monkeypatch.setattr(eit_pde_module, "PDEModel", _FakePDEModel)

    pde, model, geom = eit_pde_module.create_pde_model(eit_system)
    assert geom.n_elements == 11
    assert model.range_geometry == 22
    assert model.domain_geometry == 11
    assert isinstance(pde, _FakeEITPDE)
