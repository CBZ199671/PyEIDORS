"""Extended coverage tests for sparse Bayesian solver internals."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pyeidors.data.structures import EITData, EITImage
from pyeidors.inverse.solvers import sparse_bayesian as sparse_module


class _FakeMapResult:
    def __init__(self, values: np.ndarray):
        self._values = np.asarray(values, dtype=float)

    def to_numpy(self):
        return self._values.copy()


class _FakeLinearModel:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=float)

    def __matmul__(self, x):
        return ("linear", self.matrix, x)


class _FakeSmoothedLaplace:
    def __init__(self, location, scale, beta):
        self.location = np.asarray(location, dtype=float)
        self.scale = float(scale)
        self.beta = float(beta)


class _FakeGaussian:
    def __init__(self, model_expr, sigma):
        self.model_expr = model_expr
        self.sigma = float(sigma)


class _FakeBayesianProblem:
    def __init__(self, y, x):
        self._x = x
        self._data = None

    def set_data(self, **kwargs):
        self._data = kwargs
        return self

    def MAP(self, disp=False, x0=None):
        _ = disp
        if x0 is None:
            return _FakeMapResult(np.zeros_like(self._x.location))
        return _FakeMapResult(np.asarray(x0, dtype=float))


class _FakePDE:
    def __init__(self, jacobian: np.ndarray):
        self._jacobian = np.asarray(jacobian, dtype=float)
        self.calls = 0

    def jacobian_wrt_parameter(self, wrt: np.ndarray) -> np.ndarray:
        _ = wrt
        self.calls += 1
        return self._jacobian.copy()


def _new_reconstructor(eit_system, config: sparse_module.SparseBayesianConfig | None = None):
    rec = sparse_module.SparseBayesianReconstructor.__new__(sparse_module.SparseBayesianReconstructor)
    rec.eit_system = eit_system
    rec.fwd_model = eit_system.fwd_model
    rec.verbose = False
    rec.config = config or sparse_module.SparseBayesianConfig()
    rec.n_elements = int(eit_system.fwd_model.V_sigma.dofmap.index_map.size_local)
    rec.n_measurements = int(eit_system.fwd_model.pattern_manager.n_meas_total)
    rec._cached_jacobian = None
    rec._cached_baseline = None
    rec._cached_basis = None
    rec._cached_reduced_matrix = None
    rec._cached_U = None
    rec._cached_singular = None
    rec._coarse_levels_cache = {}
    rec._cached_coarse_matrices = {}
    return rec


def test_constructor_import_guard(eit_system, monkeypatch):
    monkeypatch.setattr(sparse_module, "_CUQI_AVAILABLE", False)
    with pytest.raises(ImportError):
        sparse_module.SparseBayesianReconstructor(eit_system)


def test_reconstruct_absolute_and_difference_modes(eit_system):
    rec = _new_reconstructor(eit_system)
    n_elem = rec.n_elements
    n_meas = rec.n_measurements
    baseline = np.ones(n_elem, dtype=float)
    jacobian = np.eye(n_meas, n_elem)
    observed = np.linspace(-0.1, 0.2, n_meas)

    rec._prepare_jacobian = lambda baseline_values: jacobian.copy()
    rec._forward_measurement = lambda values: jacobian @ np.asarray(values, dtype=float)
    rec._solve_sparse_map = lambda J, d, n, p: np.full(n_elem, 0.05, dtype=float)
    rec._estimate_noise_level = lambda dv: 1e-3

    measurement = EITData(
        meas=observed.copy(),
        stim_pattern=np.zeros((1, 1)),
        n_elec=16,
        n_stim=1,
        n_meas=n_meas,
        type="real",
    )
    reference = EITData(
        meas=np.zeros_like(observed),
        stim_pattern=np.zeros((1, 1)),
        n_elec=16,
        n_stim=1,
        n_meas=n_meas,
        type="real",
    )
    baseline_image = EITImage(elem_data=baseline.copy(), fwd_model=eit_system.fwd_model)

    absolute = rec.reconstruct(measurement_data=measurement, baseline_image=baseline_image)
    difference = rec.reconstruct(
        measurement_data=measurement,
        baseline_image=baseline_image,
        reference_data=reference,
        metadata={"case": "difference"},
    )

    assert absolute.metadata["mode"] == "absolute"
    assert difference.metadata["mode"] == "difference"
    assert absolute.conductivity.x.array.size == n_elem
    assert difference.metadata["case"] == "difference"
    assert np.isfinite(absolute.final_residual)


def test_forward_measurement_and_jacobian_cache(eit_system):
    rec = _new_reconstructor(eit_system)
    n_elem = rec.n_elements
    jacobian = np.eye(rec.n_measurements, n_elem)
    rec._eit_pde = _FakePDE(jacobian)

    class _ToNumpyResult:
        def __init__(self, arr):
            self._arr = arr

        def to_numpy(self):
            return self._arr

    rec._cuqi_model = lambda conductivity: _ToNumpyResult(np.asarray(conductivity, dtype=float)[: rec.n_measurements])

    measurement = rec._forward_measurement(np.arange(n_elem, dtype=float))
    assert measurement.shape[0] == rec.n_measurements

    baseline = np.ones(n_elem, dtype=float)
    j1 = rec._prepare_jacobian(baseline)
    j2 = rec._prepare_jacobian(baseline)
    assert np.allclose(j1, j2)
    assert rec._eit_pde.calls == 1

    rec.config.cache_jacobian = False
    _ = rec._prepare_jacobian(baseline + 0.1)
    assert rec._eit_pde.calls == 2


def test_noise_hierarchy_and_projection_helpers(eit_system):
    config = sparse_module.SparseBayesianConfig(
        coarse_levels=(4, 2),
        coarse_group_size=3,
        subspace_rank=3,
    )
    rec = _new_reconstructor(eit_system, config=config)

    noise = rec._estimate_noise_level(np.zeros(rec.n_measurements))
    assert noise >= rec.config.noise_floor

    hierarchy = rec._build_coarse_hierarchy()
    assert hierarchy
    assert hierarchy[0][0] >= hierarchy[-1][0]

    jacobian = np.eye(rec.n_measurements, rec.n_elements)
    basis, reduced, U_k, s_k = rec._compute_projection(jacobian, rank=3)
    assert basis.shape[1] == 3
    assert reduced.shape[1] == 3
    assert U_k.shape[1] == 3
    assert s_k.shape[0] == 3

    lipschitz = rec._estimate_lipschitz_constant(jacobian)
    assert lipschitz > 0


def test_fista_and_irls_cpu_paths(eit_system):
    rec = _new_reconstructor(eit_system)
    rec.config.linear_max_iterations = 80
    rec.config.linear_tolerance = 1e-8
    rec.config.use_gpu = False
    rec.config.smoothing_beta = 1e-6

    A = np.eye(rec.n_measurements, rec.n_elements)
    b = np.linspace(0.0, 1.0, rec.n_measurements)
    warm = np.zeros(rec.n_elements, dtype=float)

    x_fista = rec._solve_fista(A, b, noise_sigma=0.05, prior_scale=0.2, warm_start=warm)
    x_irls = rec._solve_irls(A, b, noise_sigma=0.05, prior_scale=0.2, warm_start=warm)

    assert x_fista.shape[0] == rec.n_elements
    assert x_irls.shape[0] == rec.n_elements
    assert np.isfinite(x_fista).all()
    assert np.isfinite(x_irls).all()


def test_multilevel_and_block_refinement(eit_system):
    config = sparse_module.SparseBayesianConfig(
        coarse_levels=(3,),
        coarse_iterations=3,
        coarse_relaxation=0.8,
        refinement_gradient_tol=1e-12,
        block_iterations=2,
        block_size=4,
    )
    rec = _new_reconstructor(eit_system, config=config)

    jacobian = np.eye(rec.n_measurements, rec.n_elements)
    data = np.linspace(-0.2, 0.2, rec.n_measurements)
    init = np.zeros(rec.n_elements, dtype=float)
    hierarchy = rec._build_coarse_hierarchy()

    corrected = rec._multilevel_correction(
        jacobian=jacobian,
        data_vector=data,
        noise_sigma=0.1,
        prior_scale=0.3,
        solution=init,
        hierarchy=hierarchy,
    )
    refined = rec._block_refinement(
        jacobian=jacobian,
        data_vector=data,
        noise_sigma=0.1,
        prior_scale=0.3,
        solution=corrected,
    )

    assert corrected.shape == init.shape
    assert refined.shape == init.shape
    assert np.isfinite(refined).all()


def test_sparse_map_solver_branches_with_fake_cuqi(eit_system, monkeypatch):
    config = sparse_module.SparseBayesianConfig(
        solver="map",
        coarse_levels=(4,),
        coarse_iterations=1,
        block_iterations=1,
        block_size=3,
        subspace_rank=2,
        use_linear_warm_start=True,
    )
    rec = _new_reconstructor(eit_system, config=config)

    monkeypatch.setattr(sparse_module, "LinearModel", _FakeLinearModel, raising=False)
    monkeypatch.setattr(sparse_module, "SmoothedLaplace", _FakeSmoothedLaplace, raising=False)
    monkeypatch.setattr(sparse_module, "Gaussian", _FakeGaussian, raising=False)
    monkeypatch.setattr(sparse_module, "BayesianProblem", _FakeBayesianProblem, raising=False)

    jacobian = np.eye(rec.n_measurements, rec.n_elements)
    data = np.linspace(-0.05, 0.05, rec.n_measurements)

    solution = rec._solve_sparse_map(jacobian, data, noise_sigma=0.1, prior_scale=0.4)
    assert solution.shape[0] == rec.n_elements
    assert np.isfinite(solution).all()

    rec.config.solver = "fista"
    rec.config.coarse_levels = None
    rec.config.subspace_rank = None
    x_fista = rec._solve_sparse_map(jacobian, data, noise_sigma=0.1, prior_scale=0.4)
    assert x_fista.shape[0] == rec.n_elements

    rec.config.solver = "irls"
    x_irls = rec._solve_sparse_map(jacobian, data, noise_sigma=0.1, prior_scale=0.4)
    assert x_irls.shape[0] == rec.n_elements

    rec.config.solver = "unsupported"
    with pytest.raises(ValueError):
        rec._solve_sparse_map(jacobian, data, noise_sigma=0.1, prior_scale=0.4)
