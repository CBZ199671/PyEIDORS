"""Additional branch coverage for sparse MAP solver helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pyeidors.inverse.solvers import sparse_map_solver as sparse_map_module


class _FakeMapResult:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def to_numpy(self):
        return self._values.copy()


class _FakeProblem:
    def __init__(self, result):
        self._result = np.asarray(result, dtype=float)
        self.last_x0 = None
        self.data = None

    def set_data(self, **kwargs):
        self.data = kwargs
        return self

    def MAP(self, disp=False, x0=None):
        _ = disp
        self.last_x0 = None if x0 is None else np.asarray(x0, dtype=float)
        return _FakeMapResult(self._result)


class _FakeLinearModel:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=float)

    def __matmul__(self, other):
        return ("linear", self.matrix, other)


def _reconstructor(**overrides):
    config = SimpleNamespace(
        subspace_rank=None,
        use_linear_warm_start=False,
        solver="map",
        coarse_iterations=0,
        coarse_relaxation=1.0,
        refinement_gradient_tol=0.0,
        block_iterations=0,
        block_size=None,
    )
    recon = SimpleNamespace(
        n_elements=4,
        config=config,
        verbose=False,
        _cached_basis=None,
        _cached_reduced_matrix=None,
        _cached_U=None,
        _cached_singular=None,
        _cached_coarse_matrices={},
        _compute_projection=lambda jacobian, rank: (
            np.eye(jacobian.shape[1], rank, dtype=float),
            jacobian[:, :rank].copy(),
            np.eye(jacobian.shape[0], rank, dtype=float),
            np.arange(1, rank + 1, dtype=float),
        ),
        _linear_model=lambda matrix: _FakeLinearModel(matrix),
        _sparse_prior=lambda dim, prior_scale: ("prior", int(dim), float(prior_scale)),
        _gaussian_likelihood=lambda latent, noise_sigma: ("gaussian", latent, float(noise_sigma)),
        _bayesian_problem=lambda y, x: _FakeProblem(np.linspace(0.1, 0.2, 4)),
        _solve_with_cuqi_map=lambda problem, warm_start: np.asarray([0.1, 0.2], dtype=float)
        if warm_start is not None
        else np.asarray([0.0, 0.1], dtype=float),
        _solve_fista=lambda *args, **kwargs: np.asarray([0.3, 0.4], dtype=float),
        _solve_irls=lambda *args, **kwargs: np.asarray([0.5, 0.6], dtype=float),
        _build_coarse_hierarchy=lambda: [],
        _multilevel_correction=lambda jacobian, data_vector, noise_sigma, prior_scale, solution, hierarchy: solution,
        _block_refinement=lambda jacobian, data_vector, noise_sigma, prior_scale, solution: solution,
        _get_coarse_matrix=lambda jacobian, groups, group_size: np.column_stack(
            [np.sum(jacobian[:, idx], axis=1) for idx in groups]
        ) if groups else np.zeros((jacobian.shape[0], 0), dtype=float),
    )
    for key, value in overrides.items():
        setattr(recon, key, value)
    return recon


def test_projection_and_warm_start_helper_branches():
    jac = np.arange(12, dtype=float).reshape(3, 4)
    recon = _reconstructor()
    recon.config.subspace_rank = 2

    linear_matrix, target_dim, basis, U_k, s_k = sparse_map_module._resolve_projection(recon, jac)
    assert target_dim == 2
    assert basis.shape == (4, 2)
    assert recon._cached_basis is not None

    linear_cached, _, basis_cached, _, _ = sparse_map_module._resolve_projection(recon, jac)
    assert basis_cached is recon._cached_basis
    np.testing.assert_allclose(linear_cached, recon._cached_reduced_matrix)

    recon.config.subspace_rank = 10
    linear_full, target_full, basis_none, _, _ = sparse_map_module._resolve_projection(recon, jac)
    assert target_full == 4
    assert basis_none is None
    np.testing.assert_allclose(linear_full, jac)

    coarse = np.array([1.0, 2.0], dtype=float)
    np.testing.assert_allclose(sparse_map_module._coarse_warm_start(None, coarse), coarse)
    np.testing.assert_allclose(
        sparse_map_module._coarse_warm_start(np.eye(2, dtype=float), coarse),
        coarse,
    )

    warm_sub = sparse_map_module._linear_warm_start_subspace(
        np.eye(2, dtype=float),
        np.array([2.0, 0.0], dtype=float),
        np.array([4.0, 5.0], dtype=float),
    )
    np.testing.assert_allclose(warm_sub, np.array([2.0, 0.0], dtype=float))

    recon.config.use_linear_warm_start = False
    assert sparse_map_module._resolve_warm_start(
        reconstructor=recon,
        basis=None,
        coarse_init=None,
        data_vector=np.array([1.0, 2.0], dtype=float),
        hierarchy=[],
        linear_matrix=np.eye(2, dtype=float),
        U_k=None,
        s_k=None,
    ) is None

    recon.config.use_linear_warm_start = True
    warm_subspace = sparse_map_module._resolve_warm_start(
        reconstructor=recon,
        basis=np.eye(2, dtype=float),
        coarse_init=None,
        data_vector=np.array([4.0, 5.0], dtype=float),
        hierarchy=[(2, [np.array([0, 1])])],
        linear_matrix=np.eye(2, dtype=float),
        U_k=np.eye(2, dtype=float),
        s_k=np.array([2.0, 0.0], dtype=float),
    )
    np.testing.assert_allclose(warm_subspace, np.array([2.0, 0.0], dtype=float))

    recon.config.use_linear_warm_start = True
    full_warm = sparse_map_module._resolve_warm_start(
        reconstructor=recon,
        basis=None,
        coarse_init=None,
        data_vector=np.array([3.0, 4.0], dtype=float),
        hierarchy=[],
        linear_matrix=np.diag([2.0, 4.0]).astype(float),
        U_k=None,
        s_k=None,
    )
    np.testing.assert_allclose(full_warm, np.array([1.5, 1.0], dtype=float))

    assert sparse_map_module._resolve_warm_start(
        reconstructor=recon,
        basis=None,
        coarse_init=None,
        data_vector=np.array([1.0, 2.0], dtype=float),
        hierarchy=[(2, [np.array([0, 1])])],
        linear_matrix=np.eye(2, dtype=float),
        U_k=None,
        s_k=None,
    ) is None

    assert sparse_map_module._resolve_solver_type(SimpleNamespace(solver="fista"), [(2, [])]) == "map"
    assert sparse_map_module._resolve_solver_type(SimpleNamespace(solver="irls"), []) == "irls"


def test_coarse_initialization_and_solve_sparse_map_paths():
    jac = np.array([[1.0, 2.0, 0.0, 0.0], [0.0, 1.0, 3.0, 4.0]], dtype=float)
    groups = [np.array([0, 1]), np.array([2, 3])]
    problem = _FakeProblem([10.0, 20.0])
    recon = _reconstructor(
        _bayesian_problem=lambda y, x: problem,
    )

    coarse = sparse_map_module.coarse_initialization(
        recon,
        jacobian=jac,
        data_vector=np.array([1.0, 2.0], dtype=float),
        noise_sigma=0.1,
        prior_scale=0.2,
        groups=groups,
        group_size=2,
        initial_guess=np.array([2.0, 4.0, 6.0, 8.0], dtype=float),
    )
    np.testing.assert_allclose(problem.last_x0, np.array([3.0, 7.0], dtype=float))
    np.testing.assert_allclose(coarse, np.array([10.0, 10.0, 20.0, 20.0], dtype=float))

    recon_map = _reconstructor()
    recon_map.config.subspace_rank = None
    recon_map.config.use_linear_warm_start = False
    recon_map._solve_with_cuqi_map = lambda problem, warm_start: np.asarray([0.0, 0.1, 0.2, 0.3], dtype=float)
    result_map = sparse_map_module.solve_sparse_map(
        recon_map,
        jacobian=np.eye(4, dtype=float),
        data_vector=np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
        noise_sigma=0.1,
        prior_scale=0.3,
    )
    assert result_map.shape == (4,)

    recon_fista = _reconstructor()
    recon_fista.config.solver = "fista"
    recon_fista._solve_fista = lambda *args, **kwargs: np.asarray([0.3, 0.4, 0.5, 0.6], dtype=float)
    result_fista = sparse_map_module.solve_sparse_map(
        recon_fista,
        jacobian=np.eye(4, dtype=float),
        data_vector=np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
        noise_sigma=0.1,
        prior_scale=0.3,
    )
    assert result_fista.shape == (4,)

    recon_irls = _reconstructor()
    recon_irls.config.solver = "irls"
    recon_irls._solve_irls = lambda *args, **kwargs: np.asarray([0.5, 0.6, 0.7, 0.8], dtype=float)
    result_irls = sparse_map_module.solve_sparse_map(
        recon_irls,
        jacobian=np.eye(4, dtype=float),
        data_vector=np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
        noise_sigma=0.1,
        prior_scale=0.3,
    )
    assert result_irls.shape == (4,)

    recon_bad = _reconstructor()
    recon_bad.config.solver = "weird"
    with pytest.raises(ValueError, match="Unknown solver type"):
        sparse_map_module.solve_sparse_map(
            recon_bad,
            jacobian=np.eye(4, dtype=float),
            data_vector=np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
            noise_sigma=0.1,
            prior_scale=0.3,
        )


def test_multilevel_and_block_refinement_fallback_paths(monkeypatch: pytest.MonkeyPatch):
    jac = np.eye(4, dtype=float)
    data = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    solution = np.zeros(4, dtype=float)

    recon = _reconstructor()
    recon.config.coarse_iterations = 1
    recon.config.coarse_relaxation = 0.5
    recon.config.refinement_gradient_tol = 0.0
    recon._get_coarse_matrix = lambda jacobian, groups, group_size: np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    solve_calls = {"count": 0}

    def _solve_then_lstsq(H, rhs):
        solve_calls["count"] += 1
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(sparse_map_module.np.linalg, "solve", _solve_then_lstsq)
    monkeypatch.setattr(
        sparse_map_module.np.linalg,
        "lstsq",
        lambda H, rhs, rcond=None: (np.ones(rhs.shape[0], dtype=float) * 0.2, None, None, None),
    )

    corrected = sparse_map_module.multilevel_correction(
        recon,
        jacobian=jac,
        data_vector=data,
        noise_sigma=1.0,
        prior_scale=1.0,
        solution=solution,
        hierarchy=[(2, [np.array([0, 1]), np.array([2, 3])]), (1, [])],
    )
    assert solve_calls["count"] >= 1
    assert corrected.shape == solution.shape

    recon_tol = _reconstructor()
    recon_tol.config.coarse_iterations = 1
    recon_tol.config.refinement_gradient_tol = 10.0
    tol_out = sparse_map_module.multilevel_correction(
        recon_tol,
        jacobian=jac,
        data_vector=np.zeros(4, dtype=float),
        noise_sigma=1.0,
        prior_scale=1.0,
        solution=np.zeros(4, dtype=float),
        hierarchy=[(2, [np.array([0, 1]), np.array([2, 3])])],
    )
    np.testing.assert_allclose(tol_out, np.zeros(4, dtype=float))

    recon_block = _reconstructor()
    recon_block.config.block_iterations = 1
    recon_block.config.block_size = 2
    recon_block.config.refinement_gradient_tol = 0.0

    monkeypatch.setattr(sparse_map_module.np.linalg, "solve", _solve_then_lstsq)
    refined = sparse_map_module.block_refinement(
        recon_block,
        jacobian=jac,
        data_vector=data,
        noise_sigma=1.0,
        prior_scale=1.0,
        solution=np.zeros(4, dtype=float),
    )
    assert refined.shape == (4,)

    recon_block_skip = _reconstructor()
    recon_block_skip.config.block_iterations = 1
    recon_block_skip.config.block_size = 2
    recon_block_skip.config.refinement_gradient_tol = 10.0
    skipped = sparse_map_module.block_refinement(
        recon_block_skip,
        jacobian=np.zeros((4, 4), dtype=float),
        data_vector=np.zeros(4, dtype=float),
        noise_sigma=1.0,
        prior_scale=1.0,
        solution=np.zeros(4, dtype=float),
    )
    np.testing.assert_allclose(skipped, np.zeros(4, dtype=float))

    empty = sparse_map_module.block_refinement(
        recon_block_skip,
        jacobian=np.zeros((0, 0), dtype=float),
        data_vector=np.zeros(0, dtype=float),
        noise_sigma=1.0,
        prior_scale=1.0,
        solution=np.zeros(0, dtype=float),
    )
    assert empty.size == 0


def test_multilevel_and_block_refinement_remaining_skip_branches(monkeypatch: pytest.MonkeyPatch):
    jac = np.eye(4, dtype=float)
    data = np.zeros(4, dtype=float)
    solution = np.zeros(4, dtype=float)

    recon = _reconstructor()
    recon.config.coarse_iterations = 1
    recon._get_coarse_matrix = lambda _jacobian, _groups, _size: np.zeros((jac.shape[0], 0), dtype=float)
    out_skip_A = sparse_map_module.multilevel_correction(
        recon,
        jacobian=jac,
        data_vector=data,
        noise_sigma=1.0,
        prior_scale=1.0,
        solution=solution,
        hierarchy=[(2, [np.array([0, 1])])],
    )
    np.testing.assert_allclose(out_skip_A, solution)

    recon_empty_delta = _reconstructor()
    recon_empty_delta.config.coarse_iterations = 1
    recon_empty_delta._get_coarse_matrix = lambda _jacobian, groups, _size: np.eye(len(groups), dtype=float)
    monkeypatch.setattr(sparse_map_module.np.linalg, "solve", lambda *_args, **_kwargs: np.array([], dtype=float))
    out_empty_delta = sparse_map_module.multilevel_correction(
        recon_empty_delta,
        jacobian=jac,
        data_vector=data,
        noise_sigma=1.0,
        prior_scale=1.0,
        solution=solution,
        hierarchy=[(2, [np.array([0]), np.array([1])])],
    )
    np.testing.assert_allclose(out_empty_delta, solution)

    recon_blocks = _reconstructor()
    recon_blocks.config.block_iterations = 1
    recon_blocks.config.block_size = 2
    recon_blocks.config.refinement_gradient_tol = 1e-9

    no_row_refine = sparse_map_module.block_refinement(
        recon_blocks,
        jacobian=np.zeros((0, 4), dtype=float),
        data_vector=np.zeros(0, dtype=float),
        noise_sigma=1.0,
        prior_scale=1.0,
        solution=np.zeros(4, dtype=float),
    )
    np.testing.assert_allclose(no_row_refine, np.zeros(4, dtype=float))

    monkeypatch.setattr(sparse_map_module.np.linalg, "solve", lambda *_args, **_kwargs: np.array([], dtype=float))
    empty_delta_refine = sparse_map_module.block_refinement(
        recon_blocks,
        jacobian=np.eye(4, dtype=float),
        data_vector=np.ones(4, dtype=float),
        noise_sigma=1.0,
        prior_scale=1.0,
        solution=np.zeros(4, dtype=float),
    )
    np.testing.assert_allclose(empty_delta_refine, np.zeros(4, dtype=float))

    monkeypatch.setattr(sparse_map_module.np.linalg, "solve", lambda *_args, **_kwargs: np.array([1e-6, 1e-6], dtype=float))
    recon_small_delta = _reconstructor()
    recon_small_delta.config.block_iterations = 1
    recon_small_delta.config.block_size = 2
    recon_small_delta.config.refinement_gradient_tol = 1e-3
    tol_refine = sparse_map_module.block_refinement(
        recon_small_delta,
        jacobian=np.eye(4, dtype=float),
        data_vector=np.ones(4, dtype=float),
        noise_sigma=1.0,
        prior_scale=1.0,
        solution=np.zeros(4, dtype=float),
    )
    np.testing.assert_allclose(tol_refine, np.zeros(4, dtype=float))

    recon_zero_rows = _reconstructor()
    recon_zero_rows.config.block_iterations = 1
    recon_zero_rows.config.block_size = 2
    recon_zero_rows.config.refinement_gradient_tol = 0.0
    zero_row_refine = sparse_map_module.block_refinement(
        recon_zero_rows,
        jacobian=np.zeros((0, 4), dtype=float),
        data_vector=np.zeros(0, dtype=float),
        noise_sigma=1.0,
        prior_scale=1.0,
        solution=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
    )
    np.testing.assert_allclose(zero_row_refine, np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
