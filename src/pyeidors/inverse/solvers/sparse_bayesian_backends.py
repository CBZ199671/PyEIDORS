"""Backend mixin for sparse Bayesian solver helpers."""

from __future__ import annotations

import numpy as np

from .sparse_map_solver import block_refinement, coarse_initialization, multilevel_correction, solve_sparse_map
from .sparse_optimizers import solve_fista, solve_irls
from .sparse_projection import compute_projection, estimate_lipschitz_constant, get_coarse_matrix


class SparseBayesianBackendMixin:
    """Shared wrapper methods for sparse MAP and refinement helpers."""

    def _solve_sparse_map(
        self,
        jacobian: np.ndarray,
        data_vector: np.ndarray,
        noise_sigma: float,
        prior_scale: float,
    ) -> np.ndarray:
        return solve_sparse_map(self, jacobian, data_vector, noise_sigma, prior_scale)

    def _coarse_initialization(
        self,
        jacobian: np.ndarray,
        data_vector: np.ndarray,
        noise_sigma: float,
        prior_scale: float,
        groups: list[np.ndarray],
        group_size: int,
        initial_guess: np.ndarray | None,
    ) -> np.ndarray:
        return coarse_initialization(
            self,
            jacobian,
            data_vector,
            noise_sigma,
            prior_scale,
            groups,
            group_size,
            initial_guess,
        )

    def _get_coarse_matrix(
        self,
        jacobian: np.ndarray,
        groups: list[np.ndarray],
        group_size: int,
    ) -> np.ndarray:
        return get_coarse_matrix(
            jacobian=jacobian,
            groups=groups,
            group_size=group_size,
            cache=self._cached_coarse_matrices,
        )

    def _compute_projection(self, jacobian: np.ndarray, rank: int):
        return compute_projection(jacobian, rank)

    def _estimate_lipschitz_constant(self, matrix: np.ndarray, iters: int = 12) -> float:
        return estimate_lipschitz_constant(matrix, iters=iters)

    def _solve_with_cuqi_map(self, problem, warm_start: np.ndarray | None) -> np.ndarray:
        if warm_start is not None:
            map_estimate = problem.MAP(disp=self.verbose, x0=warm_start)
        else:
            map_estimate = problem.MAP(disp=self.verbose)
        return np.asarray(map_estimate.to_numpy(), dtype=float)

    def _solve_fista(
        self,
        linear_matrix: np.ndarray,
        data_vector: np.ndarray,
        noise_sigma: float,
        prior_scale: float,
        warm_start: np.ndarray | None,
    ) -> np.ndarray:
        return solve_fista(
            linear_matrix=linear_matrix,
            data_vector=data_vector,
            noise_sigma=noise_sigma,
            prior_scale=prior_scale,
            warm_start=warm_start,
            config=self.config,
        )

    def _solve_irls(
        self,
        linear_matrix: np.ndarray,
        data_vector: np.ndarray,
        noise_sigma: float,
        prior_scale: float,
        warm_start: np.ndarray | None,
    ) -> np.ndarray:
        return solve_irls(
            linear_matrix=linear_matrix,
            data_vector=data_vector,
            noise_sigma=noise_sigma,
            prior_scale=prior_scale,
            warm_start=warm_start,
            config=self.config,
        )

    def _multilevel_correction(
        self,
        jacobian: np.ndarray,
        data_vector: np.ndarray,
        noise_sigma: float,
        prior_scale: float,
        solution: np.ndarray,
        hierarchy: list[tuple[int, list[np.ndarray]]],
    ) -> np.ndarray:
        return multilevel_correction(
            self,
            jacobian,
            data_vector,
            noise_sigma,
            prior_scale,
            solution,
            hierarchy,
        )

    def _block_refinement(
        self,
        jacobian: np.ndarray,
        data_vector: np.ndarray,
        noise_sigma: float,
        prior_scale: float,
        solution: np.ndarray,
    ) -> np.ndarray:
        return block_refinement(
            self,
            jacobian,
            data_vector,
            noise_sigma,
            prior_scale,
            solution,
        )

    def _linear_model(self, matrix: np.ndarray):
        from . import sparse_bayesian_engine as sparse_module

        model_cls = getattr(sparse_module, "LinearModel", None)
        if model_cls is None:
            raise ImportError("CUQIpy is required for SparseBayesianReconstructor")
        return model_cls(matrix)

    def _sparse_prior(self, target_dim: int, prior_scale: float):
        from . import sparse_bayesian_engine as sparse_module

        prior_cls = getattr(sparse_module, "SmoothedLaplace", None)
        if prior_cls is None:
            raise ImportError("CUQIpy is required for SparseBayesianReconstructor")
        return prior_cls(
            location=np.zeros(target_dim, dtype=float),
            scale=prior_scale,
            beta=self.config.smoothing_beta,
        )

    @staticmethod
    def _gaussian_likelihood(latent, noise_sigma: float):
        from . import sparse_bayesian_engine as sparse_module

        gaussian_cls = getattr(sparse_module, "Gaussian", None)
        if gaussian_cls is None:
            raise ImportError("CUQIpy is required for SparseBayesianReconstructor")
        return gaussian_cls(latent, noise_sigma)

    @staticmethod
    def _bayesian_problem(y, x):
        from . import sparse_bayesian_engine as sparse_module

        problem_cls = getattr(sparse_module, "BayesianProblem", None)
        if problem_cls is None:
            raise ImportError("CUQIpy is required for SparseBayesianReconstructor")
        return problem_cls(y, x)
