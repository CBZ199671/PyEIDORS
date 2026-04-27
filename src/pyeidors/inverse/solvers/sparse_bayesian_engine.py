"""Sparse Bayesian EIT reconstructor powered by CUQIpy.

T76 Path C consolidation: the historical ``SparseBayesianBackendMixin``
(formerly in ``sparse_bayesian_backends.py``) and the legacy alias
module ``sparse_bayesian.py`` were folded into this engine module so
the sparse solver tier collapses from 7 files to 5. The cuqi adapters
(:meth:`_linear_model` / :meth:`_sparse_prior` / :meth:`_gaussian_likelihood`
/ :meth:`_bayesian_problem` / :meth:`_solve_with_cuqi_map`) and the
thin wrappers around the FISTA / IRLS / projection / coarse-matrix /
multilevel-correction / block-refinement helpers all live directly on
:class:`SparseBayesianReconstructor` now. Tests + script callers that
monkey-patch the wrapper methods continue to work unchanged.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...cache.object_signature import (
    backend_signature_from_forward_model,
    model_signature_from_forward_model,
    pattern_signature_from_forward_model,
)
from ...data.structures import EITData, EITImage
from ...utils.cuqi_imports import suppress_known_cuqi_import_warnings
from .eit_pde import EITPDE, create_pde_model
from .sparse_map_solver import (
    block_refinement,
    coarse_initialization,
    multilevel_correction,
    solve_sparse_map,
)
from .sparse_optimizers import solve_fista, solve_irls
from .sparse_projection import (
    build_coarse_hierarchy,
    compute_projection,
    estimate_lipschitz_constant,
    get_coarse_matrix,
)
from .sparse_runtime import run_sparse_reconstruction

try:  # pragma: no cover - optional dependency guard
    with suppress_known_cuqi_import_warnings():
        from cuqi.distribution import Gaussian, SmoothedLaplace
        from cuqi.model import LinearModel
        from cuqi.problem import BayesianProblem

    _CUQI_AVAILABLE = True
except ImportError:  # pragma: no cover
    Gaussian = None
    SmoothedLaplace = None
    LinearModel = None
    BayesianProblem = None
    _CUQI_AVAILABLE = False


@dataclass
class SparseBayesianConfig:
    """Configuration parameters for the sparse Bayesian reconstructor."""

    prior_scale: float = 5e-2
    smoothing_beta: float = 1e-6
    noise_rel: float = 0.02
    noise_floor: float = 1e-6
    clip_values: tuple[float, float] | None = (1e-6, 10.0)
    cache_jacobian: bool = True
    subspace_rank: int | None = None
    use_linear_warm_start: bool = False
    solver: str = "map"  # map | fista | irls
    linear_max_iterations: int = 200
    linear_tolerance: float = 1e-6
    coarse_group_size: int | None = None
    use_gpu: bool = False
    gpu_dtype: str = "float32"
    coarse_levels: tuple[int, ...] | None = None
    block_iterations: int = 0
    block_size: int | None = None
    refinement_gradient_tol: float = 1e-5
    coarse_iterations: int = 0
    coarse_relaxation: float = 1.0


class SparseBayesianReconstructor:
    """Sparse Bayesian reconstructor using CUQIpy."""

    def __init__(
        self,
        eit_system,
        config: SparseBayesianConfig | None = None,
        verbose: bool = True,
    ) -> None:
        if not _CUQI_AVAILABLE:  # pragma: no cover
            raise ImportError(
                "CUQIpy is required for SparseBayesianReconstructor. "
                "Please install cuqipy."
            )

        self.eit_system = eit_system
        self.fwd_model = eit_system.fwd_model
        self.verbose = verbose
        self.config = config or SparseBayesianConfig()

        self._eit_pde: EITPDE
        self._cuqi_model, geometry = self._initialise_pde_model()

        self._cached_jacobian: np.ndarray | None = None
        self._cached_baseline: np.ndarray | None = None
        self._cached_basis: np.ndarray | None = None
        self._cached_reduced_matrix: np.ndarray | None = None
        self._cached_U: np.ndarray | None = None
        self._cached_singular: np.ndarray | None = None
        self._coarse_levels_cache: dict[int, list[np.ndarray]] = {}
        self._cached_coarse_matrices: dict[int, np.ndarray] = {}

        self.n_elements = geometry.n_elements
        self.n_measurements = geometry.n_measurements

    def reconstruct(
        self,
        measurement_data: EITData,
        baseline_image: EITImage | None = None,
        reference_data: EITData | None = None,
        initial_conductivity: float = 1.0,
        noise_std: float | None = None,
        prior_scale: float | None = None,
        clip_values: tuple[float, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        return run_sparse_reconstruction(
            self,
            measurement_data=measurement_data,
            baseline_image=baseline_image,
            reference_data=reference_data,
            initial_conductivity=initial_conductivity,
            noise_std=noise_std,
            prior_scale=prior_scale,
            clip_values=clip_values,
            metadata=metadata,
        )

    def _initialise_pde_model(self):
        eit_pde, model, geometry = create_pde_model(self.eit_system)
        self._eit_pde = eit_pde
        return model, geometry

    def _forward_measurement(self, conductivity_values: np.ndarray) -> np.ndarray:
        cuqi_result = self._cuqi_model(conductivity_values)
        if hasattr(cuqi_result, "to_numpy"):
            return np.asarray(cuqi_result.to_numpy(), dtype=float).ravel()
        return np.asarray(cuqi_result, dtype=float).ravel()

    def _create_homogeneous_image(self, conductivity: float) -> EITImage:
        values = np.full(self.n_elements, conductivity, dtype=float)
        return EITImage(elem_data=values, fwd_model=self.fwd_model)

    def _prepare_jacobian(self, baseline_values: np.ndarray) -> np.ndarray:
        if (
            self.config.cache_jacobian
            and self._cached_jacobian is not None
            and self._cached_baseline is not None
            and np.allclose(self._cached_baseline, baseline_values)
        ):
            return self._cached_jacobian

        cache_manager = getattr(self.eit_system, "cache_manager", None)
        if (
            cache_manager is not None
            and cache_manager.enabled
            and self.config.cache_jacobian
        ):
            baseline = np.ascontiguousarray(baseline_values, dtype=np.float64)
            payload = {
                "solver": "sparse_bayesian",
                "baseline_hash": hashlib.sha256(baseline.tobytes()).hexdigest(),
                "n_elements": self.n_elements,
                "n_measurements": self.n_measurements,
                "subspace_rank": self.config.subspace_rank,
                "coarse_levels": tuple(self.config.coarse_levels or ()),
                "model_signature": model_signature_from_forward_model(self.fwd_model),
                "pattern_signature": pattern_signature_from_forward_model(
                    self.fwd_model
                ),
                "backend_signature": backend_signature_from_forward_model(
                    self.fwd_model
                ),
            }
            jacobian, _ = cache_manager.get_or_compute_semantic(
                artifact="jacobian",
                name="calc_jacobian",
                namespace="sparse",
                cache_obj=payload,
                payload=payload,
                compute_fn=lambda: self._eit_pde.jacobian_wrt_parameter(
                    baseline_values
                ),
                persist=True,
                cost=10.0,
            )
        else:
            jacobian = self._eit_pde.jacobian_wrt_parameter(baseline_values)

        if self.config.cache_jacobian:
            self._cached_jacobian = jacobian
            self._cached_baseline = baseline_values.copy()
        else:
            self._cached_jacobian = None
            self._cached_baseline = None

        self._cached_basis = None
        self._cached_reduced_matrix = None
        self._cached_U = None
        self._cached_singular = None
        self._cached_coarse_matrices = {}
        return jacobian

    def _estimate_noise_level(self, data_vector: np.ndarray) -> float:
        noise_sigma = max(
            float(np.std(data_vector) * self.config.noise_rel), self.config.noise_floor
        )
        if not np.isfinite(noise_sigma) or noise_sigma <= 0:
            noise_sigma = self.config.noise_floor
        return noise_sigma

    def _build_coarse_hierarchy(self) -> list[tuple[int, list[np.ndarray]]]:
        cache_manager = getattr(self.eit_system, "cache_manager", None)
        if cache_manager is None or not cache_manager.enabled:
            return build_coarse_hierarchy(
                config=self.config,
                n_elements=self.n_elements,
                cache=self._coarse_levels_cache,
            )

        payload = {
            "solver": "sparse_bayesian",
            "n_elements": self.n_elements,
            "coarse_group_size": self.config.coarse_group_size,
            "coarse_levels": tuple(self.config.coarse_levels or ()),
        }
        hierarchy, _ = cache_manager.get_or_compute(
            artifact="sparse_basis",
            payload=payload,
            compute_fn=lambda: build_coarse_hierarchy(
                config=self.config,
                n_elements=self.n_elements,
                cache=self._coarse_levels_cache,
            ),
            name="coarse_hierarchy",
            namespace="sparse",
            persist=True,
            cost=8.0,
        )
        return hierarchy

    # ------------------------------------------------------------------
    # Backend-mixin methods (folded in by T76 Path C). The thin wrapper
    # layer historically lived in ``sparse_bayesian_backends.py``; tests
    # and ``sparse_map_solver`` consumers monkey-patch / reach through
    # these hooks, so they remain instance methods even though the
    # forwarding bodies are one-liners.
    # ------------------------------------------------------------------

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

    def _estimate_lipschitz_constant(
        self, matrix: np.ndarray, iters: int = 12
    ) -> float:
        return estimate_lipschitz_constant(matrix, iters=iters)

    def _solve_with_cuqi_map(
        self, problem, warm_start: np.ndarray | None
    ) -> np.ndarray:
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
        if LinearModel is None:
            raise ImportError("CUQIpy is required for SparseBayesianReconstructor")
        return LinearModel(matrix)

    def _sparse_prior(self, target_dim: int, prior_scale: float):
        if SmoothedLaplace is None:
            raise ImportError("CUQIpy is required for SparseBayesianReconstructor")
        return SmoothedLaplace(
            location=np.zeros(target_dim, dtype=float),
            scale=prior_scale,
            beta=self.config.smoothing_beta,
        )

    @staticmethod
    def _gaussian_likelihood(latent, noise_sigma: float):
        if Gaussian is None:
            raise ImportError("CUQIpy is required for SparseBayesianReconstructor")
        return Gaussian(latent, noise_sigma)

    @staticmethod
    def _bayesian_problem(y, x):
        if BayesianProblem is None:
            raise ImportError("CUQIpy is required for SparseBayesianReconstructor")
        return BayesianProblem(y, x)
