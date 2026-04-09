"""Sparse Bayesian EIT reconstructor powered by CUQIpy."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from ...data.structures import EITData, EITImage
from ...cache.object_signature import (
    backend_signature_from_forward_model,
    model_signature_from_forward_model,
    pattern_signature_from_forward_model,
)
from ...utils.cuqi_imports import suppress_known_cuqi_import_warnings
from .sparse_bayesian_backends import SparseBayesianBackendMixin
from .eit_pde import EITPDE, create_pde_model
from .sparse_projection import build_coarse_hierarchy
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


class SparseBayesianReconstructor(SparseBayesianBackendMixin):
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
        if cache_manager is not None and cache_manager.enabled and self.config.cache_jacobian:
            baseline = np.ascontiguousarray(baseline_values, dtype=np.float64)
            payload = {
                "solver": "sparse_bayesian",
                "baseline_hash": hashlib.sha256(baseline.tobytes()).hexdigest(),
                "n_elements": self.n_elements,
                "n_measurements": self.n_measurements,
                "subspace_rank": self.config.subspace_rank,
                "coarse_levels": tuple(self.config.coarse_levels or ()),
                "model_signature": model_signature_from_forward_model(self.fwd_model),
                "pattern_signature": pattern_signature_from_forward_model(self.fwd_model),
                "backend_signature": backend_signature_from_forward_model(self.fwd_model),
            }
            jacobian, _ = cache_manager.get_or_compute_semantic(
                artifact="jacobian",
                name="calc_jacobian",
                namespace="sparse",
                cache_obj=payload,
                payload=payload,
                compute_fn=lambda: self._eit_pde.jacobian_wrt_parameter(baseline_values),
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
        noise_sigma = max(float(np.std(data_vector) * self.config.noise_rel), self.config.noise_floor)
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
