"""Sparse Bayesian EIT reconstructor powered by CUQIpy.

This module provides sparse Bayesian inversion utilities that operate on top of
CUQIpy and FEniCS. It offers both direct MAP optimisation (delegating to
CUQIpy) and lightweight linearised alternatives (FISTA/IRLS) that can operate
in a truncated SVD subspace for speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
from fenics import Function

from ...data.structures import EITData, EITImage
from .eit_pde import create_pde_model, EITPDE

try:  # pragma: no cover - optional dependency guard
    from cuqi.distribution import Gaussian, SmoothedLaplace
    from cuqi.model import LinearModel
    from cuqi.problem import BayesianProblem
    _CUQI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CUQI_AVAILABLE = False


@dataclass
class SparseBayesianConfig:
    """Configuration parameters for the sparse Bayesian reconstructor."""

    prior_scale: float = 5e-2
    smoothing_beta: float = 1e-6
    noise_rel: float = 0.02
    noise_floor: float = 1e-6
    clip_values: Optional[Tuple[float, float]] = (1e-6, 10.0)
    cache_jacobian: bool = True
    subspace_rank: Optional[int] = None
    use_linear_warm_start: bool = False
    solver: str = "map"  # map | fista | irls
    linear_max_iterations: int = 200
    linear_tolerance: float = 1e-6
    coarse_group_size: Optional[int] = None
    use_gpu: bool = False
    gpu_dtype: str = "float32"
    coarse_levels: Optional[Tuple[int, ...]] = None
    block_iterations: int = 0
    block_size: Optional[int] = None
    refinement_gradient_tol: float = 1e-5
    coarse_iterations: int = 0
    coarse_relaxation: float = 1.0


class SparseBayesianReconstructor:
    """Sparse Bayesian reconstructor using CUQIpy."""

    def __init__(
        self,
        eit_system,
        config: Optional[SparseBayesianConfig] = None,
        verbose: bool = True,
    ) -> None:
        if not _CUQI_AVAILABLE:  # pragma: no cover
            raise ImportError(
                "CUQIpy is required for SparseBayesianReconstructor. "
                "Please install cuqipy and cuqipy-fenics."
            )

        self.eit_system = eit_system
        self.fwd_model = eit_system.fwd_model
        self.verbose = verbose
        self.config = config or SparseBayesianConfig()

        self._eit_pde: EITPDE
        self._cuqi_model, geometry = self._initialise_pde_model()

        self._cached_jacobian: Optional[np.ndarray] = None
        self._cached_baseline: Optional[np.ndarray] = None
        self._cached_basis: Optional[np.ndarray] = None
        self._cached_reduced_matrix: Optional[np.ndarray] = None
        self._cached_U: Optional[np.ndarray] = None
        self._cached_singular: Optional[np.ndarray] = None
        self._coarse_levels_cache: Dict[int, List[np.ndarray]] = {}
        self._cached_coarse_matrices: Dict[int, np.ndarray] = {}

        self.n_elements = geometry.n_elements
        self.n_measurements = geometry.n_measurements

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reconstruct(
        self,
        measurement_data: EITData,
        baseline_image: Optional[EITImage] = None,
        reference_data: Optional[EITData] = None,
        initial_conductivity: float = 1.0,
        noise_std: Optional[float] = None,
        prior_scale: Optional[float] = None,
        clip_values: Optional[Tuple[float, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute a sparse Bayesian reconstruction."""

        mode = "difference" if reference_data is not None else "absolute"
        clip_bounds = clip_values if clip_values is not None else self.config.clip_values

        baseline_image = baseline_image or self._create_homogeneous_image(initial_conductivity)
        baseline_values = baseline_image.elem_data.copy()

        baseline_meas = self._forward_measurement(baseline_values)
        jacobian = self._prepare_jacobian(baseline_values)

        target_vector = np.asarray(measurement_data.meas, dtype=float).ravel()

        if mode == "difference":
            reference_vector = np.asarray(reference_data.meas, dtype=float).ravel()
            data_vector = target_vector - reference_vector
            baseline_meas = reference_vector
        else:
            data_vector = target_vector - baseline_meas

        noise_sigma = noise_std or self._estimate_noise_level(data_vector)
        prior_scale = prior_scale or self.config.prior_scale

        map_delta = self._solve_sparse_map(jacobian, data_vector, noise_sigma, prior_scale)

        conductivity_values = baseline_values + map_delta
        if clip_bounds is not None:
            conductivity_values = np.clip(conductivity_values, clip_bounds[0], clip_bounds[1])

        conductivity_function = Function(self.fwd_model.V_sigma)
        conductivity_function.vector()[:] = conductivity_values

        reconstructed_image = EITImage(elem_data=conductivity_values, fwd_model=self.fwd_model)
        simulated_vector = self._forward_measurement(conductivity_values)

        if mode == "difference":
            predicted_vector = simulated_vector - baseline_meas
        else:
            predicted_vector = simulated_vector - baseline_meas

        residual_vector = predicted_vector - data_vector

        output: Dict[str, Any] = {
            "mode": mode,
            "conductivity": conductivity_function,
            "delta_sigma": map_delta,
            "baseline_conductivity": baseline_values,
            "posterior_map": map_delta,
            "jacobian": jacobian,
            "observed_data": data_vector,
            "predicted_data": predicted_vector,
            "residual_vector": residual_vector,
            "simulated_measurement": simulated_vector,
            "baseline_measurement": baseline_meas,
            "target_measurement": target_vector,
            "likelihood_noise_std": noise_sigma,
            "prior_scale": prior_scale,
            "clip_bounds": clip_bounds,
            "iterations": 1,
            "converged": True,
            "final_residual": float(np.linalg.norm(residual_vector)),
            "final_relative_change": float(
                np.linalg.norm(map_delta) / (np.linalg.norm(conductivity_values) + 1e-12)
            ),
        }

        if metadata:
            output.setdefault("metadata", {}).update(metadata)

        output.setdefault("residual_history", None)
        output.setdefault("sigma_change_history", None)

        return output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
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
            float(np.std(data_vector) * self.config.noise_rel),
            self.config.noise_floor,
        )
        if not np.isfinite(noise_sigma) or noise_sigma <= 0:
            noise_sigma = self.config.noise_floor
        return noise_sigma

    def _build_coarse_hierarchy(self) -> List[Tuple[int, List[np.ndarray]]]:
        sizes: List[int] = []
        if self.config.coarse_levels:
            sizes.extend(int(s) for s in self.config.coarse_levels if s and s > 1)
        elif self.config.coarse_group_size and self.config.coarse_group_size > 1:
            sizes.append(int(self.config.coarse_group_size))

        hierarchy: List[Tuple[int, List[np.ndarray]]] = []
        for size in sorted(set(sizes), reverse=True):
            if size >= self.n_elements:
                continue
            if size not in self._coarse_levels_cache:
                groups: List[np.ndarray] = []
                for start in range(0, self.n_elements, size):
                    stop = min(start + size, self.n_elements)
                    groups.append(np.arange(start, stop, dtype=int))
                self._coarse_levels_cache[size] = groups
            hierarchy.append((size, self._coarse_levels_cache[size]))
        return hierarchy

    def _solve_sparse_map(
        self,
        jacobian: np.ndarray,
        data_vector: np.ndarray,
        noise_sigma: float,
        prior_scale: float,
    ) -> np.ndarray:
        hierarchy = self._build_coarse_hierarchy()
        coarse_init = None
        for size, groups in hierarchy:
            coarse_init = self._coarse_initialization(
                jacobian,
                data_vector,
                noise_sigma,
                prior_scale,
                groups,
                size,
                coarse_init,
            )

        linear_matrix = jacobian
        target_dim = self.n_elements

        basis = None
        U_k = None
        s_k = None

        if self.config.subspace_rank:
            desired_rank = int(self.config.subspace_rank)
            max_rank = min(jacobian.shape)
            if desired_rank < max_rank:
                if (
                    self._cached_basis is None
                    or self._cached_reduced_matrix is None
                    or self._cached_U is None
                    or self._cached_singular is None
                ):
                    basis, reduced, U_k, s_k = self._compute_projection(linear_matrix, desired_rank)
                    self._cached_basis = basis
                    self._cached_reduced_matrix = reduced
                    self._cached_U = U_k
                    self._cached_singular = s_k
                else:
                    basis = self._cached_basis
                    reduced = self._cached_reduced_matrix
                    U_k = self._cached_U
                    s_k = self._cached_singular
                linear_matrix = reduced
                target_dim = linear_matrix.shape[1]

        model = LinearModel(linear_matrix)
        x = SmoothedLaplace(
            location=np.zeros(target_dim, dtype=float),
            scale=prior_scale,
            beta=self.config.smoothing_beta,
        )
        y = Gaussian(model @ x, noise_sigma)

        problem = BayesianProblem(y, x).set_data(y=data_vector)

        warm_start = None
        if basis is not None and coarse_init is not None:
            warm_start = basis.T @ coarse_init
        elif basis is None and coarse_init is not None:
            warm_start = coarse_init

        if warm_start is None and self.config.use_linear_warm_start and basis is not None and U_k is not None and s_k is not None:
            numerator = U_k.T @ data_vector
            warm_start = np.zeros_like(numerator)
            mask = s_k > 1e-12
            warm_start[mask] = numerator[mask] / s_k[mask]
        elif warm_start is None and self.config.use_linear_warm_start and basis is None and not hierarchy:
            U, s, Vt = np.linalg.svd(linear_matrix, full_matrices=False)
            warm_start = Vt.T @ (U.T @ data_vector)
            mask = s > 1e-12
            warm_start[mask] /= s[mask]
            warm_start[~mask] = 0.0

        solver_type = self.config.solver.lower()
        if hierarchy and solver_type in {"fista", "irls"}:
            solver_type = "map"
        if solver_type == "map":
            map_numpy = self._solve_with_cuqi_map(problem, warm_start)
        elif solver_type == "fista":
            map_numpy = self._solve_fista(
                linear_matrix,
                data_vector,
                noise_sigma,
                prior_scale,
                warm_start,
            )
        elif solver_type == "irls":
            map_numpy = self._solve_irls(
                linear_matrix,
                data_vector,
                noise_sigma,
                prior_scale,
                warm_start,
            )
        else:
            raise ValueError(f"Unknown solver type: {self.config.solver}")

        solution_param = basis @ map_numpy if basis is not None else map_numpy
        solution_param = self._multilevel_correction(
            jacobian,
            data_vector,
            noise_sigma,
            prior_scale,
            solution_param,
            hierarchy,
        )
        solution_param = self._block_refinement(
            jacobian,
            data_vector,
            noise_sigma,
            prior_scale,
            solution_param,
        )
        return solution_param

    def _coarse_initialization(
        self,
        jacobian: np.ndarray,
        data_vector: np.ndarray,
        noise_sigma: float,
        prior_scale: float,
        groups: List[np.ndarray],
        group_size: int,
        initial_guess: Optional[np.ndarray],
    ) -> np.ndarray:
        linear_matrix = self._get_coarse_matrix(jacobian, groups, group_size)
        model = LinearModel(linear_matrix)
        x = SmoothedLaplace(
            location=np.zeros(linear_matrix.shape[1], dtype=float),
            scale=prior_scale,
            beta=self.config.smoothing_beta,
        )
        y = Gaussian(model @ x, noise_sigma)
        problem = BayesianProblem(y, x).set_data(y=data_vector)
        coarse_warm = None
        if initial_guess is not None:
            coarse_warm = np.array([initial_guess[idx].mean() for idx in groups], dtype=float)
        coarse_estimate = problem.MAP(disp=self.verbose, x0=coarse_warm)
        coarse_vec = np.asarray(coarse_estimate.to_numpy(), dtype=float)

        fine = np.zeros(self.n_elements, dtype=float)
        for value, idx in zip(coarse_vec, groups):
            fine[idx] = value
        return fine

    def _get_coarse_matrix(
        self,
        jacobian: np.ndarray,
        groups: List[np.ndarray],
        group_size: int,
    ) -> np.ndarray:
        if group_size not in self._cached_coarse_matrices:
            coarse_columns = [jacobian[:, idx].sum(axis=1) for idx in groups]
            self._cached_coarse_matrices[group_size] = np.column_stack(coarse_columns)
        return self._cached_coarse_matrices[group_size]

    def _compute_projection(
        self,
        jacobian: np.ndarray,
        rank: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        U, s, Vt = np.linalg.svd(jacobian, full_matrices=False)

        k = min(rank, len(s))
        U_k = U[:, :k]
        s_k = s[:k]
        V_k = Vt[:k, :].T  # (n_elements x k)

        reduced_matrix = U_k * s_k[np.newaxis, :]
        return V_k, reduced_matrix, U_k, s_k

    def _estimate_lipschitz_constant(self, matrix: np.ndarray, iters: int = 12) -> float:
        if matrix.size == 0:
            return 1e-12
        rng = np.random.default_rng(0)
        vec = rng.standard_normal(matrix.shape[1])
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            vec = np.ones(matrix.shape[1], dtype=float)
            norm = np.linalg.norm(vec)
        vec /= norm

        for _ in range(max(iters, 1)):
            z = matrix.T @ (matrix @ vec)
            norm = np.linalg.norm(z)
            if norm < 1e-12:
                return 1e-12
            vec = z / norm

        z = matrix.T @ (matrix @ vec)
        lipschitz = float(np.dot(vec, z))
        if not np.isfinite(lipschitz) or lipschitz <= 0:
            lipschitz = 1e-12
        return lipschitz + 1e-12

    def _solve_with_cuqi_map(self, problem, warm_start: Optional[np.ndarray]) -> np.ndarray:
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
        warm_start: Optional[np.ndarray],
    ) -> np.ndarray:
        A = linear_matrix / max(noise_sigma, 1e-9)
        b = data_vector / max(noise_sigma, 1e-9)
        lambda_reg = 1.0 / max(prior_scale, 1e-12)

        n = A.shape[1]
        x = warm_start.copy() if warm_start is not None else np.zeros(n, dtype=float)
        y = x.copy()
        t = 1.0
        L = self._estimate_lipschitz_constant(A)

        use_gpu = self.config.use_gpu
        device = None
        torch = None
        if use_gpu:
            try:
                import torch  # type: ignore

                if torch.cuda.is_available():
                    device = torch.device("cuda")
                else:
                    device = None
                    use_gpu = False
            except ImportError:  # pragma: no cover
                use_gpu = False

        if use_gpu and device is not None:
            gpu_dtype = str(self.config.gpu_dtype).lower()
            if gpu_dtype == "float64":
                dtype = torch.float64
            elif gpu_dtype in {"float16", "half"}:
                dtype = torch.float16
            else:
                dtype = torch.float32
            A_t = torch.tensor(A, device=device, dtype=dtype, copy=False)
            b_t = torch.tensor(b, device=device, dtype=dtype, copy=False)
            x_t = torch.tensor(x, device=device, dtype=dtype, copy=False)
            y_t = x_t.clone()
            t_scalar = torch.tensor(1.0, device=device, dtype=dtype)
            L_t = torch.tensor(L, device=device, dtype=dtype)
            lam_over_L = torch.tensor(lambda_reg, device=device, dtype=dtype) / L_t

            for _ in range(self.config.linear_max_iterations):
                grad = torch.matmul(A_t.T, torch.matmul(A_t, y_t) - b_t)
                z = y_t - grad / L_t
                x_new = torch.sign(z) * torch.clamp(torch.abs(z) - lam_over_L, min=0.0)

                if torch.norm(x_new - x_t) <= self.config.linear_tolerance * (torch.norm(x_t) + 1e-12):
                    x_t = x_new
                    break

                t_new = (1.0 + torch.sqrt(1.0 + 4.0 * t_scalar * t_scalar)) / 2.0
                y_t = x_new + ((t_scalar - 1.0) / t_new) * (x_new - x_t)
                x_t, t_scalar = x_new, t_new

            return x_t.detach().cpu().double().numpy()

        for _ in range(self.config.linear_max_iterations):
            grad = A.T @ (A @ y - b)
            z = y - grad / L
            x_new = np.sign(z) * np.maximum(np.abs(z) - lambda_reg / L, 0.0)

            if np.linalg.norm(x_new - x) <= self.config.linear_tolerance * (np.linalg.norm(x) + 1e-12):
                x = x_new
                break

            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
            y = x_new + ((t - 1.0) / t_new) * (x_new - x)
            x, t = x_new, t_new

        return x

    def _solve_irls(
        self,
        linear_matrix: np.ndarray,
        data_vector: np.ndarray,
        noise_sigma: float,
        prior_scale: float,
        warm_start: Optional[np.ndarray],
    ) -> np.ndarray:
        A = linear_matrix / max(noise_sigma, 1e-9)
        b = data_vector / max(noise_sigma, 1e-9)
        lambda_reg = 1.0 / max(prior_scale, 1e-12)

        n = A.shape[1]
        x = warm_start.copy() if warm_start is not None else np.zeros(n, dtype=float)

        use_gpu = self.config.use_gpu
        device = None
        torch = None
        if use_gpu:
            try:
                import torch  # type: ignore

                if torch.cuda.is_available():
                    device = torch.device("cuda")
                else:
                    device = None
                    use_gpu = False
            except ImportError:  # pragma: no cover
                use_gpu = False

        if use_gpu and device is not None:
            gpu_dtype = str(self.config.gpu_dtype).lower()
            if gpu_dtype == "float64":
                dtype = torch.float64
            elif gpu_dtype in {"float16", "half"}:
                dtype = torch.float16
            else:
                dtype = torch.float32
            A_t = torch.tensor(A, device=device, dtype=dtype, copy=False)
            b_t = torch.tensor(b, device=device, dtype=dtype, copy=False)
            x_t = torch.tensor(x, device=device, dtype=dtype, copy=False)

            AtA = torch.matmul(A_t.T, A_t)
            Atb = torch.matmul(A_t.T, b_t)

            for _ in range(self.config.linear_max_iterations):
                weights = torch.rsqrt(x_t * x_t + self.config.smoothing_beta)
                M = AtA.clone()
                M.diagonal().add_(lambda_reg * weights)
                rhs = Atb
                try:
                    x_new = torch.linalg.solve(M, rhs)
                except RuntimeError:
                    x_new = torch.linalg.lstsq(M, rhs).solution

                if torch.norm(x_new - x_t) <= self.config.linear_tolerance * (torch.norm(x_t) + 1e-12):
                    x_t = x_new
                    break
                x_t = x_new

            return x_t.detach().cpu().double().numpy()

        for _ in range(self.config.linear_max_iterations):
            weights = 1.0 / np.sqrt(x * x + self.config.smoothing_beta)
            M = A.T @ A + lambda_reg * np.diag(weights)
            rhs = A.T @ b
            try:
                x_new = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                x_new = np.linalg.lstsq(M, rhs, rcond=None)[0]

            if np.linalg.norm(x_new - x) <= self.config.linear_tolerance * (np.linalg.norm(x) + 1e-12):
                x = x_new
                break
            x = x_new

        return x

    def _multilevel_correction(
        self,
        jacobian: np.ndarray,
        data_vector: np.ndarray,
        noise_sigma: float,
        prior_scale: float,
        solution: np.ndarray,
        hierarchy: List[Tuple[int, List[np.ndarray]]],
    ) -> np.ndarray:
        iterations = max(int(self.config.coarse_iterations), 0)
        if iterations == 0 or not hierarchy:
            return solution

        lambda_reg = 1.0 / max(prior_scale, 1e-12)
        inv_noise_var = 1.0 / max(noise_sigma * noise_sigma, 1e-18)
        solution = solution.copy()
        relaxation = max(float(self.config.coarse_relaxation), 0.0)
        tol = max(float(self.config.refinement_gradient_tol), 0.0)

        for _ in range(iterations):
            residual = jacobian @ solution - data_vector
            grad = inv_noise_var * (jacobian.T @ residual) + lambda_reg * solution
            max_update = 0.0

            for size, groups in hierarchy:
                if not groups:
                    continue
                A_c = self._get_coarse_matrix(jacobian, groups, size)
                if A_c.size == 0:
                    continue
                group_sizes = np.array([len(idx) for idx in groups], dtype=float)
                coarse_grad = np.array([grad[idx].sum() for idx in groups], dtype=float)
                if coarse_grad.size == 0:
                    continue
                if tol > 0.0 and np.linalg.norm(coarse_grad, ord=np.inf) <= tol:
                    continue

                H = inv_noise_var * (A_c.T @ A_c) + lambda_reg * np.diag(group_sizes)
                rhs = -coarse_grad

                try:
                    delta = np.linalg.solve(H, rhs)
                except np.linalg.LinAlgError:
                    delta = np.linalg.lstsq(H, rhs, rcond=None)[0]

                if relaxation > 0.0 and relaxation != 1.0:
                    delta *= relaxation

                if delta.size == 0:
                    continue

                max_update = max(max_update, float(np.max(np.abs(delta))))

                for g_idx, idx in enumerate(groups):
                    solution[idx] += delta[g_idx]

                residual += A_c @ delta
                grad = inv_noise_var * (jacobian.T @ residual) + lambda_reg * solution

            if tol > 0.0 and max_update <= tol:
                break

        return solution

    def _block_refinement(
        self,
        jacobian: np.ndarray,
        data_vector: np.ndarray,
        noise_sigma: float,
        prior_scale: float,
        solution: np.ndarray,
    ) -> np.ndarray:
        iterations = max(int(self.config.block_iterations), 0)
        block_size = self.config.block_size
        if iterations == 0 or not block_size or block_size <= 0:
            return solution

        n = solution.size
        block_size = min(block_size, n)
        lambda_reg = 1.0 / max(prior_scale, 1e-12)
        inv_noise_var = 1.0 / max(noise_sigma * noise_sigma, 1e-18)
        tol = max(float(self.config.refinement_gradient_tol), 0.0)
        solution = solution.copy()
        residual = jacobian @ solution - data_vector

        for _ in range(iterations):
            updated = False
            max_passes = max((n + block_size - 1) // block_size, 1)
            passes = 0

            while passes < max_passes:
                passes += 1
                grad = inv_noise_var * (jacobian.T @ residual) + lambda_reg * solution
                blocks: List[Tuple[float, int, int]] = []

                for start in range(0, n, block_size):
                    stop = min(start + block_size, n)
                    grad_block = grad[start:stop]
                    if grad_block.size == 0:
                        continue
                    score = float(np.linalg.norm(grad_block, ord=2))
                    blocks.append((score, start, stop))

                if not blocks:
                    break

                blocks.sort(key=lambda item: item[0], reverse=True)

                block_used = False
                for score, start, stop in blocks:
                    if tol > 0.0 and score <= tol:
                        continue

                    idx = slice(start, stop)
                    J_block = jacobian[:, idx]
                    if J_block.size == 0:
                        continue

                    M = inv_noise_var * (J_block.T @ J_block) + lambda_reg * np.eye(stop - start)
                    rhs = -inv_noise_var * (J_block.T @ residual) - lambda_reg * solution[idx]
                    try:
                        delta = np.linalg.solve(M, rhs)
                    except np.linalg.LinAlgError:
                        delta = np.linalg.lstsq(M, rhs, rcond=None)[0]

                    if delta.size == 0:
                        continue
                    if tol > 0.0 and np.linalg.norm(delta, ord=2) <= tol:
                        continue

                    solution[idx] += delta
                    residual += J_block @ delta
                    updated = True
                    block_used = True
                    break

                if not block_used:
                    break

            if not updated:
                break

        return solution
