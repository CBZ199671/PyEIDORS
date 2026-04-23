"""Sparse Bayesian MAP pipeline helpers."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ...utils.numeric_ops import safe_dot


def _resolve_projection(reconstructor, jacobian: np.ndarray):
    linear_matrix = jacobian
    target_dim = reconstructor.n_elements
    basis = None
    U_k = None
    s_k = None

    if reconstructor.config.subspace_rank:
        desired_rank = int(reconstructor.config.subspace_rank)
        max_rank = min(jacobian.shape)
        if desired_rank < max_rank:
            if (
                reconstructor._cached_basis is None
                or reconstructor._cached_reduced_matrix is None
                or reconstructor._cached_U is None
                or reconstructor._cached_singular is None
            ):
                basis, reduced, U_k, s_k = reconstructor._compute_projection(
                    linear_matrix, desired_rank
                )
                reconstructor._cached_basis = basis
                reconstructor._cached_reduced_matrix = reduced
                reconstructor._cached_U = U_k
                reconstructor._cached_singular = s_k
            else:
                basis = reconstructor._cached_basis
                reduced = reconstructor._cached_reduced_matrix
                U_k = reconstructor._cached_U
                s_k = reconstructor._cached_singular
            linear_matrix = reduced
            target_dim = linear_matrix.shape[1]

    return linear_matrix, target_dim, basis, U_k, s_k


def _coarse_warm_start(
    basis: Optional[np.ndarray],
    coarse_init: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if coarse_init is None:
        return None
    if basis is None:
        return coarse_init
    return safe_dot(basis.T, coarse_init, "solve_sparse_map.warm_start_subspace")


def _linear_warm_start_subspace(
    U_k: np.ndarray,
    s_k: np.ndarray,
    data_vector: np.ndarray,
) -> np.ndarray:
    numerator = safe_dot(
        U_k.T, data_vector, "solve_sparse_map.linear_warm_start_numerator"
    )
    warm_start = np.zeros_like(numerator)
    mask = s_k > 1e-12
    warm_start[mask] = numerator[mask] / s_k[mask]
    return warm_start


def _linear_warm_start_fullspace(
    linear_matrix: np.ndarray,
    data_vector: np.ndarray,
) -> np.ndarray:
    U, s, Vt = np.linalg.svd(linear_matrix, full_matrices=False)
    coeff = safe_dot(U.T, data_vector, "solve_sparse_map.fullspace_coeff")
    mask = s > 1e-12
    coeff[mask] /= s[mask]
    coeff[~mask] = 0.0
    return safe_dot(Vt.T, coeff, "solve_sparse_map.fullspace_warm_start")


def _resolve_warm_start(
    *,
    reconstructor,
    basis: Optional[np.ndarray],
    coarse_init: Optional[np.ndarray],
    data_vector: np.ndarray,
    hierarchy: List[Tuple[int, List[np.ndarray]]],
    linear_matrix: np.ndarray,
    U_k: Optional[np.ndarray],
    s_k: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    warm_start = _coarse_warm_start(basis, coarse_init)
    if warm_start is not None or not reconstructor.config.use_linear_warm_start:
        return warm_start

    if basis is not None and U_k is not None and s_k is not None:
        return _linear_warm_start_subspace(U_k, s_k, data_vector)

    if basis is None and not hierarchy:
        return _linear_warm_start_fullspace(linear_matrix, data_vector)

    return None


def _resolve_solver_type(config, hierarchy: List[Tuple[int, List[np.ndarray]]]) -> str:
    solver_type = str(config.solver).lower()
    if hierarchy and solver_type in {"fista", "irls"}:
        return "map"
    return solver_type


def solve_sparse_map(
    reconstructor,
    jacobian: np.ndarray,
    data_vector: np.ndarray,
    noise_sigma: float,
    prior_scale: float,
) -> np.ndarray:
    """Solve sparse MAP in coarse/subspace/full spaces."""
    hierarchy = reconstructor._build_coarse_hierarchy()
    coarse_init = None
    for size, groups in hierarchy:
        coarse_init = coarse_initialization(
            reconstructor,
            jacobian,
            data_vector,
            noise_sigma,
            prior_scale,
            groups,
            size,
            coarse_init,
        )

    linear_matrix, target_dim, basis, U_k, s_k = _resolve_projection(
        reconstructor, jacobian
    )

    model = reconstructor._linear_model(linear_matrix)
    x = reconstructor._sparse_prior(target_dim, prior_scale)
    y = reconstructor._gaussian_likelihood(model @ x, noise_sigma)
    problem = reconstructor._bayesian_problem(y, x).set_data(y=data_vector)

    warm_start = _resolve_warm_start(
        reconstructor=reconstructor,
        basis=basis,
        coarse_init=coarse_init,
        data_vector=data_vector,
        hierarchy=hierarchy,
        linear_matrix=linear_matrix,
        U_k=U_k,
        s_k=s_k,
    )

    solver_type = _resolve_solver_type(reconstructor.config, hierarchy)
    if solver_type == "map":
        map_numpy = reconstructor._solve_with_cuqi_map(problem, warm_start)
    elif solver_type == "fista":
        map_numpy = reconstructor._solve_fista(
            linear_matrix,
            data_vector,
            noise_sigma,
            prior_scale,
            warm_start,
        )
    elif solver_type == "irls":
        map_numpy = reconstructor._solve_irls(
            linear_matrix,
            data_vector,
            noise_sigma,
            prior_scale,
            warm_start,
        )
    else:
        raise ValueError(f"Unknown solver type: {reconstructor.config.solver}")

    solution_param = (
        safe_dot(basis, map_numpy, "solve_sparse_map.solution_projection")
        if basis is not None
        else map_numpy
    )
    solution_param = multilevel_correction(
        reconstructor,
        jacobian,
        data_vector,
        noise_sigma,
        prior_scale,
        solution_param,
        hierarchy,
    )
    solution_param = block_refinement(
        reconstructor,
        jacobian,
        data_vector,
        noise_sigma,
        prior_scale,
        solution_param,
    )
    return solution_param


def coarse_initialization(
    reconstructor,
    jacobian: np.ndarray,
    data_vector: np.ndarray,
    noise_sigma: float,
    prior_scale: float,
    groups: List[np.ndarray],
    group_size: int,
    initial_guess: Optional[np.ndarray],
) -> np.ndarray:
    """Estimate coarse-space warm start using MAP."""
    linear_matrix = reconstructor._get_coarse_matrix(jacobian, groups, group_size)
    model = reconstructor._linear_model(linear_matrix)
    x = reconstructor._sparse_prior(linear_matrix.shape[1], prior_scale)
    y = reconstructor._gaussian_likelihood(model @ x, noise_sigma)
    problem = reconstructor._bayesian_problem(y, x).set_data(y=data_vector)

    coarse_warm = None
    if initial_guess is not None:
        coarse_warm = np.array(
            [initial_guess[idx].mean() for idx in groups], dtype=float
        )

    coarse_estimate = problem.MAP(disp=reconstructor.verbose, x0=coarse_warm)
    coarse_vec = np.asarray(coarse_estimate.to_numpy(), dtype=float)

    fine = np.zeros(reconstructor.n_elements, dtype=float)
    for value, idx in zip(coarse_vec, groups):
        fine[idx] = value
    return fine


def multilevel_correction(
    reconstructor,
    jacobian: np.ndarray,
    data_vector: np.ndarray,
    noise_sigma: float,
    prior_scale: float,
    solution: np.ndarray,
    hierarchy: List[Tuple[int, List[np.ndarray]]],
) -> np.ndarray:
    """Apply optional coarse-level corrections."""
    iterations = max(int(reconstructor.config.coarse_iterations), 0)
    if iterations == 0 or not hierarchy:
        return solution

    lambda_reg = 1.0 / max(prior_scale, 1e-12)
    inv_noise_var = 1.0 / max(noise_sigma * noise_sigma, 1e-18)
    result = solution.copy()
    relaxation = max(float(reconstructor.config.coarse_relaxation), 0.0)
    tol = max(float(reconstructor.config.refinement_gradient_tol), 0.0)

    for _ in range(iterations):
        residual = (
            safe_dot(jacobian, result, "multilevel_correction.residual") - data_vector
        )
        grad = (
            inv_noise_var * safe_dot(jacobian.T, residual, "multilevel_correction.grad")
            + lambda_reg * result
        )
        max_update = 0.0

        for size, groups in hierarchy:
            if not groups:
                continue
            A_c = reconstructor._get_coarse_matrix(jacobian, groups, size)
            if A_c.size == 0:
                continue

            group_sizes = np.array([len(idx) for idx in groups], dtype=float)
            coarse_grad = np.array([grad[idx].sum() for idx in groups], dtype=float)
            if tol > 0.0 and np.linalg.norm(coarse_grad, ord=np.inf) <= tol:
                continue

            hessian = safe_dot(A_c.T, A_c, "multilevel_correction.hessian")
            H = inv_noise_var * hessian + lambda_reg * np.diag(group_sizes)
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
                result[idx] += delta[g_idx]

            residual += safe_dot(A_c, delta, "multilevel_correction.residual_update")
            grad = (
                inv_noise_var
                * safe_dot(jacobian.T, residual, "multilevel_correction.grad_update")
                + lambda_reg * result
            )

        if tol > 0.0 and max_update <= tol:
            break

    return result


def block_refinement(
    reconstructor,
    jacobian: np.ndarray,
    data_vector: np.ndarray,
    noise_sigma: float,
    prior_scale: float,
    solution: np.ndarray,
) -> np.ndarray:
    """Apply optional block-coordinate refinement over active gradients."""
    iterations = max(int(reconstructor.config.block_iterations), 0)
    block_size = reconstructor.config.block_size
    if iterations == 0 or not block_size or block_size <= 0:
        return solution

    n = solution.size
    if n == 0:
        return solution
    block_size = min(block_size, n)
    lambda_reg = 1.0 / max(prior_scale, 1e-12)
    inv_noise_var = 1.0 / max(noise_sigma * noise_sigma, 1e-18)
    tol = max(float(reconstructor.config.refinement_gradient_tol), 0.0)
    result = solution.copy()
    residual = safe_dot(jacobian, result, "block_refinement.residual") - data_vector

    for _ in range(iterations):
        updated = False
        max_passes = max((n + block_size - 1) // block_size, 1)
        passes = 0

        while passes < max_passes:
            passes += 1
            grad = (
                inv_noise_var * safe_dot(jacobian.T, residual, "block_refinement.grad")
                + lambda_reg * result
            )
            blocks: List[Tuple[float, int, int]] = []

            for start in range(0, n, block_size):
                stop = min(start + block_size, n)
                grad_block = grad[start:stop]
                blocks.append((float(np.linalg.norm(grad_block, ord=2)), start, stop))

            blocks.sort(key=lambda item: item[0], reverse=True)

            block_used = False
            for score, start, stop in blocks:
                if tol > 0.0 and score <= tol:
                    continue

                idx = slice(start, stop)
                J_block = jacobian[:, idx]
                if J_block.size == 0:
                    continue

                block_hessian = safe_dot(
                    J_block.T, J_block, "block_refinement.block_hessian"
                )
                M = inv_noise_var * block_hessian + lambda_reg * np.eye(stop - start)
                block_residual = safe_dot(
                    J_block.T, residual, "block_refinement.block_residual"
                )
                rhs = -inv_noise_var * block_residual - lambda_reg * result[idx]
                try:
                    delta = np.linalg.solve(M, rhs)
                except np.linalg.LinAlgError:
                    delta = np.linalg.lstsq(M, rhs, rcond=None)[0]

                if delta.size == 0:
                    continue
                if tol > 0.0 and np.linalg.norm(delta, ord=2) <= tol:
                    continue

                result[idx] += delta
                residual += safe_dot(J_block, delta, "block_refinement.residual_update")
                updated = True
                block_used = True
                break

            if not block_used:
                break

        if not updated:
            break

    return result
