"""Projection and coarse hierarchy helpers for sparse Bayesian solver."""

from __future__ import annotations

import numpy as np

from ...utils.numeric_ops import safe_dot


def _resolve_coarse_sizes(config) -> list[int]:
    sizes: list[int] = []
    if config.coarse_levels:
        sizes.extend(int(s) for s in config.coarse_levels if s and s > 1)
    elif config.coarse_group_size and config.coarse_group_size > 1:
        sizes.append(int(config.coarse_group_size))
    return sizes


def _build_groups(n_elements: int, size: int) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    for start in range(0, n_elements, size):
        stop = min(start + size, n_elements)
        groups.append(np.arange(start, stop, dtype=int))
    return groups


def _sum_group_columns(jacobian: np.ndarray, groups: list[np.ndarray]) -> list[np.ndarray]:
    return [jacobian[:, idx].sum(axis=1) for idx in groups]


def _init_power_vector(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    vec = rng.standard_normal(matrix.shape[1])
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        vec = np.ones(matrix.shape[1], dtype=float)
        norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec
    return vec / norm


def build_coarse_hierarchy(
    config,
    n_elements: int,
    cache: dict[int, list[np.ndarray]],
) -> list[tuple[int, list[np.ndarray]]]:
    """Build (group_size, groups) hierarchy from config and cache."""
    sizes = _resolve_coarse_sizes(config)

    hierarchy: list[tuple[int, list[np.ndarray]]] = []
    for size in sorted(set(sizes), reverse=True):
        if size >= n_elements:
            continue
        groups = cache.get(size)
        if groups is None:
            groups = _build_groups(n_elements, size)
            cache[size] = groups
        hierarchy.append((size, groups))
    return hierarchy


def get_coarse_matrix(
    jacobian: np.ndarray,
    groups: list[np.ndarray],
    group_size: int,
    cache: dict[int, np.ndarray],
) -> np.ndarray:
    """Get or build grouped coarse matrix."""
    if group_size not in cache:
        coarse_columns = _sum_group_columns(jacobian, groups)
        cache[group_size] = np.column_stack(coarse_columns)
    return cache[group_size]


def compute_projection(
    jacobian: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute SVD projection basis and reduced system."""
    U, s, Vt = np.linalg.svd(jacobian, full_matrices=False)
    k = min(rank, len(s))
    U_k = U[:, :k]
    s_k = s[:k]
    V_k = Vt[:k, :].T
    reduced_matrix = U_k * s_k[np.newaxis, :]
    return V_k, reduced_matrix, U_k, s_k


def estimate_lipschitz_constant(matrix: np.ndarray, iters: int = 12) -> float:
    """Power-iteration estimate of L = ||A^T A||."""
    if matrix.size == 0:
        return 1e-12
    rng = np.random.default_rng(0)
    vec = _init_power_vector(matrix, rng)

    for _ in range(max(iters, 1)):
        mat_vec = safe_dot(matrix, vec, "estimate_lipschitz_constant.mat_vec")
        z = safe_dot(matrix.T, mat_vec, "estimate_lipschitz_constant.ata_vec")
        norm = np.linalg.norm(z)
        if norm < 1e-12:
            return 1e-12
        vec = z / norm

    mat_vec = safe_dot(matrix, vec, "estimate_lipschitz_constant.final_mat_vec")
    z = safe_dot(matrix.T, mat_vec, "estimate_lipschitz_constant.final_ata_vec")
    lipschitz = float(np.dot(vec, z))
    if not np.isfinite(lipschitz) or lipschitz <= 0:
        lipschitz = 1e-12
    return lipschitz + 1e-12
