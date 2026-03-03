"""Projection and coarse hierarchy helpers for sparse Bayesian solver."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def build_coarse_hierarchy(config, n_elements: int, cache: Dict[int, List[np.ndarray]]) -> List[Tuple[int, List[np.ndarray]]]:
    """Build (group_size, groups) hierarchy from config and cache."""
    sizes: List[int] = []
    if config.coarse_levels:
        sizes.extend(int(s) for s in config.coarse_levels if s and s > 1)
    elif config.coarse_group_size and config.coarse_group_size > 1:
        sizes.append(int(config.coarse_group_size))

    hierarchy: List[Tuple[int, List[np.ndarray]]] = []
    for size in sorted(set(sizes), reverse=True):
        if size >= n_elements:
            continue
        if size not in cache:
            groups: List[np.ndarray] = []
            for start in range(0, n_elements, size):
                stop = min(start + size, n_elements)
                groups.append(np.arange(start, stop, dtype=int))
            cache[size] = groups
        hierarchy.append((size, cache[size]))
    return hierarchy


def get_coarse_matrix(
    jacobian: np.ndarray,
    groups: List[np.ndarray],
    group_size: int,
    cache: Dict[int, np.ndarray],
) -> np.ndarray:
    """Get or build grouped coarse matrix."""
    if group_size not in cache:
        coarse_columns = [jacobian[:, idx].sum(axis=1) for idx in groups]
        cache[group_size] = np.column_stack(coarse_columns)
    return cache[group_size]


def compute_projection(
    jacobian: np.ndarray,
    rank: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
