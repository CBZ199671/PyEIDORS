"""Low-rank Jacobian subspace extraction utilities."""

from __future__ import annotations

import numpy as np


def _rank_from_energy(singular_values: np.ndarray, energy: float, max_rank: int) -> int:
    if singular_values.size == 0:
        return 0
    total = float(np.sum(singular_values * singular_values))
    if total <= 0:
        return 0
    target = min(max(float(energy), 0.0), 1.0) * total
    accum = 0.0
    for idx, s_val in enumerate(singular_values, start=1):
        accum += float(s_val * s_val)
        if accum >= target:
            return int(min(idx, max_rank))
    return int(max_rank)


def _randomized_right_svd(jacobian: np.ndarray, q_rank: int) -> tuple[np.ndarray, np.ndarray]:
    m_rows, n_cols = jacobian.shape
    oversample = min(8, max(2, q_rank // 2))
    sample_rank = int(min(n_cols, max(1, q_rank + oversample)))
    rng = np.random.default_rng(0)
    omega = rng.standard_normal((n_cols, sample_rank))
    with np.errstate(all="ignore"):
        y_mat = jacobian @ omega
        q_mat, _ = np.linalg.qr(y_mat, mode="reduced")
        b_mat = q_mat.T @ jacobian
        _u_small, s_val, vt_mat = np.linalg.svd(b_mat, full_matrices=False)
    return s_val, vt_mat


def build_lowrank_subspace(
    jacobian: np.ndarray,
    *,
    rank: int = 16,
    energy: float = 0.995,
    method: str = "tsvd",
) -> tuple[np.ndarray, np.ndarray]:
    """Build right singular subspace basis from Jacobian matrix.

    Returns ``(basis, singular_values)`` where ``basis`` has shape ``(n_param, r)``.
    """
    j_mat = np.asarray(jacobian, dtype=np.float64)
    if j_mat.ndim != 2:
        raise ValueError("jacobian must be 2D")
    if j_mat.size == 0:
        return np.zeros((j_mat.shape[-1], 0), dtype=np.float64), np.zeros(0, dtype=np.float64)

    method_norm = str(method).strip().lower()
    cap_rank = int(max(1, rank))

    if method_norm == "randomized":
        singular_values, vt_mat = _randomized_right_svd(j_mat, cap_rank)
    else:
        with np.errstate(all="ignore"):
            _u_mat, singular_values, vt_mat = np.linalg.svd(j_mat, full_matrices=False)

    max_rank = int(min(cap_rank, vt_mat.shape[0]))
    if max_rank <= 0:
        return np.zeros((j_mat.shape[1], 0), dtype=np.float64), np.zeros(0, dtype=np.float64)

    energy_rank = _rank_from_energy(singular_values, energy, max_rank)
    keep = int(min(max_rank, max(1, energy_rank)))
    basis = np.asarray(vt_mat[:keep, :].T, dtype=np.float64)
    return basis, np.asarray(singular_values[:keep], dtype=np.float64)
