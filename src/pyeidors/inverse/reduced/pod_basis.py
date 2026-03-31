"""POD basis builders for reduced-order GN."""

from __future__ import annotations

import numpy as np


def _normalize_snapshot_matrix(snapshots: np.ndarray) -> np.ndarray:
    mat = np.asarray(snapshots, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    if mat.ndim != 2:
        raise ValueError("snapshots must be a 2D array")
    if mat.size == 0:
        return np.zeros((mat.shape[0], 0), dtype=np.float64)
    return np.ascontiguousarray(mat, dtype=np.float64)


def _rank_from_energy(singular_values: np.ndarray, energy: float, max_rank: int) -> int:
    if singular_values.size == 0:
        return 0
    total = float(np.sum(singular_values * singular_values))
    if total <= 0.0:
        return 0
    target = min(max(float(energy), 0.0), 1.0) * total
    accum = 0.0
    for idx, s_val in enumerate(singular_values, start=1):
        accum += float(s_val * s_val)
        if accum >= target:
            return int(min(idx, max_rank))
    return int(max_rank)


def compute_pod_basis(
    snapshots: np.ndarray,
    *,
    rank: int | None = None,
    energy: float = 0.995,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute orthonormal POD basis from snapshot matrix."""
    mat = _normalize_snapshot_matrix(snapshots)
    n_param = int(mat.shape[0])
    if mat.size == 0:
        return np.zeros((n_param, 0), dtype=np.float64)

    with np.errstate(all="ignore"):
        u_mat, singular_values, _ = np.linalg.svd(mat, full_matrices=False)
    if singular_values.size == 0:
        return np.zeros((n_param, 0), dtype=np.float64)

    stable = singular_values > float(max(eps, 0.0))
    if not np.any(stable):
        return np.zeros((n_param, 0), dtype=np.float64)

    max_rank = int(np.count_nonzero(stable))
    if rank is None or int(rank) <= 0:
        chosen_rank = _rank_from_energy(singular_values[stable], float(energy), max_rank)
    else:
        chosen_rank = int(min(int(rank), max_rank))

    if chosen_rank <= 0:
        return np.zeros((n_param, 0), dtype=np.float64)
    return np.ascontiguousarray(u_mat[:, :chosen_rank], dtype=np.float64)


def merge_orthonormal_bases(
    *bases: np.ndarray | None,
    rank_cap: int | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Merge candidate bases and return an orthonormal matrix."""
    valid_blocks: list[np.ndarray] = []
    n_param: int | None = None
    for basis in bases:
        if basis is None:
            continue
        arr = np.asarray(basis, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2 or arr.size == 0:
            continue
        if n_param is None:
            n_param = int(arr.shape[0])
        if int(arr.shape[0]) != int(n_param):
            continue
        valid_blocks.append(np.ascontiguousarray(arr, dtype=np.float64))

    if n_param is None:
        return np.zeros((0, 0), dtype=np.float64)
    if not valid_blocks:
        return np.zeros((n_param, 0), dtype=np.float64)

    merged = np.column_stack(valid_blocks)
    with np.errstate(all="ignore"):
        q_mat, r_mat = np.linalg.qr(merged, mode="reduced")
    diag = np.abs(np.diag(r_mat)) if r_mat.ndim == 2 else np.array([], dtype=np.float64)
    if diag.size:
        keep = diag > float(max(eps, 0.0))
        if np.any(keep):
            q_mat = q_mat[:, keep]
        else:
            q_mat = np.zeros((n_param, 0), dtype=np.float64)

    if rank_cap is not None and int(rank_cap) > 0 and q_mat.shape[1] > int(rank_cap):
        q_mat = q_mat[:, : int(rank_cap)]
    return np.ascontiguousarray(q_mat, dtype=np.float64)
