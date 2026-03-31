"""Snapshot bank utilities for reduced-order GN bases."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SnapshotBank:
    """Store recent parameter-space snapshots for POD basis construction."""

    max_snapshots: int = 24
    normalize: bool = True
    _snapshots: list[np.ndarray] = field(default_factory=list)

    def add(self, snapshot: np.ndarray) -> None:
        vec = np.asarray(snapshot, dtype=np.float64).reshape(-1)
        if vec.size == 0 or not np.isfinite(vec).all():
            return
        if self.normalize:
            norm = float(np.linalg.norm(vec))
            if norm > 1e-12:
                vec = vec / norm
        self._snapshots.append(vec)
        if len(self._snapshots) > int(max(1, self.max_snapshots)):
            self._snapshots = self._snapshots[-int(max(1, self.max_snapshots)) :]

    def matrix(self) -> np.ndarray:
        if not self._snapshots:
            return np.zeros((0, 0), dtype=np.float64)
        dim = int(self._snapshots[-1].shape[0])
        cols = [v for v in self._snapshots if int(v.shape[0]) == dim]
        if not cols:
            return np.zeros((0, 0), dtype=np.float64)
        return np.ascontiguousarray(np.column_stack(cols), dtype=np.float64)

    def snapshot_hash(self) -> str:
        mat = self.matrix()
        if mat.size == 0:
            return "empty"
        return hashlib.sha256(mat.tobytes()).hexdigest()


def _as_matrix(value: np.ndarray | None, *, n_param: int) -> np.ndarray:
    if value is None:
        return np.zeros((n_param, 0), dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        return np.zeros((n_param, 0), dtype=np.float64)
    if arr.shape[0] != n_param:
        return np.zeros((n_param, 0), dtype=np.float64)
    if arr.size == 0:
        return np.zeros((n_param, 0), dtype=np.float64)
    return np.ascontiguousarray(arr, dtype=np.float64)


def select_snapshot_matrix(
    source: str,
    *,
    n_param: int,
    bank_matrix: np.ndarray | None,
    synthetic_matrix: np.ndarray | None,
    cached_matrix: np.ndarray | None,
) -> np.ndarray:
    """Select snapshot matrix according to source policy."""
    src = str(source).strip().lower()
    if src not in {"cache", "synthetic", "hybrid"}:
        src = "hybrid"

    bank = _as_matrix(bank_matrix, n_param=n_param)
    synthetic = _as_matrix(synthetic_matrix, n_param=n_param)
    cached = _as_matrix(cached_matrix, n_param=n_param)

    blocks: list[np.ndarray] = []
    if src == "cache":
        if cached.size:
            blocks.append(cached)
        if bank.size:
            blocks.append(bank)
    elif src == "synthetic":
        if synthetic.size:
            blocks.append(synthetic)
        if bank.size:
            blocks.append(bank)
    else:
        if cached.size:
            blocks.append(cached)
        if synthetic.size:
            blocks.append(synthetic)
        if bank.size:
            blocks.append(bank)

    if not blocks:
        return np.zeros((n_param, 0), dtype=np.float64)

    stacked = np.ascontiguousarray(np.column_stack(blocks), dtype=np.float64)
    # de-duplicate nearly identical snapshots
    if stacked.shape[1] <= 1:
        return stacked

    keep_cols: list[np.ndarray] = []
    seen_hashes: set[str] = set()
    for col_idx in range(stacked.shape[1]):
        col = np.ascontiguousarray(stacked[:, col_idx], dtype=np.float64)
        h = hashlib.sha256(col.tobytes()).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        keep_cols.append(col)
    if not keep_cols:
        return np.zeros((n_param, 0), dtype=np.float64)
    return np.ascontiguousarray(np.column_stack(keep_cols), dtype=np.float64)
