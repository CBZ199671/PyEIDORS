"""Snapshot bank utilities for reduced-order GN bases."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...cache.keys import hash_array_payload
from ...utils.numeric_ops import all_finite_values


def _stack_columns_direct(
    columns: list[np.ndarray] | tuple[np.ndarray, ...],
) -> np.ndarray:
    """Stack 1D columns into a C-order matrix by direct column assignment."""

    if not columns:
        return np.zeros((0, 0), dtype=np.float64)
    first = np.asarray(columns[0], dtype=np.float64).reshape(-1)
    out = np.empty((first.shape[0], len(columns)), dtype=np.float64)
    out[:, 0] = first
    for idx, column in enumerate(columns[1:], start=1):
        arr = np.asarray(column, dtype=np.float64).reshape(-1)
        if arr.shape[0] != out.shape[0]:
            raise ValueError(
                f"column {idx} length {arr.shape[0]} does not match {out.shape[0]}."
            )
        out[:, idx] = arr
    return out


def _unique_snapshot_blocks(blocks: list[np.ndarray], *, n_param: int) -> np.ndarray:
    total_cols = sum(int(block.shape[1]) for block in blocks)
    if total_cols <= 0:
        return np.zeros((n_param, 0), dtype=np.float64)
    out = np.empty((n_param, total_cols), dtype=np.float64)
    seen_hashes: set[str] = set()
    keep_count = 0
    for block in blocks:
        for col_idx in range(int(block.shape[1])):
            column = np.asarray(block[:, col_idx], dtype=np.float64).reshape(-1)
            digest = hash_array_payload(column)
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            out[:, keep_count] = column
            keep_count += 1
    if keep_count == 0:
        return np.zeros((n_param, 0), dtype=np.float64)
    if keep_count == total_cols:
        return out
    return np.ascontiguousarray(out[:, :keep_count], dtype=np.float64)


@dataclass
class SnapshotBank:
    """Store recent parameter-space snapshots for POD basis construction."""

    max_snapshots: int = 24
    normalize: bool = True
    _snapshots: list[np.ndarray] = field(default_factory=list)

    def add(self, snapshot: np.ndarray) -> None:
        vec = np.asarray(snapshot, dtype=np.float64).reshape(-1)
        if vec.size == 0 or not all_finite_values(vec):
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
        return _stack_columns_direct(cols)

    def snapshot_hash(self) -> str:
        mat = self.matrix()
        if mat.size == 0:
            return "empty"
        return hash_array_payload(mat)


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

    return _unique_snapshot_blocks(blocks, n_param=n_param)
