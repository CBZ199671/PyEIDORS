"""Helpers for representing complex linear systems as real block systems."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse


def complex_csr_to_block_real(
    matrix: sparse.spmatrix | np.ndarray,
    *,
    dtype: Any = np.float64,
) -> sparse.csr_matrix:
    """Return the real block form of ``A x = b`` for a complex CSR matrix.

    The returned matrix maps ``[Re(x), Im(x)]`` to ``[Re(b), Im(b)]`` using
    ``[[Re(A), -Im(A)], [Im(A), Re(A)]]``. This is the system shape needed to
    test complex CEM solves through a real-scalar PETSc/AmgX runtime.
    """

    csr = sparse.csr_matrix(matrix)
    real = csr.real.astype(dtype, copy=False)
    imag = csr.imag.astype(dtype, copy=False)
    top = sparse.hstack((real, -imag), format="csr")
    bottom = sparse.hstack((imag, real), format="csr")
    return sparse.vstack((top, bottom), format="csr").astype(dtype, copy=False)


def complex_rhs_to_block_real(
    values: np.ndarray, *, dtype: Any = np.float64
) -> np.ndarray:
    """Stack complex RHS columns into the matching real block layout."""

    arr = np.asarray(values)
    if arr.ndim == 1:
        return np.concatenate((arr.real, arr.imag)).astype(dtype, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D RHS array, got shape={arr.shape!r}")
    return np.vstack((arr.real, arr.imag)).astype(dtype, copy=False)


def block_real_solution_to_complex(
    values: np.ndarray,
    *,
    original_size: int | None = None,
) -> np.ndarray:
    """Convert ``[Re(x), Im(x)]`` block-real solutions back to complex values."""

    arr = np.asarray(values)
    n_rows = int(arr.shape[0])
    if original_size is None:
        if n_rows % 2:
            raise ValueError(
                f"block-real row count must be even when original_size is omitted; got {n_rows}"
            )
        original_size = n_rows // 2
    if n_rows != int(original_size) * 2:
        raise ValueError(
            f"block-real row count mismatch: expected {int(original_size) * 2}, got {n_rows}"
        )
    real = arr[:original_size]
    imag = arr[original_size:]
    return np.asarray(real) + 1j * np.asarray(imag)


def relative_l2_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return ``||candidate-reference||_2 / max(||reference||_2, eps)``."""

    ref = np.asarray(reference)
    cand = np.asarray(candidate)
    if ref.shape != cand.shape:
        raise ValueError(
            f"shape mismatch: reference={ref.shape!r}, candidate={cand.shape!r}"
        )
    denom = float(np.linalg.norm(ref.reshape(-1)))
    if denom == 0.0:
        denom = np.finfo(np.float64).eps
    return float(np.linalg.norm((cand - ref).reshape(-1)) / denom)


def absolute_error_summary(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    """Small JSON-safe error summary for strict parity reports."""

    diff = np.asarray(candidate) - np.asarray(reference)
    flat = diff.reshape(-1)
    abs_flat = np.abs(flat)
    return {
        "relative_l2": relative_l2_error(reference, candidate),
        "max_abs": float(np.max(abs_flat)) if abs_flat.size else 0.0,
        "mean_abs": float(np.mean(abs_flat)) if abs_flat.size else 0.0,
        "rms_abs": float(np.sqrt(np.mean(abs_flat**2))) if abs_flat.size else 0.0,
    }
