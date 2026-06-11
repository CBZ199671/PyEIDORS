"""ROI-restricted TV refinement for one-step / GREIT reconstructions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy import sparse

from pyeidors.data._temporal_core import (
    as_real_float_array as _as_real_float_array,
    positive_int as _positive_int,
)
from pyeidors.inverse.prior import graph_difference_operator
from pyeidors.utils.numeric_ops import all_finite_values


_ROI_CHUNK_ITEMS = 1_048_576


@dataclass(frozen=True)
class TVRefinementResult:
    """TV-refined reconstruction plus convergence diagnostics."""

    values: np.ndarray
    metadata: MappingProxyType

    @property
    def roi_residual_norm_history(self) -> tuple[float, ...]:
        return tuple(self.metadata["roi_residual_norm_history"])

    @property
    def tv_norm_history(self) -> tuple[float, ...]:
        return tuple(self.metadata["tv_norm_history"])

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)


def _roi_relative_residual(
    current: np.ndarray,
    previous: np.ndarray,
    roi: np.ndarray,
    *,
    chunk_size: int = _ROI_CHUNK_ITEMS,
) -> float:
    work_dtype = np.result_type(
        _as_real_float_array(current).dtype,
        _as_real_float_array(previous).dtype,
        np.float32,
    )
    current_arr = np.asarray(_as_real_float_array(current), dtype=work_dtype).reshape(
        -1
    )
    previous_arr = np.asarray(_as_real_float_array(previous), dtype=work_dtype).reshape(
        -1
    )
    roi_arr = np.asarray(roi, dtype=bool).reshape(-1)
    if current_arr.shape != previous_arr.shape or current_arr.shape != roi_arr.shape:
        raise ValueError("current, previous, and roi must have matching lengths.")
    if current_arr.size == 0:
        return 0.0
    block_size = max(1, min(int(chunk_size), int(current_arr.size)))
    diff_work = np.empty(block_size, dtype=work_dtype)
    numerator_sq = 0.0
    denominator_sq = 0.0
    for start in range(0, int(current_arr.size), block_size):
        stop = min(start + block_size, int(current_arr.size))
        active = roi_arr[start:stop]
        if not bool(active.any()):
            continue
        diff = diff_work[: stop - start]
        np.subtract(current_arr[start:stop], previous_arr[start:stop], out=diff)
        np.square(diff, out=diff)
        numerator_sq += float(np.sum(diff, where=active, initial=0.0))
        np.square(previous_arr[start:stop], out=diff)
        denominator_sq += float(np.sum(diff, where=active, initial=0.0))
    return float(np.sqrt(numerator_sq)) / max(float(np.sqrt(denominator_sq)), 1.0)


def refine_tv_pdhg(
    seed: Any,
    mesh: Any,
    *,
    roi_mask: Any | None = None,
    tv_weight: float = 1.0e-2,
    max_iterations: int = 100,
    tolerance: float = 1.0e-6,
    over_relaxation: float = 1.0,
    graph_weight: str = "unit",
    return_metadata: bool = False,
    seed_source: str = "one_step_rm",
) -> np.ndarray | TVRefinementResult:
    """Refine an RM-seeded image with ROI-restricted anisotropic TV-PDHG.

    The solved model is ``0.5 ||x - seed||^2 + tv_weight * ||D x||_1``.
    Outside ``roi_mask`` the reconstruction is pinned to the input seed. The
    convergence gate is the ROI-restricted step residual norm.
    """

    y = _as_seed(seed)
    roi = _as_roi_mask(roi_mask, n=y.size)
    weight = _nonnegative_float(tv_weight, "tv_weight")
    max_it = _positive_int(max_iterations, "max_iterations")
    tol = _nonnegative_float(tolerance, "tolerance")
    theta = _nonnegative_float(over_relaxation, "over_relaxation")
    D = (
        graph_difference_operator(mesh, weight=graph_weight)
        .astype(y.dtype, copy=False)
        .tocsr()
    )
    if D.shape[1] != y.size:
        raise ValueError(
            f"mesh cell count {D.shape[1]} does not match seed length {y.size}."
        )

    if D.shape[0] == 0 or weight == 0.0:
        metadata = _metadata(
            iterations=0,
            stopped_reason="no_tv_edges" if D.shape[0] == 0 else "zero_tv_weight",
            roi=roi,
            D=D,
            residual_history=(0.0,),
            tv_history=(total_variation_norm(y, D),),
            tv_weight=weight,
            tolerance=tol,
            seed_source=seed_source,
        )
        result = TVRefinementResult(values=y.copy(), metadata=metadata)
        return result if return_metadata else result.values

    tau, sigma = _pdhg_steps(D)
    x = y.copy()
    x_bar = x.copy()
    dual = np.zeros(D.shape[0], dtype=y.dtype)
    previous = np.empty_like(x)
    x_new = np.empty_like(x)
    update = np.empty_like(x)
    np.copyto(x_new, y)
    residual_history: list[float] = []
    tv_history: list[float] = [total_variation_norm(x, D)]
    stopped_reason = "max_iterations"

    for _ in range(1, max_it + 1):
        Dxbar = np.asarray(D @ x_bar, dtype=y.dtype).reshape(-1)
        dual += sigma * Dxbar
        if weight == 0.0:
            dual.fill(0.0)
        else:
            np.clip(dual, -float(weight), float(weight), out=dual)

        np.copyto(previous, x)
        descent = np.asarray(D.T @ dual, dtype=y.dtype).reshape(-1)
        np.multiply(descent, tau, out=update)
        np.subtract(x, update, out=update)
        update += tau * y
        update /= 1.0 + tau
        np.copyto(x_new, update, where=roi)

        roi_residual = _roi_relative_residual(x_new, previous, roi)
        residual_history.append(roi_residual)
        tv_history.append(total_variation_norm(x_new, D))

        if theta == 0.0:
            np.copyto(x_bar, x_new)
        else:
            np.subtract(x_new, previous, out=x_bar)
            x_bar *= theta
            x_bar += x_new
        np.copyto(x, x_new)
        if roi_residual <= tol:
            stopped_reason = "roi_residual_tolerance"
            break

    metadata = _metadata(
        iterations=len(residual_history),
        stopped_reason=stopped_reason,
        roi=roi,
        D=D,
        residual_history=tuple(residual_history),
        tv_history=tuple(tv_history),
        tv_weight=weight,
        tolerance=tol,
        seed_source=seed_source,
    )
    result = TVRefinementResult(values=x, metadata=metadata)
    return result if return_metadata else result.values


def total_variation_norm(values: Any, operator: Any) -> float:
    """Return anisotropic graph TV norm ``sum(abs(D @ values))``."""

    vector = _as_seed(values)
    D = sparse.csr_matrix(operator, dtype=vector.dtype)
    if D.ndim != 2 or D.shape[1] != vector.size:
        raise ValueError("operator column count must match values length.")
    diff = np.asarray(D @ vector, dtype=vector.dtype).reshape(-1)
    if not all_finite_values(diff):
        raise FloatingPointError("TV differences contain non-finite values.")
    return float(np.sum(np.abs(diff)))


def _as_seed(values: Any) -> np.ndarray:
    vector = _as_real_float_array(values).reshape(-1)
    if vector.size == 0:
        raise ValueError("seed must be non-empty.")
    if not all_finite_values(vector):
        raise FloatingPointError("seed contains non-finite values.")
    return np.ascontiguousarray(vector)


def _as_roi_mask(values: Any | None, *, n: int) -> np.ndarray:
    if values is None:
        return np.ones(int(n), dtype=bool)
    mask = np.asarray(values, dtype=bool).reshape(-1)
    if mask.size != int(n):
        raise ValueError(f"roi_mask length {mask.size} does not match {n}.")
    if not np.any(mask):
        raise ValueError("roi_mask must select at least one parameter.")
    return np.ascontiguousarray(mask, dtype=bool)


def _nonnegative_float(value: float, name: str) -> float:
    resolved = float(value)
    if not np.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return resolved


def _pdhg_steps(operator: sparse.csr_matrix) -> tuple[float, float]:
    norm_estimate = float(sparse.linalg.norm(operator))
    if not np.isfinite(norm_estimate) or norm_estimate <= 0.0:
        norm_estimate = 1.0
    step = 0.99 / norm_estimate
    return step, step


def _metadata(
    *,
    iterations: int,
    stopped_reason: str,
    roi: np.ndarray,
    D: sparse.csr_matrix,
    residual_history: tuple[float, ...],
    tv_history: tuple[float, ...],
    tv_weight: float,
    tolerance: float,
    seed_source: str,
) -> MappingProxyType:
    return MappingProxyType(
        {
            "method": "tv-pdhg",
            "seed_source": str(seed_source),
            "iterations": int(iterations),
            "stopped_reason": str(stopped_reason),
            "roi_size": int(np.count_nonzero(roi)),
            "n_parameters": int(roi.size),
            "difference_operator_shape": tuple(int(v) for v in D.shape),
            "difference_operator_nnz": int(D.nnz),
            "roi_residual_norm_history": tuple(float(v) for v in residual_history),
            "tv_norm_history": tuple(float(v) for v in tv_history),
            "tv_weight": float(tv_weight),
            "tolerance": float(tolerance),
        }
    )


__all__ = ["TVRefinementResult", "refine_tv_pdhg", "total_variation_norm"]
