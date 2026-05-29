"""Regularization preparation helpers for the Gauss-Newton reconstructor."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.sparse import isspmatrix
from scipy.sparse.linalg import LinearOperator

from ..prior import RtRPrior
from ..regularization.base_regularization import BaseRegularization
from ...utils.numeric_ops import all_finite_values


def _is_rtr_prior_contract(value: Any) -> bool:
    return isinstance(value, RtRPrior) or all(
        callable(getattr(value, attr, None))
        for attr in ("apply", "diag", "as_RtR", "as_linear_operator")
    )


def _finite_min_max(values: np.ndarray) -> tuple[float, float]:
    min_val = np.inf
    max_val = -np.inf
    count = 0
    for raw_value in np.nditer(
        np.asarray(values).reshape(-1),
        flags=["refs_ok"],
        op_flags=["readonly"],
    ):
        value = float(raw_value)
        if not np.isfinite(value):
            continue
        count += 1
        if value < min_val:
            min_val = value
        if value > max_val:
            max_val = value
    if count == 0:
        return float("nan"), float("nan")
    return float(min_val), float(max_val)


def ensure_regularization_ready(reconstructor) -> None:
    """Build and validate the cached regularization tensor used by GN."""
    expected_shape = (reconstructor.n_elements, reconstructor.n_elements)
    needs_dense_tensor = (
        reconstructor.solver_mode == "strict"
        or reconstructor.line_search_mode == "full"
    )
    cache_ready = (
        reconstructor.R_matrix is not None
        and reconstructor.R_linear_operator is not None
        and (not needs_dense_tensor or reconstructor.R_torch is not None)
    )
    if cache_ready:
        return

    matrix = reconstructor.regularization.get_regularization_matrix()
    matrix_shape = tuple(getattr(matrix, "shape", ()))
    if matrix_shape != expected_shape:
        raise RuntimeError(
            "Regularization matrix shape mismatch: "
            f"expected {expected_shape}, got {matrix_shape}."
        )

    if _is_rtr_prior_contract(matrix):
        reconstructor.R_matrix = matrix
        reconstructor.R_linear_operator = matrix.as_linear_operator()
        probe = np.ones(reconstructor.n_elements, dtype=np.float64)
        check = np.asarray(matrix.apply(probe), dtype=np.float64)
        if not all_finite_values(check):
            raise FloatingPointError(
                "Regularization RtRPrior produces non-finite values."
            )
        diag = matrix.diag()
        reconstructor.R_diag = (
            None if diag is None else np.asarray(diag, dtype=np.float64).reshape(-1)
        )
        if reconstructor.R_diag is not None and not all_finite_values(
            reconstructor.R_diag
        ):
            raise FloatingPointError("Regularization RtRPrior diag is non-finite.")
        if needs_dense_tensor and reconstructor.solver_mode == "strict":
            dense_like = matrix.as_RtR(dense=True)
            if isspmatrix(dense_like):
                dense = dense_like.toarray()
            elif isinstance(dense_like, LinearOperator):
                raise RuntimeError(
                    "solver_mode='strict' requires explicit dense/sparse regularization matrix, "
                    "matrix-free RtRPrior is not supported."
                )
            else:
                dense = np.asarray(dense_like, dtype=np.float64)
            if not all_finite_values(dense):
                raise FloatingPointError(
                    "Regularization RtRPrior dense view contains non-finite values."
                )
            reconstructor.R_torch = torch.from_numpy(dense).to(
                reconstructor.device,
                dtype=reconstructor._torch_dtype,
            )
            if not torch.isfinite(reconstructor.R_torch).all():
                raise FloatingPointError(
                    "Regularization tensor contains non-finite values after transfer."
                )
        else:
            reconstructor.R_torch = None
        return

    reconstructor.R_matrix = matrix
    as_linear_operator = getattr(
        reconstructor.regularization, "as_linear_operator", None
    )
    if callable(as_linear_operator):
        reconstructor.R_linear_operator = as_linear_operator(
            matrix, shape=expected_shape
        )
    else:
        reconstructor.R_linear_operator = BaseRegularization.as_linear_operator(
            matrix, shape=expected_shape
        )

    if isspmatrix(matrix):
        if matrix.nnz == 0:
            raise FloatingPointError("Regularization sparse matrix is empty.")
        if not all_finite_values(matrix.data):
            min_val, max_val = _finite_min_max(matrix.data)
            raise FloatingPointError(
                "Regularization sparse matrix contains non-finite values: "
                f"finite_min={min_val:.6e}, finite_max={max_val:.6e}."
            )
        if matrix.format == "csr":
            diag = matrix.diagonal()
        else:
            diag = matrix.tocsr().diagonal()
        reconstructor.R_diag = np.asarray(diag, dtype=np.float64)
        if needs_dense_tensor:
            dense = matrix.toarray()
            reconstructor.R_torch = torch.from_numpy(
                np.asarray(dense, dtype=np.float64)
            ).to(
                reconstructor.device,
                dtype=reconstructor._torch_dtype,
            )
        else:
            reconstructor.R_torch = None
        return

    if isinstance(matrix, LinearOperator):
        probe = np.ones(reconstructor.n_elements, dtype=np.float64)
        check = np.asarray(matrix.matvec(probe), dtype=np.float64)
        if not all_finite_values(check):
            raise FloatingPointError(
                "Regularization LinearOperator produces non-finite values."
            )
        reconstructor.R_diag = None
        reconstructor.R_torch = None
        if reconstructor.solver_mode == "strict":
            raise RuntimeError(
                "solver_mode='strict' requires explicit dense/sparse regularization matrix, "
                "LinearOperator is not supported."
            )
        return

    dense = np.asarray(matrix, dtype=np.float64)
    if not all_finite_values(dense):
        min_val, max_val = _finite_min_max(dense)
        raise FloatingPointError(
            "Regularization matrix contains non-finite values: "
            f"finite_min={min_val:.6e}, finite_max={max_val:.6e}."
        )
    reconstructor.R_diag = np.asarray(dense.diagonal(), dtype=np.float64)
    if needs_dense_tensor:
        reconstructor.R_torch = torch.from_numpy(dense).to(
            reconstructor.device,
            dtype=reconstructor._torch_dtype,
        )
        if not torch.isfinite(reconstructor.R_torch).all():
            raise FloatingPointError(
                "Regularization tensor contains non-finite values after transfer."
            )
    else:
        reconstructor.R_torch = None
