"""Dual-mesh matrix-free Jacobian operators."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from pyeidors.inverse.dual_mesh import DualMesh

ArrayAction = Callable[[np.ndarray], Any]


class DualMeshJacobianOperator:
    """Matrix-free coarse-parameter Jacobian over a fine forward mesh.

    ``Jv`` projects ``v`` from coarse inverse cells to fine forward cells before
    applying the fine sensitivity action. ``JTr`` applies the fine adjoint
    sensitivity action and restricts the resulting fine-cell gradient back to
    the coarse inverse grid.
    """

    def __init__(
        self,
        fwd_model: Any = None,
        coarse2fine: Any | None = None,
        *,
        dual_mesh: DualMesh | sparse.spmatrix | None = None,
        fine_forward_action: ArrayAction | Any | None = None,
        fine_adjoint_action: ArrayAction | Any | None = None,
        n_measurements: int | None = None,
    ) -> None:
        # Spec-facing shape: DualMeshJacobianOperator(fwd_model, coarse2fine).
        # Convenience shape used in small tests: DualMeshJacobianOperator(dual_mesh, fine_J).
        if dual_mesh is None and isinstance(fwd_model, DualMesh):
            dual_mesh = fwd_model
            if fine_forward_action is None:
                fine_forward_action = coarse2fine
        elif (
            dual_mesh is None
            and coarse2fine is None
            and fine_forward_action is not None
        ):
            dual_mesh = fwd_model
        else:
            if dual_mesh is None:
                dual_mesh = coarse2fine
            if fine_forward_action is None:
                fine_forward_action = _extract_forward_action(fwd_model)
            if fine_adjoint_action is None:
                fine_adjoint_action = _extract_adjoint_action(fwd_model)

        if dual_mesh is None:
            raise ValueError("coarse2fine projection is required.")
        if fine_forward_action is None:
            raise ValueError("fine forward action is required.")
        self.dual_mesh = dual_mesh
        self.fine_forward_action = fine_forward_action
        self.fine_adjoint_action = fine_adjoint_action
        self.n_measurements = n_measurements
        self._configure()

    def _configure(self) -> None:
        if isinstance(self.dual_mesh, DualMesh):
            projection = self.dual_mesh.coarse2fine
        else:
            projection = sparse.csr_matrix(self.dual_mesh, dtype=np.float64)
        if projection.ndim != 2 or 0 in projection.shape:
            raise ValueError("coarse2fine projection must be a non-empty 2D matrix.")
        self._projection = projection.tocsr()
        self._fine_forward_matrix = _optional_matrix(self.fine_forward_action)
        self._fine_adjoint_matrix = _optional_matrix(self.fine_adjoint_action)
        if self._fine_forward_matrix is not None:
            if self._fine_forward_matrix.shape[1] != self.n_fine_cells:
                raise ValueError(
                    "fine_forward_action matrix column count must match fine cells."
                )
            inferred = int(self._fine_forward_matrix.shape[0])
            if self.n_measurements is None:
                self.n_measurements = inferred
            elif int(self.n_measurements) != inferred:
                raise ValueError("n_measurements does not match fine_forward_action.")
        if self._fine_adjoint_matrix is not None:
            if self._fine_adjoint_matrix.shape[0] != self.n_fine_cells:
                raise ValueError(
                    "fine_adjoint_action matrix row count must match fine cells."
                )
            inferred = int(self._fine_adjoint_matrix.shape[1])
            if self.n_measurements is None:
                self.n_measurements = inferred
            elif int(self.n_measurements) != inferred:
                raise ValueError("n_measurements does not match fine_adjoint_action.")
        if self.n_measurements is None:
            raise ValueError(
                "n_measurements is required when fine actions are callables."
            )
        self.n_measurements = int(self.n_measurements)
        if self.n_measurements <= 0:
            raise ValueError("n_measurements must be positive.")

    @property
    def n_fine_cells(self) -> int:
        return int(self._projection.shape[0])

    @property
    def n_coarse_cells(self) -> int:
        return int(self._projection.shape[1])

    @property
    def coarse2fine(self) -> sparse.csr_matrix:
        return self._projection

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.n_measurements), self.n_coarse_cells)

    def Jv(self, values: Any) -> np.ndarray:
        """Apply coarse-to-measurement sensitivity action."""

        coarse = _as_vector(values, name="v", expected=self.n_coarse_cells)
        fine = np.asarray(self._projection @ coarse, dtype=np.float64)
        out = _apply_action(
            self.fine_forward_action,
            fine,
            matrix=self._fine_forward_matrix,
            name="fine_forward_action",
        )
        return _as_vector(out, name="Jv", expected=self.n_measurements)

    def JTr(self, residual: Any) -> np.ndarray:
        """Apply measurement-to-coarse adjoint sensitivity action."""

        meas = _as_vector(residual, name="r", expected=self.n_measurements)
        action = (
            self.fine_adjoint_action
            if self.fine_adjoint_action is not None
            else self.fine_forward_action
        )
        matrix = (
            self._fine_adjoint_matrix
            if self._fine_adjoint_matrix is not None
            else self._fine_forward_matrix
        )
        fine_grad = _apply_adjoint_action(
            action,
            meas,
            matrix=matrix,
            name="fine_adjoint_action",
        )
        fine_grad = _as_vector(
            fine_grad, name="fine_gradient", expected=self.n_fine_cells
        )
        return np.asarray(self._projection.T @ fine_grad, dtype=np.float64).reshape(-1)

    def normal_matvec(
        self,
        values: Any,
        *,
        measurement_weights: Any | None = None,
        alpha: float = 0.0,
        regularization: Any | None = None,
    ) -> np.ndarray:
        """Apply ``J.T W J v + alpha R v`` without materializing ``J``."""

        coarse = _as_vector(values, name="v", expected=self.n_coarse_cells)
        meas = self.Jv(coarse)
        weighted = _apply_weight(meas, measurement_weights)
        out = self.JTr(weighted)
        if alpha:
            out = out + float(alpha) * _apply_regularization(
                regularization,
                coarse,
                n=self.n_coarse_cells,
            )
        return np.asarray(out, dtype=np.float64)

    def to_dense(self) -> np.ndarray:
        """Materialize dense coarse Jacobian for tests and tiny references."""

        cols = [self.Jv(basis) for basis in np.eye(self.n_coarse_cells)]
        return np.column_stack(cols)

    def as_linear_operator(self) -> LinearOperator:
        """Return SciPy LinearOperator exposing ``Jv`` and ``JTr``."""

        return LinearOperator(
            self.shape,
            matvec=self.Jv,
            rmatvec=self.JTr,
            dtype=np.float64,
        )


def _optional_matrix(action: Any) -> np.ndarray | None:
    if action is None or callable(action):
        return None
    if sparse.issparse(action):
        matrix = np.asarray(action.toarray(), dtype=np.float64)
    else:
        matrix = np.asarray(action, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("matrix action must be a non-empty 2D array.")
    if not np.isfinite(matrix).all():
        raise FloatingPointError("matrix action contains non-finite values.")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _extract_forward_action(source: Any) -> Any:
    if source is None:
        return None
    for name in (
        "fine_forward_action",
        "forward_action",
        "Jv",
        "matvec",
        "jacobian",
        "J",
    ):
        attr = getattr(source, name, None)
        if attr is not None:
            return attr() if name in {"jacobian", "J"} and callable(attr) else attr
    return source


def _extract_adjoint_action(source: Any) -> Any:
    if source is None:
        return None
    for name in ("fine_adjoint_action", "adjoint_action", "JTr", "rmatvec"):
        attr = getattr(source, name, None)
        if attr is not None:
            return attr
    for name in ("jacobian", "J"):
        attr = getattr(source, name, None)
        if attr is not None:
            return attr() if callable(attr) else attr
    return None


def _apply_action(
    action: ArrayAction | Any,
    values: np.ndarray,
    *,
    matrix: np.ndarray | None,
    name: str,
) -> np.ndarray:
    if matrix is not None:
        return np.asarray(matrix @ values, dtype=np.float64)
    if not callable(action):
        raise TypeError(f"{name} must be callable or a matrix.")
    return np.asarray(action(values), dtype=np.float64)


def _apply_adjoint_action(
    action: ArrayAction | Any,
    values: np.ndarray,
    *,
    matrix: np.ndarray | None,
    name: str,
) -> np.ndarray:
    if matrix is not None:
        if matrix.shape[0] == values.size:
            return np.asarray(matrix.T @ values, dtype=np.float64)
        return np.asarray(matrix @ values, dtype=np.float64)
    if not callable(action):
        raise TypeError(f"{name} must be callable or a matrix.")
    return np.asarray(action(values), dtype=np.float64)


def _as_vector(values: Any, *, name: str, expected: int | None = None) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if expected is not None and vector.size != int(expected):
        raise ValueError(f"{name} length {vector.size} does not match {expected}.")
    if not np.isfinite(vector).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(vector, dtype=np.float64)


def _apply_weight(values: np.ndarray, weights: Any | None) -> np.ndarray:
    if weights is None:
        return values
    if sparse.issparse(weights):
        matrix = weights.tocsr()
        return np.asarray(matrix @ values, dtype=np.float64)
    array = np.asarray(weights, dtype=np.float64)
    if array.ndim == 1:
        if array.size != values.size:
            raise ValueError("measurement_weights length mismatch.")
        return np.asarray(array * values, dtype=np.float64)
    if array.shape != (values.size, values.size):
        raise ValueError("measurement_weights matrix shape mismatch.")
    return np.asarray(array @ values, dtype=np.float64)


def _apply_regularization(
    values: Any | None, vector: np.ndarray, *, n: int
) -> np.ndarray:
    if values is None:
        return vector
    if sparse.issparse(values):
        matrix = values.tocsr()
        if matrix.shape != (n, n):
            raise ValueError("regularization matrix shape mismatch.")
        return np.asarray(matrix @ vector, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        if array.size != n:
            raise ValueError("regularization diagonal length mismatch.")
        return np.asarray(array * vector, dtype=np.float64)
    if array.shape != (n, n):
        raise ValueError("regularization matrix shape mismatch.")
    return np.asarray(array @ vector, dtype=np.float64)


__all__ = ["DualMeshJacobianOperator"]
