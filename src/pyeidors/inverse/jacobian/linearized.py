"""Matrix-free Jacobian actions for EIT sensitivity linearizations."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.sparse import isspmatrix
from scipy.sparse.linalg import LinearOperator

RegularizationAction = Callable[[np.ndarray], np.ndarray] | LinearOperator | np.ndarray


def compute_sigma_fingerprint(sigma_values) -> str:
    """Return a stable content hash for the conductivity values of ``sigma``."""
    values = getattr(sigma_values, "x", None)
    if values is not None:
        array = getattr(values, "array", None)
        if array is not None:
            sigma_values = array
    array = np.ascontiguousarray(np.asarray(sigma_values), dtype=np.float64).reshape(-1)
    return hashlib.sha256(array.tobytes()).hexdigest()


@dataclass
class JacobianLinearization:
    """Apply EIT Jacobian actions without materializing the dense Jacobian.

    The object stores forward and adjoint field gradients for one linearization
    point and exposes ``Jv`` and ``J^T r`` operations. Existing dense workflows
    can still call :meth:`to_dense`, but inverse solvers can use the operator
    actions directly.

    ``sigma_fingerprint`` is a content hash of the conductivity that produced
    the stored gradients. It is empty by default for backwards compatibility;
    when set, :meth:`assert_compatible` guards external reuse against silently
    applying stale gradients to a different linearization point.
    """

    grad_u_all: tuple[np.ndarray, ...]
    adjoint_gradients: tuple[np.ndarray, ...]
    cell_areas: np.ndarray
    n_meas_per_stim: tuple[int, ...]
    sign: float = 1.0
    sigma_fingerprint: str = ""

    def __post_init__(self) -> None:
        self.cell_areas = np.asarray(self.cell_areas, dtype=np.float64)
        self.grad_u_all = tuple(
            np.asarray(g, dtype=np.float64) for g in self.grad_u_all
        )
        self.adjoint_gradients = tuple(
            np.asarray(g, dtype=np.float64) for g in self.adjoint_gradients
        )
        self.n_meas_per_stim = tuple(int(v) for v in self.n_meas_per_stim)
        self.sign = float(self.sign)
        self.sigma_fingerprint = str(self.sigma_fingerprint or "")
        self._validate_shapes()

    def assert_compatible(self, sigma_fingerprint: str | None) -> None:
        """Raise if the stored gradients predate a new conductivity value.

        The check is permissive: it only fires when both the stored and
        provided fingerprints are non-empty and differ. An empty stored
        fingerprint (legacy construction) or an empty ``sigma_fingerprint``
        argument skips the guard.
        """
        stored = str(self.sigma_fingerprint or "")
        provided = str(sigma_fingerprint or "")
        if not stored or not provided:
            return
        if stored != provided:
            raise ValueError(
                "JacobianLinearization sigma fingerprint mismatch: "
                f"stored={stored[:12]}..., provided={provided[:12]}..."
            )

    @property
    def n_parameters(self) -> int:
        return int(self.cell_areas.size)

    @property
    def n_measurements(self) -> int:
        return int(sum(self.n_meas_per_stim))

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_measurements, self.n_parameters

    def _validate_shapes(self) -> None:
        if len(self.grad_u_all) != len(self.n_meas_per_stim):
            raise ValueError("grad_u_all length must match n_meas_per_stim.")
        if len(self.adjoint_gradients) != self.n_measurements:
            raise ValueError("adjoint_gradients length must match total measurements.")
        for grad_u in self.grad_u_all:
            if grad_u.ndim != 2 or grad_u.shape[0] != self.n_parameters:
                raise ValueError(
                    "Each forward gradient must have shape (n_elem, gdim)."
                )
        for grad_adj in self.adjoint_gradients:
            if grad_adj.ndim != 2 or grad_adj.shape[0] != self.n_parameters:
                raise ValueError(
                    "Each adjoint gradient must have shape (n_elem, gdim)."
                )

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        """Apply ``J v``."""
        vec = np.asarray(vector, dtype=np.float64).reshape(-1)
        if vec.size != self.n_parameters:
            raise ValueError(
                f"Expected vector length {self.n_parameters}, got {vec.size}."
            )

        out = np.zeros(self.n_measurements, dtype=np.float64)
        weighted_cell = self.cell_areas * vec
        meas_idx = 0
        for stim_idx, grad_u in enumerate(self.grad_u_all):
            n_meas = self.n_meas_per_stim[stim_idx]
            adjoint_block = np.asarray(
                self.adjoint_gradients[meas_idx : meas_idx + n_meas],
                dtype=np.float64,
            )
            out[meas_idx : meas_idx + n_meas] = self.sign * np.einsum(
                "eg,meg,e->m",
                grad_u,
                adjoint_block,
                weighted_cell,
                optimize=True,
            )
            meas_idx += n_meas
        return out

    def rmatvec(self, residual: np.ndarray) -> np.ndarray:
        """Apply ``J^T r``."""
        res = np.asarray(residual, dtype=np.float64).reshape(-1)
        if res.size != self.n_measurements:
            raise ValueError(
                f"Expected residual length {self.n_measurements}, got {res.size}."
            )

        out = np.zeros(self.n_parameters, dtype=np.float64)
        meas_idx = 0
        for stim_idx, grad_u in enumerate(self.grad_u_all):
            n_meas = self.n_meas_per_stim[stim_idx]
            adjoint_block = np.asarray(
                self.adjoint_gradients[meas_idx : meas_idx + n_meas],
                dtype=np.float64,
            )
            weighted_adjoint = np.einsum(
                "m,meg->eg",
                res[meas_idx : meas_idx + n_meas],
                adjoint_block,
                optimize=True,
            )
            out += self.sign * np.einsum(
                "eg,eg,e->e",
                grad_u,
                weighted_adjoint,
                self.cell_areas,
                optimize=True,
            )
            meas_idx += n_meas
        return out

    def as_linear_operator(self) -> LinearOperator:
        """Return a SciPy ``LinearOperator`` wrapping ``J``."""
        return LinearOperator(
            self.shape,
            matvec=self.matvec,
            rmatvec=self.rmatvec,
            dtype=np.float64,
        )

    def to_dense(self, *, block_size: int | None = None) -> np.ndarray:
        """Materialize the dense Jacobian for compatibility or debugging."""
        n_meas, n_param = self.shape
        dense = np.zeros((n_meas, n_param), dtype=np.float64)
        block = n_param if block_size is None else max(1, int(block_size))

        meas_idx = 0
        for stim_idx, grad_u in enumerate(self.grad_u_all):
            n_meas_this = self.n_meas_per_stim[stim_idx]
            adjoint_block = np.asarray(
                self.adjoint_gradients[meas_idx : meas_idx + n_meas_this],
                dtype=np.float64,
            )
            for start in range(0, n_param, block):
                end = min(start + block, n_param)
                dense[meas_idx : meas_idx + n_meas_this, start:end] = (
                    self.sign
                    * np.einsum(
                        "eg,meg->me",
                        grad_u[start:end, :],
                        adjoint_block[:, start:end, :],
                        optimize=True,
                    )
                    * self.cell_areas[None, start:end]
                )
            meas_idx += n_meas_this
        return dense

    def hessian_diag(
        self,
        *,
        measurement_weights: np.ndarray | None = None,
        alpha: float = 0.0,
        regularization_diag: np.ndarray | None = None,
        floor: float = 0.0,
    ) -> np.ndarray:
        """Return ``diag(J^T W J) [+ alpha * R_diag]`` without dense ``J``.

        Useful as a free NOSER-style diagonal preconditioner for matrix-free
        Gauss-Newton solves. The contraction reuses the same ``grad_u_all`` /
        ``adjoint_gradients`` buffers that ``matvec`` / ``rmatvec`` rely on.
        """
        weights = None
        if measurement_weights is not None:
            weights = np.asarray(measurement_weights, dtype=np.float64).reshape(-1)
            if weights.size != self.n_measurements:
                raise ValueError(
                    f"Expected {self.n_measurements} weights, got {weights.size}."
                )

        diag = np.zeros(self.n_parameters, dtype=np.float64)
        meas_idx = 0
        for stim_idx, grad_u in enumerate(self.grad_u_all):
            n_meas = self.n_meas_per_stim[stim_idx]
            adjoint_block = np.asarray(
                self.adjoint_gradients[meas_idx : meas_idx + n_meas],
                dtype=np.float64,
            )
            # Per (measurement m, element e) sensitivity before cell_area scaling.
            contrib = np.einsum(
                "eg,meg->me", grad_u, adjoint_block, optimize=True
            )
            contrib_sq = contrib * contrib
            if weights is not None:
                contrib_sq = contrib_sq * weights[meas_idx : meas_idx + n_meas, None]
            diag += contrib_sq.sum(axis=0)
            meas_idx += n_meas
        diag = diag * (float(self.sign) ** 2) * (self.cell_areas ** 2)

        if float(alpha) != 0.0 and regularization_diag is not None:
            reg = np.asarray(regularization_diag, dtype=np.float64).reshape(-1)
            if reg.size != self.n_parameters:
                raise ValueError(
                    f"Expected {self.n_parameters} regularization diag entries, "
                    f"got {reg.size}."
                )
            diag = diag + float(alpha) * reg

        if float(floor) > 0.0:
            diag = np.maximum(diag, float(floor))
        return np.asarray(diag, dtype=np.float64)

    def normal_matvec(
        self,
        vector: np.ndarray,
        *,
        measurement_weights: np.ndarray | None = None,
        alpha: float = 0.0,
        regularization: RegularizationAction | None = None,
    ) -> np.ndarray:
        """Apply ``J^T W J v + alpha R v`` without dense ``J`` or ``H``."""
        projected = self.matvec(vector)
        if measurement_weights is not None:
            weights = np.asarray(measurement_weights, dtype=np.float64).reshape(-1)
            if weights.size != self.n_measurements:
                raise ValueError(
                    f"Expected {self.n_measurements} weights, got {weights.size}."
                )
            projected = weights * projected
        out = self.rmatvec(projected)
        if regularization is not None and float(alpha) != 0.0:
            out = out + float(alpha) * self._apply_regularization(
                regularization, vector
            )
        return np.asarray(out, dtype=np.float64)

    def as_normal_operator(
        self,
        *,
        measurement_weights: np.ndarray | None = None,
        alpha: float = 0.0,
        regularization: RegularizationAction | None = None,
    ) -> LinearOperator:
        """Return ``J^T W J + alpha R`` as a SciPy ``LinearOperator``."""
        return LinearOperator(
            (self.n_parameters, self.n_parameters),
            matvec=lambda v: self.normal_matvec(
                v,
                measurement_weights=measurement_weights,
                alpha=alpha,
                regularization=regularization,
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _apply_regularization(
        regularization: RegularizationAction,
        vector: np.ndarray,
    ) -> np.ndarray:
        vec = np.asarray(vector, dtype=np.float64)
        if isinstance(regularization, LinearOperator):
            return np.asarray(regularization.matvec(vec), dtype=np.float64)
        if callable(regularization):
            return np.asarray(regularization(vec), dtype=np.float64)
        if isspmatrix(regularization):
            return np.asarray(regularization.dot(vec), dtype=np.float64)
        matrix = np.asarray(regularization, dtype=np.float64)
        return np.asarray(matrix @ vec, dtype=np.float64)

    def as_petsc_mat(self, *, comm=None):
        """Return a PETSc Python matrix for ``J`` when petsc4py is available."""
        try:
            from petsc4py import PETSc
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError(
                "petsc4py is required to create a PETSc matrix."
            ) from exc

        context = _PETScJacobianContext(self)
        mat = PETSc.Mat().createPython(self.shape, context=context, comm=comm)
        mat.setUp()
        return mat


class _PETScJacobianContext:
    """petsc4py context object for ``Mat.Type.PYTHON``."""

    def __init__(self, linearization: JacobianLinearization):
        self.linearization = linearization

    @staticmethod
    def _vec_array(vec) -> np.ndarray:
        if hasattr(vec, "array"):
            return np.asarray(vec.array, dtype=np.float64)
        return np.asarray(vec.getArray(readonly=True), dtype=np.float64)

    def mult(self, _mat, x, y) -> None:
        result = self.linearization.matvec(self._vec_array(x))
        y_arr = y.getArray(readonly=False)
        y_arr[:] = result

    def multTranspose(self, _mat, x, y) -> None:
        result = self.linearization.rmatvec(self._vec_array(x))
        y_arr = y.getArray(readonly=False)
        y_arr[:] = result
