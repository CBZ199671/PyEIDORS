"""Matrix-free Jacobian actions for EIT sensitivity linearizations.

Path C (T80) consolidates the public matrix-free operator surface that
the eager :class:`JacobianLinearization` and the lazy
:class:`LazyAdjointJacobianLinearization` historically duplicated. Both
classes now inherit from :class:`_LinearizationBase` which owns the
shape properties, the permissive fingerprint check (V9), the
``LinearOperator`` adapters, the ``J^H W J v + alpha R v`` normal-matvec
helper and the PETSc Python matrix wrapper. Each subclass keeps its own
storage layout (eager: stored adjoint gradients; lazy: ``fwd_model`` +
on-demand adjoint solve) and overrides the abstract
``matvec`` / ``rmatvec`` / ``hessian_diag`` / ``_validate_shapes`` hooks.
The V73 sign defaults (eager ``+1.0``, lazy ``-1.0``) are unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import isspmatrix
from scipy.sparse.linalg import LinearOperator

from ...cache.keys import hash_array_payload

RegularizationAction = Callable[[np.ndarray], np.ndarray] | LinearOperator | np.ndarray
GradientCallback = Callable[[Sequence[np.ndarray]], Sequence[np.ndarray]]


def _complex_preserving_dtype(*values) -> np.dtype:
    dtypes = [
        np.dtype(value)
        if isinstance(value, (np.dtype, type))
        else np.asarray(value).dtype
        for value in values
        if value is not None
    ]
    complex_dtypes = [
        dtype for dtype in dtypes if np.issubdtype(dtype, np.complexfloating)
    ]
    if complex_dtypes:
        if any(dtype != np.dtype(np.complex64) for dtype in complex_dtypes):
            return np.dtype(np.complex128)
        return np.dtype(np.complex64)
    return np.dtype(np.float64)


def compute_sigma_fingerprint(sigma_values) -> str:
    """Return a stable content hash for the conductivity values of ``sigma``."""
    values = getattr(sigma_values, "x", None)
    if values is not None:
        array = getattr(values, "array", None)
        if array is not None:
            sigma_values = array
    raw = np.asarray(sigma_values)
    if np.iscomplexobj(raw):
        array = np.asarray(
            raw,
            dtype=np.complex64 if raw.dtype == np.complex64 else np.complex128,
        )
        return hash_array_payload(
            array, prefix=str(array.dtype).encode("utf-8") + b"\0"
        )
    array = np.asarray(raw, dtype=np.float64)
    return hash_array_payload(array)


def _weighted_contrib_power_sum(
    contrib: np.ndarray,
    weights: np.ndarray,
    cell_area_sq: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        np.einsum(
            "me,me,m,e->e",
            np.conjugate(contrib),
            contrib,
            np.asarray(weights, dtype=np.float64),
            np.asarray(cell_area_sq, dtype=np.float64),
            optimize=True,
        ).real,
        dtype=np.float64,
    )


def _stack_adjoint_gradient_block(
    gradients: Sequence[np.ndarray],
    *,
    start: int,
    count: int,
    dtype: np.dtype,
) -> np.ndarray:
    if count <= 0:
        raise ValueError("adjoint gradient block must contain at least one row.")
    first = np.asarray(gradients[start], dtype=dtype)
    block = np.empty((int(count), *first.shape), dtype=dtype)
    block[0, :, :] = first
    for local_idx in range(1, int(count)):
        block[local_idx, :, :] = np.asarray(
            gradients[int(start) + local_idx], dtype=dtype
        )
    return np.ascontiguousarray(block, dtype=dtype)


class _LinearizationBase:
    """Shared public surface for EIT Jacobian matrix-free linearizations.

    Concrete subclasses own their storage layout (eager stores
    ``adjoint_gradients``; lazy stores ``fwd_model`` and computes
    adjoints on demand) and override ``matvec`` / ``rmatvec`` /
    ``hessian_diag`` / ``_validate_shapes``. This base class holds the
    shape derivations, the V9 fingerprint check, the
    ``LinearOperator`` / PETSc Python matrix adapters and the
    ``J^H W J v + alpha R v`` normal-matvec helper so the eager and
    lazy classes never re-implement them out of step.

    Subclasses MUST set the following attributes during ``__post_init__``:
    ``cell_areas`` (``ndarray``), ``n_meas_per_stim`` (``tuple[int, ...]``),
    ``sign`` (``float``), ``sigma_fingerprint`` (``str``).
    """

    cell_areas: np.ndarray
    n_meas_per_stim: tuple[int, ...]
    sign: float
    sigma_fingerprint: str

    @property
    def n_parameters(self) -> int:
        return int(self.cell_areas.size)

    @property
    def n_measurements(self) -> int:
        return int(sum(self.n_meas_per_stim))

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_measurements, self.n_parameters

    def assert_compatible(self, sigma_fingerprint: str | None) -> None:
        """Raise if the stored gradients predate a new conductivity value.

        The check is permissive (V9): it only fires when both the stored
        and provided fingerprints are non-empty and differ. An empty
        stored fingerprint (legacy construction) or an empty
        ``sigma_fingerprint`` argument skips the guard. The error
        message names the concrete subclass so callers can tell the
        eager and lazy paths apart in stack traces.
        """
        stored = str(self.sigma_fingerprint or "")
        provided = str(sigma_fingerprint or "")
        if not stored or not provided:
            return
        if stored != provided:
            raise ValueError(
                f"{type(self).__name__} sigma fingerprint mismatch: "
                f"stored={stored[:12]}..., provided={provided[:12]}..."
            )

    # ------------------------------------------------------------------
    # Abstract operations: subclasses MUST override.
    # ------------------------------------------------------------------

    def _validate_shapes(self) -> None:
        raise NotImplementedError

    def matvec(self, vector: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def rmatvec(self, residual: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def hessian_diag(self, **kwargs) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers built on top of ``matvec`` / ``rmatvec``.
    # ------------------------------------------------------------------

    def as_linear_operator(self) -> LinearOperator:
        """Return a SciPy ``LinearOperator`` wrapping ``J``."""
        return LinearOperator(
            self.shape,
            matvec=self.matvec,
            rmatvec=self.rmatvec,
            dtype=getattr(self, "dtype", np.float64),
        )

    def normal_matvec(
        self,
        vector: np.ndarray,
        *,
        measurement_weights: np.ndarray | None = None,
        alpha: float = 0.0,
        regularization: RegularizationAction | None = None,
    ) -> np.ndarray:
        """Apply ``J^H W J v + alpha R v`` without dense ``J`` or ``H``."""
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
        return np.asarray(out)

    def as_normal_operator(
        self,
        *,
        measurement_weights: np.ndarray | None = None,
        alpha: float = 0.0,
        regularization: RegularizationAction | None = None,
    ) -> LinearOperator:
        """Return ``J^H W J + alpha R`` as a SciPy ``LinearOperator``."""
        return LinearOperator(
            (self.n_parameters, self.n_parameters),
            matvec=lambda v: self.normal_matvec(
                v,
                measurement_weights=measurement_weights,
                alpha=alpha,
                regularization=regularization,
            ),
            dtype=getattr(self, "dtype", np.float64),
        )

    @staticmethod
    def _apply_regularization(
        regularization: RegularizationAction,
        vector: np.ndarray,
    ) -> np.ndarray:
        vec = np.asarray(vector)
        dtype = _complex_preserving_dtype(vec.dtype)
        vec = np.asarray(vec, dtype=dtype)
        if isinstance(regularization, LinearOperator):
            return np.asarray(regularization.matvec(vec), dtype=dtype)
        if callable(regularization):
            return np.asarray(regularization(vec), dtype=dtype)
        if isspmatrix(regularization):
            return np.asarray(regularization.dot(vec), dtype=dtype)
        matrix_raw = np.asarray(regularization)
        matrix = np.asarray(matrix_raw, dtype=np.result_type(dtype, matrix_raw.dtype))
        return np.asarray(matrix @ vec, dtype=np.result_type(matrix.dtype, dtype))

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


@dataclass
class JacobianLinearization(_LinearizationBase):
    """Apply EIT Jacobian actions without materializing the dense Jacobian.

    The object stores forward and adjoint field gradients for one linearization
    point and exposes ``Jv`` and ``J^H r`` operations. Existing dense workflows
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
        self.dtype = _complex_preserving_dtype(
            *self.grad_u_all,
            *self.adjoint_gradients,
        )
        self.cell_areas = np.asarray(self.cell_areas, dtype=np.float64)
        self.grad_u_all = tuple(
            np.asarray(g, dtype=self.dtype) for g in self.grad_u_all
        )
        self.adjoint_gradients = tuple(
            np.asarray(g, dtype=self.dtype) for g in self.adjoint_gradients
        )
        self.n_meas_per_stim = tuple(int(v) for v in self.n_meas_per_stim)
        self.sign = float(self.sign)
        self.sigma_fingerprint = str(self.sigma_fingerprint or "")
        self._validate_shapes()
        blocks: list[np.ndarray] = []
        meas_idx = 0
        for n_meas in self.n_meas_per_stim:
            blocks.append(
                _stack_adjoint_gradient_block(
                    self.adjoint_gradients,
                    start=meas_idx,
                    count=n_meas,
                    dtype=self.dtype,
                )
            )
            meas_idx += n_meas
        self._adjoint_blocks: tuple[np.ndarray, ...] = tuple(blocks)

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
        dtype = _complex_preserving_dtype(self.dtype, vector)
        vec = np.asarray(vector, dtype=dtype).reshape(-1)
        if vec.size != self.n_parameters:
            raise ValueError(
                f"Expected vector length {self.n_parameters}, got {vec.size}."
            )

        out = np.zeros(self.n_measurements, dtype=dtype)
        weighted_cell = self.cell_areas * vec
        meas_idx = 0
        for stim_idx, grad_u in enumerate(self.grad_u_all):
            n_meas = self.n_meas_per_stim[stim_idx]
            adjoint_block = self._adjoint_blocks[stim_idx]
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
        """Apply ``J^H r`` for complex data, reducing to ``J^T r`` for real data."""
        dtype = _complex_preserving_dtype(self.dtype, residual)
        res = np.asarray(residual, dtype=dtype).reshape(-1)
        if res.size != self.n_measurements:
            raise ValueError(
                f"Expected residual length {self.n_measurements}, got {res.size}."
            )

        out = np.zeros(self.n_parameters, dtype=dtype)
        meas_idx = 0
        for stim_idx, grad_u in enumerate(self.grad_u_all):
            n_meas = self.n_meas_per_stim[stim_idx]
            adjoint_block = self._adjoint_blocks[stim_idx]
            weighted_adjoint = np.einsum(
                "m,meg->eg",
                res[meas_idx : meas_idx + n_meas],
                np.conj(adjoint_block),
                optimize=True,
            )
            out += self.sign * np.einsum(
                "eg,eg,e->e",
                np.conj(grad_u),
                weighted_adjoint,
                self.cell_areas,
                optimize=True,
            )
            meas_idx += n_meas
        return out

    def to_dense(self, *, block_size: int | None = None) -> np.ndarray:
        """Materialize the dense Jacobian for compatibility or debugging."""
        n_meas, n_param = self.shape
        dense = np.zeros((n_meas, n_param), dtype=self.dtype)
        block = n_param if block_size is None else max(1, int(block_size))

        meas_idx = 0
        for stim_idx, grad_u in enumerate(self.grad_u_all):
            n_meas_this = self.n_meas_per_stim[stim_idx]
            adjoint_block = self._adjoint_blocks[stim_idx]
            for start in range(0, n_param, block):
                end = min(start + block, n_param)
                block_view = dense[meas_idx : meas_idx + n_meas_this, start:end]
                np.einsum(
                    "eg,meg->me",
                    grad_u[start:end, :],
                    adjoint_block[:, start:end, :],
                    optimize=True,
                    out=block_view,
                )
                block_view *= self.sign
                block_view *= self.cell_areas[start:end]
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
        """Return ``diag(J^H W J) [+ alpha * R_diag]`` without dense ``J``.

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
            adjoint_block = self._adjoint_blocks[stim_idx]
            # Per (measurement m, element e) sensitivity before cell_area scaling.
            contrib = np.einsum("eg,meg->me", grad_u, adjoint_block, optimize=True)
            contrib_sq = np.real(np.conj(contrib) * contrib)
            if weights is not None:
                contrib_sq = contrib_sq * weights[meas_idx : meas_idx + n_meas, None]
            diag += contrib_sq.sum(axis=0)
            meas_idx += n_meas
        diag = diag * (float(self.sign) ** 2) * (self.cell_areas**2)

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


@dataclass
class LazyAdjointJacobianLinearization(_LinearizationBase):
    """Lazy EIT Jacobian action without per-measurement adjoint storage."""

    fwd_model: Any
    sigma_values: np.ndarray
    u_all: tuple[np.ndarray, ...]
    grad_u_all: tuple[np.ndarray, ...]
    cell_areas: np.ndarray
    n_meas_per_stim: tuple[int, ...]
    meas_matrices: tuple[np.ndarray, ...]
    gradient_callback: GradientCallback
    sign: float = -1.0
    sigma_fingerprint: str = ""
    diag_exact_max_measurements: int = 512
    diag_chunk_size: int = 128

    def __post_init__(self) -> None:
        self.dtype = _complex_preserving_dtype(
            self.sigma_values,
            *self.u_all,
            *self.grad_u_all,
        )
        self.sigma_values = np.ascontiguousarray(
            np.asarray(self.sigma_values, dtype=self.dtype).reshape(-1)
        )
        self.u_all = tuple(np.asarray(u, dtype=self.dtype) for u in self.u_all)
        self.grad_u_all = tuple(
            np.asarray(g, dtype=self.dtype) for g in self.grad_u_all
        )
        self.cell_areas = np.asarray(self.cell_areas, dtype=np.float64)
        self.n_meas_per_stim = tuple(int(v) for v in self.n_meas_per_stim)
        self.meas_matrices = tuple(np.asarray(matrix) for matrix in self.meas_matrices)
        self.sign = float(self.sign)
        self.sigma_fingerprint = str(
            self.sigma_fingerprint or compute_sigma_fingerprint(self.sigma_values)
        )
        self.last_action_info: dict[str, Any] = {}
        self.last_diag_info: dict[str, Any] = {}
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        if len(self.u_all) != len(self.n_meas_per_stim):
            raise ValueError("u_all length must match n_meas_per_stim.")
        if len(self.grad_u_all) != len(self.n_meas_per_stim):
            raise ValueError("grad_u_all length must match n_meas_per_stim.")
        if len(self.meas_matrices) != len(self.n_meas_per_stim):
            raise ValueError("meas_matrices length must match n_meas_per_stim.")
        for grad_u in self.grad_u_all:
            if grad_u.ndim != 2 or grad_u.shape[0] != self.n_parameters:
                raise ValueError(
                    "Each forward gradient must have shape (n_elem, gdim)."
                )

    def _make_sigma(self):
        from dolfinx import fem

        sigma = fem.Function(self.fwd_model.V_sigma)
        sigma.x.array[:] = self.sigma_values
        return sigma

    def _gradients_for_patterns(
        self,
        patterns: np.ndarray,
        *,
        rhs_kind: str,
    ) -> tuple[np.ndarray, ...]:
        sigma = self._make_sigma()
        fields, _ = self.fwd_model.forward_solve(sigma, patterns)
        gradients = tuple(
            np.asarray(grad, dtype=_complex_preserving_dtype(grad, self.dtype))
            for grad in self.gradient_callback(fields)
        )
        self.last_action_info = {
            "action": rhs_kind,
            "solve_count": int(np.asarray(patterns).shape[0]),
            "mode": "lazy_adjoint",
        }
        return gradients

    def _assemble_sensitivity_rhs(self, vector: np.ndarray) -> np.ndarray:
        from dolfinx import fem
        import dolfinx.fem.petsc as fem_petsc
        import ufl

        vec = np.asarray(
            vector,
            dtype=_complex_preserving_dtype(vector, self.dtype),
        ).reshape(-1)
        if vec.size != self.n_parameters:
            raise ValueError(
                f"Expected vector length {self.n_parameters}, got {vec.size}."
            )

        delta_sigma = fem.Function(self.fwd_model.V_sigma)
        delta_sigma.x.array[:] = vec
        full_size = self.fwd_model.dofs + self.fwd_model.n_elec + 1
        rhs = np.zeros((full_size, len(self.u_all)), dtype=vec.dtype)
        for stim_idx, u_values in enumerate(self.u_all):
            u_fun = fem.Function(self.fwd_model.V)
            u_fun.x.array[:] = np.asarray(u_values, dtype=self.dtype)
            form = (
                -ufl.inner(
                    delta_sigma * ufl.grad(u_fun),
                    ufl.grad(self.fwd_model.phi),
                )
                * ufl.dx
            )
            vec_petsc = fem_petsc.assemble_vector(fem.form(form))
            vec_petsc.assemble()
            rhs[: self.fwd_model.dofs, stim_idx] = np.asarray(vec_petsc.array)
            destroy = getattr(vec_petsc, "destroy", None)
            if callable(destroy):
                try:
                    destroy()
                except Exception:
                    pass
        return rhs

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        rhs = self._assemble_sensitivity_rhs(vector)
        sigma = self._make_sigma()
        sol = self.fwd_model.solve_full_rhs(
            sigma,
            rhs,
            rhs_kind="sensitivity_jv",
        )
        electrode_delta = np.asarray(
            sol[self.fwd_model.dofs : self.fwd_model.dofs + self.fwd_model.n_elec, :].T,
        )
        out_dtype = _complex_preserving_dtype(electrode_delta, *self.meas_matrices)
        out = np.zeros(self.n_measurements, dtype=out_dtype)
        meas_idx = 0
        for stim_idx, meas_matrix in enumerate(self.meas_matrices):
            n_meas = self.n_meas_per_stim[stim_idx]
            out[meas_idx : meas_idx + n_meas] = meas_matrix @ electrode_delta[stim_idx]
            meas_idx += n_meas
        self.last_action_info = {
            "action": "matvec",
            "solve_count": int(len(self.u_all)),
            "mode": "lazy_sensitivity",
        }
        return np.asarray(out, dtype=out_dtype)

    def rmatvec(self, residual: np.ndarray) -> np.ndarray:
        res = np.asarray(
            residual,
            dtype=_complex_preserving_dtype(residual, self.dtype),
        ).reshape(-1)
        if res.size != self.n_measurements:
            raise ValueError(
                f"Expected residual length {self.n_measurements}, got {res.size}."
            )
        pattern_dtype = _complex_preserving_dtype(res, *self.meas_matrices)
        patterns = np.zeros(
            (len(self.n_meas_per_stim), self.fwd_model.n_elec),
            dtype=pattern_dtype,
        )
        meas_idx = 0
        for stim_idx, meas_matrix in enumerate(self.meas_matrices):
            n_meas = self.n_meas_per_stim[stim_idx]
            patterns[stim_idx, :] = meas_matrix.T @ np.conj(
                res[meas_idx : meas_idx + n_meas]
            )
            meas_idx += n_meas
        adjoint_gradients = self._gradients_for_patterns(
            patterns,
            rhs_kind="adjoint_jtr",
        )
        out_dtype = _complex_preserving_dtype(res, *self.grad_u_all, *adjoint_gradients)
        out = np.zeros(self.n_parameters, dtype=out_dtype)
        for stim_idx, grad_u in enumerate(self.grad_u_all):
            out += self.sign * np.einsum(
                "eg,eg,e->e",
                np.conj(grad_u),
                np.conj(adjoint_gradients[stim_idx]),
                self.cell_areas,
                optimize=True,
            )
        self.last_action_info = {
            "action": "rmatvec",
            "solve_count": int(len(self.n_meas_per_stim)),
            "mode": "lazy_adjoint_combined",
        }
        return np.asarray(out, dtype=out_dtype)

    def hessian_diag(
        self,
        *,
        measurement_weights: np.ndarray | None = None,
        alpha: float = 0.0,
        regularization_diag: np.ndarray | None = None,
        floor: float = 0.0,
        diag_mode: str | None = None,
        diag_batch_max_measurements: int | None = None,
    ) -> np.ndarray:
        mode_raw = str(diag_mode or "auto").strip().lower().replace("_", "-")
        if mode_raw in {"", "auto"}:
            mode_raw = (
                "approx"
                if self.n_measurements > int(self.diag_exact_max_measurements)
                else "exact"
            )

        diag_info: dict[str, Any] = {
            "n_measurements": int(self.n_measurements),
            "exact_max_measurements": int(self.diag_exact_max_measurements),
        }
        if mode_raw in {"approx", "lazy-approx"}:
            diag = self._approximate_hessian_diag(measurement_weights)
            mode = "lazy_approx"
            diag_info.update({"sampled_measurements": 0, "solve_count": 0})
        elif mode_raw in {"coarse", "coarse-hessian", "coarse-diag"}:
            diag, sampled_info = self._sampled_hessian_diag(
                measurement_weights,
                max_measurements=diag_batch_max_measurements or 128,
            )
            mode = "lazy_coarse_hessian"
            diag_info.update(sampled_info)
        elif mode_raw in {"batch", "batch-noser", "sampled-noser", "noser"}:
            diag, sampled_info = self._sampled_hessian_diag(
                measurement_weights,
                max_measurements=(
                    diag_batch_max_measurements or self.diag_exact_max_measurements
                ),
            )
            mode = "lazy_batch_noser"
            diag_info.update(sampled_info)
        elif mode_raw in {"exact", "chunked", "chunked-exact"}:
            diag = self._exact_hessian_diag_chunked(measurement_weights)
            mode = "lazy_chunked_exact"
            diag_info.update(
                {
                    "sampled_measurements": int(self.n_measurements),
                    "solve_count": int(self.n_measurements),
                }
            )
        else:
            raise ValueError(
                "diag_mode must be auto|approx|batch_noser|coarse|exact, "
                f"got {diag_mode!r}."
            )

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
        self.last_diag_info = {"mode": mode, **diag_info}
        return np.asarray(diag, dtype=np.float64)

    def _validated_measurement_weights(
        self,
        measurement_weights: np.ndarray | None,
    ) -> np.ndarray:
        weights = (
            np.ones(self.n_measurements, dtype=np.float64)
            if measurement_weights is None
            else np.asarray(measurement_weights, dtype=np.float64).reshape(-1)
        )
        if weights.size != self.n_measurements:
            raise ValueError(
                f"Expected {self.n_measurements} weights, got {weights.size}."
            )
        return weights

    def _approximate_hessian_diag(
        self,
        measurement_weights: np.ndarray | None,
    ) -> np.ndarray:
        weights = self._validated_measurement_weights(measurement_weights)
        diag = np.zeros(self.n_parameters, dtype=np.float64)
        meas_idx = 0
        for stim_idx, grad_u in enumerate(self.grad_u_all):
            n_meas = self.n_meas_per_stim[stim_idx]
            block_weight = float(np.mean(np.abs(weights[meas_idx : meas_idx + n_meas])))
            diag += (
                block_weight
                * np.einsum(
                    "eg,eg,e->e",
                    np.conj(grad_u),
                    grad_u,
                    self.cell_areas * self.cell_areas,
                    optimize=True,
                ).real
            )
            meas_idx += n_meas
        return np.maximum(diag, 1e-100)

    def _exact_hessian_diag_chunked(
        self,
        measurement_weights: np.ndarray | None,
    ) -> np.ndarray:
        weights = self._validated_measurement_weights(measurement_weights)
        diag = np.zeros(self.n_parameters, dtype=np.float64)
        meas_idx = 0
        chunk = max(1, int(self.diag_chunk_size))
        cell_area_sq = self.cell_areas * self.cell_areas
        for stim_idx, grad_u in enumerate(self.grad_u_all):
            n_meas = self.n_meas_per_stim[stim_idx]
            meas_matrix = self.meas_matrices[stim_idx]
            for start in range(0, n_meas, chunk):
                end = min(start + chunk, n_meas)
                adjoint_gradients = self._gradients_for_patterns(
                    meas_matrix[start:end, :],
                    rhs_kind="adjoint_diag",
                )
                adjoint_block = np.asarray(adjoint_gradients)
                contrib = np.einsum(
                    "eg,meg->me",
                    grad_u,
                    adjoint_block,
                    optimize=True,
                )
                block_weights = weights[meas_idx + start : meas_idx + end]
                diag += _weighted_contrib_power_sum(
                    contrib,
                    block_weights,
                    cell_area_sq,
                )
            meas_idx += n_meas
        return np.maximum(diag, 1e-100)

    def _sampled_hessian_diag(
        self,
        measurement_weights: np.ndarray | None,
        *,
        max_measurements: int,
    ) -> tuple[np.ndarray, dict[str, int]]:
        weights = self._validated_measurement_weights(measurement_weights)
        max_samples = max(1, min(int(max_measurements), int(self.n_measurements)))
        if max_samples >= self.n_measurements:
            return self._exact_hessian_diag_chunked(weights), {
                "sampled_measurements": int(self.n_measurements),
                "solve_count": int(self.n_measurements),
            }

        diag = np.zeros(self.n_parameters, dtype=np.float64)
        sampled_total = 0
        solve_total = 0
        meas_idx = 0
        chunk = max(1, int(self.diag_chunk_size))
        cell_area_sq = self.cell_areas * self.cell_areas
        for stim_idx, grad_u in enumerate(self.grad_u_all):
            n_meas = int(self.n_meas_per_stim[stim_idx])
            meas_matrix = self.meas_matrices[stim_idx]
            block_weights = weights[meas_idx : meas_idx + n_meas]
            sample_count = max(
                1,
                min(
                    n_meas,
                    int(np.ceil(max_samples * n_meas / max(self.n_measurements, 1))),
                ),
            )
            local_idx = np.unique(
                np.linspace(0, n_meas - 1, sample_count, dtype=np.int64)
            )
            sampled_total += int(local_idx.size)
            sampled_weights = block_weights[local_idx]
            sampled_weight_sum = float(np.sum(np.abs(sampled_weights)))
            block_weight_sum = float(np.sum(np.abs(block_weights)))
            if sampled_weight_sum > 0.0:
                scale = block_weight_sum / sampled_weight_sum
            else:
                scale = float(n_meas) / max(float(local_idx.size), 1.0)

            for start in range(0, int(local_idx.size), chunk):
                selected = local_idx[start : start + chunk]
                adjoint_gradients = self._gradients_for_patterns(
                    meas_matrix[selected, :],
                    rhs_kind="adjoint_diag_sampled",
                )
                solve_total += int(selected.size)
                adjoint_block = np.asarray(adjoint_gradients)
                contrib = np.einsum(
                    "eg,meg->me",
                    grad_u,
                    adjoint_block,
                    optimize=True,
                )
                selected_weights = block_weights[selected]
                diag += scale * _weighted_contrib_power_sum(
                    contrib,
                    selected_weights,
                    cell_area_sq,
                )
            meas_idx += n_meas
        return np.maximum(diag, 1e-100), {
            "sampled_measurements": int(sampled_total),
            "solve_count": int(solve_total),
        }


class _PETScJacobianContext:
    """petsc4py context object for ``Mat.Type.PYTHON``."""

    def __init__(self, linearization):
        self.linearization = linearization

    @staticmethod
    def _vec_array(vec) -> np.ndarray:
        if hasattr(vec, "array"):
            return np.asarray(vec.array)
        return np.asarray(vec.getArray(readonly=True))

    def mult(self, _mat, x, y) -> None:
        result = self.linearization.matvec(self._vec_array(x))
        y_arr = y.getArray(readonly=False)
        y_arr[:] = result

    def multTranspose(self, _mat, x, y) -> None:
        result = self.linearization.rmatvec(self._vec_array(x))
        y_arr = y.getArray(readonly=False)
        y_arr[:] = result
