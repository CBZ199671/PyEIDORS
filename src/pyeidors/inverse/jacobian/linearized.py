"""Matrix-free Jacobian actions for EIT sensitivity linearizations."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import isspmatrix
from scipy.sparse.linalg import LinearOperator

RegularizationAction = Callable[[np.ndarray], np.ndarray] | LinearOperator | np.ndarray
GradientCallback = Callable[[Sequence[np.ndarray]], Sequence[np.ndarray]]


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


@dataclass
class LazyAdjointJacobianLinearization:
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
        self.sigma_values = np.ascontiguousarray(
            np.asarray(self.sigma_values, dtype=np.float64).reshape(-1)
        )
        self.u_all = tuple(np.asarray(u, dtype=np.float64) for u in self.u_all)
        self.grad_u_all = tuple(
            np.asarray(g, dtype=np.float64) for g in self.grad_u_all
        )
        self.cell_areas = np.asarray(self.cell_areas, dtype=np.float64)
        self.n_meas_per_stim = tuple(int(v) for v in self.n_meas_per_stim)
        self.meas_matrices = tuple(
            np.asarray(matrix, dtype=np.float64) for matrix in self.meas_matrices
        )
        self.sign = float(self.sign)
        self.sigma_fingerprint = str(
            self.sigma_fingerprint or compute_sigma_fingerprint(self.sigma_values)
        )
        self.last_action_info: dict[str, Any] = {}
        self.last_diag_info: dict[str, Any] = {}
        self._validate_shapes()

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

    def assert_compatible(self, sigma_fingerprint: str | None) -> None:
        stored = str(self.sigma_fingerprint or "")
        provided = str(sigma_fingerprint or "")
        if stored and provided and stored != provided:
            raise ValueError(
                "LazyAdjointJacobianLinearization sigma fingerprint mismatch: "
                f"stored={stored[:12]}..., provided={provided[:12]}..."
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
            np.asarray(grad, dtype=np.float64)
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

        vec = np.asarray(vector, dtype=np.float64).reshape(-1)
        if vec.size != self.n_parameters:
            raise ValueError(
                f"Expected vector length {self.n_parameters}, got {vec.size}."
            )

        delta_sigma = fem.Function(self.fwd_model.V_sigma)
        delta_sigma.x.array[:] = vec
        full_size = self.fwd_model.dofs + self.fwd_model.n_elec + 1
        rhs = np.zeros((full_size, len(self.u_all)), dtype=np.float64)
        for stim_idx, u_values in enumerate(self.u_all):
            u_fun = fem.Function(self.fwd_model.V)
            u_fun.x.array[:] = np.asarray(u_values, dtype=np.float64)
            form = (
                -ufl.inner(
                    delta_sigma * ufl.grad(u_fun),
                    ufl.grad(self.fwd_model.phi),
                )
                * ufl.dx
            )
            vec_petsc = fem_petsc.assemble_vector(fem.form(form))
            vec_petsc.assemble()
            rhs[: self.fwd_model.dofs, stim_idx] = np.asarray(
                vec_petsc.array, dtype=np.float64
            )
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
            dtype=np.float64,
        )
        out = np.zeros(self.n_measurements, dtype=np.float64)
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
        return np.asarray(out, dtype=np.float64)

    def rmatvec(self, residual: np.ndarray) -> np.ndarray:
        res = np.asarray(residual, dtype=np.float64).reshape(-1)
        if res.size != self.n_measurements:
            raise ValueError(
                f"Expected residual length {self.n_measurements}, got {res.size}."
            )
        patterns = np.zeros(
            (len(self.n_meas_per_stim), self.fwd_model.n_elec),
            dtype=np.float64,
        )
        meas_idx = 0
        for stim_idx, meas_matrix in enumerate(self.meas_matrices):
            n_meas = self.n_meas_per_stim[stim_idx]
            patterns[stim_idx, :] = meas_matrix.T @ res[meas_idx : meas_idx + n_meas]
            meas_idx += n_meas
        adjoint_gradients = self._gradients_for_patterns(
            patterns,
            rhs_kind="adjoint_jtr",
        )
        out = np.zeros(self.n_parameters, dtype=np.float64)
        for stim_idx, grad_u in enumerate(self.grad_u_all):
            out += self.sign * np.einsum(
                "eg,eg,e->e",
                grad_u,
                adjoint_gradients[stim_idx],
                self.cell_areas,
                optimize=True,
            )
        self.last_action_info = {
            "action": "rmatvec",
            "solve_count": int(len(self.n_meas_per_stim)),
            "mode": "lazy_adjoint_combined",
        }
        return np.asarray(out, dtype=np.float64)

    def as_linear_operator(self) -> LinearOperator:
        return LinearOperator(
            self.shape,
            matvec=self.matvec,
            rmatvec=self.rmatvec,
            dtype=np.float64,
        )

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
            block_weight = float(
                np.mean(np.abs(weights[meas_idx : meas_idx + n_meas]))
            )
            diag += block_weight * np.einsum(
                "eg,eg,e->e",
                grad_u,
                grad_u,
                self.cell_areas * self.cell_areas,
                optimize=True,
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
        for stim_idx, grad_u in enumerate(self.grad_u_all):
            n_meas = self.n_meas_per_stim[stim_idx]
            meas_matrix = self.meas_matrices[stim_idx]
            for start in range(0, n_meas, chunk):
                end = min(start + chunk, n_meas)
                adjoint_gradients = self._gradients_for_patterns(
                    meas_matrix[start:end, :],
                    rhs_kind="adjoint_diag",
                )
                adjoint_block = np.asarray(adjoint_gradients, dtype=np.float64)
                contrib = np.einsum(
                    "eg,meg->me",
                    grad_u,
                    adjoint_block,
                    optimize=True,
                )
                block_weights = weights[meas_idx + start : meas_idx + end]
                diag += (
                    contrib
                    * contrib
                    * block_weights[:, None]
                    * (self.cell_areas[None, :] ** 2)
                ).sum(axis=0)
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
                adjoint_block = np.asarray(adjoint_gradients, dtype=np.float64)
                contrib = np.einsum(
                    "eg,meg->me",
                    grad_u,
                    adjoint_block,
                    optimize=True,
                )
                selected_weights = block_weights[selected]
                diag += scale * (
                    contrib
                    * contrib
                    * selected_weights[:, None]
                    * (self.cell_areas[None, :] ** 2)
                ).sum(axis=0)
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
