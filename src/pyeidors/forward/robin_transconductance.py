"""Complete Electrode Model through local Robin transconductance solves.

This module implements the formulation described by Deakin, Adler, and
Lionheart (EIT 2026).  It is the Schur-complement form of the classic CEM:
the body potential is solved with voltage-driven local Robin boundary
conditions, the resulting electrode transconductance is assembled, and the
requested balanced currents are inverted on the zero-sum electrode space.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import warnings

import numpy as np
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

try:  # pragma: no cover - available in the Nix runtime, optional in lean tests
    from petsc4py import PETSc
except ImportError:  # pragma: no cover
    PETSc = None

from ..utils.numeric_ops import all_finite_values
from .eit_forward_model import EITForwardModel


CLASSIC_CEM = "classic"
ROBIN_TRANSCONDUCTANCE_CEM = "robin_transconductance"
_VALID_CEM_FORMULATIONS = frozenset({CLASSIC_CEM, ROBIN_TRANSCONDUCTANCE_CEM})


def normalize_cem_formulation(value: object) -> str:
    """Normalize the public CEM formulation selector."""

    token = str(value if value is not None else CLASSIC_CEM).strip().lower()
    aliases = {
        "": CLASSIC_CEM,
        "classic_cem": CLASSIC_CEM,
        "robin": ROBIN_TRANSCONDUCTANCE_CEM,
        "transconductance": ROBIN_TRANSCONDUCTANCE_CEM,
        "robin-transconductance": ROBIN_TRANSCONDUCTANCE_CEM,
    }
    token = aliases.get(token, token)
    if token not in _VALID_CEM_FORMULATIONS:
        choices = ", ".join(sorted(_VALID_CEM_FORMULATIONS))
        raise ValueError(
            f"Unknown cem_formulation {value!r}; expected one of: {choices}"
        )
    return token


def zero_sum_helmert_basis(n_elec: int, dtype=np.float64) -> np.ndarray:
    """Return a deterministic orthonormal basis for ``sum(U) == 0``.

    Column ``k`` has equal positive entries in rows ``0..k-1`` and one
    balancing negative entry in row ``k``.  This Helmert construction avoids
    selecting a privileged ground electrode while keeping the matrix exactly
    reproducible across runs.
    """

    count = int(n_elec)
    if count < 2:
        raise ValueError("Robin transconductance CEM requires at least 2 electrodes")
    scalar_dtype = np.dtype(dtype)
    real_dtype = np.empty(1, dtype=scalar_dtype).real.dtype
    basis_real = np.zeros((count, count - 1), dtype=real_dtype)
    for column in range(1, count):
        scale = np.sqrt(float(column * (column + 1)))
        basis_real[:column, column - 1] = 1.0 / scale
        basis_real[column, column - 1] = -float(column) / scale
    return np.asarray(basis_real, dtype=scalar_dtype)


def _relative_residual(lhs: np.ndarray, rhs: np.ndarray) -> float:
    delta = np.asarray(lhs) - np.asarray(rhs)
    denominator = max(float(np.linalg.norm(rhs)), 1.0)
    return float(np.linalg.norm(delta) / denominator)


def _balance_tolerance(values: np.ndarray) -> float:
    arr = np.asarray(values)
    real_dtype = np.empty(1, dtype=arr.dtype).real.dtype
    eps = float(np.finfo(real_dtype).eps)
    magnitude = max(float(np.max(np.abs(arr), initial=0.0)), 1.0)
    return float(64.0 * eps * max(1, arr.shape[-1]) * magnitude)


def _condition_limit(dtype: np.dtype) -> float:
    real_dtype = np.empty(1, dtype=np.dtype(dtype)).real.dtype
    return float(1.0 / np.sqrt(np.finfo(real_dtype).eps))


@dataclass(frozen=True)
class RobinTransconductanceState:
    """Conductivity-specific response basis and reduced transconductance."""

    key: str
    response_basis: np.ndarray
    electrode_basis: np.ndarray
    reduced_map: np.ndarray
    reduced_factor: tuple[np.ndarray, np.ndarray]
    rank: int
    condition_number: float
    response_residual: float
    symmetry_residual: float
    backend: str
    setup_count: int
    solve_count: int


class RobinTransconductanceForwardModel(EITForwardModel):
    """CEM forward model using Robin solves and transconductance inversion."""

    cem_formulation = ROBIN_TRANSCONDUCTANCE_CEM

    def _robin_blocks(
        self,
    ) -> tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
        full = self._ensure_electrode_matrix().tocsr()
        electrode_start = self.dofs
        electrode_stop = self.dofs + self.n_elec
        robin_boundary = full[: self.dofs, : self.dofs].tocsr()
        coupling = full[: self.dofs, electrode_start:electrode_stop].tocsr()
        electrode_diag = np.asarray(
            full[
                electrode_start:electrode_stop,
                electrode_start:electrode_stop,
            ].toarray(),
            dtype=self._active_scalar_dtype(),
        )
        basis = zero_sum_helmert_basis(
            self.n_elec,
            dtype=self._active_scalar_dtype(),
        )
        coupling_basis = np.asarray(
            coupling @ basis,
            dtype=self._active_scalar_dtype(),
        )
        return robin_boundary, electrode_diag, basis, coupling_basis

    def _solve_robin_response_scipy(
        self,
        sigma,
        robin_boundary: csr_matrix,
        coupling_basis: np.ndarray,
    ) -> tuple[np.ndarray, float, str, int, int]:
        conductivity_matrix = self._assemble_conductivity_matrix(sigma)
        try:
            conductivity = self._petsc_to_csr(conductivity_matrix).astype(
                self._active_scalar_dtype(), copy=False
            )
        finally:
            destroy = getattr(conductivity_matrix, "destroy", None)
            if callable(destroy):
                destroy()
        robin_matrix = (conductivity + robin_boundary).tocsc()
        factor = splu(robin_matrix)
        response = self._as_scalar_array(
            factor.solve(coupling_basis),
            name="Robin response basis",
        )
        residual = _relative_residual(robin_matrix @ response, coupling_basis)
        return response, residual, "scipy-splu", 1, int(coupling_basis.shape[1])

    def _solve_robin_response_petsc(
        self,
        sigma,
        robin_boundary: csr_matrix,
        coupling_basis: np.ndarray,
    ) -> tuple[np.ndarray, float, str, int, int]:
        if PETSc is None:
            raise RuntimeError("petsc4py is required for Robin PETSc solves")

        conductivity = self._assemble_conductivity_matrix(sigma, mat_kind=None)
        residual_matrix = (
            self._petsc_to_csr(conductivity).astype(
                self._active_scalar_dtype(), copy=False
            )
            + robin_boundary
        ).tocsr()
        robin_matrix = self._csr_to_petsc(residual_matrix)
        requested_mat_type = self._get_requested_petsc_mat_type()
        robin_matrix = self._ensure_mat_type(robin_matrix, requested_mat_type)
        if hasattr(robin_matrix, "assemble"):
            robin_matrix.assemble()

        bundle = None
        b = None
        x = None
        try:
            bundle = self._make_petsc_solver_bundle(robin_matrix)
            solve_matrix = bundle.get("solve_A", robin_matrix)
            ksp = bundle["ksp"]
            requested_vec_type = self._get_requested_petsc_vec_type()
            b = self._ensure_vec_type(
                solve_matrix.createVecRight(),
                requested_vec_type,
            )
            x = self._ensure_vec_type(
                solve_matrix.createVecRight(),
                requested_vec_type,
            )
            response = np.zeros_like(coupling_basis)
            b_array = b.getArray(readonly=False)
            iterations = 0
            real_dtype = np.empty(1, dtype=self._active_scalar_dtype()).real.dtype
            verified_negative_limit = max(
                10.0 * float(self.backend_config.rtol),
                10.0 * float(self.backend_config.atol),
                8.0 * float(np.finfo(real_dtype).eps),
            )
            verified_negative_reasons: list[dict[str, float | int]] = []
            for column in range(coupling_basis.shape[1]):
                b_array[:] = coupling_basis[:, column]
                ksp.solve(b, x)
                reason = int(ksp.getConvergedReason())
                candidate = np.asarray(
                    x.getArray(readonly=True),
                    dtype=self._active_scalar_dtype(),
                )
                true_residual = _relative_residual(
                    residual_matrix @ candidate,
                    coupling_basis[:, column],
                )
                if reason < 0:
                    iterations_done = (
                        int(ksp.getIterationNumber())
                        if hasattr(ksp, "getIterationNumber")
                        else -1
                    )
                    reported_residual = (
                        float(ksp.getResidualNorm())
                        if hasattr(ksp, "getResidualNorm")
                        else float("nan")
                    )
                    if true_residual <= verified_negative_limit:
                        verified_negative_reasons.append(
                            {
                                "column": int(column),
                                "reason": int(reason),
                                "iterations": int(iterations_done),
                                "reported_residual": reported_residual,
                                "true_relative_residual": true_residual,
                            }
                        )
                    else:
                        raise RuntimeError(
                            "Robin PETSc response solve failed with convergence "
                            f"reason {reason} for basis column {column}; "
                            f"iterations={iterations_done}, "
                            f"reported_residual={reported_residual:.6e}, "
                            f"true_relative_residual={true_residual:.6e}, "
                            f"verified_limit={verified_negative_limit:.6e}"
                        )
                response[:, column] = candidate
                if hasattr(ksp, "getIterationNumber"):
                    iterations += int(ksp.getIterationNumber())
            response = self._as_scalar_array(
                response,
                name="Robin PETSc response basis",
            )
            residual = _relative_residual(
                residual_matrix @ response,
                coupling_basis,
            )
            self._set_backend_diagnostic(
                robin_petsc_verified_negative_reasons=verified_negative_reasons,
                robin_petsc_verified_negative_limit=verified_negative_limit,
            )
            return (
                response,
                residual,
                str(bundle.get("backend", "petsc-ksp")),
                int(bundle.get("ksp_setup_count", 1) or 1),
                int(coupling_basis.shape[1]),
            )
        finally:
            objects = [b, x, conductivity]
            if bundle is not None:
                objects.extend(
                    [
                        bundle.get("ksp"),
                        bundle.get("solve_A"),
                        bundle.get("A"),
                    ]
                )
            seen: set[int] = set()
            for obj in objects:
                if obj is None or id(obj) in seen:
                    continue
                seen.add(id(obj))
                destroy = getattr(obj, "destroy", None)
                if callable(destroy):
                    try:
                        destroy()
                    except Exception:
                        pass

    def _build_robin_state(self, sigma) -> RobinTransconductanceState:
        sigma_hash = self._sigma_fingerprint(sigma)
        state_key = ":".join(
            (
                self.cem_formulation,
                str(self.linear_backend),
                str(self._active_scalar_dtype()),
                sigma_hash,
            )
        )
        cached = getattr(self, "_robin_transconductance_state", None)
        if cached is not None and cached.key == state_key:
            self._set_backend_diagnostic(
                robin_transconductance_cache_hit=True,
                cem_formulation_requested=self.cem_formulation,
                cem_formulation_effective=self.cem_formulation,
            )
            return cached

        (
            robin_boundary,
            electrode_diag,
            basis,
            coupling_basis,
        ) = self._robin_blocks()
        fallback_reason = ""
        if self.linear_backend == "scipy":
            response, residual, backend, setup_count, solve_count = (
                self._solve_robin_response_scipy(
                    sigma,
                    robin_boundary,
                    coupling_basis,
                )
            )
        elif self.linear_backend == "petsc":
            try:
                response, residual, backend, setup_count, solve_count = (
                    self._solve_robin_response_petsc(
                        sigma,
                        robin_boundary,
                        coupling_basis,
                    )
                )
            except Exception as exc:
                solver_preset = self._solver_token(
                    getattr(self.backend_config, "solver_preset", "")
                )
                pc_type = self._solver_token(
                    getattr(self.backend_config, "pc_type", ""), ""
                )
                if pc_type == "amgx" or "amgx" in solver_preset:
                    raise
                fallback_reason = f"robin_petsc_failed:{exc}"
                response, residual, backend, setup_count, solve_count = (
                    self._solve_robin_response_scipy(
                        sigma,
                        robin_boundary,
                        coupling_basis,
                    )
                )
        else:
            raise ValueError(
                f"Unsupported linear_backend: {self.linear_backend}. "
                "Expected one of: 'petsc', 'scipy'."
            )

        if not all_finite_values(response):
            raise RuntimeError("Robin response basis contains non-finite values")
        reduced_map = np.asarray(
            basis.T @ (electrode_diag @ basis) - coupling_basis.T @ response,
            dtype=self._active_scalar_dtype(),
        )
        if not all_finite_values(reduced_map):
            raise RuntimeError(
                "Reduced Robin transconductance contains non-finite values"
            )
        rank = int(np.linalg.matrix_rank(reduced_map))
        expected_rank = self.n_elec - 1
        if rank != expected_rank:
            raise RuntimeError(
                "Reduced Robin transconductance is rank deficient: "
                f"rank={rank}, expected={expected_rank}"
            )
        condition_number = float(np.linalg.cond(reduced_map))
        condition_limit = _condition_limit(self._active_scalar_dtype())
        if not np.isfinite(condition_number) or condition_number > condition_limit:
            raise RuntimeError(
                "Reduced Robin transconductance is ill-conditioned: "
                f"condition={condition_number:.6e}, limit={condition_limit:.6e}"
            )
        symmetry_residual = _relative_residual(reduced_map.T, reduced_map)
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinAlgWarning)
            reduced_factor = lu_factor(reduced_map)

        response.setflags(write=False)
        basis.setflags(write=False)
        reduced_map.setflags(write=False)
        state = RobinTransconductanceState(
            key=state_key,
            response_basis=response,
            electrode_basis=basis,
            reduced_map=reduced_map,
            reduced_factor=reduced_factor,
            rank=rank,
            condition_number=condition_number,
            response_residual=residual,
            symmetry_residual=symmetry_residual,
            backend=backend,
            setup_count=setup_count,
            solve_count=solve_count,
        )
        self._robin_transconductance_state = state
        self._set_backend_diagnostic(
            cem_formulation_requested=self.cem_formulation,
            cem_formulation_effective=self.cem_formulation,
            robin_transconductance_cache_hit=False,
            robin_transconductance_backend=backend,
            robin_transconductance_rank=rank,
            robin_transconductance_condition_number=condition_number,
            robin_transconductance_condition_limit=condition_limit,
            robin_response_residual=residual,
            robin_transconductance_symmetry_residual=symmetry_residual,
            robin_basis_rhs_count=int(coupling_basis.shape[1]),
            forward_ksp_setup_count=int(setup_count),
            forward_ksp_solve_count=(
                int(solve_count) if self.linear_backend == "petsc" else 0
            ),
            fallback_reason=fallback_reason or None,
        )
        return state

    def forward_solve(self, sigma, current_patterns=None):
        """Solve balanced currents through local Robin transconductance."""

        started = time.perf_counter()
        pattern_matrix = self._resolve_pattern_matrix(current_patterns)
        if self.forward_backend == "cuda_structured":
            result = super().forward_solve(sigma, pattern_matrix)
            self._set_backend_diagnostic(
                cem_formulation_requested=self.cem_formulation,
                cem_formulation_effective=self.cem_formulation,
                robin_transconductance_backend="cuda-structured-schur",
                forward_solve_seconds=float(time.perf_counter() - started),
            )
            return result

        pattern_matrix = self._as_scalar_array(
            pattern_matrix,
            name="Robin current patterns",
        )
        current_sums = np.sum(pattern_matrix, axis=1)
        current_balance_residual = float(np.max(np.abs(current_sums), initial=0.0))
        balance_tolerance = _balance_tolerance(pattern_matrix)
        if current_balance_residual > balance_tolerance:
            raise ValueError(
                "Robin transconductance CEM requires balanced current patterns; "
                f"max abs sum={current_balance_residual:.6e}, "
                f"tolerance={balance_tolerance:.6e}"
            )

        state = self._build_robin_state(sigma)
        reduced_rhs = state.electrode_basis.T @ pattern_matrix.T
        reduced_coefficients = self._as_scalar_array(
            lu_solve(state.reduced_factor, reduced_rhs),
            name="Robin reduced electrode coefficients",
        )
        reduced_residual = _relative_residual(
            state.reduced_map @ reduced_coefficients,
            reduced_rhs,
        )
        electrode_columns = state.electrode_basis @ reduced_coefficients
        potential_columns = -(state.response_basis @ reduced_coefficients)
        electrode_block = np.asarray(
            electrode_columns.T,
            dtype=self._active_scalar_dtype(),
        )
        voltage_balance_residual = float(
            np.max(np.abs(np.sum(electrode_block, axis=1)), initial=0.0)
        )
        if not all_finite_values(electrode_block) or not all_finite_values(
            potential_columns
        ):
            raise RuntimeError(
                "Robin transconductance solution contains non-finite values"
            )

        u_views = []
        for column_index in range(pattern_matrix.shape[0]):
            column = potential_columns[:, column_index]
            column.setflags(write=False)
            u_views.append(column)
        self._set_backend_diagnostic(
            cem_formulation_requested=self.cem_formulation,
            cem_formulation_effective=self.cem_formulation,
            forward_rhs_count=int(pattern_matrix.shape[0]),
            robin_current_balance_residual=current_balance_residual,
            robin_current_balance_tolerance=balance_tolerance,
            robin_voltage_balance_residual=voltage_balance_residual,
            robin_reduced_solve_residual=reduced_residual,
            forward_solve_seconds=float(time.perf_counter() - started),
        )
        return tuple(u_views), electrode_block


__all__ = [
    "CLASSIC_CEM",
    "ROBIN_TRANSCONDUCTANCE_CEM",
    "RobinTransconductanceForwardModel",
    "RobinTransconductanceState",
    "normalize_cem_formulation",
    "zero_sum_helmert_basis",
]
