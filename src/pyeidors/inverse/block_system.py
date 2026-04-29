"""Block metadata helpers for joint inverse parameter systems."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

try:  # pragma: no cover - optional in non-Nix/unit environments
    from petsc4py import PETSc as _PETSc
except Exception:  # pragma: no cover
    _PETSc = None

ArrayAction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ParameterBlock:
    """Contiguous parameter block in a joint inverse vector."""

    name: str
    kind: str
    size: int
    offset: int
    preconditioner: str
    regularization: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def stop(self) -> int:
        return int(self.offset + self.size)

    @property
    def slice(self) -> slice:
        return slice(int(self.offset), int(self.stop))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "size": int(self.size),
            "offset": int(self.offset),
            "stop": int(self.stop),
            "preconditioner": self.preconditioner,
            "regularization": self.regularization,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BlockCoupling:
    """Shape-only coupling metadata between two block spaces."""

    name: str
    row: str
    col: str
    shape: tuple[int, int]
    role: str
    approximation: str = "matrix-free-action"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "row": self.row,
            "col": self.col,
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "role": self.role,
            "approximation": self.approximation,
        }


@dataclass(frozen=True)
class JointInverseBlockMetadata:
    """Block-ready contract for joint inverse systems such as sigma + z_contact."""

    blocks: tuple[ParameterBlock, ...]
    couplings: tuple[BlockCoupling, ...]
    fieldsplit_type: str = "additive"
    schur_approximation: str = "block-diagonal-first"
    notes: tuple[str, ...] = ()

    @property
    def total_size(self) -> int:
        if not self.blocks:
            return 0
        return max(block.stop for block in self.blocks)

    def block(self, name: str) -> ParameterBlock:
        for block in self.blocks:
            if block.name == name:
                return block
        raise KeyError(f"Unknown parameter block: {name!r}")

    def block_slices(self) -> dict[str, slice]:
        return {block.name: block.slice for block in self.blocks}

    def fieldsplit_plan(self) -> dict[str, Any]:
        return {
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": self.fieldsplit_type,
            "schur_approximation": self.schur_approximation,
            "blocks": [block.to_dict() for block in self.blocks],
            "couplings": [coupling.to_dict() for coupling in self.couplings],
            "upgrade_path": [
                "block-diagonal",
                "multiplicative",
                "schur",
            ],
            "notes": list(self.notes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_size": int(self.total_size),
            "fieldsplit_plan": self.fieldsplit_plan(),
        }


@dataclass(frozen=True)
class SigmaContactNormalSystem:
    """Sparse normal-equation block system for joint sigma/contact updates."""

    matrix: sparse.csr_matrix
    rhs: np.ndarray
    metadata: JointInverseBlockMetadata
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.matrix.shape)


@dataclass(frozen=True)
class JointFieldSplitSolveResult:
    """Result of a joint sigma/contact fieldsplit solve."""

    solution: np.ndarray
    metadata: JointInverseBlockMetadata
    backend: str
    fieldsplit_type: str
    converged: bool
    iterations: int
    residual_norm: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def sigma_delta(self) -> np.ndarray:
        sigma = self.metadata.block("sigma")
        return np.asarray(self.solution[sigma.slice], dtype=np.float64)

    @property
    def contact_delta(self) -> np.ndarray:
        contact = self.metadata.block("z_contact")
        return np.asarray(self.solution[contact.slice], dtype=np.float64)


def _positive_int(name: str, value: int) -> int:
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return resolved


def _nonnegative_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}.")
    return resolved


def _optional_positive_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    return _positive_int(name, value)


def _positive_float(name: str, value: float) -> float:
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return resolved


def _nonnegative_float(name: str, value: float) -> float:
    resolved = float(value)
    if not np.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}.")
    return resolved


def _finite_vector(
    name: str, value: np.ndarray, *, length: int | None = None
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if length is not None and arr.size != int(length):
        raise ValueError(f"{name} length {arr.size} does not match {int(length)}.")
    if not np.isfinite(arr).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _as_csr_matrix(name: str, value: Any) -> sparse.csr_matrix:
    if sparse.issparse(value):
        matrix = value.tocsr().astype(np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 2D matrix.")
        matrix = sparse.csr_matrix(arr)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError(f"{name} must be a non-empty 2D matrix.")
    if matrix.nnz and not np.isfinite(matrix.data).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
    matrix.sort_indices()
    return matrix


def _matrix_is_symmetric(matrix: sparse.csr_matrix, *, atol: float = 1e-12) -> bool:
    diff = (matrix - matrix.T).tocoo()
    if diff.nnz == 0:
        return True
    return bool(np.max(np.abs(diff.data)) <= float(atol))


def _regularization_to_csr(
    name: str,
    value: Any | None,
    *,
    size: int,
) -> sparse.csr_matrix:
    if value is None:
        return sparse.csr_matrix((size, size), dtype=np.float64)
    if np.isscalar(value):
        weight = _nonnegative_float(name, float(value))
        return (sparse.eye(size, format="csr", dtype=np.float64) * weight).tocsr()
    if sparse.issparse(value):
        matrix = value.tocsr().astype(np.float64)
        if matrix.shape != (size, size):
            raise ValueError(
                f"{name} shape {matrix.shape} does not match {(size, size)}."
            )
        if matrix.nnz and not np.isfinite(matrix.data).all():
            raise FloatingPointError(f"{name} contains non-finite values.")
        matrix.sort_indices()
        return matrix

    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size != size:
            raise ValueError(f"{name} length {arr.size} does not match {size}.")
        if not np.isfinite(arr).all() or np.any(arr < 0.0):
            raise FloatingPointError(
                f"{name} diagonal must be finite and non-negative."
            )
        return sparse.diags(arr, format="csr", dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape != (size, size):
            raise ValueError(f"{name} shape {arr.shape} does not match {(size, size)}.")
        if not np.isfinite(arr).all():
            raise FloatingPointError(f"{name} contains non-finite values.")
        return sparse.csr_matrix(arr)
    raise ValueError(f"{name} must be scalar, diagonal vector, or square matrix.")


def _measurement_weights_to_csr(
    value: Any | None,
    *,
    size: int,
) -> tuple[sparse.csr_matrix, str]:
    if value is None:
        return sparse.eye(size, format="csr", dtype=np.float64), "identity"
    if np.isscalar(value):
        weight = _nonnegative_float("measurement_weights", float(value))
        return (
            sparse.eye(size, format="csr", dtype=np.float64) * weight
        ).tocsr(), "scalar"
    if sparse.issparse(value):
        matrix = value.tocsr().astype(np.float64)
        source = "sparse-matrix"
    else:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 1:
            if arr.size != size:
                raise ValueError(
                    f"measurement_weights length {arr.size} does not match {size}."
                )
            if not np.isfinite(arr).all() or np.any(arr < 0.0):
                raise FloatingPointError(
                    "measurement_weights diagonal must be finite and non-negative."
                )
            return sparse.diags(arr, format="csr", dtype=np.float64), "diagonal"
        if arr.ndim != 2:
            raise ValueError("measurement_weights must be scalar, vector, or matrix.")
        matrix = sparse.csr_matrix(arr)
        source = "dense-matrix"

    if matrix.shape != (size, size):
        raise ValueError(
            f"measurement_weights shape {matrix.shape} does not match {(size, size)}."
        )
    if matrix.nnz and not np.isfinite(matrix.data).all():
        raise FloatingPointError("measurement_weights contains non-finite values.")
    if not _matrix_is_symmetric(matrix):
        raise ValueError("measurement_weights matrix must be symmetric.")
    matrix.sort_indices()
    return matrix, source


def build_sigma_contact_block_metadata(
    *,
    n_sigma: int,
    n_contact: int,
    n_movement: int | None = None,
    n_measurements: int | None = None,
    sigma_preconditioner: str = "prior-preconditioned-cg",
    contact_preconditioner: str = "dense-lu-or-jacobi",
    movement_preconditioner: str = "small-dense-or-jacobi",
    sigma_regularization: str = "prior-or-smoothness",
    contact_regularization: str = "diagonal-scale",
    movement_regularization: str = "prior_movement",
    fieldsplit_type: str = "additive",
) -> JointInverseBlockMetadata:
    """Create shape-safe metadata for sigma, contact, and optional movement blocks."""
    n_sigma = _positive_int("n_sigma", n_sigma)
    n_contact = _positive_int("n_contact", n_contact)
    n_movement = _optional_positive_int("n_movement", n_movement)
    n_measurements = _nonnegative_int("n_measurements", n_measurements)
    fieldsplit_type = str(fieldsplit_type).strip().lower()
    if fieldsplit_type not in {"additive", "multiplicative", "schur"}:
        raise ValueError(
            "fieldsplit_type must be one of: additive, multiplicative, schur."
        )

    sigma_block = ParameterBlock(
        name="sigma",
        kind="cell-conductivity",
        size=n_sigma,
        offset=0,
        preconditioner=str(sigma_preconditioner),
        regularization=str(sigma_regularization),
        metadata={"scale": "cell-count"},
    )
    contact_block = ParameterBlock(
        name="z_contact",
        kind="electrode-contact-impedance",
        size=n_contact,
        offset=n_sigma,
        preconditioner=str(contact_preconditioner),
        regularization=str(contact_regularization),
        metadata={"scale": "electrode-count"},
    )
    blocks: list[ParameterBlock] = [sigma_block, contact_block]
    movement_block: ParameterBlock | None = None
    if n_movement is not None:
        dofs_per_electrode: float | None = None
        if n_contact > 0:
            dofs_per_electrode = float(n_movement) / float(n_contact)
        movement_block = ParameterBlock(
            name="e",
            kind="electrode-pose-movement",
            size=n_movement,
            offset=n_sigma + n_contact,
            preconditioner=str(movement_preconditioner),
            regularization=str(movement_regularization),
            metadata={
                "scale": "electrode-motion-dofs",
                "dofs_per_electrode": dofs_per_electrode,
            },
        )
        blocks.append(movement_block)

    couplings: list[BlockCoupling] = [
        BlockCoupling(
            name="H_sigma_z",
            row="sigma",
            col="z_contact",
            shape=(n_sigma, n_contact),
            role="hessian-coupling",
            approximation="drop-or-low-rank-until-schur",
        ),
        BlockCoupling(
            name="H_z_sigma",
            row="z_contact",
            col="sigma",
            shape=(n_contact, n_sigma),
            role="hessian-coupling",
            approximation="transpose-action",
        ),
    ]
    if movement_block is not None:
        couplings.extend(
            [
                BlockCoupling(
                    name="H_sigma_e",
                    row="sigma",
                    col="e",
                    shape=(n_sigma, n_movement),
                    role="hessian-coupling",
                    approximation="matrix-free-or-low-rank-until-schur",
                ),
                BlockCoupling(
                    name="H_e_sigma",
                    row="e",
                    col="sigma",
                    shape=(n_movement, n_sigma),
                    role="hessian-coupling",
                    approximation="transpose-action",
                ),
                BlockCoupling(
                    name="H_z_e",
                    row="z_contact",
                    col="e",
                    shape=(n_contact, n_movement),
                    role="hessian-coupling",
                    approximation="drop-or-diag-until-schur",
                ),
                BlockCoupling(
                    name="H_e_z",
                    row="e",
                    col="z_contact",
                    shape=(n_movement, n_contact),
                    role="hessian-coupling",
                    approximation="transpose-action",
                ),
                BlockCoupling(
                    name="H_ee",
                    row="e",
                    col="e",
                    shape=(n_movement, n_movement),
                    role="hessian-block",
                    approximation="prior_movement-plus-gn-diag",
                ),
            ]
        )
    if n_measurements is not None:
        measurement_couplings = [
            BlockCoupling(
                name="J_sigma",
                row="measurement",
                col="sigma",
                shape=(n_measurements, n_sigma),
                role="measurement-jacobian",
            ),
            BlockCoupling(
                name="J_z_contact",
                row="measurement",
                col="z_contact",
                shape=(n_measurements, n_contact),
                role="measurement-jacobian",
            ),
        ]
        if movement_block is not None:
            measurement_couplings.append(
                BlockCoupling(
                    name="J_e",
                    row="measurement",
                    col="e",
                    shape=(n_measurements, n_movement),
                    role="movement-jacobian",
                    approximation="finite-difference-or-adjoint-action",
                )
            )
        couplings.extend(measurement_couplings)

    notes = (
        "sigma+z_contact PCFIELDSPLIT solve is available via solve_sigma_contact_fieldsplit.",
        "Use fieldsplit additive first; multiplicative and Schur remain selectable upgrade paths.",
        "Do not merge sigma, z_contact, or e into an opaque dense monolith.",
    )
    if movement_block is not None:
        notes = notes + (
            "Electrode movement e is a nuisance block regularized by prior_movement; e solve remains metadata/prior only.",
        )
    return JointInverseBlockMetadata(
        blocks=tuple(blocks),
        couplings=tuple(couplings),
        fieldsplit_type=fieldsplit_type,
        schur_approximation=(
            "diag-z-and-prior-sigma"
            if fieldsplit_type == "schur"
            else "block-diagonal-first"
        ),
        notes=notes,
    )


def assemble_sigma_contact_normal_system(
    j_sigma: Any,
    j_contact: Any,
    residual: np.ndarray,
    *,
    sigma_regularization: Any | None = None,
    contact_regularization: Any | None = None,
    measurement_weights: Any | None = None,
    fieldsplit_type: str = "additive",
) -> SigmaContactNormalSystem:
    """Assemble the sparse joint normal system for ``[delta_sigma, delta_z]``."""
    j_sigma_csr = _as_csr_matrix("j_sigma", j_sigma)
    j_contact_csr = _as_csr_matrix("j_contact", j_contact)
    if j_sigma_csr.shape[0] != j_contact_csr.shape[0]:
        raise ValueError(
            "j_sigma and j_contact must have the same measurement row count."
        )

    n_measurements = int(j_sigma_csr.shape[0])
    n_sigma = int(j_sigma_csr.shape[1])
    n_contact = int(j_contact_csr.shape[1])
    rhs_residual = _finite_vector("residual", residual, length=n_measurements)
    metadata = build_sigma_contact_block_metadata(
        n_sigma=n_sigma,
        n_contact=n_contact,
        n_measurements=n_measurements,
        fieldsplit_type=fieldsplit_type,
    )

    weights, weight_source = _measurement_weights_to_csr(
        measurement_weights,
        size=n_measurements,
    )
    sigma_reg = _regularization_to_csr(
        "sigma_regularization",
        sigma_regularization,
        size=n_sigma,
    )
    contact_reg = _regularization_to_csr(
        "contact_regularization",
        contact_regularization,
        size=n_contact,
    )

    weighted_sigma = weights @ j_sigma_csr
    weighted_contact = weights @ j_contact_csr
    weighted_residual = np.asarray(weights @ rhs_residual, dtype=np.float64).reshape(-1)

    h_sigma_sigma = (j_sigma_csr.T @ weighted_sigma + sigma_reg).tocsr()
    h_sigma_contact = (j_sigma_csr.T @ weighted_contact).tocsr()
    h_contact_sigma = (j_contact_csr.T @ weighted_sigma).tocsr()
    h_contact_contact = (j_contact_csr.T @ weighted_contact + contact_reg).tocsr()
    matrix = sparse.bmat(
        [
            [h_sigma_sigma, h_sigma_contact],
            [h_contact_sigma, h_contact_contact],
        ],
        format="csr",
        dtype=np.float64,
    )
    rhs = np.concatenate(
        [
            np.asarray(j_sigma_csr.T @ weighted_residual, dtype=np.float64).reshape(-1),
            np.asarray(j_contact_csr.T @ weighted_residual, dtype=np.float64).reshape(
                -1
            ),
        ]
    )
    if matrix.nnz and not np.isfinite(matrix.data).all():
        raise FloatingPointError(
            "sigma/contact normal matrix contains non-finite values."
        )
    if not np.isfinite(rhs).all():
        raise FloatingPointError("sigma/contact normal rhs contains non-finite values.")
    matrix.sort_indices()
    return SigmaContactNormalSystem(
        matrix=matrix,
        rhs=np.ascontiguousarray(rhs, dtype=np.float64),
        metadata=metadata,
        diagnostics={
            "n_measurements": n_measurements,
            "n_sigma": n_sigma,
            "n_contact": n_contact,
            "matrix_nnz": int(matrix.nnz),
            "measurement_weights": weight_source,
            "sigma_regularization_nnz": int(sigma_reg.nnz),
            "contact_regularization_nnz": int(contact_reg.nnz),
        },
    )


def _petsc_composite_type(petsc_module: Any, fieldsplit_type: str) -> Any:
    fallback = str(fieldsplit_type).upper()
    pc_cls = getattr(petsc_module, "PC", None)
    composite = getattr(pc_cls, "CompositeType", None)
    return getattr(composite, fallback, str(fieldsplit_type))


def _petsc_schur_fact_type(petsc_module: Any) -> Any:
    pc_cls = getattr(petsc_module, "PC", None)
    schur_fact = getattr(pc_cls, "SchurFactType", None)
    return getattr(schur_fact, "FULL", "full")


def _create_petsc_stride_is(
    petsc_module: Any,
    block: ParameterBlock,
    *,
    comm: Any,
) -> Any:
    is_factory = getattr(petsc_module, "IS", None)
    if is_factory is None:
        return np.arange(block.offset, block.stop, dtype=np.int32)
    return is_factory().createStride(
        int(block.size),
        first=int(block.offset),
        step=1,
        comm=comm,
    )


def configure_petsc_fieldsplit_solver(
    ksp: Any,
    metadata: JointInverseBlockMetadata,
    *,
    petsc_module: Any | None = None,
    index_sets: Mapping[str, Any] | None = None,
    ksp_type: str = "gmres",
    rtol: float | None = None,
    maxiter: int | None = None,
) -> dict[str, Any]:
    """Configure a PETSc KSP/PC pair for sigma/contact block field splitting."""
    petsc = petsc_module if petsc_module is not None else _PETSc
    if petsc is None:
        raise RuntimeError("petsc4py is required for PETSc fieldsplit configuration.")
    if not hasattr(ksp, "getPC"):
        raise TypeError("ksp must provide getPC().")

    if ksp_type and hasattr(ksp, "setType"):
        ksp.setType(str(ksp_type))
    pc = ksp.getPC()
    if hasattr(pc, "setType"):
        pc.setType("fieldsplit")
    if hasattr(pc, "setFieldSplitType"):
        pc.setFieldSplitType(_petsc_composite_type(petsc, metadata.fieldsplit_type))
    if metadata.fieldsplit_type == "schur" and hasattr(
        pc, "setFieldSplitSchurFactType"
    ):
        pc.setFieldSplitSchurFactType(_petsc_schur_fact_type(petsc))

    comm = getattr(ksp, "comm", getattr(petsc, "COMM_SELF", None))
    resolved_sets: dict[str, Any] = {}
    for block in metadata.blocks:
        if index_sets is not None and block.name in index_sets:
            resolved_sets[block.name] = index_sets[block.name]
        else:
            resolved_sets[block.name] = _create_petsc_stride_is(
                petsc,
                block,
                comm=comm,
            )
    if hasattr(pc, "setFieldSplitIS"):
        pc.setFieldSplitIS(
            *((block.name, resolved_sets[block.name]) for block in metadata.blocks)
        )
    if (rtol is not None or maxiter is not None) and hasattr(ksp, "setTolerances"):
        kwargs: dict[str, Any] = {}
        if rtol is not None:
            kwargs["rtol"] = float(rtol)
        if maxiter is not None:
            kwargs["max_it"] = int(max(1, maxiter))
        ksp.setTolerances(**kwargs)

    return {
        "ksp_type": str(ksp_type),
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": metadata.fieldsplit_type,
        "fieldsplit_blocks": [block.to_dict() for block in metadata.blocks],
        "schur_approximation": metadata.schur_approximation,
    }


def _petsc_mat_from_csr(matrix: sparse.csr_matrix, petsc_module: Any, comm: Any) -> Any:
    mat = petsc_module.Mat().createAIJ(
        size=matrix.shape,
        csr=(matrix.indptr, matrix.indices, matrix.data),
        comm=comm,
    )
    if hasattr(mat, "assemble"):
        mat.assemble()
    return mat


def _petsc_vec_array(vec: Any) -> np.ndarray:
    if hasattr(vec, "getArray"):
        try:
            return np.asarray(vec.getArray(readonly=False), dtype=np.float64)
        except TypeError:
            return np.asarray(vec.getArray(), dtype=np.float64)
    if hasattr(vec, "array"):
        return np.asarray(vec.array, dtype=np.float64)
    raise TypeError("Unsupported PETSc Vec wrapper.")


def _destroy_petsc_objects(*objects: Any) -> None:
    for obj in objects:
        destroy = getattr(obj, "destroy", None)
        if callable(destroy):
            try:
                destroy()
            except Exception:
                pass


def _solve_sigma_contact_with_petsc(
    system: SigmaContactNormalSystem,
    *,
    petsc_module: Any,
    rtol: float,
    maxiter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    comm = getattr(petsc_module, "COMM_SELF", None)
    mat = _petsc_mat_from_csr(system.matrix, petsc_module, comm)
    ksp = petsc_module.KSP().create(comm=comm)
    b = None
    x = None
    try:
        ksp.setOperators(mat)
        plan = configure_petsc_fieldsplit_solver(
            ksp,
            system.metadata,
            petsc_module=petsc_module,
            rtol=rtol,
            maxiter=maxiter,
        )
        if hasattr(ksp, "setFromOptions"):
            ksp.setFromOptions()
        if hasattr(ksp, "setUp"):
            ksp.setUp()
        b = mat.createVecRight()
        x = mat.createVecRight()
        _petsc_vec_array(b)[:] = system.rhs
        ksp.solve(b, x)
        solution = np.asarray(_petsc_vec_array(x), dtype=np.float64).reshape(-1).copy()
        iterations = (
            int(ksp.getIterationNumber()) if hasattr(ksp, "getIterationNumber") else 0
        )
        reason = (
            int(ksp.getConvergedReason()) if hasattr(ksp, "getConvergedReason") else 1
        )
        if not np.isfinite(solution).all():
            raise FloatingPointError(
                "PETSc fieldsplit solve produced non-finite values."
            )
        if reason <= 0:
            raise RuntimeError(f"petsc_fieldsplit_not_converged:{reason}")
        return solution, {
            **plan,
            "backend": "petsc",
            "converged_reason": reason,
            "iterations": iterations,
        }
    finally:
        _destroy_petsc_objects(b, x, ksp, mat)


def _solve_sigma_contact_with_scipy(
    system: SigmaContactNormalSystem,
    *,
    rtol: float,
    maxiter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    with warnings.catch_warnings():
        warnings.simplefilter("error", spla.MatrixRankWarning)
        try:
            solution = spla.spsolve(system.matrix, system.rhs)
            out = np.asarray(solution, dtype=np.float64).reshape(-1)
            if np.isfinite(out).all():
                return np.ascontiguousarray(out), {
                    "backend": "scipy",
                    "solver": "spsolve",
                    "converged": True,
                    "iterations": 0,
                }
        except (spla.MatrixRankWarning, RuntimeError, ValueError):
            pass
    lsmr = spla.lsmr(
        system.matrix,
        system.rhs,
        atol=float(rtol),
        btol=float(rtol),
        maxiter=int(max(1, maxiter)),
    )
    out = np.asarray(lsmr[0], dtype=np.float64).reshape(-1)
    if not np.isfinite(out).all():
        raise FloatingPointError(
            "sigma/contact fieldsplit solve produced non-finite values."
        )
    return np.ascontiguousarray(out), {
        "backend": "scipy",
        "solver": "lsmr",
        "converged": bool(int(lsmr[1]) in {1, 2}),
        "iterations": int(lsmr[2]),
        "lsmr_stop_code": int(lsmr[1]),
    }


def solve_sigma_contact_fieldsplit(
    j_sigma: Any,
    j_contact: Any,
    residual: np.ndarray,
    *,
    sigma_regularization: Any | None = None,
    contact_regularization: Any | None = None,
    measurement_weights: Any | None = None,
    fieldsplit_type: str = "additive",
    backend: str = "auto",
    rtol: float = 1e-8,
    maxiter: int = 1000,
    petsc_module: Any | None = None,
) -> JointFieldSplitSolveResult:
    """Solve a joint ``sigma + z_contact`` Gauss-Newton block system."""
    system = assemble_sigma_contact_normal_system(
        j_sigma,
        j_contact,
        residual,
        sigma_regularization=sigma_regularization,
        contact_regularization=contact_regularization,
        measurement_weights=measurement_weights,
        fieldsplit_type=fieldsplit_type,
    )
    requested_backend = str(backend).strip().lower() or "auto"
    if requested_backend not in {"auto", "petsc", "scipy"}:
        raise ValueError("backend must be one of: auto, petsc, scipy.")
    rtol = _positive_float("rtol", rtol)
    maxiter = _positive_int("maxiter", maxiter)

    petsc = petsc_module if petsc_module is not None else _PETSc
    diagnostics: dict[str, Any] = {
        **dict(system.diagnostics),
        "backend_requested": requested_backend,
        "fieldsplit_plan": system.metadata.fieldsplit_plan(),
    }
    solution: np.ndarray | None = None
    solve_meta: dict[str, Any] = {}
    if requested_backend in {"auto", "petsc"} and petsc is not None:
        try:
            solution, solve_meta = _solve_sigma_contact_with_petsc(
                system,
                petsc_module=petsc,
                rtol=rtol,
                maxiter=maxiter,
            )
        except Exception as exc:
            diagnostics["petsc_fallback_reason"] = (
                f"petsc_fieldsplit_failed:{type(exc).__name__}"
            )
    elif requested_backend == "petsc":
        diagnostics["petsc_fallback_reason"] = "petsc_backend_unavailable"

    if solution is None:
        solution, solve_meta = _solve_sigma_contact_with_scipy(
            system,
            rtol=rtol,
            maxiter=maxiter,
        )
    diagnostics.update(solve_meta)
    residual_vec = np.asarray(system.matrix @ solution - system.rhs, dtype=np.float64)
    residual_norm = float(np.linalg.norm(residual_vec))
    return JointFieldSplitSolveResult(
        solution=np.ascontiguousarray(solution, dtype=np.float64),
        metadata=system.metadata,
        backend=str(solve_meta.get("backend", "scipy")),
        fieldsplit_type=system.metadata.fieldsplit_type,
        converged=bool(solve_meta.get("converged", True)),
        iterations=int(solve_meta.get("iterations", 0)),
        residual_norm=residual_norm,
        diagnostics=diagnostics,
    )


def make_block_diagonal_inverse_action(
    metadata: JointInverseBlockMetadata,
    *,
    sigma_inverse_action: ArrayAction,
    contact_inverse_action: ArrayAction,
    movement_inverse_action: ArrayAction | None = None,
) -> ArrayAction:
    """Create a shape-checked block diagonal inverse action for joint blocks."""
    sigma = metadata.block("sigma")
    contact = metadata.block("z_contact")
    movement = next((block for block in metadata.blocks if block.name == "e"), None)
    if movement is not None and movement_inverse_action is None:
        raise ValueError(
            "movement_inverse_action is required when metadata contains e."
        )

    def _apply(vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float64).reshape(-1)
        if arr.shape[0] != metadata.total_size:
            raise ValueError(
                f"Expected vector length {metadata.total_size}, got {arr.shape[0]}."
            )

        out = np.zeros_like(arr, dtype=np.float64)
        sigma_out = np.asarray(
            sigma_inverse_action(arr[sigma.slice]), dtype=np.float64
        ).reshape(-1)
        contact_out = np.asarray(
            contact_inverse_action(arr[contact.slice]), dtype=np.float64
        ).reshape(-1)
        if sigma_out.shape[0] != sigma.size:
            raise ValueError(
                f"sigma inverse action returned length {sigma_out.shape[0]}, expected {sigma.size}."
            )
        if contact_out.shape[0] != contact.size:
            raise ValueError(
                "contact inverse action returned length "
                f"{contact_out.shape[0]}, expected {contact.size}."
            )
        out[sigma.slice] = sigma_out
        out[contact.slice] = contact_out
        if movement is not None:
            movement_out = np.asarray(
                movement_inverse_action(arr[movement.slice]), dtype=np.float64
            ).reshape(-1)
            if movement_out.shape[0] != movement.size:
                raise ValueError(
                    "movement inverse action returned length "
                    f"{movement_out.shape[0]}, expected {movement.size}."
                )
            out[movement.slice] = movement_out
        return out

    return _apply


def build_electrode_movement_jacobian(
    baseline_measurements: np.ndarray,
    perturbed_measurements: np.ndarray,
    perturbation_steps: np.ndarray | float,
    *,
    orientation: str = "movement-major",
) -> np.ndarray:
    """Finite-difference measurement Jacobian for electrode movement parameters."""

    baseline = np.asarray(baseline_measurements, dtype=np.float64).reshape(-1)
    if baseline.size == 0:
        raise ValueError("baseline_measurements must be non-empty.")
    if not np.isfinite(baseline).all():
        raise FloatingPointError("baseline_measurements contain non-finite values.")

    perturbed = np.asarray(perturbed_measurements, dtype=np.float64)
    if perturbed.ndim != 2 or 0 in perturbed.shape:
        raise ValueError("perturbed_measurements must be a non-empty 2D array.")
    if not np.isfinite(perturbed).all():
        raise FloatingPointError("perturbed_measurements contain non-finite values.")

    orientation = str(orientation).strip().lower()
    if orientation == "movement-major":
        if perturbed.shape[1] != baseline.size:
            raise ValueError(
                "movement-major perturbed_measurements must have shape "
                "(n_movement, n_measurements)."
            )
        movement_major = perturbed
    elif orientation == "measurement-major":
        if perturbed.shape[0] != baseline.size:
            raise ValueError(
                "measurement-major perturbed_measurements must have shape "
                "(n_measurements, n_movement)."
            )
        movement_major = perturbed.T
    else:
        raise ValueError(
            "orientation must be one of: movement-major, measurement-major."
        )

    n_movement = int(movement_major.shape[0])
    steps = np.asarray(perturbation_steps, dtype=np.float64).reshape(-1)
    if steps.size == 1:
        steps = np.full(n_movement, float(steps[0]), dtype=np.float64)
    if steps.size != n_movement:
        raise ValueError(
            f"perturbation_steps length {steps.size} does not match {n_movement}."
        )
    if not np.isfinite(steps).all() or np.any(steps == 0.0):
        raise FloatingPointError("perturbation_steps must be finite and non-zero.")

    jacobian = (movement_major - baseline[np.newaxis, :]) / steps[:, np.newaxis]
    return np.ascontiguousarray(jacobian.T, dtype=np.float64)


def prior_movement(
    n_movement: int,
    *,
    weight: float = 1.0,
    floor: float = 0.0,
) -> sparse.csr_matrix:
    """Diagonal prior for electrode pose / movement nuisance parameters."""

    n = _positive_int("n_movement", n_movement)
    weight = _positive_float("weight", weight)
    floor = _nonnegative_float("floor", floor)
    return (sparse.eye(n, format="csr", dtype=np.float64) * (weight + floor)).tocsr()


def scale_contact_impedance_update(
    current_z: np.ndarray,
    delta_z: np.ndarray,
    *,
    max_relative_step: float = 0.5,
    floor: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Apply a finite, globally scaled contact-impedance update."""
    z = np.asarray(current_z, dtype=np.float64).reshape(-1)
    delta = np.asarray(delta_z, dtype=np.float64).reshape(-1)
    if z.shape != delta.shape:
        raise ValueError(
            f"contact update shape mismatch: current={z.shape}, delta={delta.shape}."
        )
    if not np.isfinite(z).all() or not np.isfinite(delta).all():
        raise FloatingPointError("contact impedance update contains non-finite values.")

    max_relative_step = float(max_relative_step)
    floor = float(floor)
    if not np.isfinite(max_relative_step) or max_relative_step <= 0.0:
        raise ValueError("max_relative_step must be positive.")
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("floor must be positive.")

    safe_z = np.maximum(np.abs(z), floor)
    nonzero = np.abs(delta) > 0.0
    if not np.any(nonzero):
        return np.maximum(z, floor), 1.0

    allowed = max_relative_step * safe_z[nonzero]
    ratios = allowed / np.maximum(np.abs(delta[nonzero]), floor)
    step = float(min(1.0, np.min(ratios)))
    updated = np.maximum(z + step * delta, floor)
    if not np.isfinite(updated).all():
        raise FloatingPointError("contact impedance update produced non-finite values.")
    return updated.astype(np.float64), step
