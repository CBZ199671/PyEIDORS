"""Block metadata helpers for joint inverse parameter systems."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse

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
        "Initial implementation is block-diagonal metadata, not production Schur solve.",
        "Use fieldsplit additive first; multiplicative and Schur are upgrade paths.",
        "Do not merge sigma, z_contact, or e into an opaque dense monolith.",
    )
    if movement_block is not None:
        notes = notes + (
            "Electrode movement e is a nuisance block regularized by prior_movement.",
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
