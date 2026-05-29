"""PETSc scalar-type helpers for real and complex FEniCSx runtimes."""

from __future__ import annotations

from typing import Any

import numpy as np

_PETSC_UNSET = object()
PETSc: Any = _PETSC_UNSET


def _petsc_module() -> Any:
    """Resolve petsc4py lazily; scalar helpers are often imported by light CLIs."""

    global PETSc
    if PETSc is _PETSC_UNSET:
        try:  # pragma: no cover - petsc4py is present in full Nix runtimes
            from petsc4py import PETSc as petsc
        except ImportError:  # pragma: no cover
            petsc = None
        PETSc = petsc
    return PETSc


def petsc_scalar_dtype() -> np.dtype:
    """Return the active PETSc scalar dtype, falling back to float64."""

    petsc = _petsc_module()
    if petsc is None:
        return np.dtype(np.float64)
    return np.dtype(getattr(petsc, "ScalarType", np.float64))


def petsc_scalar_dtype_name() -> str:
    """Return the active PETSc scalar dtype name."""

    return str(petsc_scalar_dtype())


def petsc_scalar_is_complex() -> bool:
    """Return whether the active PETSc runtime was built for complex scalars."""

    return bool(np.issubdtype(petsc_scalar_dtype(), np.complexfloating))


def runtime_scalar_summary() -> dict[str, object]:
    """Small diagnostic payload for forward-model runtime reports."""

    dtype = petsc_scalar_dtype()
    return {
        "petsc_scalar_type": str(dtype),
        "petsc_scalar_is_complex": bool(np.issubdtype(dtype, np.complexfloating)),
    }


def require_complex_scalar_support(feature: str = "complex admittivity CEM") -> None:
    """Fail fast when a complex-only feature is used in a real PETSc runtime."""

    if petsc_scalar_is_complex():
        return
    raise RuntimeError(
        f"{feature} requires a complex PETSc/DOLFINx runtime. "
        "Enter `nix develop .#complex` / `nix develop .#complex64` for CPU, "
        "or `nix develop .#complex-cuda` / `nix develop .#complex64-cuda` "
        "for PETSc CUDA, and retry."
    )


def require_runtime_scalar_dtype(
    *,
    complex_required: bool,
    feature: str = "complex admittivity CEM",
) -> None:
    """Validate the PETSc runtime scalar mode for feature gates."""

    if complex_required:
        require_complex_scalar_support(feature)
