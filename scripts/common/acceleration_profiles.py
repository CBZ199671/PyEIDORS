"""Shared CLI/script helpers for high-level acceleration profiles."""

from __future__ import annotations

import argparse

from pyeidors.perf import (
    ACCELERATION_PROFILE_VALUES,
    DEFAULT_3D_GENERATOR_REVISION,
    DEFAULT_ACCELERATION_PROFILE,
    DEFAULT_FORWARD_BACKEND,
    DEFAULT_INEXACT_MODE,
    DEFAULT_LOWRANK_MODE,
    DEFAULT_MESH_FAMILY,
    DEFAULT_PETSC_DEVICE,
    DEFAULT_ROM_MODE,
    FORWARD_BACKEND_CUDA_STRUCTURED,
    MESH_FAMILY_HEX,
    PETSC_DEVICE_CUDA,
    normalize_acceleration_profile,
    prefers_3d_gpu_pipeline,
    prefers_fused_3d_gpu_pipeline,
)
from pyeidors.perf.policy import DEFAULT_3D_GEOMETRY_VERSION


def add_acceleration_profile_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str = DEFAULT_ACCELERATION_PROFILE,
    flag: str = "--acceleration-profile",
    help_suffix: str = "",
) -> None:
    """Register a shared high-level acceleration profile argument."""
    suffix = f" {help_suffix.strip()}" if str(help_suffix).strip() else ""
    parser.add_argument(
        flag,
        choices=list(ACCELERATION_PROFILE_VALUES),
        default=default,
        help=(
            "High-level runtime preset. "
            "`gpu3d` prefers 3D hex + cuda_structured + PETSc/Torch CUDA; "
            "`gpu3d_fused` also enables fused ROM/inexact/low-rank defaults."
            f"{suffix}"
        ),
    )


def resolve_acceleration_profile(profile: object) -> str:
    """Normalize a profile token to a supported acceleration preset."""
    return normalize_acceleration_profile(profile, default=DEFAULT_ACCELERATION_PROFILE)


def apply_acceleration_profile_overrides(args, *, mesh_dim: int) -> None:
    """Mutate argparse namespaces so 3D GPU workflows need fewer low-level flags."""
    profile = resolve_acceleration_profile(
        getattr(args, "acceleration_profile", DEFAULT_ACCELERATION_PROFILE)
    )
    setattr(args, "acceleration_profile", profile)
    if int(mesh_dim) != 3 or not prefers_3d_gpu_pipeline(profile):
        return

    if (
        hasattr(args, "forward_backend")
        and str(getattr(args, "forward_backend", DEFAULT_FORWARD_BACKEND))
        == DEFAULT_FORWARD_BACKEND
    ):
        setattr(args, "forward_backend", FORWARD_BACKEND_CUDA_STRUCTURED)
    if (
        hasattr(args, "mesh_family")
        and str(getattr(args, "mesh_family", DEFAULT_MESH_FAMILY))
        == DEFAULT_MESH_FAMILY
    ):
        setattr(args, "mesh_family", MESH_FAMILY_HEX)
    if (
        hasattr(args, "petsc_device")
        and str(getattr(args, "petsc_device", DEFAULT_PETSC_DEVICE))
        == DEFAULT_PETSC_DEVICE
    ):
        setattr(args, "petsc_device", PETSC_DEVICE_CUDA)
    if hasattr(args, "device") and str(getattr(args, "device", "auto")) == "auto":
        setattr(args, "device", PETSC_DEVICE_CUDA)
    if prefers_fused_3d_gpu_pipeline(profile):
        if (
            hasattr(args, "rom_mode")
            and str(getattr(args, "rom_mode", DEFAULT_ROM_MODE)) == DEFAULT_ROM_MODE
        ):
            setattr(args, "rom_mode", "on")
        if (
            hasattr(args, "inexact_mode")
            and str(getattr(args, "inexact_mode", DEFAULT_INEXACT_MODE))
            == DEFAULT_INEXACT_MODE
        ):
            setattr(args, "inexact_mode", "auto")
        if (
            hasattr(args, "lowrank_mode")
            and str(getattr(args, "lowrank_mode", DEFAULT_LOWRANK_MODE))
            == DEFAULT_LOWRANK_MODE
        ):
            setattr(args, "lowrank_mode", "auto")


def resolve_3d_mesh_contract(
    *,
    acceleration_profile: object,
    mesh_family: object | None = None,
    geometry_version: object | None = None,
    generator_revision: object | None = None,
) -> tuple[str, str, str]:
    """Resolve the 3D mesh contract preferred by the selected acceleration preset."""
    profile = resolve_acceleration_profile(acceleration_profile)
    wants_gpu = prefers_3d_gpu_pipeline(profile)
    resolved_family = (
        MESH_FAMILY_HEX
        if wants_gpu and mesh_family is None
        else str(mesh_family or DEFAULT_MESH_FAMILY).strip().lower()
        or DEFAULT_MESH_FAMILY
    )
    resolved_geometry = (
        DEFAULT_3D_GEOMETRY_VERSION
        if wants_gpu and geometry_version is None
        else str(geometry_version or DEFAULT_3D_GEOMETRY_VERSION).strip().lower()
        or DEFAULT_3D_GEOMETRY_VERSION
    )
    resolved_revision = (
        DEFAULT_3D_GENERATOR_REVISION
        if wants_gpu and generator_revision is None
        else str(generator_revision or DEFAULT_3D_GENERATOR_REVISION).strip().lower()
        or DEFAULT_3D_GENERATOR_REVISION
    )
    return resolved_family, resolved_geometry, resolved_revision
