"""Contract tests for script-level acceleration profile helpers."""

from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path
import sys


def _load_module():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "common"
        / "acceleration_profiles.py"
    )
    spec = importlib.util.spec_from_file_location("script_acceleration_profiles", script)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise AssertionError("failed to load acceleration_profiles.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_apply_acceleration_profile_overrides_promotes_3d_gpu_defaults():
    module = _load_module()
    args = Namespace(
        acceleration_profile="gpu3d",
        forward_backend="dolfinx",
        mesh_family="tetra",
        petsc_device="auto",
        device="auto",
        rom_mode="off",
        inexact_mode="off",
        lowrank_mode="off",
    )

    module.apply_acceleration_profile_overrides(args, mesh_dim=3)

    assert args.forward_backend == "cuda_structured"
    assert args.mesh_family == "hex"
    assert args.petsc_device == "cuda"
    assert args.device == "cuda"


def test_gpu3d_fused_enables_fused_defaults_and_mesh_contract():
    module = _load_module()
    args = Namespace(
        acceleration_profile="gpu3d_fused",
        forward_backend="dolfinx",
        mesh_family="tetra",
        petsc_device="auto",
        device="auto",
        rom_mode="off",
        inexact_mode="off",
        lowrank_mode="off",
    )

    module.apply_acceleration_profile_overrides(args, mesh_dim=3)
    mesh_family, geometry_version, generator_revision = module.resolve_3d_mesh_contract(
        acceleration_profile="gpu3d_fused",
    )

    assert args.rom_mode == "on"
    assert args.inexact_mode == "auto"
    assert args.lowrank_mode == "auto"
    assert mesh_family == "hex"
    assert geometry_version == "geomv2"
    assert generator_revision == "g3d3"


def test_2d_profile_does_not_override_runtime_defaults():
    module = _load_module()
    args = Namespace(
        acceleration_profile="gpu3d",
        forward_backend="dolfinx",
        mesh_family="tetra",
        petsc_device="auto",
        device="auto",
    )

    module.apply_acceleration_profile_overrides(args, mesh_dim=2)

    assert args.forward_backend == "dolfinx"
    assert args.mesh_family == "tetra"
    assert args.petsc_device == "auto"
    assert args.device == "auto"
