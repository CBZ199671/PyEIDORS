"""Tests for PETSc forward solver preset resolution."""

from __future__ import annotations

from pyeidors.forward.eit_forward_model import EITForwardModel, LinearBackendConfig


def _model_with_dim(mesh_dim: int):
    model = object.__new__(EITForwardModel)
    model.mesh_tdim = mesh_dim
    return model


def test_auto_preset_keeps_2d_on_direct_solver() -> None:
    config = EITForwardModel._resolve_linear_backend_config(
        _model_with_dim(2),
        LinearBackendConfig(),
    )

    assert config.solver_preset == "direct"
    assert config.ksp_type == "preonly"
    assert config.pc_type == "lu"


def test_auto_preset_uses_portable_3d_amg_solver() -> None:
    config = EITForwardModel._resolve_linear_backend_config(
        _model_with_dim(3),
        LinearBackendConfig(),
    )

    assert config.solver_preset == "3d_gamg"
    assert config.ksp_type == "fgmres"
    assert config.pc_type == "gamg"
    assert config.pc_gamg_type == "agg"
    assert config.petsc_options["mg_levels_ksp_type"] == "chebyshev"


def test_explicit_hypre_preset_sets_boomeramg() -> None:
    config = EITForwardModel._resolve_linear_backend_config(
        _model_with_dim(3),
        LinearBackendConfig.from_dict({"solver_preset": "3d_hypre"}),
    )

    assert config.ksp_type == "fgmres"
    assert config.pc_type == "hypre"
    assert config.pc_hypre_type == "boomeramg"


def test_explicit_ksp_pc_are_not_overridden_by_auto_preset() -> None:
    config = EITForwardModel._resolve_linear_backend_config(
        _model_with_dim(3),
        LinearBackendConfig.from_dict({"ksp_type": "minres", "pc_type": "jacobi"}),
    )

    assert config.solver_preset == "custom"
    assert config.ksp_type == "minres"
    assert config.pc_type == "jacobi"
