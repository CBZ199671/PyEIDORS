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


def test_cuda_amgx_preset_requests_cuda_cg_amgx() -> None:
    config = EITForwardModel._resolve_linear_backend_config(
        _model_with_dim(3),
        LinearBackendConfig.from_dict({"solver_preset": "cuda_amgx"}),
    )

    assert config.ksp_type == "cg"
    assert config.pc_type == "amgx"
    assert config.petsc_device == "cuda"


def test_explicit_ksp_pc_are_not_overridden_by_auto_preset() -> None:
    config = EITForwardModel._resolve_linear_backend_config(
        _model_with_dim(3),
        LinearBackendConfig.from_dict({"ksp_type": "minres", "pc_type": "jacobi"}),
    )

    assert config.solver_preset == "custom"
    assert config.ksp_type == "minres"
    assert config.pc_type == "jacobi"


# --- Canonical preset/PC matrix drift guard (T8) -------------------------------
# If anyone renames, adds or drops a preset, this test fails so SPEC.md §V.V6,
# the canonical solver/PC matrix, and the R11 documentation trail must be
# updated in lockstep before the change lands.

# Full expected preset shape. KSP/PC values use "?" when the preset can ride on
# the auto defaults; non-None entries hard-lock the driver/AMG type.
_CANONICAL_PRESET_MATRIX: dict[str, dict[str, str | None]] = {
    "custom": {
        "ksp_type": None,
        "pc_type": None,
        "pc_gamg_type": None,
        "pc_hypre_type": None,
        "pc_factor_mat_solver_type": None,
    },
    "direct": {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_gamg_type": None,
        "pc_hypre_type": None,
        "pc_factor_mat_solver_type": None,
    },
    "legacy_direct": {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_gamg_type": None,
        "pc_hypre_type": None,
        "pc_factor_mat_solver_type": None,
    },
    "debug_direct": {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_gamg_type": None,
        "pc_hypre_type": None,
        "pc_factor_mat_solver_type": None,
    },
    "mumps": {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_gamg_type": None,
        "pc_hypre_type": None,
        "pc_factor_mat_solver_type": "mumps",
    },
    "debug_mumps": {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_gamg_type": None,
        "pc_hypre_type": None,
        "pc_factor_mat_solver_type": "mumps",
    },
    "3d_gamg": {
        "ksp_type": "fgmres",
        "pc_type": "gamg",
        "pc_gamg_type": "agg",
        "pc_hypre_type": None,
        "pc_factor_mat_solver_type": None,
    },
    "3d_amg": {
        "ksp_type": "fgmres",
        "pc_type": "gamg",
        "pc_gamg_type": "agg",
        "pc_hypre_type": None,
        "pc_factor_mat_solver_type": None,
    },
    "3d_hypre": {
        "ksp_type": "fgmres",
        "pc_type": "hypre",
        "pc_gamg_type": None,
        "pc_hypre_type": "boomeramg",
        "pc_factor_mat_solver_type": None,
    },
    "hypre_boomeramg": {
        "ksp_type": "fgmres",
        "pc_type": "hypre",
        "pc_gamg_type": None,
        "pc_hypre_type": "boomeramg",
        "pc_factor_mat_solver_type": None,
    },
    "spd_gamg": {
        "ksp_type": "cg",
        "pc_type": "gamg",
        "pc_gamg_type": "agg",
        "pc_hypre_type": None,
        "pc_factor_mat_solver_type": None,
    },
    "spd_hypre": {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "pc_gamg_type": None,
        "pc_hypre_type": "boomeramg",
        "pc_factor_mat_solver_type": None,
    },
    "cg_hypre": {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "pc_gamg_type": None,
        "pc_hypre_type": "boomeramg",
        "pc_factor_mat_solver_type": None,
    },
    "amgx": {
        "ksp_type": "cg",
        "pc_type": "amgx",
        "pc_gamg_type": None,
        "pc_hypre_type": None,
        "pc_factor_mat_solver_type": None,
    },
    "cuda_amgx": {
        "ksp_type": "cg",
        "pc_type": "amgx",
        "pc_gamg_type": None,
        "pc_hypre_type": None,
        "pc_factor_mat_solver_type": None,
    },
}


def test_preset_name_set_matches_canonical_matrix() -> None:
    """T8: lock preset names; any change must update SPEC §V.V6 + canonical matrix."""
    import pytest

    for name, expected in _CANONICAL_PRESET_MATRIX.items():
        try:
            config = EITForwardModel._resolve_linear_backend_config(
                _model_with_dim(3),
                LinearBackendConfig.from_dict({"solver_preset": name}),
            )
        except Exception as exc:  # pragma: no cover - guarded by test
            pytest.fail(
                f"preset {name!r} unexpectedly raised {type(exc).__name__}: {exc}. "
                "If removed intentionally, update SPEC §V.V5/V6 + canonical matrix."
            )
        for field, expected_value in expected.items():
            if expected_value is None:
                continue
            assert getattr(config, field) == expected_value, (
                f"preset {name!r}: {field}={getattr(config, field)!r} != "
                f"{expected_value!r} (canonical matrix drift; update SPEC)."
            )


def test_unknown_preset_still_raises_with_sorted_choices() -> None:
    """T8 companion: V5 semantics must surface the full valid preset list."""
    import pytest

    with pytest.raises(ValueError) as exc_info:
        EITForwardModel._resolve_linear_backend_config(
            _model_with_dim(3),
            LinearBackendConfig.from_dict({"solver_preset": "not-a-real-preset"}),
        )
    message = str(exc_info.value)
    for name in _CANONICAL_PRESET_MATRIX:
        assert name in message, (
            f"error text missing canonical preset {name!r}; update resolver "
            "to list every supported preset."
        )


def test_auto_dispatch_matches_canonical_decision() -> None:
    """T8: auto dispatch rule (V6) stays 3d_gamg for tdim>=3 and direct otherwise."""
    for tdim, expected in (
        (1, "direct"),
        (2, "direct"),
        (3, "3d_gamg"),
        (4, "3d_gamg"),
    ):
        config = EITForwardModel._resolve_linear_backend_config(
            _model_with_dim(tdim),
            LinearBackendConfig(),
        )
        assert config.solver_preset == expected, (
            f"auto dispatch for tdim={tdim} resolved to {config.solver_preset!r}; "
            f"expected {expected!r} per SPEC §V.V6."
        )
