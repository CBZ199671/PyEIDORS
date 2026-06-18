"""Tests for PETSc forward solver preset resolution."""

from __future__ import annotations

from pyeidors.forward.eit_forward_model import (
    EITForwardModel,
    LinearBackendConfig,
    _solver_route_metadata,
)


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


def test_cuda_amgx_preset_requests_cuda_fgmres_amgx() -> None:
    config = EITForwardModel._resolve_linear_backend_config(
        _model_with_dim(3),
        LinearBackendConfig.from_dict({"solver_preset": "cuda_amgx"}),
    )

    assert config.ksp_type == "fgmres"
    assert config.pc_type == "amgx"
    assert config.petsc_device == "cuda"
    assert config.petsc_options["pc_amgx_smoother"] == "JACOBI_L1"
    assert config.petsc_options["pc_amgx_exact_coarse_solve"] == "0"
    assert config.petsc_options["pc_amgx_presweeps"] == "2"
    assert config.petsc_options["pc_amgx_postsweeps"] == "2"
    assert config.petsc_options["pc_amgx_coarse_solver"] == "NOSOLVER"


def test_complex_cuda_amgx_preset_uses_native_complex_safe_options() -> None:
    config = EITForwardModel._resolve_linear_backend_config(
        _model_with_dim(3),
        LinearBackendConfig.from_dict({"solver_preset": "complex_cuda_amgx"}),
    )

    assert config.ksp_type == "fgmres"
    assert config.pc_type == "amgx"
    assert config.petsc_device == "cuda"
    assert config.petsc_options["pc_amgx_amg_method"] == "AGGREGATION"
    assert config.petsc_options["pc_amgx_selector"] == "SIZE_8"
    assert config.petsc_options["pc_amgx_smoother"] == "BLOCK_JACOBI"
    assert config.petsc_options["pc_amgx_exact_coarse_solve"] == "0"
    assert config.petsc_options["pc_amgx_presweeps"] == "2"
    assert config.petsc_options["pc_amgx_postsweeps"] == "2"
    assert config.petsc_options["pc_amgx_coarse_solver"] == "NOSOLVER"


def test_complex_cuda_amgx_route_metadata_marks_numeric_delta_experiment() -> None:
    meta = _solver_route_metadata("complex_cuda_amgx")

    assert meta["solver_route_family"] == "native_complex_amgx"
    assert meta["solver_route_status"] == "experimental_known_numeric_delta"
    assert "numerical differences versus CPU direct reference" in str(
        meta["solver_route_caveat"]
    )


def test_complex_block_real_amgx_route_metadata_marks_strict_accuracy() -> None:
    config = EITForwardModel._resolve_linear_backend_config(
        _model_with_dim(3),
        LinearBackendConfig.from_dict({"solver_preset": "complex_block_real_amgx"}),
    )
    meta = _solver_route_metadata("complex_block_real_amgx")

    assert config.solver_preset == "complex_block_real_amgx"
    assert config.petsc_device == "cuda"
    assert config.petsc_options["block_real_amgx_profile"] == "real_jacobi_l1"
    assert config.petsc_options["block_real_amgx_ksp_type"] == "bcgs"
    assert config.petsc_options["block_real_amgx_rtol"] == "1e-6"
    assert config.petsc_options["block_real_amgx_max_relative_residual"] == "1e-6"
    assert meta["solver_route_family"] == "complex_block_real_amgx"
    assert meta["solver_route_status"] == "strict_accuracy_complex_gpu"


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
        "ksp_type": "fgmres",
        "pc_type": "amgx",
        "pc_gamg_type": None,
        "pc_hypre_type": None,
        "pc_factor_mat_solver_type": None,
    },
    "cuda_amgx": {
        "ksp_type": "fgmres",
        "pc_type": "amgx",
        "pc_gamg_type": None,
        "pc_hypre_type": None,
        "pc_factor_mat_solver_type": None,
    },
    "complex_cuda_amgx": {
        "ksp_type": "fgmres",
        "pc_type": "amgx",
        "pc_gamg_type": None,
        "pc_hypre_type": None,
        "pc_factor_mat_solver_type": None,
    },
    "complex_block_real_amgx": {
        "ksp_type": "fgmres",
        "pc_type": "gamg",
        "pc_gamg_type": "agg",
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


def test_retired_mumps_presets_are_not_supported() -> None:
    """Retired 3D complex-CEM MUMPS routes must not re-enter preset surface."""
    import pytest

    for name in ("mumps", "debug_mumps"):
        with pytest.raises(ValueError, match="Unsupported PETSc solver_preset"):
            EITForwardModel._resolve_linear_backend_config(
                _model_with_dim(3),
                LinearBackendConfig.from_dict({"solver_preset": name}),
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
