from __future__ import annotations

from eit_app.controllers import reconstruction_controller as rc
from eit_app.ui.dialogs.reconstruction_settings_panel import (
    FORWARD_SOLVER_PRESET_CHOICES,
)
from eit_app.ui.simulation.inverse_problem_panel import (
    SIMULATION_DEBUG_INVERSE_METHODS,
    normalize_simulation_inverse_method,
    simulation_inverse_methods_for_mesh_dimension,
)


def test_v663_reconstruction_runtime_preserves_tetra_cuda_solver_policy(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "gpu")
    monkeypatch.setattr(
        rc,
        "probe_petsc_cuda_runtime",
        lambda: {
            "petsc_cuda": True,
            "petsc_hypre": True,
            "petsc_amgx": True,
        },
    )

    tetra = rc._resolve_reconstruction_runtime(
        {"mesh_family": "tetra", "forward_backend": "cuda_structured"},
        mesh_dim=3,
    )

    assert tetra["mesh_family"] == "tetra"
    assert tetra["forward_backend"] == "dolfinx"
    assert tetra["petsc_device"] == "cuda"
    assert tetra["forward_solver_preset"] == "cuda_amgx"
    assert tetra["forward_solver_policy_reason"] == "tetra_real_cuda_amgx_default"
    assert tetra["forward_mat_solve"] == "off"
    assert (
        tetra["forward_mat_solve_policy_reason"]
        == "cuda_amgx_matsolve_disabled_mainline"
    )

    requested_amgx = rc._resolve_reconstruction_runtime(
        {"mesh_family": "tetra", "forward_solver_preset": "cuda_amgx"},
        mesh_dim=3,
    )
    assert requested_amgx["forward_solver_preset_requested"] == "cuda_amgx"
    assert requested_amgx["forward_solver_preset"] == "cuda_amgx"
    assert requested_amgx["forward_solver_policy_reason"] == ""

    complex_default = rc._resolve_reconstruction_runtime(
        {"mesh_family": "tetra", "background_conductivity": "1+0.2j"},
        mesh_dim=3,
    )
    assert complex_default["forward_solver_preset"] == "3d_gamg"
    assert (
        complex_default["forward_solver_policy_reason"]
        == "complex_cuda_native_gamg_default"
    )

    complex_strict = rc._resolve_reconstruction_runtime(
        {
            "mesh_family": "tetra",
            "background_conductivity": "1+0.2j",
            "complex_gpu_high_accuracy": True,
        },
        mesh_dim=3,
    )
    assert complex_strict["forward_solver_preset"] == "complex_block_real_amgx"
    assert (
        complex_strict["forward_solver_policy_reason"]
        == "complex_cuda_block_real_amgx_default"
    )

    hex_runtime = rc._resolve_reconstruction_runtime({"mesh_family": "hex"}, mesh_dim=3)
    assert hex_runtime["mesh_family"] == "hex"
    assert hex_runtime["forward_backend"] == "cuda_structured"


def test_v663_simulation_inverse_default_methods_hide_debug_routes() -> None:
    methods_2d = simulation_inverse_methods_for_mesh_dimension(2)
    methods_3d = simulation_inverse_methods_for_mesh_dimension(3)

    for debug_method in SIMULATION_DEBUG_INVERSE_METHODS:
        assert debug_method not in methods_2d
        assert debug_method not in methods_3d

    debug_methods_3d = simulation_inverse_methods_for_mesh_dimension(
        3,
        include_debug=True,
    )
    for debug_method in SIMULATION_DEBUG_INVERSE_METHODS:
        assert debug_method in debug_methods_3d

    assert normalize_simulation_inverse_method("eidors_one_step_noser") == "noser_rm"


def test_v663_gui_forward_solver_choices_hide_blacklisted_hypre_routes() -> None:
    assert "3d_hypre" not in FORWARD_SOLVER_PRESET_CHOICES
    assert "spd_hypre" not in FORWARD_SOLVER_PRESET_CHOICES
    assert "cg_hypre" not in FORWARD_SOLVER_PRESET_CHOICES
    assert "hypre_boomeramg" not in FORWARD_SOLVER_PRESET_CHOICES
    assert "cuda_amgx" in FORWARD_SOLVER_PRESET_CHOICES
    assert "complex_block_real_amgx" in FORWARD_SOLVER_PRESET_CHOICES
