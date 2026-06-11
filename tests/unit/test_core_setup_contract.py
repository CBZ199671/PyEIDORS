"""Contract tests for explicit EITSystem setup flows."""

from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

import pyeidors.core_system as core_module
from pyeidors.core_system import EITSystem
from pyeidors.data.structures import PatternConfig
from pyeidors.geometry.mesh3d_generator import create_cylinder_3d_eit_mesh


def _new_system() -> EITSystem:
    pattern = PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="normalized",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
        use_meas_current=False,
        rotate_meas=True,
    )
    return EITSystem(
        n_elec=16,
        pattern_config=pattern,
        contact_impedance=np.full(16, 1e-5, dtype=float),
        regularization_type="noser",
        regularization_alpha=1.0,
    )


def test_setup_requires_explicit_source():
    system = _new_system()
    with pytest.raises(ValueError, match="requires an explicit mesh source"):
        system.setup()


def test_setup_rejects_unknown_source():
    system = _new_system()
    with pytest.raises(ValueError, match="requires an explicit mesh source"):
        system.setup(mesh_source="unknown")


def test_setup_dispatches_cache(monkeypatch):
    system = _new_system()
    calls = {}

    def _fake_setup_from_cache(
        *,
        mesh_dir: str,
        mesh_name: str | None,
        gdim: int = 2,
        initialize_inverse: bool = True,
    ):
        calls["mesh_dir"] = mesh_dir
        calls["mesh_name"] = mesh_name
        calls["gdim"] = gdim
        calls["initialize_inverse"] = initialize_inverse

    monkeypatch.setattr(system, "setup_from_cache", _fake_setup_from_cache)
    system.setup(
        mesh_source="cache",
        mesh_dir="my_meshes",
        mesh_name="mesh_001",
        initialize_inverse=False,
    )

    assert calls["mesh_dir"] == "my_meshes"
    assert calls["mesh_name"] == "mesh_001"
    assert calls["gdim"] == 2
    assert calls["initialize_inverse"] is False


def test_setup_dispatches_generated(monkeypatch):
    system = _new_system()
    calls = {}

    def _fake_setup_generated_mesh(
        *,
        radius: float | None,
        mesh_size: float | None,
        dimension: int = 2,
        mesh_dir: str = "eit_meshes",
        height: float | None = None,
        electrode_coverage: float | None = None,
        electrode_height_ratio: float | None = None,
        electrode_level_fractions: tuple[float, ...] | list[float] | None = None,
        z_center: float | None = None,
        mesh_family: str | None = None,
        geometry_version: str | None = None,
        electrode_layout: str | None = None,
        initialize_inverse: bool = True,
    ):
        calls["radius"] = radius
        calls["mesh_size"] = mesh_size
        calls["dimension"] = dimension
        calls["mesh_dir"] = mesh_dir
        calls["height"] = height
        calls["electrode_coverage"] = electrode_coverage
        calls["electrode_height_ratio"] = electrode_height_ratio
        calls["electrode_level_fractions"] = electrode_level_fractions
        calls["z_center"] = z_center
        calls["mesh_family"] = mesh_family
        calls["geometry_version"] = geometry_version
        calls["electrode_layout"] = electrode_layout
        calls["initialize_inverse"] = initialize_inverse

    monkeypatch.setattr(system, "setup_generated_mesh", _fake_setup_generated_mesh)
    system.setup(
        mesh_source="generated",
        radius=1.5,
        mesh_size=0.08,
        mesh_family="hex",
        geometry_version="geomv2",
        initialize_inverse=False,
    )

    assert calls["radius"] == 1.5
    assert calls["mesh_size"] == 0.08
    assert calls["dimension"] == 2
    assert calls["mesh_dir"] == "eit_meshes"
    assert calls["mesh_family"] == "hex"
    assert calls["geometry_version"] == "geomv2"
    assert calls["electrode_coverage"] is None
    assert calls["initialize_inverse"] is False


def test_setup_with_mesh_type_guard():
    system = _new_system()
    with pytest.raises(TypeError, match="expects an EITMesh"):
        system.setup_with_mesh(object())  # type: ignore[arg-type]


def test_setup_with_mesh_normalizes_3d_drive_mode(tmp_path, monkeypatch):
    pattern = PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="line_current_density",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
    )
    system = EITSystem(
        n_elec=16,
        pattern_config=pattern,
        contact_impedance=np.full(16, 1e-5, dtype=float),
    )
    mesh = create_cylinder_3d_eit_mesh(
        n_elec=16,
        radius=0.18,
        height=0.16,
        refinement=1,
        electrode_coverage=0.5,
        electrode_height_ratio=0.2,
        output_dir=str(tmp_path),
        mesh_name="drive_mode_norm",
        mesh_family="hex",
        geometry_version="geomv2",
    )
    monkeypatch.setattr(
        system,
        "_initialize_components",
        lambda *, initialize_inverse=True: None,
    )
    system.setup_with_mesh(mesh)
    assert system.pattern_config.drive_mode == "total_current"
    assert system._pattern_config_diagnostics == {
        "drive_mode_requested": "line_current_density",
        "drive_mode_effective": "total_current",
    }


def test_setup_from_cache_calls_loader_paths(monkeypatch):
    system = _new_system()
    captured = {}

    class _FakeLoader:
        def __init__(self, mesh_dir: str, gdim: int = 2):
            captured["mesh_dir"] = mesh_dir
            captured["gdim"] = gdim

        def load_mesh(self, mesh_name: str):
            return f"mesh:{mesh_name}"

        def get_default_mesh(self):
            return "mesh:default"

    monkeypatch.setattr("pyeidors.core_system.MeshLoader", _FakeLoader)
    monkeypatch.setattr(
        system,
        "setup_with_mesh",
        lambda mesh, **_kwargs: captured.__setitem__("mesh", mesh),
    )

    system.setup_from_cache(mesh_dir="cache_dir", mesh_name="foo")
    assert captured["mesh_dir"] == "cache_dir"
    assert captured["gdim"] == 2
    assert captured["mesh"] == "mesh:foo"

    system.setup_from_cache(mesh_dir="cache_dir", mesh_name=None)
    assert captured["mesh"] == "mesh:default"


def test_setup_generated_mesh_uses_defaults_and_overrides(monkeypatch):
    system = _new_system()
    system.mesh_config.radius = 1.23
    system.mesh_config.mesh_size = 0.07
    system.mesh_config.electrode_coverage = 0.4

    generated_calls = []
    monkeypatch.setattr(
        "pyeidors.core_system.load_or_create_mesh",
        lambda **kwargs: generated_calls.append(kwargs) or "generated-mesh",
    )
    monkeypatch.setattr(
        system,
        "setup_with_mesh",
        lambda mesh, **kwargs: generated_calls.append({"mesh": mesh, **kwargs}),
    )

    system.setup_generated_mesh()
    system.setup_generated_mesh(radius=2.0, mesh_size=0.05)

    assert generated_calls[0]["radius"] == 1.23
    assert generated_calls[0]["refinement"] == 9
    assert generated_calls[0]["electrode_coverage"] == 0.4
    assert generated_calls[0]["mesh_dir"] == "eit_meshes"
    assert generated_calls[1]["mesh"] == "generated-mesh"
    assert generated_calls[1]["initialize_inverse"] is True
    assert generated_calls[2]["radius"] == 2.0
    assert generated_calls[2]["refinement"] == 20
    assert generated_calls[2]["electrode_coverage"] == 0.4

    system.setup_generated_mesh(electrode_coverage=0.6)
    assert generated_calls[4]["electrode_coverage"] == 0.6


def test_system_stores_public_device_policy():
    system = _new_system()
    assert system.device == "auto"
    assert system.forward_backend == "dolfinx"

    system_gpu = EITSystem(
        n_elec=16,
        pattern_config=system.pattern_config,
        contact_impedance=np.full(16, 1e-5, dtype=float),
        regularization_type="noser",
        regularization_alpha=1.0,
        device="cuda",
        forward_backend="cuda_structured",
    )
    assert system_gpu.device == "cuda"
    assert system_gpu.forward_backend == "cuda_structured"

    system_unknown = EITSystem(
        n_elec=16,
        pattern_config=system.pattern_config,
        contact_impedance=np.full(16, 1e-5, dtype=float),
        regularization_type="noser",
        regularization_alpha=1.0,
        device="cpu",
        forward_backend="unexpected",
    )
    assert system_unknown.device == "cpu"
    assert system_unknown.forward_backend == "dolfinx"


def test_forward_only_initialization_skips_inverse_components(monkeypatch):
    system = _new_system()
    system.mesh = SimpleNamespace(topology=SimpleNamespace(dim=2))

    class _FakeForwardModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.mesh = kwargs["mesh"]
            self.backend_diagnostic = {}

        def _set_backend_diagnostic(self, **kwargs) -> None:
            self.backend_diagnostic = kwargs

    def _fail_inverse_init(*_args, **_kwargs):
        raise AssertionError("forward-only setup must not build Jacobian")

    monkeypatch.setattr(core_module, "EITForwardModel", _FakeForwardModel)
    monkeypatch.setattr(core_module, "DirectJacobianCalculator", _fail_inverse_init)

    system._initialize_components(initialize_inverse=False)

    assert isinstance(system.fwd_model, _FakeForwardModel)
    assert system.reconstructor is None
    assert system._is_initialized is True
    system._require_initialized(require_reconstructor=False)
    with pytest.raises(RuntimeError, match="System not initialized"):
        system._require_initialized()


def test_setup_generated_mesh_prefers_hex_for_gpu3d_profile(monkeypatch):
    system = EITSystem(
        n_elec=16,
        pattern_config=PatternConfig(n_elec=16),
        contact_impedance=np.full(16, 1e-5, dtype=float),
        acceleration_profile="gpu3d",
    )
    generated_calls = []
    monkeypatch.setattr(
        "pyeidors.core_system.load_or_create_mesh",
        lambda **kwargs: generated_calls.append(kwargs) or "generated-3d-mesh",
    )
    monkeypatch.setattr(
        system,
        "setup_with_mesh",
        lambda mesh, **kwargs: generated_calls.append({"mesh": mesh, **kwargs}),
    )

    system.setup_generated_mesh(dimension=3, initialize_inverse=False)

    assert generated_calls[0]["dimension"] == 3
    assert generated_calls[0]["mesh_family"] == "hex"
    assert generated_calls[0]["geometry_version"] == "geomv2"
    assert generated_calls[1]["mesh"] == "generated-3d-mesh"
    assert generated_calls[1]["initialize_inverse"] is False


def test_setup_generated_mesh_uses_eidors_ring_order_for_multi_ring_3d(monkeypatch):
    system = EITSystem(
        n_elec=16,
        pattern_config=PatternConfig(n_elec=8, n_rings=2),
        contact_impedance=np.full(16, 1e-5, dtype=float),
    )
    generated_calls = []
    monkeypatch.setattr(
        "pyeidors.core_system.load_or_create_mesh",
        lambda **kwargs: generated_calls.append(kwargs) or "generated-3d-mesh",
    )
    monkeypatch.setattr(
        system,
        "setup_with_mesh",
        lambda mesh, **kwargs: generated_calls.append({"mesh": mesh, **kwargs}),
    )

    system.setup_generated_mesh(
        dimension=3,
        electrode_level_fractions=(0.25, 0.75),
    )

    assert generated_calls[0]["dimension"] == 3
    assert generated_calls[0]["n_elec"] == 16
    assert generated_calls[0]["electrode_layout"] == "ring_major"
    assert generated_calls[1]["mesh"] == "generated-3d-mesh"
    assert generated_calls[1]["initialize_inverse"] is True


def test_v632_setup_generated_3d_mesh_routes_through_disk_cache(monkeypatch):
    system = _new_system()
    system.mesh_config.radius = 0.18
    system.mesh_config.mesh_size = 0.045

    cache_calls = []
    monkeypatch.setattr(
        "pyeidors.core_system.load_or_create_mesh",
        lambda **kwargs: cache_calls.append(kwargs) or "cached-3d-mesh",
    )
    setup_calls = []
    monkeypatch.setattr(
        system,
        "setup_with_mesh",
        lambda mesh, **kwargs: setup_calls.append(mesh),
    )

    system.setup_generated_mesh(
        dimension=3,
        mesh_dir="custom_meshes",
        height=0.16,
        electrode_coverage=0.5,
        electrode_height_ratio=0.2,
        z_center=0.08,
    )

    assert setup_calls == ["cached-3d-mesh"]
    call = cache_calls[0]
    assert call["mesh_dir"] == "custom_meshes"
    assert call["dimension"] == 3
    assert call["n_elec"] == 16
    assert call["radius"] == 0.18
    assert call["height"] == 0.16
    assert call["refinement"] == 2
    assert call["electrode_coverage"] == 0.5
    assert call["electrode_height_ratio"] == 0.2
    assert call["z_center"] == 0.08


def test_runtime_policy_promotes_gpu3d_on_supported_structured_mesh():
    system = EITSystem(
        n_elec=16,
        pattern_config=PatternConfig(n_elec=16),
        contact_impedance=np.full(16, 1e-5, dtype=float),
        acceleration_profile="gpu3d",
    )
    system.mesh = SimpleNamespace(
        topology=SimpleNamespace(dim=3),
        mesh_family="hex",
        geometry_version="geomv2",
        generator_revision="g3d4",
        mesh_file="mesh.msh",
    )

    policy = system._resolve_runtime_policy()

    assert policy["acceleration_profile_effective"] == "gpu3d"
    assert policy["forward_backend_effective"] == "cuda_structured"
    assert policy["petsc_device_effective"] == "cuda"
    assert policy["device_effective"] == "cuda"
    assert policy["solver_mode_effective"] == "fast"
    assert policy["line_search_mode_effective"] == "fast"


def test_runtime_policy_routes_complex_gpu3d_to_dolfinx_petsc_cuda(monkeypatch):
    monkeypatch.setattr(core_module, "petsc_scalar_is_complex", lambda: True)
    monkeypatch.setattr(core_module, "petsc_scalar_dtype_name", lambda: "complex128")
    monkeypatch.setattr(
        core_module,
        "probe_petsc_cuda_runtime",
        lambda: {
            "petsc_cuda": True,
            "mat_type_name": "aijcusparse",
            "vec_type_name": "cuda",
            "dense_mat_type_name": "densecuda",
        },
    )
    system = EITSystem(
        n_elec=16,
        pattern_config=PatternConfig(n_elec=16),
        contact_impedance=np.full(16, 1e-5 + 2e-6j, dtype=np.complex128),
        acceleration_profile="gpu3d",
    )
    system.mesh = SimpleNamespace(
        topology=SimpleNamespace(dim=3),
        mesh_family="hex",
        geometry_version="geomv2",
        generator_revision="g3d4",
        mesh_file="mesh.msh",
    )

    policy = system._resolve_runtime_policy()

    assert policy["complex_admittivity_requested"] is True
    assert policy["petsc_scalar_is_complex"] is True
    assert policy["complex_route_effective"] is True
    assert policy["forward_backend_effective"] == "dolfinx"
    assert policy["petsc_device_effective"] == "cuda"
    assert policy["device_effective"] == "cuda"
    assert policy["forward_backend_fallback_reason"] is None


def test_runtime_policy_falls_back_from_cuda_structured_in_complex_runtime(
    monkeypatch,
):
    monkeypatch.setattr(core_module, "petsc_scalar_is_complex", lambda: True)
    monkeypatch.setattr(core_module, "petsc_scalar_dtype_name", lambda: "complex64")
    system = EITSystem(
        n_elec=16,
        pattern_config=PatternConfig(n_elec=16),
        contact_impedance=np.full(16, 1e-5, dtype=float),
        forward_backend="cuda_structured",
    )
    system.mesh = SimpleNamespace(
        topology=SimpleNamespace(dim=3),
        mesh_family="hex",
        geometry_version="geomv2",
        generator_revision="g3d4",
        mesh_file="mesh.msh",
    )

    policy = system._resolve_runtime_policy()

    assert policy["complex_admittivity_requested"] is False
    assert policy["petsc_scalar_is_complex"] is True
    assert policy["forward_backend_effective"] == "dolfinx"
    assert (
        policy["forward_backend_fallback_reason"]
        == "cuda_structured_unavailable_in_complex_petsc_runtime"
    )


def test_v83_runtime_policy_keeps_2d_auto_petsc_on_cpu_for_gpu_profile():
    system = EITSystem(
        n_elec=16,
        pattern_config=PatternConfig(n_elec=16),
        contact_impedance=np.full(16, 1e-5, dtype=float),
        acceleration_profile="gpu3d",
        petsc_device="auto",
    )
    system.mesh = SimpleNamespace(
        topology=SimpleNamespace(dim=2),
        mesh_family="simple",
        geometry_version="geomv2",
        generator_revision="g2d1",
        mesh_file="mesh.xdmf",
    )

    policy = system._resolve_runtime_policy()

    assert policy["acceleration_profile_effective"] == "default"
    assert policy["forward_backend_effective"] == "dolfinx"
    assert policy["petsc_device_requested"] == "auto"
    assert policy["petsc_device_effective"] == "cpu"
    assert policy["device_effective"] == "auto"


def test_runtime_policy_gpu3d_fused_enables_fused_defaults():
    system = EITSystem(
        n_elec=16,
        pattern_config=PatternConfig(n_elec=16),
        contact_impedance=np.full(16, 1e-5, dtype=float),
        acceleration_profile="gpu3d_fused",
    )
    system.mesh = SimpleNamespace(
        topology=SimpleNamespace(dim=3),
        mesh_family="tetra",
        geometry_version="geomv2",
        generator_revision="g3d2",
        mesh_file="mesh.msh",
    )

    policy = system._resolve_runtime_policy()

    assert policy["forward_backend_effective"] == "dolfinx"
    assert policy["petsc_device_effective"] == "cuda"
    assert policy["device_effective"] == "cuda"
    assert policy["rom_mode_effective"] == "on"
    assert policy["inexact_mode_effective"] == "auto"
    assert policy["lowrank_mode_effective"] == "auto"


def test_runtime_policy_downgrades_missing_amgx_to_spd_gamg(monkeypatch):
    monkeypatch.setattr(
        "pyeidors.core_system.probe_petsc_cuda_runtime",
        lambda: {
            "petsc_cuda": True,
            "petsc_hypre": True,
            "petsc_amgx": False,
        },
    )
    system = EITSystem(
        n_elec=16,
        pattern_config=PatternConfig(n_elec=16),
        contact_impedance=np.full(16, 1e-5, dtype=float),
        acceleration_profile="gpu3d",
        linear_backend_config={"solver_preset": "cuda_amgx"},
    )
    system.mesh = SimpleNamespace(
        topology=SimpleNamespace(dim=3),
        mesh_family="tetra",
        geometry_version="geomv2",
        generator_revision="g3d2",
        mesh_file="mesh.msh",
    )

    policy = system._resolve_runtime_policy()

    assert policy["forward_solver_preset_requested"] == "cuda_amgx"
    assert policy["forward_solver_preset_effective"] == "spd_gamg"
    assert (
        policy["forward_solver_policy_reason"]
        == "amgx_unavailable_downgraded_to_spd_gamg"
    )
    assert policy["petsc_amgx_available"] is False
    assert policy["forward_mat_solve_effective_policy"] == "off"
    assert (
        policy["forward_mat_solve_policy_reason"]
        == "cuda_spd_gamg_matsolve_disabled_b6"
    )


def test_runtime_policy_blacklists_hypre_cuda_to_spd_gamg(monkeypatch):
    monkeypatch.setattr(
        "pyeidors.core_system.probe_petsc_cuda_runtime",
        lambda: {
            "petsc_cuda": True,
            "petsc_hypre": True,
            "petsc_amgx": False,
        },
    )
    system = EITSystem(
        n_elec=16,
        pattern_config=PatternConfig(n_elec=16),
        contact_impedance=np.full(16, 1e-5, dtype=float),
        acceleration_profile="gpu3d",
        linear_backend_config={"solver_preset": "spd_hypre"},
    )
    system.mesh = SimpleNamespace(
        topology=SimpleNamespace(dim=3),
        mesh_family="tetra",
        geometry_version="geomv2",
        generator_revision="g3d2",
        mesh_file="mesh.msh",
    )

    policy = system._resolve_runtime_policy()

    assert policy["forward_solver_preset_effective"] == "spd_gamg"
    assert policy["forward_solver_policy_reason"] == "hypre_cuda_blacklisted_sigsegv_b4"
    assert policy["petsc_hypre_cuda_blacklisted"] is True
    assert policy["forward_mat_solve_effective_policy"] == "off"


def test_runtime_policy_preserves_explicit_cuda_matsolve_on(monkeypatch):
    monkeypatch.setattr(
        "pyeidors.core_system.probe_petsc_cuda_runtime",
        lambda: {
            "petsc_cuda": True,
            "petsc_hypre": True,
            "petsc_amgx": False,
        },
    )
    system = EITSystem(
        n_elec=16,
        pattern_config=PatternConfig(n_elec=16),
        contact_impedance=np.full(16, 1e-5, dtype=float),
        acceleration_profile="gpu3d",
        linear_backend_config={
            "solver_preset": "spd_gamg",
            "mat_solve_mode": "on",
        },
    )
    system.mesh = SimpleNamespace(
        topology=SimpleNamespace(dim=3),
        mesh_family="tetra",
        geometry_version="geomv2",
        generator_revision="g3d2",
        mesh_file="mesh.msh",
    )

    policy = system._resolve_runtime_policy()

    assert policy["forward_solver_preset_effective"] == "spd_gamg"
    assert policy["forward_mat_solve_requested"] == "on"
    assert policy["forward_mat_solve_effective_policy"] == "on"
    assert policy["forward_mat_solve_policy_reason"] == ""


def test_system_cache_lifecycle_defaults_to_session_and_supports_persistent():
    system = _new_system()
    assert system.cache_lifecycle == "session"
    assert system.cache_manager.disk_lifecycle == "session"
    assert system.cache_manager.session_cache_enabled is True

    persistent = EITSystem(
        n_elec=16,
        pattern_config=system.pattern_config,
        contact_impedance=np.full(16, 1e-5, dtype=float),
        regularization_type="noser",
        regularization_alpha=1.0,
        cache_lifecycle="persistent",
    )
    assert persistent.cache_lifecycle == "persistent"
    assert persistent.cache_manager.disk_lifecycle == "persistent"
    assert persistent.cache_manager.session_cache_enabled is False


def test_system_stores_eidors_alignment_defaults():
    system = _new_system()
    assert system.difference_preset == "eidors_one_step_noser"
    assert system.absolute_preset == "eidors_abs_gn"
    assert system.hyperparameter is None
    assert system.jacobian_background_conductivity == system.base_conductivity
    assert system.difference_step_size_mode == "off"
    assert system.best_homog_mode == "off"
