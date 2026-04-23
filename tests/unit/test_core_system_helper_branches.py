"""Additional branch coverage for core system orchestration helpers."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import pyeidors.core_system as core_module
from pyeidors.cache import CachePolicy
from pyeidors.core_system import (
    EITSystem,
    _matches_previous,
    _normalize_absolute_preset,
    _normalize_best_homog_mode,
    _normalize_bounds,
    _normalize_difference_preset,
    _normalize_difference_step_size_mode,
)
from pyeidors.data.structures import PatternConfig


def _pattern() -> PatternConfig:
    return PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="line_current_density",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
    )


def _new_system(**kwargs) -> EITSystem:
    return EITSystem(
        n_elec=16,
        pattern_config=_pattern(),
        contact_impedance=np.full(16, 1e-5, dtype=float),
        regularization_type="noser",
        regularization_alpha=1.0,
        **kwargs,
    )


def _bare_system() -> EITSystem:
    system = EITSystem.__new__(EITSystem)
    system.n_elec = 16
    system.pattern_config = _pattern()
    system.contact_impedance = np.full(16, 1e-5, dtype=float)
    system.base_conductivity = 1.0
    system.difference_mode = "normalized"
    system.difference_orientation = "target_minus_reference"
    system.regularization_type = "noser"
    system.regularization_alpha = 1.0
    system.hyperparameter = None
    system.jacobian_background_conductivity = 1.0
    system.noser_exponent = 0.5
    system.noser_floor = 1e-12
    system.difference_step_size_mode = "off"
    system._difference_step_size_mode_explicit = False
    system.difference_step_size_value = None
    system.difference_step_size_bounds = (0.0, 4.0)
    system.difference_step_size_fmin_options = {"maxiter": 10}
    system.difference_preset = "eidors_one_step_noser"
    system.absolute_preset = "eidors_abs_gn"
    system.best_homog_mode = "off"
    system._best_homog_mode_explicit = False
    system.linear_backend = "petsc"
    system.linear_backend_config = {"petsc_device": "cpu"}
    system.forward_backend = "dolfinx"
    system.performance_mode = "aggressive"
    system.device = "auto"
    system.solver_mode = "fast"
    system.linear_solver = "auto"
    system.jacobian_update_every = 1
    system.jacobian_reuse_tol = 0.0
    system.line_search_mode = "full"
    system.preconditioner = "auto"
    system.fast_linear_path = "auto"
    system.rom_mode = "off"
    system.rom_rank_global = 32
    system.rom_rank_adaptive = 16
    system.rom_refresh_every = 2
    system.rom_snapshot_source = "hybrid"
    system.inexact_mode = "off"
    system.inexact_forcing = "eisenstat-walker"
    system.inexact_eta0 = 0.2
    system.inexact_eta_min = 0.01
    system.inexact_eta_max = 0.9
    system.lowrank_mode = "off"
    system.lowrank_rank = 16
    system.lowrank_method = "tsvd"
    system.lowrank_energy = 0.995
    system.absolute_startup_cache = True
    system.cholmod_max_n = 50000
    system.cholmod_max_memory_gib = 4.0
    system.jacobian_block_tune = "auto"
    system.jacobian_block_size = 0
    system.jacobian_block_candidates = (64, 128, 256)
    system.cache_manager = SimpleNamespace(
        stats=lambda: {"disk_hits": 1},
        clear=lambda scope="both": None,
    )
    system.cache_policy = CachePolicy(disk_lifecycle="session")
    system.cache_lifecycle = "session"
    system.mesh_config = SimpleNamespace(
        radius=1.0,
        mesh_size=0.1,
        height=2.0,
        electrode_height_ratio=0.25,
        electrode_level_fractions=(0.3, 0.7),
        z_center=0.1,
        mesh_family="hex",
        geometry_version="geomv2",
    )
    system.mesh = None
    system.fwd_model = None
    system.reconstructor = None
    system._pattern_config_diagnostics = {
        "drive_mode_requested": "line_current_density",
        "drive_mode_effective": "line_current_density",
    }
    system._is_initialized = False
    system._last_reconstructor_controls = {}
    system._active_inverse_preset_name = None
    return system


def test_normalizer_helpers_and_matches_previous_cover_edge_cases():
    assert _normalize_difference_preset(None) == "eidors_one_step_noser"
    assert _normalize_difference_preset(" EIDORS_DEMO3D_TV ") == "eidors_demo3d_tv"
    with pytest.raises(ValueError, match="Unsupported difference_preset"):
        _normalize_difference_preset("bad")

    assert _normalize_absolute_preset(None) == "eidors_abs_gn"
    with pytest.raises(ValueError, match="Unsupported absolute_preset"):
        _normalize_absolute_preset("bad")

    assert _normalize_difference_step_size_mode(None) == "off"
    assert _normalize_difference_step_size_mode("FIXED") == "fixed"
    with pytest.raises(ValueError, match="Unsupported difference_step_size_mode"):
        _normalize_difference_step_size_mode("bad")

    assert _normalize_best_homog_mode("on") == "optimize"
    assert _normalize_best_homog_mode("off") == "off"
    with pytest.raises(ValueError, match="Unsupported best_homog_mode"):
        _normalize_best_homog_mode("bad")

    assert _normalize_bounds(None) == (0.0, 4.0)
    assert _normalize_bounds([0.1, 1.5]) == (0.1, 1.5)
    with pytest.raises(ValueError, match="exactly two values"):
        _normalize_bounds([0.1])
    with pytest.raises(ValueError, match="finite lower < upper"):
        _normalize_bounds([1.0, 1.0])
    with pytest.raises(ValueError, match="finite lower < upper"):
        _normalize_bounds([np.nan, 2.0])

    assert _matches_previous("same", None) is True
    assert _matches_previous(np.array([1.0, 2.0]), np.array([1.0, 2.0])) is True
    assert _matches_previous((1, 2), [1, 2]) is True
    assert _matches_previous({"a": 1}, {"a": 1}) is True
    assert _matches_previous(1.0, 1.0 + 1e-13) is True
    assert _matches_previous("left", "right") is False


def test_constructor_validation_and_cache_policy_override():
    with pytest.raises(ValueError, match="Unsupported jacobian_block_tune"):
        _new_system(jacobian_block_tune="bad")
    with pytest.raises(ValueError, match="Unsupported performance_mode"):
        _new_system(performance_mode="bad")

    policy = CachePolicy(disk_lifecycle="persistent")
    system = _new_system(cache_policy=policy, cache_lifecycle="session")
    assert system.cache_policy.disk_lifecycle == "session"
    assert policy.disk_lifecycle == "persistent"


def test_setup_generated_mesh_3d_and_initialize_components_branches(
    monkeypatch: pytest.MonkeyPatch,
):
    system = _new_system()
    system.mesh_config.height = 0.3
    system.mesh_config.electrode_height_ratio = 0.2
    system.mesh_config.electrode_level_fractions = (0.25, 0.75)
    system.mesh_config.z_center = -0.1
    system.mesh_config.mesh_family = "tetra"
    system.mesh_config.geometry_version = "legacy"

    generated_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        core_module,
        "create_cylinder_3d_eit_mesh",
        lambda **kwargs: generated_calls.append(dict(kwargs)) or "mesh3d",
    )
    captured_mesh = {}
    monkeypatch.setattr(
        system, "setup_with_mesh", lambda mesh: captured_mesh.__setitem__("mesh", mesh)
    )

    system.setup_generated_mesh(dimension=3, radius=0.4, mesh_size=0.04)
    assert captured_mesh["mesh"] == "mesh3d"
    assert generated_calls[0]["height"] == 0.3
    assert generated_calls[0]["electrode_height_ratio"] == 0.2
    assert generated_calls[0]["electrode_level_fractions"] == (0.25, 0.75)
    assert generated_calls[0]["z_center"] == -0.1
    assert generated_calls[0]["mesh_family"] == "tetra"
    assert generated_calls[0]["geometry_version"] == "legacy"
    assert generated_calls[0]["refinement"] >= 2

    with pytest.raises(ValueError, match="dimension must be 2 or 3"):
        system.setup_generated_mesh(dimension=5)

    bare = _bare_system()
    with pytest.raises(RuntimeError, match="without mesh"):
        bare._initialize_components()

    bare.mesh = SimpleNamespace(topology=SimpleNamespace(dim=2))

    class _FakeForward:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)
            self.backend_diag = {}

        def _set_backend_diagnostic(self, **kwargs):
            self.backend_diag.update(kwargs)

    class _FakeJacobian:
        def __init__(self, fwd_model, **kwargs):
            self.fwd_model = fwd_model
            self.kwargs = dict(kwargs)

    class _FakeRegularization:
        def __init__(self, kind: str):
            self.kind = kind

    class _FakeReconstructor:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.max_iterations = 2
            self.min_step = 0.0
            self.max_step = 1.0
            self.line_search_mode = kwargs["line_search_mode"]
            self.difference_step_size_fmin_options = dict(
                kwargs["difference_step_size_fmin_options"]
            )
            self.best_homog_mode = kwargs["best_homog_mode"]

        def set_regularization(self, regularization):
            self.regularization = regularization

    monkeypatch.setattr(core_module, "EITForwardModel", _FakeForward)
    monkeypatch.setattr(core_module, "DirectJacobianCalculator", _FakeJacobian)
    monkeypatch.setattr(
        bare,
        "_build_regularization",
        lambda jac, regularization_type=None: _FakeRegularization(
            regularization_type or "noser"
        ),
    )
    monkeypatch.setattr(core_module, "GaussNewtonReconstructor", _FakeReconstructor)

    bare._initialize_components()
    assert bare._is_initialized is True
    assert bare.fwd_model.backend_diag["drive_mode_effective"] == "line_current_density"
    assert bare.reconstructor.regularization.kind == "noser"
    assert bare._last_reconstructor_controls["line_search_mode"] == "full"


def test_runtime_policy_helper_guards_cover_missing_mesh_branch():
    bare = _bare_system()
    assert bare._supports_cuda_structured_backend() is False
    with pytest.raises(RuntimeError, match="before mesh setup"):
        bare._resolve_runtime_policy()


def test_regularization_and_preset_application_helpers(monkeypatch: pytest.MonkeyPatch):
    system = _bare_system()
    system.fwd_model = SimpleNamespace(name="fwd")

    monkeypatch.setattr(
        core_module,
        "NOSERRegularization",
        lambda *args, **kwargs: ("noser", args, kwargs),
    )
    monkeypatch.setattr(
        core_module,
        "TikhonovRegularization",
        lambda *args, **kwargs: ("tikhonov", args, kwargs),
    )
    monkeypatch.setattr(
        core_module,
        "SmoothnessRegularization",
        lambda *args, **kwargs: ("smoothness", args, kwargs),
    )
    monkeypatch.setattr(
        core_module,
        "TotalVariationRegularization",
        lambda *args, **kwargs: ("tv", args, kwargs),
    )

    jac = SimpleNamespace(name="jac")
    assert system._build_regularization(jac)[0] == "noser"
    assert (
        system._build_regularization(jac, regularization_type="tikhonov")[0]
        == "tikhonov"
    )
    assert (
        system._build_regularization(jac, regularization_type="smoothness")[0]
        == "smoothness"
    )
    assert system._build_regularization(jac, regularization_type="tv")[0] == "tv"
    with pytest.raises(ValueError, match="Unsupported regularization_type"):
        system._build_regularization(jac, regularization_type="bad")

    system.reconstructor = None
    assert system._capture_reconstructor_controls() == {}
    assert system._default_hyperparameter("eidors_one_step_noser") == 1e-1
    assert np.isclose(system._default_hyperparameter("eidors_abs_gn"), np.sqrt(1e-3))
    assert np.isclose(
        system._default_hyperparameter("sphere_multistep_noser"), np.sqrt(1e-3)
    )
    assert system._default_hyperparameter("eidors_demo3d_tv") == 1e-2
    system.hyperparameter = 0.25
    assert system._default_hyperparameter("anything") == 0.25
    system.hyperparameter = None

    system.difference_preset = "eidors_demo3d_tv"
    diff_tv = system._preset_config("difference")
    assert diff_tv["regularization_type"] == "tv"
    assert diff_tv["difference_step_size_mode"] == "optimize"

    system.difference_preset = "sphere_multistep_noser"
    diff_sphere = system._preset_config("difference")
    assert diff_sphere["max_iterations"] == 3
    assert diff_sphere["difference_step_size_mode"] == "off"

    absolute_cfg = system._preset_config("absolute")
    assert absolute_cfg["best_homog_mode"] == "optimize"

    with pytest.raises(RuntimeError, match="Reconstructor is not initialized"):
        system._apply_inverse_preset("difference")

    system.difference_preset = "eidors_demo3d_tv"
    calls: list[object] = []

    class _FakeTVReg:
        pass

    reconstructor = SimpleNamespace(
        regularization=object(),
        jacobian_calculator=SimpleNamespace(name="jac"),
        hyperparameter=0.5,
        max_iterations=99,
        jacobian_update_every=1,
        jacobian_reuse_tol=0.0,
        line_search_mode="keep",
        difference_step_size_mode="off",
        difference_step_size_value=None,
        difference_step_size_bounds=(0.0, 4.0),
        difference_step_size_fmin_options={"maxiter": 10},
        best_homog_mode="off",
        min_step=0.0,
        max_step=1.0,
        set_regularization=lambda reg: calls.append(reg),
    )
    system.reconstructor = reconstructor
    system.regularization_type = "noser"
    system._last_reconstructor_controls = {
        "hyperparameter": 0.5,
        "max_iterations": 5,
        "jacobian_update_every": 1,
        "jacobian_reuse_tol": 0.0,
        "line_search_mode": "keep",
        "difference_step_size_mode": "off",
        "difference_step_size_value": None,
        "difference_step_size_bounds": (0.0, 4.0),
        "difference_step_size_fmin_options": {"maxiter": 10},
        "best_homog_mode": "off",
        "min_step": 0.0,
        "max_step": 1.0,
    }
    monkeypatch.setattr(
        system,
        "_build_regularization",
        lambda jacobian_calculator, regularization_type=None: _FakeTVReg(),
    )
    monkeypatch.setattr(core_module, "TotalVariationRegularization", _FakeTVReg)

    system._apply_inverse_preset("difference")
    assert isinstance(calls[0], _FakeTVReg)
    assert reconstructor.hyperparameter == 1e-2
    assert reconstructor.max_iterations == 99
    assert reconstructor.active_preset_name == "eidors_demo3d_tv"
    assert system._active_inverse_preset_name == "eidors_demo3d_tv"


def test_runtime_wrappers_precheck_and_cache_helpers(monkeypatch: pytest.MonkeyPatch):
    system = _bare_system()
    with pytest.raises(RuntimeError, match="System not initialized"):
        system._require_initialized()

    system._is_initialized = True
    system.fwd_model = SimpleNamespace(fwd_solve=lambda image: ("data-out", "voltages"))
    system.reconstructor = SimpleNamespace(
        ensure_regularization_ready=lambda: None,
        reconstruct=lambda diff_data, initial_guess: (
            "recon",
            diff_data,
            initial_guess,
        ),
    )

    monkeypatch.setattr(
        core_module,
        "conductivity_to_image",
        lambda fwd_model, conductivity: ("image", conductivity),
    )
    assert system.forward_solve(np.array([1.0], dtype=float)) == "data-out"

    applied_modes: list[str] = []
    monkeypatch.setattr(
        system,
        "_apply_inverse_preset",
        lambda inverse_mode: applied_modes.append(inverse_mode),
    )
    monkeypatch.setattr(
        core_module,
        "difference_measurement",
        lambda data, reference_data, mode, orientation: {
            "data": data,
            "reference_data": reference_data,
            "mode": mode,
            "orientation": orientation,
        },
    )
    diff_out = system.inverse_solve(
        "target", reference_data="reference", initial_guess=np.array([0.2], dtype=float)
    )
    assert applied_modes[-1] == "difference"
    assert diff_out[1]["reference_data"] == "reference"

    abs_out = system.inverse_solve("target", reference_data=None, initial_guess=None)
    assert applied_modes[-1] == "absolute"
    assert abs_out[1]["reference_data"] is None

    system.reconstructor.ensure_regularization_ready = lambda: (_ for _ in ()).throw(
        ValueError("boom")
    )
    with pytest.raises(RuntimeError, match="regularization warmup failed: boom"):
        system.inverse_solve("target")

    report = SimpleNamespace(has_errors=True, summary_lines=lambda: ["bad-a", "bad-b"])
    monkeypatch.setattr(
        core_module,
        "run_unit_consistency_checks",
        lambda fwd_model, expected_domain_size_m=None: report,
    )
    assert system.run_unit_precheck(expected_domain_size_m=0.2, strict=False) is report
    with pytest.raises(ValueError, match="bad-a \\| bad-b"):
        system.run_unit_precheck(expected_domain_size_m=0.2, strict=True)

    clear_calls: list[str] = []
    system.cache_manager = SimpleNamespace(
        stats=lambda: {"disk_hits": 2},
        clear=lambda scope="both": clear_calls.append(f"cache:{scope}"),
    )
    monkeypatch.setattr(
        core_module, "process_forward_setup_cache_stats", lambda: {"warm": 1}
    )
    monkeypatch.setattr(
        core_module, "clear_process_mesh_cache", lambda: clear_calls.append("mesh")
    )
    monkeypatch.setattr(
        core_module,
        "clear_process_forward_setup_cache",
        lambda: clear_calls.append("forward"),
    )
    assert system.get_cache_stats() == {
        "disk_hits": 2,
        "process_forward_setup_cache": {"warm": 1},
    }

    system.clear_cache(scope="disk")
    assert clear_calls == ["cache:disk"]
    system.clear_cache(scope="process")
    assert clear_calls[-3:] == ["cache:process", "mesh", "forward"]
