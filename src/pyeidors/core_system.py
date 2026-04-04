"""PyEIDORS core system orchestration."""

from __future__ import annotations

from dataclasses import replace
import logging
from typing import Any, Dict, Optional, Union

import numpy as np

from .cache import (
    DEFAULT_CACHE_LIFECYCLE,
    CacheManager,
    CachePolicy,
    CacheScope,
    normalize_cache_lifecycle,
)
from .core_system_facade import CoreSystemFacadeMixin
from .core_system_helpers import (
    conductivity_to_image,
    difference_measurement,
)
from .data.difference import (
    DEFAULT_DIFFERENCE_MODE,
    DEFAULT_DIFFERENCE_ORIENTATION,
    normalize_difference_mode,
    normalize_difference_orientation,
)
from .data.structures import EITData, EITImage, EITMesh, MeshConfig, PatternConfig
from .forward.eit_forward_model import EITForwardModel
from .forward.process_setup_cache import (
    clear_process_forward_setup_cache,
    process_forward_setup_cache_stats,
)
from .geometry.mesh_loader import MeshLoader
from .geometry.process_mesh_cache import clear_process_mesh_cache
from .geometry.mesh3d_generator import create_cylinder_3d_eit_mesh
from .geometry.simple_mesh_generator import create_simple_eit_mesh
from .inverse.contracts import SolverOutput
from .inverse.jacobian.direct_jacobian import DirectJacobianCalculator
from .inverse.regularization.smoothness import (
    NOSERRegularization,
    SmoothnessRegularization,
    TotalVariationRegularization,
    TikhonovRegularization,
)
from .inverse.solvers.gauss_newton import GaussNewtonReconstructor
from .inverse.solvers.gauss_newton_device import normalize_runtime_device
from .physics import UnitCheckReport, run_unit_consistency_checks
from .physics.current_drive import normalize_pattern_config_for_mesh
from .perf.policy import (
    DEFAULT_3D_GEOMETRY_VERSION,
    DEFAULT_CHOLMOD_MAX_MEMORY_GIB,
    DEFAULT_CHOLMOD_MAX_N,
    DEFAULT_FORWARD_BACKEND,
    DEFAULT_MESH_FAMILY,
    DEFAULT_INEXACT_ETA0,
    DEFAULT_INEXACT_ETA_MAX,
    DEFAULT_INEXACT_ETA_MIN,
    DEFAULT_INEXACT_FORCING,
    DEFAULT_INEXACT_MODE,
    DEFAULT_JACOBIAN_BLOCK_CANDIDATES,
    DEFAULT_JACOBIAN_BLOCK_SIZE,
    DEFAULT_JACOBIAN_BLOCK_TUNE,
    DEFAULT_LOWRANK_ENERGY,
    DEFAULT_LOWRANK_METHOD,
    DEFAULT_LOWRANK_MODE,
    DEFAULT_LOWRANK_RANK,
    DEFAULT_PETSC_DEVICE,
    DEFAULT_PRECONDITIONER,
    DEFAULT_ROM_MODE,
    DEFAULT_ROM_RANK_ADAPTIVE,
    DEFAULT_ROM_RANK_GLOBAL,
    DEFAULT_ROM_REFRESH_EVERY,
    DEFAULT_ROM_SNAPSHOT_SOURCE,
    normalize_forward_backend,
    normalize_mesh_family,
    normalize_petsc_device,
)

logger = logging.getLogger(__name__)


DEFAULT_DIFFERENCE_PRESET = "eidors_one_step_noser"
DEFAULT_ABSOLUTE_PRESET = "eidors_abs_gn"
_VALID_DIFFERENCE_PRESETS = {
    "eidors_one_step_noser",
    "eidors_demo3d_tv",
    "sphere_multistep_noser",
}
_VALID_ABSOLUTE_PRESETS = {"eidors_abs_gn"}
_VALID_DIFFERENCE_STEP_SIZE_MODES = {"off", "optimize", "fixed"}
_VALID_BEST_HOMOG_MODES = {"off", "optimize", "on"}


def _normalize_difference_preset(name: str | None) -> str:
    resolved = str(name or DEFAULT_DIFFERENCE_PRESET).strip().lower()
    if resolved not in _VALID_DIFFERENCE_PRESETS:
        raise ValueError(
            f"Unsupported difference_preset={name!r}. "
            "Expected one of: 'eidors_one_step_noser', 'eidors_demo3d_tv', 'sphere_multistep_noser'."
        )
    return resolved


def _normalize_absolute_preset(name: str | None) -> str:
    resolved = str(name or DEFAULT_ABSOLUTE_PRESET).strip().lower()
    if resolved not in _VALID_ABSOLUTE_PRESETS:
        raise ValueError(
            f"Unsupported absolute_preset={name!r}. Expected 'eidors_abs_gn'."
        )
    return resolved


def _normalize_difference_step_size_mode(mode: str | None) -> str:
    resolved = str(mode or "off").strip().lower()
    if resolved not in _VALID_DIFFERENCE_STEP_SIZE_MODES:
        raise ValueError(
            f"Unsupported difference_step_size_mode={mode!r}. "
            "Expected one of: 'off', 'optimize', 'fixed'."
        )
    return resolved


def _normalize_best_homog_mode(mode: str | None) -> str:
    resolved = str(mode or "off").strip().lower()
    if resolved == "on":
        return "optimize"
    if resolved not in _VALID_BEST_HOMOG_MODES:
        raise ValueError(
            f"Unsupported best_homog_mode={mode!r}. Expected one of: 'off', 'optimize'."
        )
    return resolved


def _normalize_bounds(bounds: tuple[float, float] | list[float] | None) -> tuple[float, float]:
    if bounds is None:
        return (0.0, 4.0)
    if len(bounds) != 2:
        raise ValueError("difference_step_size_bounds must contain exactly two values.")
    lower, upper = float(bounds[0]), float(bounds[1])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("difference_step_size_bounds must satisfy finite lower < upper.")
    return (lower, upper)


def _matches_previous(current: Any, previous: Any) -> bool:
    if previous is None:
        return True
    if isinstance(current, np.ndarray) or isinstance(previous, np.ndarray):
        return np.array_equal(np.asarray(current), np.asarray(previous))
    if isinstance(current, (list, tuple)) or isinstance(previous, (list, tuple)):
        return tuple(current) == tuple(previous)
    if isinstance(current, dict) or isinstance(previous, dict):
        return dict(current or {}) == dict(previous or {})
    if isinstance(current, (int, float)) and isinstance(previous, (int, float)):
        return bool(np.isclose(float(current), float(previous), rtol=1e-12, atol=1e-12))
    return current == previous


class EITSystem(CoreSystemFacadeMixin):
    """Facade that wires mesh, forward model and reconstruction workflow."""

    def __init__(
        self,
        n_elec: int = 16,
        pattern_config: Optional[PatternConfig] = None,
        mesh_config: Optional[MeshConfig] = None,
        contact_impedance: Optional[np.ndarray] = None,
        base_conductivity: float = 1.0,
        difference_mode: str = DEFAULT_DIFFERENCE_MODE,
        difference_orientation: str = DEFAULT_DIFFERENCE_ORIENTATION,
        regularization_type: str = "noser",
        regularization_alpha: float = 1.0,
        hyperparameter: float | None = None,
        jacobian_background_conductivity: float | None = None,
        noser_exponent: float = 0.5,
        noser_floor: float = 1e-12,
        difference_step_size_mode: str | None = None,
        difference_step_size_value: float | None = None,
        difference_step_size_bounds: tuple[float, float] | list[float] | None = None,
        difference_step_size_fmin_options: Optional[dict[str, Any]] = None,
        difference_preset: str = DEFAULT_DIFFERENCE_PRESET,
        absolute_preset: str = DEFAULT_ABSOLUTE_PRESET,
        best_homog_mode: str | None = None,
        linear_backend: str = "petsc",
        linear_backend_config: Optional[dict[str, Any]] = None,
        forward_backend: str = DEFAULT_FORWARD_BACKEND,
        mesh_family: str = DEFAULT_MESH_FAMILY,
        petsc_device: str = DEFAULT_PETSC_DEVICE,
        device: str = "auto",
        performance_mode: str = "aggressive",
        solver_mode: str = "strict",
        linear_solver: str = "auto",
        jacobian_update_every: int = 1,
        jacobian_reuse_tol: float = 0.0,
        line_search_mode: str = "full",
        preconditioner: str = DEFAULT_PRECONDITIONER,
        fast_linear_path: str = "auto",
        rom_mode: str = DEFAULT_ROM_MODE,
        rom_rank_global: int = DEFAULT_ROM_RANK_GLOBAL,
        rom_rank_adaptive: int = DEFAULT_ROM_RANK_ADAPTIVE,
        rom_refresh_every: int = DEFAULT_ROM_REFRESH_EVERY,
        rom_snapshot_source: str = DEFAULT_ROM_SNAPSHOT_SOURCE,
        inexact_mode: str = DEFAULT_INEXACT_MODE,
        inexact_forcing: str = DEFAULT_INEXACT_FORCING,
        inexact_eta0: float = DEFAULT_INEXACT_ETA0,
        inexact_eta_min: float = DEFAULT_INEXACT_ETA_MIN,
        inexact_eta_max: float = DEFAULT_INEXACT_ETA_MAX,
        lowrank_mode: str = DEFAULT_LOWRANK_MODE,
        lowrank_rank: int = DEFAULT_LOWRANK_RANK,
        lowrank_method: str = DEFAULT_LOWRANK_METHOD,
        lowrank_energy: float = DEFAULT_LOWRANK_ENERGY,
        absolute_startup_cache: bool = True,
        cholmod_max_n: int = DEFAULT_CHOLMOD_MAX_N,
        cholmod_max_memory_gib: float = DEFAULT_CHOLMOD_MAX_MEMORY_GIB,
        jacobian_block_tune: str = DEFAULT_JACOBIAN_BLOCK_TUNE,
        jacobian_block_size: int = DEFAULT_JACOBIAN_BLOCK_SIZE,
        jacobian_block_candidates: tuple[int, ...] | list[int] = DEFAULT_JACOBIAN_BLOCK_CANDIDATES,
        cache_scope: CacheScope = "both",
        cache_dir: str = ".pyeidors_cache/v2",
        cache_lifecycle: str | None = None,
        cache_policy: Optional[CachePolicy] = None,
        **kwargs,
    ) -> None:
        _ = kwargs
        self.n_elec = n_elec
        self.pattern_config = pattern_config or PatternConfig(
            n_elec=n_elec,
            stim_pattern="{ad}",
            meas_pattern="{ad}",
            drive_mode="line_current_density",
            drive_value=1.0,
            geometry_scale_to_m=1.0,
        )
        self.mesh_config = mesh_config or MeshConfig(radius=1.0, refinement=8)
        self.contact_impedance = (
            np.ones(n_elec, dtype=float) * 0.01
            if contact_impedance is None
            else np.asarray(contact_impedance, dtype=float)
        )

        self.base_conductivity = float(base_conductivity)
        self.difference_mode = normalize_difference_mode(
            difference_mode,
            default=DEFAULT_DIFFERENCE_MODE,
        )
        self.difference_orientation = normalize_difference_orientation(
            difference_orientation,
            default=DEFAULT_DIFFERENCE_ORIENTATION,
        )
        self.regularization_type = regularization_type.lower()
        self.regularization_alpha = float(regularization_alpha)
        self.hyperparameter = (
            None if hyperparameter is None else float(max(0.0, hyperparameter))
        )
        self.jacobian_background_conductivity = (
            self.base_conductivity
            if jacobian_background_conductivity is None
            else float(jacobian_background_conductivity)
        )
        self.noser_exponent = float(noser_exponent)
        self.noser_floor = float(noser_floor)
        self._difference_step_size_mode_explicit = difference_step_size_mode is not None
        self.difference_step_size_mode = _normalize_difference_step_size_mode(difference_step_size_mode)
        self.difference_step_size_value = (
            None
            if difference_step_size_value is None
            else float(difference_step_size_value)
        )
        self.difference_step_size_bounds = _normalize_bounds(difference_step_size_bounds)
        self.difference_step_size_fmin_options = dict(difference_step_size_fmin_options or {})
        self.difference_preset = _normalize_difference_preset(difference_preset)
        self.absolute_preset = _normalize_absolute_preset(absolute_preset)
        self._best_homog_mode_explicit = best_homog_mode is not None
        self.best_homog_mode = _normalize_best_homog_mode(best_homog_mode)
        self.linear_backend = str(linear_backend).strip().lower()
        self.forward_backend = normalize_forward_backend(
            forward_backend,
            default=DEFAULT_FORWARD_BACKEND,
        )
        self.mesh_family = normalize_mesh_family(
            mesh_family,
            default=DEFAULT_MESH_FAMILY,
        )
        self.petsc_device = normalize_petsc_device(petsc_device, default=DEFAULT_PETSC_DEVICE)
        self.device = normalize_runtime_device(device, default="auto")
        self.linear_backend_config = dict(linear_backend_config or {})
        self.linear_backend_config["petsc_device"] = self.petsc_device
        self.performance_mode = str(performance_mode).strip().lower()
        self.solver_mode = str(solver_mode).strip().lower()
        self.linear_solver = str(linear_solver).strip().lower()
        self.jacobian_update_every = int(max(1, jacobian_update_every))
        self.jacobian_reuse_tol = float(max(0.0, jacobian_reuse_tol))
        self.line_search_mode = str(line_search_mode).strip().lower()
        self.preconditioner = str(preconditioner).strip().lower()
        self.fast_linear_path = str(fast_linear_path).strip().lower()
        self.rom_mode = str(rom_mode).strip().lower()
        self.rom_rank_global = int(max(1, rom_rank_global))
        self.rom_rank_adaptive = int(max(0, rom_rank_adaptive))
        self.rom_refresh_every = int(max(1, rom_refresh_every))
        self.rom_snapshot_source = str(rom_snapshot_source).strip().lower()
        self.inexact_mode = str(inexact_mode).strip().lower()
        self.inexact_forcing = str(inexact_forcing).strip().lower()
        self.inexact_eta0 = float(inexact_eta0)
        self.inexact_eta_min = float(inexact_eta_min)
        self.inexact_eta_max = float(inexact_eta_max)
        self.lowrank_mode = str(lowrank_mode).strip().lower()
        self.lowrank_rank = int(max(1, lowrank_rank))
        self.lowrank_method = str(lowrank_method).strip().lower()
        self.lowrank_energy = float(lowrank_energy)
        self.absolute_startup_cache = bool(absolute_startup_cache)
        self.cholmod_max_n = int(max(1, cholmod_max_n))
        self.cholmod_max_memory_gib = float(max(0.25, cholmod_max_memory_gib))
        self.jacobian_block_tune = str(jacobian_block_tune).strip().lower()
        if self.jacobian_block_tune not in {"auto", "off"}:
            raise ValueError(
                f"Unsupported jacobian_block_tune={jacobian_block_tune!r}. "
                "Expected one of: 'auto', 'off'."
            )
        self.jacobian_block_size = int(max(0, jacobian_block_size))
        self.jacobian_block_candidates = tuple(
            sorted({int(v) for v in jacobian_block_candidates if int(v) > 0})
        ) or (64, 128, 256, 512)
        if self.performance_mode not in {"safe", "aggressive"}:
            raise ValueError(
                f"Unsupported performance_mode={performance_mode!r}. "
                "Expected one of: 'safe', 'aggressive'."
            )
        self.cache_scope: CacheScope = cache_scope
        requested_cache_lifecycle = (
            normalize_cache_lifecycle(cache_lifecycle, default=DEFAULT_CACHE_LIFECYCLE)
            if cache_lifecycle is not None
            else None
        )
        resolved_cache_policy = cache_policy
        if resolved_cache_policy is None:
            resolved_cache_policy = CachePolicy(
                disk_lifecycle=requested_cache_lifecycle or DEFAULT_CACHE_LIFECYCLE
            )
        elif requested_cache_lifecycle is not None:
            resolved_cache_policy = replace(
                resolved_cache_policy,
                disk_lifecycle=requested_cache_lifecycle,
            )
        self.cache_policy = resolved_cache_policy
        self.cache_lifecycle = normalize_cache_lifecycle(
            getattr(self.cache_policy, "disk_lifecycle", DEFAULT_CACHE_LIFECYCLE),
            default=DEFAULT_CACHE_LIFECYCLE,
        )
        self.cache_manager = CacheManager(
            scope=cache_scope,
            cache_dir=cache_dir,
            policy=self.cache_policy,
        )
        self._pattern_config_diagnostics: dict[str, str] = {
            "drive_mode_requested": str(self.pattern_config.drive_mode).strip().lower(),
            "drive_mode_effective": str(self.pattern_config.drive_mode).strip().lower(),
        }

        self.mesh: Optional[EITMesh] = None
        self.fwd_model: Optional[EITForwardModel] = None
        self.reconstructor: Optional[GaussNewtonReconstructor] = None
        self._is_initialized = False
        self._last_reconstructor_controls: dict[str, Any] = {}
        self._active_inverse_preset_name: str | None = None

    def setup(
        self,
        mesh: Optional[EITMesh] = None,
        *,
        mesh_source: Optional[str] = None,
        mesh_dir: str = "eit_meshes",
        mesh_name: Optional[str] = None,
        radius: Optional[float] = None,
        mesh_size: Optional[float] = None,
        dimension: Optional[int] = None,
        gdim: Optional[int] = None,
        height: Optional[float] = None,
        electrode_height_ratio: Optional[float] = None,
        electrode_level_fractions: Optional[tuple[float, ...] | list[float]] = None,
        z_center: Optional[float] = None,
        mesh_family: Optional[str] = None,
        geometry_version: Optional[str] = None,
    ) -> None:
        """Set up the system with an explicit mesh source."""
        if mesh is not None:
            self.setup_with_mesh(mesh)
            return
        if mesh_source == "cache":
            resolved_gdim = int(gdim if gdim is not None else (dimension if dimension is not None else 2))
            self.setup_from_cache(mesh_dir=mesh_dir, mesh_name=mesh_name, gdim=resolved_gdim)
            return
        if mesh_source == "generated":
            resolved_dim = int(dimension if dimension is not None else (gdim if gdim is not None else 2))
            self.setup_generated_mesh(
                radius=radius,
                mesh_size=mesh_size,
                dimension=resolved_dim,
                height=height,
                electrode_height_ratio=electrode_height_ratio,
                electrode_level_fractions=electrode_level_fractions,
                z_center=z_center,
                mesh_family=mesh_family,
                geometry_version=geometry_version,
            )
            return
        raise ValueError(
            "EITSystem.setup requires an explicit mesh source. "
            "Use setup(mesh=...), setup(mesh_source='cache', ...), "
            "or setup(mesh_source='generated', ...)."
        )

    def setup_with_mesh(self, mesh: EITMesh) -> None:
        if not isinstance(mesh, EITMesh):
            raise TypeError("EITSystem.setup_with_mesh expects an EITMesh instance")
        self.mesh = mesh
        normalized_pattern, diagnostics = normalize_pattern_config_for_mesh(
            self.pattern_config,
            mesh_tdim=int(mesh.topology.dim),
        )
        self.pattern_config = normalized_pattern
        self._pattern_config_diagnostics = diagnostics
        self._initialize_components()

    def setup_from_cache(
        self,
        mesh_dir: str = "eit_meshes",
        mesh_name: Optional[str] = None,
        gdim: int = 2,
    ) -> None:
        loader = MeshLoader(mesh_dir=mesh_dir, gdim=gdim)
        selected = loader.load_mesh(mesh_name) if mesh_name else loader.get_default_mesh()
        logger.info("Loaded cached mesh from %s (mesh_name=%s, gdim=%d)", mesh_dir, mesh_name, gdim)
        self.setup_with_mesh(selected)

    def setup_generated_mesh(
        self,
        *,
        radius: Optional[float] = None,
        mesh_size: Optional[float] = None,
        dimension: int = 2,
        height: Optional[float] = None,
        electrode_height_ratio: Optional[float] = None,
        electrode_level_fractions: Optional[tuple[float, ...] | list[float]] = None,
        z_center: Optional[float] = None,
        mesh_family: Optional[str] = None,
        geometry_version: Optional[str] = None,
    ) -> None:
        if int(dimension) not in {2, 3}:
            raise ValueError(f"dimension must be 2 or 3, got {dimension!r}")

        resolved_radius = self.mesh_config.radius if radius is None else float(radius)
        resolved_mesh_size = self.mesh_config.mesh_size if mesh_size is None else float(mesh_size)
        if int(dimension) == 2:
            generated = create_simple_eit_mesh(
                n_elec=self.n_elec,
                radius=resolved_radius,
                mesh_size=resolved_mesh_size,
            )
        else:
            resolved_height = self.mesh_config.height if height is None else float(height)
            resolved_ratio = (
                self.mesh_config.electrode_height_ratio
                if electrode_height_ratio is None
                else float(electrode_height_ratio)
            )
            resolved_level_fractions = (
                self.mesh_config.electrode_level_fractions
                if electrode_level_fractions is None
                else tuple(float(v) for v in electrode_level_fractions)
            )
            resolved_z = self.mesh_config.z_center if z_center is None else float(z_center)
            resolved_mesh_family = normalize_mesh_family(
                self.mesh_config.mesh_family if mesh_family is None else mesh_family,
                default=DEFAULT_MESH_FAMILY,
            )
            resolved_geometry_version = (
                self.mesh_config.geometry_version
                if geometry_version is None
                else str(geometry_version).strip().lower()
            ) or DEFAULT_3D_GEOMETRY_VERSION
            resolved_refinement = max(
                2,
                int(round(resolved_radius / max(resolved_mesh_size, 1e-6) / 2)),
            )
            generated = create_cylinder_3d_eit_mesh(
                n_elec=self.n_elec,
                radius=resolved_radius,
                height=resolved_height,
                refinement=resolved_refinement,
                electrode_height_ratio=resolved_ratio,
                electrode_level_fractions=resolved_level_fractions,
                z_center=resolved_z,
                mesh_family=resolved_mesh_family,
                geometry_version=resolved_geometry_version,
            )
        logger.info(
            "Generated mesh on demand (n_elec=%d, dim=%d, radius=%s, mesh_size=%s)",
            self.n_elec,
            int(dimension),
            resolved_radius,
            resolved_mesh_size,
        )
        self.setup_with_mesh(generated)

    def _initialize_components(self) -> None:
        if self.mesh is None:
            raise RuntimeError("Cannot initialize EITSystem without mesh")
        self.fwd_model = EITForwardModel(
            n_elec=self.n_elec,
            pattern_config=self.pattern_config,
            z=self.contact_impedance,
            mesh=self.mesh,
            linear_backend=self.linear_backend,
            backend_config=self.linear_backend_config,
            forward_backend=self.forward_backend,
            cache_manager=self.cache_manager,
            performance_mode=self.performance_mode,
        )
        self.fwd_model._set_backend_diagnostic(**self._pattern_config_diagnostics)
        jacobian_calculator = DirectJacobianCalculator(
            self.fwd_model,
            block_tune_mode=self.jacobian_block_tune,
            block_size=self.jacobian_block_size,
            block_candidates=self.jacobian_block_candidates,
            runtime_device=self.device,
        )
        regularization = self._build_regularization(jacobian_calculator)
        self.reconstructor = GaussNewtonReconstructor(
            fwd_model=self.fwd_model,
            jacobian_calculator=jacobian_calculator,
            regularization=regularization,
            hyperparameter=self.hyperparameter,
            difference_mode=self.difference_mode,
            difference_orientation=self.difference_orientation,
            jacobian_background_conductivity=self.jacobian_background_conductivity,
            difference_step_size_mode=self.difference_step_size_mode,
            difference_step_size_value=self.difference_step_size_value,
            difference_step_size_bounds=self.difference_step_size_bounds,
            difference_step_size_fmin_options=self.difference_step_size_fmin_options,
            difference_preset=self.difference_preset,
            absolute_preset=self.absolute_preset,
            best_homog_mode=self.best_homog_mode,
            cache_manager=self.cache_manager,
            performance_mode=self.performance_mode,
            device=self.device,
            solver_mode=self.solver_mode,
            linear_solver=self.linear_solver,
            jacobian_update_every=self.jacobian_update_every,
            jacobian_reuse_tol=self.jacobian_reuse_tol,
            line_search_mode=self.line_search_mode,
            preconditioner=self.preconditioner,
            fast_linear_path=self.fast_linear_path,
            rom_mode=self.rom_mode,
            rom_rank_global=self.rom_rank_global,
            rom_rank_adaptive=self.rom_rank_adaptive,
            rom_refresh_every=self.rom_refresh_every,
            rom_snapshot_source=self.rom_snapshot_source,
            inexact_mode=self.inexact_mode,
            inexact_forcing=self.inexact_forcing,
            inexact_eta0=self.inexact_eta0,
            inexact_eta_min=self.inexact_eta_min,
            inexact_eta_max=self.inexact_eta_max,
            lowrank_mode=self.lowrank_mode,
            lowrank_rank=self.lowrank_rank,
            lowrank_method=self.lowrank_method,
            lowrank_energy=self.lowrank_energy,
            absolute_startup_cache=self.absolute_startup_cache,
            cholmod_max_n=self.cholmod_max_n,
            cholmod_max_memory_gib=self.cholmod_max_memory_gib,
        )
        self._last_reconstructor_controls = self._capture_reconstructor_controls()
        self._is_initialized = True

    def _build_regularization(self, jacobian_calculator, *, regularization_type: Optional[str] = None):
        resolved_type = (regularization_type or self.regularization_type).lower()
        if resolved_type == "noser":
            return NOSERRegularization(
                self.fwd_model,
                jacobian_calculator,
                base_conductivity=self.jacobian_background_conductivity,
                alpha=self.regularization_alpha,
                exponent=self.noser_exponent,
                floor=self.noser_floor,
            )
        if resolved_type == "tikhonov":
            return TikhonovRegularization(self.fwd_model, alpha=self.regularization_alpha)
        if resolved_type == "smoothness":
            return SmoothnessRegularization(self.fwd_model, alpha=self.regularization_alpha)
        if resolved_type == "tv":
            return TotalVariationRegularization(
                self.fwd_model,
                alpha=self.regularization_alpha,
                epsilon=1e-6,
                reference_conductivity=self.jacobian_background_conductivity,
            )
        raise ValueError(
            f"Unsupported regularization_type={resolved_type!r}. "
            "Expected one of: 'noser', 'tikhonov', 'smoothness', 'tv'."
        )

    def _capture_reconstructor_controls(self) -> dict[str, Any]:
        if self.reconstructor is None:
            return {}
        return {
            "hyperparameter": self.reconstructor.hyperparameter,
            "max_iterations": self.reconstructor.max_iterations,
            "jacobian_update_every": self.reconstructor.jacobian_update_every,
            "jacobian_reuse_tol": self.reconstructor.jacobian_reuse_tol,
            "line_search_mode": self.reconstructor.line_search_mode,
            "difference_step_size_mode": self.reconstructor.difference_step_size_mode,
            "difference_step_size_value": self.reconstructor.difference_step_size_value,
            "difference_step_size_bounds": self.reconstructor.difference_step_size_bounds,
            "difference_step_size_fmin_options": dict(
                self.reconstructor.difference_step_size_fmin_options
            ),
            "best_homog_mode": self.reconstructor.best_homog_mode,
            "min_step": self.reconstructor.min_step,
            "max_step": self.reconstructor.max_step,
        }

    def _default_hyperparameter(self, preset_name: str) -> float:
        if self.hyperparameter is not None:
            return float(self.hyperparameter)
        if preset_name == "eidors_demo3d_tv":
            return 1e-2
        if preset_name in {"sphere_multistep_noser", "eidors_abs_gn"}:
            return float(np.sqrt(1e-3))
        return 1e-1

    def _preset_config(self, inverse_mode: str) -> dict[str, Any]:
        resolved_mode = str(inverse_mode).strip().lower()
        if resolved_mode == "difference":
            preset_name = self.difference_preset
            config: dict[str, Any] = {
                "preset_name": preset_name,
                "difference_step_size_mode": (
                    self.difference_step_size_mode
                    if self._difference_step_size_mode_explicit
                    else "optimize"
                ),
                "difference_step_size_value": self.difference_step_size_value,
                "difference_step_size_bounds": self.difference_step_size_bounds,
                "difference_step_size_fmin_options": dict(self.difference_step_size_fmin_options),
                "best_homog_mode": self.best_homog_mode,
                "min_step": 0.0,
                "max_step": 1.0,
            }
            if preset_name == "eidors_one_step_noser":
                config.update(
                    {
                        "regularization_type": "noser",
                        "hyperparameter": self._default_hyperparameter(preset_name),
                        "max_iterations": 1,
                        "jacobian_update_every": 1,
                        "jacobian_reuse_tol": 0.0,
                        "line_search_mode": "full",
                    }
                )
            elif preset_name == "eidors_demo3d_tv":
                config.update(
                    {
                        "regularization_type": "tv",
                        "hyperparameter": self._default_hyperparameter(preset_name),
                        "max_iterations": 1,
                        "jacobian_update_every": 1,
                        "jacobian_reuse_tol": 0.0,
                        "line_search_mode": "full",
                    }
                )
            else:
                config.update(
                    {
                        "regularization_type": "noser",
                        "hyperparameter": self._default_hyperparameter(preset_name),
                        "max_iterations": 3,
                        "jacobian_update_every": 1,
                        "jacobian_reuse_tol": 0.0,
                        "line_search_mode": "full",
                        "difference_step_size_mode": "off",
                        "max_step": 1.0,
                    }
                )
            return config

        preset_name = self.absolute_preset
        return {
            "preset_name": preset_name,
            "regularization_type": "noser",
            "hyperparameter": self._default_hyperparameter(preset_name),
            "max_iterations": 3,
            "jacobian_update_every": 1,
            "jacobian_reuse_tol": 0.0,
            "line_search_mode": "full",
            "difference_step_size_mode": "off",
            "difference_step_size_value": self.difference_step_size_value,
            "difference_step_size_bounds": self.difference_step_size_bounds,
            "difference_step_size_fmin_options": dict(self.difference_step_size_fmin_options),
            "best_homog_mode": (
                self.best_homog_mode if self._best_homog_mode_explicit else "optimize"
            ),
            "min_step": 0.0,
            "max_step": 1.0,
        }

    def _apply_inverse_preset(self, inverse_mode: str) -> None:
        if self.reconstructor is None:
            raise RuntimeError("Reconstructor is not initialized.")

        config = self._preset_config(inverse_mode)
        regularization_type = str(config.pop("regularization_type", self.regularization_type)).lower()
        if regularization_type != self.regularization_type or not isinstance(
            self.reconstructor.regularization,
            {
                "noser": NOSERRegularization,
                "tikhonov": TikhonovRegularization,
                "smoothness": SmoothnessRegularization,
                "tv": TotalVariationRegularization,
            }[regularization_type],
        ):
            self.regularization_type = regularization_type
            self.reconstructor.set_regularization(
                self._build_regularization(
                    self.reconstructor.jacobian_calculator,
                    regularization_type=regularization_type,
                )
            )

        for field, value in config.items():
            if field == "preset_name":
                continue
            previous = self._last_reconstructor_controls.get(field)
            current = getattr(self.reconstructor, field)
            if _matches_previous(current, previous):
                setattr(self.reconstructor, field, value)
                self._last_reconstructor_controls[field] = value

        self.reconstructor.jacobian_background_conductivity = self.jacobian_background_conductivity
        self.reconstructor.difference_preset = self.difference_preset
        self.reconstructor.absolute_preset = self.absolute_preset
        self.reconstructor.active_preset_name = str(config["preset_name"])
        self._active_inverse_preset_name = str(config["preset_name"])

    def _require_initialized(self) -> None:
        if not self._is_initialized or self.fwd_model is None or self.reconstructor is None:
            raise RuntimeError("System not initialized. Please call setup(...) first.")

    def forward_solve(self, conductivity: Union[np.ndarray, EITImage, Any]) -> EITData:
        self._require_initialized()
        image = conductivity_to_image(self.fwd_model, conductivity)
        data, _ = self.fwd_model.fwd_solve(image)
        return data

    def inverse_solve(
        self,
        data: EITData,
        reference_data: Optional[EITData] = None,
        initial_guess: Optional[np.ndarray] = None,
    ) -> SolverOutput:
        self._require_initialized()
        inverse_mode = "difference" if reference_data is not None else "absolute"
        self._apply_inverse_preset(inverse_mode)
        diff_data = difference_measurement(
            data,
            reference_data,
            mode=self.difference_mode,
            orientation=self.difference_orientation,
        )
        try:
            self.reconstructor.ensure_regularization_ready()
        except Exception as exc:
            raise RuntimeError(f"regularization warmup failed: {exc}") from exc
        return self.reconstructor.reconstruct(diff_data, initial_guess)

    def run_unit_precheck(
        self,
        expected_domain_size_m: float | None = None,
        strict: bool = True,
    ) -> UnitCheckReport:
        """Run unit consistency checks before experiments.

        Args:
            expected_domain_size_m: Optional expected physical size (max bbox extent).
            strict: Raise ``ValueError`` if any blocking check fails.
        """
        self._require_initialized()
        report = run_unit_consistency_checks(
            self.fwd_model,
            expected_domain_size_m=expected_domain_size_m,
        )
        if strict and report.has_errors:
            details = " | ".join(report.summary_lines())
            raise ValueError(f"Unit precheck failed: {details}")
        return report

    def get_cache_stats(self) -> dict[str, Any]:
        """Return runtime cache hit/miss and footprint statistics."""
        stats = dict(self.cache_manager.stats())
        stats["process_forward_setup_cache"] = process_forward_setup_cache_stats()
        return stats

    def clear_cache(self, scope: CacheScope = "both") -> None:
        """Clear cache entries for selected scope."""

        self.cache_manager.clear(scope=scope)
        if scope in {"process", "both"}:
            clear_process_mesh_cache()
            clear_process_forward_setup_cache()
