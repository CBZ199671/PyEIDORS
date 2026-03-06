"""PyEIDORS core system orchestration."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

import numpy as np

from .cache import CacheManager, CachePolicy, CacheScope
from .core_system_facade import CoreSystemFacadeMixin
from .core_system_helpers import (
    conductivity_to_image,
    difference_measurement,
)
from .data.structures import EITData, EITImage, EITMesh, MeshConfig, PatternConfig
from .forward.eit_forward_model import EITForwardModel
from .geometry.mesh_loader import MeshLoader
from .geometry.mesh3d_generator import create_cylinder_3d_eit_mesh
from .geometry.simple_mesh_generator import create_simple_eit_mesh
from .inverse.contracts import SolverOutput
from .inverse.jacobian.direct_jacobian import DirectJacobianCalculator
from .inverse.regularization.smoothness import (
    NOSERRegularization,
    SmoothnessRegularization,
    TikhonovRegularization,
)
from .inverse.solvers.gauss_newton import GaussNewtonReconstructor
from .inverse.solvers.gauss_newton_device import normalize_runtime_device
from .physics import UnitCheckReport, run_unit_consistency_checks
from .perf.policy import (
    DEFAULT_CHOLMOD_MAX_MEMORY_GIB,
    DEFAULT_CHOLMOD_MAX_N,
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
    normalize_petsc_device,
)

logger = logging.getLogger(__name__)


class EITSystem(CoreSystemFacadeMixin):
    """Facade that wires mesh, forward model and reconstruction workflow."""

    def __init__(
        self,
        n_elec: int = 16,
        pattern_config: Optional[PatternConfig] = None,
        mesh_config: Optional[MeshConfig] = None,
        contact_impedance: Optional[np.ndarray] = None,
        base_conductivity: float = 1.0,
        regularization_type: str = "noser",
        regularization_alpha: float = 1.0,
        noser_exponent: float = 0.5,
        noser_floor: float = 1e-12,
        linear_backend: str = "petsc",
        linear_backend_config: Optional[dict[str, Any]] = None,
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
        self.regularization_type = regularization_type.lower()
        self.regularization_alpha = float(regularization_alpha)
        self.noser_exponent = float(noser_exponent)
        self.noser_floor = float(noser_floor)
        self.linear_backend = str(linear_backend).strip().lower()
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
        self.cache_manager = CacheManager(
            scope=cache_scope,
            cache_dir=cache_dir,
            policy=cache_policy,
        )

        self.mesh: Optional[EITMesh] = None
        self.fwd_model: Optional[EITForwardModel] = None
        self.reconstructor: Optional[GaussNewtonReconstructor] = None
        self._is_initialized = False

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
        z_center: Optional[float] = None,
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
                z_center=z_center,
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
        z_center: Optional[float] = None,
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
            resolved_z = self.mesh_config.z_center if z_center is None else float(z_center)
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
                z_center=resolved_z,
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
            cache_manager=self.cache_manager,
            performance_mode=self.performance_mode,
        )
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
        self._is_initialized = True

    def _build_regularization(self, jacobian_calculator):
        if self.regularization_type == "noser":
            return NOSERRegularization(
                self.fwd_model,
                jacobian_calculator,
                base_conductivity=self.base_conductivity,
                alpha=self.regularization_alpha,
                exponent=self.noser_exponent,
                floor=self.noser_floor,
            )
        if self.regularization_type == "tikhonov":
            return TikhonovRegularization(self.fwd_model, alpha=self.regularization_alpha)
        if self.regularization_type == "smoothness":
            return SmoothnessRegularization(self.fwd_model, alpha=self.regularization_alpha)
        raise ValueError(
            f"Unsupported regularization_type={self.regularization_type!r}. "
            "Expected one of: 'noser', 'tikhonov', 'smoothness'."
        )

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
        diff_data = difference_measurement(data, reference_data)
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

        return self.cache_manager.stats()

    def clear_cache(self, scope: CacheScope = "both") -> None:
        """Clear cache entries for selected scope."""

        self.cache_manager.clear(scope=scope)
