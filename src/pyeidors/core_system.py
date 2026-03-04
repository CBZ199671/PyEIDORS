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
from .geometry.simple_mesh_generator import create_simple_eit_mesh
from .inverse.contracts import SolverOutput
from .inverse.jacobian.direct_jacobian import DirectJacobianCalculator
from .inverse.regularization.smoothness import (
    NOSERRegularization,
    SmoothnessRegularization,
    TikhonovRegularization,
)
from .inverse.solvers.gauss_newton import GaussNewtonReconstructor
from .physics import UnitCheckReport, run_unit_consistency_checks

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
        performance_mode: str = "aggressive",
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
        self.linear_backend_config = dict(linear_backend_config or {})
        self.performance_mode = str(performance_mode).strip().lower()
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
    ) -> None:
        """Set up the system with an explicit mesh source."""
        if mesh is not None:
            self.setup_with_mesh(mesh)
            return
        if mesh_source == "cache":
            self.setup_from_cache(mesh_dir=mesh_dir, mesh_name=mesh_name)
            return
        if mesh_source == "generated":
            self.setup_generated_mesh(radius=radius, mesh_size=mesh_size)
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

    def setup_from_cache(self, mesh_dir: str = "eit_meshes", mesh_name: Optional[str] = None) -> None:
        loader = MeshLoader(mesh_dir=mesh_dir)
        selected = loader.load_mesh(mesh_name) if mesh_name else loader.get_default_mesh()
        logger.info("Loaded cached mesh from %s (mesh_name=%s)", mesh_dir, mesh_name)
        self.setup_with_mesh(selected)

    def setup_generated_mesh(
        self,
        *,
        radius: Optional[float] = None,
        mesh_size: Optional[float] = None,
    ) -> None:
        resolved_radius = self.mesh_config.radius if radius is None else float(radius)
        resolved_mesh_size = self.mesh_config.mesh_size if mesh_size is None else float(mesh_size)
        generated = create_simple_eit_mesh(
            n_elec=self.n_elec,
            radius=resolved_radius,
            mesh_size=resolved_mesh_size,
        )
        logger.info(
            "Generated mesh on demand (n_elec=%d, radius=%s, mesh_size=%s)",
            self.n_elec,
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
        jacobian_calculator = DirectJacobianCalculator(self.fwd_model)
        regularization = self._build_regularization(jacobian_calculator)
        self.reconstructor = GaussNewtonReconstructor(
            fwd_model=self.fwd_model,
            jacobian_calculator=jacobian_calculator,
            regularization=regularization,
            cache_manager=self.cache_manager,
            performance_mode=self.performance_mode,
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
