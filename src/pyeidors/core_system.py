"""PyEIDORS Core System Module."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

import numpy as np
from dolfinx import fem

from .data.structures import EITData, EITImage, EITMesh, MeshConfig, PatternConfig
from .forward.eit_forward_model import EITForwardModel
from .geometry.mesh_loader import MeshLoader
from .geometry.simple_mesh_generator import create_simple_eit_mesh
from .inverse import (
    ReconstructionResult,
    perform_absolute_reconstruction,
    perform_difference_reconstruction,
)
from .inverse.jacobian.direct_jacobian import DirectJacobianCalculator
from .inverse.regularization.smoothness import (
    NOSERRegularization,
    SmoothnessRegularization,
    TikhonovRegularization,
)
from .inverse.solvers.gauss_newton import GaussNewtonReconstructor

logger = logging.getLogger(__name__)


class EITSystem:
    """PyEIDORS Core System Class.

    Integrates all major EIT system functionality:
    - Mesh generation and management
    - Forward problem solving
    - Inverse problem reconstruction
    - Data processing and visualization
    """

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
        **kwargs,
    ):
        """Initialize the EIT system.

        Args:
            n_elec: Number of electrodes.
            pattern_config: Stimulation and measurement pattern configuration.
            mesh_config: Mesh configuration.
            contact_impedance: Contact impedance array.
            base_conductivity: Baseline conductivity value.
            regularization_type: Regularization type ("noser", "tikhonov", "smoothness").
            regularization_alpha: Regularization parameter.
            noser_exponent: NOSER regularization exponent (EIDORS default: 0.5).
            noser_floor: Minimum value for NOSER diagonal elements.
            **kwargs: Additional configuration parameters.
        """
        _ = kwargs
        self.n_elec = n_elec

        # Set default configuration
        if pattern_config is None:
            pattern_config = PatternConfig(
                n_elec=n_elec,
                stim_pattern="{ad}",
                meas_pattern="{ad}",
                amplitude=1.0,
            )
        self.pattern_config = pattern_config

        if mesh_config is None:
            mesh_config = MeshConfig(radius=1.0, refinement=8)
        self.mesh_config = mesh_config

        # Set contact impedance
        if contact_impedance is None:
            contact_impedance = np.ones(n_elec) * 0.01
        self.contact_impedance = contact_impedance

        self.base_conductivity = base_conductivity
        self.regularization_type = regularization_type.lower()
        self.regularization_alpha = regularization_alpha
        self.noser_exponent = noser_exponent
        self.noser_floor = noser_floor

        # Initialize components
        self.mesh = None
        self.fwd_model = None
        self.reconstructor = None
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
        """Set up the EIT system with an explicit mesh source.

        Allowed paths:
        - `setup(mesh=eit_mesh)` for a pre-built :class:`EITMesh`.
        - `setup(mesh_source="cache", ...)` to load from `.msh` cache.
        - `setup(mesh_source="generated", ...)` to generate a mesh.
        """
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
            "EITSystem.setup now requires an explicit mesh source. "
            "Use one of: setup(mesh=...), setup(mesh_source='cache', ...), "
            "or setup(mesh_source='generated', ...)."
        )

    def setup_with_mesh(self, mesh: EITMesh) -> None:
        """Initialise with a provided :class:`EITMesh`."""
        if not isinstance(mesh, EITMesh):
            raise TypeError("EITSystem.setup_with_mesh expects an EITMesh instance")
        self.mesh = mesh
        self._initialize_components()

    def setup_from_cache(self, mesh_dir: str = "eit_meshes", mesh_name: Optional[str] = None) -> None:
        """Initialise from a cached `.msh` mesh."""
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
        """Initialise from a newly generated mesh."""
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
        )
        jacobian_calculator = DirectJacobianCalculator(self.fwd_model)
        regularization = self._build_regularization(jacobian_calculator)
        self.reconstructor = GaussNewtonReconstructor(
            fwd_model=self.fwd_model,
            jacobian_calculator=jacobian_calculator,
            regularization=regularization,
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
        if not self._is_initialized:
            raise RuntimeError("System not initialized. Please call setup(...) first.")

    def forward_solve(self, conductivity: Union[np.ndarray, fem.Function, EITImage]) -> EITData:
        """Perform forward solve.

        Args:
            conductivity: Conductivity distribution.

        Returns:
            EIT measurement data.
        """
        self._require_initialized()

        # Handle different conductivity input types
        if isinstance(conductivity, np.ndarray):
            img = EITImage(elem_data=conductivity, fwd_model=self.fwd_model)
        elif isinstance(conductivity, fem.Function):
            img = EITImage(elem_data=conductivity.x.array.copy(), fwd_model=self.fwd_model)
        elif isinstance(conductivity, EITImage):
            img = conductivity
        else:
            raise ValueError("Unsupported conductivity input type")

        # Execute forward solve
        data, _ = self.fwd_model.fwd_solve(img)
        return data

    def inverse_solve(
        self,
        data: EITData,
        reference_data: Optional[EITData] = None,
        initial_guess: Optional[np.ndarray] = None,
    ):
        """Perform inverse reconstruction.

        Args:
            data: Measurement data.
            reference_data: Reference data (optional).
            initial_guess: Initial guess for reconstruction (optional).

        Returns:
            Reconstructed conductivity distribution.
        """
        self._require_initialized()

        # Handle difference measurements
        if reference_data is not None:
            diff_data = EITData(
                meas=data.meas - reference_data.meas,
                stim_pattern=data.stim_pattern,
                n_elec=data.n_elec,
                n_stim=data.n_stim,
                n_meas=data.n_meas,
                type="difference",
            )
        else:
            diff_data = data

        # Execute reconstruction
        result = self.reconstructor.reconstruct(diff_data, initial_guess)
        
        return result

    def absolute_reconstruct(
        self,
        measurement_data: EITData,
        baseline_image: Optional[EITImage] = None,
        initial_image: Optional[EITImage] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReconstructionResult:
        """Convenience method for absolute imaging reconstruction."""

        if baseline_image is None and self._is_initialized:
            baseline_image = self.create_homogeneous_image()

        return perform_absolute_reconstruction(
            eit_system=self,
            measurement_data=measurement_data,
            baseline_image=baseline_image,
            initial_image=initial_image,
            metadata=metadata,
        )

    def difference_reconstruct(
        self,
        measurement_data: EITData,
        reference_data: EITData,
        initial_image: Optional[EITImage] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReconstructionResult:
        """Convenience method for difference imaging reconstruction."""

        return perform_difference_reconstruction(
            eit_system=self,
            measurement_data=measurement_data,
            reference_data=reference_data,
            initial_image=initial_image,
            metadata=metadata,
        )

    def create_homogeneous_image(self, conductivity: Optional[float] = None) -> EITImage:
        """Create a homogeneous conductivity image.

        Args:
            conductivity: Conductivity value.

        Returns:
            Homogeneous conductivity image.
        """
        self._require_initialized()

        if conductivity is None:
            conductivity = self.base_conductivity

        n_elements = int(fem.Function(self.fwd_model.V_sigma).x.array.size)
        elem_data = np.ones(n_elements) * conductivity
        return EITImage(elem_data=elem_data, fwd_model=self.fwd_model)

    def add_phantom(
        self,
        base_conductivity: float = 1.0,
        phantom_conductivity: float = 2.0,
        phantom_center: tuple = (0.3, 0.3),
        phantom_radius: float = 0.2,
    ) -> EITImage:
        """Add a circular phantom.

        Args:
            base_conductivity: Background conductivity.
            phantom_conductivity: Phantom conductivity.
            phantom_center: Phantom center coordinates.
            phantom_radius: Phantom radius.

        Returns:
            Conductivity image with phantom.
        """
        self._require_initialized()

        # Get mesh centroid coordinates
        V_sigma = self.fwd_model.V_sigma
        dof_coordinates = V_sigma.tabulate_dof_coordinates()

        # Create base conductivity distribution
        elem_data = np.ones(len(dof_coordinates)) * base_conductivity

        # Add circular phantom
        for i, coord in enumerate(dof_coordinates):
            x, y = coord[0], coord[1]
            distance = np.sqrt((x - phantom_center[0]) ** 2 + (y - phantom_center[1]) ** 2)
            if distance <= phantom_radius:
                elem_data[i] = phantom_conductivity

        return EITImage(elem_data=elem_data, fwd_model=self.fwd_model)

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information.

        Returns:
            Dictionary containing system configuration information.
        """
        info = {
            "n_elec": self.n_elec,
            "pattern_config": self.pattern_config,
            "mesh_config": self.mesh_config,
            "initialized": self._is_initialized,
        }

        if self._is_initialized:
            info.update(
                {
                    "n_elements": int(fem.Function(self.fwd_model.V_sigma).x.array.size),
                    "n_nodes": int(fem.Function(self.fwd_model.V).x.array.size),
                    "n_measurements": self.fwd_model.pattern_manager.n_meas_total,
                    "n_stimulation_patterns": self.fwd_model.pattern_manager.n_stim,
                }
            )

        return info
