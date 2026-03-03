"""PyEIDORS Core System Module.

This is the main interface for the EIT system, integrating forward modeling,
inverse problem solvers, and data processing functionality.
"""

import numpy as np
from typing import Optional, Union, Dict, Any
from fenics import Function

from .data.structures import EITData, EITImage, PatternConfig, MeshConfig
from .forward.eit_forward_model import EITForwardModel
from .inverse.solvers.gauss_newton import ModularGaussNewtonReconstructor
from .inverse.jacobian.direct_jacobian import DirectJacobianCalculator
from .inverse.regularization.smoothness import SmoothnessRegularization, NOSERRegularization, TikhonovRegularization
from .inverse import (
    perform_absolute_reconstruction,
    perform_difference_reconstruction,
    ReconstructionResult,
)
from .electrodes.patterns import StimMeasPatternManager
from .geometry.mesh_loader import MeshLoader
from .geometry.simple_mesh_generator import create_simple_eit_mesh


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
        self.n_elec = n_elec

        # Set default configuration
        if pattern_config is None:
            pattern_config = PatternConfig(
                n_elec=n_elec,
                stim_pattern='{ad}',
                meas_pattern='{ad}',
                amplitude=1.0
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

    def setup(self, mesh=None):
        """Set up the EIT system.

        Args:
            mesh: Optional external mesh. If not provided, will attempt to load or generate one.
        """
        # Set up mesh
        if mesh is not None:
            self.mesh = mesh
        else:
            # First try to load existing mesh
            try:
                mesh_loader = MeshLoader()
                self.mesh = mesh_loader.get_default_mesh()
                print(f"Loaded existing mesh: {self.mesh.get_info()}")
            except Exception as load_error:
                # If loading fails, try to generate new mesh
                print(f"Failed to load existing mesh: {load_error}")
                print("Generating new EIT mesh...")
                try:
                    self.mesh = create_simple_eit_mesh(
                        n_elec=self.n_elec,
                        radius=1.0,
                        mesh_size=0.1
                    )
                    print(f"New mesh generated successfully: {self.mesh.get_info()}")
                except Exception as gen_error:
                    raise RuntimeError(f"Unable to generate mesh: {gen_error}. Please check Gmsh installation or provide a mesh object.")

        # Initialize forward model
        self.fwd_model = EITForwardModel(
            n_elec=self.n_elec,
            pattern_config=self.pattern_config,
            z=self.contact_impedance,
            mesh=self.mesh
        )

        # Initialize reconstructor
        jacobian_calculator = DirectJacobianCalculator(self.fwd_model)
        if self.regularization_type == "noser":
            regularization = NOSERRegularization(
                self.fwd_model,
                jacobian_calculator,
                base_conductivity=self.base_conductivity,
                alpha=self.regularization_alpha,
                exponent=self.noser_exponent,
                floor=self.noser_floor,
            )
        elif self.regularization_type == "tikhonov":
            regularization = TikhonovRegularization(self.fwd_model, alpha=self.regularization_alpha)
        else:
            regularization = SmoothnessRegularization(self.fwd_model, alpha=self.regularization_alpha)
        
        self.reconstructor = ModularGaussNewtonReconstructor(
            fwd_model=self.fwd_model,
            jacobian_calculator=jacobian_calculator,
            regularization=regularization
        )
        
        self._is_initialized = True

    def forward_solve(self, conductivity: Union[np.ndarray, Function, EITImage]) -> EITData:
        """Perform forward solve.

        Args:
            conductivity: Conductivity distribution.

        Returns:
            EIT measurement data.
        """
        if not self._is_initialized:
            raise RuntimeError("System not initialized. Please call setup() first.")

        # Handle different conductivity input types
        if isinstance(conductivity, np.ndarray):
            img = EITImage(elem_data=conductivity, fwd_model=self.fwd_model)
        elif isinstance(conductivity, EITImage):
            img = conductivity
        else:
            raise ValueError("Unsupported conductivity input type")

        # Execute forward solve
        data, _ = self.fwd_model.fwd_solve(img)
        return data

    def inverse_solve(self, data: EITData,
                     reference_data: Optional[EITData] = None,
                     initial_guess: Optional[np.ndarray] = None) -> EITImage:
        """Perform inverse reconstruction.

        Args:
            data: Measurement data.
            reference_data: Reference data (optional).
            initial_guess: Initial guess for reconstruction (optional).

        Returns:
            Reconstructed conductivity distribution.
        """
        if not self._is_initialized:
            raise RuntimeError("System not initialized. Please call setup() first.")

        # Handle difference measurements
        if reference_data is not None:
            diff_data = EITData(
                meas=data.meas - reference_data.meas,
                stim_pattern=data.stim_pattern,
                n_elec=data.n_elec,
                n_stim=data.n_stim,
                n_meas=data.n_meas,
                type='difference'
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
        if not self._is_initialized:
            raise RuntimeError("System not initialized. Please call setup() first.")

        if conductivity is None:
            conductivity = self.base_conductivity

        n_elements = len(Function(self.fwd_model.V_sigma).vector()[:])
        elem_data = np.ones(n_elements) * conductivity
        
        return EITImage(elem_data=elem_data, fwd_model=self.fwd_model)

    def add_phantom(self, base_conductivity: float = 1.0,
                   phantom_conductivity: float = 2.0,
                   phantom_center: tuple = (0.3, 0.3),
                   phantom_radius: float = 0.2) -> EITImage:
        """Add a circular phantom.

        Args:
            base_conductivity: Background conductivity.
            phantom_conductivity: Phantom conductivity.
            phantom_center: Phantom center coordinates.
            phantom_radius: Phantom radius.

        Returns:
            Conductivity image with phantom.
        """
        if not self._is_initialized:
            raise RuntimeError("System not initialized. Please call setup() first.")

        # Get mesh centroid coordinates
        V_sigma = self.fwd_model.V_sigma
        dof_coordinates = V_sigma.tabulate_dof_coordinates()

        # Create base conductivity distribution
        elem_data = np.ones(len(dof_coordinates)) * base_conductivity

        # Add circular phantom
        for i, coord in enumerate(dof_coordinates):
            x, y = coord[0], coord[1]
            distance = np.sqrt((x - phantom_center[0])**2 + (y - phantom_center[1])**2)
            if distance <= phantom_radius:
                elem_data[i] = phantom_conductivity
        
        return EITImage(elem_data=elem_data, fwd_model=self.fwd_model)

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information.

        Returns:
            Dictionary containing system configuration information.
        """
        info = {
            'n_elec': self.n_elec,
            'pattern_config': self.pattern_config,
            'mesh_config': self.mesh_config,
            'initialized': self._is_initialized
        }
        
        if self._is_initialized:
            info.update({
                'n_elements': len(Function(self.fwd_model.V_sigma).vector()[:]),
                'n_nodes': self.fwd_model.V.dim(),
                'n_measurements': self.fwd_model.pattern_manager.n_meas_total,
                'n_stimulation_patterns': self.fwd_model.pattern_manager.n_stim
            })
        
        return info
