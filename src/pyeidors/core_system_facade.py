"""High-level workflow facade methods for :class:`EITSystem`."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .core_system_helpers import add_circular_phantom, collect_system_info, create_homogeneous_image
from .data.structures import EITData, EITImage
from .inverse import (
    ReconstructionResult,
    perform_absolute_reconstruction,
    perform_difference_reconstruction,
)


class CoreSystemFacadeMixin:
    """Convenience reconstruction/image helpers shared by ``EITSystem``."""

    def absolute_reconstruct(
        self,
        measurement_data: EITData,
        baseline_image: Optional[EITImage] = None,
        initial_image: Optional[EITImage] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReconstructionResult:
        baseline = baseline_image
        if baseline is None and self._is_initialized:
            baseline = self.create_homogeneous_image()
        return perform_absolute_reconstruction(
            eit_system=self,
            measurement_data=measurement_data,
            baseline_image=baseline,
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
        return perform_difference_reconstruction(
            eit_system=self,
            measurement_data=measurement_data,
            reference_data=reference_data,
            initial_image=initial_image,
            metadata=metadata,
        )

    def create_homogeneous_image(self, conductivity: Optional[float] = None) -> EITImage:
        self._require_initialized()
        return create_homogeneous_image(self, conductivity)

    def add_phantom(
        self,
        base_conductivity: float = 1.0,
        phantom_conductivity: float = 2.0,
        phantom_center: tuple = (0.3, 0.3),
        phantom_radius: float = 0.2,
    ) -> EITImage:
        self._require_initialized()
        return add_circular_phantom(
            self,
            base_conductivity=base_conductivity,
            phantom_conductivity=phantom_conductivity,
            phantom_center=phantom_center,
            phantom_radius=phantom_radius,
        )

    def get_system_info(self) -> Dict[str, Any]:
        return collect_system_info(self)
