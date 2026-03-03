"""EIT Jacobian matrix calculator base class."""

import numpy as np
from abc import ABC, abstractmethod
from fenics import Function

from ...data.structures import EITImage


class BaseJacobianCalculator(ABC):
    """Jacobian calculator base class."""

    def __init__(self, fwd_model):
        self.fwd_model = fwd_model
        self.n_elements = len(Function(fwd_model.V_sigma).vector()[:])
        self.n_measurements = fwd_model.pattern_manager.n_meas_total

    @abstractmethod
    def calculate(self, sigma: Function, **kwargs) -> np.ndarray:
        """Calculate Jacobian matrix.

        Args:
            sigma: Current conductivity distribution.

        Returns:
            Jacobian matrix (n_measurements x n_elements).
        """
        pass

    def calculate_from_image(self, img: EITImage, **kwargs) -> np.ndarray:
        """Calculate Jacobian matrix from EIT image."""
        sigma = Function(self.fwd_model.V_sigma)
        sigma.vector()[:] = img.get_conductivity()
        return self.calculate(sigma, **kwargs)