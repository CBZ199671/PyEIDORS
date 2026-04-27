"""EIT Jacobian matrix calculator base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from dolfinx import fem

from ...data.structures import EITImage


class BaseJacobianCalculator(ABC):
    """Jacobian calculator base class."""

    def __init__(self, fwd_model):
        self.fwd_model = fwd_model
        self.n_elements = int(fem.Function(fwd_model.V_sigma).x.array.size)
        self.n_measurements = fwd_model.pattern_manager.n_meas_total

    @abstractmethod
    def calculate(self, sigma: fem.Function, **kwargs) -> np.ndarray:
        """Calculate Jacobian matrix."""

    def calculate_from_image(self, img: EITImage, **kwargs) -> np.ndarray:
        sigma = fem.Function(self.fwd_model.V_sigma)
        sigma.x.array[:] = img.get_conductivity()
        return self.calculate(sigma, **kwargs)
