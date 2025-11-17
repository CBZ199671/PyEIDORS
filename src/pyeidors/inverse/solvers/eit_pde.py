"""EIT-specific PDE adapter for CUQIpy-FEniCS.

This module exposes the forward model used in PyEidors as a CUQI-compatible
PDE object so that it can be wrapped by ``cuqipy_fenics.testproblem.PDEModel``.
It bridges the existing ``EITForwardModel`` (which already relies on FEniCS)
with the higher-level CUQI abstractions, reducing the amount of bespoke glue
code required inside the Bayesian reconstructor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from fenics import Function

import cuqi.pde
from cuqipy_fenics.testproblem import PDEModel

from ...data.structures import EITImage
from ..jacobian.direct_jacobian import DirectJacobianCalculator


@dataclass
class EITGeometryInfo:
    """Simple container describing the discretisation sizes."""

    n_elements: int
    n_measurements: int


class EITPDE(cuqi.pde.PDE):
    """Wrap the PyEidors forward model as a CUQI PDE.

    The implementation delegates the assembly/solve steps to the existing
    :class:`EITForwardModel` while exposing the observation and Jacobian
    evaluations in the interface expected by CUQI.
    """

    def __init__(self, eit_system):
        super().__init__(PDE_form=None)
        self._eit_system = eit_system
        self._fwd_model = eit_system.fwd_model
        self._V_sigma = self._fwd_model.V_sigma
        self._sigma_function = Function(self._V_sigma)

        self._jacobian_calculator = DirectJacobianCalculator(self._fwd_model)
        self._current_image: Optional[EITImage] = None
        self._cached_jacobian: Optional[np.ndarray] = None
        self._cached_sigma_vector: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # cuqi.pde.PDE interface
    # ------------------------------------------------------------------
    def assemble(self, parameter) -> None:
        """Store the conductivity parameter and reset caches."""
        param_array = np.asarray(parameter, dtype=float).ravel()
        if param_array.size != self._sigma_function.vector().size():
            raise ValueError(
                f"Parameter length mismatch: got {param_array.size}, "
                f"expected {self._sigma_function.vector().size()}"
            )

        self._sigma_function.vector()[:] = param_array
        self._current_image = EITImage(elem_data=param_array, fwd_model=self._fwd_model)

        # Invalidate Jacobian cache whenever the parameter changes.
        self._cached_sigma_vector = param_array.copy()
        self._cached_jacobian = None

    def solve(self) -> Tuple[object, dict]:
        """Run the existing forward solver and keep the raw outputs."""
        if self._current_image is None:
            raise RuntimeError("assemble() must be called before solve().")

        data, potentials = self._fwd_model.fwd_solve(self._current_image)
        return data, {"potentials": potentials}

    def observe(self, solution) -> np.ndarray:
        """Return the measurement vector from the forward solution."""
        data = solution[0] if isinstance(solution, tuple) else solution
        if hasattr(data, "meas"):
            return np.asarray(data.meas, dtype=float)
        return np.asarray(data, dtype=float)

    # ------------------------------------------------------------------
    # Jacobian helpers
    # ------------------------------------------------------------------
    def _ensure_sigma(self, wrt: np.ndarray) -> Function:
        wrt_array = np.asarray(wrt, dtype=float).ravel()
        if wrt_array.size != self._sigma_function.vector().size():
            raise ValueError(
                f"wrt size mismatch: got {wrt_array.size}, "
                f"expected {self._sigma_function.vector().size()}"
            )
        sigma = Function(self._V_sigma)
        sigma.vector()[:] = wrt_array
        return sigma

    def gradient_wrt_parameter(self, direction, wrt):
        """Compute J^T * direction for the current parameter."""
        sigma = self._ensure_sigma(wrt)
        jacobian = self._jacobian_calculator.calculate(sigma)
        direction_vec = np.asarray(direction, dtype=float).ravel()
        return jacobian.T @ direction_vec

    def jacobian_wrt_parameter(self, wrt):
        """Return the measurement Jacobian evaluated at `wrt`."""
        wrt_array = np.asarray(wrt, dtype=float).ravel()
        if (
            self._cached_jacobian is not None
            and self._cached_sigma_vector is not None
            and np.allclose(self._cached_sigma_vector, wrt_array, atol=1e-14, rtol=1e-12)
        ):
            return self._cached_jacobian

        sigma = self._ensure_sigma(wrt_array)
        jacobian = self._jacobian_calculator.calculate(sigma)
        self._cached_sigma_vector = wrt_array.copy()
        self._cached_jacobian = jacobian
        return jacobian

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @property
    def geometry_info(self) -> EITGeometryInfo:
        return EITGeometryInfo(
            n_elements=self._sigma_function.vector().size(),
            n_measurements=self._fwd_model.pattern_manager.n_meas_total,
        )

    def forward(self, parameter: np.ndarray) -> np.ndarray:
        """Run assemble/solve/observe in one step (utility for callers)."""
        self.assemble(parameter)
        return self.observe(self.solve())


def create_pde_model(eit_system) -> Tuple[EITPDE, PDEModel, EITGeometryInfo]:
    """Construct the CUQI PDEModel for a given :class:`EITSystem`."""
    eit_pde = EITPDE(eit_system)
    geom = eit_pde.geometry_info
    model = PDEModel(
        PDE=eit_pde,
        range_geometry=geom.n_measurements,
        domain_geometry=geom.n_elements,
    )
    return eit_pde, model, geom
