"""EIT-specific PDE adapter for CUQIpy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from dolfinx import fem

try:
    import cuqi.pde as cuqi_pde
except ImportError:  # pragma: no cover
    cuqi_pde = None

try:
    from cuqi.model import PDEModel
except ImportError:  # pragma: no cover
    PDEModel = None  # type: ignore[assignment]

from ...data.structures import EITImage
from ..jacobian.direct_jacobian import DirectJacobianCalculator


if cuqi_pde is not None:
    _PDEBase = cuqi_pde.PDE
else:  # pragma: no cover
    class _PDEBase:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("CUQIpy is required for EITPDE. Install cuqipy to use sparse Bayesian solvers.")


@dataclass
class EITGeometryInfo:
    """Container describing discretization sizes."""

    n_elements: int
    n_measurements: int


class EITPDE(_PDEBase):
    """Wrap the PyEIDORS forward model as a CUQI PDE."""

    def __init__(self, eit_system):
        if cuqi_pde is None:  # pragma: no cover
            raise ImportError("CUQIpy is required for EITPDE. Install cuqipy to use sparse Bayesian solvers.")
        super().__init__(PDE_form=None)
        self._eit_system = eit_system
        self._fwd_model = eit_system.fwd_model
        self._V_sigma = self._fwd_model.V_sigma
        self._sigma_function = fem.Function(self._V_sigma)

        self._jacobian_calculator = DirectJacobianCalculator(self._fwd_model)
        self._current_image: Optional[EITImage] = None
        self._cached_jacobian: Optional[np.ndarray] = None
        self._cached_sigma_vector: Optional[np.ndarray] = None

    def assemble(self, parameter) -> None:
        param_array = np.asarray(parameter, dtype=float).ravel()
        expected = int(self._sigma_function.x.array.size)
        if param_array.size != expected:
            raise ValueError(
                f"Parameter length mismatch: got {param_array.size}, expected {expected}"
            )

        self._sigma_function.x.array[:] = param_array
        self._current_image = EITImage(elem_data=param_array, fwd_model=self._fwd_model)

        self._cached_sigma_vector = param_array.copy()
        self._cached_jacobian = None

    def solve(self) -> Tuple[object, dict]:
        if self._current_image is None:
            raise RuntimeError("assemble() must be called before solve().")

        data, potentials = self._fwd_model.fwd_solve(self._current_image)
        return data, {"potentials": potentials}

    def observe(self, solution) -> np.ndarray:
        data = solution[0] if isinstance(solution, tuple) else solution
        if hasattr(data, "meas"):
            return np.asarray(data.meas, dtype=float)
        return np.asarray(data, dtype=float)

    def _ensure_sigma(self, wrt: np.ndarray) -> fem.Function:
        wrt_array = np.asarray(wrt, dtype=float).ravel()
        expected = int(self._sigma_function.x.array.size)
        if wrt_array.size != expected:
            raise ValueError(
                f"wrt size mismatch: got {wrt_array.size}, expected {expected}"
            )
        sigma = fem.Function(self._V_sigma)
        sigma.x.array[:] = wrt_array
        return sigma

    def gradient_wrt_parameter(self, direction, wrt):
        sigma = self._ensure_sigma(wrt)
        jacobian = self._jacobian_calculator.calculate(sigma)
        direction_vec = np.asarray(direction, dtype=float).ravel()
        return jacobian.T @ direction_vec

    def jacobian_wrt_parameter(self, wrt):
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

    @property
    def geometry_info(self) -> EITGeometryInfo:
        return EITGeometryInfo(
            n_elements=int(self._sigma_function.x.array.size),
            n_measurements=self._fwd_model.pattern_manager.n_meas_total,
        )

    def forward(self, parameter: np.ndarray) -> np.ndarray:
        self.assemble(parameter)
        return self.observe(self.solve())


def create_pde_model(eit_system) -> Tuple[EITPDE, "PDEModel", EITGeometryInfo]:
    """Construct CUQI PDEModel for ``EITSystem``."""

    if PDEModel is None:
        raise ImportError(
            "cuqi.model.PDEModel is unavailable. Install a CUQIpy version that provides PDEModel."
        )

    eit_pde = EITPDE(eit_system)
    geom = eit_pde.geometry_info
    model = PDEModel(
        PDE=eit_pde,
        range_geometry=geom.n_measurements,
        domain_geometry=geom.n_elements,
    )
    return eit_pde, model, geom
