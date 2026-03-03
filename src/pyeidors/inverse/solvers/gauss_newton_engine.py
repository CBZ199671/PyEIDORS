"""Modular PyTorch-accelerated Gauss-Newton EIT Reconstructor."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
from dolfinx import fem

from ...femx import function_set_array
from ..contracts import SolverOutput
from ..jacobian.direct_jacobian import DirectJacobianCalculator
from ..regularization.smoothness import SmoothnessRegularization
from .gauss_newton_device import resolve_torch_device
from .gauss_newton_line_search import (
    calc_perturb_limits,
    line_search_torch,
    update_perturb_eidors_style,
)
from .gauss_newton_runtime import ensure_measurement_weights, run_reconstruction
from .gauss_newton_weights import difference_with_baseline, scale_baseline_to_measured

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency fallback
    class _NoOpTqdm:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def update(self, n: int = 1):
            pass

        def set_postfix_str(self, s: str):
            pass

    def tqdm(*args, **kwargs):  # type: ignore[override]
        return _NoOpTqdm(*args, **kwargs)


class GaussNewtonReconstructor:
    """Gauss-Newton solver with optional measurement weighting and line-search."""

    def __init__(
        self,
        fwd_model,
        jacobian_calculator=None,
        regularization=None,
        max_iterations: int = 15,
        convergence_tol: float = 1e-4,
        regularization_param: float = 0.01,
        line_search_steps: int = 8,
        clip_values: Tuple[float, float] = (1e-6, 10.0),
        device: str = "cuda:0",
        verbose: bool = True,
        use_measurement_weights: bool = False,
        weight_floor: float = 1e-9,
        measurement_weight_strategy: str = "none",
        max_step: float = 1.0,
        min_step: float = 0.1,
        negate_jacobian: bool = True,
        min_iterations: int = 1,
        use_prior_term: bool = True,
    ):
        self.fwd_model = fwd_model
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.regularization_param = regularization_param
        self.line_search_steps = line_search_steps
        self.clip_values = clip_values
        self.verbose = verbose
        self.measurement_weight_strategy = measurement_weight_strategy
        self.use_measurement_weights = use_measurement_weights or measurement_weight_strategy != "none"
        self.weight_floor = weight_floor
        self.negate_jacobian = negate_jacobian
        self.max_step = max_step
        self.min_step = min_step
        self.step_schedule: Optional[list[float]] = None
        self.min_iterations = int(max(1, min_iterations))
        self.use_prior_term = use_prior_term
        self._prior_data: Optional[np.ndarray] = None
        self._meas_weight_sqrt: Optional[torch.Tensor] = None
        self._baseline_measurement: Optional[np.ndarray] = None
        self._measured_vector: Optional[np.ndarray] = None
        self._line_search_perturb: Optional[np.ndarray] = None

        self.device = resolve_torch_device(device, verbose=self.verbose)
        self._torch_dtype = torch.float64
        self.jacobian_calculator = jacobian_calculator or DirectJacobianCalculator(fwd_model)
        self.regularization = regularization or SmoothnessRegularization(fwd_model, alpha=1.0)

        self.n_elements = int(fem.Function(fwd_model.V_sigma).x.array.size)
        self.n_measurements = fwd_model.pattern_manager.n_meas_total
        self.R_torch = None

        if self.verbose:
            print(
                f"[INFO] GN config: lambda={self.regularization_param:.3e}, "
                f"use_prior_term={self.use_prior_term}"
            )
            print("Modular PyTorch Gauss-Newton Reconstructor initialized:")
            print(f"  Elements: {self.n_elements}")
            print(f"  Measurements: {self.n_measurements}")
            print(f"  Jacobian calculator: {type(self.jacobian_calculator).__name__}")
            print(f"  Regularization: {type(self.regularization).__name__}")
            print(f"  Device: {self.device}")

    def _progress(self, total: int):
        return tqdm(total=total, disable=not self.verbose)

    def reconstruct(
        self,
        measured_data: Union[object, np.ndarray],
        initial_conductivity: float = 1.0,
        jacobian_method: str = "efficient",
        prior_data: Optional[np.ndarray] = None,
        record_conductivity_history: bool = False,
        conductivity_history_stride: int = 1,
    ) -> SolverOutput:
        return run_reconstruction(
            reconstructor=self,
            measured_data=measured_data,
            initial_conductivity=initial_conductivity,
            jacobian_method=jacobian_method,
            prior_data=prior_data,
            record_conductivity_history=record_conductivity_history,
            conductivity_history_stride=conductivity_history_stride,
        )

    # Private API wrappers retained for tests and focused diagnostics.
    def _ensure_measurement_weights(self, sigma_function: fem.Function) -> None:
        ensure_measurement_weights(self, sigma_function)

    def _scale_baseline_to_measured(self, baseline_vector: np.ndarray) -> np.ndarray:
        return scale_baseline_to_measured(baseline_vector, self._measured_vector)

    def _difference_with_baseline(self, baseline_vector: np.ndarray) -> np.ndarray:
        return difference_with_baseline(baseline_vector, self._measured_vector, self.weight_floor)

    def _line_search_torch(
        self,
        sigma_current,
        delta_sigma_torch,
        meas_target_torch,
        current_weighted_residual,
        weight_vector=None,
        prior_torch=None,
        lambda_eff=None,
        retry: int = 0,
    ):
        return line_search_torch(
            self,
            sigma_current,
            delta_sigma_torch,
            meas_target_torch,
            current_weighted_residual,
            weight_vector,
            prior_torch,
            lambda_eff,
            retry=retry,
        )

    def _calc_perturb_limits(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        return calc_perturb_limits(self, x, dx)

    def _update_perturb_eidors_style(
        self,
        chosen_step: float,
        perturb: np.ndarray,
        mlist: np.ndarray,
        valid_idx: np.ndarray,
    ) -> None:
        update_perturb_eidors_style(self, chosen_step, perturb, mlist, valid_idx)

    def set_regularization(self, regularization):
        self.regularization = regularization
        self.R_torch = None
        if self.verbose:
            print(f"Regularization updated to: {type(regularization).__name__}")

    def set_jacobian_calculator(self, jacobian_calculator):
        self.jacobian_calculator = jacobian_calculator
        if self.verbose:
            print(f"Jacobian calculator updated to: {type(jacobian_calculator).__name__}")
