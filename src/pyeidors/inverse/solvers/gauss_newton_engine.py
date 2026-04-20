"""Modular PyTorch-accelerated Gauss-Newton EIT Reconstructor."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from dolfinx import fem
from scipy.sparse import isspmatrix
from scipy.sparse.linalg import LinearOperator

from ...data.difference import (
    DEFAULT_DIFFERENCE_MODE,
    DEFAULT_DIFFERENCE_ORIENTATION,
    normalize_difference_mode,
    normalize_difference_orientation,
)
from ...femx import function_set_array
from ..contracts import SolverOutput
from ..jacobian.direct_jacobian import DirectJacobianCalculator
from ..regularization.base_regularization import BaseRegularization
from ..regularization.smoothness import SmoothnessRegularization
from .gauss_newton_device import resolve_torch_device
from .gauss_newton_line_search import (
    calc_perturb_limits,
    line_search_torch,
    update_perturb_eidors_style,
)
from .gauss_newton_runtime import ensure_measurement_weights, run_reconstruction
from .gauss_newton_weights import difference_with_baseline, scale_baseline_to_measured
from ...perf.policy import (
    DEFAULT_CHOLMOD_MAX_MEMORY_GIB,
    DEFAULT_CHOLMOD_MAX_N,
    DEFAULT_INEXACT_ETA0,
    DEFAULT_INEXACT_ETA_MAX,
    DEFAULT_INEXACT_ETA_MIN,
    DEFAULT_INEXACT_FORCING,
    DEFAULT_INEXACT_MODE,
    DEFAULT_LOWRANK_ENERGY,
    DEFAULT_LOWRANK_METHOD,
    DEFAULT_LOWRANK_MODE,
    DEFAULT_LOWRANK_RANK,
    DEFAULT_PRECONDITIONER,
    DEFAULT_ROM_MODE,
    DEFAULT_ROM_RANK_ADAPTIVE,
    DEFAULT_ROM_RANK_GLOBAL,
    DEFAULT_ROM_REFRESH_EVERY,
    DEFAULT_ROM_SNAPSHOT_SOURCE,
)

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


def _validate_option(name: str, value: str, allowed: set[str]) -> None:
    """Raise ``ValueError`` if *value* is not in *allowed*."""
    if value not in allowed:
        options = ", ".join(repr(v) for v in sorted(allowed))
        raise ValueError(f"Unsupported {name}={value!r}. Expected one of: {options}.")


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
        hyperparameter: float | None = None,
        line_search_steps: int = 8,
        clip_values: tuple[float, float] = (1e-6, 10.0),
        device: str = "auto",
        verbose: bool = True,
        use_measurement_weights: bool = False,
        weight_floor: float = 1e-9,
        measurement_weight_strategy: str = "none",
        max_step: float = 1.0,
        min_step: float = 0.1,
        negate_jacobian: bool = True,
        min_iterations: int = 1,
        use_prior_term: bool = True,
        difference_mode: str = DEFAULT_DIFFERENCE_MODE,
        difference_orientation: str = DEFAULT_DIFFERENCE_ORIENTATION,
        jacobian_background_conductivity: float = 1.0,
        difference_step_size_mode: str = "off",
        difference_step_size_value: float | None = None,
        difference_step_size_bounds: tuple[float, float] = (0.0, 4.0),
        difference_step_size_fmin_options: dict[str, Any] | None = None,
        difference_preset: str = "eidors_one_step_noser",
        absolute_preset: str = "eidors_abs_gn",
        best_homog_mode: str = "off",
        cache_manager=None,
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
    ):
        self.fwd_model = fwd_model
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self._regularization_param = 0.0
        self._hyperparameter = 0.0
        if hyperparameter is None:
            self.regularization_param = regularization_param
        else:
            self.hyperparameter = hyperparameter
        self.line_search_steps = line_search_steps
        self.clip_values = clip_values
        self.verbose = verbose
        self.measurement_weight_strategy = measurement_weight_strategy
        self.use_measurement_weights = use_measurement_weights or measurement_weight_strategy != "none"
        self.weight_floor = weight_floor
        self.negate_jacobian = negate_jacobian
        self.max_step = max_step
        self.min_step = min_step
        self.step_schedule: list[float] | None = None
        self.min_iterations = int(max(1, min_iterations))
        self.use_prior_term = use_prior_term
        self.difference_mode = normalize_difference_mode(
            difference_mode,
            default=DEFAULT_DIFFERENCE_MODE,
        )
        self.difference_orientation = normalize_difference_orientation(
            difference_orientation,
            default=DEFAULT_DIFFERENCE_ORIENTATION,
        )
        self.jacobian_background_conductivity = float(jacobian_background_conductivity)
        self.difference_step_size_mode = str(difference_step_size_mode).strip().lower()
        self.difference_step_size_value = (
            None if difference_step_size_value is None else float(difference_step_size_value)
        )
        self.difference_step_size_bounds = (
            float(difference_step_size_bounds[0]),
            float(difference_step_size_bounds[1]),
        )
        self.difference_step_size_fmin_options = dict(difference_step_size_fmin_options or {})
        self.difference_preset = str(difference_preset).strip().lower()
        self.absolute_preset = str(absolute_preset).strip().lower()
        self.active_preset_name = self.difference_preset
        self.best_homog_mode = str(best_homog_mode).strip().lower()
        self._prior_data: np.ndarray | None = None
        self._meas_weight_sqrt: torch.Tensor | None = None
        self._baseline_measurement: np.ndarray | None = None
        self._measured_vector: np.ndarray | None = None
        self._line_search_perturb: np.ndarray | None = None
        self._measurement_space_type = "real"
        self._difference_reference_meas: np.ndarray | None = None
        self._difference_target_meas: np.ndarray | None = None
        self._difference_mode_effective = self.difference_mode
        self._difference_orientation_effective = self.difference_orientation
        self.cache_manager = cache_manager
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
        _validate_option("performance_mode", self.performance_mode, {"safe", "aggressive"})
        _validate_option("solver_mode", self.solver_mode, {"strict", "fast"})
        _validate_option("linear_solver", self.linear_solver, {"auto", "petsc-ksp", "scipy-lsmr", "pyamg-cg", "cholmod"})
        _validate_option("line_search_mode", self.line_search_mode, {"full", "fast"})
        _validate_option(
            "preconditioner",
            self.preconditioner,
            {
                "auto",
                "diag",
                "noser",
                "prior",
                "pmat",
                "coarse",
                "custom",
                "pyamg",
                "cholmod",
                "petsc-gamg",
            },
        )
        _validate_option("fast_linear_path", self.fast_linear_path, {"auto", "woodbury", "pcg", "cholmod-direct", "strict"})
        _validate_option("rom_mode", self.rom_mode, {"off", "auto", "on"})
        _validate_option("rom_snapshot_source", self.rom_snapshot_source, {"cache", "synthetic", "hybrid"})
        _validate_option("inexact_mode", self.inexact_mode, {"off", "auto", "on"})
        _validate_option("inexact_forcing", self.inexact_forcing, {"fixed", "eisenstat-walker"})
        _validate_option("lowrank_mode", self.lowrank_mode, {"off", "auto", "on"})
        _validate_option("lowrank_method", self.lowrank_method, {"tsvd", "randomized"})
        if self.inexact_eta_min <= 0.0 or self.inexact_eta_max <= 0.0:
            raise ValueError("inexact eta bounds must be positive.")
        if self.inexact_eta_min > self.inexact_eta_max:
            raise ValueError("inexact_eta_min must be <= inexact_eta_max.")
        if not (0.0 < self.lowrank_energy <= 1.0):
            raise ValueError("lowrank_energy must be in (0, 1].")

        petsc_backend_info = getattr(self.fwd_model, "_petsc_backend_info", {}) or {}
        petsc_device_effective = str(petsc_backend_info.get("petsc_device_effective", "cpu"))
        device_resolution = resolve_torch_device(
            device,
            verbose=self.verbose,
            petsc_device_effective=petsc_device_effective,
        )
        self.device_requested = str(device_resolution.requested)
        self.device_effective = str(device_resolution.effective)
        self.device_fallback_reason = device_resolution.fallback_reason
        self.device = device_resolution.torch_device
        self._torch_dtype = torch.float64
        self.jacobian_calculator = jacobian_calculator or DirectJacobianCalculator(
            fwd_model,
            runtime_device=self.device_requested,
        )
        if hasattr(self.jacobian_calculator, "set_runtime_device"):
            self.jacobian_calculator.set_runtime_device(
                requested=self.device_requested,
                effective=self.device_effective,
                torch_device=self.device,
            )
        self.regularization = regularization or SmoothnessRegularization(fwd_model, alpha=1.0)

        self.n_elements = int(fem.Function(fwd_model.V_sigma).x.array.size)
        self.n_measurements = fwd_model.pattern_manager.n_meas_total
        self.R_torch = None
        self.R_matrix = None
        self.R_linear_operator: LinearOperator | None = None
        self.R_diag: np.ndarray | None = None

        if self.verbose:
            print(
                f"[INFO] GN config: lambda={self.regularization_param:.3e}, "
                f"hp={self.hyperparameter:.3e}, "
                f"use_prior_term={self.use_prior_term}"
            )
            print("Modular PyTorch Gauss-Newton Reconstructor initialized:")
            print(f"  Elements: {self.n_elements}")
            print(f"  Measurements: {self.n_measurements}")
            print(f"  Jacobian calculator: {type(self.jacobian_calculator).__name__}")
            print(f"  Regularization: {type(self.regularization).__name__}")
            print(f"  Device: {self.device}")
            print(
                f"  Solver mode: {self.solver_mode}, linear solver: {self.linear_solver}, "
                f"line-search: {self.line_search_mode}, preconditioner: {self.preconditioner}, "
                f"fast-linear-path: {self.fast_linear_path}"
            )
            print(
                "  Fused reduced mode: "
                f"rom={self.rom_mode} (rg={self.rom_rank_global}, ra={self.rom_rank_adaptive}, "
                f"refresh={self.rom_refresh_every}, source={self.rom_snapshot_source}), "
                f"inexact={self.inexact_mode} ({self.inexact_forcing}, eta0={self.inexact_eta0:.3g}), "
                f"lowrank={self.lowrank_mode} ({self.lowrank_method}, rank={self.lowrank_rank}, "
                f"energy={self.lowrank_energy:.3f})"
            )
            print(
                f"  CHOLMOD guard: max_n={self.cholmod_max_n}, "
                f"max_memory_gib={self.cholmod_max_memory_gib:.2f}"
            )

    @property
    def regularization_param(self) -> float:
        return float(self._regularization_param)

    @regularization_param.setter
    def regularization_param(self, value: float) -> None:
        resolved = max(0.0, float(value))
        self._regularization_param = resolved
        self._hyperparameter = float(np.sqrt(resolved))

    @property
    def hyperparameter(self) -> float:
        return float(self._hyperparameter)

    @hyperparameter.setter
    def hyperparameter(self, value: float | None) -> None:
        resolved = 0.0 if value is None else max(0.0, float(value))
        self._hyperparameter = resolved
        self._regularization_param = float(resolved * resolved)

    def _progress(self, total: int):
        return tqdm(total=total, disable=not self.verbose)

    def ensure_regularization_ready(self) -> None:
        """Build and validate the cached regularization tensor used by GN."""
        expected_shape = (self.n_elements, self.n_elements)
        needs_dense_tensor = self.solver_mode == "strict" or self.line_search_mode == "full"
        cache_ready = (
            self.R_matrix is not None
            and self.R_linear_operator is not None
            and (not needs_dense_tensor or self.R_torch is not None)
        )
        if cache_ready:
            return

        matrix = self.regularization.get_regularization_matrix()
        matrix_shape = tuple(getattr(matrix, "shape", ()))
        if matrix_shape != expected_shape:
            raise RuntimeError(
                "Regularization matrix shape mismatch: "
                f"expected {expected_shape}, got {matrix_shape}."
            )
        self.R_matrix = matrix
        as_linear_operator = getattr(self.regularization, "as_linear_operator", None)
        if callable(as_linear_operator):
            self.R_linear_operator = as_linear_operator(matrix, shape=expected_shape)
        else:
            self.R_linear_operator = BaseRegularization.as_linear_operator(matrix, shape=expected_shape)

        if isspmatrix(matrix):
            if matrix.nnz == 0:
                raise FloatingPointError("Regularization sparse matrix is empty.")
            if not np.isfinite(matrix.data).all():
                finite = matrix.data[np.isfinite(matrix.data)]
                min_val = float(finite.min()) if finite.size else float("nan")
                max_val = float(finite.max()) if finite.size else float("nan")
                raise FloatingPointError(
                    "Regularization sparse matrix contains non-finite values: "
                    f"finite_min={min_val:.6e}, finite_max={max_val:.6e}."
                )
            if matrix.format == "csr":
                diag = matrix.diagonal()
            else:
                diag = matrix.tocsr().diagonal()
            self.R_diag = np.asarray(diag, dtype=np.float64)
            if needs_dense_tensor:
                dense = matrix.toarray()
                self.R_torch = torch.from_numpy(np.asarray(dense, dtype=np.float64)).to(
                    self.device,
                    dtype=self._torch_dtype,
                )
            else:
                self.R_torch = None
            return

        if isinstance(matrix, LinearOperator):
            probe = np.ones(self.n_elements, dtype=np.float64)
            check = np.asarray(matrix.matvec(probe), dtype=np.float64)
            if not np.isfinite(check).all():
                raise FloatingPointError("Regularization LinearOperator produces non-finite values.")
            self.R_diag = None
            self.R_torch = None
            if self.solver_mode == "strict":
                raise RuntimeError(
                    "solver_mode='strict' requires explicit dense/sparse regularization matrix, "
                    "LinearOperator is not supported."
                )
            return

        dense = np.asarray(matrix, dtype=np.float64)
        if not np.isfinite(dense).all():
            finite = dense[np.isfinite(dense)]
            min_val = float(finite.min()) if finite.size else float("nan")
            max_val = float(finite.max()) if finite.size else float("nan")
            raise FloatingPointError(
                "Regularization matrix contains non-finite values: "
                f"finite_min={min_val:.6e}, finite_max={max_val:.6e}."
            )
        self.R_diag = np.asarray(np.diag(dense), dtype=np.float64)
        if needs_dense_tensor:
            self.R_torch = torch.from_numpy(dense).to(
                self.device,
                dtype=self._torch_dtype,
            )
            if not torch.isfinite(self.R_torch).all():
                raise FloatingPointError("Regularization tensor contains non-finite values after transfer.")
        else:
            self.R_torch = None

    def reconstruct(
        self,
        measured_data: object | np.ndarray,
        initial_conductivity: float = 1.0,
        jacobian_method: str = "efficient",
        prior_data: np.ndarray | None = None,
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
        self.R_matrix = None
        self.R_linear_operator = None
        self.R_diag = None
        if self.verbose:
            print(f"Regularization updated to: {type(regularization).__name__}")

    def set_jacobian_calculator(self, jacobian_calculator):
        self.jacobian_calculator = jacobian_calculator
        if self.verbose:
            print(f"Jacobian calculator updated to: {type(jacobian_calculator).__name__}")
