"""Runtime iteration helpers for the Gauss-Newton solver."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np
import torch
from dolfinx import fem
from scipy.sparse.linalg import LinearOperator
from scipy.optimize import minimize_scalar


from ...data.difference import project_measurement_vector
from ...data.structures import EITImage
from ...femx import function_get_array, function_set_array
from ...utils.numeric_ops import has_nonzero_imaginary
from ..contracts import SolverOutput
from ..jacobian.linearized import JacobianLinearization, compute_sigma_fingerprint
from ..jacobian.process_jacobian_cache import (
    build_process_jacobian_key,
    get_process_cached_jacobian,
    put_process_cached_jacobian,
)
from ..matrix_free.dual_mesh import DualMeshJacobianOperator
from .gauss_newton_iteration_log import (  # noqa: F401  re-exported for V73 contract
    _IterationLog,
    _record_iteration_log,
)
from .gauss_newton_measurement_space import (  # noqa: F401  re-exported for V73 contract
    _configure_measurement_space,
    _extract_measured_vector,
    _measurement_space_kwargs,
    _project_measurement_jacobian,
    _project_simulated_measurements,
)
from .gauss_newton_weights import build_weight_reference

from . import gauss_newton_linear_system as _linear_system
from . import gauss_newton_startup_cache as _startup_cache
from . import gauss_newton_step_size as _step_size


def _diagnostic_real_or_complex_scalar(value) -> float | str:
    array = np.asarray(value)
    if array.size != 1:
        return repr(value)
    scalar = array.reshape(-1)[0]
    if has_nonzero_imaginary(array):
        z = complex(scalar)
        return f"{z.real:g}{z.imag:+g}j"
    return float(np.real(scalar))


def _torch_dtype_for_values(*values) -> torch.dtype:
    saw_complex64 = False
    saw_complex128 = False
    for value in values:
        if value is None:
            continue
        arr = np.asarray(value)
        if not np.iscomplexobj(arr):
            continue
        if arr.dtype == np.complex64:
            saw_complex64 = True
        else:
            saw_complex128 = True
    if saw_complex128:
        return torch.complex128
    if saw_complex64:
        return torch.complex64
    return torch.float64


def _is_complex_array_like(value) -> bool:
    try:
        return bool(np.iscomplexobj(np.asarray(value)))
    except Exception:
        return False


def _vdot_real_torch(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    if lhs.is_complex() or rhs.is_complex():
        return float(torch.vdot(lhs.reshape(-1), rhs.reshape(-1)).real.item())
    return float(torch.dot(lhs.reshape(-1), rhs.reshape(-1)).item())


def _hermitian_transpose(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.is_complex():
        return matrix.conj().transpose(0, 1)
    return matrix.t()


def _clip_real_sigma(values: np.ndarray, clip_values) -> np.ndarray:
    arr = np.asarray(values)
    if clip_values is None or np.iscomplexobj(arr):
        return arr
    return np.clip(arr, clip_values[0], clip_values[1])


# T77 phase 2 keeps these legacy runtime-level names patchable/importable.
_JacobianActionBundle = _linear_system._JacobianActionBundle
_PETScMatrixFreeHessianContext = _linear_system._PETScMatrixFreeHessianContext
_PETScMatrixFreePCContext = _linear_system._PETScMatrixFreePCContext
_apply_regularization_np = _linear_system._apply_regularization_np
_as_sparse_regularization_matrix = _linear_system._as_sparse_regularization_matrix
_build_matrix_free_custom_pc_operator = (
    _linear_system._build_matrix_free_custom_pc_operator
)
_build_matrix_free_pmat_inverse_operator = (
    _linear_system._build_matrix_free_pmat_inverse_operator
)
_build_matrix_free_explicit_pc_operator = (
    _linear_system._build_matrix_free_explicit_pc_operator
)
_coerce_preconditioner_diag = _linear_system._coerce_preconditioner_diag
_diag_preconditioner = _linear_system._diag_preconditioner
_finite_summary = _linear_system._finite_summary
_is_jv_jtr_action = _linear_system._is_jv_jtr_action
_jv_jtr_action_shape = _linear_system._jv_jtr_action_shape
_jv_jtr_action_representation = _linear_system._jv_jtr_action_representation
_matrix_free_pc_floor = _linear_system._matrix_free_pc_floor
_matrix_free_pmat_candidates = _linear_system._matrix_free_pmat_candidates
_operator_diag_preconditioner = _linear_system._operator_diag_preconditioner
_petsc_vec_to_numpy = _linear_system._petsc_vec_to_numpy
_regularization_looks_like_noser = _linear_system._regularization_looks_like_noser
_require_finite = _linear_system._require_finite
_require_scalar_finite = _linear_system._require_scalar_finite
_sanitize_preconditioner_diag = _linear_system._sanitize_preconditioner_diag

_PETSc = _linear_system._PETSc
pyamg = _linear_system.pyamg
cholmod_cholesky = _linear_system.cholmod_cholesky
cg = _linear_system.cg
lsmr = _linear_system.lsmr
cho_factor = _linear_system.cho_factor
cho_solve = _linear_system.cho_solve
InexactController = _linear_system.InexactController
SnapshotBank = _linear_system.SnapshotBank
backend_signature_from_forward_model = (
    _linear_system.backend_signature_from_forward_model
)
build_lowrank_subspace = _linear_system.build_lowrank_subspace
build_reduced_operator = _linear_system.build_reduced_operator
compute_pod_basis = _linear_system.compute_pod_basis
detect_performance_capabilities = _linear_system.detect_performance_capabilities
merge_orthonormal_bases = _linear_system.merge_orthonormal_bases
model_signature_from_forward_model = _linear_system.model_signature_from_forward_model
pattern_signature_from_forward_model = (
    _linear_system.pattern_signature_from_forward_model
)
rom_signature = _linear_system.rom_signature
safe_dot = _linear_system.safe_dot
select_fast_linear_path = _linear_system.select_fast_linear_path
select_fused_strategy = _linear_system.select_fused_strategy
select_preconditioner = _linear_system.select_preconditioner
select_snapshot_matrix = _linear_system.select_snapshot_matrix
solve_reduced_step = _linear_system.solve_reduced_step

_LINEAR_SYSTEM_PATCHABLE_NAMES = (
    "_PETSc",
    "_apply_regularization_np",
    "backend_signature_from_forward_model",
    "build_lowrank_subspace",
    "build_reduced_operator",
    "cg",
    "cho_factor",
    "cho_solve",
    "cholmod_cholesky",
    "compute_pod_basis",
    "detect_performance_capabilities",
    "InexactController",
    "lsmr",
    "merge_orthonormal_bases",
    "model_signature_from_forward_model",
    "pattern_signature_from_forward_model",
    "pyamg",
    "rom_signature",
    "safe_dot",
    "select_fast_linear_path",
    "select_fused_strategy",
    "select_preconditioner",
    "select_snapshot_matrix",
    "SnapshotBank",
    "solve_reduced_step",
)
_STARTUP_CACHE_PATCHABLE_NAMES = (
    "backend_signature_from_forward_model",
    "function_get_array",
    "model_signature_from_forward_model",
    "pattern_signature_from_forward_model",
)
_STARTUP_CACHE_DEFAULT_PATCHES = {
    name: getattr(_startup_cache, name) for name in _STARTUP_CACHE_PATCHABLE_NAMES
}
_STARTUP_CACHE_LAST_SYNCED: dict[str, object] = {}
_STEP_SIZE_PATCHABLE_NAMES = (
    "EITImage",
    "_project_simulated_measurements",
    "_require_scalar_finite",
    "minimize_scalar",
)
_STEP_SIZE_DEFAULT_PATCHES = {
    name: getattr(_step_size, name) for name in _STEP_SIZE_PATCHABLE_NAMES
}
_STEP_SIZE_LAST_SYNCED: dict[str, object] = {}


def _sync_linear_system_runtime_overrides() -> None:
    for name in _LINEAR_SYSTEM_PATCHABLE_NAMES:
        if name in globals():
            setattr(_linear_system, name, globals()[name])


def _sync_startup_cache_runtime_overrides() -> None:
    for name in _STARTUP_CACHE_PATCHABLE_NAMES:
        if name in globals():
            runtime_value = globals()[name]
            default_value = _STARTUP_CACHE_DEFAULT_PATCHES[name]
            if runtime_value is not default_value:
                setattr(_startup_cache, name, runtime_value)
                _STARTUP_CACHE_LAST_SYNCED[name] = runtime_value
                continue
            if getattr(_startup_cache, name) is _STARTUP_CACHE_LAST_SYNCED.get(name):
                setattr(_startup_cache, name, default_value)
                _STARTUP_CACHE_LAST_SYNCED.pop(name, None)


def _sync_step_size_runtime_overrides() -> None:
    for name in _STEP_SIZE_PATCHABLE_NAMES:
        if name in globals():
            runtime_value = globals()[name]
            default_value = _STEP_SIZE_DEFAULT_PATCHES[name]
            if runtime_value is not default_value:
                setattr(_step_size, name, runtime_value)
                _STEP_SIZE_LAST_SYNCED[name] = runtime_value
                continue
            if getattr(_step_size, name) is _STEP_SIZE_LAST_SYNCED.get(name):
                setattr(_step_size, name, default_value)
                _STEP_SIZE_LAST_SYNCED.pop(name, None)


def _startup_cache_payload(
    reconstructor, sigma_array: np.ndarray, jacobian_method: str
) -> dict[str, object]:
    _sync_startup_cache_runtime_overrides()
    return _startup_cache._startup_cache_payload(
        reconstructor, sigma_array, jacobian_method
    )


def _startup_cache_lookup(
    reconstructor,
    sigma_current: fem.Function,
    jacobian_method: str,
) -> tuple[np.ndarray | None, dict[str, object]]:
    _sync_startup_cache_runtime_overrides()
    return _startup_cache._startup_cache_lookup(
        reconstructor,
        sigma_current,
        jacobian_method,
    )


def _difference_step_size_objective(
    reconstructor,
    *,
    prior_sigma: np.ndarray,
    delta_sigma: np.ndarray,
    measured_vector: np.ndarray,
    alpha: float,
) -> float:
    _sync_step_size_runtime_overrides()
    return _step_size._difference_step_size_objective(
        reconstructor,
        prior_sigma=prior_sigma,
        delta_sigma=delta_sigma,
        measured_vector=measured_vector,
        alpha=alpha,
    )


def _apply_difference_step_size(
    reconstructor,
    *,
    sigma_final: np.ndarray,
    measured_vector: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    _sync_step_size_runtime_overrides()
    return _step_size._apply_difference_step_size(
        reconstructor,
        sigma_final=sigma_final,
        measured_vector=measured_vector,
    )


def _solve_matrix_free_hessian_via_petsc(
    h_op: LinearOperator,
    rhs: np.ndarray,
    m_op: LinearOperator | None,
    *,
    rtol: float,
    maxiter: int,
    petsc_module=None,
) -> tuple[np.ndarray, int, bool, str | None]:
    _sync_linear_system_runtime_overrides()
    effective_petsc = _PETSc if petsc_module is None else petsc_module
    return _linear_system._solve_matrix_free_hessian_via_petsc(
        h_op,
        rhs,
        m_op,
        rtol=rtol,
        maxiter=maxiter,
        petsc_module=effective_petsc,
    )


def _as_jacobian_action_bundle(
    jacobian,
    *,
    measurement_weight_np: np.ndarray | None = None,
) -> _JacobianActionBundle:
    _sync_linear_system_runtime_overrides()
    return _linear_system._as_jacobian_action_bundle(
        jacobian,
        measurement_weight_np=measurement_weight_np,
    )


def _solve_linear_system_fast(
    reconstructor,
    *,
    J_weighted_np,
    weighted_residual_np: np.ndarray,
    de_current_np: np.ndarray,
    lambda_eff: float,
    iteration: int,
    measurement_weight_np: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    _sync_linear_system_runtime_overrides()
    return _linear_system._solve_linear_system_fast(
        reconstructor,
        J_weighted_np=J_weighted_np,
        weighted_residual_np=weighted_residual_np,
        de_current_np=de_current_np,
        lambda_eff=lambda_eff,
        iteration=iteration,
        measurement_weight_np=measurement_weight_np,
    )


def _select_step_size(
    reconstructor,
    iteration: int,
    sigma_current: fem.Function,
    delta_sigma_torch: torch.Tensor,
    meas_torch: torch.Tensor,
    residual_norm_weighted: float,
    prior_torch: torch.Tensor,
    lambda_eff: float,
) -> float:
    _sync_step_size_runtime_overrides()
    return _step_size._select_step_size(
        reconstructor,
        iteration,
        sigma_current,
        delta_sigma_torch,
        meas_torch,
        residual_norm_weighted,
        prior_torch,
        lambda_eff,
    )


def _to_runtime_tensor(reconstructor, values) -> torch.Tensor:
    if isinstance(values, np.ndarray) and not values.flags.writeable:
        values = values.copy()
    return torch.as_tensor(
        values,
        device=reconstructor.device,
        dtype=reconstructor._torch_dtype,
    )


def _to_runtime_tensor_cached(
    reconstructor,
    name: str,
    values,
) -> torch.Tensor:
    """Reuse per-iteration tensor buffers to reduce repeated allocations."""
    source = _to_runtime_tensor(reconstructor, values)
    cache = getattr(reconstructor, "_runtime_tensor_cache", None)
    if cache is None:
        cache = {}
        reconstructor._runtime_tensor_cache = cache
    target = cache.get(name)
    if (
        target is None
        or tuple(target.shape) != tuple(source.shape)
        or target.dtype != source.dtype
        or target.device != source.device
    ):
        target = source.clone()
        cache[name] = target
        return target
    target.copy_(source)
    return target


def ensure_measurement_weights(reconstructor, sigma_function: fem.Function) -> None:
    """Compute measurement weights based on baseline forward solution."""
    strategy = reconstructor.measurement_weight_strategy
    if not reconstructor.use_measurement_weights or strategy == "none":
        reconstructor._meas_weight_sqrt = None
        reconstructor._baseline_measurement = None
        return

    img = EITImage(
        elem_data=function_get_array(sigma_function), fwd_model=reconstructor.fwd_model
    )
    baseline_data, _ = reconstructor.fwd_model.fwd_solve(img)
    baseline_vector = project_measurement_vector(
        baseline_data.meas, **_measurement_space_kwargs(reconstructor)
    )
    reconstructor._baseline_measurement = baseline_vector.copy()

    reference_vector = build_weight_reference(
        strategy=strategy,
        baseline_vector=baseline_vector,
        measured_vector=reconstructor._measured_vector,
        floor=reconstructor.weight_floor,
    )

    weights = np.abs(reference_vector)
    np.square(weights, out=weights)
    np.nan_to_num(weights, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(weights, reconstructor.weight_floor, out=weights)
    median = np.median(weights)
    if median > 0:
        weights /= median

    reconstructor._meas_weight_sqrt = _to_runtime_tensor(
        reconstructor,
        np.sqrt(weights),
    )
    if reconstructor.verbose:
        w_min = float(np.min(weights)) if weights.size else float("nan")
        w_max = float(np.max(weights)) if weights.size else float("nan")
        w_med = float(np.median(weights)) if weights.size else float("nan")
        print(
            f"[INFO] measurement weights ({strategy}): min={w_min:.3e}, med={w_med:.3e}, max={w_max:.3e}"
        )


def _is_operator_jacobian_method(jacobian_method: str) -> bool:
    method = str(jacobian_method).strip().lower().replace("_", "-")
    return method in {"linearized", "operator", "matrix-free"}


def _is_matrix_free_jacobian(jacobian) -> bool:
    return isinstance(
        jacobian,
        (JacobianLinearization, LinearOperator, DualMeshJacobianOperator),
    ) or _is_jv_jtr_action(jacobian)


def _scale_jacobian_action(jacobian, scale: float):
    scale = float(scale)
    if scale == 1.0:
        return jacobian
    if isinstance(jacobian, JacobianLinearization):
        return replace(jacobian, sign=jacobian.sign * scale)
    if isinstance(jacobian, LinearOperator):
        dtype = getattr(jacobian, "dtype", np.float64)
        return LinearOperator(
            jacobian.shape,
            matvec=lambda x: scale * np.asarray(jacobian.matvec(x)),
            rmatvec=lambda x: scale * np.asarray(jacobian.rmatvec(x)),
            dtype=dtype,
        )
    if _is_jv_jtr_action(jacobian):
        shape = _jv_jtr_action_shape(jacobian)
        dtype = getattr(jacobian, "dtype", np.float64)
        return LinearOperator(
            shape,
            matvec=lambda x: scale * np.asarray(jacobian.Jv(np.asarray(x))),
            rmatvec=lambda x: scale * np.asarray(jacobian.JTr(np.asarray(x))),
            dtype=dtype,
        )
    return scale * np.asarray(jacobian)


def _persistent_jacobian_cache_key(
    reconstructor,
    sigma_current,
    jacobian_method: str,
) -> str | None:
    """Build a process-cache key for the current Jacobian, or ``None``.

    Returns ``None`` when the persistent cache is disabled, when the
    sigma fingerprint cannot be computed, when the mesh has no stable
    identifier (V17), or when the forward model is missing required
    signatures.
    """
    if not bool(getattr(reconstructor, "persistent_jacobian_cache", False)):
        return None
    fwd_model = getattr(reconstructor, "fwd_model", None)
    if fwd_model is None:
        return None
    try:
        sigma_fp = compute_sigma_fingerprint(sigma_current)
    except Exception:
        return None
    if not sigma_fp:
        return None
    mesh = getattr(fwd_model, "mesh", None) or getattr(fwd_model, "eit_mesh", None)
    mesh_file = str(getattr(mesh, "mesh_file", "") or "")
    mesh_content_hash = ""
    try:
        from ...forward.eit_forward_model import _hash_mesh_content

        dolfinx_mesh = getattr(mesh, "mesh", mesh)
        mesh_content_hash = _hash_mesh_content(dolfinx_mesh) or ""
    except Exception:
        mesh_content_hash = ""
    if not mesh_file and not mesh_content_hash:
        return None
    try:
        return build_process_jacobian_key(
            sigma_fingerprint=sigma_fp,
            mesh_file=mesh_file or None,
            mesh_content_hash=mesh_content_hash or None,
            jacobian_method=str(jacobian_method),
            calculator_signature=_jacobian_calculator_cache_signature(
                getattr(reconstructor, "jacobian_calculator", None)
            ),
            model_signature=model_signature_from_forward_model(fwd_model),
            pattern_signature=pattern_signature_from_forward_model(fwd_model),
            backend_signature=backend_signature_from_forward_model(fwd_model),
            extra={
                "measurement_space": str(
                    getattr(reconstructor, "_measurement_space_type", "real")
                ),
                "difference_mode": str(
                    getattr(reconstructor, "_difference_mode_effective", "")
                ),
                "difference_orientation": str(
                    getattr(reconstructor, "_difference_orientation_effective", "")
                ),
            },
        )
    except ValueError:
        return None


def _persistent_jacobian_lookup_state(
    *,
    hit: bool = False,
    stored: bool = False,
    key: str | None = None,
    reason: str | None = None,
) -> dict[str, object]:
    lookup: dict[str, object] = {
        "hit": bool(hit),
        "stored": bool(stored),
        "key": key,
        "artifact": "persistent_jacobian",
    }
    if reason:
        lookup["reason"] = str(reason)
    return lookup


def _jacobian_calculator_cache_signature(jacobian_calculator) -> dict[str, object]:
    cls = type(jacobian_calculator)
    signature: dict[str, object] = {
        "module": cls.__module__,
        "qualname": cls.__qualname__,
        "sign_convention": str(getattr(jacobian_calculator, "sign_convention", "")),
    }
    for attr in (
        "use_torch",
        "torch_batch_all",
        "torch_dtype",
        "block_tune_mode",
    ):
        if hasattr(jacobian_calculator, attr):
            signature[attr] = str(getattr(jacobian_calculator, attr))
    return signature


def _calculate_iteration_jacobian(
    reconstructor,
    sigma_current,
    *,
    jacobian_method: str,
):
    if _is_operator_jacobian_method(jacobian_method):
        if getattr(reconstructor, "_measurement_space_type", "real") != "real":
            raise RuntimeError(
                "Matrix-free Jacobian method currently supports real measurement space only; "
                "difference measurement projection requires dense Jacobian."
            )
        linearize = getattr(reconstructor.jacobian_calculator, "linearize", None)
        if not callable(linearize):
            raise RuntimeError(
                "jacobian_method='linearized' requires jacobian_calculator.linearize()."
            )
        jacobian = linearize(sigma_current, method="efficient")
        # Guard external cache / stale-linearization reuse.
        if isinstance(jacobian, JacobianLinearization):
            try:
                current_fp = compute_sigma_fingerprint(sigma_current)
            except Exception:
                current_fp = ""
            jacobian.assert_compatible(current_fp)
    else:
        cache_key = _persistent_jacobian_cache_key(
            reconstructor, sigma_current, jacobian_method
        )
        cached = (
            get_process_cached_jacobian(cache_key) if cache_key is not None else None
        )
        if cached is not None:
            jacobian = cached
            reconstructor._last_persistent_jacobian_lookup = (
                _persistent_jacobian_lookup_state(hit=True, key=cache_key)
            )
        else:
            jacobian = reconstructor.jacobian_calculator.calculate(
                sigma_current,
                method=jacobian_method,
            )
            jacobian = _project_measurement_jacobian(
                reconstructor,
                jacobian,
            )
            stored = False
            if cache_key is not None and not _is_matrix_free_jacobian(jacobian):
                put_process_cached_jacobian(cache_key, np.asarray(jacobian))
                stored = True
            reconstructor._last_persistent_jacobian_lookup = (
                _persistent_jacobian_lookup_state(stored=stored, key=cache_key)
            )

    if reconstructor.negate_jacobian:
        jacobian = _scale_jacobian_action(jacobian, -1.0)
    if not _is_matrix_free_jacobian(jacobian):
        _require_finite("measurement_jacobian_np", jacobian, None)
    return jacobian


def _init_sigma_function(
    reconstructor,
    initial_conductivity,
) -> tuple[fem.Function, float | np.ndarray]:
    if initial_conductivity is None:
        initial_conductivity = 1.0
    sigma_current = fem.Function(reconstructor.fwd_model.V_sigma)
    sigma_storage = function_get_array(sigma_current)
    if np.isscalar(initial_conductivity):
        if _is_complex_array_like(initial_conductivity) and not np.iscomplexobj(
            sigma_storage
        ):
            raise RuntimeError(
                "Complex conductivity requires a complex DOLFINx/PETSc scalar build."
            )
        dtype = np.result_type(
            sigma_storage.dtype, np.asarray(initial_conductivity).dtype
        )
        function_set_array(
            sigma_current,
            np.full(reconstructor.n_elements, initial_conductivity, dtype=dtype),
        )
    else:
        values = np.asarray(initial_conductivity).reshape(-1)
        if np.iscomplexobj(values) and not np.iscomplexobj(sigma_storage):
            raise RuntimeError(
                "Complex conductivity requires a complex DOLFINx/PETSc scalar build."
            )
        function_set_array(sigma_current, values)
    return sigma_current, initial_conductivity


def _prepare_prior(
    reconstructor,
    prior_data: np.ndarray | None,
    initial_conductivity: float | np.ndarray,
) -> torch.Tensor:
    if prior_data is not None:
        reconstructor._prior_data = np.asarray(prior_data).reshape(-1)
    elif np.isscalar(initial_conductivity):
        reconstructor._prior_data = np.full(
            reconstructor.n_elements, initial_conductivity
        )
    else:
        reconstructor._prior_data = np.asarray(initial_conductivity).reshape(-1)
    return _to_runtime_tensor(reconstructor, reconstructor._prior_data)


def _best_homog_bounds(
    reconstructor,
    initial_conductivity: float | np.ndarray,
) -> tuple[float, float]:
    if np.isscalar(initial_conductivity):
        center = max(float(initial_conductivity), 1e-6)
    else:
        arr = np.asarray(initial_conductivity, dtype=np.float64).reshape(-1)
        center = max(float(np.mean(arr)), 1e-6)
    clip_values = getattr(reconstructor, "clip_values", None)
    if clip_values is not None:
        lower = max(float(clip_values[0]), 1e-6)
        upper = max(float(clip_values[1]), lower * 1.01)
        return (lower, upper)
    return (max(center * 0.2, 1e-6), max(center * 5.0, center + 1e-6))


def _estimate_best_homogeneous_conductivity(
    reconstructor,
    *,
    measured_vector: np.ndarray,
    initial_conductivity: float | np.ndarray,
) -> dict[str, object]:
    mode = str(getattr(reconstructor, "best_homog_mode", "off")).strip().lower()
    info: dict[str, object] = {
        "mode": mode,
        "applied": False,
        "value": None,
        "objective": None,
    }
    if mode != "optimize":
        info["reason"] = "disabled"
        return info
    if np.iscomplexobj(measured_vector) or _is_complex_array_like(initial_conductivity):
        info["reason"] = "complex_measurement"
        return info
    if getattr(reconstructor, "_measurement_space_type", "real") != "real":
        info["reason"] = "difference_measurement_space"
        return info

    bounds = _best_homog_bounds(reconstructor, initial_conductivity)
    info["bounds"] = [float(bounds[0]), float(bounds[1])]
    calls = {"count": 0}

    def _objective(conductivity_value: float) -> float:
        calls["count"] += 1
        image = EITImage(
            elem_data=np.full(
                reconstructor.n_elements,
                float(conductivity_value),
                dtype=np.float64,
            ),
            fwd_model=reconstructor.fwd_model,
        )
        simulated, _ = reconstructor.fwd_model.fwd_solve(image)
        simulated_vector = _project_simulated_measurements(
            reconstructor, simulated.meas
        )
        residual = np.asarray(simulated_vector, dtype=np.float64) - measured_vector
        return float(np.dot(residual, residual))

    try:
        result = minimize_scalar(
            _objective,
            bounds=bounds,
            method="bounded",
            options={"xatol": 1e-4, "maxiter": 32},
        )
        value = float(result.x)
        objective = float(result.fun)
        info["success"] = bool(result.success)
        info["message"] = str(result.message)
    except Exception as exc:
        info["reason"] = f"optimization_failed:{type(exc).__name__}"
        info["eval_count"] = int(calls["count"])
        return info

    info["eval_count"] = int(calls["count"])
    info["applied"] = True
    info["value"] = value
    info["objective"] = objective
    return info


def _compute_residuals(
    reconstructor,
    simulated_meas: np.ndarray,
    meas_torch: torch.Tensor,
    iteration: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
    float,
    float,
]:
    data_sim_torch = _to_runtime_tensor_cached(
        reconstructor, "simulated_meas", simulated_meas
    )
    residual_torch = data_sim_torch - meas_torch
    if reconstructor._meas_weight_sqrt is not None:
        weighted_residual_torch = residual_torch * reconstructor._meas_weight_sqrt
    else:
        weighted_residual_torch = residual_torch

    residual_norm_weighted = torch.norm(weighted_residual_torch).item()
    residual_norm = torch.norm(residual_torch).item()
    residual_max = torch.max(torch.abs(residual_torch)).item()
    _require_scalar_finite("residual_norm_weighted", residual_norm_weighted, iteration)
    _require_scalar_finite("residual_norm", residual_norm, iteration)
    _require_scalar_finite("residual_max", residual_max, iteration)
    return (
        data_sim_torch,
        residual_torch,
        weighted_residual_torch,
        residual_norm_weighted,
        residual_norm,
        residual_max,
    )


def _compute_objective(
    reconstructor,
    weighted_residual_torch: torch.Tensor,
    de_current: torch.Tensor,
    lambda_eff: float,
    iteration: int,
) -> tuple[float, float, float, torch.Tensor]:
    meas_misfit = 0.5 * _vdot_real_torch(
        weighted_residual_torch,
        weighted_residual_torch,
    )
    if reconstructor.R_torch is not None:
        RtR_de = torch.mv(reconstructor.R_torch, de_current)
        prior_misfit = 0.5 * lambda_eff * _vdot_real_torch(de_current, RtR_de)
    else:
        de_np = de_current.detach().cpu().numpy()
        rde_np = _apply_regularization_np(reconstructor, de_np)
        prior_misfit = 0.5 * lambda_eff * float(np.vdot(de_np, rde_np).real)
        RtR_de = _to_runtime_tensor_cached(reconstructor, "RtR_de_fast", rde_np)
    total_objective = meas_misfit + prior_misfit
    _require_scalar_finite("meas_misfit", meas_misfit, iteration)
    _require_scalar_finite("prior_misfit", prior_misfit, iteration)
    _require_scalar_finite("total_objective", total_objective, iteration)
    return meas_misfit, prior_misfit, total_objective, RtR_de


def _build_linear_system(
    reconstructor,
    JTJ: torch.Tensor,
    JTr: torch.Tensor,
    de_torch: torch.Tensor,
    lambda_eff: float,
    iteration: int,
    RtR_de: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    A = JTJ + lambda_eff * reconstructor.R_torch
    if reconstructor.use_prior_term:
        if RtR_de is None:
            RtR_de = torch.mv(reconstructor.R_torch, de_torch)
        b = -(JTr + lambda_eff * RtR_de)
    else:
        b = -JTr
    _require_finite("A", A, iteration)
    _require_finite("b", b, iteration)
    return A, b


def _solve_linear_system_torch_cg(
    A: torch.Tensor,
    b: torch.Tensor,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_iter: int | None = None,
) -> torch.Tensor:
    if max_iter is None:
        max_iter = max(512, min(int(A.shape[0]) * 2, 4096))
    x = torch.zeros_like(b)
    r = b.clone()
    diag = torch.diagonal(A)
    safe_diag = torch.where(torch.abs(diag) > 1e-18, diag, torch.ones_like(diag))
    z = r / safe_diag
    p = z.clone()
    rz_old = torch.vdot(r, z) if (r.is_complex() or z.is_complex()) else torch.dot(r, z)
    b_norm = float(torch.linalg.vector_norm(b).item())
    tol = max(float(atol), float(rtol) * max(b_norm, 1e-18))
    for _ in range(int(max_iter)):
        Ap = torch.mv(A, p)
        denom = (
            torch.vdot(p, Ap)
            if (p.is_complex() or Ap.is_complex())
            else torch.dot(p, Ap)
        )
        if (
            not bool(torch.isfinite(denom).all().item())
            or float(torch.abs(denom).item()) <= 1e-30
        ):
            break
        alpha = rz_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        if float(torch.linalg.vector_norm(r).item()) <= tol:
            return x
        z = r / safe_diag
        rz_new = (
            torch.vdot(r, z) if (r.is_complex() or z.is_complex()) else torch.dot(r, z)
        )
        if not bool(torch.isfinite(rz_new).all().item()):
            break
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
    raise RuntimeError("torch CG fallback did not converge")


def _solve_linear_system(
    reconstructor,
    A: torch.Tensor,
    b: torch.Tensor,
    JTJ: torch.Tensor,
    iteration: int,
) -> tuple[torch.Tensor, float]:
    try:
        delta_sigma_torch = torch.linalg.solve(A, b)
    except RuntimeError as exc:
        message = str(exc).lower()
        runtime_linalg_unavailable = any(
            token in message
            for token in (
                "libtorch_cuda_linalg",
                "cusolver",
                "undefined symbol",
                "dlopen",
            )
        )
        if runtime_linalg_unavailable and A.device.type == "cuda":
            delta_sigma_torch = _solve_linear_system_torch_cg(
                A,
                b,
                rtol=1e-12,
                atol=1e-14,
                max_iter=max(2048, min(int(A.shape[0]) * 4, 8192)),
            )
        else:
            A_regularized = (
                JTJ + (reconstructor.regularization_param * 10) * reconstructor.R_torch
            )
            _require_finite("A_regularized", A_regularized, iteration)
            try:
                delta_sigma_torch = torch.linalg.solve(A_regularized, b)
            except RuntimeError:
                delta_sigma_torch = _solve_linear_system_torch_cg(
                    A_regularized if A_regularized.device.type == "cuda" else A,
                    b,
                    rtol=1e-10,
                    atol=1e-12,
                )
    _require_finite("delta_sigma_torch", delta_sigma_torch, iteration)
    delta_norm = torch.norm(delta_sigma_torch).item()
    _require_scalar_finite("delta_norm", delta_norm, iteration)
    return delta_sigma_torch, delta_norm


def _maybe_rollback(
    reconstructor,
    sigma_current: fem.Function,
    sigma_old_values: np.ndarray,
    residual_norm: float,
    prev_residual: float | None,
    residual_history: list[float],
    sigma_change_history: list[float],
    consecutive_rollbacks: int,
    max_consecutive_rollbacks: int,
) -> tuple[bool, bool, int]:
    if prev_residual is None or residual_norm <= prev_residual:
        return False, False, consecutive_rollbacks

    consecutive_rollbacks += 1
    if reconstructor.verbose:
        print(
            f"[WARN] residual increased ({residual_norm:.3e} > {prev_residual:.3e}), "
            f"rolling back step ({consecutive_rollbacks}/{max_consecutive_rollbacks})"
        )
    function_set_array(sigma_current, sigma_old_values)
    residual_history[-1] = prev_residual
    sigma_change_history[-1] = 0.0

    if consecutive_rollbacks >= max_consecutive_rollbacks:
        if reconstructor.verbose:
            print(
                f"[STOP] {max_consecutive_rollbacks} consecutive rollbacks, terminating early"
            )
        return True, True, consecutive_rollbacks

    return True, False, consecutive_rollbacks


def run_reconstruction(
    reconstructor,
    measured_data: object | np.ndarray,
    initial_conductivity: float = 1.0,
    jacobian_method: str = "efficient",
    prior_data: np.ndarray | None = None,
    record_conductivity_history: bool = False,
    conductivity_history_stride: int = 1,
) -> SolverOutput:
    """Execute Gauss-Newton iterations and return typed solver output."""
    reconstructor._meas_weight_sqrt = None
    reconstructor._baseline_measurement = None

    meas_vector = _extract_measured_vector(measured_data)
    _configure_measurement_space(reconstructor, measured_data)
    reconstructor._difference_step_size_info = {
        "mode": str(getattr(reconstructor, "difference_step_size_mode", "off")),
        "applied": False,
        "value": 1.0,
    }
    reconstructor._best_homog_info = {
        "mode": str(getattr(reconstructor, "best_homog_mode", "off")),
        "applied": False,
        "value": None,
    }

    if len(meas_vector) != reconstructor.n_measurements:
        raise ValueError(
            f"Measurement data length mismatch: {len(meas_vector)} vs {reconstructor.n_measurements}"
        )
    _require_finite("measured_data", meas_vector)

    runtime_dtype = _torch_dtype_for_values(
        meas_vector,
        initial_conductivity,
        prior_data,
        getattr(reconstructor, "jacobian_background_conductivity", None),
    )
    if getattr(reconstructor, "_torch_dtype", None) != runtime_dtype:
        reconstructor._torch_dtype = runtime_dtype
        reconstructor.R_torch = None

    meas_torch = _to_runtime_tensor(reconstructor, meas_vector)
    reconstructor._measured_vector = meas_vector.copy()
    reconstructor.ensure_regularization_ready()

    meas_norm = torch.norm(meas_torch).item()
    meas_max = torch.max(torch.abs(meas_torch)).item()
    _require_scalar_finite("meas_norm", meas_norm)
    _require_scalar_finite("meas_max", meas_max)
    meas_weighted_norm = None

    if prior_data is None:
        best_homog_info = _estimate_best_homogeneous_conductivity(
            reconstructor,
            measured_vector=meas_vector,
            initial_conductivity=initial_conductivity,
        )
        reconstructor._best_homog_info = best_homog_info
        if best_homog_info.get("applied") and best_homog_info.get("value") is not None:
            initial_conductivity = float(best_homog_info["value"])

    sigma_current, initial_conductivity = _init_sigma_function(
        reconstructor, initial_conductivity
    )
    reconstructor._ensure_measurement_weights(sigma_current)

    if reconstructor._meas_weight_sqrt is not None:
        meas_weighted_norm = torch.norm(
            meas_torch * reconstructor._meas_weight_sqrt
        ).item()
        _require_scalar_finite("meas_weighted_norm", meas_weighted_norm)

    prior_torch = _prepare_prior(reconstructor, prior_data, initial_conductivity)

    residual_history = []
    sigma_change_history = []
    iteration_logs: list[_IterationLog] = []
    conductivity_history = []
    history_stride = max(1, int(conductivity_history_stride))
    if record_conductivity_history:
        conductivity_history.append(function_get_array(sigma_current).copy())

    consecutive_rollbacks = 0
    max_consecutive_rollbacks = 5

    if reconstructor.verbose:
        print(
            f"[INFO] lambda={reconstructor.regularization_param:.3e}, "
            f"hp={reconstructor.hyperparameter:.3e}"
        )
        print("\nStarting modular Gauss-Newton reconstruction...")
        print(f"Using Jacobian method: {jacobian_method}")

    prev_residual = None
    relative_change = float("inf")
    reconstructor._runtime_tensor_cache = {}
    reconstructor._force_jacobian_refresh = False
    reconstructor._last_persistent_jacobian_lookup = _persistent_jacobian_lookup_state(
        reason="not_attempted"
    )
    use_operator_jacobian = _is_operator_jacobian_method(jacobian_method)
    if use_operator_jacobian and reconstructor.solver_mode != "fast":
        raise RuntimeError("Operator Jacobian methods require solver_mode='fast'.")
    prev_jacobian = None
    prev_jacobian_iter = -1
    fast_fallback_reason: str | None = None
    resolved_preconditioner: str | None = None
    fast_solver_path: str | None = None
    fast_linear_path_selected: str | None = None
    fast_linear_path_reason: str | None = None
    jacobian_representation: str | None = None
    jacobian_shape: list[int] | None = None
    dense_jacobian_materialized: bool | None = None
    degrade_stage_counts: dict[str, int] = {}
    effective_solver_path_counts: dict[str, int] = {}
    inexact_eta_history: list[float] = []
    rom_rank_effective: int = 0
    lowrank_rank_effective: int = 0
    rom_enabled_effective = False
    timing_totals = {
        "forward": 0.0,
        "jacobian": 0.0,
        "linear_solve": 0.0,
        "line_search": 0.0,
    }
    if use_operator_jacobian:
        startup_jacobian_np = None
        startup_cache_lookup = {
            "hit": False,
            "layer": "disabled",
            "artifact": "absolute_startup_jacobian",
            "reason": "operator_jacobian",
        }
        reconstructor._last_persistent_jacobian_lookup = (
            _persistent_jacobian_lookup_state(reason="operator_jacobian")
        )
    else:
        startup_jacobian_np, startup_cache_lookup = _startup_cache_lookup(
            reconstructor,
            sigma_current,
            jacobian_method,
        )
        if startup_jacobian_np is not None:
            startup_jacobian_np = _project_measurement_jacobian(
                reconstructor, startup_jacobian_np
            )
    final_simulated_measurement: np.ndarray | None = None

    lambda_eff = reconstructor.regularization_param
    with reconstructor._progress(total=reconstructor.max_iterations) as pbar:
        for iteration in range(reconstructor.max_iterations):
            sigma_array = function_get_array(sigma_current)
            _require_finite("sigma_array", sigma_array, iteration)
            img_current = EITImage(
                elem_data=sigma_array, fwd_model=reconstructor.fwd_model
            )
            forward_start = perf_counter()
            data_simulated, _ = reconstructor.fwd_model.fwd_solve(img_current)
            timing_totals["forward"] += perf_counter() - forward_start
            _require_finite("data_simulated.meas", data_simulated.meas, iteration)
            simulated_measurement = _project_simulated_measurements(
                reconstructor,
                data_simulated.meas,
            )
            _require_finite("simulated_measurement", simulated_measurement, iteration)

            lambda_eff = reconstructor.regularization_param
            (
                data_sim_torch,
                residual_torch,
                weighted_residual_torch,
                residual_norm_weighted,
                residual_norm,
                residual_max,
            ) = _compute_residuals(
                reconstructor,
                simulated_measurement,
                meas_torch,
                iteration,
            )

            sigma_vec_torch = _to_runtime_tensor_cached(
                reconstructor,
                "sigma_vector",
                sigma_array,
            )
            de_current = sigma_vec_torch - prior_torch
            meas_misfit, prior_misfit, total_objective, RtR_de = _compute_objective(
                reconstructor,
                weighted_residual_torch,
                de_current,
                lambda_eff,
                iteration,
            )

            residual_history.append(residual_norm)
            res_drop = None if prev_residual is None else prev_residual - residual_norm
            reconstructor._outer_prev_residual = prev_residual

            jacobian_reused = False
            reuse_tol = float(reconstructor.jacobian_reuse_tol)
            large_fast_problem = (
                reconstructor.solver_mode == "fast" and reconstructor.n_elements >= 5000
            )
            reuse_change_ok = float(relative_change) <= reuse_tol
            if large_fast_problem:
                # For large 3D runs, update cadence + residual trend already guard stability.
                # Skipping per-iteration Jacobian rebuilds is the dominant speed lever.
                reuse_change_ok = True
            force_refresh = bool(
                getattr(reconstructor, "_force_jacobian_refresh", False)
            )
            if force_refresh:
                reconstructor._force_jacobian_refresh = False
            can_reuse = (
                reconstructor.solver_mode == "fast"
                and prev_jacobian is not None
                and (iteration - prev_jacobian_iter)
                < max(1, reconstructor.jacobian_update_every)
                and reuse_change_ok
                and not force_refresh
                and (
                    prev_residual is None
                    or residual_norm <= (float(prev_residual) * 1.05)
                )
            )
            if can_reuse:
                measurement_jacobian_np = prev_jacobian
                jacobian_reused = True
            elif iteration == 0 and startup_jacobian_np is not None:
                # Defence in depth: operator-mode sets startup_jacobian_np=None above,
                # but if an external cache layer ever returns an operator here we
                # would otherwise try np.asarray(JacobianLinearization, dtype=float)
                # which silently yields a 0-d object array. Prefer a clean rebuild.
                if _is_matrix_free_jacobian(startup_jacobian_np):
                    jacobian_start = perf_counter()
                    measurement_jacobian_np = _calculate_iteration_jacobian(
                        reconstructor,
                        sigma_current,
                        jacobian_method=jacobian_method,
                    )
                    timing_totals["jacobian"] += perf_counter() - jacobian_start
                else:
                    measurement_jacobian_np = np.asarray(startup_jacobian_np)
                prev_jacobian = measurement_jacobian_np
                prev_jacobian_iter = iteration
            else:
                jacobian_start = perf_counter()
                measurement_jacobian_np = _calculate_iteration_jacobian(
                    reconstructor,
                    sigma_current,
                    jacobian_method=jacobian_method,
                )
                timing_totals["jacobian"] += perf_counter() - jacobian_start
                prev_jacobian = measurement_jacobian_np
                prev_jacobian_iter = iteration

            if reconstructor._meas_weight_sqrt is not None:
                meas_weight_np = reconstructor._meas_weight_sqrt.detach().cpu().numpy()
                weighted_residual_np = (
                    residual_torch.detach().cpu().numpy() * meas_weight_np
                )
            else:
                meas_weight_np = None
                weighted_residual_np = residual_torch.detach().cpu().numpy()

            def _solve_strict_path(
                *,
                measurement_jacobian_np=measurement_jacobian_np,
                weighted_residual_torch=weighted_residual_torch,
                de_current=de_current,
                lambda_eff=lambda_eff,
                iteration=iteration,
                RtR_de=RtR_de,
            ) -> tuple[torch.Tensor, float, float]:
                J_torch_local = _to_runtime_tensor_cached(
                    reconstructor,
                    "measurement_jacobian",
                    measurement_jacobian_np,
                )
                if reconstructor._meas_weight_sqrt is not None:
                    J_weighted_local = (
                        J_torch_local * reconstructor._meas_weight_sqrt.unsqueeze(1)
                    )
                else:
                    J_weighted_local = J_torch_local

                J_h_local = _hermitian_transpose(J_weighted_local)
                JTJ_local = torch.mm(J_h_local, J_weighted_local)
                JTr_local = torch.mv(J_h_local, weighted_residual_torch)

                de_torch_local = de_current
                A_local, b_local = _build_linear_system(
                    reconstructor,
                    JTJ_local,
                    JTr_local,
                    de_torch_local,
                    lambda_eff,
                    iteration,
                    RtR_de=RtR_de,
                )
                jtr_norm_local = torch.norm(JTr_local).item()
                delta_sigma_torch_local, delta_norm_local = _solve_linear_system(
                    reconstructor,
                    A_local,
                    b_local,
                    JTJ_local,
                    iteration,
                )
                return delta_sigma_torch_local, delta_norm_local, float(jtr_norm_local)

            if reconstructor.solver_mode == "fast":
                if _is_matrix_free_jacobian(measurement_jacobian_np):
                    J_weighted_np = measurement_jacobian_np
                    measurement_weight_for_solver = meas_weight_np
                else:
                    J_weighted_np = measurement_jacobian_np
                    measurement_weight_for_solver = meas_weight_np
                de_current_np = de_current.detach().cpu().numpy()
                linear_start = perf_counter()
                try:
                    delta_sigma_np, delta_norm, jtr_norm = _solve_linear_system_fast(
                        reconstructor,
                        J_weighted_np=J_weighted_np,
                        weighted_residual_np=weighted_residual_np,
                        de_current_np=de_current_np,
                        lambda_eff=lambda_eff,
                        iteration=iteration,
                        measurement_weight_np=measurement_weight_for_solver,
                    )
                    delta_sigma_torch = _to_runtime_tensor_cached(
                        reconstructor,
                        "delta_sigma_fast",
                        delta_sigma_np,
                    )
                    fast_meta = getattr(reconstructor, "_last_fast_linear_meta", {})
                    if isinstance(fast_meta, dict):
                        path = fast_meta.get("path")
                        if isinstance(path, str):
                            fast_solver_path = path
                        resolved = fast_meta.get("resolved_preconditioner")
                        if isinstance(resolved, str):
                            resolved_preconditioner = resolved
                        reason = fast_meta.get("fallback_reason")
                        if isinstance(reason, str) and reason:
                            fast_fallback_reason = reason
                        selected_path = fast_meta.get("fast_linear_path_selected")
                        if isinstance(selected_path, str):
                            fast_linear_path_selected = selected_path
                        selected_reason = fast_meta.get("fast_linear_path_reason")
                        if isinstance(selected_reason, str):
                            fast_linear_path_reason = selected_reason
                        representation = fast_meta.get("jacobian_representation")
                        if isinstance(representation, str):
                            jacobian_representation = representation
                        shape = fast_meta.get("jacobian_shape")
                        if isinstance(shape, list):
                            jacobian_shape = [int(value) for value in shape]
                        if "dense_jacobian_materialized" in fast_meta:
                            dense_jacobian_materialized = bool(
                                fast_meta.get("dense_jacobian_materialized")
                            )
                        degrade_stage = fast_meta.get("degrade_stage")
                        if isinstance(degrade_stage, str) and degrade_stage:
                            degrade_stage_counts[degrade_stage] = (
                                degrade_stage_counts.get(degrade_stage, 0) + 1
                            )
                        effective_path = fast_meta.get("effective_solver_path")
                        if isinstance(effective_path, str) and effective_path:
                            effective_solver_path_counts[effective_path] = (
                                effective_solver_path_counts.get(effective_path, 0) + 1
                            )
                        else:
                            if isinstance(fast_solver_path, str) and fast_solver_path:
                                effective_solver_path_counts[fast_solver_path] = (
                                    effective_solver_path_counts.get(
                                        fast_solver_path, 0
                                    )
                                    + 1
                                )
                        eta_value = fast_meta.get("inexact_eta")
                        if isinstance(eta_value, (int, float)):
                            inexact_eta_history.append(float(eta_value))
                        rank_effective = fast_meta.get("rom_rank_effective")
                        if isinstance(rank_effective, int):
                            rom_rank_effective = max(
                                rom_rank_effective, int(rank_effective)
                            )
                        lowrank_effective = fast_meta.get("lowrank_rank_effective")
                        if isinstance(lowrank_effective, int):
                            lowrank_rank_effective = max(
                                lowrank_rank_effective, int(lowrank_effective)
                            )
                        if bool(fast_meta.get("rom_enabled_effective", False)):
                            rom_enabled_effective = True
                except Exception as exc:
                    fast_fallback_reason = (
                        f"fast_linear_solver_failed:{type(exc).__name__}"
                    )
                    fast_solver_path = "strict-fallback"
                    fast_linear_path_selected = "strict"
                    fast_linear_path_reason = "fast_solver_exception"
                    if _is_matrix_free_jacobian(measurement_jacobian_np):
                        raise RuntimeError(
                            "fast_linear_solver_failed_for_operator_jacobian"
                        ) from exc
                    delta_sigma_torch, delta_norm, jtr_norm = _solve_strict_path()
                timing_totals["linear_solve"] += perf_counter() - linear_start
            else:
                linear_start = perf_counter()
                delta_sigma_torch, delta_norm, jtr_norm = _solve_strict_path()
                timing_totals["linear_solve"] += perf_counter() - linear_start

            pred_norm = torch.norm(data_sim_torch).item()
            pred_max = torch.max(torch.abs(data_sim_torch)).item()
            rel_residual = residual_norm / (meas_norm + 1e-12)
            rel_residual_weighted = (
                residual_norm_weighted / (meas_weighted_norm + 1e-12)
                if meas_weighted_norm
                else None
            )
            _require_scalar_finite("pred_norm", pred_norm, iteration)
            _require_scalar_finite("pred_max", pred_max, iteration)
            _require_scalar_finite("jtr_norm", jtr_norm, iteration)
            _require_scalar_finite("rel_residual", rel_residual, iteration)
            if rel_residual_weighted is not None:
                _require_scalar_finite(
                    "rel_residual_weighted", rel_residual_weighted, iteration
                )

            if reconstructor.verbose and jacobian_reused:
                print(f"[INFO] iteration={iteration}: reused Jacobian (fast mode)")

            line_search_start = perf_counter()
            optimal_step_size = _select_step_size(
                reconstructor,
                iteration,
                sigma_current,
                delta_sigma_torch,
                meas_torch,
                residual_norm_weighted,
                prior_torch,
                lambda_eff,
            )
            timing_totals["line_search"] += perf_counter() - line_search_start

            needs_snapshot = (
                prev_residual is not None or reconstructor.clip_values is not None
            )
            sigma_old_values = sigma_array.copy() if needs_snapshot else None
            delta_sigma_np = delta_sigma_torch.detach().cpu().numpy()
            sigma_array[:] += optimal_step_size * delta_sigma_np
            _require_finite("sigma_array_updated", sigma_array, iteration)

            if reconstructor.clip_values is not None:
                function_set_array(
                    sigma_current,
                    _clip_real_sigma(sigma_array, reconstructor.clip_values),
                )
                sigma_array = function_get_array(sigma_current)
            _require_finite("sigma_array_clipped", sigma_array, iteration)

            sigma_new_norm = float(np.linalg.norm(sigma_array))
            if sigma_old_values is None:
                sigma_change = abs(optimal_step_size) * delta_norm
            else:
                sigma_change = float(np.linalg.norm(sigma_array - sigma_old_values))
            relative_change = sigma_change / (sigma_new_norm + 1e-12)
            _require_scalar_finite("sigma_change", sigma_change, iteration)
            _require_scalar_finite("relative_change", relative_change, iteration)

            rolled_back, should_stop, consecutive_rollbacks = _maybe_rollback(
                reconstructor,
                sigma_current,
                sigma_old_values if sigma_old_values is not None else sigma_array,
                residual_norm,
                prev_residual,
                residual_history,
                sigma_change_history,
                consecutive_rollbacks,
                max_consecutive_rollbacks,
            )
            if rolled_back:
                if should_stop:
                    break
                continue

            consecutive_rollbacks = 0
            sigma_change_history.append(relative_change)
            if record_conductivity_history and (iteration + 1) % history_stride == 0:
                conductivity_history.append(function_get_array(sigma_current).copy())
            final_simulated_measurement = simulated_measurement.copy()

            _record_iteration_log(
                iteration_logs,
                iteration=iteration,
                residual_norm=residual_norm,
                residual_norm_weighted=residual_norm_weighted,
                rel_residual=rel_residual,
                rel_residual_weighted=rel_residual_weighted,
                residual_max=residual_max,
                meas_norm=meas_norm,
                pred_norm=pred_norm,
                meas_max=meas_max,
                pred_max=pred_max,
                jtr_norm=jtr_norm,
                delta_norm=delta_norm,
                optimal_step_size=optimal_step_size,
                lambda_eff=lambda_eff,
                relative_change=relative_change,
                res_drop=res_drop,
                meas_misfit=meas_misfit,
                prior_misfit=prior_misfit,
                total_objective=total_objective,
            )
            prev_residual = residual_norm

            if (
                relative_change < reconstructor.convergence_tol
                and iteration + 1 >= reconstructor.min_iterations
            ):
                if reconstructor.verbose:
                    print(
                        f"\nConverged! Iteration {iteration}, relative change: {relative_change:.2e}"
                    )
                break

            if reconstructor.verbose:
                pbar.set_postfix_str(
                    f"residual={residual_norm:.2e}, step={optimal_step_size:.3f}, Δσ={relative_change:.2e}"
                )
                pbar.update(1)

    cache_stats = {}
    if getattr(reconstructor, "cache_manager", None) is not None:
        cache_stats = reconstructor.cache_manager.stats()
    sigma_final_array = function_get_array(sigma_current).copy()
    sigma_final_array, difference_step_size_info = _apply_difference_step_size(
        reconstructor,
        sigma_final=sigma_final_array,
        measured_vector=meas_vector,
    )
    reconstructor._difference_step_size_info = difference_step_size_info
    if reconstructor.clip_values is not None:
        sigma_final_array = _clip_real_sigma(
            sigma_final_array,
            reconstructor.clip_values,
        )
    function_set_array(sigma_current, sigma_final_array)
    final_img = EITImage(
        elem_data=sigma_final_array,
        fwd_model=reconstructor.fwd_model,
    )
    final_data_simulated, _ = reconstructor.fwd_model.fwd_solve(final_img)
    final_simulated_measurement = _project_simulated_measurements(
        reconstructor,
        final_data_simulated.meas,
    )
    jacobian_block_tune_info = (
        reconstructor.jacobian_calculator.block_tuning_info()
        if hasattr(reconstructor.jacobian_calculator, "block_tuning_info")
        else {}
    )
    petsc_backend_info = (
        getattr(reconstructor.fwd_model, "_petsc_backend_info", {}) or {}
    )
    inverse_device_requested = str(getattr(reconstructor, "device_requested", "cpu"))
    inverse_device_effective = str(
        getattr(
            reconstructor,
            "device_effective",
            str(getattr(reconstructor, "device", "cpu")),
        )
    )
    petsc_effective = str(petsc_backend_info.get("petsc_device_effective", "cpu"))
    jacobian_block_backend = (
        jacobian_block_tune_info.get("jacobian_block_backend")
        if isinstance(jacobian_block_tune_info, dict)
        else None
    )
    if (
        petsc_effective == "cuda"
        and inverse_device_effective == "cuda"
        and jacobian_block_backend == "torch-cuda"
    ):
        execution_profile = "cuda"
    elif (
        petsc_effective == "cuda"
        or inverse_device_effective == "cuda"
        or jacobian_block_backend == "torch-cuda"
    ):
        execution_profile = "mixed"
    else:
        execution_profile = "cpu"

    if jacobian_representation is None and prev_jacobian is not None:
        jacobian_representation = "dense"
    if (
        jacobian_shape is None
        and prev_jacobian is not None
        and hasattr(prev_jacobian, "shape")
    ):
        jacobian_shape = [int(prev_jacobian.shape[0]), int(prev_jacobian.shape[1])]
    if dense_jacobian_materialized is None and jacobian_representation == "dense":
        dense_jacobian_materialized = True

    backend_info = {
        "linear_backend": getattr(reconstructor.fwd_model, "linear_backend", "unknown"),
        "performance_mode": getattr(reconstructor, "performance_mode", "aggressive"),
        "solver_mode": getattr(reconstructor, "solver_mode", "strict"),
        "linear_solver": getattr(reconstructor, "linear_solver", "auto"),
        "line_search_mode": getattr(reconstructor, "line_search_mode", "full"),
        "preconditioner": getattr(reconstructor, "preconditioner", "auto"),
        "resolved_preconditioner": resolved_preconditioner,
        "rom_mode": getattr(reconstructor, "rom_mode", "off"),
        "rom_snapshot_source": getattr(reconstructor, "rom_snapshot_source", "hybrid"),
        "inexact_mode": getattr(reconstructor, "inexact_mode", "off"),
        "inexact_forcing": getattr(
            reconstructor, "inexact_forcing", "eisenstat-walker"
        ),
        "lowrank_mode": getattr(reconstructor, "lowrank_mode", "off"),
        "lowrank_method": getattr(reconstructor, "lowrank_method", "tsvd"),
        "fast_solver_path": fast_solver_path,
        "fast_linear_path_selected": fast_linear_path_selected,
        "fast_linear_path_reason": fast_linear_path_reason,
        "fallback_reason": fast_fallback_reason,
        "fast_fallback_reason": fast_fallback_reason,
        "jacobian_representation": jacobian_representation,
        "jacobian_shape": jacobian_shape,
        "dense_jacobian_materialized": dense_jacobian_materialized,
        "rom_enabled_effective": bool(rom_enabled_effective),
        "rom_rank_effective": int(rom_rank_effective),
        "lowrank_rank_effective": int(lowrank_rank_effective),
        "inexact_eta_history": inexact_eta_history[-32:],
        "degrade_stage_counts": degrade_stage_counts,
        "effective_solver_path_counts": effective_solver_path_counts,
        "startup_cache_lookup": startup_cache_lookup,
        "forward_cache_lookup": getattr(
            reconstructor.fwd_model, "_last_cache_lookup", {}
        ),
        "jacobian_cache_lookup": getattr(
            reconstructor.jacobian_calculator, "_last_cache_lookup", {}
        ),
        "persistent_jacobian_cache_lookup": getattr(
            reconstructor, "_last_persistent_jacobian_lookup", {}
        ),
        "petsc_device_requested": petsc_backend_info.get(
            "petsc_device_requested", "auto"
        ),
        "petsc_device_effective": petsc_effective,
        "petsc_mat_type": petsc_backend_info.get("petsc_mat_type"),
        "petsc_vec_type": petsc_backend_info.get("petsc_vec_type"),
        "forward_mat_solve_effective": petsc_backend_info.get(
            "forward_mat_solve_effective"
        ),
        "gpu_fallback_reason": petsc_backend_info.get("gpu_fallback_reason"),
        "forward_factor_backend": petsc_backend_info.get("forward_factor_backend"),
        "inverse_device_requested": inverse_device_requested,
        "inverse_device_effective": inverse_device_effective,
        "inverse_device_fallback_reason": getattr(
            reconstructor, "device_fallback_reason", None
        ),
        "execution_profile": execution_profile,
        "jacobian_backend_requested": (
            jacobian_block_tune_info.get("jacobian_backend_requested")
            if isinstance(jacobian_block_tune_info, dict)
            else None
        ),
        "jacobian_backend_effective": (
            jacobian_block_tune_info.get("jacobian_backend_effective")
            if isinstance(jacobian_block_tune_info, dict)
            else None
        ),
        "jacobian_block_backend": jacobian_block_backend,
        "jacobian_transfer_estimate": (
            jacobian_block_tune_info.get("jacobian_transfer_estimate")
            if isinstance(jacobian_block_tune_info, dict)
            else None
        ),
        "jacobian_cuda_threshold_hit": (
            jacobian_block_tune_info.get("jacobian_cuda_threshold_hit")
            if isinstance(jacobian_block_tune_info, dict)
            else None
        ),
        "jacobian_block_tune": jacobian_block_tune_info,
        "jacobian_assembly_elapsed_only": (
            float(jacobian_block_tune_info.get("assembly_elapsed_only", 0.0))
            if isinstance(jacobian_block_tune_info, dict)
            else 0.0
        ),
        "forward_static_setup_lookup": (
            dict(
                reconstructor.fwd_model.get_backend_diagnostics().get(
                    "static_setup_lookup", {}
                )
            )
            if hasattr(reconstructor.fwd_model, "get_backend_diagnostics")
            else {}
        ),
        "timing_totals": timing_totals,
    }

    results = SolverOutput(
        conductivity=sigma_current,
        residual_history=residual_history,
        sigma_change_history=sigma_change_history,
        iterations=len(residual_history),
        converged=relative_change < reconstructor.convergence_tol,
        final_residual=residual_history[-1],
        final_relative_change=relative_change,
        jacobian_method=jacobian_method,
        regularization_type=type(reconstructor.regularization).__name__,
        iteration_logs=[item.to_payload() for item in iteration_logs],
        conductivity_history=(
            conductivity_history if record_conductivity_history else None
        ),
        baseline_measurement=(
            reconstructor._baseline_measurement.copy()
            if reconstructor._baseline_measurement is not None
            else None
        ),
        measurement_weight=(
            reconstructor._meas_weight_sqrt.detach().cpu().numpy() ** 2
            if reconstructor._meas_weight_sqrt is not None
            else None
        ),
        simulated_measurement=final_simulated_measurement,
        diagnostics={
            "cache_hits": cache_stats.get("total_hits", 0),
            "cache_misses": cache_stats.get("total_misses", 0),
            "cache_stats": cache_stats,
            "backend_info": backend_info,
            "timing": timing_totals,
            "hyperparameter": float(reconstructor.hyperparameter),
            "lambda_eff": float(reconstructor.regularization_param),
            "difference_step_size": reconstructor._difference_step_size_info,
            "best_homog": reconstructor._best_homog_info,
            "preset_name": str(getattr(reconstructor, "active_preset_name", "")),
            "jacobian_background_conductivity": _diagnostic_real_or_complex_scalar(
                getattr(reconstructor, "jacobian_background_conductivity", 1.0)
            ),
            "measurement_space": {
                "type": getattr(reconstructor, "_measurement_space_type", "real"),
                "difference_mode": getattr(
                    reconstructor,
                    "_difference_mode_effective",
                    reconstructor.difference_mode,
                ),
                "difference_orientation": getattr(
                    reconstructor,
                    "_difference_orientation_effective",
                    reconstructor.difference_orientation,
                ),
            },
        },
    )

    if reconstructor.verbose:
        print("\nReconstruction complete:")
        print(f"  Iterations: {results.iterations}")
        print(f"  Final residual: {results.final_residual:.2e}")
        print(f"  Jacobian method: {jacobian_method}")
        print(f"  Regularization type: {results.regularization_type}")

    return results
