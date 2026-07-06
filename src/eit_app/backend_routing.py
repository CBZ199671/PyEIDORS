"""Runtime-profile routing for GUI backend workers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np

from eit_app.controllers.forward_solver_controller import ForwardSolverRequest
from eit_app.controllers.reconstruction_controller import ReconstructionRequest
from eit_app.models.forward_model_config import ForwardModelConfig
from pyeidors.utils.numeric_ops import has_nonzero_imaginary


@dataclass(frozen=True)
class BackendRoute:
    """Resolved GUI backend execution route."""

    profile: str
    reason: str
    external: bool


def _has_nonzero_imag(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return any(_has_nonzero_imag(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_nonzero_imag(item) for item in value)
    try:
        arr = np.asarray(value)
    except Exception:
        try:
            return abs(complex(value).imag) > 1.0e-12
        except Exception:
            return False
    if arr.size == 0 or not np.iscomplexobj(arr):
        return False
    return has_nonzero_imaginary(arr, tol=1.0e-12)


def forward_request_requires_complex(request: ForwardSolverRequest) -> bool:
    cfg = ForwardModelConfig.from_mapping(
        request.forward_model_config
        or {
            "mesh_dimension": request.mesh_dimension,
            "mesh_refinement": request.mesh_refinement,
            "n_elec": request.n_electrodes,
            "background_conductivity": request.background_conductivity,
            "noise_level": request.noise_level,
        }
    )
    return bool(
        _has_nonzero_imag(cfg.background_conductivity)
        or _has_nonzero_imag(cfg.contact_impedance)
        or _has_nonzero_imag(cfg.custom_stim_matrix)
        or _has_nonzero_imag(cfg.custom_meas_matrices)
        or any(_has_nonzero_imag(spec.conductivity) for spec in request.inhomogeneities)
    )


def _target_precision() -> str:
    raw = os.getenv("EIT_APP_GUI_PRECISION", "complex64").strip().lower()
    return "complex128" if raw in {"complex128", "128", "double"} else "complex64"


def _target_uses_gpu(cfg: ForwardModelConfig) -> bool:
    gui_profile = os.getenv("EIT_APP_GUI_PROFILE", "").strip().lower()
    acceleration = str(cfg.acceleration_profile or "").strip().lower()
    petsc_device = str(cfg.petsc_device or "").strip().lower()
    device = str(cfg.device or "").strip().lower()
    backend = str(cfg.forward_backend or "").strip().lower()
    return bool(
        int(cfg.mesh_dimension) == 3
        and (
            gui_profile == "gpu"
            or acceleration in {"gpu", "gpu3d", "gpu3d_fused"}
            or petsc_device == "cuda"
            or device == "cuda"
            or backend == "cuda_structured"
        )
    )


def _metadata_uses_gpu(metadata: dict[str, Any], *, mesh_dimension: int) -> bool:
    gui_profile = os.getenv("EIT_APP_GUI_PROFILE", "").strip().lower()
    acceleration = str(metadata.get("acceleration_profile", "")).strip().lower()
    petsc_device = str(metadata.get("petsc_device", "")).strip().lower()
    device = str(metadata.get("device", "")).strip().lower()
    forward_backend = str(metadata.get("forward_backend", "")).strip().lower()
    return bool(
        int(mesh_dimension) == 3
        and (
            gui_profile == "gpu"
            or acceleration in {"gpu", "gpu3d", "gpu3d_fused"}
            or petsc_device == "cuda"
            or device == "cuda"
            or forward_backend == "cuda_structured"
        )
    )


def _current_profile() -> str:
    return os.getenv("EIT_APP_GUI_RUNTIME_PROFILE", "default").strip() or "default"


def _can_use_inprocess_fast_path(
    *,
    target_profile: str,
    mesh_dimension: int,
    wants_gpu: bool,
    complex_required: bool,
) -> tuple[bool, str]:
    current = _current_profile()
    if target_profile == current:
        return True, "target_profile_matches_current_runtime"
    if int(mesh_dimension) < 3 and not wants_gpu:
        if complex_required:
            if current.startswith("complex"):
                return True, "safe_2d_complex_current_runtime_fast_path"
            return False, "complex_2d_requires_complex_runtime"
        return False, "real_2d_prefers_real_profile_isolation"
    return False, "profile_isolation_required"


def _route_for_traits(
    *,
    complex_required: bool,
    wants_gpu: bool,
    mesh_dimension: int,
) -> BackendRoute:
    def _current_cuda_variant_profile(profile: str) -> str:
        current = _current_profile()
        if not current.endswith("-sm61"):
            return profile
        if profile == "cuda-amgx":
            return "cuda-sm61"
        if profile == "complex64-cuda":
            return "complex64-cuda-sm61"
        return profile

    if complex_required:
        precision = _target_precision()
        profile = (
            "complex-cuda"
            if wants_gpu and precision == "complex128"
            else "complex64-cuda"
            if wants_gpu
            else "complex"
            if precision == "complex128"
            else "complex64"
        )
        if wants_gpu:
            profile = _current_cuda_variant_profile(profile)
        reason = "complex_input_requires_complex_petsc_runtime"
    else:
        profile = "cuda-amgx" if wants_gpu else "default"
        if wants_gpu:
            profile = _current_cuda_variant_profile(profile)
        reason = (
            "real_input_uses_real_cuda_petsc_runtime"
            if profile == "cuda-sm61"
            else "real_input_uses_real_amgx_petsc_runtime"
            if wants_gpu
            else "real_input_uses_real_petsc_runtime"
        )

    current = _current_profile()
    fast_path_ok, fast_path_reason = _can_use_inprocess_fast_path(
        target_profile=profile,
        mesh_dimension=mesh_dimension,
        wants_gpu=wants_gpu,
        complex_required=complex_required,
    )
    mode = os.getenv("EIT_APP_BACKEND_WORKER_MODE", "smart").strip().lower()
    if mode in {"inprocess", "inline"}:
        external = False
        reason = f"{reason}:forced_inprocess"
    elif mode in {"process", "external", "worker"}:
        external = True
        reason = f"{reason}:forced_process"
    elif mode in {"hybrid", "profile-mismatch"}:
        external = profile != current
        if not external:
            reason = f"{reason}:{fast_path_reason}"
    else:
        external = not fast_path_ok
        reason = f"{reason}:{fast_path_reason}"
    return BackendRoute(profile=profile, reason=reason, external=external)


def select_forward_backend_route(request: ForwardSolverRequest) -> BackendRoute:
    cfg = ForwardModelConfig.from_mapping(
        request.forward_model_config
        or {
            "mesh_dimension": request.mesh_dimension,
            "mesh_refinement": request.mesh_refinement,
            "n_elec": request.n_electrodes,
            "background_conductivity": request.background_conductivity,
            "noise_level": request.noise_level,
        }
    )
    return _route_for_traits(
        complex_required=forward_request_requires_complex(request),
        wants_gpu=_target_uses_gpu(cfg),
        mesh_dimension=int(cfg.mesh_dimension),
    )


def reconstruction_request_requires_complex(request: ReconstructionRequest) -> bool:
    metadata = dict(request.metadata or {})
    use_part = str(request.use_part or "real").strip().lower()
    mode = " ".join(
        str(metadata.get(key, ""))
        for key in (
            "eit_value_mode",
            "complex_measurement_mode",
            "complex_reconstruction_dispatch",
        )
    ).lower()
    if use_part == "complex" or ("complex" in mode and "split" not in mode):
        return True
    return False


def select_reconstruction_backend_route(
    request: ReconstructionRequest,
) -> BackendRoute:
    metadata = dict(request.metadata or {})
    return _route_for_traits(
        complex_required=reconstruction_request_requires_complex(request),
        wants_gpu=_metadata_uses_gpu(metadata, mesh_dimension=request.mesh_dimension),
        mesh_dimension=int(request.mesh_dimension),
    )
