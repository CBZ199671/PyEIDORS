"""GPU-aware kernels for online reconstruction-matrix application."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy import sparse

from pyeidors.utils.numeric_ops import safe_dot

try:  # pragma: no cover - availability depends on active dev shell
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RMMatmulResult:
    """Batched RM application result plus execution metadata."""

    values: np.ndarray
    metadata: MappingProxyType

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)


@dataclass(frozen=True)
class RMMatmulHandle:
    """Prepared reconstruction matrix for repeated online RM application."""

    matrix: np.ndarray
    matrix_tensor: Any | None
    matrix_shape: tuple[int, int]
    device_requested: str
    device_effective: str
    backend: str
    dtype: str
    fallback_reason: str | None = None
    cache_key: str | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix_shape

    @property
    def metadata(self) -> MappingProxyType:
        return MappingProxyType(
            {
                "backend": self.backend,
                "device_requested": self.device_requested,
                "device_effective": self.device_effective,
                "fallback_reason": self.fallback_reason,
                "rm_shape": self.matrix_shape,
                "rm_dtype": self.dtype,
                "rm_persistent": True,
                "rm_matrix_resident": "device" if self.matrix_tensor is not None else "cpu",
                "rm_cache_key": self.cache_key,
            }
        )

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.matrix, dtype=dtype)


def prepare_rm_matmul(
    rm: Any,
    *,
    device: str = "auto",
    dtype: str | np.dtype[Any] = "float64",
    cache_key: str | None = None,
) -> RMMatmulHandle:
    """Prepare ``RM`` once for repeated online application.

    CUDA preparation copies only the reconstruction matrix to the device.
    Per-frame calls still copy the incoming ``ΔV`` batch and final image back
    to NumPy because the public API returns NumPy arrays.
    """

    requested = _normalize_device(device)
    np_dtype, dtype_name = _normalize_dtype(dtype)
    if isinstance(rm, RMMatmulHandle):
        if _handle_matches(rm, requested=requested, dtype_name=dtype_name):
            return rm
        matrix = _as_rm_matrix(rm.matrix, dtype=np_dtype)
    else:
        matrix = _as_rm_matrix(rm, dtype=np_dtype)
    effective, fallback_reason = _resolve_effective_device(requested)
    matrix_tensor = None
    backend = "numpy"
    if effective == "cuda":
        matrix_tensor = _torch_as_tensor(matrix, device="cuda", dtype=np_dtype)
        backend = "torch"
    return RMMatmulHandle(
        matrix=matrix,
        matrix_tensor=matrix_tensor,
        matrix_shape=tuple(int(v) for v in matrix.shape),
        device_requested=requested,
        device_effective=effective,
        backend=backend,
        dtype=dtype_name,
        fallback_reason=fallback_reason,
        cache_key=cache_key,
    )


def rm_matmul(
    rm: Any,
    delta_v: Any,
    *,
    device: str = "auto",
    dtype: str | np.dtype[Any] = "float64",
    return_metadata: bool = False,
) -> np.ndarray | RMMatmulResult:
    """Apply ``RM @ ΔV`` for one frame or a frame batch.

    ``delta_v`` may be shape ``(n_meas,)`` or ``(n_frames, n_meas)``.
    Batched output is shape ``(n_frames, n_param)``. ``device="auto"``
    chooses CUDA when Torch CUDA is available, otherwise NumPy CPU.
    """

    requested = _normalize_device(device)
    np_dtype, dtype_name = _normalize_dtype(dtype)
    reused_handle = isinstance(rm, RMMatmulHandle) and _handle_matches(
        rm,
        requested=requested,
        dtype_name=dtype_name,
    )
    handle = (
        rm
        if reused_handle
        else prepare_rm_matmul(rm, device=requested, dtype=np_dtype)
    )
    batch, was_vector = _as_delta_batch(
        delta_v,
        n_measurements=handle.shape[1],
        dtype=np_dtype,
    )

    if handle.device_effective == "cuda":
        values = _torch_rm_matmul(handle, batch, device="cuda", dtype=np_dtype)
    else:
        values = _numpy_rm_matmul(handle.matrix, batch)
    if was_vector:
        values = values.reshape(-1)
    values = np.asarray(values, dtype=np_dtype)
    if not np.isfinite(values).all():
        raise FloatingPointError("RM matmul produced non-finite values.")

    metadata = MappingProxyType(
        {
            "backend": handle.backend,
            "device_requested": handle.device_requested,
            "device_effective": handle.device_effective,
            "fallback_reason": handle.fallback_reason,
            "batched": not was_vector,
            "n_frames": int(batch.shape[0]),
            "rm_shape": tuple(int(v) for v in handle.shape),
            "delta_v_shape": tuple(int(v) for v in batch.shape),
            "output_shape": tuple(int(v) for v in values.shape),
            "rm_dtype": handle.dtype,
            "rm_persistent": True,
            "rm_tensor_reused": bool(
                reused_handle and handle.matrix_tensor is not None
            ),
            "rm_prepare_mode": "reused_handle" if reused_handle else "per_call",
            "rm_matrix_resident": "device"
            if handle.matrix_tensor is not None
            else "cpu",
            "rm_cache_key": handle.cache_key,
            "host_device_transfer": _host_device_transfer_label(handle),
        }
    )
    if return_metadata:
        return RMMatmulResult(values=values, metadata=metadata)
    return values


def _as_rm_matrix(rm: Any, *, dtype: np.dtype[Any] | type = np.float64) -> np.ndarray:
    if sparse.issparse(rm):
        matrix = np.asarray(rm.toarray(), dtype=dtype)
    else:
        matrix = np.asarray(rm, dtype=dtype)
    if matrix.ndim != 2:
        raise ValueError("rm must be a 2D reconstruction matrix.")
    if 0 in matrix.shape:
        raise ValueError("rm must be non-empty.")
    if not np.isfinite(matrix).all():
        raise FloatingPointError("rm contains non-finite values.")
    return np.ascontiguousarray(matrix, dtype=dtype)


def _as_delta_batch(
    delta_v: Any,
    *,
    n_measurements: int,
    dtype: np.dtype[Any] | type = np.float64,
) -> tuple[np.ndarray, bool]:
    values = np.asarray(delta_v, dtype=dtype)
    if values.ndim == 1:
        batch = values.reshape(1, -1)
        was_vector = True
    elif values.ndim == 2:
        batch = values
        was_vector = False
    else:
        raise ValueError("delta_v must be a 1D vector or 2D frame batch.")
    if batch.shape[1] != int(n_measurements):
        raise ValueError(
            f"delta_v measurement dimension {batch.shape[1]} does not match RM columns "
            f"{n_measurements}."
        )
    if batch.shape[0] == 0:
        raise ValueError("delta_v batch must contain at least one frame.")
    if not np.isfinite(batch).all():
        raise FloatingPointError("delta_v contains non-finite values.")
    return np.ascontiguousarray(batch, dtype=dtype), was_vector


def _normalize_device(device: str | None) -> str:
    resolved = str(device or "auto").strip().lower()
    aliases = {"gpu": "cuda", "torch-cuda": "cuda", "numpy": "cpu"}
    resolved = aliases.get(resolved, resolved)
    if resolved not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: 'auto', 'cpu', 'cuda'.")
    return resolved


def _torch_cuda_available() -> bool:
    return bool(
        torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()
    )


def _resolve_effective_device(requested: str) -> tuple[str, str | None]:
    if requested == "cpu":
        return "cpu", None
    if _torch_cuda_available():
        return "cuda", None
    if requested == "cuda":
        raise RuntimeError(
            "RM CUDA matmul requested but Torch CUDA is unavailable. "
            "Use `nix develop .#cuda` and verify torch.cuda.is_available()."
        )
    return "cpu", "torch_cuda_not_available"


def _normalize_dtype(dtype: str | np.dtype[Any]) -> tuple[np.dtype[Any], str]:
    resolved = np.dtype(dtype)
    if resolved == np.dtype(np.float64):
        return np.dtype(np.float64), "float64"
    if resolved == np.dtype(np.float32):
        return np.dtype(np.float32), "float32"
    raise ValueError("dtype must be 'float64' or 'float32'.")


def _handle_matches(
    handle: RMMatmulHandle,
    *,
    requested: str,
    dtype_name: str,
) -> bool:
    if handle.dtype != dtype_name:
        return False
    if requested == "auto":
        return True
    return handle.device_requested == requested or handle.device_effective == requested


def _torch_dtype(dtype: np.dtype[Any]):
    if torch is None:
        raise RuntimeError("Torch is unavailable.")
    if dtype == np.dtype(np.float32):
        return torch.float32
    return torch.float64


def _torch_as_tensor(values: np.ndarray, *, device: str, dtype: np.dtype[Any]):
    if torch is None:
        raise RuntimeError("Torch is unavailable.")
    return torch.as_tensor(values, device=device, dtype=_torch_dtype(dtype))


def _host_device_transfer_label(handle: RMMatmulHandle) -> str:
    if handle.device_effective != "cuda":
        return "none"
    return "delta_v_to_device+output_to_host"


def _numpy_rm_matmul(matrix: np.ndarray, batch: np.ndarray) -> np.ndarray:
    return np.asarray(
        safe_dot(batch, matrix.T, "rm_matmul.cpu.batch"), dtype=matrix.dtype
    )


def _torch_rm_matmul(
    handle: RMMatmulHandle,
    batch: np.ndarray,
    *,
    device: str,
    dtype: np.dtype[Any],
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("Torch is unavailable.")
    matrix_t = handle.matrix_tensor
    if matrix_t is None:
        matrix_t = _torch_as_tensor(handle.matrix, device=device, dtype=dtype)
    batch_t = _torch_as_tensor(batch, device=device, dtype=dtype)
    out = torch.matmul(batch_t, matrix_t.T)
    return np.asarray(out.detach().cpu().numpy(), dtype=dtype)


__all__ = ["RMMatmulHandle", "RMMatmulResult", "prepare_rm_matmul", "rm_matmul"]
