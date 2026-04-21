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


def rm_matmul(
    rm: Any,
    delta_v: Any,
    *,
    device: str = "auto",
    return_metadata: bool = False,
) -> np.ndarray | RMMatmulResult:
    """Apply ``RM @ ΔV`` for one frame or a frame batch.

    ``delta_v`` may be shape ``(n_meas,)`` or ``(n_frames, n_meas)``.
    Batched output is shape ``(n_frames, n_param)``. ``device="auto"``
    chooses CUDA when Torch CUDA is available, otherwise NumPy CPU.
    """

    matrix = _as_rm_matrix(rm)
    batch, was_vector = _as_delta_batch(delta_v, n_measurements=matrix.shape[1])
    requested = _normalize_device(device)
    effective, fallback_reason = _resolve_effective_device(requested)

    if effective == "cuda":
        values = _torch_rm_matmul(matrix, batch, device="cuda")
        backend = "torch"
    else:
        values = _numpy_rm_matmul(matrix, batch)
        backend = "numpy"
    if was_vector:
        values = values.reshape(-1)
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError("RM matmul produced non-finite values.")

    metadata = MappingProxyType(
        {
            "backend": backend,
            "device_requested": requested,
            "device_effective": effective,
            "fallback_reason": fallback_reason,
            "batched": not was_vector,
            "n_frames": int(batch.shape[0]),
            "rm_shape": tuple(int(v) for v in matrix.shape),
            "delta_v_shape": tuple(int(v) for v in batch.shape),
            "output_shape": tuple(int(v) for v in values.shape),
        }
    )
    if return_metadata:
        return RMMatmulResult(values=values, metadata=metadata)
    return values


def _as_rm_matrix(rm: Any) -> np.ndarray:
    if sparse.issparse(rm):
        matrix = np.asarray(rm.toarray(), dtype=np.float64)
    else:
        matrix = np.asarray(rm, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("rm must be a 2D reconstruction matrix.")
    if 0 in matrix.shape:
        raise ValueError("rm must be non-empty.")
    if not np.isfinite(matrix).all():
        raise FloatingPointError("rm contains non-finite values.")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _as_delta_batch(delta_v: Any, *, n_measurements: int) -> tuple[np.ndarray, bool]:
    values = np.asarray(delta_v, dtype=np.float64)
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
    return np.ascontiguousarray(batch, dtype=np.float64), was_vector


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


def _numpy_rm_matmul(matrix: np.ndarray, batch: np.ndarray) -> np.ndarray:
    return np.asarray(
        safe_dot(batch, matrix.T, "rm_matmul.cpu.batch"), dtype=np.float64
    )


def _torch_rm_matmul(
    matrix: np.ndarray, batch: np.ndarray, *, device: str
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("Torch is unavailable.")
    matrix_t = torch.as_tensor(matrix, device=device, dtype=torch.float64)
    batch_t = torch.as_tensor(batch, device=device, dtype=torch.float64)
    out = torch.matmul(batch_t, matrix_t.T)
    return np.asarray(out.detach().cpu().numpy(), dtype=np.float64)


__all__ = ["RMMatmulResult", "rm_matmul"]
