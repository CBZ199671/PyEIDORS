"""Device selection helpers for Gauss-Newton solver."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ResolvedTorchDevice:
    """Resolved runtime device policy for inverse-side Torch computation."""

    requested: str
    effective: str
    torch_device: torch.device
    fallback_reason: str | None = None

    @property
    def type(self) -> str:
        """Mirror ``torch.device.type`` for callers that need the compact device label."""
        return str(self.torch_device.type)


RUNTIME_DEVICE_AUTO = "auto"
RUNTIME_DEVICE_CPU = "cpu"
RUNTIME_DEVICE_CUDA = "cuda"


def normalize_runtime_device(
    value: object, *, default: str = RUNTIME_DEVICE_AUTO
) -> str:
    """Normalize public runtime device values while preserving explicit CUDA indices."""
    normalized = str(value if value is not None else default).strip().lower()
    if not normalized:
        normalized = str(default).strip().lower() or RUNTIME_DEVICE_AUTO
    if normalized in {RUNTIME_DEVICE_AUTO, RUNTIME_DEVICE_CPU, RUNTIME_DEVICE_CUDA}:
        return normalized
    if normalized.startswith("cuda:"):
        return normalized
    if normalized.startswith("mps"):
        return normalized
    return str(default).strip().lower() or RUNTIME_DEVICE_AUTO


def normalize_runtime_device_label(
    value: object, *, default: str = RUNTIME_DEVICE_CPU
) -> str:
    """Collapse explicit runtime devices to stable diagnostics labels."""
    normalized = normalize_runtime_device(value, default=default)
    if normalized.startswith("cuda"):
        return RUNTIME_DEVICE_CUDA
    if normalized.startswith("mps"):
        return "mps"
    return normalized


def _disable_tf32() -> None:
    def _set_fp32_precision(backend: object, value: str) -> bool:
        if backend is None or not hasattr(backend, "fp32_precision"):
            return False
        try:
            setattr(backend, "fp32_precision", value)
        except Exception:
            return False
        return True

    cuda_backend = getattr(torch.backends, "cuda", None)
    matmul_backend = getattr(cuda_backend, "matmul", None)
    if not _set_fp32_precision(matmul_backend, "ieee") and matmul_backend is not None:
        try:
            matmul_backend.allow_tf32 = False
        except Exception:
            pass
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    conv_backend = getattr(cudnn_backend, "conv", None)
    cudnn_updated = _set_fp32_precision(conv_backend, "ieee")
    cudnn_updated = _set_fp32_precision(cudnn_backend, "ieee") or cudnn_updated
    if not cudnn_updated and cudnn_backend is not None:
        try:
            cudnn_backend.allow_tf32 = False
        except Exception:
            pass
    set_precision = getattr(torch, "set_float32_matmul_precision", None)
    if callable(set_precision):
        try:
            set_precision("highest")
        except Exception:
            pass


def resolve_torch_device(
    requested: str,
    *,
    verbose: bool,
    petsc_device_effective: str = RUNTIME_DEVICE_CPU,
) -> ResolvedTorchDevice:
    """Resolve inverse runtime device with explicit auto/cpu/cuda semantics."""
    normalized = normalize_runtime_device(requested, default=RUNTIME_DEVICE_AUTO)
    petsc_effective = normalize_runtime_device_label(
        petsc_device_effective, default=RUNTIME_DEVICE_CPU
    )

    if normalized == RUNTIME_DEVICE_AUTO:
        if petsc_effective == RUNTIME_DEVICE_CUDA and torch.cuda.is_available():
            _disable_tf32()
            device = torch.device(RUNTIME_DEVICE_CUDA)
            if verbose:
                print(f"Using GPU: {torch.cuda.get_device_name(device)}")
            return ResolvedTorchDevice(
                requested=RUNTIME_DEVICE_AUTO,
                effective=RUNTIME_DEVICE_CUDA,
                torch_device=device,
                fallback_reason=None,
            )
        fallback_reason = None
        if petsc_effective == RUNTIME_DEVICE_CUDA and not torch.cuda.is_available():
            fallback_reason = "torch_cuda_unavailable"
        elif petsc_effective != RUNTIME_DEVICE_CUDA:
            fallback_reason = "auto_cpu_policy"
        if verbose:
            print("Using CPU for computation")
        return ResolvedTorchDevice(
            requested=RUNTIME_DEVICE_AUTO,
            effective=RUNTIME_DEVICE_CPU,
            torch_device=torch.device(RUNTIME_DEVICE_CPU),
            fallback_reason=fallback_reason,
        )

    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' requires torch.cuda.is_available(). Enter `nix develop .#cuda` and retry."
            )
        _disable_tf32()
        device = torch.device(
            normalized if normalized.startswith("cuda:") else RUNTIME_DEVICE_CUDA
        )
        if verbose:
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        return ResolvedTorchDevice(
            requested=normalize_runtime_device_label(
                normalized, default=RUNTIME_DEVICE_CUDA
            ),
            effective=RUNTIME_DEVICE_CUDA,
            torch_device=device,
            fallback_reason=None,
        )

    if normalized.startswith("mps"):
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
            if verbose:
                print("Using Apple MPS device")
            return ResolvedTorchDevice(
                requested="mps",
                effective="mps",
                torch_device=device,
                fallback_reason=None,
            )
        raise RuntimeError(
            "device='mps' requested, but torch MPS runtime is unavailable."
        )

    if verbose:
        print("Using CPU for computation")
    return ResolvedTorchDevice(
        requested=RUNTIME_DEVICE_CPU,
        effective=RUNTIME_DEVICE_CPU,
        torch_device=torch.device(RUNTIME_DEVICE_CPU),
        fallback_reason=None,
    )
