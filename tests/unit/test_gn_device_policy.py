"""Tests for inverse runtime device policy helpers."""

from __future__ import annotations

import pytest

from pyeidors.inverse.solvers.gauss_newton_device import resolve_torch_device


class _CudaNamespace:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def get_device_name(self, device):
        _ = device
        return "Fake CUDA"


class _TorchBackends:
    class _CudaMatmul:
        allow_tf32 = True

    class _Cudnn:
        allow_tf32 = True

    cuda = type("CudaBackend", (), {"matmul": _CudaMatmul()})()
    cudnn = _Cudnn()
    mps = None


class _TorchStub:
    def __init__(self, cuda_available: bool):
        self.cuda = _CudaNamespace(cuda_available)
        self.backends = _TorchBackends()

    @staticmethod
    def device(name: str):
        return name

    @staticmethod
    def set_float32_matmul_precision(mode: str):
        _ = mode
        return None


def test_resolve_torch_device_auto_uses_cpu_when_petsc_cpu(monkeypatch):
    import pyeidors.inverse.solvers.gauss_newton_device as dev_mod

    monkeypatch.setattr(dev_mod, "torch", _TorchStub(cuda_available=True))
    resolved = resolve_torch_device("auto", verbose=False, petsc_device_effective="cpu")

    assert resolved.requested == "auto"
    assert resolved.effective == "cpu"
    assert resolved.fallback_reason == "auto_cpu_policy"


def test_resolve_torch_device_auto_uses_cuda_when_petsc_cuda(monkeypatch):
    import pyeidors.inverse.solvers.gauss_newton_device as dev_mod

    monkeypatch.setattr(dev_mod, "torch", _TorchStub(cuda_available=True))
    resolved = resolve_torch_device("auto", verbose=False, petsc_device_effective="cuda")

    assert resolved.requested == "auto"
    assert resolved.effective == "cuda"
    assert resolved.torch_device == "cuda"


def test_resolve_torch_device_cuda_requires_torch_cuda(monkeypatch):
    import pyeidors.inverse.solvers.gauss_newton_device as dev_mod

    monkeypatch.setattr(dev_mod, "torch", _TorchStub(cuda_available=False))
    with pytest.raises(RuntimeError, match="device='cuda'"):
        resolve_torch_device("cuda", verbose=False, petsc_device_effective="cpu")
