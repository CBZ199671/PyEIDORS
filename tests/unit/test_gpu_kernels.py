"""Tests for online RM matmul kernels."""

from __future__ import annotations

import numpy as np
import pytest

import pyeidors.perf.gpu_kernels as gpu_kernels
from pyeidors.perf.gpu_kernels import rm_matmul


class _FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    @property
    def T(self):
        return _FakeTensor(self.values.T)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True


class _FakeTorch:
    cuda = _FakeCuda()
    float64 = np.float64

    @staticmethod
    def as_tensor(values, *, device=None, dtype=None):
        _ = (device, dtype)
        return _FakeTensor(values)

    @staticmethod
    def matmul(lhs, rhs):
        return _FakeTensor(lhs.values @ rhs.values)


def test_rm_matmul_cpu_single_and_batch() -> None:
    rm = np.array([[1.0, 2.0, -1.0], [0.5, 0.0, 4.0]], dtype=float)
    vector = np.array([0.25, -0.5, 2.0], dtype=float)
    batch = np.array([[0.25, -0.5, 2.0], [1.0, 2.0, 3.0]], dtype=float)

    single = rm_matmul(rm, vector, device="cpu", return_metadata=True)
    batched = rm_matmul(rm, batch, device="cpu", return_metadata=True)

    np.testing.assert_allclose(single.values, rm @ vector)
    np.testing.assert_allclose(batched.values, batch @ rm.T)
    assert single.metadata["backend"] == "numpy"
    assert single.metadata["batched"] is False
    assert batched.metadata["batched"] is True
    assert batched.metadata["output_shape"] == (2, 2)


def test_rm_matmul_auto_falls_back_to_cpu_when_torch_cuda_missing(monkeypatch) -> None:
    monkeypatch.setattr(gpu_kernels, "torch", None)
    rm = np.eye(2)
    result = rm_matmul(rm, np.ones((3, 2)), device="auto", return_metadata=True)

    np.testing.assert_allclose(result.values, np.ones((3, 2)))
    assert result.metadata["device_effective"] == "cpu"
    assert result.metadata["fallback_reason"] == "torch_cuda_not_available"


def test_rm_matmul_cuda_requires_torch_cuda(monkeypatch) -> None:
    monkeypatch.setattr(gpu_kernels, "torch", None)

    with pytest.raises(RuntimeError, match="Torch CUDA is unavailable"):
        rm_matmul(np.eye(2), np.ones(2), device="cuda")


def test_rm_matmul_cuda_path_uses_torch_backend(monkeypatch) -> None:
    monkeypatch.setattr(gpu_kernels, "torch", _FakeTorch)
    rm = np.array([[1.0, 2.0], [-1.0, 0.5]], dtype=float)
    batch = np.array([[3.0, 4.0], [1.0, -2.0]], dtype=float)

    result = rm_matmul(rm, batch, device="cuda", return_metadata=True)

    np.testing.assert_allclose(result.values, batch @ rm.T)
    assert result.metadata["backend"] == "torch"
    assert result.metadata["device_effective"] == "cuda"


def test_rm_matmul_validates_shapes_and_device() -> None:
    with pytest.raises(ValueError, match="measurement dimension"):
        rm_matmul(np.eye(3), np.ones((2, 2)), device="cpu")
    with pytest.raises(ValueError, match="device must be"):
        rm_matmul(np.eye(2), np.ones(2), device="bad")
