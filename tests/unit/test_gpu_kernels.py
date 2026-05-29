"""Tests for online RM matmul kernels."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import pyeidors.perf.gpu_kernels as gpu_kernels
from pyeidors.perf.gpu_kernels import prepare_rm_matmul, rm_matmul


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
    float32 = np.float32

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


def test_v485_rm_matmul_finite_guards_use_bounded_scanner() -> None:
    checked_functions = (
        gpu_kernels.rm_matmul,
        gpu_kernels._as_rm_matrix,
        gpu_kernels._as_delta_batch,
    )
    old_payload_scans = (
        "np.isfinite(values).all()",
        "np.isfinite(matrix).all()",
        "np.isfinite(batch).all()",
    )

    for func in checked_functions:
        source = inspect.getsource(func)
        assert "all_finite_values(" in source
        for old_payload_scan in old_payload_scans:
            assert old_payload_scan not in source


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


def test_prepare_rm_matmul_reuses_cuda_matrix_tensor(monkeypatch) -> None:
    class _CountingFakeTorch(_FakeTorch):
        tensor_shapes: list[tuple[int, ...]] = []

        @staticmethod
        def as_tensor(values, *, device=None, dtype=None):
            _ = (device, dtype)
            _CountingFakeTorch.tensor_shapes.append(tuple(np.asarray(values).shape))
            return _FakeTensor(values)

    monkeypatch.setattr(gpu_kernels, "torch", _CountingFakeTorch)
    rm = np.array([[1.0, 2.0], [-1.0, 0.5]], dtype=float)
    batch = np.array([[3.0, 4.0], [1.0, -2.0]], dtype=float)

    handle = prepare_rm_matmul(rm, device="cuda", cache_key="unit-rm")
    first = rm_matmul(handle, batch, device="cuda", return_metadata=True)
    second = rm_matmul(handle, batch * 2.0, device="cuda", return_metadata=True)

    np.testing.assert_allclose(first.values, batch @ rm.T)
    np.testing.assert_allclose(second.values, (batch * 2.0) @ rm.T)
    assert _CountingFakeTorch.tensor_shapes == [rm.shape, batch.shape, batch.shape]
    assert first.metadata["rm_prepare_mode"] == "reused_handle"
    assert first.metadata["rm_tensor_reused"] is True
    assert first.metadata["rm_cache_key"] == "unit-rm"
    assert first.metadata["host_device_transfer"] == "delta_v_to_device+output_to_host"


def test_prepare_rm_matmul_can_use_torch_compile(monkeypatch) -> None:
    class _CompileFakeTorch(_FakeTorch):
        compile_calls = 0
        compiled_calls = 0

        @staticmethod
        def compile(fn, **kwargs):
            assert kwargs["mode"] == "reduce-overhead"
            assert kwargs["fullgraph"] is True
            _CompileFakeTorch.compile_calls += 1

            def _compiled(lhs, rhs):
                _CompileFakeTorch.compiled_calls += 1
                return fn(lhs, rhs)

            return _compiled

    monkeypatch.setattr(gpu_kernels, "torch", _CompileFakeTorch)
    rm = np.array([[1.0, 2.0], [-1.0, 0.5]], dtype=float)
    batch = np.array([[3.0, 4.0], [1.0, -2.0]], dtype=float)

    handle = prepare_rm_matmul(rm, device="cuda", compile_mode="force")
    result = rm_matmul(handle, batch, device="cuda", return_metadata=True)

    np.testing.assert_allclose(result.values, batch @ rm.T)
    assert _CompileFakeTorch.compile_calls == 1
    assert _CompileFakeTorch.compiled_calls == 1
    assert result.metadata["rm_matmul_compiled"] is True
    assert result.metadata["rm_matmul_compile_mode"] == "force"
    assert result.metadata["rm_matmul_compile_status"] == "compiled"
    assert result.metadata["online_hot_path"] == "rm_torch_compile_matmul"


def test_prepare_rm_matmul_records_cpu_dtype_policy() -> None:
    rm = np.array([[1.0, 2.0], [-1.0, 0.5]], dtype=float)
    batch = np.array([[3.0, 4.0], [1.0, -2.0]], dtype=float)

    handle = prepare_rm_matmul(rm, device="cpu", dtype="float32")
    result = rm_matmul(
        handle,
        batch,
        device="cpu",
        dtype="float32",
        return_metadata=True,
    )

    np.testing.assert_allclose(result.values, batch @ rm.T, rtol=1e-6)
    assert result.metadata["backend"] == "numpy"
    assert result.metadata["rm_dtype"] == "float32"
    assert result.metadata["rm_matrix_resident"] == "cpu"
    assert result.metadata["rm_prepare_mode"] == "reused_handle"


def test_rm_matmul_validates_shapes_and_device() -> None:
    with pytest.raises(ValueError, match="measurement dimension"):
        rm_matmul(np.eye(3), np.ones((2, 2)), device="cpu")
    with pytest.raises(ValueError, match="device must be"):
        rm_matmul(np.eye(2), np.ones(2), device="bad")
    with pytest.raises(ValueError, match="dtype must be"):
        rm_matmul(np.eye(2), np.ones(2), device="cpu", dtype="float16")
