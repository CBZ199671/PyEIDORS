"""Allocation-focused tests for Gauss-Newton runtime helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from pyeidors.inverse.solvers import gauss_newton_runtime as gn_runtime


def test_to_runtime_tensor_shares_memory_on_cpu():
    reconstructor = SimpleNamespace(device="cpu", _torch_dtype=torch.float64)
    values = np.arange(8, dtype=np.float64)

    tensor = gn_runtime._to_runtime_tensor(reconstructor, values)
    values[0] = 42.0

    assert tensor.device.type == "cpu"
    assert tensor.dtype == torch.float64
    assert float(tensor[0].item()) == 42.0


def test_runtime_uses_tensor_adapter_in_reconstruction(eit_system, monkeypatch):
    reconstructor = eit_system.reconstructor
    reconstructor.max_iterations = 1
    reconstructor.min_iterations = 1
    reconstructor.clip_values = None
    reconstructor.verbose = False

    baseline = eit_system.create_homogeneous_image(conductivity=1.0)
    measured, _ = eit_system.fwd_model.fwd_solve(baseline)

    calls = {"count": 0}
    original = gn_runtime._to_runtime_tensor

    def _counting_adapter(runtime_reconstructor, values):
        calls["count"] += 1
        return original(runtime_reconstructor, values)

    monkeypatch.setattr(gn_runtime, "_to_runtime_tensor", _counting_adapter)

    result = reconstructor.reconstruct(
        measured_data=measured,
        initial_conductivity=1.0,
        jacobian_method="efficient",
    )

    assert result.iterations >= 1
    assert calls["count"] >= 4
