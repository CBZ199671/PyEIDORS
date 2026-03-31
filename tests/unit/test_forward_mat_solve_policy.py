"""Policy tests for PETSc matSolve auto|off|on routing."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import pyeidors.forward.eit_forward_model as forward_module

import pyeidors.perf.capabilities as perf_caps
from pyeidors.forward.eit_forward_model import EITForwardModel


class _FakeVec:
    def __init__(self, size: int):
        self.arr = np.zeros(size, dtype=float)

    def getArray(self, readonly=False):
        _ = readonly
        return self.arr


class _FakeDenseMat:
    def __init__(self):
        self.arr = None

    def createDense(self, size, array=None, comm=None):
        _ = comm
        out = _FakeDenseMat()
        if array is None:
            out.arr = np.zeros(size, dtype=float, order="F")
        else:
            out.arr = np.array(array, dtype=float, copy=True, order="F")
        return out

    def getDenseArray(self):
        return self.arr

    def destroy(self):
        return None


class _FakePETSc:
    class Mat(_FakeDenseMat):
        pass


class _FakeA:
    def __init__(self, size: int):
        self.size = size

    def createVecRight(self):
        return _FakeVec(self.size)


class _FakeKSP:
    def __init__(self):
        self.mat_calls = 0
        self.solve_calls = 0

    def matSolve(self, B, X):
        self.mat_calls += 1
        X.arr[:, :] = B.arr * 2.0

    def solve(self, b, x):
        self.solve_calls += 1
        x.arr[:] = b.arr + 1.0

    def getConvergedReason(self):
        return 1


def _make_model(*, mat_solve_mode: str, mesh_tdim: int) -> tuple[EITForwardModel, _FakeKSP]:
    model = EITForwardModel.__new__(EITForwardModel)
    model.cache_manager = None
    model.dofs = 2
    model.n_elec = 1
    model.mesh = SimpleNamespace(comm=None)
    model.mesh_tdim = mesh_tdim
    model.performance_mode = "aggressive"
    model.backend_config = SimpleNamespace(
        mat_solve_mode=mat_solve_mode,
        use_mat_solve=False,
    )
    model._sigma_fingerprint = lambda sigma: "sigma-hash"
    model._base_cache_payload = lambda sigma_hash, n_patterns: {
        "sigma_hash": sigma_hash,
        "n_patterns": n_patterns,
    }
    ksp = _FakeKSP()
    model._create_full_matrix_petsc = lambda sigma: _FakeA(model.dofs + model.n_elec + 1)
    model._make_petsc_solver_bundle = lambda system_matrix: {"A": system_matrix, "ksp": ksp}
    model._last_cache_lookup = {}
    return model, ksp


def test_forward_mat_solve_mode_off_uses_vector_loop(monkeypatch):
    model, ksp = _make_model(mat_solve_mode="off", mesh_tdim=3)
    monkeypatch.setattr(forward_module, "PETSc", _FakePETSc)

    pattern_matrix = np.array([[1.0], [2.0]], dtype=float)
    sol = model._solve_with_petsc(sigma=None, pattern_matrix=pattern_matrix)

    assert ksp.mat_calls == 0
    assert ksp.solve_calls == pattern_matrix.shape[0]
    assert sol.shape == (model.dofs + model.n_elec + 1, pattern_matrix.shape[0])


def test_forward_mat_solve_mode_auto_prefers_mat_solve_for_3d(monkeypatch):
    model, ksp = _make_model(mat_solve_mode="auto", mesh_tdim=3)
    monkeypatch.setattr(forward_module, "PETSc", _FakePETSc)

    pattern_matrix = np.array([[1.0], [2.0]], dtype=float)
    _ = model._solve_with_petsc(sigma=None, pattern_matrix=pattern_matrix)

    assert ksp.mat_calls == 1
    assert ksp.solve_calls == 0


def test_resolve_petsc_backend_info_auto_falls_back_to_cpu(monkeypatch):
    model = EITForwardModel.__new__(EITForwardModel)
    model.linear_backend = "petsc"
    model.backend_config = SimpleNamespace(petsc_device="auto")
    monkeypatch.setattr(forward_module, "PETSc", object())
    monkeypatch.setattr(
        perf_caps,
        "probe_petsc_cuda_runtime",
        lambda: {"petsc_cuda": False, "errors": {"mat": "Unknown type"}},
    )

    info = EITForwardModel._resolve_petsc_backend_info(model)

    assert info["petsc_device_requested"] == "auto"
    assert info["petsc_device_effective"] == "cpu"
    assert info["gpu_fallback_reason"] == "petsc_cuda_not_available"


def test_resolve_petsc_backend_info_cuda_requires_real_capability(monkeypatch):
    model = EITForwardModel.__new__(EITForwardModel)
    model.linear_backend = "petsc"
    model.backend_config = SimpleNamespace(petsc_device="cuda")
    monkeypatch.setattr(forward_module, "PETSc", object())
    monkeypatch.setattr(
        perf_caps,
        "probe_petsc_cuda_runtime",
        lambda: {"petsc_cuda": False, "errors": {"mat": "Unknown type", "vec": "Unknown type"}},
    )

    try:
        EITForwardModel._resolve_petsc_backend_info(model)
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected RuntimeError for unavailable CUDA PETSc runtime")

    assert "petsc_device='cuda'" in message
    assert "nix develop .#cuda" in message


def test_resolve_mfem_runtime_device_prefers_requested_cuda_when_petsc_is_cpu():
    model = EITForwardModel.__new__(EITForwardModel)
    model.backend_config = SimpleNamespace(petsc_device="cuda")
    model._petsc_backend_info = {"petsc_device_effective": "cpu"}

    assert EITForwardModel._resolve_mfem_runtime_device(model) == "cuda"


def test_resolve_mfem_runtime_device_keeps_effective_cuda():
    model = EITForwardModel.__new__(EITForwardModel)
    model.backend_config = SimpleNamespace(petsc_device="auto")
    model._petsc_backend_info = {"petsc_device_effective": "cuda"}

    assert EITForwardModel._resolve_mfem_runtime_device(model) == "cuda"


def test_ensure_electrode_matrix_is_lazy():
    model = EITForwardModel.__new__(EITForwardModel)
    calls = {"count": 0}

    def fake_assemble():
        calls["count"] += 1
        return "assembled"

    model.M = None
    model._assemble_electrode_matrix = fake_assemble

    assert EITForwardModel._ensure_electrode_matrix(model) == "assembled"
    assert EITForwardModel._ensure_electrode_matrix(model) == "assembled"
    assert calls["count"] == 1
