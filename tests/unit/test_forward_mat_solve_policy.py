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

    def getType(self):
        return "seq"


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

    def getType(self):
        return "aij"


class _FakeKSP:
    def __init__(self):
        self.mat_calls = 0
        self.solve_calls = 0
        self.raise_on_mat_solve = False
        self.converged_reason = 1
        self.mat_converged_reason = 1
        self.solve_converged_reason = 1

    def matSolve(self, B, X):
        self.mat_calls += 1
        if self.raise_on_mat_solve:
            raise RuntimeError("matSolve failed")
        X.arr[:, :] = B.arr * 2.0
        self.converged_reason = self.mat_converged_reason

    def solve(self, b, x):
        self.solve_calls += 1
        x.arr[:] = b.arr + 1.0
        self.converged_reason = self.solve_converged_reason

    def getConvergedReason(self):
        return self.converged_reason


def _make_model(
    *,
    mat_solve_mode: str,
    mesh_tdim: int,
    solve_mat_type: str = "aij",
) -> tuple[EITForwardModel, _FakeKSP]:
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
        reuse_preconditioner=True,
    )
    model._sigma_fingerprint = lambda sigma: "sigma-hash"
    model._base_cache_payload = lambda sigma_hash, n_patterns: {
        "sigma_hash": sigma_hash,
        "n_patterns": n_patterns,
    }
    calls = {"matrix": 0, "bundle": 0}

    def _create_full_matrix(_sigma):
        calls["matrix"] += 1
        return _FakeA(model.dofs + model.n_elec + 1)

    def _make_bundle(system_matrix):
        calls["bundle"] += 1
        return {
            "A": system_matrix,
            "solve_A": system_matrix,
            "ksp": ksp,
            "backend": "petsc-ksp",
            "ksp_type": "cg",
            "pc_type": "gamg",
            "factor_solver_type": None,
            "solve_mat_type": solve_mat_type,
            "ksp_setup_count": 1,
            "reuse_preconditioner": bool(model.backend_config.reuse_preconditioner),
            "reuse_preconditioner_applied": True,
        }

    ksp = _FakeKSP()
    model._create_full_matrix_petsc = _create_full_matrix
    model._make_petsc_solver_bundle = _make_bundle
    model._last_cache_lookup = {}
    model._test_calls = calls
    return model, ksp


def test_forward_mat_solve_mode_off_uses_vector_loop(monkeypatch):
    model, ksp = _make_model(mat_solve_mode="off", mesh_tdim=3)
    monkeypatch.setattr(forward_module, "PETSc", _FakePETSc)

    pattern_matrix = np.array([[1.0], [2.0]], dtype=float)
    sol = model._solve_with_petsc(sigma=None, pattern_matrix=pattern_matrix)

    assert ksp.mat_calls == 0
    assert ksp.solve_calls == pattern_matrix.shape[0]
    assert sol.shape == (model.dofs + model.n_elec + 1, pattern_matrix.shape[0])
    assert model._test_calls == {"matrix": 1, "bundle": 1}
    diag = model.get_backend_diagnostics()
    assert diag["forward_rhs_count"] == pattern_matrix.shape[0]
    assert diag["forward_ksp_solve_count"] == pattern_matrix.shape[0]
    assert diag["forward_ksp_mat_solve_count"] == 0
    assert diag["forward_mat_solve_effective"] == "vec-loop"
    assert diag["forward_factor_cache_hit"] is False
    assert diag["ksp_type"] == "cg"
    assert diag["pc_type"] == "gamg"
    assert diag["petsc_mat_type"] == "aij"
    assert diag["petsc_vec_type"] == "seq"
    assert diag["forward_ksp_setup_count"] == 1
    assert diag["forward_ksp_setup_attempts"] == 1
    assert diag["forward_reuse_preconditioner_requested"] is True
    assert diag["forward_reuse_preconditioner_applied"] is True


def test_forward_mat_solve_mode_off_overrides_cuda_dense_auto(monkeypatch):
    model, ksp = _make_model(
        mat_solve_mode="off",
        mesh_tdim=3,
        solve_mat_type="densecuda",
    )
    model._petsc_backend_info = {
        "petsc_device_requested": "cuda",
        "petsc_device_effective": "cuda",
        "capability": {"petsc_cuda_dense": True},
    }
    monkeypatch.setattr(forward_module, "PETSc", _FakePETSc)

    pattern_matrix = np.array([[1.0], [2.0]], dtype=float)
    _ = model._solve_with_petsc(sigma=None, pattern_matrix=pattern_matrix)

    assert model._should_use_mat_solve(pattern_matrix.shape[0]) is False
    assert ksp.mat_calls == 0
    assert ksp.solve_calls == pattern_matrix.shape[0]
    diag = model.get_backend_diagnostics()
    assert diag["forward_mat_solve_effective"] == "vec-loop"
    assert diag["forward_ksp_mat_solve_count"] == 0


def test_forward_mat_solve_mode_auto_prefers_mat_solve_for_3d(monkeypatch):
    model, ksp = _make_model(mat_solve_mode="auto", mesh_tdim=3)
    monkeypatch.setattr(forward_module, "PETSc", _FakePETSc)

    pattern_matrix = np.array([[1.0], [2.0]], dtype=float)
    _ = model._solve_with_petsc(sigma=None, pattern_matrix=pattern_matrix)

    assert ksp.mat_calls == 1
    assert ksp.solve_calls == 0
    assert model._test_calls == {"matrix": 1, "bundle": 1}
    diag = model.get_backend_diagnostics()
    assert diag["forward_rhs_count"] == pattern_matrix.shape[0]
    assert diag["forward_ksp_mat_solve_count"] == 1
    assert diag["forward_ksp_solve_count"] == 0
    assert diag["forward_ksp_converged_reason"] == 1
    assert diag["forward_ksp_converged"] is True
    assert diag["forward_mat_solve_effective"] == "matsolve"


def test_forward_mat_solve_mode_on_forces_mat_solve(monkeypatch):
    model, ksp = _make_model(mat_solve_mode="on", mesh_tdim=2)
    monkeypatch.setattr(forward_module, "PETSc", _FakePETSc)

    pattern_matrix = np.array([[1.0]], dtype=float)
    _ = model._solve_with_petsc(sigma=None, pattern_matrix=pattern_matrix)

    assert ksp.mat_calls == 1
    assert ksp.solve_calls == 0
    assert model._test_calls == {"matrix": 1, "bundle": 1}
    diag = model.get_backend_diagnostics()
    assert diag["forward_rhs_count"] == 1
    assert diag["forward_ksp_mat_solve_count"] == 1
    assert diag["forward_ksp_solve_count"] == 0
    assert diag["forward_ksp_converged_reason"] == 1
    assert diag["forward_ksp_converged"] is True
    assert diag["forward_mat_solve_effective"] == "matsolve"


def test_forward_mat_solve_cpu_failure_falls_back_to_vector_loop(monkeypatch):
    model, ksp = _make_model(mat_solve_mode="on", mesh_tdim=3)
    ksp.raise_on_mat_solve = True
    monkeypatch.setattr(forward_module, "PETSc", _FakePETSc)

    pattern_matrix = np.array([[1.0], [2.0]], dtype=float)
    sol = model._solve_with_petsc(sigma=None, pattern_matrix=pattern_matrix)

    assert ksp.mat_calls == 1
    assert ksp.solve_calls == pattern_matrix.shape[0]
    assert sol.shape == (model.dofs + model.n_elec + 1, pattern_matrix.shape[0])
    diag = model.get_backend_diagnostics()
    assert diag["forward_mat_solve_effective"] == "vec-loop"
    assert diag["forward_ksp_mat_solve_count"] == 0
    assert diag["forward_ksp_solve_count"] == pattern_matrix.shape[0]
    assert diag["forward_mat_solve_fallback_reason"] == "matSolve failed"
    assert str(diag["gpu_fallback_reason"]).startswith("matSolve_failed:")


def test_forward_mat_solve_negative_reason_falls_back_to_vector_loop(monkeypatch):
    model, ksp = _make_model(mat_solve_mode="on", mesh_tdim=3)
    ksp.mat_converged_reason = -3
    monkeypatch.setattr(forward_module, "PETSc", _FakePETSc)

    pattern_matrix = np.array([[1.0], [2.0]], dtype=float)
    _ = model._solve_with_petsc(sigma=None, pattern_matrix=pattern_matrix)

    assert ksp.mat_calls == 1
    assert ksp.solve_calls == pattern_matrix.shape[0]
    diag = model.get_backend_diagnostics()
    assert diag["forward_mat_solve_effective"] == "vec-loop"
    assert diag["forward_ksp_converged_reason"] == 1
    assert diag["forward_ksp_converged"] is True
    assert (
        "negative convergence reason (-3)" in diag["forward_mat_solve_fallback_reason"]
    )


def test_forward_reuse_preconditioner_disabled_is_diagnosed(monkeypatch):
    model, _ksp = _make_model(mat_solve_mode="off", mesh_tdim=3)
    model.backend_config.reuse_preconditioner = False
    monkeypatch.setattr(forward_module, "PETSc", _FakePETSc)

    pattern_matrix = np.array([[1.0], [2.0]], dtype=float)
    _ = model._solve_with_petsc(sigma=None, pattern_matrix=pattern_matrix)

    diag = model.get_backend_diagnostics()
    assert diag["forward_reuse_preconditioner_requested"] is False
    assert diag["forward_reuse_preconditioner_applied"] is True
    assert diag["forward_ksp_setup_count"] == 1


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
        lambda: {
            "petsc_cuda": False,
            "errors": {"mat": "Unknown type", "vec": "Unknown type"},
        },
    )

    try:
        EITForwardModel._resolve_petsc_backend_info(model)
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected RuntimeError for unavailable CUDA PETSc runtime")

    assert "petsc_device='cuda'" in message
    assert "nix develop .#cuda" in message


def test_resolve_petsc_backend_info_cuda_amgx_requires_pcamgx(monkeypatch):
    model = EITForwardModel.__new__(EITForwardModel)
    model.linear_backend = "petsc"
    model.backend_config = SimpleNamespace(
        petsc_device="cuda",
        solver_preset="cuda_amgx",
        pc_type="amgx",
    )
    monkeypatch.setattr(forward_module, "PETSc", object())
    monkeypatch.setattr(
        perf_caps,
        "probe_petsc_cuda_runtime",
        lambda: {
            "petsc_cuda": True,
            "petsc_cuda_mat": True,
            "petsc_cuda_vec": True,
            "petsc_cuda_dense": True,
            "petsc_amgx": False,
            "mat_type_name": "aijcusparse",
            "vec_type_name": "cuda",
            "dense_mat_type_name": "densecuda",
            "errors": {},
        },
    )

    try:
        EITForwardModel._resolve_petsc_backend_info(model)
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected RuntimeError for missing PETSc PCAMGX")

    assert "当前 PETSc 未启用 AmgX" in message
    assert "PCAMGX unavailable" in message


def test_should_use_mat_solve_respects_min_patterns_threshold():
    model, _ksp = _make_model(mat_solve_mode="auto", mesh_tdim=3)
    model.backend_config.forward_mat_solve_min_patterns = 0
    assert model._should_use_mat_solve(2) is True
    assert model._should_use_mat_solve(1) is False

    model.backend_config.forward_mat_solve_min_patterns = 4
    assert model._should_use_mat_solve(2) is False
    assert model._should_use_mat_solve(3) is False
    assert model._should_use_mat_solve(4) is True
    assert model._should_use_mat_solve(16) is True


def test_should_use_mat_solve_cuda_auto_still_respects_v2_formula():
    model, _ksp = _make_model(mat_solve_mode="auto", mesh_tdim=3)
    model._petsc_backend_info = {
        "petsc_device_effective": "cuda",
        "capability": {"petsc_cuda_dense": True},
    }

    model.backend_config.forward_mat_solve_min_patterns = 4
    assert model._should_use_mat_solve(2) is False
    assert model._should_use_mat_solve(4) is True

    model.backend_config.forward_mat_solve_min_patterns = 0
    model.performance_mode = "conservative"
    assert model._should_use_mat_solve(16) is False


def test_should_use_mat_solve_min_patterns_ignored_in_on_mode():
    model, _ksp = _make_model(mat_solve_mode="on", mesh_tdim=3)
    model.backend_config.forward_mat_solve_min_patterns = 64
    # Explicit "on" overrides the auto-mode threshold.
    assert model._should_use_mat_solve(2) is True


def test_should_use_mat_solve_min_patterns_defaults_to_current_behaviour():
    model, _ksp = _make_model(mat_solve_mode="auto", mesh_tdim=3)
    # No explicit threshold attribute → defaults to 0 → existing heuristic wins.
    if hasattr(model.backend_config, "forward_mat_solve_min_patterns"):
        del model.backend_config.forward_mat_solve_min_patterns
    assert model._should_use_mat_solve(2) is True
    assert model._should_use_mat_solve(1) is False


def test_gpu_gauge_fix_enabled_tracks_effective_cuda_backend():
    model = EITForwardModel.__new__(EITForwardModel)
    model._petsc_backend_info = {"petsc_device_effective": "cpu"}
    assert EITForwardModel._gpu_gauge_fix_enabled(model) is False

    model._petsc_backend_info = {"petsc_device_effective": "cuda"}
    assert EITForwardModel._gpu_gauge_fix_enabled(model) is True


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
