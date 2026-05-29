"""Persistent PETSc KSP session reuse across forward solves (G1)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import pyeidors.forward.eit_forward_model as forward_module
from pyeidors.forward.eit_forward_model import EITForwardModel, ForwardKSPSession


class _FakeVec:
    def __init__(self, size: int):
        self.arr = np.zeros(size, dtype=float)

    def getArray(self, readonly=False):
        _ = readonly
        return self.arr

    def getType(self):
        return "seq"


class _FakeA:
    def __init__(self, size: int):
        self.size = size
        self.destroyed = False

    def createVecRight(self):
        return _FakeVec(self.size)

    def getType(self):
        return "aij"

    def destroy(self):
        self.destroyed = True


class _FakeKSP:
    def __init__(self):
        self.set_operators_calls: list[object] = []
        self.set_reuse_calls: list[bool] = []
        self.solve_calls = 0
        self.mat_calls = 0
        self.destroy_calls = 0
        self.iteration_number = 1
        self.solve_converged_reason = 1
        self.mat_converged_reason = 1

    def setOperators(self, A):
        self.set_operators_calls.append(A)

    def setReusePreconditioner(self, value):
        self.set_reuse_calls.append(bool(value))

    def matSolve(self, B, X):
        self.mat_calls += 1
        X.arr[:, :] = B.arr * 2.0

    def solve(self, b, x):
        self.solve_calls += 1
        x.arr[:] = b.arr + 1.0

    def getIterationNumber(self):
        return int(self.iteration_number)

    def getConvergedReason(self):
        return int(self.solve_converged_reason)

    def destroy(self):
        self.destroy_calls += 1


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


def _build_model(
    *,
    mat_solve_mode: str = "off",
    mesh_tdim: int = 3,
    forward_pc_refresh_policy: str = "auto",
    forward_pc_refresh_iter_threshold: int = 0,
    forward_pc_refresh_lag: int = 0,
    reuse_preconditioner: bool = True,
) -> tuple[EITForwardModel, _FakeKSP, dict]:
    model = EITForwardModel.__new__(EITForwardModel)
    model.cache_manager = None
    model.dofs = 2
    model.n_elec = 1
    model.mesh = SimpleNamespace(comm=None)
    model.mesh_tdim = mesh_tdim
    model.performance_mode = "aggressive"
    model.linear_backend = "petsc"
    model.forward_backend = "dolfinx"
    model.backend_config = SimpleNamespace(
        solver_preset="3d_gamg",
        ksp_type="fgmres",
        pc_type="gamg",
        rtol=1e-10,
        atol=1e-12,
        max_it=2000,
        mat_solve_mode=mat_solve_mode,
        use_mat_solve=False,
        reuse_preconditioner=reuse_preconditioner,
        petsc_device="cpu",
        pc_factor_mat_solver_type=None,
        pc_hypre_type=None,
        pc_gamg_type="agg",
        petsc_options={},
        forward_pc_refresh_policy=forward_pc_refresh_policy,
        forward_pc_refresh_iter_threshold=forward_pc_refresh_iter_threshold,
        forward_pc_refresh_lag=forward_pc_refresh_lag,
    )
    model._forward_ksp_session = None
    model._sigma_fingerprint = lambda sigma: f"sigma-hash-{id(sigma)}"
    model._base_cache_payload = lambda sigma_hash, n_patterns: {
        "sigma_hash": sigma_hash,
        "n_patterns": n_patterns,
    }

    calls = {"matrix": 0, "bundle": 0}
    ksp = _FakeKSP()

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
            "ksp_type": "fgmres",
            "pc_type": "gamg",
            "factor_solver_type": None,
            "solve_mat_type": "aij",
            "ksp_setup_count": 1,
            "reuse_preconditioner": bool(reuse_preconditioner),
            "reuse_preconditioner_applied": True,
        }

    model._create_full_matrix_petsc = _create_full_matrix
    model._make_petsc_solver_bundle = _make_bundle
    model._last_cache_lookup = {}
    return model, ksp, calls


def test_forward_ksp_session_reuses_across_calls(monkeypatch):
    model, ksp, calls = _build_model(mat_solve_mode="off")
    monkeypatch.setattr(forward_module, "PETSc", _FakePETSc)

    patterns = np.array([[1.0]], dtype=float)

    model._solve_with_petsc(sigma=object(), pattern_matrix=patterns)
    diag1 = dict(model.get_backend_diagnostics())

    # Second call with a distinct sigma should reuse the session bundle.
    model._solve_with_petsc(sigma=object(), pattern_matrix=patterns)
    diag2 = dict(model.get_backend_diagnostics())

    assert calls["bundle"] == 1, "bundle should be built only on the first solve"
    assert calls["matrix"] == 2, "matrix is reassembled for each new sigma"

    assert diag1["forward_pc_session_reused"] is False
    assert diag1["forward_pc_refresh_triggered"] is True
    assert diag1["forward_ksp_setup_count"] == 1
    assert diag1["forward_pc_session_total_setups"] == 1
    assert diag1["forward_pc_refresh_policy"] == "auto"

    assert diag2["forward_pc_session_reused"] is True
    assert diag2["forward_pc_refresh_triggered"] is False
    assert diag2["forward_pc_refresh_reason"] is None
    assert diag2["forward_ksp_setup_count"] == 0
    assert diag2["forward_factor_cache_hit"] is True
    assert diag2["forward_pc_session_total_setups"] == 1
    assert diag2["forward_reuse_preconditioner_applied"] is True
    assert diag2["forward_ksp_session"]["schema"].endswith("telemetry-v1")
    assert diag2["forward_ksp_session"]["cache_hit"] is True
    assert diag2["forward_ksp_session"]["session_reused"] is True
    assert diag2["forward_ksp_session"]["structural_fingerprint_short"]

    # Second call set PETSc operators once and asked PETSc to reuse PC.
    assert len(ksp.set_operators_calls) == 1
    assert ksp.set_reuse_calls[-1] is True


def test_forward_ksp_session_never_policy_disposes_each_call(monkeypatch):
    model, ksp, calls = _build_model(forward_pc_refresh_policy="never")
    monkeypatch.setattr(forward_module, "PETSc", _FakePETSc)

    patterns = np.array([[1.0]], dtype=float)
    model._solve_with_petsc(sigma=object(), pattern_matrix=patterns)
    model._solve_with_petsc(sigma=object(), pattern_matrix=patterns)
    diag = dict(model.get_backend_diagnostics())

    assert calls["bundle"] == 2, "'never' policy must rebuild the bundle every call"
    assert ksp.destroy_calls == 1, "prior KSP must be destroyed before rebuilding"
    assert diag["forward_pc_session_reused"] is False
    assert diag["forward_pc_refresh_triggered"] is True


def test_forward_ksp_session_lag_policy_triggers_refresh(monkeypatch):
    model, ksp, calls = _build_model(
        forward_pc_refresh_policy="lag",
        forward_pc_refresh_lag=2,
    )
    monkeypatch.setattr(forward_module, "PETSc", _FakePETSc)

    patterns = np.array([[1.0]], dtype=float)

    model._solve_with_petsc(sigma=object(), pattern_matrix=patterns)  # setup
    model._solve_with_petsc(sigma=object(), pattern_matrix=patterns)  # reuse
    diag2 = dict(model.get_backend_diagnostics())
    model._solve_with_petsc(sigma=object(), pattern_matrix=patterns)  # refresh
    diag3 = dict(model.get_backend_diagnostics())

    assert calls["bundle"] == 1, "lag policy reuses the same KSP; no rebuild of bundle"
    assert diag2["forward_pc_session_reused"] is True
    assert diag2["forward_pc_refresh_triggered"] is False
    assert diag3["forward_pc_session_reused"] is True
    assert diag3["forward_pc_refresh_triggered"] is True
    assert diag3["forward_pc_refresh_reason"] == "policy_lag_2_exceeded"
    assert diag3["forward_pc_session_total_setups"] == 2
    # Third call set PETSc PC-reuse flag to False, asking PETSc to rebuild.
    assert ksp.set_reuse_calls[-1] is False


def test_forward_ksp_session_iter_threshold_triggers_refresh(monkeypatch):
    model, ksp, calls = _build_model(
        forward_pc_refresh_policy="auto",
        forward_pc_refresh_iter_threshold=5,
    )
    monkeypatch.setattr(forward_module, "PETSc", _FakePETSc)

    ksp.iteration_number = 12  # simulate PC reuse degrading convergence

    patterns = np.array([[1.0]], dtype=float)
    model._solve_with_petsc(sigma=object(), pattern_matrix=patterns)  # setup
    diag1 = dict(model.get_backend_diagnostics())
    model._solve_with_petsc(sigma=object(), pattern_matrix=patterns)  # refresh
    diag2 = dict(model.get_backend_diagnostics())

    assert calls["bundle"] == 1
    assert diag1["forward_pc_last_iter_count"] == 12
    assert diag2["forward_pc_refresh_triggered"] is True
    assert diag2["forward_pc_refresh_reason"] == ("iter_count_12_gt_threshold_5")
    assert ksp.set_reuse_calls[-1] is False


def test_forward_ksp_session_structural_change_rebuilds(monkeypatch):
    model, ksp, calls = _build_model()
    monkeypatch.setattr(forward_module, "PETSc", _FakePETSc)

    patterns = np.array([[1.0]], dtype=float)
    model._solve_with_petsc(sigma=object(), pattern_matrix=patterns)
    # Change a field that belongs to the structural fingerprint between solves.
    model.backend_config.solver_preset = "3d_hypre"
    model._solve_with_petsc(sigma=object(), pattern_matrix=patterns)
    diag = dict(model.get_backend_diagnostics())

    assert calls["bundle"] == 2, "structural change must invalidate the session"
    assert ksp.destroy_calls == 1
    assert diag["forward_pc_session_reused"] is False
    assert diag["forward_pc_refresh_triggered"] is True


def test_forward_ksp_session_reuse_disabled_rebuilds_pc_without_new_bundle(monkeypatch):
    model, ksp, calls = _build_model(reuse_preconditioner=False)
    monkeypatch.setattr(forward_module, "PETSc", _FakePETSc)

    patterns = np.array([[1.0]], dtype=float)
    model._solve_with_petsc(sigma=object(), pattern_matrix=patterns)
    model._solve_with_petsc(sigma=object(), pattern_matrix=patterns)
    diag = dict(model.get_backend_diagnostics())

    assert calls["bundle"] == 1, "session is still reused; only PC is refreshed"
    assert diag["forward_pc_session_reused"] is True
    assert diag["forward_pc_refresh_triggered"] is True
    assert diag["forward_pc_refresh_reason"] == "reuse_preconditioner_disabled"
    assert ksp.set_reuse_calls[-1] is False


def test_direct_preset_forces_pc_refresh_across_sigma(monkeypatch):
    """preonly + lu/cholesky/qr must NOT reuse factorization across sigma updates."""
    model, ksp, calls = _build_model(
        forward_pc_refresh_policy="auto",
        forward_pc_refresh_iter_threshold=0,
        forward_pc_refresh_lag=0,
    )
    # Switch the backend config to a direct-solve preset.
    model.backend_config.ksp_type = "preonly"
    model.backend_config.pc_type = "lu"
    model.backend_config.solver_preset = "direct"
    # Bundle factory also exposes the direct types so the session metadata matches.
    original_bundle_factory = model._make_petsc_solver_bundle

    def _direct_bundle(system_matrix):
        bundle = original_bundle_factory(system_matrix)
        bundle["ksp_type"] = "preonly"
        bundle["pc_type"] = "lu"
        return bundle

    model._make_petsc_solver_bundle = _direct_bundle
    monkeypatch.setattr(forward_module, "PETSc", _FakePETSc)

    patterns = np.array([[1.0]], dtype=float)

    # First sigma — fresh bundle, PC setup runs (not a reuse decision).
    model._solve_with_petsc(sigma=object(), pattern_matrix=patterns)
    diag1 = dict(model.get_backend_diagnostics())
    assert diag1["forward_pc_session_reused"] is False

    # Second sigma — guard must fire: session reused, but PC FORCED to refresh.
    model._solve_with_petsc(sigma=object(), pattern_matrix=patterns)
    diag2 = dict(model.get_backend_diagnostics())

    assert diag2["forward_pc_session_reused"] is True
    assert diag2["forward_pc_refresh_triggered"] is True
    assert diag2["forward_pc_refresh_reason"] == "direct_factor_requires_rebuild"
    assert diag2["forward_reuse_preconditioner_requested"] is False
    assert diag2["forward_reuse_preconditioner_applied"] is False
    assert ksp.set_reuse_calls[-1] is False, (
        "direct factor reuse must set reusePreconditioner(False)"
    )


def test_iterative_preset_still_reuses_pc_across_sigma(monkeypatch):
    """Control: fgmres+gamg (iterative+AMG) retains reuse — only direct is guarded."""
    model, ksp, _calls = _build_model(forward_pc_refresh_policy="auto")
    monkeypatch.setattr(forward_module, "PETSc", _FakePETSc)

    patterns = np.array([[1.0]], dtype=float)
    model._solve_with_petsc(sigma=object(), pattern_matrix=patterns)
    model._solve_with_petsc(sigma=object(), pattern_matrix=patterns)
    diag = dict(model.get_backend_diagnostics())

    assert diag["forward_pc_session_reused"] is True
    assert diag["forward_pc_refresh_triggered"] is False
    assert diag["forward_pc_refresh_reason"] is None
    assert ksp.set_reuse_calls[-1] is True


def test_forward_ksp_session_as_bundle_and_record_solve_roundtrip():
    session = ForwardKSPSession(
        ksp=object(),
        current_A=object(),
        current_solve_A=object(),
        backend_name="petsc-ksp",
        ksp_type="fgmres",
        pc_type="gamg",
        factor_solver_type=None,
        solve_mat_type="aij",
        structural_fingerprint="fp",
    )
    session.current_solve_A = session.current_A  # align with reuse check contract

    bundle_initial = session.as_bundle()
    assert bundle_initial["ksp_setup_count"] == 1
    assert bundle_initial["ksp_setup_attempts"] == 1

    session.record_solve(7)
    assert session.total_solves == 1
    assert session.solves_since_setup == 1
    assert session.last_iter_count == 7

    session.mark_reuse()
    bundle_reuse = session.as_bundle()
    assert bundle_reuse["ksp_setup_count"] == 0, "reuse must not count as new PC setup"

    session.mark_refresh("iter_count_12_gt_threshold_5")
    assert session.total_setups == 2
    telemetry = session.as_observability(
        cache_hit=True,
        session_reused=True,
        setup_seconds=0.125,
        rhs_count=16,
        rhs_kind="unit",
    )
    assert telemetry["schema"] == "pyeidors-forward-ksp-session-telemetry-v1"
    assert telemetry["rhs_count"] == 16
    assert telemetry["rhs_kind"] == "unit"
    assert telemetry["structural_fingerprint_short"] == "fp"
    assert session.solves_since_setup == 0
    bundle_refresh = session.as_bundle()
    assert bundle_refresh["ksp_setup_count"] == 1
    assert bundle_refresh["ksp_setup_attempts"] == 2
