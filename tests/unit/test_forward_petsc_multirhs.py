"""PETSc multi-RHS solve branch tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import pyeidors.forward.eit_forward_model as forward_module
from pyeidors.forward.eit_forward_model import EITForwardModel


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
    def createVecRight(self):  # pragma: no cover - should not be used in this test
        raise RuntimeError("fallback vector path should not execute")


class _FakeKSP:
    def __init__(self):
        self.calls = 0

    def matSolve(self, B, X):
        self.calls += 1
        X.arr[:, :] = B.arr * 2.0


def test_solve_with_petsc_uses_mat_solve_when_available(monkeypatch):
    model = EITForwardModel.__new__(EITForwardModel)
    model.cache_manager = None
    model.dofs = 2
    model.n_elec = 1
    model.mesh = SimpleNamespace(comm=None)
    model._sigma_fingerprint = lambda sigma: "sigma-hash"
    model._base_cache_payload = lambda sigma_hash, n_patterns: {
        "sigma_hash": sigma_hash,
        "n_patterns": n_patterns,
    }
    ksp = _FakeKSP()
    model._create_full_matrix_petsc = lambda sigma: _FakeA()
    model._make_petsc_solver_bundle = lambda system_matrix: {
        "A": system_matrix,
        "ksp": ksp,
    }
    model._last_cache_lookup = {}
    monkeypatch.setattr(forward_module, "PETSc", _FakePETSc)

    pattern_matrix = np.array([[1.0], [2.0]], dtype=float)
    sol = model._solve_with_petsc(sigma=None, pattern_matrix=pattern_matrix)

    assert ksp.calls == 1
    assert sol.shape == (model.dofs + model.n_elec + 1, pattern_matrix.shape[0])
    # RHS rows [dofs:dofs+n_elec] are pattern values; matSolve multiplies by 2.
    assert np.allclose(sol[model.dofs, :], np.array([2.0, 4.0], dtype=float))
