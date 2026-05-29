"""Additional branch coverage for forward-model PETSc solve/runtime paths."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import pyeidors.forward.eit_forward_model as forward_module
from pyeidors.forward.eit_forward_model import EITForwardModel


class _FakeVec:
    def __init__(self, size: int, vec_type: str = "seq"):
        self.arr = np.zeros(int(size), dtype=float)
        self.vec_type = str(vec_type)

    def getArray(self, readonly=False):
        _ = readonly
        return self.arr

    def getType(self):
        return self.vec_type

    def setType(self, vec_type):
        self.vec_type = str(vec_type)


class _FakeMat:
    def __init__(self, size=(4, 4), mat_type: str = "seqaij", csr=None):
        self.size = tuple(int(v) for v in size)
        self.mat_type = str(mat_type)
        self.csr = csr
        self.values = []
        self.set_values = []
        self.options = []
        self.assembled = 0
        self.axpy_calls = []
        self.destroyed = False
        self.dense_array = None

    def getValuesCSR(self):
        if self.csr is not None:
            return self.csr
        matrix = csr_matrix(np.eye(self.size[0], self.size[1], dtype=float))
        return matrix.indptr, matrix.indices, matrix.data

    def getSize(self):
        return self.size

    def createAIJ(self, size, csr, comm=None):
        _ = comm
        return _FakeMat(size=size, csr=csr)

    def createDense(self, size, array=None, comm=None):
        _ = comm
        out = _FakeMat(size=size, mat_type="dense")
        if array is None:
            out.dense_array = np.zeros(size, dtype=float, order="F")
        else:
            out.dense_array = np.array(array, dtype=float, copy=True, order="F")
        return out

    def getDenseArray(self):
        return self.dense_array

    def setValues(self, row, cols, vals):
        if np.isscalar(row):
            rows = (int(row),)
        else:
            rows = tuple(int(r) for r in np.asarray(row).reshape(-1))
        if np.isscalar(cols):
            cols_tuple = (int(cols),)
        else:
            cols_tuple = tuple(int(c) for c in np.asarray(cols).reshape(-1))
        vals_tuple = tuple(float(v) for v in np.asarray(vals).reshape(-1))
        self.set_values.append((rows, cols_tuple, vals_tuple))

    def setValue(self, row, col, value):
        self.values.append((int(row), int(col), float(value)))

    def setOption(self, option, value):
        self.options.append((option, value))

    def assemblyBegin(self):
        self.assembled += 1

    def assemblyEnd(self):
        self.assembled += 1

    def assemble(self):
        self.assembled += 1

    def getType(self):
        return self.mat_type

    def setType(self, mat_type):
        self.mat_type = str(mat_type)

    def copy(self):
        copied = _FakeMat(size=self.size, mat_type=self.mat_type, csr=self.csr)
        copied.dense_array = (
            None
            if self.dense_array is None
            else np.array(self.dense_array, copy=True, order="F")
        )
        return copied

    def axpy(self, alpha, other, structure=None):
        self.axpy_calls.append((float(alpha), other, structure))

    def createVecRight(self):
        return _FakeVec(self.size[0])

    def destroy(self):
        self.destroyed = True


class _FakePC:
    def __init__(self):
        self.pc_type = None
        self.factor_solver_type = None

    def setType(self, pc_type):
        self.pc_type = str(pc_type)

    def getType(self):
        return self.pc_type

    def setFactorSolverType(self, solver_type):
        self.factor_solver_type = str(solver_type)


class _FakeKSP:
    def __init__(self, fail_setup=False, fail_factor_types=None):
        self.fail_setup = bool(fail_setup)
        self.fail_factor_types = set(fail_factor_types or [])
        self.pc = _FakePC()
        self.ksp_type = None
        self.tolerances = None
        self.reuse = None
        self.monitor = None
        self.A = None
        self.mat_solve_calls = 0
        self.solve_calls = 0
        self.converged_reason = 1
        self.solve_value = 3.0
        self.mat_solve_value = 2.0
        self.raise_on_matsolve = False

    def create(self, _comm):
        return self

    def setOperators(self, A):
        self.A = A

    def setType(self, ksp_type):
        self.ksp_type = str(ksp_type)

    def getType(self):
        return self.ksp_type

    def getPC(self):
        return self.pc

    def setTolerances(self, **kwargs):
        self.tolerances = dict(kwargs)

    def setReusePreconditioner(self, enabled):
        self.reuse = bool(enabled)

    def setMonitor(self, monitor):
        self.monitor = monitor

    def setUp(self):
        if self.fail_setup or self.pc.factor_solver_type in self.fail_factor_types:
            raise RuntimeError("setup failed")

    def matSolve(self, B, X):
        self.mat_solve_calls += 1
        if self.raise_on_matsolve:
            raise RuntimeError("matSolve failed")
        X.dense_array[:, :] = B.dense_array * self.mat_solve_value

    def solve(self, b, x):
        self.solve_calls += 1
        x.arr[:] = b.arr + self.solve_value

    def getConvergedReason(self):
        return self.converged_reason


class _FakePETSc:
    class Mat(_FakeMat):
        class Option:
            NEW_NONZERO_ALLOCATION_ERR = "new_nonzero_allocation_err"

        class Structure:
            DIFFERENT_NONZERO_PATTERN = "different"


def _make_model(**overrides):
    model = EITForwardModel.__new__(EITForwardModel)
    model.cache_manager = None
    model.dofs = 2
    model.n_elec = 2
    model.mesh = SimpleNamespace(comm=None)
    model.mesh_tdim = 3
    model.performance_mode = "aggressive"
    model.backend_config = SimpleNamespace(
        ksp_type="preonly",
        pc_type="lu",
        rtol=1e-10,
        atol=1e-12,
        max_it=200,
        reuse_preconditioner=True,
        monitor=False,
        mat_solve_mode="auto",
        use_mat_solve=False,
        petsc_device="auto",
    )
    model._petsc_backend_info = {"petsc_device_effective": "cpu", "capability": {}}
    model._M_petsc = {}
    model.z = np.array([1.0, 2.0], dtype=float)
    model.electrode_lengths_m = np.array([0.5, 0.8], dtype=float)
    model.electrode_tags = [2, 3]
    model.boundary_scale_to_m = 1.0
    model.u = 1.0
    model.phi = 1.0
    model.ds_electrodes = lambda _tag: 1.0
    model.pattern_manager = SimpleNamespace(
        stim_matrix=np.array([[1.0, -1.0], [0.5, -0.5]], dtype=float),
        n_stim=2,
        n_meas_total=3,
        apply_meas_pattern=lambda U: np.sum(U, axis=1),
    )
    model.linear_backend = "petsc"
    model.forward_backend = "dolfinx"
    model._last_cache_lookup = {}
    model._set_backend_diagnostic = EITForwardModel._set_backend_diagnostic.__get__(
        model, EITForwardModel
    )
    model.get_backend_diagnostics = EITForwardModel.get_backend_diagnostics.__get__(
        model, EITForwardModel
    )
    for key, value in overrides.items():
        setattr(model, key, value)
    return model


def test_electrode_matrix_petsc_helpers_cover_none_cache_and_full_expansion(
    monkeypatch: pytest.MonkeyPatch,
):
    model = _make_model()
    monkeypatch.setattr(forward_module, "PETSc", None)
    with pytest.raises(RuntimeError, match="petsc4py is not available"):
        EITForwardModel._assemble_electrode_matrix_petsc(model)
    with pytest.raises(RuntimeError, match="petsc4py is not available"):
        EITForwardModel._get_electrode_matrix_petsc(model)

    monkeypatch.setattr(forward_module, "PETSc", _FakePETSc)
    monkeypatch.setattr(forward_module.ufl, "inner", lambda _a, _b: 1.0)
    monkeypatch.setattr(forward_module.fem, "form", lambda expr: expr)

    top_left = _FakeMat(size=(2, 2))
    full_matrix = _FakeMat(size=(5, 5))
    vec1 = _FakeVec(2)
    vec1.array = np.array([1.0, 0.0], dtype=float)
    vec1.destroy = lambda: setattr(vec1, "destroyed", True)
    vec2 = _FakeVec(2)
    vec2.array = np.array([0.0, 2.0], dtype=float)
    vec2.destroy = lambda: setattr(vec2, "destroyed", True)
    monkeypatch.setattr(
        model, "_assemble_form_matrix", lambda _form, mat_kind=None: top_left
    )
    monkeypatch.setattr(
        model,
        "_expand_conductivity_csr_to_full",
        lambda _mat, mat_type=None: full_matrix,
    )
    monkeypatch.setattr(
        model,
        "_assemble_form_vector",
        lambda _form, vec_kind=None: vec1 if not hasattr(model, "_seen_vec") else vec2,
    )
    monkeypatch.setattr(
        model, "_vec_to_numpy", lambda vec: np.asarray(vec.array, dtype=float)
    )
    monkeypatch.setattr(model, "_ensure_mat_type", lambda mat, _kind: mat)
    model._seen_vec = False

    def _assemble_vec(_form, vec_kind=None):
        if not model._seen_vec:
            model._seen_vec = True
            return vec1
        return vec2

    monkeypatch.setattr(model, "_assemble_form_vector", _assemble_vec)
    out = EITForwardModel._assemble_electrode_matrix_petsc(
        model, mat_type="aij", vec_type="seq"
    )
    assert out is full_matrix
    assert any(item[0] == (model.dofs,) for item in full_matrix.set_values)
    assert any(item[0] == (model.dofs + 1,) for item in full_matrix.set_values)
    assert any(item[1] == (model.dofs,) for item in full_matrix.set_values)
    assert any(item[1] == (model.dofs + 1,) for item in full_matrix.set_values)
    assert (model.dofs, model.dofs, 0.5) in full_matrix.values
    assert (model.dofs + 1, model.dofs + 1, 0.4) in full_matrix.values
    assert full_matrix.assembled >= 2
    assert getattr(vec1, "destroyed", False) is True
    assert getattr(vec2, "destroyed", False) is True

    csr = csr_matrix(np.array([[1.0, 2.0], [0.0, 3.0]], dtype=float))
    conductivity = _FakeMat(size=(2, 2), csr=(csr.indptr, csr.indices, csr.data))
    expanded = EITForwardModel._expand_conductivity_csr_to_full(
        model, conductivity, mat_type="cuda"
    )
    assert expanded.size == (5, 5)
    assert expanded.assembled >= 2

    base_electrode = _FakeMat(size=(5, 5))
    monkeypatch.setattr(model, "_csr_to_petsc", lambda _csr: base_electrode)
    monkeypatch.setattr(
        model, "_ensure_electrode_matrix", lambda: csr_matrix(np.eye(5, dtype=float))
    )
    cached_first = EITForwardModel._get_electrode_matrix_petsc(model, mat_type="cuda")
    cached_second = EITForwardModel._get_electrode_matrix_petsc(model, mat_type="cuda")
    assert cached_first is cached_second
    assert any(
        value[:2] == (model.dofs + model.n_elec, model.dofs + model.n_elec)
        for value in base_electrode.values
    )


def test_v417_electrode_coupling_nonzero_arrays_direct_fill():
    indices, values = forward_module._nonzero_index_value_arrays(
        np.array([0.0, 2.0, np.nan, 0.0, -3.0], dtype=np.float64)
    )

    np.testing.assert_array_equal(indices, np.array([1, 2, 4], dtype=np.int32))
    np.testing.assert_allclose(
        values,
        np.array([2.0, np.nan, -3.0], dtype=np.float64),
        equal_nan=True,
    )
    complex_indices, complex_values = forward_module._nonzero_index_value_arrays(
        np.array([0.0 + 0.0j, 1.0 - 2.0j], dtype=np.complex128)
    )
    np.testing.assert_array_equal(complex_indices, np.array([1], dtype=np.int32))
    np.testing.assert_allclose(
        complex_values,
        np.array([1.0 - 2.0j], dtype=np.complex128),
    )

    helper_source = inspect.getsource(forward_module._nonzero_index_value_arrays)
    assemble_source = inspect.getsource(
        EITForwardModel._assemble_electrode_matrix_petsc
    )
    assert "np.flatnonzero" not in assemble_source
    assert "c_i[nz]" not in assemble_source
    assert "np.nditer" in helper_source
    assert "out_values = np.empty" in helper_source


def test_electrode_matrix_petsc_helpers_cover_destroy_and_ground_value_exceptions(
    monkeypatch: pytest.MonkeyPatch,
):
    model = _make_model()
    monkeypatch.setattr(forward_module, "PETSc", _FakePETSc)
    monkeypatch.setattr(forward_module.ufl, "inner", lambda _a, _b: 1.0)
    monkeypatch.setattr(forward_module.fem, "form", lambda expr: expr)

    top_left = _FakeMat(size=(2, 2))
    full_matrix = _FakeMat(size=(5, 5))

    class _ExplodingVec(_FakeVec):
        def destroy(self):
            raise RuntimeError("destroy failed")

    vec = _ExplodingVec(2)
    vec.arr[:] = np.array([1.0, 0.0], dtype=float)
    monkeypatch.setattr(
        model, "_assemble_form_matrix", lambda _form, mat_kind=None: top_left
    )
    monkeypatch.setattr(
        model,
        "_expand_conductivity_csr_to_full",
        lambda _mat, mat_type=None: full_matrix,
    )
    monkeypatch.setattr(
        model, "_assemble_form_vector", lambda _form, vec_kind=None: vec
    )
    monkeypatch.setattr(
        model, "_vec_to_numpy", lambda _vec: np.asarray(vec.arr, dtype=float)
    )
    monkeypatch.setattr(model, "_ensure_mat_type", lambda mat, _kind: mat)

    out = EITForwardModel._assemble_electrode_matrix_petsc(
        model, mat_type="aij", vec_type="seq"
    )
    assert out is full_matrix
    assert full_matrix.assembled >= 2

    class _ExplodingGroundMat(_FakeMat):
        def setValue(self, row, col, value):
            raise RuntimeError("ground failed")

    ground_mat = _ExplodingGroundMat(size=(5, 5))
    monkeypatch.setattr(model, "_csr_to_petsc", lambda _csr: ground_mat)
    monkeypatch.setattr(
        model, "_ensure_electrode_matrix", lambda: csr_matrix(np.eye(5, dtype=float))
    )
    model._M_petsc = {}
    out_ground = EITForwardModel._get_electrode_matrix_petsc(model, mat_type="cuda")
    assert out_ground is ground_mat


def test_create_full_matrix_petsc_predict_payload_and_scipy_solver_cover_branches(
    monkeypatch: pytest.MonkeyPatch,
):
    model = _make_model()
    monkeypatch.setattr(forward_module, "PETSc", _FakePETSc)

    with pytest.raises(RuntimeError, match="petsc4py is not available"):
        monkeypatch.setattr(forward_module, "PETSc", None)
        EITForwardModel._create_full_matrix_petsc(model, sigma=None)

    monkeypatch.setattr(forward_module, "PETSc", _FakePETSc)
    monkeypatch.setattr(model, "_get_requested_petsc_mat_type", lambda: "aijcusparse")
    monkeypatch.setattr(model, "_ensure_mat_type", lambda mat, _kind: mat)
    monkeypatch.setattr(model, "_sigma_fingerprint", lambda _sigma: "sigma")

    model._petsc_backend_info = {"petsc_device_effective": "cuda", "capability": {}}
    monkeypatch.setattr(model, "_gpu_gauge_fix_enabled", lambda: True)
    monkeypatch.setattr(
        model,
        "_create_full_matrix_scipy",
        lambda _sigma: csr_matrix(np.eye(5, dtype=float)),
    )
    gpu_mat = _FakeMat(size=(5, 5), mat_type="aijcusparse")
    monkeypatch.setattr(model, "_csr_to_petsc", lambda _csr: gpu_mat)
    out_gpu = EITForwardModel._create_full_matrix_petsc(model, sigma=None)
    assert out_gpu is gpu_mat
    assert (
        model.get_backend_diagnostics()["gpu_constraint_strategy"]
        == "reference-electrode-row"
    )

    model._petsc_backend_info = {"petsc_device_effective": "cpu", "capability": {}}
    monkeypatch.setattr(model, "_gpu_gauge_fix_enabled", lambda: False)
    cond_mat = _FakeMat(size=(2, 2))
    aug_mat = _FakeMat(size=(5, 5))
    aug_mat.destroy = lambda: setattr(aug_mat, "destroyed", True)
    electrode_mat = _FakeMat(size=(5, 5))
    monkeypatch.setattr(
        model, "_assemble_conductivity_matrix", lambda _sigma, mat_kind=None: cond_mat
    )
    monkeypatch.setattr(
        model, "_expand_conductivity_csr_to_full", lambda _mat, mat_type=None: aug_mat
    )
    monkeypatch.setattr(
        model, "_get_electrode_matrix_petsc", lambda mat_type=None: electrode_mat
    )
    called = {"struct_diag": 0}
    monkeypatch.setattr(
        model,
        "_ensure_structural_diagonal",
        lambda _mat: called.__setitem__("struct_diag", called["struct_diag"] + 1),
    )
    out_cpu = EITForwardModel._create_full_matrix_petsc(model, sigma=None)
    assert out_cpu is not electrode_mat
    assert out_cpu.axpy_calls
    assert called["struct_diag"] == 1
    assert getattr(aug_mat, "destroyed", False) is True

    model.mesh_tdim = 2
    model.performance_mode = "safe"
    model.backend_config = SimpleNamespace(
        mat_solve_mode="weird", use_mat_solve=True, petsc_device="auto"
    )
    assert EITForwardModel._predict_forward_mat_solve_effective(model, 1) == "matsolve"
    model.backend_config = SimpleNamespace(
        mat_solve_mode="auto", use_mat_solve=False, petsc_device="auto"
    )
    assert EITForwardModel._predict_forward_mat_solve_effective(model, 1) == "vec-loop"
    model.mesh_tdim = 3
    model.performance_mode = "aggressive"
    assert EITForwardModel._predict_forward_mat_solve_effective(model, 2) == "matsolve"
    model._petsc_backend_info = {
        "petsc_device_effective": "cuda",
        "capability": {"petsc_cuda_dense": True},
    }
    assert EITForwardModel._predict_forward_mat_solve_effective(model, 2) == "matsolve"
    model._petsc_backend_info = {
        "petsc_device_effective": "cuda",
        "capability": {"petsc_cuda_dense": False},
    }
    assert EITForwardModel._predict_forward_mat_solve_effective(model, 2) == "vec-loop"

    model.backend_config = SimpleNamespace(
        ksp_type="preonly",
        pc_type="lu",
        rtol=1e-10,
        atol=1e-12,
        max_it=200,
        reuse_preconditioner=True,
        monitor=False,
        mat_solve_mode="auto",
        use_mat_solve=False,
        petsc_device="auto",
    )
    model._petsc_backend_info = {"petsc_device_effective": "cpu"}
    monkeypatch.setattr(model, "_stable_cpu_petsc_types", lambda: ("seqaij", "seq"))
    payload = EITForwardModel._base_cache_payload(model, sigma_hash="abc", n_patterns=2)
    assert payload["petsc_backend"]["mat_type"] == "seqaij"
    assert payload["petsc_backend"]["vec_type"] == "seq"

    class _FakeLU:
        def solve(self, rhs):
            return np.asarray(rhs, dtype=float) + 1.0

    class _Lookup:
        key = "lu-key"
        hit = True
        layer = "memory"
        artifact = "forward_factor"

    class _Cache:
        enabled = True

        def get_or_compute(self, **kwargs):
            return _FakeLU(), _Lookup()

    model.cache_manager = _Cache()
    monkeypatch.setattr(
        model,
        "_create_full_matrix_scipy",
        lambda _sigma: csr_matrix(np.eye(5, dtype=float)),
    )
    monkeypatch.setattr(model, "_apply_cuda_gauge_fix_rhs", lambda rhs: rhs)
    out_cached = EITForwardModel._solve_with_scipy(
        model, sigma=None, pattern_matrix=np.array([[1.0, -1.0]], dtype=float)
    )
    assert out_cached.shape == (5, 1)
    assert model._last_cache_lookup["hit"] is True

    model.cache_manager = None
    out_disabled = EITForwardModel._solve_with_scipy(
        model, sigma=None, pattern_matrix=np.array([[1.0, -1.0]], dtype=float)
    )
    assert out_disabled.shape == (5, 1)
    assert model._last_cache_lookup["layer"] == "disabled"


def test_make_petsc_solver_bundle_covers_direct_dense_and_gmres_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    model = _make_model()
    monkeypatch.setattr(forward_module, "PETSc", None)
    with pytest.raises(RuntimeError, match="petsc4py is required"):
        EITForwardModel._make_petsc_solver_bundle(
            model, csr_matrix(np.eye(2, dtype=float))
        )

    converted = _FakeMat(size=(5, 5))
    monkeypatch.setattr(forward_module, "PETSc", _FakePETSc)
    monkeypatch.setattr(model, "_csr_to_petsc", lambda _csr: converted)

    ksp_success = _FakeKSP(fail_factor_types={"cuda"})
    monkeypatch.setattr(
        forward_module,
        "PETSc",
        SimpleNamespace(Mat=_FakePETSc.Mat, KSP=lambda: ksp_success),
    )
    model._petsc_backend_info = {"petsc_device_effective": "cuda"}
    bundle_direct = EITForwardModel._make_petsc_solver_bundle(
        model, csr_matrix(np.eye(5, dtype=float))
    )
    assert bundle_direct["backend"] == "petsc-ksp-cusparse-lu"
    assert bundle_direct["factor_solver_type"] == "cusparse"

    class _KSPFactory:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            if self.calls <= 2:
                return _FakeKSP(fail_factor_types={"cusparse", "cuda"})
            return _FakeKSP()

    dense_factory = _KSPFactory()
    monkeypatch.setattr(
        forward_module, "PETSc", SimpleNamespace(Mat=_FakePETSc.Mat, KSP=dense_factory)
    )
    monkeypatch.setattr(model, "_get_requested_dense_mat_type", lambda: "densecuda")
    monkeypatch.setattr(
        model, "_ensure_mat_type", lambda mat, mat_type: (mat.setType(mat_type), mat)[1]
    )
    bundle_dense = EITForwardModel._make_petsc_solver_bundle(model, converted)
    assert bundle_dense["backend"] == "petsc-ksp-densecuda-lu"
    assert bundle_dense["solve_mat_type"] == "densecuda"

    gmres_factory = _KSPFactory()
    monkeypatch.setattr(
        forward_module, "PETSc", SimpleNamespace(Mat=_FakePETSc.Mat, KSP=gmres_factory)
    )
    model._petsc_backend_info = {"petsc_device_effective": "cpu"}
    monkeypatch.setattr(model, "_get_requested_dense_mat_type", lambda: None)

    def _bad_config_ksp():
        if gmres_factory.calls == 0:
            gmres_factory.calls += 1
            return _FakeKSP(fail_setup=True)
        gmres_factory.calls += 1
        return _FakeKSP()

    monkeypatch.setattr(
        forward_module,
        "PETSc",
        SimpleNamespace(Mat=_FakePETSc.Mat, KSP=_bad_config_ksp),
    )
    bundle_gmres = EITForwardModel._make_petsc_solver_bundle(model, converted)
    assert bundle_gmres["backend"] == "petsc-ksp-gmres+none"
    assert bundle_gmres["ksp_type"] == "gmres"


def test_make_petsc_solver_bundle_covers_reuse_monitor_and_cuda_dense_setup_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    model = _make_model()
    model.backend_config = SimpleNamespace(
        ksp_type="preonly",
        pc_type="lu",
        rtol=1e-10,
        atol=1e-12,
        max_it=200,
        reuse_preconditioner=True,
        monitor=True,
        mat_solve_mode="auto",
        use_mat_solve=False,
        petsc_device="auto",
    )

    class _ReuseFailKSP(_FakeKSP):
        def setReusePreconditioner(self, enabled):
            _ = enabled
            raise RuntimeError("reuse failed")

    monkeypatch.setattr(
        forward_module,
        "PETSc",
        SimpleNamespace(Mat=_FakePETSc.Mat, KSP=lambda: _ReuseFailKSP()),
    )
    model._petsc_backend_info = {"petsc_device_effective": "cpu"}
    bundle = EITForwardModel._make_petsc_solver_bundle(model, _FakeMat(size=(5, 5)))
    assert callable(bundle["ksp"].monitor)
    assert bundle["ksp_setup_count"] == 1
    assert bundle["reuse_preconditioner"] is True
    assert bundle["reuse_preconditioner_applied"] is False
    bundle["ksp"].monitor(bundle["ksp"], 1, 1e-3)
    assert "[KSP] iter=1" in capsys.readouterr().out

    class _Factory:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            if self.calls <= 3:
                return _FakeKSP(fail_setup=True)
            return _FakeKSP()

    factory = _Factory()
    monkeypatch.setattr(
        forward_module, "PETSc", SimpleNamespace(Mat=_FakePETSc.Mat, KSP=factory)
    )
    model._petsc_backend_info = {"petsc_device_effective": "cuda"}
    monkeypatch.setattr(model, "_get_requested_dense_mat_type", lambda: "densecuda")
    monkeypatch.setattr(
        model, "_ensure_mat_type", lambda mat, mat_type: (mat.setType(mat_type), mat)[1]
    )
    bundle_gmres = EITForwardModel._make_petsc_solver_bundle(
        model, _FakeMat(size=(5, 5))
    )
    assert bundle_gmres["backend"] == "petsc-ksp-gmres+none"
    assert bundle_gmres["ksp_type"] == "gmres"
    assert bundle_gmres["ksp_setup_count"] == 4
    assert bundle_gmres["reuse_preconditioner"] is True


def test_solve_with_petsc_and_forward_interfaces_cover_mat_solve_fallbacks_and_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    model = _make_model()
    model._set_backend_diagnostic = EITForwardModel._set_backend_diagnostic.__get__(
        model, EITForwardModel
    )
    monkeypatch.setattr(forward_module, "PETSc", _FakePETSc)
    monkeypatch.setattr(model, "_sigma_fingerprint", lambda _sigma: "sig")
    monkeypatch.setattr(model, "_base_cache_payload", lambda **_kwargs: {})
    monkeypatch.setattr(model, "_apply_cuda_gauge_fix_rhs", lambda rhs: rhs)
    monkeypatch.setattr(model, "_get_requested_dense_mat_type", lambda: "densecuda")
    monkeypatch.setattr(model, "_ensure_mat_type", lambda obj, _kind: obj)
    monkeypatch.setattr(model, "_get_requested_petsc_vec_type", lambda: "cuda")
    monkeypatch.setattr(model, "_recenter_cuda_gauge_solution", lambda sol: sol + 10.0)
    monkeypatch.setattr(
        model, "_create_full_matrix_petsc", lambda _sigma: _FakeMat(size=(5, 5))
    )

    matsolve_ksp = _FakeKSP()
    matsolve_bundle = {
        "A": _FakeMat(size=(5, 5), mat_type="aij"),
        "solve_A": _FakeMat(size=(5, 5), mat_type="densecuda"),
        "ksp": matsolve_ksp,
        "backend": "petsc-ksp-densecuda-lu",
        "solve_mat_type": "densecuda",
    }
    monkeypatch.setattr(model, "_make_petsc_solver_bundle", lambda _A: matsolve_bundle)
    model._petsc_backend_info = {
        "petsc_device_requested": "auto",
        "petsc_device_effective": "cuda",
        "capability": {"petsc_cuda_dense": True},
    }
    sol = EITForwardModel._solve_with_petsc(
        model,
        sigma=None,
        pattern_matrix=np.array([[1.0, -1.0], [0.5, -0.5]], dtype=float),
    )
    assert sol.shape == (5, 2)
    assert matsolve_ksp.mat_solve_calls == 1
    assert model.get_backend_diagnostics()["forward_mat_solve_effective"] == "matsolve"

    failing_ksp = _FakeKSP()
    failing_ksp.raise_on_matsolve = True
    failing_bundle = dict(matsolve_bundle, ksp=failing_ksp, backend="petsc-ksp")
    dense_ksp = _FakeKSP()
    monkeypatch.setattr(model, "_make_petsc_solver_bundle", lambda _A: failing_bundle)
    monkeypatch.setattr(
        model,
        "_make_petsc_dense_solver_bundle",
        lambda _A: {
            "A": _FakeMat(size=(5, 5)),
            "solve_A": _FakeMat(size=(5, 5), mat_type="densecuda"),
            "ksp": dense_ksp,
            "backend": "petsc-ksp-densecuda-lu",
            "solve_mat_type": "densecuda",
        },
    )
    sol_fallback = EITForwardModel._solve_with_petsc(
        model, sigma=None, pattern_matrix=np.array([[1.0, -1.0]], dtype=float)
    )
    assert sol_fallback.shape == (5, 1)
    assert dense_ksp.mat_solve_calls == 1
    assert "matSolve_fallback" in str(
        model.get_backend_diagnostics()["gpu_fallback_reason"]
    )

    raising_ksp = _FakeKSP()
    raising_ksp.raise_on_matsolve = True
    monkeypatch.setattr(
        model,
        "_make_petsc_solver_bundle",
        lambda _A: {
            "A": _FakeMat(size=(5, 5), mat_type="aij"),
            "solve_A": _FakeMat(size=(5, 5), mat_type="densecuda"),
            "ksp": raising_ksp,
            "backend": "petsc-ksp",
            "solve_mat_type": "densecuda",
        },
    )
    monkeypatch.setattr(
        model,
        "_make_petsc_dense_solver_bundle",
        lambda _A: (_ for _ in ()).throw(RuntimeError("dense fail")),
    )
    model._petsc_backend_info = {
        "petsc_device_requested": "cuda",
        "petsc_device_effective": "cuda",
        "capability": {"petsc_cuda_dense": True},
    }
    with pytest.raises(RuntimeError, match="PETSc CUDA matSolve failed"):
        EITForwardModel._solve_with_petsc(
            model, sigma=None, pattern_matrix=np.array([[1.0, -1.0]], dtype=float)
        )

    vec_ksp = _FakeKSP()
    vec_ksp.converged_reason = -7
    vec_bundle = {
        "A": _FakeMat(size=(5, 5), mat_type="aij"),
        "solve_A": _FakeMat(size=(5, 5), mat_type="aij"),
        "ksp": vec_ksp,
        "backend": "petsc-ksp",
        "solve_mat_type": "aij",
    }
    monkeypatch.setattr(model, "_make_petsc_solver_bundle", lambda _A: vec_bundle)
    monkeypatch.setattr(
        model,
        "_solve_with_scipy",
        lambda _sigma, _pattern: np.full((5, 1), 9.0, dtype=float),
    )
    model._petsc_backend_info = {
        "petsc_device_requested": "auto",
        "petsc_device_effective": "cpu",
        "capability": {},
    }
    vec_sol = EITForwardModel._solve_with_petsc(
        model, sigma=None, pattern_matrix=np.array([[1.0, -1.0]], dtype=float)
    )
    assert vec_sol.shape == (5, 1)
    assert np.allclose(vec_sol, 9.0)

    model._petsc_backend_info = {
        "petsc_device_requested": "auto",
        "petsc_device_effective": "cpu",
        "capability": {},
    }
    monkeypatch.setattr(
        model,
        "_solve_with_petsc",
        lambda sigma, pattern_matrix: np.arange(10, dtype=float).reshape(5, 2),
    )
    monkeypatch.setattr(
        model,
        "_solve_with_scipy",
        lambda sigma, pattern_matrix: np.arange(10, dtype=float).reshape(5, 2),
    )
    sigma = SimpleNamespace(x=SimpleNamespace(array=np.array([1.0, 2.0], dtype=float)))
    model.linear_backend = "petsc"
    u_all, U_all = EITForwardModel.forward_solve(model, sigma)
    assert len(u_all) == 2
    assert U_all.shape == (2, 2)

    model.forward_backend = "cuda_structured"
    model._cuda_structured_backend = None
    with pytest.raises(RuntimeError, match="was not initialized"):
        EITForwardModel.forward_solve(model, sigma)
    model._cuda_structured_backend = SimpleNamespace(
        backend_diagnostics=lambda: {"operator_backend": "cuda_structured"},
        solve_batch=lambda _sigma, _patterns: (
            ("u0",),
            np.array([[1.0, 2.0]], dtype=float),
        ),
    )
    out_cuda = EITForwardModel.forward_solve(model, sigma)
    assert out_cuda[1].shape == (1, 2)

    model.forward_backend = "dolfinx"
    model.linear_backend = "other"
    with pytest.raises(ValueError, match="Unsupported linear_backend"):
        EITForwardModel.forward_solve(model, sigma)

    class _FakeFemFunction:
        def __init__(self, _space):
            self.x = SimpleNamespace(array=np.zeros(2, dtype=float))

    monkeypatch.setattr(forward_module.fem, "Function", _FakeFemFunction)
    model.linear_backend = "petsc"
    model.V_sigma = object()
    model.forward_solve = lambda _sigma: (
        ("u0",),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
    )
    model.pattern_manager = SimpleNamespace(
        apply_meas_pattern=lambda U: np.sum(U, axis=1),
        stim_matrix=np.array([[1.0, -1.0]], dtype=float),
        n_stim=1,
        n_meas_total=2,
    )
    data, U = EITForwardModel.fwd_solve(
        model,
        SimpleNamespace(get_conductivity=lambda: np.array([5.0, 6.0], dtype=float)),
    )
    np.testing.assert_allclose(data.meas, np.array([3.0, 7.0], dtype=float))
    assert data.type == "simulated"
    assert U.shape == (2, 2)


def test_solve_with_petsc_covers_cuda_dense_unavailable_dense_fallback_failure_and_cuda_ksp_error(
    monkeypatch: pytest.MonkeyPatch,
):
    model = _make_model()
    model._set_backend_diagnostic = EITForwardModel._set_backend_diagnostic.__get__(
        model, EITForwardModel
    )
    monkeypatch.setattr(forward_module, "PETSc", _FakePETSc)
    monkeypatch.setattr(model, "_sigma_fingerprint", lambda _sigma: "sig")
    monkeypatch.setattr(model, "_base_cache_payload", lambda **_kwargs: {})
    monkeypatch.setattr(model, "_apply_cuda_gauge_fix_rhs", lambda rhs: rhs)
    monkeypatch.setattr(model, "_get_requested_dense_mat_type", lambda: "densecuda")
    monkeypatch.setattr(model, "_ensure_mat_type", lambda obj, _kind: obj)
    monkeypatch.setattr(model, "_get_requested_petsc_vec_type", lambda: None)
    monkeypatch.setattr(
        model, "_create_full_matrix_petsc", lambda _sigma: _FakeMat(size=(5, 5))
    )

    vec_loop_bundle = {
        "A": _FakeMat(size=(5, 5), mat_type="aij"),
        "solve_A": _FakeMat(size=(5, 5), mat_type="aij"),
        "ksp": _FakeKSP(),
        "backend": "petsc-ksp",
        "solve_mat_type": "aij",
    }
    monkeypatch.setattr(model, "_make_petsc_solver_bundle", lambda _A: vec_loop_bundle)
    monkeypatch.setattr(model, "_should_use_mat_solve", lambda _n: True)
    model._petsc_backend_info = {
        "petsc_device_requested": "auto",
        "petsc_device_effective": "cuda",
        "capability": {"petsc_cuda_dense": False},
    }
    sol_unavailable = EITForwardModel._solve_with_petsc(
        model,
        sigma=None,
        pattern_matrix=np.array([[1.0, -1.0], [0.5, -0.5]], dtype=float),
    )
    assert sol_unavailable.shape == (5, 2)
    assert (
        model.get_backend_diagnostics()["gpu_fallback_reason"]
        == "petsc_densecuda_unavailable"
    )

    failing_ksp = _FakeKSP()
    failing_ksp.raise_on_matsolve = True
    dense_failing_ksp = _FakeKSP()
    dense_failing_ksp.raise_on_matsolve = True
    mat_fail_bundle = {
        "A": _FakeMat(size=(5, 5), mat_type="aij"),
        "solve_A": _FakeMat(size=(5, 5), mat_type="densecuda"),
        "ksp": failing_ksp,
        "backend": "petsc-ksp",
        "solve_mat_type": "densecuda",
    }
    monkeypatch.setattr(model, "_make_petsc_solver_bundle", lambda _A: mat_fail_bundle)
    monkeypatch.setattr(
        model,
        "_make_petsc_dense_solver_bundle",
        lambda _A: {
            "A": _FakeMat(size=(5, 5)),
            "solve_A": _FakeMat(size=(5, 5), mat_type="densecuda"),
            "ksp": dense_failing_ksp,
            "backend": "petsc-ksp-densecuda-lu",
            "solve_mat_type": "densecuda",
        },
    )
    model._petsc_backend_info = {
        "petsc_device_requested": "auto",
        "petsc_device_effective": "cuda",
        "capability": {"petsc_cuda_dense": True},
    }
    sol_fallback_fail = EITForwardModel._solve_with_petsc(
        model,
        sigma=None,
        pattern_matrix=np.array([[1.0, -1.0]], dtype=float),
    )
    assert sol_fallback_fail.shape == (5, 1)
    assert str(model.get_backend_diagnostics()["gpu_fallback_reason"]).startswith(
        "matSolve_failed:"
    )
    assert model.get_backend_diagnostics()["forward_mat_solve_effective"] == "vec-loop"

    bad_ksp = _FakeKSP()
    bad_ksp.converged_reason = -9
    monkeypatch.setattr(
        model,
        "_make_petsc_solver_bundle",
        lambda _A: {
            "A": _FakeMat(size=(5, 5), mat_type="aij"),
            "solve_A": _FakeMat(size=(5, 5), mat_type="aij"),
            "ksp": bad_ksp,
            "backend": "petsc-ksp",
            "solve_mat_type": "aij",
        },
    )
    monkeypatch.setattr(model, "_should_use_mat_solve", lambda _n: False)
    model._petsc_backend_info = {
        "petsc_device_requested": "auto",
        "petsc_device_effective": "cuda",
        "capability": {},
    }
    with pytest.raises(
        RuntimeError, match="PETSc CUDA solve failed with a negative convergence reason"
    ):
        EITForwardModel._solve_with_petsc(
            model,
            sigma=None,
            pattern_matrix=np.array([[1.0, -1.0]], dtype=float),
        )
