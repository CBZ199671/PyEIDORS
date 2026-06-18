"""Additional branch coverage for PETSc/GPU helper logic in the forward model."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

import pyeidors.forward.eit_forward_model as forward_module
import pyeidors.perf.capabilities as perf_caps
from pyeidors.forward.eit_forward_model import EITForwardModel


class _ExplodingComm:
    def Get_size(self):
        raise RuntimeError("boom")


class _FakeMat:
    def __init__(self, mat_type: str = "seqaij"):
        self.mat_type = str(mat_type)
        self.destroyed = False
        self.assembled = 0
        self.set_type_calls: list[str] = []
        self.convert_result = None
        self.raise_on_convert = False

    def getType(self):
        return self.mat_type

    def convert(self, mat_type):
        if self.raise_on_convert:
            raise RuntimeError("convert failed")
        return self.convert_result

    def setType(self, mat_type):
        self.set_type_calls.append(str(mat_type))
        self.mat_type = str(mat_type)

    def destroy(self):
        self.destroyed = True

    def assemble(self):
        self.assembled += 1

    def copy(self):
        return _FakeMat(self.mat_type)


class _FakeVec:
    def __init__(self, vec_type: str = "seq"):
        self.vec_type = str(vec_type)
        self.set_type_calls: list[str] = []
        self.assembled = 0
        self.array = np.array([1.0, 2.0, 3.0], dtype=float)

    def getType(self):
        return self.vec_type

    def setType(self, vec_type):
        self.set_type_calls.append(str(vec_type))
        self.vec_type = str(vec_type)

    def assemble(self):
        self.assembled += 1


class _StructuralMat:
    def __init__(self, shape=(4, 4), fail_get_size: bool = False):
        self.shape = shape
        self.fail_get_size = fail_get_size
        self.options = []
        self.values = []

    def setOption(self, option, value):
        self.options.append((option, value))

    def getSize(self):
        if self.fail_get_size:
            raise RuntimeError("size failed")
        return self.shape

    def setValue(self, row, col, value):
        self.values.append((int(row), int(col), float(value)))


class _FakePC:
    def __init__(self):
        self.pc_type = None

    def setType(self, pc_type):
        self.pc_type = pc_type

    def getType(self):
        return self.pc_type


class _FakeKSPInstance:
    def __init__(self):
        self.solve_mat = None
        self.ksp_type = None
        self.pc = _FakePC()
        self.tolerances = None
        self.reuse = None
        self.did_setup = False

    def create(self, _comm):
        return self

    def setOperators(self, solve_mat):
        self.solve_mat = solve_mat

    def setType(self, ksp_type):
        self.ksp_type = ksp_type

    def getPC(self):
        return self.pc

    def setTolerances(self, **kwargs):
        self.tolerances = dict(kwargs)

    def setReusePreconditioner(self, enabled):
        self.reuse = bool(enabled)

    def setUp(self):
        self.did_setup = True

    def getType(self):
        return self.ksp_type


class _FakePETScDense:
    class KSP:
        def create(self, _comm):
            return _FakeKSPInstance()

    class Mat:
        class Option:
            NEW_NONZERO_ALLOCATION_ERR = "new_nonzero_allocation_err"


def _make_model(**overrides):
    model = EITForwardModel.__new__(EITForwardModel)
    model.mesh = SimpleNamespace(comm=SimpleNamespace(Get_size=lambda: 1))
    model.backend_config = SimpleNamespace(
        petsc_device="auto",
        ksp_type="preonly",
        pc_type="lu",
        rtol=1e-10,
        atol=1e-12,
        max_it=200,
        reuse_preconditioner=True,
    )
    model.linear_backend = "petsc"
    model.forward_backend = "dolfinx"
    model._petsc_backend_info = {"petsc_device_effective": "cpu"}
    model.dofs = 2
    model.n_elec = 2
    for key, value in overrides.items():
        setattr(model, key, value)
    return model


def test_apply_ksp_options_database_scopes_and_cleans_project_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeOptions(dict):
        def __init__(self):
            super().__init__()
            self.deleted: list[str] = []

        def delValue(self, key):
            self.deleted.append(str(key))
            self.pop(str(key), None)

    class _OptionsKSP:
        def __init__(self, options, *, fail: bool = False):
            self.options = options
            self.fail = bool(fail)
            self.prefix = ""
            self.snapshot: dict[str, str] = {}

        def setOptionsPrefix(self, prefix):
            self.prefix = str(prefix)

        def setFromOptions(self):
            self.snapshot = dict(self.options)
            if self.fail:
                raise RuntimeError("setFromOptions failed")

    model = _make_model()
    model.backend_config = SimpleNamespace(
        pc_type="gamg",
        pc_hypre_type="boomeramg",
        pc_gamg_type="agg",
        pc_factor_mat_solver_type="superlu_dist",
        petsc_options={"ksp_error_if_not_converged": True},
    )
    options = _FakeOptions()
    monkeypatch.setattr(
        forward_module,
        "PETSc",
        SimpleNamespace(Options=lambda: options),
    )
    ksp = _OptionsKSP(options)

    EITForwardModel._apply_ksp_options_database(model, ksp)

    assert ksp.snapshot == {
        f"{ksp.prefix}ksp_error_if_not_converged": "true",
        f"{ksp.prefix}pc_gamg_type": "agg",
    }
    assert f"{ksp.prefix}pc_hypre_type" not in ksp.snapshot
    assert f"{ksp.prefix}pc_factor_mat_solver_type" not in ksp.snapshot
    assert dict(options) == {}
    assert sorted(options.deleted) == sorted(ksp.snapshot)

    model.backend_config = SimpleNamespace(
        pc_type="hypre",
        pc_hypre_type="boomeramg",
        pc_gamg_type="agg",
        pc_factor_mat_solver_type=None,
        petsc_options={},
    )
    failing_options = _FakeOptions()
    monkeypatch.setattr(
        forward_module,
        "PETSc",
        SimpleNamespace(Options=lambda: failing_options),
    )
    failing_ksp = _OptionsKSP(failing_options, fail=True)

    with pytest.raises(RuntimeError, match="setFromOptions failed"):
        EITForwardModel._apply_ksp_options_database(model, failing_ksp)

    assert failing_ksp.snapshot == {f"{failing_ksp.prefix}pc_hypre_type": "boomeramg"}
    assert f"{failing_ksp.prefix}pc_gamg_type" not in failing_ksp.snapshot
    assert dict(failing_options) == {}
    assert failing_options.deleted == [f"{failing_ksp.prefix}pc_hypre_type"]


def test_stable_cpu_petsc_types_handles_none_mpi_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    model = _make_model()

    monkeypatch.setattr(forward_module, "PETSc", None)
    assert EITForwardModel._stable_cpu_petsc_types(model) == (None, None)

    fake_petsc = SimpleNamespace(
        Mat=SimpleNamespace(Type=SimpleNamespace(MPIAIJ="MATMPIAIJ", AIJ="MATAIJ")),
        Vec=SimpleNamespace(Type=SimpleNamespace(MPI="VECMPI")),
    )
    monkeypatch.setattr(forward_module, "PETSc", fake_petsc)
    model.mesh = SimpleNamespace(comm=SimpleNamespace(Get_size=lambda: 4))
    assert EITForwardModel._stable_cpu_petsc_types(model) == ("MATMPIAIJ", "VECMPI")

    model.mesh = SimpleNamespace(comm=SimpleNamespace(size=3))
    assert EITForwardModel._stable_cpu_petsc_types(model) == ("MATMPIAIJ", "VECMPI")

    model.mesh = SimpleNamespace(comm=_ExplodingComm())
    monkeypatch.setattr(
        forward_module,
        "PETSc",
        SimpleNamespace(
            Mat=SimpleNamespace(Type=SimpleNamespace()),
            Vec=SimpleNamespace(Type=SimpleNamespace()),
        ),
    )
    assert EITForwardModel._stable_cpu_petsc_types(model) == ("seqaij", "seq")


def test_v135_large_cuda_complex_cem_skips_dense_lu_fallback() -> None:
    model = _make_model(dofs=20000, n_elec=16)
    model.backend_config = SimpleNamespace(cuda_dense_fallback_max_gib=0.01)
    model._petsc_backend_info = {"petsc_device_effective": "cuda"}
    model._active_scalar_dtype = lambda: np.complex64
    session = SimpleNamespace(ksp_type="gmres", pc_type="gamg")

    assert (
        EITForwardModel._cuda_cem_requires_direct_solve(
            model,
            session,
            rhs_count=208,
        )
        is False
    )

    diag = model.get_backend_diagnostics()
    assert diag["cuda_dense_lu_fallback_skipped"] is True
    assert diag["cuda_dense_lu_fallback_scalar_dtype"] == "complex64"
    assert diag["cuda_dense_lu_fallback_estimated_gib"] > 0.01
    assert str(diag["cuda_dense_lu_fallback_skip_reason"]).startswith(
        "cuda_dense_lu_estimated_memory_exceeds_limit"
    )
    assert str(diag["gpu_fallback_reason"]).startswith("cuda_dense_lu_fallback_skipped")


def test_v135_small_cuda_cem_can_still_use_dense_lu_fallback() -> None:
    model = _make_model(dofs=64, n_elec=16)
    model.backend_config = SimpleNamespace(cuda_dense_fallback_max_gib=2.0)
    model._petsc_backend_info = {"petsc_device_effective": "cuda"}
    model._active_scalar_dtype = lambda: np.complex64
    session = SimpleNamespace(ksp_type="gmres", pc_type="gamg")

    assert (
        EITForwardModel._cuda_cem_requires_direct_solve(
            model,
            session,
            rhs_count=16,
        )
        is True
    )

    diag = model.get_backend_diagnostics()
    assert diag["cuda_dense_lu_fallback_skipped"] is False


def test_mpi_backend_info_reports_current_single_rank_boundary():
    model = _make_model()
    model.mesh = SimpleNamespace(
        comm=SimpleNamespace(Get_size=lambda: 4, Get_rank=lambda: 2)
    )

    info = EITForwardModel._resolve_mpi_backend_info(model)
    assert info["mpi_size"] == 4
    assert info["mpi_rank"] == 2
    assert info["mpi_parallel"] is True
    assert info["mpi_size_supported"] is False
    assert (
        info["mpi_fallback_reason"]
        == "mpi_size_gt_1_not_supported_phase2_single_rank_only"
    )

    with pytest.raises(RuntimeError, match="Detected MPI size=4"):
        EITForwardModel._assert_supported_mpi_runtime(model)


def test_resolve_petsc_backend_info_handles_non_petsc_missing_probe_and_capabilities(
    monkeypatch: pytest.MonkeyPatch,
):
    model = _make_model(linear_backend="scipy")
    info = EITForwardModel._resolve_petsc_backend_info(model)
    assert info["forward_factor_backend"] == "scipy"
    assert info["petsc_device_effective"] == "cpu"
    assert info["mpi_size"] == 1
    assert info["mpi_size_supported"] is True

    model = _make_model()
    monkeypatch.setattr(forward_module, "PETSc", None)
    info = EITForwardModel._resolve_petsc_backend_info(model)
    assert info["gpu_fallback_reason"] == "petsc_unavailable"

    cuda_missing = _make_model()
    cuda_missing.backend_config = SimpleNamespace(petsc_device="cuda")
    with pytest.raises(RuntimeError, match="petsc4py/PETSc support"):
        EITForwardModel._resolve_petsc_backend_info(cuda_missing)

    monkeypatch.setattr(forward_module, "PETSc", object())
    monkeypatch.delattr(perf_caps, "probe_petsc_cuda_runtime", raising=False)
    info = EITForwardModel._resolve_petsc_backend_info(model)
    assert str(info["gpu_fallback_reason"]).startswith("capability_probe_failed:")

    cuda_model = _make_model()
    cuda_model.backend_config = SimpleNamespace(petsc_device="cuda")
    with pytest.raises(RuntimeError, match="successful PETSc CUDA capability probe"):
        EITForwardModel._resolve_petsc_backend_info(cuda_model)


def test_resolve_petsc_backend_info_cpu_cuda_structured_and_cuda_success(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(forward_module, "PETSc", object())

    cpu_model = _make_model()
    cpu_model.backend_config = SimpleNamespace(petsc_device="cpu")
    monkeypatch.setattr(
        perf_caps,
        "probe_petsc_cuda_runtime",
        lambda: {
            "petsc_cuda": True,
            "mat_type_name": "aijcusparse",
            "vec_type_name": "cuda",
        },
    )
    cpu_info = EITForwardModel._resolve_petsc_backend_info(cpu_model)
    assert cpu_info["petsc_device_effective"] == "cpu"
    assert cpu_info["capability"]["petsc_cuda"] is True
    assert cpu_info["petsc_mat_type"] == "seqaij"
    assert cpu_info["petsc_vec_type"] == "seq"

    fallback_model = _make_model(forward_backend="cuda_structured")
    fallback_model.backend_config = SimpleNamespace(petsc_device="cuda")
    monkeypatch.setattr(
        perf_caps,
        "probe_petsc_cuda_runtime",
        lambda: {"petsc_cuda": False, "errors": {"mat": "missing"}},
    )
    fallback_info = EITForwardModel._resolve_petsc_backend_info(fallback_model)
    assert (
        fallback_info["gpu_fallback_reason"]
        == "petsc_cuda_not_required_for_cuda_structured"
    )

    cuda_model = _make_model()
    monkeypatch.setattr(
        perf_caps,
        "probe_petsc_cuda_runtime",
        lambda: {
            "petsc_cuda": True,
            "mat_type_name": "aijcusparse",
            "vec_type_name": "cuda",
            "dense_mat_type_name": "densecuda",
        },
    )
    cuda_info = EITForwardModel._resolve_petsc_backend_info(cuda_model)
    assert cuda_info["petsc_device_effective"] == "cuda"
    assert cuda_info["petsc_mat_type"] == "aijcusparse"
    assert cuda_info["petsc_vec_type"] == "cuda"
    assert cuda_info["petsc_dense_mat_type"] == "densecuda"
    assert cuda_info["gpu_transfer_risk"] == "mixed_dolfinx_assembly_to_petsc_cuda"


def test_requested_petsc_type_helpers_use_explicit_and_namespace_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_petsc = SimpleNamespace(
        Mat=SimpleNamespace(
            Type=SimpleNamespace(AIJCUSPARSE="AIJCUSPARSE", DENSECUDA="DENSECUDA")
        ),
        Vec=SimpleNamespace(Type=SimpleNamespace(CUDA="VECCUDA")),
    )
    monkeypatch.setattr(forward_module, "PETSc", fake_petsc)

    model = _make_model()
    model._petsc_backend_info = {"petsc_device_effective": "cpu"}
    assert EITForwardModel._get_requested_petsc_mat_type(model) is None
    assert EITForwardModel._get_requested_dense_mat_type(model) is None
    assert EITForwardModel._get_requested_petsc_vec_type(model) is None

    model._petsc_backend_info = {"petsc_device_effective": "cuda"}
    assert EITForwardModel._get_requested_petsc_mat_type(model) == "AIJCUSPARSE"
    assert EITForwardModel._get_requested_dense_mat_type(model) == "DENSECUDA"
    assert EITForwardModel._get_requested_petsc_vec_type(model) == "VECCUDA"

    model._petsc_backend_info = {
        "petsc_device_effective": "cuda",
        "petsc_mat_type": "mat-explicit",
        "petsc_dense_mat_type": "dense-explicit",
        "petsc_vec_type": "vec-explicit",
    }
    assert EITForwardModel._get_requested_petsc_mat_type(model) == "mat-explicit"
    assert EITForwardModel._get_requested_dense_mat_type(model) == "dense-explicit"
    assert EITForwardModel._get_requested_petsc_vec_type(model) == "vec-explicit"

    monkeypatch.setattr(
        forward_module,
        "PETSc",
        SimpleNamespace(
            Mat=SimpleNamespace(Type=SimpleNamespace()),
            Vec=SimpleNamespace(Type=SimpleNamespace()),
        ),
    )
    model._petsc_backend_info = {"petsc_device_effective": "cuda"}
    assert (
        EITForwardModel._get_cuda_type(model, "petsc_mat_type", "Mat", "AIJCUSPARSE")
        is None
    )


def test_gpu_gauge_fix_flag_and_csr_to_petsc_helpers_cover_cpu_and_unavailable_paths(
    monkeypatch: pytest.MonkeyPatch,
):
    model = _make_model()
    model._petsc_backend_info = {"petsc_device_effective": "cpu"}
    assert EITForwardModel._gpu_gauge_fix_enabled(model) is False

    model._petsc_backend_info = {"petsc_device_effective": "cuda"}
    assert EITForwardModel._gpu_gauge_fix_enabled(model) is True

    monkeypatch.setattr(forward_module, "PETSc", None)
    with pytest.raises(RuntimeError, match="petsc4py is not available"):
        EITForwardModel._csr_to_petsc(sparse.identity(2, format="csr"))


def test_ensure_mat_and_vec_type_cover_convert_settype_and_passthrough(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(forward_module, "PETSc", object())

    mat = _FakeMat("aij")
    assert EITForwardModel._ensure_mat_type(mat, "aij") is mat

    converted = _FakeMat("densecuda")
    mat2 = _FakeMat("seqaij")
    mat2.convert_result = converted
    assert EITForwardModel._ensure_mat_type(mat2, "densecuda") is converted
    assert mat2.destroyed is True

    mat3 = _FakeMat("seqaij")
    mat3.raise_on_convert = True
    out = EITForwardModel._ensure_mat_type(mat3, "mpiaij")
    assert out is mat3
    assert mat3.set_type_calls == ["mpiaij"]

    vec = _FakeVec("seq")
    assert EITForwardModel._ensure_vec_type(vec, "seq") is vec
    out_vec = EITForwardModel._ensure_vec_type(vec, "cuda")
    assert out_vec is vec
    assert vec.set_type_calls == ["cuda"]


def test_assemble_form_helpers_route_gpu_kinds_and_kind_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    model = _make_model()
    mat_calls = []
    vec_calls = []

    def fake_assemble_matrix(_form_obj, kind=None):
        mat_calls.append(kind)
        if kind == "seqaij":
            raise TypeError("legacy assemble signature")
        return _FakeMat("assembled")

    def fake_assemble_vector(_form_obj, kind=None):
        vec_calls.append(kind)
        if kind == "seq":
            raise TypeError("legacy assemble signature")
        return _FakeVec("assembled")

    monkeypatch.setattr(
        forward_module.fem_petsc, "assemble_matrix", fake_assemble_matrix
    )
    monkeypatch.setattr(
        forward_module.fem_petsc, "assemble_vector", fake_assemble_vector
    )
    monkeypatch.setattr(
        model, "_ensure_mat_type", lambda mat, mat_type: (mat, mat_type)
    )
    monkeypatch.setattr(
        model, "_ensure_vec_type", lambda vec, vec_type: (vec, vec_type)
    )

    gpu_mat, gpu_kind = EITForwardModel._assemble_form_matrix(
        model, "f", mat_kind="AIJCUSPARSE"
    )
    assert isinstance(gpu_mat, _FakeMat)
    assert gpu_kind == "AIJCUSPARSE"

    cpu_mat, cpu_kind = EITForwardModel._assemble_form_matrix(
        model, "f", mat_kind="seqaij"
    )
    assert isinstance(cpu_mat, _FakeMat)
    assert cpu_kind == "seqaij"

    gpu_vec, gpu_vec_kind = EITForwardModel._assemble_form_vector(
        model, "g", vec_kind="cuda"
    )
    assert isinstance(gpu_vec, _FakeVec)
    assert gpu_vec_kind == "cuda"

    cpu_vec, cpu_vec_kind = EITForwardModel._assemble_form_vector(
        model, "g", vec_kind="seq"
    )
    assert isinstance(cpu_vec, _FakeVec)
    assert cpu_vec_kind == "seq"

    assert mat_calls == [None, "seqaij", None]
    assert vec_calls == [None, "seq", None]


def test_structural_diagonal_and_cuda_gauge_fix_helpers(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(forward_module, "PETSc", _FakePETScDense)
    model = _make_model()

    structural = _StructuralMat(shape=(3, 4))
    EITForwardModel._ensure_structural_diagonal(model, structural)
    assert structural.options == [("new_nonzero_allocation_err", False)]
    assert structural.values == [(0, 0, 0.0), (1, 1, 0.0), (2, 2, 0.0)]

    EITForwardModel._ensure_structural_diagonal(
        model, _StructuralMat(fail_get_size=True)
    )

    monkeypatch.setattr(forward_module, "PETSc", None)
    mat_passthrough = _FakeMat("aijcusparse")
    EITForwardModel._ensure_structural_diagonal(model, mat_passthrough)
    assert (
        EITForwardModel._apply_cuda_gauge_fix_matrix(model, mat_passthrough)
        is mat_passthrough
    )

    monkeypatch.setattr(forward_module, "PETSc", _FakePETScDense)
    model._petsc_backend_info = {"petsc_device_effective": "cuda"}
    captured = {}
    base = sparse.lil_matrix((5, 5), dtype=float)
    base[0, 2] = 3.0
    base[0, 4] = 5.0
    base[1, 2] = 7.0
    base[3, 4] = 11.0
    base[4, 0] = 13.0
    base = base.tocsr()
    original = _FakeMat("aijcusparse")
    fixed = _FakeMat("aijcusparse")
    monkeypatch.setattr(model, "_petsc_to_csr", lambda _mat: base.copy())

    def fake_csr_to_petsc(csr):
        captured["matrix"] = csr.copy()
        return fixed

    monkeypatch.setattr(model, "_csr_to_petsc", fake_csr_to_petsc)

    out = EITForwardModel._apply_cuda_gauge_fix_matrix(model, original)
    assert out is fixed
    assert original.destroyed is True
    assert (
        model.get_backend_diagnostics()["gpu_constraint_strategy"]
        == "reference-electrode-row"
    )
    fixed_csr = captured["matrix"].tocsr()
    assert np.array_equal(fixed_csr.getrow(4).indices, np.array([2], dtype=np.int32))
    np.testing.assert_allclose(fixed_csr.getrow(4).data, np.array([1.0]))
    assert 2 in fixed_csr.getrow(0).indices
    assert 4 in fixed_csr.getrow(0).indices

    monkeypatch.setattr(
        model,
        "_petsc_to_csr",
        lambda _mat: (_ for _ in ()).throw(RuntimeError("bad csr")),
    )
    original2 = _FakeMat("aijcusparse")
    assert EITForwardModel._apply_cuda_gauge_fix_matrix(model, original2) is original2

    class _ExplodingDestroyMat(_FakeMat):
        def destroy(self):
            self.destroyed = True
            raise RuntimeError("destroy failed")

    monkeypatch.setattr(model, "_petsc_to_csr", lambda _mat: base.copy())
    exploding = _ExplodingDestroyMat("aijcusparse")
    fixed_again = _FakeMat("aijcusparse")
    monkeypatch.setattr(model, "_csr_to_petsc", lambda _csr: fixed_again)
    out = EITForwardModel._apply_cuda_gauge_fix_matrix(model, exploding)
    assert out is fixed_again
    assert exploding.destroyed is True


def test_cuda_rhs_and_solution_recentering_respect_gpu_enablement():
    model = _make_model()
    rhs = np.arange(10, dtype=float).reshape(5, 2)
    sol = np.arange(10, dtype=float).reshape(5, 2)

    model._petsc_backend_info = {"petsc_device_effective": "cpu"}
    assert EITForwardModel._apply_cuda_gauge_fix_rhs(model, rhs) is rhs
    assert EITForwardModel._recenter_cuda_gauge_solution(model, sol) is sol

    model._petsc_backend_info = {"petsc_device_effective": "cuda"}
    rhs_fixed = EITForwardModel._apply_cuda_gauge_fix_rhs(model, rhs.copy())
    np.testing.assert_allclose(rhs_fixed[2, :], rhs[2, :])
    assert np.allclose(rhs_fixed[4, :], 0.0)

    sol_fixed = EITForwardModel._recenter_cuda_gauge_solution(model, sol.copy())
    offsets = sol[2:4, :].mean(axis=0, keepdims=True)
    np.testing.assert_allclose(sol_fixed[:2, :], sol[:2, :] - offsets)
    np.testing.assert_allclose(sol_fixed[2:4, :], sol[2:4, :] - offsets)
    np.testing.assert_allclose(sol_fixed[4, :], 0.0)


def test_cuda_cem_requires_direct_solve_for_reference_gauge():
    model = _make_model()
    model._petsc_backend_info = {"petsc_device_effective": "cpu"}
    iterative = forward_module.ForwardKSPSession(
        ksp=object(),
        current_A=object(),
        current_solve_A=object(),
        backend_name="petsc-ksp",
        ksp_type="cg",
        pc_type="gamg",
        factor_solver_type=None,
        solve_mat_type=None,
        structural_fingerprint="fp",
    )
    assert EITForwardModel._cuda_cem_requires_direct_solve(model, iterative) is False

    model._petsc_backend_info = {"petsc_device_effective": "cuda"}
    assert EITForwardModel._cuda_cem_requires_direct_solve(model, iterative) is True

    direct = forward_module.ForwardKSPSession(
        ksp=object(),
        current_A=object(),
        current_solve_A=object(),
        backend_name="petsc-ksp-densecuda-lu",
        ksp_type="preonly",
        pc_type="lu",
        factor_solver_type=None,
        solve_mat_type="densecuda",
        structural_fingerprint="fp",
    )
    assert EITForwardModel._cuda_cem_requires_direct_solve(model, direct) is False


def test_make_petsc_dense_solver_bundle_validates_dense_type_and_builds_solver(
    monkeypatch: pytest.MonkeyPatch,
):
    model = _make_model()

    monkeypatch.setattr(forward_module, "PETSc", None)
    with pytest.raises(RuntimeError, match="petsc4py is required"):
        EITForwardModel._make_petsc_dense_solver_bundle(model, _FakeMat())

    monkeypatch.setattr(forward_module, "PETSc", _FakePETScDense)
    monkeypatch.setattr(model, "_get_requested_dense_mat_type", lambda: None)
    with pytest.raises(RuntimeError, match="CUDA dense PETSc Mat type is unavailable"):
        EITForwardModel._make_petsc_dense_solver_bundle(model, _FakeMat())

    monkeypatch.setattr(model, "_get_requested_dense_mat_type", lambda: "densecuda")
    monkeypatch.setattr(model, "_stable_cpu_petsc_types", lambda: ("seqaij", "seq"))

    def fake_ensure(mat, mat_type):
        mat.setType(mat_type)
        return mat

    monkeypatch.setattr(model, "_ensure_mat_type", fake_ensure)
    bundle = EITForwardModel._make_petsc_dense_solver_bundle(model, _FakeMat("aij"))
    assert bundle["backend"] == "petsc-ksp-densecuda-lu"
    assert bundle["ksp_type"] == "preonly"
    assert bundle["pc_type"] == "lu"
    assert bundle["solve_mat_type"] == "densecuda"
    assert bundle["ksp"].tolerances == {"rtol": 1e-10, "atol": 1e-12, "max_it": 200}
    assert bundle["ksp"].reuse is True
    assert bundle["ksp"].did_setup is True

    class _ReuseFailKSP(_FakeKSPInstance):
        def setReusePreconditioner(self, enabled):
            _ = enabled
            raise RuntimeError("reuse failed")

    monkeypatch.setattr(
        forward_module,
        "PETSc",
        SimpleNamespace(KSP=lambda: _ReuseFailKSP()),
    )
    bundle_reuse_fail = EITForwardModel._make_petsc_dense_solver_bundle(
        model, _FakeMat("aij")
    )
    assert bundle_reuse_fail["ksp"].did_setup is True
