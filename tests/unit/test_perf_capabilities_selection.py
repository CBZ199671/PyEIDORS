"""Tests for performance capability discovery and preconditioner selection."""

from __future__ import annotations

from types import SimpleNamespace

import pyeidors.perf.capabilities as perf_caps
from pyeidors.perf.capabilities import (
    probe_mpi_runtime,
    probe_petsc_cuda_runtime,
    select_fast_linear_path,
    select_fused_strategy,
    select_preconditioner,
)
from pyeidors.perf.forward_solver_policy import (
    resolve_3d_cuda_forward_solver_policy,
    resolve_3d_cuda_mat_solve_policy,
)


def test_select_preconditioner_auto_priority_order():
    caps = {
        "pyamg": True,
        "cholmod": True,
        "petsc_mat_solve": True,
        "petsc_gamg": True,
    }
    assert select_preconditioner("auto", caps) == "cholmod"


def test_tetra_cuda_auto_solver_policy_prefers_real_amgx() -> None:
    capability = {"petsc_cuda": True, "petsc_hypre": True, "petsc_amgx": True}

    tetra = resolve_3d_cuda_forward_solver_policy(
        requested_solver_preset="auto",
        mesh_dim=3,
        petsc_device="cuda",
        forward_backend="dolfinx",
        mesh_family="tetra",
        capability=capability,
    )
    assert tetra["forward_solver_preset_effective"] == "cuda_amgx"
    assert tetra["forward_solver_policy_reason"] == "tetra_real_cuda_amgx_default"

    matsolve = resolve_3d_cuda_mat_solve_policy(
        requested_mat_solve="auto",
        mesh_dim=3,
        petsc_device="cuda",
        forward_backend="dolfinx",
        solver_preset=tetra["forward_solver_preset_effective"],
    )
    assert matsolve["forward_mat_solve_effective_policy"] == "off"
    assert (
        matsolve["forward_mat_solve_policy_reason"]
        == "cuda_amgx_matsolve_disabled_mainline"
    )

    complex_tetra = resolve_3d_cuda_forward_solver_policy(
        requested_solver_preset="auto",
        mesh_dim=3,
        petsc_device="cuda",
        forward_backend="dolfinx",
        mesh_family="tetra",
        capability={"petsc_cuda": True, "petsc_hypre": True, "petsc_amgx": False},
        complex_admittivity_requested=True,
    )
    assert complex_tetra["forward_solver_preset_effective"] == "3d_gamg"
    assert (
        complex_tetra["forward_solver_policy_reason"]
        == "complex_cuda_native_gamg_default"
    )

    strict_complex_tetra = resolve_3d_cuda_forward_solver_policy(
        requested_solver_preset="auto",
        mesh_dim=3,
        petsc_device="cuda",
        forward_backend="dolfinx",
        mesh_family="tetra",
        capability={"petsc_cuda": True, "petsc_hypre": True, "petsc_amgx": True},
        complex_admittivity_requested=True,
        complex_high_accuracy=True,
    )
    assert (
        strict_complex_tetra["forward_solver_preset_effective"]
        == "complex_block_real_amgx"
    )
    assert (
        strict_complex_tetra["forward_solver_policy_reason"]
        == "complex_cuda_block_real_amgx_default"
    )

    missing_amgx = resolve_3d_cuda_forward_solver_policy(
        requested_solver_preset="auto",
        mesh_dim=3,
        petsc_device="cuda",
        forward_backend="dolfinx",
        mesh_family="tetra",
        capability={"petsc_cuda": True, "petsc_hypre": True, "petsc_amgx": False},
    )
    assert missing_amgx["forward_solver_preset_effective"] == "3d_gamg"
    assert (
        missing_amgx["forward_solver_policy_reason"]
        == "tetra_amgx_unavailable_downgraded_to_3d_gamg"
    )

    legacy_unknown_family = resolve_3d_cuda_forward_solver_policy(
        requested_solver_preset="auto",
        mesh_dim=3,
        petsc_device="cuda",
        forward_backend="dolfinx",
        capability={"petsc_cuda": True, "petsc_hypre": True, "petsc_amgx": False},
    )
    assert legacy_unknown_family["forward_solver_preset_effective"] == "spd_gamg"


def test_select_preconditioner_auto_fallback_chain():
    assert (
        select_preconditioner(
            "auto",
            {
                "pyamg": True,
                "cholmod": False,
                "petsc_mat_solve": False,
                "petsc_gamg": True,
            },
        )
        == "pyamg"
    )
    assert (
        select_preconditioner(
            "auto",
            {
                "pyamg": False,
                "cholmod": False,
                "petsc_mat_solve": False,
                "petsc_gamg": True,
            },
        )
        == "petsc-gamg"
    )
    assert (
        select_preconditioner(
            "auto",
            {
                "pyamg": False,
                "cholmod": False,
                "petsc_mat_solve": False,
                "petsc_gamg": False,
            },
        )
        == "diag"
    )


def test_select_preconditioner_explicit_unavailable_falls_back_to_diag():
    caps = {
        "pyamg": False,
        "cholmod": False,
        "petsc_mat_solve": False,
        "petsc_gamg": False,
    }
    assert select_preconditioner("pyamg", caps) == "diag"
    assert select_preconditioner("cholmod", caps) == "diag"
    assert select_preconditioner("petsc-gamg", caps) == "diag"


def test_select_preconditioner_explicit_cholmod_when_available():
    caps = {
        "pyamg": True,
        "cholmod": True,
        "petsc_mat_solve": True,
        "petsc_gamg": True,
    }
    assert select_preconditioner("cholmod", caps) == "cholmod"


def test_select_preconditioner_explicit_matrix_free_diag_modes():
    caps = {
        "pyamg": False,
        "cholmod": False,
        "petsc_mat_solve": False,
        "petsc_gamg": False,
    }
    assert select_preconditioner("diag", caps) == "diag"
    assert select_preconditioner("noser", caps) == "noser"
    assert select_preconditioner("prior", caps) == "prior"
    assert select_preconditioner("pmat", caps) == "pmat"
    assert select_preconditioner("coarse", caps) == "coarse"
    assert select_preconditioner("custom", caps) == "custom"


def test_select_fast_linear_path_auto_prefers_woodbury_for_diagonal_regularization():
    selected = select_fast_linear_path(
        "auto",
        regularization_is_diagonal=True,
        regularization_is_sparse_spd=True,
        capabilities={"cholmod": True},
    )
    assert selected == "woodbury"


def test_select_fast_linear_path_auto_prefers_pcg_for_sparse_spd_non_diagonal():
    selected = select_fast_linear_path(
        "auto",
        regularization_is_diagonal=False,
        regularization_is_sparse_spd=True,
        capabilities={"cholmod": True},
    )
    assert selected == "pcg"


def test_select_fused_strategy_auto_is_disabled_by_policy():
    strategy = select_fused_strategy(
        solver_mode="fast",
        mesh_dim=3,
        n_param=12000,
        n_meas=208,
        rom_mode="auto",
        inexact_mode="auto",
        lowrank_mode="auto",
        regularization_is_diagonal=True,
        capabilities={"cholmod": True},
    )
    assert strategy["enabled"] is False
    assert strategy["rom"] is False


def test_select_fused_strategy_explicit_rom_on_enables_experimental_path():
    strategy = select_fused_strategy(
        solver_mode="fast",
        mesh_dim=3,
        n_param=12000,
        n_meas=208,
        rom_mode="on",
        inexact_mode="auto",
        lowrank_mode="auto",
        regularization_is_diagonal=True,
        capabilities={"cholmod": True},
    )
    assert strategy["enabled"] is True
    assert strategy["rom"] is True


def test_select_fused_strategy_disables_for_2d_or_strict():
    s1 = select_fused_strategy(
        solver_mode="strict",
        mesh_dim=3,
        n_param=12000,
        n_meas=208,
        rom_mode="auto",
        inexact_mode="auto",
        lowrank_mode="auto",
        regularization_is_diagonal=True,
        capabilities={"cholmod": True},
    )
    assert s1["enabled"] is False
    s2 = select_fused_strategy(
        solver_mode="fast",
        mesh_dim=2,
        n_param=12000,
        n_meas=208,
        rom_mode="auto",
        inexact_mode="auto",
        lowrank_mode="auto",
        regularization_is_diagonal=True,
        capabilities={"cholmod": True},
    )
    assert s2["enabled"] is False


class _FailingCudaMat:
    class Type:
        AIJCUSPARSE = "aijcusparse"
        DENSECUDA = "densecuda"

    def create(self, comm=None):
        _ = comm
        return self

    def setSizes(self, size):
        _ = size

    def setType(self, kind):
        raise RuntimeError(f"Unknown Mat type given: {kind}")

    def setPreallocationNNZ(self, value):
        _ = value

    def setUp(self):
        return None

    def setValue(self, i, j, value):
        _ = (i, j, value)

    def assemblyBegin(self):
        return None

    def assemblyEnd(self):
        return None

    def destroy(self):
        return None


class _FailingCudaVec:
    class Type:
        CUDA = "cuda"

    def create(self, comm=None):
        _ = comm
        return self

    def setSizes(self, size):
        _ = size

    def setType(self, kind):
        raise RuntimeError(f"Unknown vector type: {kind}")

    def setUp(self):
        return None

    def setValue(self, i, value):
        _ = (i, value)

    def assemblyBegin(self):
        return None

    def assemblyEnd(self):
        return None

    def destroy(self):
        return None


class _WorkingCudaMat(_FailingCudaMat):
    def setType(self, kind):
        self.kind = kind


class _WorkingCudaVec(_FailingCudaVec):
    def setType(self, kind):
        self.kind = kind


class _FailingCudaPETSc:
    COMM_SELF = object()
    Mat = _FailingCudaMat
    Vec = _FailingCudaVec

    class KSP:
        pass

    class PC:
        class Type:
            GAMG = "gamg"
            HYPRE = "hypre"
            AMGX = "amgx"


class _WorkingCudaPETSc:
    COMM_SELF = object()
    Mat = _WorkingCudaMat
    Vec = _WorkingCudaVec

    class KSP:
        pass

    class PC:
        class Type:
            GAMG = "gamg"
            HYPRE = "hypre"
            AMGX = "amgx"


class _FakeMPIComm:
    def __init__(self, *, size: int, rank: int = 0):
        self._size = int(size)
        self._rank = int(rank)

    def Get_size(self):
        return self._size

    def Get_rank(self):
        return self._rank


class _RecordingPetscOptions(dict):
    def __init__(self):
        super().__init__()
        self.writes: dict[str, str] = {}

    def __setitem__(self, key, value):
        self.writes[str(key)] = str(value)
        super().__setitem__(key, value)


class _ProbeVec:
    def __init__(self, size: int = 0, norm_value: float = 1.0):
        self.size = size
        self.kind = None
        self.norm_value = norm_value

    def createSeq(self, size):
        self.size = int(size)
        return self

    def setType(self, kind):
        self.kind = kind

    def set(self, _value):
        return None

    def assemblyBegin(self):
        return None

    def assemblyEnd(self):
        return None

    def duplicate(self):
        return _ProbeVec(self.size)

    def norm(self):
        return self.norm_value

    def axpy(self, _alpha, _other):
        self.norm_value = 0.0

    def destroy(self):
        return None


class _ProbeMat:
    def createAIJ(self, shape, nnz=None):
        self.shape = shape
        self.nnz = nnz
        self.kind = None
        self.values = {}
        return self

    def setType(self, kind):
        self.kind = kind

    def setUp(self):
        return None

    def __setitem__(self, key, value):
        self.values[key] = value

    def assemblyBegin(self):
        return None

    def assemblyEnd(self):
        return None

    def mult(self, _x, residual):
        residual.norm_value = 1.0

    def destroy(self):
        return None


class _ProbePC:
    def setType(self, kind):
        self.kind = kind


class _ProbeKSP:
    def __init__(self):
        self.pc = _ProbePC()
        self.ksp_type = None
        self.prefix = None

    def create(self):
        return self

    def setOptionsPrefix(self, prefix):
        self.prefix = prefix

    def setOperators(self, _mat):
        return None

    def setType(self, kind):
        self.ksp_type = kind

    def getPC(self):
        return self.pc

    def setTolerances(self, **_kwargs):
        return None

    def setFromOptions(self):
        return None

    def solve(self, _b, x):
        x.norm_value = 1.0

    def destroy(self):
        return None


def test_petsc_amgx_smoke_uses_complex_safe_options_for_complex_scalar() -> None:
    options = _RecordingPetscOptions()
    ksp = _ProbeKSP()
    fake_petsc = SimpleNamespace(
        ScalarType=complex,
        Options=lambda: options,
        Mat=_ProbeMat,
        Vec=_ProbeVec,
        KSP=lambda: ksp,
    )

    ok, error = perf_caps._probe_petsc_amgx_setup_solve(fake_petsc)

    assert ok is True, error
    assert ksp.ksp_type == "fgmres"
    assert options.writes["pyeidors_amgx_probe_pc_amgx_amg_method"] == "AGGREGATION"
    assert options.writes["pyeidors_amgx_probe_pc_amgx_selector"] == "SIZE_8"
    assert options.writes["pyeidors_amgx_probe_pc_amgx_smoother"] == "BLOCK_JACOBI"
    assert options.writes["pyeidors_amgx_probe_pc_amgx_coarse_solver"] == "NOSOLVER"


def test_petsc_amgx_smoke_keeps_real_jacobi_l1_probe() -> None:
    options = _RecordingPetscOptions()
    ksp = _ProbeKSP()
    fake_petsc = SimpleNamespace(
        ScalarType=float,
        Options=lambda: options,
        Mat=_ProbeMat,
        Vec=_ProbeVec,
        KSP=lambda: ksp,
    )

    ok, error = perf_caps._probe_petsc_amgx_setup_solve(fake_petsc)

    assert ok is True, error
    assert ksp.ksp_type == "cg"
    assert options.writes["pyeidors_amgx_probe_pc_amgx_smoother"] == "JACOBI_L1"
    assert options.writes["pyeidors_amgx_probe_pc_amgx_exact_coarse_solve"] == "0"
    assert "pyeidors_amgx_probe_pc_amgx_selector" not in options.writes


def test_probe_petsc_cuda_runtime_rejects_unknown_type_symbols(monkeypatch):
    perf_caps.probe_petsc_cuda_runtime.cache_clear()
    perf_caps.detect_performance_capabilities.cache_clear()
    monkeypatch.setattr(perf_caps, "_load_petsc_runtime", lambda: _FailingCudaPETSc)

    probe = probe_petsc_cuda_runtime()

    assert probe["petsc_available"] is True
    assert probe["petsc_cuda"] is False
    assert probe["petsc_cuda_mat"] is False
    assert probe["petsc_cuda_vec"] is False
    assert probe["petsc_hypre"] is True
    assert probe["petsc_amgx"] is True
    assert probe["petsc_amgx_cuda_candidate"] is False
    assert "Unknown Mat type" in probe["errors"]["mat"]
    assert "Unknown vector type" in probe["errors"]["vec"]


def test_probe_petsc_cuda_runtime_accepts_working_types(monkeypatch):
    perf_caps.probe_petsc_cuda_runtime.cache_clear()
    perf_caps.detect_performance_capabilities.cache_clear()
    monkeypatch.setattr(perf_caps, "_load_petsc_runtime", lambda: _WorkingCudaPETSc)
    monkeypatch.setattr(
        perf_caps,
        "_probe_petsc_amgx_setup_solve",
        lambda _petsc: (True, None),
    )

    probe = probe_petsc_cuda_runtime()

    assert probe["petsc_available"] is True
    assert probe["petsc_cuda"] is True
    assert probe["petsc_cuda_mat"] is True
    assert probe["petsc_cuda_vec"] is True
    assert probe["petsc_hypre"] is True
    assert probe["petsc_amgx"] is True
    assert probe["petsc_amgx_smoke"] is True
    assert probe["petsc_amgx_cuda_candidate"] is True


def test_probe_petsc_cuda_runtime_rejects_broken_amgx_smoke(monkeypatch):
    perf_caps.probe_petsc_cuda_runtime.cache_clear()
    perf_caps.detect_performance_capabilities.cache_clear()
    monkeypatch.setattr(perf_caps, "_load_petsc_runtime", lambda: _WorkingCudaPETSc)
    monkeypatch.setattr(
        perf_caps,
        "_probe_petsc_amgx_setup_solve",
        lambda _petsc: (False, "Incorrect amgx configuration provided."),
    )

    probe = probe_petsc_cuda_runtime()

    assert probe["petsc_cuda"] is True
    assert probe["petsc_amgx"] is True
    assert probe["petsc_amgx_smoke"] is False
    assert probe["petsc_amgx_cuda_candidate"] is False
    assert "Incorrect amgx configuration" in probe["errors"]["amgx"]


def test_v325_probe_petsc_cuda_runtime_uses_opt_in_disk_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("PYEIDORS_PETSC_CUDA_PROBE_CACHE", "1")
    monkeypatch.setenv("PYEIDORS_PETSC_CUDA_PROBE_CACHE_DIR", str(tmp_path))
    perf_caps.probe_petsc_cuda_runtime.cache_clear()
    perf_caps.detect_performance_capabilities.cache_clear()
    monkeypatch.setattr(perf_caps, "_load_petsc_runtime", lambda: _WorkingCudaPETSc)
    monkeypatch.setattr(
        perf_caps,
        "_probe_petsc_amgx_setup_solve",
        lambda _petsc: (True, None),
    )

    first = probe_petsc_cuda_runtime()

    assert first["petsc_cuda"] is True
    assert first["probe_cache"]["enabled"] is True
    assert first["probe_cache"]["hit"] is False
    assert first["probe_cache"]["stored"] is True

    perf_caps.probe_petsc_cuda_runtime.cache_clear()
    monkeypatch.setattr(
        perf_caps,
        "_probe_petsc_mat_type",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("disk cache hit must skip PETSc Mat probes")
        ),
    )

    second = probe_petsc_cuda_runtime()

    assert second["petsc_cuda"] is True
    assert second["probe_cache"]["hit"] is True
    assert second["probe_cache"]["layer"] == "disk"


def test_probe_petsc_cuda_runtime_cache_tracks_runtime_identity(monkeypatch):
    perf_caps.probe_petsc_cuda_runtime.cache_clear()
    perf_caps.detect_performance_capabilities.cache_clear()
    monkeypatch.setattr(perf_caps, "_load_petsc_runtime", lambda: _FailingCudaPETSc)
    probe_fail = probe_petsc_cuda_runtime()

    monkeypatch.setattr(perf_caps, "_load_petsc_runtime", lambda: _WorkingCudaPETSc)
    monkeypatch.setattr(
        perf_caps,
        "_probe_petsc_amgx_setup_solve",
        lambda _petsc: (True, None),
    )
    probe_working = probe_petsc_cuda_runtime()

    assert probe_fail["petsc_cuda"] is False
    assert probe_working["petsc_cuda"] is True


def test_probe_mpi_runtime_reports_single_rank_limit():
    single = probe_mpi_runtime(comm=_FakeMPIComm(size=1, rank=0))
    assert single["mpi_size"] == 1
    assert single["mpi_rank"] == 0
    assert single["mpi_size_supported"] is True
    assert single["mpi_fallback_reason"] is None

    parallel = probe_mpi_runtime(comm=_FakeMPIComm(size=4, rank=2))
    assert parallel["mpi_parallel"] is True
    assert parallel["mpi_size_supported"] is False
    assert parallel["mpi_fallback_reason"] == perf_caps.MPI_SINGLE_RANK_FALLBACK_REASON
    assert "MPI size=1" in str(parallel["mpi_guidance"])

    supported = probe_mpi_runtime(
        comm=_FakeMPIComm(size=4, rank=1),
        supports_parallel=True,
    )
    assert supported["mpi_size_supported"] is True
    assert supported["mpi_fallback_reason"] is None


def test_detect_performance_capabilities_cache_tracks_runtime_identity(monkeypatch):
    perf_caps.probe_petsc_cuda_runtime.cache_clear()
    perf_caps.detect_performance_capabilities.cache_clear()
    monkeypatch.setattr(perf_caps, "_has_pyamg", lambda: False)
    monkeypatch.setattr(perf_caps, "_has_cholmod", lambda: False)
    monkeypatch.setattr(perf_caps, "_load_petsc_runtime", lambda: _FailingCudaPETSc)
    caps_fail = perf_caps.detect_performance_capabilities()

    monkeypatch.setattr(perf_caps, "_load_petsc_runtime", lambda: _WorkingCudaPETSc)
    monkeypatch.setattr(
        perf_caps,
        "_probe_petsc_amgx_setup_solve",
        lambda _petsc: (True, None),
    )
    caps_working = perf_caps.detect_performance_capabilities()

    assert caps_fail["petsc_cuda"] is False
    assert caps_fail["petsc_hypre"] is True
    assert caps_fail["petsc_amgx"] is True
    assert caps_fail["petsc_amgx_cuda_candidate"] is False
    assert caps_working["petsc_cuda"] is True
    assert caps_working["petsc_hypre"] is True
    assert caps_working["petsc_amgx"] is True
    assert caps_working["petsc_amgx_cuda_candidate"] is True
    assert "mpi_size_supported" in caps_working
