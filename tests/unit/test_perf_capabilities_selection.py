"""Tests for performance capability discovery and preconditioner selection."""

from __future__ import annotations

import pyeidors.perf.capabilities as perf_caps
from pyeidors.perf.capabilities import (
    probe_mpi_runtime,
    probe_petsc_cuda_runtime,
    select_fast_linear_path,
    select_fused_strategy,
    select_preconditioner,
)


def test_select_preconditioner_auto_priority_order():
    caps = {
        "pyamg": True,
        "cholmod": True,
        "petsc_mat_solve": True,
        "petsc_gamg": True,
    }
    assert select_preconditioner("auto", caps) == "cholmod"


def test_select_preconditioner_auto_fallback_chain():
    assert (
        select_preconditioner(
            "auto",
            {"pyamg": True, "cholmod": False, "petsc_mat_solve": False, "petsc_gamg": True},
        )
        == "pyamg"
    )
    assert (
        select_preconditioner(
            "auto",
            {"pyamg": False, "cholmod": False, "petsc_mat_solve": False, "petsc_gamg": True},
        )
        == "petsc-gamg"
    )
    assert (
        select_preconditioner(
            "auto",
            {"pyamg": False, "cholmod": False, "petsc_mat_solve": False, "petsc_gamg": False},
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


class _WorkingCudaPETSc:
    COMM_SELF = object()
    Mat = _WorkingCudaMat
    Vec = _WorkingCudaVec

    class KSP:
        pass

    class PC:
        class Type:
            GAMG = "gamg"


class _FakeMPIComm:
    def __init__(self, *, size: int, rank: int = 0):
        self._size = int(size)
        self._rank = int(rank)

    def Get_size(self):
        return self._size

    def Get_rank(self):
        return self._rank


def test_probe_petsc_cuda_runtime_rejects_unknown_type_symbols(monkeypatch):
    perf_caps.probe_petsc_cuda_runtime.cache_clear()
    perf_caps.detect_performance_capabilities.cache_clear()
    monkeypatch.setattr(perf_caps, "_load_petsc_runtime", lambda: _FailingCudaPETSc)

    probe = probe_petsc_cuda_runtime()

    assert probe["petsc_available"] is True
    assert probe["petsc_cuda"] is False
    assert probe["petsc_cuda_mat"] is False
    assert probe["petsc_cuda_vec"] is False
    assert "Unknown Mat type" in probe["errors"]["mat"]
    assert "Unknown vector type" in probe["errors"]["vec"]


def test_probe_petsc_cuda_runtime_accepts_working_types(monkeypatch):
    perf_caps.probe_petsc_cuda_runtime.cache_clear()
    perf_caps.detect_performance_capabilities.cache_clear()
    monkeypatch.setattr(perf_caps, "_load_petsc_runtime", lambda: _WorkingCudaPETSc)

    probe = probe_petsc_cuda_runtime()

    assert probe["petsc_available"] is True
    assert probe["petsc_cuda"] is True
    assert probe["petsc_cuda_mat"] is True
    assert probe["petsc_cuda_vec"] is True


def test_probe_petsc_cuda_runtime_cache_tracks_runtime_identity(monkeypatch):
    perf_caps.probe_petsc_cuda_runtime.cache_clear()
    perf_caps.detect_performance_capabilities.cache_clear()
    monkeypatch.setattr(perf_caps, "_load_petsc_runtime", lambda: _FailingCudaPETSc)
    probe_fail = probe_petsc_cuda_runtime()

    monkeypatch.setattr(perf_caps, "_load_petsc_runtime", lambda: _WorkingCudaPETSc)
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
    caps_working = perf_caps.detect_performance_capabilities()

    assert caps_fail["petsc_cuda"] is False
    assert caps_working["petsc_cuda"] is True
    assert "mpi_size_supported" in caps_working
