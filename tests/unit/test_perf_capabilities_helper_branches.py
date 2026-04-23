"""Additional branch coverage for performance capability helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pyeidors.perf.capabilities as perf_caps


class _AttrFlipsOnSecondRead:
    def __init__(self):
        self.calls = 0

    def __getattr__(self, name):
        if name != "TOKEN":
            raise AttributeError(name)
        self.calls += 1
        if self.calls == 1:
            return "value"
        raise RuntimeError("boom")


class _CreatorWithoutKwargs:
    def __init__(self):
        self.created = False

    def create(self):
        self.created = True


class _DestroyRaisesMat:
    def create(self, comm=None):
        _ = comm
        return self

    def destroy(self):
        raise RuntimeError("destroy boom")


def test_enum_name_create_probe_and_runtime_none_paths(monkeypatch):
    assert perf_caps._enum_name(None, "X") is None
    assert perf_caps._enum_name(SimpleNamespace(), "X") is None
    assert perf_caps._enum_name(_AttrFlipsOnSecondRead(), "TOKEN") is None

    fake_petsc = SimpleNamespace(COMM_SELF=object(), Mat=_CreatorWithoutKwargs)
    created = perf_caps._create_petsc_object(fake_petsc, "Mat")
    assert created.created is True

    assert perf_caps._probe_petsc_type(
        None,
        cls_name="Mat",
        missing_label="missing",
        setup_fn=lambda obj, type_name: None,
    ) == (False, "missing")

    monkeypatch.setattr(perf_caps, "_load_petsc_runtime", lambda: None)
    assert perf_caps._probe_petsc_type(
        "cuda",
        cls_name="Mat",
        missing_label="missing",
        setup_fn=lambda obj, type_name: None,
    ) == (False, "petsc_unavailable")
    assert perf_caps._petsc_runtime_cache_key() == ("petsc:none",)

    runtime_none = perf_caps._probe_petsc_cuda_runtime_cached(("petsc:none",))
    assert runtime_none["petsc_available"] is False
    assert runtime_none["errors"]["petsc"] == "petsc_unavailable"


def test_probe_type_destroy_cleanup_and_selection_defaults(monkeypatch):
    fake_petsc = SimpleNamespace(COMM_SELF=object(), Mat=_DestroyRaisesMat)
    monkeypatch.setattr(perf_caps, "_load_petsc_runtime", lambda: fake_petsc)

    ok, err = perf_caps._probe_petsc_type(
        "cuda",
        cls_name="Mat",
        missing_label="missing",
        setup_fn=lambda obj, type_name: None,
    )
    assert ok is True
    assert err is None

    monkeypatch.setattr(
        perf_caps,
        "detect_performance_capabilities",
        lambda: {"cholmod": False, "pyamg": True, "petsc_gamg": False},
    )
    assert perf_caps.select_preconditioner("auto") == "pyamg"
    assert (
        perf_caps.select_fast_linear_path(
            "weird",
            regularization_is_diagonal=False,
            regularization_is_sparse_spd=False,
            capabilities=None,
        )
        == "pcg"
    )


def test_fused_strategy_reason_branches_and_detect_cache(monkeypatch):
    perf_caps.detect_performance_capabilities.cache_clear()
    monkeypatch.setattr(
        perf_caps,
        "probe_petsc_cuda_runtime",
        lambda: {
            "petsc_cuda_mat": True,
            "petsc_cuda_vec": False,
            "petsc_cuda_dense": True,
            "petsc_cuda": False,
        },
    )
    monkeypatch.setattr(perf_caps, "_has_pyamg", lambda: True)
    monkeypatch.setattr(perf_caps, "_has_cholmod", lambda: False)
    monkeypatch.setattr(perf_caps, "_has_cuda_structured", lambda: True)
    monkeypatch.setattr(perf_caps, "_has_petsc_mat_solve", lambda: True)
    monkeypatch.setattr(perf_caps, "_has_petsc_gamg", lambda: False)
    monkeypatch.setattr(perf_caps, "_petsc_runtime_cache_key", lambda: ("stub",))

    caps = perf_caps.detect_performance_capabilities()
    assert caps["pyamg"] is True
    assert caps["cuda_structured"] is True
    assert caps["petsc_cuda_dense"] is True

    monkeypatch.setattr(
        perf_caps,
        "detect_performance_capabilities",
        lambda: {"cholmod": False},
    )
    lowrank_reason = perf_caps.select_fused_strategy(
        solver_mode="fast",
        mesh_dim=3,
        n_param=12000,
        n_meas=1000,
        rom_mode="on",
        inexact_mode="on",
        lowrank_mode="auto",
        regularization_is_diagonal=False,
        capabilities=None,
    )
    assert lowrank_reason["enabled"] is True
    assert lowrank_reason["reason"] == "enabled_without_lowrank"

    inexact_reason = perf_caps.select_fused_strategy(
        solver_mode="fast",
        mesh_dim=3,
        n_param=12000,
        n_meas=4000,
        rom_mode="on",
        inexact_mode="auto",
        lowrank_mode="on",
        regularization_is_diagonal=True,
        capabilities=None,
    )
    assert inexact_reason["enabled"] is True
    assert inexact_reason["lowrank"] is True
    assert inexact_reason["reason"] == "enabled_without_inexact"
