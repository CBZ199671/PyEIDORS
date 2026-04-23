"""Additional branch coverage for semantic cache object signatures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pyeidors.cache.object_signature as sig_module


@dataclass
class _DemoData:
    alpha: int
    beta: float


def _demo_callable(value):
    return value + 1


def test_normalize_callable_and_signature_payload_cover_edge_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        sig_module.inspect,
        "getsourcefile",
        lambda _func: (_ for _ in ()).throw(RuntimeError("no src")),
    )
    monkeypatch.setattr(
        sig_module.inspect,
        "getfile",
        lambda _func: (_ for _ in ()).throw(RuntimeError("no file")),
    )
    payload = sig_module._normalize_callable(_demo_callable)
    assert payload["module"] == __name__
    assert "source_hash" not in payload

    sample = tmp_path / "sample.txt"
    sample.write_text("hello", encoding="utf-8")
    normalized = sig_module._normalize_for_signature(
        {
            "arr": np.arange(4, dtype=np.float64).reshape(2, 2),
            "scalar": np.float64(1.5),
            "path": sample,
            "blob": b"abc",
            "items": ({3, 1, 2}, [_DemoData(alpha=1, beta=2.0)]),
            "callable": _demo_callable,
        }
    )
    assert normalized["arr"]["__ndarray__"] is True
    assert normalized["scalar"] == 1.5
    assert normalized["path"]["__path__"].endswith("sample.txt")
    assert normalized["blob"]["__bytes__"]
    assert normalized["items"][0] == [1, 2, 3]
    assert normalized["callable"]["__callable__"]["qualname"].endswith("_demo_callable")


def test_forward_model_signature_helpers_cover_comm_backend_and_model_paths(
    monkeypatch: pytest.MonkeyPatch,
):
    pattern_manager = SimpleNamespace(
        stim_matrix=np.array([[1.0, -1.0]], dtype=float),
        meas_matrices=[np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)],
        n_stim=1,
        n_meas_total=2,
        n_meas_per_stim=[2],
    )
    assert sig_module.pattern_signature_from_forward_model(
        SimpleNamespace(pattern_manager=pattern_manager)
    )

    assert (
        sig_module._forward_model_comm_size(
            SimpleNamespace(
                mesh=SimpleNamespace(comm=SimpleNamespace(Get_size=lambda: 4))
            )
        )
        == 4
    )
    assert (
        sig_module._forward_model_comm_size(
            SimpleNamespace(mesh=SimpleNamespace(comm=SimpleNamespace(size=3)))
        )
        == 3
    )

    class _BadComm:
        def Get_size(self):
            raise RuntimeError("boom")

    assert (
        sig_module._forward_model_comm_size(
            SimpleNamespace(mesh=SimpleNamespace(comm=_BadComm()))
        )
        == 1
    )

    assert sig_module._canonicalize_cuda_mat_type(None, comm_size=1) is None
    assert (
        sig_module._canonicalize_cuda_mat_type("AIJCUSPARSE", comm_size=1)
        == "seqaijcusparse"
    )
    assert (
        sig_module._canonicalize_cuda_mat_type("densecuda", comm_size=4)
        == "mpidensecuda"
    )
    assert sig_module._canonicalize_cuda_mat_type("custom", comm_size=1) == "custom"

    backend_cfg = SimpleNamespace(
        ksp_type="preonly",
        pc_type="lu",
        rtol=1e-10,
        atol=1e-12,
        max_it=200,
        reuse_preconditioner=True,
        mat_solve_mode="auto",
        petsc_device="auto",
    )
    cpu_model = SimpleNamespace(
        backend_config=backend_cfg,
        linear_backend="petsc",
        forward_backend="dolfinx",
        mesh_family="tetra",
        geometry_version="geomv2",
        generator_revision="g3d3",
        performance_mode="aggressive",
        mesh=SimpleNamespace(comm=SimpleNamespace(Get_size=lambda: 1)),
        eit_mesh=SimpleNamespace(structured_sidecar_version="v1"),
        _petsc_backend_info={"petsc_device_effective": "cpu"},
        _stable_cpu_petsc_types=lambda: ("seqaij", "seq"),
    )
    cpu_sig = sig_module.backend_signature_from_forward_model(cpu_model)
    assert cpu_sig

    bad_cpu_model = SimpleNamespace(**cpu_model.__dict__)
    bad_cpu_model._stable_cpu_petsc_types = lambda: (_ for _ in ()).throw(
        RuntimeError("bad stable types")
    )
    assert sig_module.backend_signature_from_forward_model(bad_cpu_model)

    cuda_model = SimpleNamespace(
        backend_config=backend_cfg,
        linear_backend="petsc",
        forward_backend="dolfinx",
        mesh_family="tetra",
        geometry_version="geomv2",
        generator_revision="g3d3",
        performance_mode="aggressive",
        mesh=SimpleNamespace(comm=SimpleNamespace(Get_size=lambda: 2)),
        eit_mesh=SimpleNamespace(structured_sidecar_version=None),
        _petsc_backend_info={
            "petsc_device_effective": "cuda",
            "petsc_mat_type": "AIJCUSPARSE",
            "petsc_vec_type": "cuda",
            "petsc_dense_mat_type": "densecuda",
            "forward_backend_effective": "dolfinx",
        },
    )
    cuda_sig = sig_module.backend_signature_from_forward_model(cuda_model)
    assert cuda_sig
    assert cpu_sig != cuda_sig


def test_model_signature_from_forward_model_covers_cached_mesh_file_array_and_missing_mesh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cached_model = SimpleNamespace(_semantic_model_signature="cached-hash")
    assert sig_module.model_signature_from_forward_model(cached_model) == "cached-hash"

    mesh_file = tmp_path / "demo.msh"
    mesh_file.write_text("mesh", encoding="utf-8")
    mesh = SimpleNamespace(
        association_table={"domain": 1},
        mesh=SimpleNamespace(topology=SimpleNamespace(dim=3)),
        mesh_file=str(mesh_file),
        mesh_family="tetra",
        geometry_version="geomv2",
        generator_revision="g3d3",
        structured_sidecar_file=None,
        structured_sidecar_version="v1",
    )
    model = SimpleNamespace(
        n_elec=16, z=np.ones(16, dtype=float), geometry_scale_to_m=1.0, eit_mesh=mesh
    )
    sig = sig_module.model_signature_from_forward_model(model)
    assert sig == model._semantic_model_signature

    monkeypatch.setattr(
        sig_module,
        "hash_path",
        lambda _path: (_ for _ in ()).throw(RuntimeError("hash failed")),
    )
    sig_fallback = sig_module.model_signature_from_forward_model(
        SimpleNamespace(
            n_elec=8,
            z=np.ones(8, dtype=float),
            geometry_scale_to_m=0.5,
            eit_mesh=mesh,
        )
    )
    assert sig_fallback

    coord_mesh = SimpleNamespace(
        association_table={"domain": 1},
        mesh=SimpleNamespace(topology=SimpleNamespace(dim=2)),
        mesh_file=None,
        mesh_family="tetra",
        geometry_version="legacy",
        generator_revision="g3d0",
        structured_sidecar_file=None,
        structured_sidecar_version=None,
        coordinates=lambda: np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        cells=lambda: np.array([[0, 1]], dtype=np.int32),
    )
    assert sig_module.model_signature_from_forward_model(
        SimpleNamespace(
            n_elec=4,
            z=np.ones(4, dtype=float),
            geometry_scale_to_m=1.0,
            eit_mesh=coord_mesh,
        )
    )

    missing_mesh_sig = sig_module.model_signature_from_forward_model(
        SimpleNamespace(
            n_elec=2, z=np.ones(2, dtype=float), geometry_scale_to_m=1.0, eit_mesh=None
        )
    )
    assert missing_mesh_sig
