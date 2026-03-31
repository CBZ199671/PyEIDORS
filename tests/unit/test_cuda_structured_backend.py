"""Tests for the single-rank cuda_structured forward backend helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pyeidors.forward import cuda_structured_backend as module


def _write_sidecar(path: Path, *, generator_revision: str = "g3d3") -> None:
    payload = {
        "version": module.STRUCTURED_SIDECAR_VERSION,
        "mesh_family": "hex",
        "geometry_version": "geomv2",
        "generator_revision": generator_revision,
        "block_topology": ["core", "east", "north", "west", "south"],
        "blocks": [
            {
                "id": 0,
                "name": "core",
                "logical_cells": [4, 4, 4],
                "logical_nodes": [5, 5, 5],
            }
        ],
        "structured_node_to_mesh_node": [0, 1, 2, 3],
        "structured_cell_to_block": [0, 0],
        "structured_cell_local_ijk": [[0, 0, 0], [1, 0, 0]],
        "boundary_faces": [],
        "field_tags": {"domain": 1, "gaps": 18},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_resolve_cuda_structured_runtime_accepts_narrow_happy_path(tmp_path, monkeypatch):
    mesh_file = tmp_path / "mesh.msh"
    mesh_file.write_text("$MeshFormat\n2.2 0 8\n", encoding="utf-8")
    _write_sidecar(module.structured_sidecar_path_for_mesh(mesh_file))
    monkeypatch.setattr(module, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(module, "meshio", object())

    payload = module.resolve_cuda_structured_runtime(
        mesh_dim=3,
        mesh_file=str(mesh_file),
        mesh_family="hex",
        geometry_version="geomv2",
        generator_revision="g3d3",
        petsc_device_requested="cuda",
        scalar_type="real",
        mesh_comm_size=1,
    )

    assert payload["forward_backend_effective"] == "cuda_structured"
    assert payload["structured_sidecar_version"] == module.STRUCTURED_SIDECAR_VERSION
    assert payload["structured_backend_version"] == module.CUDA_STRUCTURED_BACKEND_VERSION
    assert payload["operator_backend"] == "torch-cuda"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "mesh_dim": 2,
                "mesh_file": "mesh.msh",
                "mesh_family": "hex",
                "geometry_version": "geomv2",
                "generator_revision": "g3d3",
                "petsc_device_requested": "cuda",
                "scalar_type": "real",
                "mesh_comm_size": 1,
            },
            "3D meshes only",
        ),
        (
            {
                "mesh_dim": 3,
                "mesh_file": "mesh.msh",
                "mesh_family": "tetra",
                "geometry_version": "geomv2",
                "generator_revision": "g3d3",
                "petsc_device_requested": "cuda",
                "scalar_type": "real",
                "mesh_comm_size": 1,
            },
            "mesh_family='hex' only",
        ),
        (
            {
                "mesh_dim": 3,
                "mesh_file": "mesh.msh",
                "mesh_family": "hex",
                "geometry_version": "geomv2",
                "generator_revision": "g3d2",
                "petsc_device_requested": "cuda",
                "scalar_type": "real",
                "mesh_comm_size": 1,
            },
            "generator_revision='g3d3' only",
        ),
        (
            {
                "mesh_dim": 3,
                "mesh_file": "mesh.msh",
                "mesh_family": "hex",
                "geometry_version": "geomv2",
                "generator_revision": "g3d3",
                "petsc_device_requested": "cpu",
                "scalar_type": "real",
                "mesh_comm_size": 1,
            },
            "requires petsc_device='cuda'",
        ),
        (
            {
                "mesh_dim": 3,
                "mesh_file": "mesh.msh",
                "mesh_family": "hex",
                "geometry_version": "geomv2",
                "generator_revision": "g3d3",
                "petsc_device_requested": "cuda",
                "scalar_type": "complex",
                "mesh_comm_size": 1,
            },
            "real-valued conductivity only",
        ),
    ],
)
def test_resolve_cuda_structured_runtime_rejects_unsupported_cases(tmp_path, monkeypatch, kwargs, match):
    mesh_file = tmp_path / "mesh.msh"
    mesh_file.write_text("$MeshFormat\n2.2 0 8\n", encoding="utf-8")
    _write_sidecar(module.structured_sidecar_path_for_mesh(mesh_file))
    monkeypatch.setattr(module, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(module, "meshio", object())
    payload = dict(kwargs)
    payload["mesh_file"] = str(mesh_file)

    with pytest.raises(ValueError, match=match):
        module.resolve_cuda_structured_runtime(**payload)


def test_resolve_cuda_structured_runtime_rejects_missing_sidecar(tmp_path, monkeypatch):
    mesh_file = tmp_path / "mesh.msh"
    mesh_file.write_text("$MeshFormat\n2.2 0 8\n", encoding="utf-8")
    monkeypatch.setattr(module, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(module, "meshio", object())

    with pytest.raises(ValueError, match="requires a structured sidecar"):
        module.resolve_cuda_structured_runtime(
            mesh_dim=3,
            mesh_file=str(mesh_file),
            mesh_family="hex",
            geometry_version="geomv2",
            generator_revision="g3d3",
            petsc_device_requested="cuda",
            scalar_type="real",
            mesh_comm_size=1,
        )


@pytest.mark.skipif(module.torch is None, reason="torch not importable in current runtime")
def test_block_pcg_solves_small_spd_system():
    device = "cuda" if module._torch_cuda_available() else "cpu"
    if device != "cuda":
        pytest.skip("CUDA runtime not available")

    matrix = csr_matrix(np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64))
    rhs_np = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    expected = np.linalg.solve(matrix.toarray(), rhs_np)
    A = module.CudaStructuredForwardBackend._csr_to_torch(matrix, module.torch.device(device))
    rhs = module.torch.as_tensor(rhs_np, device=device, dtype=module.torch.float64)
    diag_inv = module.torch.as_tensor(
        (1.0 / matrix.diagonal())[:, None],
        device=device,
        dtype=module.torch.float64,
    )

    solved, iterations = module.CudaStructuredForwardBackend._block_pcg(
        A,
        rhs,
        diag_inv=diag_inv,
        rtol=1e-12,
        atol=1e-14,
        max_it=128,
    )

    assert iterations > 0
    np.testing.assert_allclose(solved.detach().cpu().numpy(), expected, rtol=1e-10, atol=1e-10)
