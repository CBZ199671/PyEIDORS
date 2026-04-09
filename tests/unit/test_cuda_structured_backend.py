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


def test_resolve_cuda_structured_runtime_additional_validation_paths(tmp_path, monkeypatch):
    mesh_file = tmp_path / "mesh.bad"
    mesh_file.write_text("$MeshFormat\n2.2 0 8\n", encoding="utf-8")
    sidecar = module.structured_sidecar_path_for_mesh(mesh_file)
    _write_sidecar(sidecar)

    monkeypatch.setattr(module, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(module, "meshio", object())

    with pytest.raises(ValueError, match="single-rank execution only"):
        module.resolve_cuda_structured_runtime(
            mesh_dim=3,
            mesh_file=str(mesh_file),
            mesh_family="hex",
            geometry_version="geomv2",
            generator_revision="g3d3",
            petsc_device_requested="cuda",
            scalar_type="real",
            mesh_comm_size=2,
        )

    with pytest.raises(ValueError, match="requires a Gmsh .msh mesh file"):
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

    mesh_file_ok = tmp_path / "mesh.msh"
    mesh_file_ok.write_text("$MeshFormat\n2.2 0 8\n", encoding="utf-8")
    _write_sidecar(module.structured_sidecar_path_for_mesh(mesh_file_ok))

    monkeypatch.setattr(module, "_torch_cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires torch.cuda"):
        module.resolve_cuda_structured_runtime(
            mesh_dim=3,
            mesh_file=str(mesh_file_ok),
            mesh_family="hex",
            geometry_version="geomv2",
            generator_revision="g3d3",
            petsc_device_requested="cuda",
            scalar_type="real",
            mesh_comm_size=1,
        )

    monkeypatch.setattr(module, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(module, "meshio", None)
    with pytest.raises(RuntimeError, match="requires meshio"):
        module.resolve_cuda_structured_runtime(
            mesh_dim=3,
            mesh_file=str(mesh_file_ok),
            mesh_family="hex",
            geometry_version="geomv2",
            generator_revision="g3d3",
            petsc_device_requested="cuda",
            scalar_type="real",
            mesh_comm_size=1,
        )


def test_resolve_cuda_structured_runtime_covers_missing_file_geometry_sidecar_revision_and_zero_rhs_pcg(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(module, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(module, "meshio", object())

    with pytest.raises(ValueError, match="file-backed 3D mesh"):
        module.resolve_cuda_structured_runtime(
            mesh_dim=3,
            mesh_file=None,
            mesh_family="hex",
            geometry_version="geomv2",
            generator_revision="g3d3",
            petsc_device_requested="cuda",
            scalar_type="real",
            mesh_comm_size=1,
        )

    mesh_file = tmp_path / "mesh.msh"
    mesh_file.write_text("$MeshFormat\n2.2 0 8\n", encoding="utf-8")
    _write_sidecar(module.structured_sidecar_path_for_mesh(mesh_file))

    with pytest.raises(ValueError, match="geometry_version='geomv2' only"):
        module.resolve_cuda_structured_runtime(
            mesh_dim=3,
            mesh_file=str(mesh_file),
            mesh_family="hex",
            geometry_version="legacy",
            generator_revision="g3d3",
            petsc_device_requested="cuda",
            scalar_type="real",
            mesh_comm_size=1,
        )

    _write_sidecar(
        module.structured_sidecar_path_for_mesh(mesh_file),
        generator_revision="g3d2",
    )
    with pytest.raises(ValueError, match="matching the mesh"):
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

    if module.torch is not None:
        matrix = csr_matrix(np.eye(2, dtype=np.float64))
        A = module.CudaStructuredForwardBackend._csr_to_torch(matrix, module.torch.device("cpu"))
        zeros = module.torch.zeros((2, 1), dtype=module.torch.float64)
        diag_inv = module.torch.ones((2, 1), dtype=module.torch.float64)
        solved, iterations = module.CudaStructuredForwardBackend._block_pcg(
            A,
            zeros,
            diag_inv=diag_inv,
            rtol=1e-10,
            atol=1e-12,
            max_it=8,
        )
        assert iterations == 0
        np.testing.assert_allclose(solved.detach().cpu().numpy(), np.zeros((2, 1), dtype=np.float64))


def test_cuda_structured_helper_methods_cover_remaining_edge_paths(monkeypatch):
    backend = module.CudaStructuredForwardBackend.__new__(module.CudaStructuredForwardBackend)
    backend.sidecar = {"blocks": []}
    assert backend._estimate_mg_levels() == 1

    backend.sidecar = {"structured_node_to_mesh_node": []}
    backend.model = type(
        "Model",
        (),
        {
            "mesh": type("Mesh", (), {"geometry": type("Geom", (), {"dim": 3})()})(),
            "V": type("VSpace", (), {"tabulate_dof_coordinates": staticmethod(lambda: np.zeros((3, 3), dtype=float))})(),
        },
    )()
    backend.mesh_file = "demo.msh"
    monkeypatch.setattr(backend, "_load_mesh_points", lambda: np.zeros((2, 3), dtype=float))
    with pytest.raises(RuntimeError, match="structured_node_to_mesh_node"):
        backend._build_dof_bijection()

    backend.sidecar = {"structured_node_to_mesh_node": [0, 1]}
    with pytest.raises(RuntimeError, match="node count mismatch"):
        backend._build_dof_bijection()

    backend.model = type(
        "Model",
        (),
        {
            "mesh": type("Mesh", (), {"geometry": type("Geom", (), {"dim": 3})()})(),
            "V": type("VSpace", (), {"tabulate_dof_coordinates": staticmethod(lambda: np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float))})(),
        },
    )()
    monkeypatch.setattr(backend, "_load_mesh_points", lambda: np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float))
    with pytest.raises(RuntimeError, match="exceeds tolerance"):
        backend._build_dof_bijection()

    backend.model = type(
        "Model",
        (),
        {
            "mesh": type("Mesh", (), {"geometry": type("Geom", (), {"dim": 3})()})(),
            "V": type("VSpace", (), {"tabulate_dof_coordinates": staticmethod(lambda: np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float))})(),
        },
    )()
    monkeypatch.setattr(backend, "_load_mesh_points", lambda: np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float))
    with pytest.raises(RuntimeError, match="not bijective"):
        backend._build_dof_bijection()

    if module.torch is not None:
        matrix = csr_matrix(np.eye(2, dtype=np.float64))
        rhs = module.torch.ones((2, 1), dtype=module.torch.float64)
        diag_inv = module.torch.ones((2, 1), dtype=module.torch.float64)
        with pytest.raises(RuntimeError, match="failed to converge"):
            module.CudaStructuredForwardBackend._block_pcg(
                module.CudaStructuredForwardBackend._csr_to_torch(matrix, module.torch.device("cpu")),
                rhs,
                diag_inv=diag_inv,
                rtol=0.0,
                atol=0.0,
                max_it=0,
            )

        build_backend = module.CudaStructuredForwardBackend.__new__(module.CudaStructuredForwardBackend)
        build_backend.model = type(
            "Model",
            (),
            {
                "V_sigma": object(),
                "dofs": 2,
                "_petsc_to_csr": staticmethod(lambda _mat: csr_matrix(np.diag([0.0, 1.0]))),
                "_assemble_conductivity_matrix": staticmethod(lambda _sigma: object()),
            },
        )()
        build_backend._top_left_robin = csr_matrix((2, 2), dtype=np.float64)
        build_backend._coupling_columns = np.zeros((2, 1), dtype=np.float64)
        build_backend.device = module.torch.device("cpu")
        build_backend._sigma_state = None
        monkeypatch.setattr(module.fem, "Function", lambda _space: type("Sigma", (), {"x": type("X", (), {"array": np.zeros(2, dtype=np.float64)})()})())
        with pytest.raises(RuntimeError, match="invalid diagonal"):
            build_backend._build_sigma_state(np.array([1.0, 2.0], dtype=np.float64))
