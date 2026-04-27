"""DOLFINx-native mesh cache tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pyeidors.geometry.dolfinx_mesh_cache import (
    dolfinx_cache_metadata_path_for_mesh,
    load_dolfinx_mesh_cache,
    write_dolfinx_mesh_cache,
    xdmf_cache_path_for_mesh,
)
import pyeidors.geometry.mesh_loader as mesh_loader_module


def _unit_square_mesh_data():
    mesh = dmesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    tdim = mesh.topology.dim
    fdim = tdim - 1
    cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    cell_tags = dmesh.meshtags(
        mesh, tdim, cells, np.full(cells.shape, 1, dtype=np.int32)
    )

    facets = dmesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
    ).astype(np.int32)
    order = np.argsort(facets)
    facet_tags = dmesh.meshtags(
        mesh,
        fdim,
        facets[order],
        np.full(facets.shape, 2, dtype=np.int32)[order],
    )

    class _Group:
        def __init__(self, dim: int, tag: int):
            self.dim = dim
            self.tag = tag

    return SimpleNamespace(
        mesh=mesh,
        facet_tags=facet_tags,
        cell_tags=cell_tags,
        physical_groups={
            "domain": _Group(2, 1),
            "electrode_1": _Group(1, 2),
        },
    )


def test_mesh_loader_prefers_dolfinx_xdmf_cache_over_gmsh(tmp_path: Path, monkeypatch):
    source_msh = tmp_path / "cached.msh"
    source_msh.write_text("placeholder source", encoding="utf-8")
    mesh_data = _unit_square_mesh_data()

    assert write_dolfinx_mesh_cache(
        mesh_data,
        source_msh_file=source_msh,
        association_table={"domain": 1, "electrode_1": 2},
        gdim=2,
    )
    assert xdmf_cache_path_for_mesh(source_msh).exists()
    assert dolfinx_cache_metadata_path_for_mesh(source_msh).exists()

    monkeypatch.setattr(
        mesh_loader_module.gmshio,
        "read_from_msh",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Gmsh .msh import should not run")
        ),
    )

    loaded = mesh_loader_module.MeshLoader(mesh_dir=str(tmp_path), gdim=2).load_mesh(
        "cached"
    )
    assert loaded.num_cells() > 0
    assert loaded.facet_tags is not None
    assert loaded.cell_tags is not None
    assert loaded.association_table == {"domain": 1, "electrode_1": 2}


def test_xdmf_cache_loads_after_source_msh_is_removed(tmp_path: Path, monkeypatch):
    source_msh = tmp_path / "hdf5_only.msh"
    source_msh.write_text("placeholder source", encoding="utf-8")
    sidecar = tmp_path / "hdf5_only_structured_sidecar.json"
    sidecar.write_text("{}", encoding="utf-8")
    mesh_data = _unit_square_mesh_data()
    association = {"domain": 1, "electrode_1": 2}

    assert write_dolfinx_mesh_cache(
        mesh_data,
        source_msh_file=source_msh,
        association_table=association,
        gdim=2,
        mesh_family="hex",
        geometry_version="geomv2",
        generator_revision="test-rev",
        structured_sidecar_file=sidecar,
        structured_sidecar_version="test-sidecar-v1",
    )
    xdmf_file = xdmf_cache_path_for_mesh(source_msh)
    source_msh.unlink()

    direct = load_dolfinx_mesh_cache(xdmf_file, gdim=2)
    assert direct is not None
    assert direct.source_msh_file is None
    assert direct.association_table == association
    assert direct.physical_groups["electrode_1"].tag == 2
    assert direct.facet_tags is not None
    assert direct.cell_tags is not None
    assert direct.metadata["structured_sidecar_file"] == str(sidecar)
    assert len(direct.metadata["artifact_key"]) == 64
    manifest = direct.metadata["artifact_manifest"]
    assert manifest["artifact_key"] == direct.metadata["artifact_key"]
    assert manifest["artifact_kind"] == "dolfinx-mesh-cache"
    assert manifest["key_payload"]["mesh_content_signature"]["geometry_hash"]
    assert manifest["key_payload"]["source_msh_signature"]["sha256"]
    assert manifest["files"]["xdmf"]["path"].endswith(".xdmf")
    assert manifest["files"]["hdf5"]["path"].endswith(".h5")

    direct_from_h5 = load_dolfinx_mesh_cache(xdmf_file.with_suffix(".h5"), gdim=2)
    assert direct_from_h5 is not None
    assert direct_from_h5.association_table == association

    monkeypatch.setattr(
        mesh_loader_module.gmshio,
        "read_from_msh",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Gmsh .msh import should not run")
        ),
    )

    loaded = mesh_loader_module.MeshLoader(mesh_dir=str(tmp_path), gdim=2).load_mesh(
        "hdf5_only"
    )
    assert loaded.mesh_file == str(xdmf_file)
    assert loaded.association_table == association
    assert loaded.mesh_family == "hex"
    assert loaded.geometry_version == "geomv2"
    assert loaded.generator_revision == "test-rev"
    assert loaded.structured_sidecar_file == str(sidecar)
    assert loaded.structured_sidecar_version == "test-sidecar-v1"


def test_legacy_xdmf_metadata_without_manifest_gets_in_memory_artifact_key(
    tmp_path: Path,
) -> None:
    source_msh = tmp_path / "legacy_manifest.msh"
    source_msh.write_text("placeholder source", encoding="utf-8")
    mesh_data = _unit_square_mesh_data()
    association = {"domain": 1, "electrode_1": 2}

    assert write_dolfinx_mesh_cache(
        mesh_data,
        source_msh_file=source_msh,
        association_table=association,
        gdim=2,
    )
    metadata_file = dolfinx_cache_metadata_path_for_mesh(source_msh)
    original = json.loads(metadata_file.read_text(encoding="utf-8"))
    expected_key = original["artifact_key"]
    original.pop("artifact_key", None)
    original.pop("artifact_manifest", None)
    metadata_file.write_text(json.dumps(original, sort_keys=True), encoding="utf-8")

    loaded = load_dolfinx_mesh_cache(xdmf_cache_path_for_mesh(source_msh), gdim=2)

    assert loaded is not None
    assert loaded.metadata["artifact_key"] == expected_key
    assert loaded.metadata["artifact_manifest"]["artifact_kind"] == "dolfinx-mesh-cache"
    persisted = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert "artifact_key" not in persisted
    assert "artifact_manifest" not in persisted
