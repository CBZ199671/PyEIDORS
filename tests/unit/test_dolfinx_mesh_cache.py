"""DOLFINx-native mesh cache tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pyeidors.geometry.dolfinx_mesh_cache import (
    dolfinx_cache_metadata_path_for_mesh,
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
