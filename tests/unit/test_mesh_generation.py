"""Geometry pipeline tests (gmsh path is mandatory)."""

from __future__ import annotations

import numpy as np
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pyeidors.femx import (
    build_eit_mesh,
    cell_midpoints,
    mesh_cell_vertices,
    mesh_coordinates,
)
from tests.utils import run_python


def test_gmsh_generated_files_exist(gmsh_mesh_artifacts):
    assert gmsh_mesh_artifacts["msh_file"].exists()
    assert gmsh_mesh_artifacts["association_file"].exists()


def test_mesh_loader_reads_gmsh_cache(gmsh_mesh_artifacts):
    code = f"""
from pyeidors.geometry.mesh_loader import MeshLoader
loader = MeshLoader(mesh_dir={str(gmsh_mesh_artifacts["mesh_dir"])!r})
mesh = loader.load_mesh({gmsh_mesh_artifacts["mesh_name"]!r})
assert mesh.num_cells() > 0
assert mesh.num_vertices() > 0
assert len(mesh.association_table) >= 8
print(mesh.mesh_file)
"""
    proc = run_python(code)
    assert proc.returncode == 0, proc.stderr


def test_mesh_converter_roundtrip(gmsh_mesh_artifacts):
    code = f"""
from pyeidors.geometry.mesh_converter import MeshConverter
converter = MeshConverter(
    mesh_file={str(gmsh_mesh_artifacts["msh_file"])!r},
    output_dir={str(gmsh_mesh_artifacts["mesh_dir"])!r},
)
mesh, facet_tags, association = converter.convert()
assert mesh.num_cells() > 0
assert facet_tags is not None
assert set(association.keys()) >= {{\"domain\", \"gaps\"}}
"""
    proc = run_python(code)
    assert proc.returncode == 0, proc.stderr


def test_build_eit_mesh_from_local_unit_square():
    mesh = dmesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    tdim = mesh.topology.dim
    fdim = tdim - 1
    facets = dmesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
    )
    facets = np.asarray(facets, dtype=np.int32)
    tags = np.full(facets.shape, 2, dtype=np.int32)
    order = np.argsort(facets)
    facet_tags = dmesh.meshtags(mesh, fdim, facets[order], tags[order])

    eit_mesh = build_eit_mesh(
        mesh,
        facet_tags=facet_tags,
        association_table={"electrode_1": 2},
        radius=1.0,
    )

    assert eit_mesh.num_cells() > 0
    assert eit_mesh.num_vertices() > 0
    assert mesh_coordinates(eit_mesh.mesh).shape[1] == 2
    assert mesh_cell_vertices(eit_mesh.mesh).shape[0] == eit_mesh.num_cells()
    assert cell_midpoints(eit_mesh.mesh).shape[0] == eit_mesh.num_cells()
