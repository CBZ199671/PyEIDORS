"""Unit tests for 3D cylindrical mesh generation."""

from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from pyeidors.geometry.mesh3d_generator import (
    GMSH_AVAILABLE,
    STRUCTURED_SIDECAR_VERSION,
    create_cylinder_3d_eit_mesh,
    load_structured_sidecar,
)
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
def test_create_cylinder_3d_mesh_tags_and_measures(tmp_path):
    mesh = create_cylinder_3d_eit_mesh(
        n_elec=16,
        radius=0.25,
        height=0.2,
        refinement=3,
        electrode_coverage=0.5,
        output_dir=str(tmp_path),
        mesh_name="cyl3d_unit",
    )

    assert mesh.mesh.geometry.dim == 3
    assert (tmp_path / "cyl3d_unit.msh").exists()
    assert (tmp_path / "cyl3d_unit_association_table.ini").exists()

    association = mesh.association_table
    assert association.get("domain") == 1
    assert "gaps" in association
    for idx in range(1, 17):
        assert f"electrode_{idx}" in association

    ds = ufl.Measure("ds", domain=mesh.mesh, subdomain_data=mesh.facet_tags)
    one = fem.Constant(mesh.mesh, 1.0)
    measures = []
    for idx in range(1, 17):
        tag = association[f"electrode_{idx}"]
        value = mesh.comm.allreduce(
            fem.assemble_scalar(fem.form(one * ds(tag))),
            op=MPI.SUM,
        )
        measures.append(float(value))

    arr = np.asarray(measures, dtype=float)
    assert np.all(np.isfinite(arr))
    assert np.min(arr) > 0.0


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
def test_geomv2_tetra_electrode_measure_tracks_height_ratio(tmp_path):
    level_fractions = (0.1, 0.9)
    mesh_small = create_cylinder_3d_eit_mesh(
        n_elec=16,
        radius=0.25,
        height=0.2,
        refinement=3,
        electrode_coverage=0.5,
        electrode_height_ratio=0.2,
        electrode_level_fractions=level_fractions,
        output_dir=str(tmp_path),
        mesh_name="tetra_geomv2_small",
        mesh_family="tetra",
        geometry_version="geomv2",
    )
    mesh_large = create_cylinder_3d_eit_mesh(
        n_elec=16,
        radius=0.25,
        height=0.2,
        refinement=3,
        electrode_coverage=0.5,
        electrode_height_ratio=0.6,
        electrode_level_fractions=level_fractions,
        output_dir=str(tmp_path),
        mesh_name="tetra_geomv2_large",
        mesh_family="tetra",
        geometry_version="geomv2",
    )

    def _mean_electrode_measure(mesh):
        ds = ufl.Measure("ds", domain=mesh.mesh, subdomain_data=mesh.facet_tags)
        one = fem.Constant(mesh.mesh, 1.0)
        values = []
        for idx in range(1, 17):
            tag = mesh.association_table[f"electrode_{idx}"]
            value = mesh.comm.allreduce(
                fem.assemble_scalar(fem.form(one * ds(tag))),
                op=MPI.SUM,
            )
            values.append(float(value))
        return float(np.mean(values))

    assert _mean_electrode_measure(mesh_large) > _mean_electrode_measure(mesh_small)


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
def test_geomv2_tetra_zigzag_layout_resolves_all_electrodes(tmp_path):
    mesh = create_cylinder_3d_eit_mesh(
        n_elec=16,
        radius=0.25,
        height=0.2,
        refinement=3,
        electrode_coverage=0.5,
        electrode_height_ratio=0.18,
        electrode_level_fractions=(0.25, 0.75),
        output_dir=str(tmp_path),
        mesh_name="tetra_geomv2_zigzag",
        mesh_family="tetra",
        geometry_version="geomv2",
    )

    ds = ufl.Measure("ds", domain=mesh.mesh, subdomain_data=mesh.facet_tags)
    one = fem.Constant(mesh.mesh, 1.0)
    measures = []
    for idx in range(1, 17):
        tag = mesh.association_table[f"electrode_{idx}"]
        value = mesh.comm.allreduce(
            fem.assemble_scalar(fem.form(one * ds(tag))),
            op=MPI.SUM,
        )
        measures.append(float(value))

    arr = np.asarray(measures, dtype=float)
    assert np.all(np.isfinite(arr))
    assert np.min(arr) > 0.0


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
def test_geomv2_tetra_ring_order_matches_eidors_plane_order(tmp_path):
    mesh = create_cylinder_3d_eit_mesh(
        n_elec=16,
        radius=0.25,
        height=0.2,
        refinement=1,
        electrode_coverage=0.5,
        electrode_height_ratio=0.2,
        electrode_level_fractions=(0.25, 0.75),
        output_dir=str(tmp_path),
        mesh_name="tetra_geomv2_ring_order",
        mesh_family="tetra",
        geometry_version="geomv2",
        electrode_layout="ring_major",
    )

    coords = mesh.mesh.geometry.x[:, :3]
    fdim = mesh.mesh.topology.dim - 1
    mesh.mesh.topology.create_connectivity(fdim, 0)
    f2v = mesh.mesh.topology.connectivity(fdim, 0)
    z_means = []
    for idx in range(1, 17):
        tag = mesh.association_table[f"electrode_{idx}"]
        facets = mesh.facet_tags.indices[mesh.facet_tags.values == tag]
        z_values = []
        for facet in facets:
            z_values.extend(coords[f2v.links(int(facet)), 2].tolist())
        z_means.append(float(np.mean(z_values)))

    assert np.all(np.asarray(z_means[:8]) < 0.0)
    assert np.all(np.asarray(z_means[8:]) > 0.0)


def test_geomv2_hex_mesh_is_file_backed_and_pure_hex(tmp_path):
    mesh = create_cylinder_3d_eit_mesh(
        n_elec=16,
        radius=0.25,
        height=0.2,
        refinement=2,
        electrode_coverage=0.5,
        electrode_height_ratio=0.3,
        output_dir=str(tmp_path),
        mesh_name="hex_geomv2_unit",
        mesh_family="hex",
        geometry_version="geomv2",
    )

    assert mesh.mesh_family == "hex"
    assert mesh.geometry_version == "geomv2"
    assert mesh.mesh_file is not None
    assert (tmp_path / "hex_geomv2_unit.msh").exists()
    sidecar = tmp_path / "hex_geomv2_unit_structured.json"
    assert sidecar.exists()
    payload = load_structured_sidecar(sidecar)
    assert payload["version"] == STRUCTURED_SIDECAR_VERSION
    assert payload["generator_revision"] == "g3d3"
    assert payload["mesh_family"] == "hex"
    assert payload["geometry_version"] == "geomv2"
    assert payload["block_topology"] == ["core", "east", "north", "west", "south"]
    assert len(payload["structured_cell_to_block"]) == mesh.num_cells()
    assert len(payload["structured_node_to_mesh_node"]) == mesh.num_vertices()
    assert mesh.cells().shape[1] == 8
    assert mesh.association_table["domain"] == 1
    assert "gaps" in mesh.association_table


def test_load_or_create_mesh_rejects_unknown_electrode_layout(tmp_path):
    with pytest.raises(ValueError, match="Unsupported electrode_layout"):
        load_or_create_mesh(
            mesh_dir=str(tmp_path),
            n_elec=16,
            dimension=3,
            electrode_layout="band",
        )
