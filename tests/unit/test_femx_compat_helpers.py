"""Coverage tests for femx compatibility helpers."""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem

from pyeidors.femx import (
    build_eit_mesh,
    cell_midpoints,
    create_ds_measure,
    estimate_radius,
    function_get_array,
    function_set_array,
    function_size,
    mesh_cell_vertices,
    mesh_coordinates,
    mesh_facet_vertices,
    mesh_num_cells,
    mesh_num_edges,
    mesh_num_vertices,
)


def test_mesh_helpers_and_function_array_ops(eit_system):
    mesh = eit_system.mesh.mesh

    coords = mesh_coordinates(mesh)
    cells = mesh_cell_vertices(mesh)
    facets = mesh_facet_vertices(mesh)
    mids = cell_midpoints(mesh)
    radius = estimate_radius(mesh)

    assert coords.ndim == 2 and coords.shape[1] == 2
    assert cells.shape[0] == mesh_num_cells(mesh)
    assert facets.shape[0] >= 0
    assert mids.shape[0] == mesh_num_cells(mesh)
    assert mesh_num_vertices(mesh) > 0
    assert mesh_num_edges(mesh) > 0
    assert radius > 0

    ds = create_ds_measure(mesh, eit_system.mesh.facet_tags)
    assert isinstance(ds, ufl.Measure)

    sigma = fem.Function(eit_system.fwd_model.V_sigma)
    vals = np.linspace(0.9, 1.1, sigma.x.array.size)
    function_set_array(sigma, vals)
    out = function_get_array(sigma)
    assert np.allclose(out, vals)
    assert function_size(sigma) == vals.size

    wrapped = build_eit_mesh(
        mesh,
        facet_tags=eit_system.mesh.facet_tags,
        association_table={"domain": 1, "electrode_1": 2},
        radius=1.23,
        mesh_file="dummy.msh",
    )
    assert wrapped.radius == 1.23
    assert wrapped.mesh_file == "dummy.msh"
    assert wrapped.association_table["domain"] == 1
