"""Smoke coverage for same-mesh forward p-refinement."""

from __future__ import annotations

import numpy as np
from dolfinx import mesh as dmesh
from mpi4py import MPI

from eit_app.models.forward_model_config import ForwardModelConfig
from pyeidors.data.structures import PatternConfig
from pyeidors.femx import build_eit_mesh
from pyeidors.forward.eit_forward_model import EITForwardModel
from pyeidors.forward.process_setup_cache import clear_process_forward_setup_cache


def _dof_count(function_space) -> int:
    dofmap = function_space.dofmap
    return int(dofmap.index_map.size_local * dofmap.index_map_bs)


def _make_tagged_square(n_elec: int = 4):
    mesh = dmesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    fdim = mesh.topology.dim - 1
    boundary_facets = dmesh.locate_entities_boundary(
        mesh,
        fdim,
        lambda x: np.full(x.shape[1], True, dtype=bool),
    ).astype(np.int32)
    mesh.topology.create_connectivity(fdim, 0)
    f2v = mesh.topology.connectivity(fdim, 0)
    coords = mesh.geometry.x[:, :2]
    centroids = np.zeros((boundary_facets.size, 2), dtype=np.float64)
    for idx, facet in enumerate(boundary_facets):
        centroids[idx, :] = coords[f2v.links(int(facet))].mean(axis=0)
    x = centroids[:, 0]
    y = centroids[:, 1]
    t = np.zeros_like(x)
    left = np.isclose(x, 0.0)
    top = (~left) & np.isclose(y, 1.0)
    right = (~left) & (~top) & np.isclose(x, 1.0)
    bottom = (~left) & (~top) & (~right) & np.isclose(y, 0.0)
    t[left] = y[left]
    t[top] = 1.0 + x[top]
    t[right] = 2.0 + (1.0 - y[right])
    t[bottom] = 3.0 + (1.0 - x[bottom])
    tags = (
        np.floor(np.clip(t, 0.0, 4.0 - 1e-12) / (4.0 / n_elec)).astype(np.int32) + 2
    ).astype(np.int32)
    order = np.argsort(boundary_facets)
    facet_tags = dmesh.meshtags(mesh, fdim, boundary_facets[order], tags[order])
    association = {f"electrode_{idx + 1}": idx + 2 for idx in range(n_elec)}
    return build_eit_mesh(
        mesh,
        facet_tags=facet_tags,
        association_table=association,
        radius=1.0,
    )


def test_forward_model_p_refinement_raises_potential_dofs_only():
    clear_process_forward_setup_cache()
    eit_mesh = _make_tagged_square()
    pattern = PatternConfig(
        n_elec=4,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="line_current_density",
        drive_value=1.0,
    )
    z = np.full(4, 1e-3, dtype=float)

    p1 = EITForwardModel(
        n_elec=4,
        pattern_config=pattern,
        z=z,
        mesh=eit_mesh,
        linear_backend="scipy",
        potential_order=1,
    )
    p2 = EITForwardModel(
        n_elec=4,
        pattern_config=pattern,
        z=z,
        mesh=eit_mesh,
        linear_backend="scipy",
        potential_order=2,
    )

    assert p1.potential_order == 1
    assert p2.potential_order == 2
    assert p2.dofs > p1.dofs
    assert _dof_count(p2.V_sigma) == _dof_count(p1.V_sigma)
    assert _dof_count(p1.V_sigma) == eit_mesh.num_cells()
    assert p2.get_backend_diagnostics()["potential_order"] == 2


def test_forward_model_config_round_trips_potential_order_aliases():
    config = ForwardModelConfig.from_mapping({"p_order": 3})

    assert config.potential_order == 3
    assert config.to_mapping()["potential_order"] == 3


def test_forward_model_config_round_trips_complex_gpu_high_accuracy():
    config = ForwardModelConfig.from_mapping({"complex_gpu_high_accuracy": "true"})

    assert config.complex_gpu_high_accuracy is True
    assert config.to_mapping()["complex_gpu_high_accuracy"] is True
    assert (
        ForwardModelConfig.from_mapping(
            {"complex_gpu_high_accuracy": "false"}
        ).complex_gpu_high_accuracy
        is False
    )
