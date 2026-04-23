"""Tests for unit-aware current drive semantics."""

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pyeidors.data.structures import EITImage, PatternConfig
from pyeidors.femx import build_eit_mesh
from pyeidors.forward.eit_forward_model import EITForwardModel
from pyeidors.physics.current_drive import (
    build_stim_currents,
    normalize_pattern_config_for_mesh,
    resolve_electrode_lengths_m,
    validate_drive_config,
)


def _build_square_eit_mesh(scale: float):
    mesh = dmesh.create_unit_square(MPI.COMM_WORLD, 18, 18)
    mesh.geometry.x[:, : mesh.geometry.dim] *= float(scale)
    tdim = mesh.topology.dim
    fdim = tdim - 1

    boundary_facets = dmesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
    ).astype(np.int32)
    mesh.topology.create_connectivity(fdim, 0)
    f2v = mesh.topology.connectivity(fdim, 0)

    coords = mesh.geometry.x[:, :2]
    centroids = np.zeros((boundary_facets.size, 2), dtype=float)
    for i, facet in enumerate(boundary_facets):
        vertices = f2v.links(int(facet))
        centroids[i, :] = coords[vertices].mean(axis=0)

    x = centroids[:, 0]
    y = centroids[:, 1]
    eps = 1e-10 * max(scale, 1.0)
    t = np.zeros_like(x)
    xmin, ymin = 0.0, 0.0
    xmax, ymax = float(scale), float(scale)
    left = np.isclose(x, xmin, atol=eps)
    top = (~left) & np.isclose(y, ymax, atol=eps)
    right = (~left) & (~top) & np.isclose(x, xmax, atol=eps)
    bottom = (~left) & (~top) & (~right) & np.isclose(y, ymin, atol=eps)
    t[left] = (y[left] - ymin) / (ymax - ymin)
    t[top] = 1.0 + (x[top] - xmin) / (xmax - xmin)
    t[right] = 2.0 + (ymax - y[right]) / (ymax - ymin)
    t[bottom] = 3.0 + (xmax - x[bottom]) / (xmax - xmin)
    seg_len = 4.0 / 16
    tags = (
        np.floor(np.clip(t, 0.0, 4.0 - 1e-12) / seg_len).astype(np.int32) + 2
    ).astype(np.int32)
    order = np.argsort(boundary_facets)
    facet_tags = dmesh.meshtags(mesh, fdim, boundary_facets[order], tags[order])
    association = {f"electrode_{idx + 1}": idx + 2 for idx in range(16)}
    return build_eit_mesh(
        mesh, facet_tags=facet_tags, association_table=association, radius=float(scale)
    )


def test_validate_drive_config_rules():
    assert (
        validate_drive_config(
            drive_mode="line_current_density",
            drive_value=1.0,
            geometry_scale_to_m=1.0,
            mesh_tdim=2,
        )
        == "line_current_density"
    )
    with pytest.raises(ValueError, match="drive_value"):
        validate_drive_config(
            drive_mode="total_current",
            drive_value=0.0,
            geometry_scale_to_m=1.0,
        )
    with pytest.raises(ValueError, match="geometry_scale_to_m"):
        validate_drive_config(
            drive_mode="normalized",
            drive_value=1.0,
            geometry_scale_to_m=0.0,
        )
    with pytest.raises(ValueError, match="2D meshes only"):
        validate_drive_config(
            drive_mode="line_current_density",
            drive_value=1.0,
            geometry_scale_to_m=1.0,
            mesh_tdim=3,
        )


def test_normalize_pattern_config_for_mesh_promotes_3d_line_density():
    pattern = PatternConfig(
        n_elec=16,
        drive_mode="line_current_density",
        drive_value=1.0,
    )
    normalized, diag = normalize_pattern_config_for_mesh(pattern, mesh_tdim=3)
    assert normalized.drive_mode == "total_current"
    assert diag == {
        "drive_mode_requested": "line_current_density",
        "drive_mode_effective": "total_current",
    }


def test_resolve_electrode_lengths_with_override():
    mesh_lengths = np.array([0.5, 0.5, 1.0])
    lengths_m = resolve_electrode_lengths_m(
        electrode_lengths_mesh=mesh_lengths,
        geometry_scale_to_m=0.01,
        electrode_length_m_override=None,
        n_elec=3,
    )
    assert np.allclose(lengths_m, np.array([0.005, 0.005, 0.01]))

    scalar_override = resolve_electrode_lengths_m(
        electrode_lengths_mesh=mesh_lengths,
        geometry_scale_to_m=0.01,
        electrode_length_m_override=0.02,
        n_elec=3,
    )
    assert np.allclose(scalar_override, np.array([0.02, 0.02, 0.02]))

    vector_override = resolve_electrode_lengths_m(
        electrode_lengths_mesh=mesh_lengths,
        geometry_scale_to_m=0.01,
        electrode_length_m_override=[0.01, 0.015, 0.02],
        n_elec=3,
    )
    assert np.allclose(vector_override, np.array([0.01, 0.015, 0.02]))

    with pytest.raises(ValueError, match="length mismatch"):
        resolve_electrode_lengths_m(
            electrode_lengths_mesh=mesh_lengths,
            geometry_scale_to_m=0.01,
            electrode_length_m_override=[0.01, 0.015],
            n_elec=3,
        )


def test_build_stim_currents_modes():
    indices = [0, 1]
    weights = np.array([1.0, -1.0])
    lengths = np.array([0.01, 0.02])

    total = build_stim_currents(
        drive_mode="total_current",
        drive_value=2.0,
        inj_indices=indices,
        inj_weights=weights,
        electrode_lengths_m=None,
    )
    assert np.allclose(total, np.array([2.0, -2.0]))

    normalized = build_stim_currents(
        drive_mode="normalized",
        drive_value=1.5,
        inj_indices=indices,
        inj_weights=weights,
        electrode_lengths_m=None,
    )
    assert np.allclose(normalized, np.array([1.5, -1.5]))

    line = build_stim_currents(
        drive_mode="line_current_density",
        drive_value=3.0,
        inj_indices=indices,
        inj_weights=weights,
        electrode_lengths_m=lengths,
    )
    assert np.allclose(line, np.array([0.03, -0.06]))

    with pytest.raises(ValueError, match="requires electrode_lengths_m"):
        build_stim_currents(
            drive_mode="line_current_density",
            drive_value=3.0,
            inj_indices=indices,
            inj_weights=weights,
            electrode_lengths_m=None,
        )


def test_line_current_density_voltage_invariance_m_vs_cm():
    mesh_m = _build_square_eit_mesh(scale=1.0)
    mesh_cm = _build_square_eit_mesh(scale=100.0)

    config_m = PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="line_current_density",
        drive_value=5e-5,
        geometry_scale_to_m=1.0,
        use_meas_current=False,
        rotate_meas=True,
    )
    config_cm = PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="line_current_density",
        drive_value=5e-5,
        geometry_scale_to_m=0.01,
        use_meas_current=False,
        rotate_meas=True,
    )

    z = np.full(16, 1e-5, dtype=float)
    model_m = EITForwardModel(n_elec=16, pattern_config=config_m, z=z, mesh=mesh_m)
    model_cm = EITForwardModel(n_elec=16, pattern_config=config_cm, z=z, mesh=mesh_cm)

    sigma_m = np.ones(
        int(
            model_m.V_sigma.dofmap.index_map.size_local
            * model_m.V_sigma.dofmap.index_map_bs
        )
    )
    sigma_cm = np.ones(
        int(
            model_cm.V_sigma.dofmap.index_map.size_local
            * model_cm.V_sigma.dofmap.index_map_bs
        )
    )

    data_m, _ = model_m.fwd_solve(EITImage(elem_data=sigma_m, fwd_model=model_m))
    data_cm, _ = model_cm.fwd_solve(EITImage(elem_data=sigma_cm, fwd_model=model_cm))

    amp_m = float(np.max(np.abs(data_m.meas)))
    amp_cm = float(np.max(np.abs(data_cm.meas)))
    ratio = amp_cm / amp_m
    assert 0.98 <= ratio <= 1.02
