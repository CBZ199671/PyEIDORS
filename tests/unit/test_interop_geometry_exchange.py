#!/usr/bin/env python3
"""Tests for the standardized EIDORS <-> PyEIDORS interop helpers."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from pyeidors.data.structures import EITMesh
import pyeidors.interop.geometry_exchange as geometry_exchange_module
from pyeidors.interop import (
    STANDARD_INTEROP_FORMAT,
    build_boundary_edges,
    build_electrode_arrays,
    build_mesh_from_exchange_mat,
    export_forward_csv,
    load_forward_csv,
    save_exchange_mat,
    validate_exchange_payload,
)


def make_standard_payload() -> dict[str, object]:
    return {
        "exchange_format": STANDARD_INTEROP_FORMAT,
        "source_framework": "pyeidors",
        "nodes": np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=float,
        ),
        "elems": np.array([[1, 2, 3], [1, 3, 4]], dtype=np.int64),
        "boundary_edges": np.array([[1, 2], [2, 3], [3, 4], [4, 1]], dtype=np.int64),
        "electrode_nodes": np.array([[1, 2], [2, 3], [3, 4], [4, 1]], dtype=np.int64),
        "electrode_node_counts": np.array([2, 2, 2, 2], dtype=np.int64),
        "n_elec": 4,
        "background": 1.0,
        "truth_elem_data": np.array([1.0, 2.0], dtype=float),
        "contact_impedance": 1e-6,
        "mesh_name": "unit_square",
        "mesh_level": "unit",
        "scenario_name": "unit_case",
    }


def test_forward_csv_roundtrip(tmp_path: Path) -> None:
    baseline = np.array([1.0, 2.0, 3.0], dtype=float)
    phantom = np.array([1.5, 1.0, 4.0], dtype=float)
    out_csv = tmp_path / "forward.csv"

    export_forward_csv(out_csv, baseline, phantom)
    loaded_baseline, loaded_phantom, loaded_diff = load_forward_csv(out_csv)

    np.testing.assert_allclose(loaded_baseline, baseline)
    np.testing.assert_allclose(loaded_phantom, phantom)
    np.testing.assert_allclose(loaded_diff, phantom - baseline)


def test_validate_exchange_payload_rejects_missing_fields() -> None:
    payload = make_standard_payload()
    payload.pop("electrode_nodes")

    with pytest.raises(ValueError, match="missing required fields"):
        validate_exchange_payload(payload)


def test_save_exchange_mat_persists_standard_metadata(tmp_path: Path) -> None:
    out_mat = tmp_path / "exchange.mat"
    payload = make_standard_payload()

    save_exchange_mat(out_mat, payload)
    loaded = loadmat(out_mat, squeeze_me=True, struct_as_record=False)

    assert (
        str(np.asarray(loaded["exchange_format"]).reshape(-1)[0])
        == STANDARD_INTEROP_FORMAT
    )
    assert str(np.asarray(loaded["source_framework"]).reshape(-1)[0]) == "pyeidors"
    np.testing.assert_allclose(
        np.asarray(loaded["truth_elem_data"], dtype=float).reshape(-1), [1.0, 2.0]
    )


def test_build_mesh_from_exchange_mat_standard_payload(tmp_path: Path) -> None:
    out_mat = tmp_path / "exchange.mat"
    save_exchange_mat(out_mat, make_standard_payload())

    mesh, payload = build_mesh_from_exchange_mat(out_mat)

    assert isinstance(mesh, EITMesh)
    assert mesh.mesh_name == "unit_square"
    assert mesh.exchange_format == STANDARD_INTEROP_FORMAT
    assert mesh.n_electrodes == 4
    assert mesh.num_vertices() == 4
    assert mesh.num_cells() == 2
    assert mesh.facet_tags is not None
    assert mesh.cell_tags is not None
    assert mesh.association_table == {
        "domain": 1,
        "electrode_1": 2,
        "electrode_2": 3,
        "electrode_3": 4,
        "electrode_4": 5,
        "gaps": 6,
    }

    boundary_edges = build_boundary_edges(mesh)
    assert {tuple(sorted(edge)) for edge in boundary_edges.tolist()} == {
        (1, 2),
        (2, 3),
        (3, 4),
        (1, 4),
    }

    electrode_nodes, electrode_counts = build_electrode_arrays(mesh)
    assert electrode_counts.tolist() == [2, 2, 2, 2]
    assert {
        tuple(sorted(row[:count].tolist()))
        for row, count in zip(electrode_nodes, electrode_counts, strict=True)
    } == {
        (1, 2),
        (2, 3),
        (3, 4),
        (1, 4),
    }
    np.testing.assert_allclose(
        np.asarray(payload["truth_elem_data"], dtype=float).reshape(-1), [1.0, 2.0]
    )


def test_v698_exchange_facet_tags_survive_dolfinx_vertex_reordering(
    tmp_path: Path,
) -> None:
    n_electrodes = 16
    n_boundary = 2 * n_electrodes
    theta = 2.0 * np.pi * np.arange(n_boundary, dtype=np.float64) / n_boundary
    nodes = np.vstack(
        (
            np.zeros((1, 2), dtype=np.float64),
            np.column_stack((np.cos(theta), np.sin(theta))),
        )
    )
    elems = np.asarray(
        [[1, 2 + edge, 2 + ((edge + 1) % n_boundary)] for edge in range(n_boundary)],
        dtype=np.int64,
    )
    boundary_edges = np.asarray(
        [[2 + edge, 2 + ((edge + 1) % n_boundary)] for edge in range(n_boundary)],
        dtype=np.int64,
    )
    electrode_nodes = boundary_edges[::2].copy()
    payload = make_standard_payload()
    payload.update(
        {
            "nodes": nodes,
            "elems": elems,
            "boundary_edges": boundary_edges,
            "electrode_nodes": electrode_nodes,
            "electrode_node_counts": np.full(n_electrodes, 2, dtype=np.int64),
            "n_elec": n_electrodes,
            "truth_elem_data": np.ones(n_boundary, dtype=np.float64),
            "mesh_name": "reordered_fan",
        }
    )
    out_mat = tmp_path / "reordered_fan.mat"
    save_exchange_mat(out_mat, payload)

    mesh, _ = build_mesh_from_exchange_mat(out_mat)

    assert mesh.num_vertices() == 33
    assert mesh.num_cells() == 32
    assert mesh.facet_tags is not None
    assert mesh.facet_tags.indices.size == 32
    for electrode in range(1, n_electrodes + 1):
        tag = mesh.association_table[f"electrode_{electrode}"]
        assert np.count_nonzero(mesh.facet_tags.values == tag) == 1
    gap_tag = mesh.association_table["gaps"]
    assert np.count_nonzero(mesh.facet_tags.values == gap_tag) == n_electrodes


def test_v300_boundary_edges_direct_fill_without_vstack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_mat = tmp_path / "exchange.mat"
    save_exchange_mat(out_mat, make_standard_payload())
    mesh, _payload = build_mesh_from_exchange_mat(out_mat)

    def _fail_vstack(*_args, **_kwargs):
        raise AssertionError("boundary edge assembly must not call np.vstack")

    monkeypatch.setattr(geometry_exchange_module.np, "vstack", _fail_vstack)
    source = inspect.getsource(geometry_exchange_module.build_boundary_edges)
    assert "np.vstack" not in source

    boundary_edges = geometry_exchange_module.build_boundary_edges(mesh)
    assert boundary_edges.shape == (4, 2)
    assert boundary_edges.dtype == np.int64


def test_v501_geometry_exchange_index_guards_use_min_reductions() -> None:
    electrode_source = inspect.getsource(
        geometry_exchange_module._load_standard_electrode_node_lists
    )
    connectivity_source = inspect.getsource(
        geometry_exchange_module._load_one_based_connectivity
    )

    assert "np.min(active_nodes, initial=1)" in electrode_source
    assert "np.min(data, initial=1)" in connectivity_source
    assert "np.any(active_nodes < 1)" not in electrode_source
    assert "np.any(data < 1)" not in connectivity_source
