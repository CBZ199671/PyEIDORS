#!/usr/bin/env python3
"""Tests for the standardized EIDORS <-> PyEIDORS interop helpers."""

from __future__ import annotations

import inspect
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import loadmat

from pyeidors.data.structures import EITMesh
import pyeidors.interop.geometry_exchange as geometry_exchange_module
from pyeidors.interop import (
    LEGACY_INTEROP_FORMAT,
    STANDARD_INTEROP_FORMAT,
    build_boundary_facets,
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
        "schema_version": 2,
        "index_base": 1,
        "source_framework": "pyeidors",
        "dimension": 2,
        "cell_type": "triangle",
        "boundary_entity_type": "edge",
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
        "boundary_facets": np.array([[1, 2], [2, 3], [3, 4], [4, 1]], dtype=np.int64),
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


def make_tetrahedron_payload() -> dict[str, object]:
    boundary_facets = np.array(
        [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]],
        dtype=np.int64,
    )
    return {
        "exchange_format": STANDARD_INTEROP_FORMAT,
        "schema_version": 2,
        "index_base": 1,
        "source_framework": "pyeidors",
        "dimension": 3,
        "cell_type": "tetrahedron",
        "boundary_entity_type": "triangle",
        "nodes": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        "elems": np.array([[1, 2, 3, 4]], dtype=np.int64),
        "boundary_edges": boundary_facets,
        "boundary_facets": boundary_facets,
        "electrode_nodes": boundary_facets.copy(),
        "electrode_node_counts": np.full(4, 3, dtype=np.int64),
        "n_elec": 4,
        "background": 1.0,
        "truth_elem_data": np.array([1.0], dtype=float),
        "contact_impedance": np.full(4, 0.01, dtype=float),
        "mesh_name": "unit_tetrahedron",
        "mesh_level": "unit",
        "scenario_name": "unit_3d_case",
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


def test_validate_exchange_payload_accepts_legacy_v1_2d() -> None:
    payload = make_standard_payload()
    payload["exchange_format"] = LEGACY_INTEROP_FORMAT
    for field in (
        "schema_version",
        "index_base",
        "dimension",
        "cell_type",
        "boundary_entity_type",
        "boundary_facets",
    ):
        payload.pop(field)

    validate_exchange_payload(payload)


def test_v739_validator_restores_squeezed_one_node_per_electrode_axis() -> None:
    payload = make_standard_payload()
    payload["electrode_nodes"] = np.array([1, 2, 3, 4], dtype=np.int64)
    payload["electrode_node_counts"] = np.ones(4, dtype=np.int64)

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


def test_v735_build_3d_tetrahedron_mesh_from_geometry_v2(tmp_path: Path) -> None:
    out_mat = tmp_path / "tetrahedron.mat"
    save_exchange_mat(out_mat, make_tetrahedron_payload())

    mesh, payload = build_mesh_from_exchange_mat(out_mat)

    assert isinstance(mesh, EITMesh)
    assert mesh.mesh_name == "unit_tetrahedron"
    assert mesh.exchange_format == STANDARD_INTEROP_FORMAT
    assert mesh.mesh_family == "tetrahedron"
    assert mesh.geometry_version == "interop-v2"
    assert mesh.electrode_projection == "exact_surface_nodes"
    assert mesh.topology.dim == 3
    assert mesh.geometry.dim == 3
    assert mesh.num_vertices() == 4
    assert mesh.num_cells() == 1
    boundary_facets = build_boundary_facets(mesh)
    assert boundary_facets.shape == (4, 3)
    assert {tuple(sorted(facet)) for facet in boundary_facets.tolist()} == {
        (1, 2, 3),
        (1, 2, 4),
        (1, 3, 4),
        (2, 3, 4),
    }
    with pytest.raises(ValueError, match="only supports 2D"):
        build_boundary_edges(mesh)
    for electrode in range(1, 5):
        tag = mesh.association_table[f"electrode_{electrode}"]
        assert np.count_nonzero(mesh.facet_tags.values == tag) == 1
    np.testing.assert_allclose(
        np.asarray(payload["contact_impedance"], dtype=float).reshape(-1),
        np.full(4, 0.01),
    )


def test_v740_point_electrodes_expand_to_unique_incident_facets(
    tmp_path: Path,
) -> None:
    payload = make_standard_payload()
    payload["electrode_nodes"] = np.array([1, 2, 3, 4], dtype=np.int64)
    payload["electrode_node_counts"] = np.ones(4, dtype=np.int64)
    out_mat = tmp_path / "point_electrodes.mat"
    save_exchange_mat(out_mat, payload)

    mesh, _ = build_mesh_from_exchange_mat(out_mat)

    assert mesh.electrode_projection == "incident_boundary_facets"
    for electrode in range(1, 5):
        tag = mesh.association_table[f"electrode_{electrode}"]
        assert np.count_nonzero(mesh.facet_tags.values == tag) >= 1


def test_v743_legacy_distributed_point_sets_require_projection(
    tmp_path: Path,
) -> None:
    payload = make_standard_payload()
    payload["electrode_nodes"] = np.array([[1, 3], [2, 4]], dtype=np.int64)
    payload["electrode_node_counts"] = np.array([2, 2], dtype=np.int64)
    payload["n_elec"] = 2
    out_mat = tmp_path / "distributed_point_electrodes.mat"
    save_exchange_mat(out_mat, payload)

    mesh, _ = build_mesh_from_exchange_mat(out_mat)

    assert mesh.source_electrode_models == [
        "distributed_point",
        "distributed_point",
    ]
    assert mesh.electrode_projection == "incident_boundary_facets"
    for electrode in range(1, 3):
        tag = mesh.association_table[f"electrode_{electrode}"]
        assert np.count_nonzero(mesh.facet_tags.values == tag) >= 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("index_base", 0, "index_base"),
        ("dimension", 2, "inconsistent"),
        ("cell_type", "triangle", "requires cell_type"),
    ],
)
def test_v735_geometry_v2_rejects_contract_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = make_tetrahedron_payload()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        validate_exchange_payload(payload)


def test_v733_imported_mesh_uses_active_dolfinx_geometry_dtype(
    tmp_path: Path,
) -> None:
    import dolfinx

    out_mat = tmp_path / "exchange.mat"
    save_exchange_mat(out_mat, make_standard_payload())

    mesh, _payload = build_mesh_from_exchange_mat(out_mat)

    assert mesh.mesh.geometry.x.dtype == np.dtype(dolfinx.default_real_type)


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
            "boundary_facets": boundary_edges,
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


def test_v738_geometry_v2_rejects_disagreeing_boundary_aliases() -> None:
    payload = make_standard_payload()
    payload["boundary_edges"] = np.array([[1, 2]], dtype=np.int64)

    with pytest.raises(ValueError, match="same boundary entities"):
        validate_exchange_payload(payload)


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


@pytest.mark.parametrize(
    "script_name",
    [
        "export_geometry_from_pyeidors.py",
        "import_geometry_from_eidors.py",
    ],
)
def test_v733_interop_cli_help_is_decoupled_from_reviewer_benchmark(
    script_name: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "interop" / script_name

    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    source = script_path.read_text(encoding="utf-8")
    assert "benchmark_reviewer_case" not in source
    assert "benchmark_difference_runtime" not in source
    if script_name == "export_geometry_from_pyeidors.py":
        assert "initialize_default_reconstructor" not in source
        assert "system.setup(mesh=mesh)" in source
        assert ".vector()" not in source
        assert "function_get_array(sigma)" in source
        assert "real_vector(baseline_data.meas" in source
        assert "real_vector(phantom_data.meas" in source
    else:
        assert ".dofmap()" not in source
        assert "imported_mesh.num_cells()" in source
        assert 'name="predicted voltage difference"' in source
        assert 'name="reconstructed conductivity"' in source
