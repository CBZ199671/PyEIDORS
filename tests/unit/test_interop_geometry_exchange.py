#!/usr/bin/env python3
"""Tests for the Bridge v3 EIDORS <-> PyEIDORS geometry helpers."""

from __future__ import annotations

import inspect
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import loadmat

from pyeidors.data.structures import EITImage, EITMesh
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
        "schema_version": 3,
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
        "schema_version": 3,
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


def test_v753_validate_exchange_payload_rejects_legacy_v1_2d() -> None:
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

    with pytest.raises(ValueError, match="Unsupported exchange format"):
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


def test_v753_build_3d_tetrahedron_mesh_from_geometry_v3(tmp_path: Path) -> None:
    out_mat = tmp_path / "tetrahedron.mat"
    save_exchange_mat(out_mat, make_tetrahedron_payload())

    mesh, payload = build_mesh_from_exchange_mat(out_mat)

    assert isinstance(mesh, EITMesh)
    assert mesh.mesh_name == "unit_tetrahedron"
    assert mesh.exchange_format == STANDARD_INTEROP_FORMAT
    assert mesh.mesh_family == "tetrahedron"
    assert mesh.geometry_version == "interop-v3"
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


def test_v747_point_electrodes_preserve_exact_nodes_without_facet_projection(
    tmp_path: Path,
) -> None:
    payload = make_standard_payload()
    source_background = np.array([10.0, 20.0])
    source_target = np.array([30.0, 40.0])
    payload["background_elem_data"] = source_background
    payload["target_elem_data"] = source_target
    payload["electrode_nodes"] = np.array([1, 2, 3, 4], dtype=np.int64)
    payload["electrode_node_counts"] = np.ones(4, dtype=np.int64)
    payload["electrode_model"] = ["point"] * 4
    payload["effective_gnd_node"] = 1
    out_mat = tmp_path / "point_electrodes.mat"
    save_exchange_mat(out_mat, payload)

    mesh, imported = build_mesh_from_exchange_mat(out_mat)

    assert mesh.electrode_model == "pem"
    assert mesh.electrode_projection == "none"
    assert mesh.gnd_node_source == 0
    np.testing.assert_array_equal(mesh.point_electrode_source_nodes, [0, 1, 2, 3])
    np.testing.assert_array_equal(
        imported["background_elem_data"],
        source_background[mesh.source_cell_indices],
    )
    np.testing.assert_array_equal(
        imported["target_elem_data"],
        source_target[mesh.source_cell_indices],
    )
    np.testing.assert_array_equal(
        imported["truth_elem_data"],
        np.asarray(payload["truth_elem_data"])[mesh.source_cell_indices],
    )
    assert imported["element_data_order"] == "dolfinx_local"
    electrode_nodes, electrode_counts = build_electrode_arrays(mesh)
    np.testing.assert_array_equal(electrode_nodes, [[1], [2], [3], [4]])
    np.testing.assert_array_equal(electrode_counts, [1, 1, 1, 1])
    for electrode in range(1, 5):
        tag = mesh.association_table[f"electrode_{electrode}"]
        assert np.count_nonzero(mesh.facet_tags.values == tag) == 0


def test_v747_native_pem_uses_exact_node_currents_and_ignores_contact_impedance(
    tmp_path: Path,
) -> None:
    from pyeidors import EITSystem
    from pyeidors.data import PatternConfig

    payload = make_standard_payload()
    payload["electrode_nodes"] = np.array([1, 2, 3, 4], dtype=np.int64)
    payload["electrode_node_counts"] = np.ones(4, dtype=np.int64)
    payload["electrode_model"] = ["point"] * 4
    payload["effective_gnd_node"] = 1
    out_mat = tmp_path / "native_pem.mat"
    save_exchange_mat(out_mat, payload)
    mesh, _ = build_mesh_from_exchange_mat(out_mat)

    stim = np.array([[0.0, 1.0, -1.0, 0.0]])
    meas = [np.array([[1.0, 0.0, -1.0, 0.0]])]
    pattern = PatternConfig(
        n_elec=4,
        measurement_protocol="custom",
        custom_stim_matrix=stim,
        custom_meas_matrices=meas,
        drive_mode="total_current",
    )

    electrode_voltages = []
    for contact_impedance in (None, np.full(4, 1e9)):
        system = EITSystem(
            n_elec=4,
            pattern_config=pattern,
            electrode_model="pem",
            contact_impedance=contact_impedance,
            linear_backend="scipy",
            forward_backend="dolfinx",
        )
        system.setup(mesh=mesh, initialize_inverse=False)
        data, voltages = system.fwd_model.fwd_solve(
            EITImage(
                elem_data=np.ones(mesh.num_cells()),
                fwd_model=system.fwd_model,
            )
        )
        assert system.fwd_model.contact_impedance_applicable is False
        assert system.fwd_model.ground_dof >= 0
        assert system.fwd_model.point_electrode_matrix.shape == (4, 4)
        np.testing.assert_allclose(
            data.meas,
            system.fwd_model.pattern_manager.apply_meas_pattern(voltages),
        )
        electrode_voltages.append(voltages)

    np.testing.assert_allclose(
        electrode_voltages[0],
        electrode_voltages[1],
        rtol=0,
        atol=0,
    )


def test_v754_weighted_pem_preserves_exact_weights_without_projection(
    tmp_path: Path,
) -> None:
    payload = make_standard_payload()
    payload["electrode_nodes"] = np.array([[1, 3], [2, 4]], dtype=np.int64)
    payload["electrode_node_counts"] = np.array([2, 2], dtype=np.int64)
    payload["n_elec"] = 2
    payload["electrode_model"] = ["distributed_point", "distributed_point"]
    payload["pem_node_weights"] = np.array([[0.25, 0.75], [0.6, 0.4]])
    payload["effective_gnd_node"] = 1
    out_mat = tmp_path / "distributed_point_electrodes.mat"
    save_exchange_mat(out_mat, payload)

    mesh, _ = build_mesh_from_exchange_mat(out_mat)

    assert mesh.source_electrode_models == [
        "distributed_point",
        "distributed_point",
    ]
    assert mesh.electrode_model == "pem"
    assert mesh.electrode_projection == "none"
    assert [spec.kind for spec in mesh.electrode_specs] == ["pem", "pem"]
    np.testing.assert_allclose(mesh.electrode_specs[0].node_weights, [0.25, 0.75])
    np.testing.assert_allclose(mesh.electrode_specs[1].node_weights, [0.6, 0.4])
    for electrode in range(1, 3):
        tag = mesh.association_table[f"electrode_{electrode}"]
        assert np.count_nonzero(mesh.facet_tags.values == tag) == 0


def test_v754_weighted_pem_without_weights_fails_closed() -> None:
    payload = make_standard_payload()
    payload["electrode_nodes"] = np.array([[1, 3], [2, 4]], dtype=np.int64)
    payload["electrode_node_counts"] = np.array([2, 2], dtype=np.int64)
    payload["n_elec"] = 2
    payload["electrode_model"] = ["distributed_point", "distributed_point"]

    with pytest.raises(ValueError, match="requires N2E or pem_node_weights"):
        validate_exchange_payload(payload)


def test_v755_cem_n2e_preserves_augmented_electrode_unknowns() -> None:
    payload = make_standard_payload()
    n_nodes = np.asarray(payload["nodes"]).shape[0]
    n_elec = int(payload["n_elec"])
    payload["electrode_model"] = ["cem"] * n_elec
    payload["N2E"] = np.zeros((n_elec, n_nodes + n_elec))

    validate_exchange_payload(payload)

    payload["N2E"] = np.zeros((n_nodes + n_elec, n_elec))
    validate_exchange_payload(payload)

    payload["N2E"] = np.zeros((n_elec, n_nodes - 1))
    with pytest.raises(ValueError, match="n_system_unknowns >= n_nodes"):
        validate_exchange_payload(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("index_base", 0, "index_base"),
        ("dimension", 2, "inconsistent"),
        ("cell_type", "triangle", "requires cell_type"),
    ],
)
def test_v753_geometry_v3_rejects_contract_drift(
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


def test_v753_geometry_v3_rejects_disagreeing_boundary_aliases() -> None:
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
