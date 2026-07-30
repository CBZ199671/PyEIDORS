"""Exact weighted-PEM, mixed-electrode, and interior-CEM operator tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from dolfinx import fem

from pyeidors import EITSystem
from pyeidors.data import EITImage, PatternConfig
from pyeidors.interop import (
    STANDARD_INTEROP_FORMAT,
    build_mesh_from_exchange_mat,
    save_exchange_mat,
)


def _square_payload() -> dict[str, object]:
    return {
        "exchange_format": STANDARD_INTEROP_FORMAT,
        "schema_version": 3,
        "index_base": 1,
        "source_framework": "pyeidors",
        "dimension": 2,
        "cell_type": "triangle",
        "boundary_entity_type": "edge",
        "nodes": np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        "elems": np.asarray([[1, 2, 3], [1, 3, 4]], dtype=np.int64),
        "boundary_edges": np.asarray(
            [[1, 2], [2, 3], [3, 4], [4, 1]],
            dtype=np.int64,
        ),
        "boundary_facets": np.asarray(
            [[1, 2], [2, 3], [3, 4], [4, 1]],
            dtype=np.int64,
        ),
        "background": 1.0,
        "truth_elem_data": np.ones(2),
        "mesh_name": "mixed_operator_square",
        "mesh_level": "unit",
        "scenario_name": "mixed_operator",
    }


def _solve(
    tmp_path: Path,
    payload: dict[str, object],
    *,
    electrode_model: str,
    stim: np.ndarray,
    meas: np.ndarray,
    contact_impedance: np.ndarray | None = None,
):
    path = tmp_path / f"{electrode_model}.mat"
    save_exchange_mat(path, payload)
    mesh, _ = build_mesh_from_exchange_mat(path)
    pattern = PatternConfig(
        n_elec=int(payload["n_elec"]),
        measurement_protocol="custom",
        custom_stim_matrix=stim,
        custom_meas_matrices=[meas],
        drive_mode="total_current",
    )
    system = EITSystem(
        n_elec=int(payload["n_elec"]),
        pattern_config=pattern,
        electrode_model=electrode_model,
        contact_impedance=contact_impedance,
        linear_backend="scipy",
        forward_backend="dolfinx",
    )
    system.setup(mesh=mesh, initialize_inverse=False)
    data, electrode_voltages = system.fwd_model.fwd_solve(
        EITImage(
            elem_data=np.ones(mesh.num_cells()),
            fwd_model=system.fwd_model,
        )
    )
    return mesh, system.fwd_model, data, electrode_voltages


def test_v754_weighted_pem_uses_exact_w_for_current_and_voltage(
    tmp_path: Path,
) -> None:
    payload = _square_payload()
    payload.update(
        {
            "electrode_nodes": np.asarray([[1, 3], [2, 4]], dtype=np.int64),
            "electrode_node_counts": np.asarray([2, 2], dtype=np.int64),
            "electrode_model": ["distributed_point", "distributed_point"],
            "pem_node_weights": np.asarray([[0.25, 0.75], [0.6, 0.4]]),
            "n_elec": 2,
            "contact_impedance": np.asarray([np.nan, np.nan]),
            "contact_impedance_present": np.asarray([False, False]),
            "effective_gnd_node": 1,
        }
    )
    stim = np.asarray([[1.0, -1.0]])
    meas = np.asarray([[1.0, -1.0]])

    results = []
    for impedance in (None, np.asarray([1.0e-9, 1.0e9])):
        mesh, model, data, voltages = _solve(
            tmp_path / ("none" if impedance is None else "provenance"),
            payload,
            electrode_model="pem",
            stim=stim,
            meas=meas,
            contact_impedance=impedance,
        )
        w_matrix = model.point_electrode_matrix.toarray()
        np.testing.assert_allclose(np.sum(w_matrix, axis=1), [1.0, 1.0])
        np.testing.assert_allclose(
            np.sort(w_matrix[0, w_matrix[0] != 0]),
            [0.25, 0.75],
        )
        np.testing.assert_allclose(
            np.sort(w_matrix[1, w_matrix[1] != 0]),
            [0.4, 0.6],
        )
        sigma = fem.Function(model.V_sigma)
        sigma.x.array[:] = 1.0
        potentials, exact_voltages = model.forward_solve(sigma)
        np.testing.assert_allclose(
            exact_voltages[0],
            w_matrix @ potentials[0],
        )
        assert np.all(np.isfinite(data.meas))
        assert np.all(np.isfinite(voltages))
        results.append((data.meas.copy(), voltages.copy()))

    np.testing.assert_allclose(results[0][0], results[1][0], rtol=0, atol=0)
    np.testing.assert_allclose(results[0][1], results[1][1], rtol=0, atol=0)


def test_v754_mixed_cem_pem_keeps_order_and_cem_only_gauge(
    tmp_path: Path,
) -> None:
    payload = _square_payload()
    payload.update(
        {
            "electrode_nodes": np.asarray(
                [[1, 2], [3, 0], [3, 4], [4, 0]],
                dtype=np.int64,
            ),
            "electrode_node_counts": np.asarray([2, 1, 2, 1], dtype=np.int64),
            "electrode_model": ["cem", "point", "cem", "point"],
            "n_elec": 4,
            "contact_impedance": np.asarray([0.01, np.nan, 0.02, np.nan]),
            "contact_impedance_present": np.asarray([True, False, True, False]),
            "effective_gnd_node": 1,
        }
    )
    mesh, model, data, voltages = _solve(
        tmp_path,
        payload,
        electrode_model="mixed",
        stim=np.asarray([[1.0, -1.0, 0.0, 0.0]]),
        meas=np.asarray([[0.0, 1.0, 0.0, -1.0]]),
    )

    assert mesh.electrode_model == "mixed"
    assert model.cem_electrode_indices == (0, 2)
    assert model.pem_electrode_indices == (1, 3)
    gauge_row = model.M.getrow(model.dofs + model.n_elec).toarray().reshape(-1)
    np.testing.assert_allclose(
        gauge_row[model.dofs : model.dofs + model.n_elec],
        [1.0, 0.0, 1.0, 0.0],
    )
    assert np.all(np.isfinite(data.meas))
    assert voltages.shape == (1, 4)


def test_v754_interior_cem_uses_dS_and_solves(tmp_path: Path) -> None:
    payload = _square_payload()
    payload.update(
        {
            "electrode_nodes": np.asarray([[1, 3], [2, 3]], dtype=np.int64),
            "electrode_node_counts": np.asarray([2, 2], dtype=np.int64),
            "electrode_model": ["cem", "cem"],
            "electrode_boundary_kind": ["interior", "exterior"],
            "cem_face_nodes": np.asarray([[1, 3]], dtype=np.int64),
            "cem_face_node_counts": np.asarray([2], dtype=np.int64),
            "cem_face_electrode": np.asarray([1], dtype=np.int64),
            "n_elec": 2,
            "contact_impedance": np.asarray([0.01, 0.02]),
            "contact_impedance_present": np.asarray([True, True]),
        }
    )
    mesh, model, data, voltages = _solve(
        tmp_path,
        payload,
        electrode_model="cem",
        stim=np.asarray([[1.0, -1.0]]),
        meas=np.asarray([[1.0, -1.0]]),
    )

    assert mesh.interior_facet_tags is not None
    assert model.dS_electrodes is not None
    assert model.electrode_specs[0].boundary_kind == "interior"
    assert np.all(np.isfinite(data.meas))
    assert np.all(np.isfinite(voltages))
