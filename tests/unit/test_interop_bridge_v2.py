"""Bridge Package v2 novice CLI and exact-mesh integration tests."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from eit_app.controllers.forward_solver_controller import (
    _setup_generated_forward_system,
)
from eit_app.interop import (
    InteropBundleImporter,
    build_geometry_payload_from_result,
    save_bridge_package,
    validate_bridge_package,
)
from eit_app.interop.bridge_package import default_manifest
from eit_app.interop.matlab_templates import (
    CAPTURE_SCRIPT_TEMPLATE,
    RUN_IN_EIDORS_TEMPLATE,
)
from eit_app.models.forward_model_config import ForwardModelConfig


def _custom_protocol() -> tuple[np.ndarray, list[np.ndarray]]:
    stim = np.array(
        [
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
            [-1.0, 0.0, 0.0, 1.0],
        ]
    )
    meas = [
        np.array([[0.0, 0.0, 1.0, -1.0]]),
        np.array([[1.0, 0.0, 0.0, -1.0]]),
        np.array([[1.0, -1.0, 0.0, 0.0]]),
        np.array([[0.0, 1.0, -1.0, 0.0]]),
    ]
    return stim, meas


def _tetra_geometry_payload() -> tuple[dict[str, object], ForwardModelConfig]:
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cells = np.array([[0, 1, 2, 3]], dtype=np.int64)
    facets = np.array(
        [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]],
        dtype=np.int64,
    )
    stim, meas = _custom_protocol()
    config = ForwardModelConfig(
        mesh_dimension=3,
        mesh_family="tetrahedron",
        n_elec=4,
        n_rings=1,
        measurement_protocol="custom",
        custom_stim_matrix=stim,
        custom_meas_matrices=meas,
        contact_impedance=[0.01] * 4,
    )
    payload = build_geometry_payload_from_result(
        node_coords=nodes,
        cell_connectivity=cells,
        forward_model_config=config,
        truth_elem_data=np.array([1.0]),
        boundary_facets=facets,
        electrode_nodes=facets,
        electrode_node_counts=np.full(4, 3, dtype=np.int64),
        mesh_name="unit_tetrahedron",
        scenario_name="bridge_v2_unit",
    )
    return payload, config


def _make_v2_package(tmp_path: Path) -> Path:
    payload, config = _tetra_geometry_payload()
    root = tmp_path / "bridge_v2_3d"
    save_bridge_package(
        root,
        default_manifest(source_framework="pyeidors", package_kind="unit_test"),
        geometry_payload=payload,
        forward_model_config=config,
        include_run_in_eidors_script=True,
    )
    return root


def test_v734_v735_bridge_v2_validates_and_loads_exact_3d_config(
    tmp_path: Path,
) -> None:
    root = _make_v2_package(tmp_path)

    report = validate_bridge_package(root)
    loaded = InteropBundleImporter().load_package(root)
    preview = InteropBundleImporter().preview_loaded_package(loaded)
    config = preview.forward_model_config

    assert report["valid"] is True
    assert report["package_format"] == "eidors_pyeidors_bridge_v2"
    assert report["geometry_format"] == "eidors_pyeidors_geometry_v2"
    assert report["dimension"] == 3
    assert report["cell_type"] == "tetrahedron"
    assert report["n_nodes"] == 4
    assert report["n_elements"] == 1
    assert report["n_boundary_facets"] == 4
    assert report["n_electrodes"] == 4
    assert report["n_stimulations"] == 4
    assert report["n_measurements"] == 4
    assert config.mesh_source == "interop"
    assert Path(config.mesh_path) == (root / "geometry.mat").resolve()
    assert config.measurement_protocol == "custom"
    np.testing.assert_allclose(
        np.asarray(config.custom_stim_matrix),
        _custom_protocol()[0],
    )
    assert [matrix.shape for matrix in config.custom_meas_matrices] == [(1, 4)] * 4


def test_v734_bridge_validator_rejects_unsafe_manifest_path(tmp_path: Path) -> None:
    root = _make_v2_package(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["geometry"] = "../geometry.mat"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_bridge_package(root)

    assert report["valid"] is False
    assert any("safe relative path" in message for message in report["errors"])


def test_v734_cli_validate_inspect_and_import_geometry(tmp_path: Path) -> None:
    root = _make_v2_package(tmp_path)
    repo_root = Path(__file__).resolve().parents[2]

    for command in ("validate", "inspect", "import-geometry"):
        command_args = [command, str(root)]
        if command == "import-geometry":
            command_args.append("--forward-smoke")
        result = subprocess.run(
            [sys.executable, "-m", "pyeidors.interop", *command_args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        if command == "import-geometry":
            assert payload["status"] == "imported"
            assert payload["dimension"] == 3
            assert payload["n_boundary_facets"] == 4
            assert payload["forward_smoke"] == "passed"
            assert payload["forward_measurements_finite"] is True
            assert payload["n_forward_measurements"] == 4
        else:
            assert payload["valid"] is True


def test_v736_forward_setup_uses_imported_mesh_not_generator(tmp_path: Path) -> None:
    root = _make_v2_package(tmp_path)
    config = InteropBundleImporter().preview_package(root)[1].forward_model_config
    calls: list[dict[str, object]] = []
    fake_system = SimpleNamespace(
        setup=lambda **kwargs: calls.append(kwargs),
    )

    _setup_generated_forward_system(
        fake_system,
        forward_cfg=config,
        runtime={},
    )

    assert len(calls) == 1
    assert "mesh" in calls[0]
    assert calls[0]["mesh"].topology.dim == 3
    assert calls[0]["mesh"].num_cells() == 1
    assert calls[0]["initialize_inverse"] is False
    assert "mesh_source" not in calls[0]


def test_v736_matlab_templates_preserve_geometry_v2_and_exact_protocol() -> None:
    assert "eidors_pyeidors_geometry_v2" in CAPTURE_SCRIPT_TEMPLATE
    assert "boundary_facets" in CAPTURE_SCRIPT_TEMPLATE
    assert "local_build_pattern_arrays" in CAPTURE_SCRIPT_TEMPLATE
    assert "'stim_matrix'" in CAPTURE_SCRIPT_TEMPLATE
    assert "'meas_matrices'" in CAPTURE_SCRIPT_TEMPLATE
    assert "payload.stim_matrix" in RUN_IN_EIDORS_TEMPLATE
    assert "fmdl.stimulation(i).stim_pattern" in RUN_IN_EIDORS_TEMPLATE
    assert "fmdl.stimulation(i).meas_pattern" in RUN_IN_EIDORS_TEMPLATE
