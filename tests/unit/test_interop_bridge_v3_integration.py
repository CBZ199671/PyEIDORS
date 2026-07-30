"""Bridge Package v3 novice CLI and exact-model integration tests."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from eit_app.controllers.forward_solver_controller import (
    _setup_generated_forward_system,
)
from eit_app.interop import (
    EidorsBridgeRunner,
    EidorsEnvironment,
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
from pyeidors.interop import BridgeV3Package, ElectrodeSpec


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
        scenario_name="bridge_v3_unit",
    )
    return payload, config


def _make_v3_package(tmp_path: Path) -> Path:
    payload, config = _tetra_geometry_payload()
    root = tmp_path / "bridge_v3_3d"
    save_bridge_package(
        root,
        default_manifest(source_framework="pyeidors", package_kind="unit_test"),
        geometry_payload=payload,
        forward_model_config=config,
        include_run_in_eidors_script=True,
    )
    return root


def test_v753_v754_bridge_v3_validates_and_loads_exact_3d_config(
    tmp_path: Path,
) -> None:
    root = _make_v3_package(tmp_path)

    report = validate_bridge_package(root)
    loaded = InteropBundleImporter().load_package(root)
    preview = InteropBundleImporter().preview_loaded_package(loaded)
    config = preview.forward_model_config

    assert report["valid"] is True
    assert report["package_format"] == "eidors_pyeidors_bridge_v3"
    assert report["geometry_format"] == "eidors_pyeidors_geometry_v3"
    assert report["dimension"] == 3
    assert report["cell_type"] == "tetrahedron"
    assert report["n_nodes"] == 4
    assert report["n_elements"] == 1
    assert report["n_boundary_facets"] == 4
    assert report["n_electrodes"] == 4
    assert report["contact_impedance_present"] == [True] * 4
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


def test_v755_protocol_and_fields_files_are_the_only_runtime_authority(
    tmp_path: Path,
) -> None:
    geometry, config = _tetra_geometry_payload()
    protocol_stim = np.asarray(geometry["stim_matrix"]) * 0.125
    package = BridgeV3Package.write(
        tmp_path / "separate_authority",
        model={
            "schema_version": 3,
            "n_elec": 4,
            "dimension": 3,
            "potential_order": 1,
            "forward_model_config": config.to_mapping(),
        },
        geometry=geometry,
        protocol={
            "stim_matrix": protocol_stim,
            "meas_matrices": geometry["meas_matrices"],
            "measurement_counts": geometry["measurement_counts"],
            "stimulation_supported": True,
        },
        fields={
            "background": 2.5,
            "background_present": True,
            "background_elem_data": np.asarray([2.5]),
            "target_elem_data": np.asarray([3.0]),
        },
    )

    loaded = InteropBundleImporter().load_package(package.root)
    imported = (
        InteropBundleImporter().preview_loaded_package(loaded).forward_model_config
    )

    np.testing.assert_allclose(imported.custom_stim_matrix, protocol_stim)
    assert imported.background_conductivity == pytest.approx(2.5)
    np.testing.assert_allclose(
        loaded.protocol_payload["stim_matrix"],
        protocol_stim,
    )
    assert loaded.fields_payload["background"] == pytest.approx(2.5)


def test_v734_bridge_validator_rejects_unsafe_manifest_path(tmp_path: Path) -> None:
    root = _make_v3_package(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["geometry"]["path"] = "../geometry.mat"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_bridge_package(root)

    assert report["valid"] is False
    assert any(
        "Unsafe Bridge v3 relative path" in message for message in report["errors"]
    )


def test_v734_cli_validate_inspect_and_import_geometry(tmp_path: Path) -> None:
    root = _make_v3_package(tmp_path)
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
    root = _make_v3_package(tmp_path)
    config = InteropBundleImporter().preview_package(root)[1].forward_model_config
    calls: list[dict[str, object]] = []
    fake_system = SimpleNamespace(
        setup=lambda **kwargs: calls.append(kwargs),
    )

    with pytest.raises(ValueError, match="Unmanaged interop geometry"):
        _setup_generated_forward_system(
            fake_system,
            forward_cfg=config,
            runtime={},
        )

    assert calls == []


def test_v754_matlab_templates_preserve_geometry_v3_and_exact_protocol() -> None:
    assert "eidors_pyeidors_geometry_v3" in CAPTURE_SCRIPT_TEMPLATE
    assert "boundary_facets" in CAPTURE_SCRIPT_TEMPLATE
    assert "local_build_pattern_arrays" in CAPTURE_SCRIPT_TEMPLATE
    assert "'stim_matrix'" in CAPTURE_SCRIPT_TEMPLATE
    assert "'meas_matrices'" in CAPTURE_SCRIPT_TEMPLATE
    assert "protocol.stim_matrix" in RUN_IN_EIDORS_TEMPLATE
    assert "fmdl.stimulation(i).stim_pattern" in RUN_IN_EIDORS_TEMPLATE
    assert "fmdl.stimulation(i).meas_pattern" in RUN_IN_EIDORS_TEMPLATE
    assert "local_expand_logical_electrodes" in RUN_IN_EIDORS_TEMPLATE
    assert "logical_to_physical * stim_matrix(i, :)'" in RUN_IN_EIDORS_TEMPLATE
    assert "logical_meas * logical_to_physical'" in RUN_IN_EIDORS_TEMPLATE
    assert "local_assert_logical_n2e" in RUN_IN_EIDORS_TEMPLATE
    assert "local_assert_logical_v2meas" in RUN_IN_EIDORS_TEMPLATE


def test_v762_real_eidors_gate_compares_common_parameter_jacobian() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    acceptance_source = (
        repo_root / "scripts/interop/run_eidors_source_semantics_acceptance.py"
    ).read_text(encoding="utf-8")
    matlab_source = (
        repo_root / "examples/interop/export_bridge_jacobian_v3.m"
    ).read_text(encoding="utf-8")
    roundtrip_source = (
        repo_root / "examples/interop/validate_bridge_in_eidors.m"
    ).read_text(encoding="utf-8")

    assert "pem_native_eidors_jacobian_parity" in acceptance_source
    assert 'pem_parity["jacobian_relative_l2"] <= 5e-3' in acceptance_source
    assert "source_cell_data_to_local" in acceptance_source
    assert "EidorsJacobianAdapter" in acceptance_source
    assert "calc_jacobian(img_background)" in matlab_source
    assert "size(jacobian, 2) ~= size(fmdl.elems, 1)" in matlab_source
    assert "all(~logical(payload.contact_impedance_applicable(:)))" in roundtrip_source


def test_v755_public_matlab_v3_entry_points_are_shipped() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    exporter = (repo_root / "matlab" / "pyeidors_export_v3.m").read_text(
        encoding="utf-8"
    )
    importer = (repo_root / "matlab" / "pyeidors_import_v3.m").read_text(
        encoding="utf-8"
    )

    assert "function package_dir = pyeidors_export_v3(" in exporter
    assert "'Background'" in exporter
    assert "'Target'" in exporter
    assert "pyeidors-interop" in exporter
    assert "'--fwd-model-var'" in exporter
    assert "function imported = pyeidors_import_v3(" in importer
    assert "'eidors_pyeidors_bridge_v3'" in importer
    assert "'validate'" in importer
    assert "imported.fwd_model = fmdl" in importer


def test_v755_fresh_capture_staging_is_packaged_before_strict_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import eit_app.interop.services as services_module
    from scipy.io import savemat

    payload, _ = _tetra_geometry_payload()
    output = tmp_path / "fresh_capture"

    def fake_matlab_capture(command: list[str], *, timeout: int | None = None):
        del command, timeout
        output.mkdir(parents=True, exist_ok=True)
        savemat(output / "geometry.mat", payload)
        return 0, "", ""

    monkeypatch.setattr(
        services_module,
        "_run_command_capture",
        fake_matlab_capture,
    )
    source = tmp_path / "source.m"
    source.write_text("fmdl = mk_common_model('a3cr', 4);\n", encoding="utf-8")

    root = EidorsBridgeRunner().run_capture(
        EidorsEnvironment(
            name="unit",
            matlab_command="/opt/matlab/bin/matlab",
            eidors_startup="/opt/eidors/startup.m",
        ),
        source,
        output,
    )

    report = validate_bridge_package(root)
    assert report["valid"] is True
    assert (root / "manifest.json").is_file()
    assert (root / "protocol.mat").is_file()
    assert (root / "fields.mat").is_file()


def _make_source_semantics_package(
    tmp_path: Path,
    *,
    point_electrodes: bool = False,
    missing_contact_impedance: bool = False,
    raw_stim_scale: float = 1.0,
    effective_stim_scale: float = 1.0,
) -> Path:
    payload, config = _tetra_geometry_payload()
    payload["stim_matrix_raw"] = _custom_protocol()[0] * raw_stim_scale
    payload["stim_matrix"] = _custom_protocol()[0] * effective_stim_scale
    payload["stimulation_supported"] = True
    payload["current_density"] = (
        raw_stim_scale / effective_stim_scale
        if raw_stim_scale != effective_stim_scale
        else np.nan
    )
    payload["current_density_present"] = raw_stim_scale != effective_stim_scale
    payload["current_density_applied"] = raw_stim_scale != effective_stim_scale
    if point_electrodes:
        payload["electrode_nodes"] = np.arange(1, 5, dtype=np.int64).reshape(-1, 1)
        payload["electrode_node_counts"] = np.ones(4, dtype=np.int64)
        payload["electrode_model"] = ["point"] * 4
        payload["pem_node_weights"] = np.ones((4, 1))
        payload["cem_face_nodes"] = np.empty((0, 3), dtype=np.int64)
        payload["cem_face_node_counts"] = np.empty(0, dtype=np.int64)
        payload["cem_face_electrode"] = np.empty(0, dtype=np.int64)
        payload["electrode_projection_required"] = np.ones(4, dtype=bool)
        metadata = json.loads(str(payload["capture_metadata_json"]))
        metadata["electrode_models"] = ["point"] * 4
        payload["capture_metadata_json"] = json.dumps(metadata)
    if missing_contact_impedance:
        payload["contact_impedance"] = np.full(4, np.nan)
        payload["contact_impedance_present"] = np.zeros(4, dtype=bool)
    root = tmp_path / (
        "source_semantics_point"
        if point_electrodes
        else "source_semantics_missing_contact"
    )
    save_bridge_package(
        root,
        default_manifest(source_framework="eidors", package_kind="unit_test"),
        geometry_payload=payload,
        forward_model_config=config,
    )
    return root


def test_v741_capture_template_has_discovery_provenance_without_fake_defaults() -> None:
    assert "local_discover_workspace" in CAPTURE_SCRIPT_TEMPLATE
    assert "fwd_model_var" in CAPTURE_SCRIPT_TEMPLATE
    assert "background_image_var" in CAPTURE_SCRIPT_TEMPLATE
    assert "target_image_var" in CAPTURE_SCRIPT_TEMPLATE
    assert "data_mapper(working)" in CAPTURE_SCRIPT_TEMPLATE
    assert "convert_img_units(mapped, 'conductivity')" in CAPTURE_SCRIPT_TEMPLATE
    assert "capture_metadata.model.coarse2fine" in CAPTURE_SCRIPT_TEMPLATE
    assert "capture_metadata.model.model_reduction" in CAPTURE_SCRIPT_TEMPLATE
    assert "stim_matrix = stim_matrix ./ current_density;" in CAPTURE_SCRIPT_TEMPLATE
    assert "contact_impedance(i) = 0.01" not in CAPTURE_SCRIPT_TEMPLATE
    assert "median(img.elem_data" not in CAPTURE_SCRIPT_TEMPLATE
    assert "truth_elem_data = ones" not in CAPTURE_SCRIPT_TEMPLATE


def test_v744_import_uses_eidors_effective_current_pattern(tmp_path: Path) -> None:
    root = _make_source_semantics_package(
        tmp_path,
        raw_stim_scale=0.02,
        effective_stim_scale=0.01,
    )

    preview = InteropBundleImporter().preview_package(root)[1]
    config = preview.forward_model_config

    np.testing.assert_allclose(
        np.asarray(config.custom_stim_matrix),
        _custom_protocol()[0] * 0.01,
    )
    assert config.interop_semantics["forward_ready"] is True
    config.require_interop_forward_ready()


def test_v743_missing_contact_impedance_is_preserved_and_blocks_forward(
    tmp_path: Path,
) -> None:
    root = _make_source_semantics_package(
        tmp_path,
        missing_contact_impedance=True,
    )

    report = validate_bridge_package(root)
    preview = InteropBundleImporter().preview_package(root)[1]
    config = preview.forward_model_config

    assert report["valid"] is True
    assert report["forward_ready"] is False
    assert "contact_impedance_missing_no_eidors_default" in report["forward_blockers"]
    assert config.contact_impedance is None
    with pytest.raises(ValueError, match="contact_impedance_missing"):
        config.require_interop_forward_ready()


def test_v748_point_electrodes_route_to_native_pem_without_projection_opt_in(
    tmp_path: Path,
) -> None:
    root = _make_source_semantics_package(tmp_path, point_electrodes=True)

    report = validate_bridge_package(root)
    preview = InteropBundleImporter().preview_package(root)[1]
    config = preview.forward_model_config

    assert report["valid"] is True
    assert report["electrode_models"] == ["point"] * 4
    assert report["electrode_model"] == "pem"
    assert report["electrode_projection"] == "exact_weighted_n2e"
    assert report["forward_ready"] is True
    assert config.electrode_model == "pem"
    assert config.drive_mode == "total_current"
    assert config.potential_order == 1
    assert config.interop_semantics["contact_impedance_applicable"] is False
    assert config.interop_semantics["electrode_projection"] == "exact_weighted_n2e"
    config.require_interop_forward_ready()


def test_v749_missing_point_contact_is_not_a_physical_pem_blocker(
    tmp_path: Path,
) -> None:
    root = _make_source_semantics_package(
        tmp_path,
        point_electrodes=True,
        missing_contact_impedance=True,
    )

    report = validate_bridge_package(root)
    config = InteropBundleImporter().preview_package(root)[1].forward_model_config

    assert report["contact_impedance_present"] == [False] * 4
    assert (
        "contact_impedance_missing_no_eidors_default" not in report["forward_blockers"]
    )
    assert config.contact_impedance is None
    assert config.electrode_model == "pem"
    config.require_interop_forward_ready()


def test_v749_pem_export_uses_singleton_nodes_and_labeled_eidors_placeholder() -> None:
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
        n_elec=4,
        electrode_model="pem",
        measurement_protocol="custom",
        custom_stim_matrix=stim,
        custom_meas_matrices=meas,
        drive_mode="total_current",
        contact_impedance=None,
        interop_semantics={"effective_gnd_node": 2},
    )

    payload = build_geometry_payload_from_result(
        node_coords=nodes,
        cell_connectivity=cells,
        forward_model_config=config,
        boundary_facets=facets,
        electrode_nodes=np.arange(4, dtype=np.int64),
        electrode_node_counts=np.ones(4, dtype=np.int64),
    )
    metadata = json.loads(str(payload["capture_metadata_json"]))

    assert payload["electrode_model"] == ["point"] * 4
    np.testing.assert_array_equal(payload["electrode_node_counts"], [1, 1, 1, 1])
    np.testing.assert_array_equal(payload["electrode_nodes"], [[1], [2], [3], [4]])
    assert np.isnan(np.asarray(payload["contact_impedance"])).all()
    assert not np.asarray(payload["contact_impedance_present"], dtype=bool).any()
    assert not np.asarray(payload["contact_impedance_applicable"], dtype=bool).any()
    assert not np.asarray(
        payload["contact_impedance_physical_present"],
        dtype=bool,
    ).any()
    assert int(payload["effective_gnd_node"]) == 2
    assert metadata["fields"]["contact_impedance"]["status"] == "exact_config"


def test_v763_weighted_pem_and_cem_export_restores_singleton_protocol_axes(
    tmp_path: Path,
) -> None:
    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    config = ForwardModelConfig(
        mesh_dimension=2,
        n_elec=2,
        electrode_model="mixed",
        measurement_protocol="custom",
        custom_stim_matrix=np.asarray([[1.0, -1.0]]),
        custom_meas_matrices=[np.asarray([[1.0, -1.0]])],
        contact_impedance=None,
    )
    specs = (
        ElectrodeSpec(
            kind="cem",
            index_base=0,
            source_nodes=(0, 1),
            source_faces=((0, 1),),
            boundary_kind="exterior",
            contact_impedance=0.02,
            contact_impedance_present=True,
            contact_impedance_applicable=True,
        ),
        ElectrodeSpec(
            kind="pem",
            index_base=0,
            source_nodes=(2, 3),
            node_weights=(0.25, 0.75),
            boundary_kind="none",
            contact_impedance=None,
            contact_impedance_present=False,
            contact_impedance_applicable=False,
        ),
    )

    payload = build_geometry_payload_from_result(
        node_coords=nodes,
        cell_connectivity=cells,
        forward_model_config=config,
        electrode_specs=specs,
    )
    root = tmp_path / "mixed_exact"
    save_bridge_package(
        root,
        default_manifest(source_framework="pyeidors", package_kind="unit_test"),
        geometry_payload=payload,
        forward_model_config=config,
    )

    report = validate_bridge_package(root)
    imported = InteropBundleImporter().preview_package(root)[1].forward_model_config

    assert report["valid"] is True
    assert report["electrode_model"] == "mixed"
    assert report["electrode_projection"] == "exact_per_electrode"
    assert report["forward_ready"] is True
    assert report["contact_impedance_present"] == [True, False]
    assert imported.electrode_model == "mixed"
    np.testing.assert_allclose(payload["pem_node_weights"][1, :2], [0.25, 0.75])
    np.testing.assert_array_equal(payload["cem_face_nodes"], [[1, 2]])
    assert (
        "mixed_cem_pem_electrode_models_not_supported"
        not in (imported.interop_semantics["forward_blockers"])
    )


def test_v745_nonuniform_background_is_preserved_without_scalar_inference(
    tmp_path: Path,
) -> None:
    payload, config = _tetra_geometry_payload()
    payload["elems"] = np.vstack([payload["elems"], payload["elems"]])
    payload["background"] = np.nan
    payload["background_present"] = False
    payload["background_elem_data"] = np.array([1.0, 2.0])
    payload["background_elem_data_present"] = True
    payload["truth_elem_data"] = np.array([1.0, 1.0])
    payload["target_elem_data"] = np.array([1.0, 1.0])
    root = tmp_path / "source_semantics_nonuniform_background"
    save_bridge_package(
        root,
        default_manifest(source_framework="eidors", package_kind="unit_test"),
        geometry_payload=payload,
        forward_model_config=config,
    )

    report = validate_bridge_package(root)
    preview = InteropBundleImporter().preview_package(root)[1]
    config = preview.forward_model_config

    assert report["valid"] is True
    assert report["background_present"] is False
    assert "background_image_missing_or_unmappable" not in report["forward_blockers"]
    assert (
        "background_uses_exact_element_field_without_scalar_summary"
        in config.interop_semantics["forward_warnings"]
    )
    config.require_interop_forward_ready()


def test_v744_pyeidors_export_records_current_and_eidors_runtime_semantics() -> None:
    payload, _ = _tetra_geometry_payload()

    np.testing.assert_allclose(payload["stim_positive_current"], np.ones(4))
    np.testing.assert_allclose(payload["stim_negative_current"], np.ones(4))
    np.testing.assert_allclose(payload["stim_net_current"], np.zeros(4))
    np.testing.assert_allclose(payload["stim_max_abs_current"], np.ones(4))
    assert np.asarray(payload["stim_balanced"], dtype=bool).all()
    assert bool(payload["gnd_node_present"]) is False
    assert int(payload["effective_gnd_node"]) == 1
    assert bool(payload["normalize_measurements"]) is False
    assert "EIDORS has no universal z_contact default" in RUN_IN_EIDORS_TEMPLATE
    assert "payload.effective_gnd_node" in RUN_IN_EIDORS_TEMPLATE
    assert "payload.electrode_faces" in RUN_IN_EIDORS_TEMPLATE
    assert "cfg.measurements_mat" in RUN_IN_EIDORS_TEMPLATE


def test_v744_complex_measurements_use_mat_without_silent_real_cast(
    tmp_path: Path,
) -> None:
    payload, config = _tetra_geometry_payload()
    root = tmp_path / "complex_measurement_bridge"
    homogeneous = np.arange(4, dtype=float) + 1j * np.linspace(0.1, 0.4, 4)
    target = homogeneous + (0.01 + 0.02j)
    save_bridge_package(
        root,
        default_manifest(source_framework="pyeidors", package_kind="unit_test"),
        geometry_payload=payload,
        measurements={"homogeneous": homogeneous, "target": target},
        forward_model_config=config,
        include_run_in_eidors_script=True,
    )

    loaded = InteropBundleImporter().load_package(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    assert not (root / "measurements.csv").exists()
    assert (root / "measurements.mat").exists()
    assert manifest["files"]["measurements"]["path"] == "measurements.mat"
    np.testing.assert_allclose(loaded.measurements["homogeneous"], homogeneous)
    np.testing.assert_allclose(loaded.measurements["target"], target)
