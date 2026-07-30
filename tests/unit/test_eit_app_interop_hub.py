from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from eit_app.interop import (
    EidorsEnvironment,
    EidorsExportJob,
    EidorsScriptCaptureService,
    InteropBridgeManifest,
)
from eit_app.interop import (
    InteropBundleExporter,
    InteropBundleImporter,
    save_bridge_package,
)
from eit_app.interop.bridge_package import default_manifest, load_bridge_package
from eit_app.interop.matlab_templates import (
    CAPTURE_SCRIPT_TEMPLATE,
    RUN_IN_EIDORS_TEMPLATE,
)
from eit_app.interop.services import EidorsBridgeRunner, InteropSmokeValidator
from eit_app.interop.environment import (
    _guess_host_os_from_path,
    matlab_runtime_path,
)
from eit_app.i18n import t
from eit_app.models.forward_model_config import ForwardModelConfig
import eit_app.ui.path_explorer as path_explorer_module
from eit_app.ui.dialogs.interop_hub_dialog import InteropHubDialog
from eit_app.ui.dialogs.model_asset_manager_dialog import ModelAssetManagerDialog
from eit_app.ui.main_window import EITWorkstation
from pyeidors.interop import ModelRegistry


def _get_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _standard_geometry_payload(*, n_elec: int = 8) -> dict:
    angles = np.linspace(0.0, 2.0 * np.pi, n_elec, endpoint=False)
    boundary_nodes = np.column_stack([np.cos(angles), np.sin(angles)])
    center = np.zeros((1, 2), dtype=float)
    nodes = np.vstack([center, boundary_nodes])
    elems = []
    for index in range(n_elec):
        a = index + 1
        b = ((index + 1) % n_elec) + 1
        elems.append([1, a + 1, b + 1])
    elems_array = np.asarray(elems, dtype=np.int64)
    boundary_edges = np.asarray(
        [[index + 2, ((index + 1) % n_elec) + 2] for index in range(n_elec)],
        dtype=np.int64,
    )
    electrode_nodes = np.asarray(
        [[index + 2] for index in range(n_elec)], dtype=np.int64
    )
    electrode_counts = np.ones(n_elec, dtype=np.int64)
    stim_matrix = np.zeros((n_elec, n_elec), dtype=float)
    meas_matrices = np.zeros((n_elec, 1, n_elec), dtype=float)
    for index in range(n_elec):
        stim_matrix[index, index] = 1.0
        stim_matrix[index, (index + 1) % n_elec] = -1.0
        meas_matrices[index, 0, (index + 2) % n_elec] = 1.0
        meas_matrices[index, 0, (index + 3) % n_elec] = -1.0
    return {
        "exchange_format": "eidors_pyeidors_geometry_v3",
        "source_framework": "eidors",
        "dimension": 2,
        "cell_type": "triangle",
        "boundary_entity_type": "edge",
        "nodes": nodes,
        "elems": elems_array,
        "boundary_edges": boundary_edges,
        "electrode_nodes": electrode_nodes,
        "electrode_node_counts": electrode_counts,
        "n_elec": n_elec,
        "background": 1.0,
        "background_elem_data": np.ones(len(elems_array), dtype=float),
        "target_elem_data": np.full(len(elems_array), 1.1, dtype=float),
        "truth_elem_data": np.ones(len(elems_array), dtype=float),
        "contact_impedance": 0.01,
        "contact_impedance_present": np.ones(n_elec, dtype=bool),
        "stim_matrix": stim_matrix,
        "stim_matrix_raw": stim_matrix,
        "meas_matrices": meas_matrices,
        "measurement_counts": np.ones(n_elec, dtype=np.int64),
        "normalize_measurements": False,
        "stimulation_supported": True,
        "effective_gnd_node": 1,
        "mesh_name": "unit_test_mesh",
        "mesh_level": "unit",
        "scenario_name": "unit_case",
    }


def _make_bridge_dir(tmp_path: Path, *, n_elec: int = 8) -> Path:
    geometry = _standard_geometry_payload(n_elec=n_elec)
    config = ForwardModelConfig(
        n_elec=n_elec,
        measurement_protocol="custom",
        custom_stim_matrix=geometry["stim_matrix"],
        custom_meas_matrices=[
            matrix for matrix in np.asarray(geometry["meas_matrices"])
        ],
        drive_mode="total_current",
        stim_pattern="{ad}",
        meas_pattern="{ad}",
    )
    n_points = config.point_count()
    measurements = {
        "homogeneous": np.linspace(0.0, 1.0, n_points, dtype=float),
        "target": np.linspace(0.1, 1.1, n_points, dtype=float),
        "difference": np.full(n_points, 0.1, dtype=float),
    }
    manifest = default_manifest(source_framework="eidors", package_kind="unit_test")
    root = tmp_path / "bridge_case"
    save_bridge_package(
        root,
        manifest,
        geometry_payload=geometry,
        measurements=measurements,
        forward_model_config=config,
        include_capture_script=True,
    )
    return root


def test_interop_import_preview_round_trip(tmp_path: Path) -> None:
    bridge_dir = _make_bridge_dir(tmp_path, n_elec=8)

    importer = InteropBundleImporter()
    loaded, preview = importer.preview_package(bridge_dir)

    assert loaded.forward_model_config is not None
    assert preview.forward_model_config.n_elec == 8
    assert preview.capability_report.can_import_geometry is True
    assert preview.capability_report.can_import_measurements is True
    assert preview.measurement_summary["points"] == str(
        preview.forward_model_config.point_count()
    )
    assert preview.geometry_summary["electrodes"] == "8"


def test_interop_exporter_writes_runtime_artifacts(tmp_path: Path) -> None:
    exporter = InteropBundleExporter()
    out_dir = tmp_path / "export_bridge"
    cfg = ForwardModelConfig(n_elec=8)
    measurements = {
        "homogeneous": np.linspace(0.0, 1.0, cfg.point_count(), dtype=float),
        "target": np.linspace(0.2, 1.2, cfg.point_count(), dtype=float),
    }
    env = EidorsEnvironment(
        name="MATLAB / EIDORS",
        matlab_command=r"C:\Program Files\MATLAB\R2025a\bin\matlab.exe",
        matlab_root=r"C:\Program Files\MATLAB\R2025a",
        eidors_startup=r"C:\eidors\startup.m",
    )
    root = exporter.export_bundle(
        EidorsExportJob(
            source_kind="simulation",
            output_dir=str(out_dir),
            include_geometry=True,
            include_measurements=True,
            include_scripts=True,
            source_name="simulation",
        ),
        forward_model_config=cfg,
        environment=env,
        geometry_payload=_standard_geometry_payload(n_elec=8),
        measurements=measurements,
    )

    assert (root / "manifest.json").exists()
    assert (root / "bridge_manifest.json").exists()
    assert (root / "bridge_runtime.json").exists()
    assert (root / "run_import_from_pyeidors.m").exists()
    payload = json.loads((root / "bridge_runtime.json").read_text(encoding="utf-8"))
    assert payload["stim_pattern"] == "{ad}"
    assert "startup.m" in payload["eidors_startup"]
    assert payload["measurements_csv"].endswith("measurements.csv")
    assert payload["measurements_mat"].endswith("measurements.mat")


@pytest.mark.gui
def test_interop_hub_can_preview_and_import_into_simulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _get_app()
    bridge_dir = _make_bridge_dir(tmp_path, n_elec=8)
    window = EITWorkstation()
    registry = ModelRegistry(tmp_path / "model_registry")
    window._bridge_model_registry = registry
    window.show()
    _get_app().processEvents()

    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda *args, **kwargs: QMessageBox.StandardButton.Ok,
    )
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.StandardButton.Ok
    )
    monkeypatch.setattr(
        QMessageBox, "critical", lambda *args, **kwargs: QMessageBox.StandardButton.Ok
    )

    dialog = InteropHubDialog(
        window,
        capture_service=EidorsScriptCaptureService(),
        importer=window._interop_importer,
        exporter=window._interop_exporter,
        export_snapshot_provider=window._interop_export_snapshots,
        apply_import_callback=window._apply_interop_import,
    )
    dialog._source_edit.setText(str(bridge_dir))
    dialog._generate_preview()

    assert dialog._loaded_bundle is not None
    assert dialog._apply_import_btn.isEnabled() is True
    assert dialog._source_table.rowCount() > 0
    assert dialog._mapping_table.rowCount() > 0

    dialog._auto_smoke_check.setChecked(False)
    dialog._import_target_combo.setCurrentIndex(1)  # simulation
    dialog._apply_import()
    _get_app().processEvents()

    sim_cfg = window._sim_tab.mesh_setup_panel.get_config()
    assert sim_cfg["n_electrodes"] == 8
    assert window._sim_forward_model_config.n_elec == 8
    imported_cfg = window._current_sim_forward_model_config()
    assert imported_cfg.mesh_source == "interop"
    registered = registry.get(imported_cfg.interop_semantics["model_id"])
    assert (
        Path(imported_cfg.mesh_path)
        == (registered.asset_path / "geometry.mat").resolve()
    )
    assert registry.bound_model("simulation").model_id == registered.model_id
    request = window._build_sim_forward_request(request_source="interop_test")
    assert request.forward_model_config["mesh_source"] == "interop"
    assert (
        Path(request.forward_model_config["mesh_path"])
        == (registered.asset_path / "geometry.mat").resolve()
    )
    assert window._tab_widget.currentWidget() is window._sim_tab

    dialog.close()
    window.close()


@pytest.mark.gui
def test_v761_model_asset_manager_lists_and_binds_v3_assets(
    tmp_path: Path,
) -> None:
    _get_app()
    bridge_dir = _make_bridge_dir(tmp_path, n_elec=8)
    registry = ModelRegistry(tmp_path / "registry")
    registered = registry.register(bridge_dir, display_name="EIDORS unit model")

    dialog = ModelAssetManagerDialog(None, registry=registry)
    dialog.show()
    _get_app().processEvents()

    assert dialog.table.rowCount() == 1
    assert dialog.table.item(0, 1).text() == registered.model_id
    assert registered.model_id in dialog.details.toPlainText()
    dialog._apply_selected_to_all()
    assert {flow: model.model_id for flow, model in registry.bindings().items()} == {
        "simulation": registered.model_id,
        "dataset": registered.model_id,
        "realtime": registered.model_id,
    }
    dialog.close()


@pytest.mark.gui
def test_interop_hub_manual_smoke_callback_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _get_app()
    called: list[bool] = []

    def _fake_export_snapshots() -> dict[str, dict[str, object]]:
        return {}

    def _fake_apply_import(_target: str, _bundle) -> str:
        return "ok"

    def _fake_smoke(_bundle) -> str:
        called.append(True)
        return "smoke ok"

    dialog = InteropHubDialog(
        None,
        capture_service=EidorsScriptCaptureService(),
        importer=InteropBundleImporter(),
        exporter=InteropBundleExporter(),
        export_snapshot_provider=_fake_export_snapshots,
        apply_import_callback=_fake_apply_import,
        smoke_validate_callback=_fake_smoke,
    )
    dialog._loaded_bundle = object()  # type: ignore[assignment]
    dialog._run_smoke_btn.setEnabled(True)

    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda *args, **kwargs: QMessageBox.StandardButton.Ok,
    )
    message = dialog._run_smoke_validation(show_dialog=False)

    assert called == [True]
    assert "smoke ok" in message
    assert "smoke ok" in dialog._validation_log.toPlainText()

    dialog.close()


@pytest.mark.gui
def test_interop_hub_uses_single_qt_path_picker_buttons() -> None:
    _get_app()
    dialog = InteropHubDialog(
        None,
        capture_service=EidorsScriptCaptureService(),
        importer=InteropBundleImporter(),
        exporter=InteropBundleExporter(),
    )

    texts = [button.text() for button in dialog.findChildren(QPushButton)]

    assert texts.count(t("dlg.interop.path_pick_button")) >= 4
    assert "WSL..." not in texts
    assert "Win..." not in texts
    assert "Open" not in texts

    dialog.close()


@pytest.mark.gui
def test_interop_hub_open_does_not_auto_detect_environments_and_has_no_diagnostics_tab() -> (
    None
):
    _get_app()
    called = {"detect": 0}

    class _FakeCaptureService:
        def detect_environments(self):
            called["detect"] += 1
            raise AssertionError("Dialog should not auto-detect environments on open.")

        def load_profiles(self):
            return []

        def save_profiles(self, profiles):
            return None

        def save_last_environment(self, environment):
            return None

        def test_matlab(self, environment):
            return True, "ok"

        def test_startup(self, environment):
            return True, "ok"

        def infer_startup_from_source(self, source_path):
            return ""

    dialog = InteropHubDialog(
        None,
        capture_service=_FakeCaptureService(),
        importer=InteropBundleImporter(),
        exporter=InteropBundleExporter(),
    )

    assert called["detect"] == 0
    assert dialog._tabs.count() == 3
    assert [dialog._tabs.tabText(index) for index in range(dialog._tabs.count())] == [
        "Import from EIDORS",
        "Export to EIDORS",
        # Raw tab text carries the Qt-escaped "&&" so the rendered label
        # shows a literal ampersand instead of a mnemonic underline.
        "Profiles && Paths",
    ]

    dialog.close()


@pytest.mark.gui
def test_v128_interop_profile_save_uses_profiles_page_fields(
    tmp_path: Path,
) -> None:
    _get_app()

    class _MemoryCaptureService:
        def __init__(self) -> None:
            self.profiles: list[EidorsEnvironment] = []

        def load_profiles(self) -> list[EidorsEnvironment]:
            return list(self.profiles)

        def save_profiles(self, profiles: list[EidorsEnvironment]) -> None:
            self.profiles = list(profiles)

    capture_service = _MemoryCaptureService()
    dialog = InteropHubDialog(
        None,
        capture_service=capture_service,  # type: ignore[arg-type]
        importer=InteropBundleImporter(),
        exporter=InteropBundleExporter(),
    )
    try:
        dialog._matlab_edit.setText("/wrong/import/matlab")
        dialog._startup_edit.setText("/wrong/import/startup.m")
        dialog._source_edit.setText("/wrong/import/source.m")
        dialog._capture_output_edit.setText("/wrong/import/output")

        dialog._profile_name_edit.setText("Profile Form")
        dialog._profile_matlab_edit.setText("/opt/MATLAB/bin/matlab")
        dialog._profile_startup_edit.setText("/opt/eidors/startup.m")
        dialog._profile_script_edit.setText(str(tmp_path / "case.m"))
        dialog._profile_output_edit.setText(str(tmp_path / "bridge"))
        dialog._save_current_profile()

        assert len(capture_service.profiles) == 1
        saved = capture_service.profiles[0]
        assert saved.name == "Profile Form"
        assert saved.matlab_command == "/opt/MATLAB/bin/matlab"
        assert saved.eidors_startup == "/opt/eidors/startup.m"
        assert saved.last_script_path == str(tmp_path / "case.m")
        assert saved.last_output_dir == str(tmp_path / "bridge")
        assert "wrong/import" not in saved.to_mapping()["matlab_command"]
    finally:
        dialog.close()


@pytest.mark.gui
def test_v128_interop_hub_secondary_menu_actions_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _get_app()
    bridge_dir = _make_bridge_dir(tmp_path, n_elec=8)
    picked_path = tmp_path / "picked_source"
    saved_profiles: list[EidorsEnvironment] = [
        EidorsEnvironment(
            name="Saved EIDORS",
            matlab_command="/opt/matlab/bin/matlab",
            eidors_startup="/opt/eidors/startup.m",
            last_script_path=str(bridge_dir),
            last_output_dir=str(tmp_path / "capture"),
        )
    ]

    class _MemoryCaptureService:
        def load_profiles(self) -> list[EidorsEnvironment]:
            return list(saved_profiles)

        def save_profiles(self, profiles: list[EidorsEnvironment]) -> None:
            saved_profiles[:] = list(profiles)

        def save_last_environment(self, _environment: EidorsEnvironment) -> None:
            return None

        def capture_or_load(
            self,
            source_path: str | Path,
            *,
            environment: EidorsEnvironment | None = None,
            output_dir: str | Path | None = None,
            selectors: dict[str, str] | None = None,
        ):
            return InteropBundleImporter().load_package(source_path)

    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda *args, **kwargs: QMessageBox.StandardButton.Ok,
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **kwargs: QMessageBox.StandardButton.Ok,
    )
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda *args, **kwargs: QMessageBox.StandardButton.Ok,
    )
    import eit_app.ui.dialogs.interop_hub_dialog as interop_dialog_module

    monkeypatch.setattr(
        interop_dialog_module,
        "pick_visual_path",
        lambda *args, **kwargs: str(picked_path),
    )

    applied_targets: list[str] = []
    smoke_calls: list[bool] = []
    cfg = ForwardModelConfig(n_elec=8)
    measurements = {
        "homogeneous": np.linspace(0.0, 1.0, cfg.point_count(), dtype=float),
        "target": np.linspace(0.1, 1.1, cfg.point_count(), dtype=float),
    }

    def _fake_apply(target: str, _bundle) -> str:
        applied_targets.append(target)
        return f"applied {target}"

    def _fake_smoke(_bundle) -> str:
        smoke_calls.append(True)
        return "smoke ok"

    def _snapshots() -> dict[str, dict[str, object]]:
        return {
            key: {
                "name": key,
                "forward_model_config": cfg,
                "geometry_payload": _standard_geometry_payload(n_elec=8),
                "measurements": measurements,
                "reconstruction_preset": None,
                "notes": [],
            }
            for key in ("simulation", "hardware", "recording")
        }

    dialog = InteropHubDialog(
        None,
        capture_service=_MemoryCaptureService(),  # type: ignore[arg-type]
        importer=InteropBundleImporter(),
        exporter=InteropBundleExporter(),
        export_snapshot_provider=_snapshots,
        apply_import_callback=_fake_apply,
        smoke_validate_callback=_fake_smoke,
    )
    try:
        dialog._generate_preview()
        assert dialog._loaded_bundle is None

        dialog._reload_current_bundle()
        assert t("dlg.interop.msg.bundle_no_preview") in dialog._import_status.text()

        dialog._browse_into(
            dialog._source_edit,
            title_key="dlg.interop.source.pick_title",
            mode="file_or_directory",
        )
        assert dialog._source_edit.text() == str(picked_path)

        script_path = tmp_path / "script_case.m"
        script_path.write_text("% EIDORS script placeholder\n", encoding="utf-8")
        dialog._matlab_edit.clear()
        dialog._startup_edit.clear()
        dialog._source_edit.setText(str(script_path))
        dialog._generate_preview()
        assert dialog._loaded_bundle is None

        dialog._env_combo.setCurrentIndex(1)
        assert dialog._matlab_edit.text() == "/opt/matlab/bin/matlab"
        assert dialog._startup_edit.text() == "/opt/eidors/startup.m"

        dialog._source_edit.setText(str(bridge_dir))
        dialog._generate_preview()
        assert dialog._loaded_bundle is not None
        assert dialog._apply_import_btn.isEnabled()
        assert dialog._run_smoke_btn.isEnabled()
        assert dialog._source_table.rowCount() > 0
        assert dialog._mapping_table.rowCount() > 0

        dialog._reload_current_bundle()
        assert dialog._preview is not None

        target_order = [
            "hardware",
            "simulation",
            "dataset",
            "all",
            "measurements",
            "geometry",
        ]
        for index, target in enumerate(target_order):
            dialog._import_target_combo.setCurrentIndex(index)
            dialog._apply_import()
            assert applied_targets[-1] == target
        assert applied_targets == target_order
        assert len(smoke_calls) == len(target_order)

        message = dialog._run_smoke_validation(show_dialog=False)
        assert "smoke ok" in message

        for index, source_kind in enumerate(("simulation", "hardware", "recording")):
            output_dir = tmp_path / f"export_{source_kind}"
            dialog._export_source_combo.setCurrentIndex(index)
            dialog._export_dir_edit.setText(str(output_dir))
            dialog._generate_export()
            assert (output_dir / "manifest.json").exists()
            assert (output_dir / "bridge_runtime.json").exists()

        dialog._profile_name_edit.setText("New Profile")
        dialog._profile_matlab_edit.setText("/new/matlab")
        dialog._profile_startup_edit.setText("/new/startup.m")
        dialog._profile_script_edit.setText(str(bridge_dir))
        dialog._profile_output_edit.setText(str(tmp_path / "new_capture"))
        dialog._save_current_profile()
        assert any(profile.name == "New Profile" for profile in saved_profiles)

        for row in range(dialog._profiles_list.count()):
            item = dialog._profiles_list.item(row)
            profile = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(profile, EidorsEnvironment) and profile.name == "New Profile":
                dialog._profiles_list.setCurrentRow(row)
                break
        dialog._remove_selected_profile()
        assert all(profile.name != "New Profile" for profile in saved_profiles)
    finally:
        dialog.close()


@pytest.mark.gui
def test_v128_tools_menu_opens_interop_hub_dialog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _get_app()
    import eit_app.ui.dialogs.interop_hub_dialog as interop_dialog_module

    opened: list[InteropHubDialog] = []

    def _fake_exec(dialog: InteropHubDialog) -> int:
        opened.append(dialog)
        return 0

    monkeypatch.setattr(
        interop_dialog_module.InteropHubDialog,
        "exec",
        _fake_exec,
    )

    window = EITWorkstation()
    window.show()
    _get_app().processEvents()
    try:
        window._action_interop_hub.trigger()
        _get_app().processEvents()
        assert opened
        dialog = opened[0]
        assert dialog._capture_service is window._interop_capture_service
        assert dialog._importer is window._interop_importer
        assert dialog._exporter is window._interop_exporter
        assert dialog._apply_import_callback is not None
        assert dialog._smoke_validate_callback is not None
    finally:
        for dialog in opened:
            dialog.close()
        window.close()


def test_v129_matlab_bridge_templates_match_real_eidors_roundtrip() -> None:
    assert "local_discover_workspace" in CAPTURE_SCRIPT_TEMPLATE
    assert "vars = evalin('caller', 'whos')" in CAPTURE_SCRIPT_TEMPLATE
    assert "nested_path = [name, '.fwd_model']" in CAPTURE_SCRIPT_TEMPLATE
    assert "Multiple EIDORS %s objects were discovered" in CAPTURE_SCRIPT_TEMPLATE

    assert "mk_common_gridmdl('backproj')" not in RUN_IN_EIDORS_TEMPLATE
    assert "pyeidors_bridge_homogeneous" in RUN_IN_EIDORS_TEMPLATE
    assert "EIDORS bridge measurements loaded" in RUN_IN_EIDORS_TEMPLATE


def test_v734_wsl_mounted_matlab_executable_uses_windows_runtime_paths() -> None:
    env = EidorsEnvironment(
        name="WSL-mounted Windows MATLAB",
        matlab_command="/mnt/d/Program Files/MATLAB/R2023b/bin/matlab.exe",
    )

    assert _guess_host_os_from_path(env.matlab_command) == "windows"
    assert (
        matlab_runtime_path(
            "/mnt/d/eidors/eidors/startup.m",
            env,
        )
        == r"D:\eidors\eidors\startup.m"
    )


def test_v733_smoke_validator_counts_complex_conductivity_without_real_cast() -> None:
    source = inspect.getsource(InteropSmokeValidator.validate)

    assert 'getattr(recon, "conductivity", np.asarray([]))' in source
    assert 'getattr(recon, "conductivity", np.asarray([])), dtype=float' not in source


def test_v129_bridge_runner_uses_tolerant_matlab_output_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import eit_app.interop.services as services_module

    calls: list[list[str]] = []

    def _fake_run_command_capture(command: list[str], *, timeout=None):
        calls.append(command)
        return 1, "", "MATLAB 输出包含非 UTF-8 字节但已容错解码"

    monkeypatch.setattr(
        services_module, "_run_command_capture", _fake_run_command_capture
    )

    source = tmp_path / "target.m"
    source.write_text("fmdl = struct();\n", encoding="utf-8")
    runner = EidorsBridgeRunner()
    env = EidorsEnvironment(
        name="MATLAB",
        matlab_command="/matlab/bin/matlab",
        eidors_startup="/eidors/startup.m",
    )

    with pytest.raises(RuntimeError, match="非 UTF-8"):
        output = tmp_path / "out"
        runner.run_capture(env, source, output)

    assert calls
    assert calls[0][1] == "-batch"
    request = json.loads((output / "capture_request.json").read_text(encoding="utf-8"))
    assert "work_dir" in request
    assert not Path(request["work_dir"]).exists()
    assert "addpath(script_dir)" in CAPTURE_SCRIPT_TEMPLATE


def test_v753_legacy_single_mat_package_fails_closed(
    tmp_path: Path,
) -> None:
    from scipy.io import savemat

    root = tmp_path / "bridge"
    root.mkdir()
    savemat(
        root / "geometry.mat",
        {
            "exchange_format": "eidors_pyeidors_bridge_v1",
            "source_framework": "eidors",
            "nodes": np.zeros((3, 2), dtype=float),
            "elems": np.array([[1, 2, 3]], dtype=np.int64),
            "n_elec": 8,
            "contact_impedance": 0.01,
        },
    )
    manifest = InteropBridgeManifest(
        files={"geometry": "geometry.mat"},
        source_framework="eidors",
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest.to_mapping()), encoding="utf-8"
    )

    with pytest.raises(
        ValueError,
        match="missing required file roles: model, protocol, fields",
    ):
        load_bridge_package(root)


def test_visual_path_roots_include_wsl_and_windows_mounts_in_wsl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(path_explorer_module, "running_in_wsl", lambda: True)
    monkeypatch.setattr(path_explorer_module, "running_on_windows", lambda: False)
    monkeypatch.setattr(
        path_explorer_module, "_available_windows_drives", lambda: ["/mnt/c", "/mnt/d"]
    )

    roots = path_explorer_module.visual_path_roots()

    assert (t("path_picker.sidebar.wsl_home"), str(Path.home())) in roots
    assert (t("path_picker.sidebar.wsl_root"), "/") in roots
    assert ("Windows C:", "/mnt/c") in roots
    assert ("Windows D:", "/mnt/d") in roots


def test_visual_path_roots_include_only_windows_entries_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(path_explorer_module, "running_in_wsl", lambda: False)
    monkeypatch.setattr(path_explorer_module, "running_on_windows", lambda: True)
    monkeypatch.setattr(
        path_explorer_module, "_available_windows_drives", lambda: ["C:\\", "D:\\"]
    )

    roots = path_explorer_module.visual_path_roots()

    assert (t("path_picker.sidebar.windows_home"), str(Path.home())) in roots
    assert ("Windows C:", "C:\\") in roots
    assert ("Windows D:", "D:\\") in roots
    assert all("WSL" not in label for label, _root in roots)
