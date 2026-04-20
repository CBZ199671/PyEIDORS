from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from eit_app.interop import (
    EidorsEnvironment,
    EidorsExportJob,
    EidorsScriptCaptureService,
)
from eit_app.interop import InteropBundleExporter, InteropBundleImporter, save_bridge_package
from eit_app.interop.bridge_package import default_manifest
from eit_app.i18n import t
from eit_app.models.forward_model_config import ForwardModelConfig
import eit_app.ui.path_explorer as path_explorer_module
from eit_app.ui.dialogs.interop_hub_dialog import InteropHubDialog
from eit_app.ui.main_window import EITWorkstation


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
    electrode_nodes = np.asarray([[index + 2] for index in range(n_elec)], dtype=np.int64)
    electrode_counts = np.ones(n_elec, dtype=np.int64)
    return {
        "exchange_format": "eidors_pyeidors_bridge_v1",
        "source_framework": "eidors",
        "nodes": nodes,
        "elems": elems_array,
        "boundary_edges": boundary_edges,
        "electrode_nodes": electrode_nodes,
        "electrode_node_counts": electrode_counts,
        "n_elec": n_elec,
        "background": 1.0,
        "truth_elem_data": np.ones(len(elems_array), dtype=float),
        "contact_impedance": 0.01,
        "mesh_name": "unit_test_mesh",
        "mesh_level": "unit",
        "scenario_name": "unit_case",
    }


def _make_bridge_dir(tmp_path: Path, *, n_elec: int = 8) -> Path:
    config = ForwardModelConfig(n_elec=n_elec, stim_pattern="{ad}", meas_pattern="{ad}")
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
        geometry_payload=_standard_geometry_payload(n_elec=n_elec),
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
    assert preview.measurement_summary["points"] == str(preview.forward_model_config.point_count())
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


@pytest.mark.gui
def test_interop_hub_can_preview_and_import_into_simulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _get_app()
    bridge_dir = _make_bridge_dir(tmp_path, n_elec=8)
    window = EITWorkstation()
    window.show()
    _get_app().processEvents()

    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)

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
    assert window._tab_widget.currentWidget() is window._sim_tab

    dialog.close()
    window.close()


@pytest.mark.gui
def test_interop_hub_manual_smoke_callback_runs(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
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
def test_interop_hub_open_does_not_auto_detect_environments_and_has_no_diagnostics_tab() -> None:
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
        "Profiles & Paths",
    ]

    dialog.close()


def test_visual_path_roots_include_wsl_and_windows_mounts_in_wsl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(path_explorer_module, "running_in_wsl", lambda: True)
    monkeypatch.setattr(path_explorer_module, "running_on_windows", lambda: False)
    monkeypatch.setattr(path_explorer_module, "_available_windows_drives", lambda: ["/mnt/c", "/mnt/d"])

    roots = path_explorer_module.visual_path_roots()

    assert (t("path_picker.sidebar.wsl_home"), str(Path.home())) in roots
    assert (t("path_picker.sidebar.wsl_root"), "/") in roots
    assert ("Windows C:", "/mnt/c") in roots
    assert ("Windows D:", "/mnt/d") in roots


def test_visual_path_roots_include_only_windows_entries_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(path_explorer_module, "running_in_wsl", lambda: False)
    monkeypatch.setattr(path_explorer_module, "running_on_windows", lambda: True)
    monkeypatch.setattr(path_explorer_module, "_available_windows_drives", lambda: ["C:\\", "D:\\"])

    roots = path_explorer_module.visual_path_roots()

    assert (t("path_picker.sidebar.windows_home"), str(Path.home())) in roots
    assert ("Windows C:", "C:\\") in roots
    assert ("Windows D:", "D:\\") in roots
    assert all("WSL" not in label for label, _root in roots)
