"""Comprehensive button-driver smoke test.

Goal: instantiate EITWorkstation in offscreen Qt, walk every QPushButton on
every panel/dialog, and click each one without exception. Catches latent
crashes that the more-targeted tests in test_eit_app_gui_smoke.py miss.

Modal QFileDialog/QMessageBox calls are stubbed out so clicks don't block.
Hardware connect/disconnect uses the simulator with fps=0.
"""

from __future__ import annotations

import gc
import os
from unittest.mock import patch

import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QMessageBox,
    QPushButton,
    QWidget,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _get_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(autouse=True)
def _cleanup_after():
    _get_app()
    yield
    app = _get_app()
    app.processEvents()
    from eit_app.ui.main_window import EITWorkstation

    for w in list(app.topLevelWidgets()):
        try:
            if isinstance(w, EITWorkstation):
                w.close()
                deadline = 100
                while w._device_ctrl._thread.isRunning() and deadline:
                    app.processEvents()
                    deadline -= 1
            else:
                w.close()
        except Exception:
            pass
    app.processEvents()
    for obj in gc.get_objects():
        try:
            if isinstance(obj, QThread) and obj.isRunning():
                obj.requestInterruption()
                obj.quit()
                obj.wait(500)
        except Exception:
            pass
    try:
        from eit_app.ui.theme import _mode_listeners

        _mode_listeners.clear()
    except Exception:
        pass
    app.processEvents()


def _patch_modal_dialogs(stack):
    """Replace every modal dialog call with a no-op so clicks don't block."""
    stack.enter_context(
        patch.object(
            QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: "")
        )
    )
    stack.enter_context(
        patch.object(
            QFileDialog,
            "getOpenFileName",
            staticmethod(lambda *a, **k: ("", "")),
        )
    )
    stack.enter_context(
        patch.object(
            QFileDialog,
            "getSaveFileName",
            staticmethod(lambda *a, **k: ("", "")),
        )
    )
    stack.enter_context(patch.object(QMessageBox, "exec", lambda self: 0))
    stack.enter_context(
        patch.object(QMessageBox, "warning", staticmethod(lambda *a, **k: 0))
    )
    stack.enter_context(
        patch.object(QMessageBox, "information", staticmethod(lambda *a, **k: 0))
    )
    stack.enter_context(
        patch.object(QMessageBox, "critical", staticmethod(lambda *a, **k: 0))
    )
    stack.enter_context(
        patch.object(QMessageBox, "question", staticmethod(lambda *a, **k: 0))
    )
    # QDialog.exec on any dialog → just return 0 (cancel/closed)
    stack.enter_context(patch.object(QDialog, "exec", lambda self: 0))


def _walk_pushbuttons(widget: QWidget) -> list[QPushButton]:
    """Return every QPushButton descendent (including hidden ones)."""
    return widget.findChildren(QPushButton)


def _is_safe_to_click(btn: QPushButton) -> bool:
    """Skip buttons that are disabled or never get rendered enabled."""
    if not btn.isEnabled():
        return False
    return True


def _try_click(btn: QPushButton, label: str, errors: list[tuple[str, str]]) -> None:
    """Click a button and capture any exception."""
    try:
        btn.click()
        _get_app().processEvents()
    except Exception as exc:  # noqa: BLE001 — we want all
        errors.append((label, f"{type(exc).__name__}: {exc}"))


# ---------------------------------------------------------------------------
# Panel-level tests: each panel constructed standalone, every button clicked
# ---------------------------------------------------------------------------


def test_connection_panel_buttons_no_crash() -> None:
    from eit_app.ui.hardware.connection_panel import ConnectionPanel

    panel = ConnectionPanel()
    panel.show()
    _get_app().processEvents()

    errors: list[tuple[str, str]] = []
    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_modal_dialogs(stack)
        for btn in _walk_pushbuttons(panel):
            if _is_safe_to_click(btn):
                _try_click(
                    btn, f"connection.{btn.objectName() or btn.text()!r}", errors
                )

    panel.close()
    assert not errors, f"Buttons crashed: {errors}"


def test_control_panel_buttons_no_crash() -> None:
    from eit_app.ui.hardware.control_panel import ControlPanel

    panel = ControlPanel()
    panel.show()
    _get_app().processEvents()

    errors: list[tuple[str, str]] = []
    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_modal_dialogs(stack)
        for btn in _walk_pushbuttons(panel):
            if _is_safe_to_click(btn):
                _try_click(btn, f"control.{btn.text()!r}", errors)

    panel.close()
    assert not errors, f"Buttons crashed: {errors}"


def test_acquisition_panel_buttons_no_crash() -> None:
    from eit_app.ui.hardware.acquisition_panel import AcquisitionPanel

    panel = AcquisitionPanel()
    panel.show()
    _get_app().processEvents()

    errors: list[tuple[str, str]] = []
    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_modal_dialogs(stack)
        for btn in _walk_pushbuttons(panel):
            if _is_safe_to_click(btn):
                _try_click(btn, f"acquisition.{btn.text()!r}", errors)

    panel.close()
    assert not errors, f"Buttons crashed: {errors}"


def test_frame_browser_buttons_no_crash() -> None:
    from eit_app.ui.hardware.frame_browser_widget import FrameBrowserWidget

    panel = FrameBrowserWidget()
    panel.show()
    _get_app().processEvents()

    errors: list[tuple[str, str]] = []
    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_modal_dialogs(stack)
        for btn in _walk_pushbuttons(panel):
            if _is_safe_to_click(btn):
                _try_click(btn, f"framebrowser.{btn.text()!r}", errors)

    panel.close()
    assert not errors, f"Buttons crashed: {errors}"


def test_equipotential_widget_buttons_no_crash() -> None:
    from eit_app.ui.hardware.equipotential_plot_widget import EquipotentialPlotWidget

    panel = EquipotentialPlotWidget()
    panel.show()
    _get_app().processEvents()

    errors: list[tuple[str, str]] = []
    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_modal_dialogs(stack)
        for btn in _walk_pushbuttons(panel):
            if _is_safe_to_click(btn):
                _try_click(btn, f"equipotential.{btn.text()!r}", errors)

    panel.close()
    assert not errors, f"Buttons crashed: {errors}"


def test_forward_problem_panel_buttons_no_crash() -> None:
    from eit_app.ui.simulation.forward_problem_panel import ForwardProblemPanel

    panel = ForwardProblemPanel()
    panel.show()
    _get_app().processEvents()

    errors: list[tuple[str, str]] = []
    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_modal_dialogs(stack)
        for btn in _walk_pushbuttons(panel):
            if _is_safe_to_click(btn):
                _try_click(btn, f"forward.{btn.text()!r}", errors)

    panel.close()
    assert not errors, f"Buttons crashed: {errors}"


def test_inverse_problem_panel_buttons_no_crash() -> None:
    from eit_app.ui.simulation.inverse_problem_panel import (
        SIMULATION_INVERSE_METHODS,
        InverseProblemPanel,
    )

    for method in SIMULATION_INVERSE_METHODS:
        panel = InverseProblemPanel()
        panel.show()
        _get_app().processEvents()

        idx = panel._method_combo.findText(method)
        if idx >= 0:
            panel._method_combo.setCurrentIndex(idx)
        _get_app().processEvents()

        errors: list[tuple[str, str]] = []
        from contextlib import ExitStack

        with ExitStack() as stack:
            _patch_modal_dialogs(stack)
            for btn in _walk_pushbuttons(panel):
                if _is_safe_to_click(btn):
                    _try_click(btn, f"inverse.{method}.{btn.text()!r}", errors)

        panel.close()
        assert not errors, f"Buttons crashed for method {method}: {errors}"


def test_inhomogeneity_editor_shape_buttons_per_dim() -> None:
    from eit_app.ui.simulation.inhomogeneity_editor import InhomogeneityEditor

    for dim in (2, 3):
        editor = InhomogeneityEditor()
        editor.set_domain_context(
            mesh_dimension=dim, radius=1.0, height=1.0, z_center=0.0
        )
        editor.show()
        _get_app().processEvents()

        errors: list[tuple[str, str]] = []
        from contextlib import ExitStack

        with ExitStack() as stack:
            _patch_modal_dialogs(stack)
            for btn in _walk_pushbuttons(editor):
                if _is_safe_to_click(btn):
                    _try_click(btn, f"inhom.dim{dim}.{btn.text()!r}", errors)

        # After clicking shape buttons, table should have rows.
        assert editor.get_inhomogeneities(), f"No specs added for dim={dim}"
        # Then click remove on the first row.
        view = editor._table
        view.selectRow(0)
        editor._remove_btn.click()
        _get_app().processEvents()
        editor.close()
        assert not errors, f"Buttons crashed (dim={dim}): {errors}"


def test_mesh_setup_panel_dimension_toggle_no_crash() -> None:
    from eit_app.ui.simulation.mesh_setup_panel import MeshSetupPanel

    panel = MeshSetupPanel()
    panel.show()
    _get_app().processEvents()
    # Toggle 2D ↔ 3D ↔ 2D — exercises every gating path
    panel._dim_combo.setCurrentIndex(1)
    _get_app().processEvents()
    panel._dim_combo.setCurrentIndex(0)
    _get_app().processEvents()
    # And every measurement-protocol option
    panel._dim_combo.setCurrentIndex(1)
    _get_app().processEvents()
    for i in range(panel._measurement_protocol_combo.count()):
        panel._measurement_protocol_combo.setCurrentIndex(i)
        _get_app().processEvents()
    cfg = panel.get_config()
    assert cfg["mesh_dimension"] == 3
    panel.close()


def test_dataset_generator_panel_buttons_no_crash() -> None:
    from eit_app.ui.simulation.dataset_generator_panel import (
        DatasetGeneratorPanel,
    )

    panel_obj = DatasetGeneratorPanel()
    # DatasetGeneratorPanel is a QObject coordinator; the actual widgets
    # are .randomization_panel and .run_panel — both QGroupBoxes.
    holder = QWidget()
    from PySide6.QtWidgets import QVBoxLayout

    layout = QVBoxLayout(holder)
    layout.addWidget(panel_obj.randomization_panel)
    layout.addWidget(panel_obj.run_panel)
    holder.show()
    _get_app().processEvents()

    errors: list[tuple[str, str]] = []
    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_modal_dialogs(stack)
        for btn in _walk_pushbuttons(holder):
            if _is_safe_to_click(btn):
                # Skip the Generate button — it would actually try to
                # spin up the dataset generator controller without one.
                if btn is panel_obj.run_panel._gen_btn:
                    continue
                _try_click(btn, f"dataset.{btn.text()!r}", errors)

    holder.close()
    assert not errors, f"Buttons crashed: {errors}"


# ---------------------------------------------------------------------------
# Dialog-level tests
# ---------------------------------------------------------------------------


def test_about_dialog_buttons_no_crash() -> None:
    from eit_app.ui.dialogs.about_dialog import AboutDialog

    dialog = AboutDialog()
    dialog.show()
    _get_app().processEvents()

    assert dialog._credit_label.text() == ""
    assert not dialog._credit_label.isVisible()

    # Only the Close button.  Dialog should have a QDialogButtonBox.
    btns = dialog._buttons.buttons()
    assert btns, "About dialog has no buttons"
    for b in btns:
        b.click()
        _get_app().processEvents()
    # After clicking Close, dialog should be hidden.
    assert not dialog.isVisible()


def test_about_dialog_version_uses_designer_credit(monkeypatch) -> None:
    from eit_app.ui.dialogs import about_dialog as about_dialog_module
    from eit_app.ui.main_window import EITWorkstation

    captured: dict[str, str] = {}

    class _FakeLabel:
        def __init__(self) -> None:
            self._text = "版本 {version} · {build}"

        def text(self) -> str:
            return self._text

        def setText(self, value: str) -> None:
            self._text = value
            captured["version_line"] = value

    class _FakeAboutDialog:
        def __init__(self, _parent=None) -> None:
            self._version_label = _FakeLabel()

        def exec(self) -> None:
            captured["exec"] = "1"

    monkeypatch.setattr(about_dialog_module, "AboutDialog", _FakeAboutDialog)

    EITWorkstation._open_about_dialog(object())

    assert captured["exec"] == "1"
    assert captured["version_line"].endswith("designed by Bing-Zhou Chen!")
    assert "design-system gui-polish-v4" not in captured["version_line"]


def test_reconstruction_dialog_buttons_no_crash() -> None:
    from eit_app.ui.dialogs.reconstruction_dialog import ReconstructionDialog

    target = {"frame_index": 0, "csv_path": "/tmp/x.csv"}
    ref = {"frame_index": 1, "csv_path": "/tmp/y.csv"}
    dlg = ReconstructionDialog(reference_entry=ref, target_entry=target)
    dlg.show()
    _get_app().processEvents()

    errors: list[tuple[str, str]] = []
    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_modal_dialogs(stack)
        # Iterate algorithms — each rebuilds reference visibility
        for i in range(dlg._algo_combo.count()):
            dlg._algo_combo.setCurrentIndex(i)
            _get_app().processEvents()
        # Click browse button (mocked)
        _try_click(dlg._dir_browse_btn, "recon_dlg.browse", errors)
        # Click Cancel — dialog hides
        _try_click(dlg._cancel_btn, "recon_dlg.cancel", errors)

    assert not errors, f"Buttons crashed: {errors}"


def test_batch_reconstruction_dialog_buttons_no_crash() -> None:
    from eit_app.ui.dialogs.batch_reconstruction_dialog import (
        BatchReconstructionDialog,
    )

    dlg = BatchReconstructionDialog()
    dlg.show()
    _get_app().processEvents()

    errors: list[tuple[str, str]] = []
    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_modal_dialogs(stack)
        for i in range(dlg._algo_combo.count()):
            dlg._algo_combo.setCurrentIndex(i)
            _get_app().processEvents()
        # Click each browse button + cancel/close (Run would emit signal — skip)
        for btn in (
            dlg._input_browse_btn,
            dlg._output_browse_btn,
            dlg._ref_browse_btn,
            dlg._cancel_btn,
            dlg._close_btn,
        ):
            _try_click(btn, f"batch_dlg.{btn.text()!r}", errors)

    assert not errors, f"Buttons crashed: {errors}"


def test_difference_dialog_buttons_no_crash() -> None:
    from eit_app.ui.dialogs.difference_dialog import DifferenceDialog

    # Dialog needs frame entries — feed empty list, just verify no crash on open
    dlg = DifferenceDialog(frame_entries=[])
    dlg.show()
    _get_app().processEvents()

    # Walk all buttons
    errors: list[tuple[str, str]] = []
    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_modal_dialogs(stack)
        for btn in _walk_pushbuttons(dlg):
            if _is_safe_to_click(btn):
                _try_click(btn, f"diff_dlg.{btn.text()!r}", errors)

    dlg.close()
    assert not errors, f"Buttons crashed: {errors}"


def test_interop_hub_dialog_buttons_no_crash() -> None:
    from eit_app.ui.dialogs.interop_hub_dialog import InteropHubDialog

    dlg = InteropHubDialog()
    dlg.show()
    _get_app().processEvents()

    errors: list[tuple[str, str]] = []
    from contextlib import ExitStack

    with ExitStack() as stack:
        _patch_modal_dialogs(stack)
        # Switch through all three tabs
        for i in range(dlg._tabs.count()):
            dlg._tabs.setCurrentIndex(i)
            _get_app().processEvents()
            for btn in _walk_pushbuttons(dlg._tabs.currentWidget()):
                if _is_safe_to_click(btn):
                    _try_click(btn, f"interop.tab{i}.{btn.text()!r}", errors)

    dlg.close()
    assert not errors, f"Buttons crashed: {errors}"


# ---------------------------------------------------------------------------
# Full main-window walkthrough: every tab, every menu, every button
# ---------------------------------------------------------------------------


def test_main_window_every_button_click_no_crash() -> None:
    """Top-level smoke: walk every QPushButton in the live window.

    Buttons that depend on a connected device (connect, power, start/stop
    acquisition) are skipped — those are exercised in panel-level tests
    with the simulator wired up.  Walking them here without state would
    block on real serial-port discovery when run after other tests that
    leak threads.
    """
    from eit_app.ui.main_window import EITWorkstation

    win = EITWorkstation()
    win.show()
    _get_app().processEvents()

    errors: list[tuple[str, str]] = []
    from contextlib import ExitStack

    skip_objects: set[int] = set()
    # Connection panel — connect/disconnect would touch real hardware
    for attr in ("_connect_btn", "_disconnect_btn", "_refresh_btn"):
        btn = getattr(win._conn_panel, attr, None)
        if btn is not None:
            skip_objects.add(id(btn))
    # Control panel — power/spt/imp blocks on device commands
    for attr in (
        "_power_on_btn",
        "_power_off_btn",
        "_spt_btn",
        "_imp_btn",
        "_freq_apply",
        "_stim_apply",
        "_vamp_apply",
    ):
        btn = getattr(win._control_panel, attr, None)
        if btn is not None:
            skip_objects.add(id(btn))
    # Acquisition panel — start/stop/single-frame need a connected device
    for attr in ("_start_btn", "_stop_btn", "_single_frame_btn"):
        btn = getattr(win._acq_panel, attr, None)
        if btn is not None:
            skip_objects.add(id(btn))

    with ExitStack() as stack:
        _patch_modal_dialogs(stack)

        # Switch through every tab — exercises lazy initialisation.
        tabs = win._tab_widget
        for i in range(tabs.count()):
            tabs.setCurrentIndex(i)
            _get_app().processEvents()

        # Walk every safe button.
        for btn in _walk_pushbuttons(win):
            if id(btn) in skip_objects:
                continue
            if not _is_safe_to_click(btn):
                continue
            label = (btn.text() or btn.objectName() or "<anon>").strip()
            _try_click(btn, f"main.{label!r}", errors)
            _get_app().processEvents()

    win.close()
    assert not errors, f"Buttons crashed: {errors}"


def test_main_window_menu_actions_no_crash() -> None:
    """Trigger every menu action and ensure none raises."""
    from eit_app.ui.main_window import EITWorkstation
    import eit_app.ui.main_window as mw_module

    win = EITWorkstation()
    win.show()
    _get_app().processEvents()

    errors: list[tuple[str, str]] = []
    from contextlib import ExitStack

    # Skip menu actions that spawn external processes / file managers —
    # those are validated separately and would leak Popen handles in the
    # offscreen test runner.
    skip_actions = {
        getattr(win, "_action_exit", None),
        getattr(win, "_action_open_recordings", None),
        getattr(win, "_action_open_output", None),
    }

    with ExitStack() as stack:
        _patch_modal_dialogs(stack)
        # Stub the file-manager helpers so they don't actually spawn
        # xdg-open / explorer.exe (would leak Popen until process exit).
        stack.enter_context(
            patch.object(
                mw_module, "_open_folder_in_file_manager", lambda *a, **k: True
            )
        )
        stack.enter_context(
            patch.object(mw_module, "_open_with_explorer_exe", lambda *a, **k: True)
        )

        def _trigger(action) -> None:
            if action is None or action.isSeparator():
                return
            if action in skip_actions:
                return
            label = action.text()
            try:
                action.trigger()
                _get_app().processEvents()
            except Exception as exc:  # noqa: BLE001
                errors.append((label, f"{type(exc).__name__}: {exc}"))
            # Close any modeless dialog the trigger may have opened so
            # the subsequent actions don't stack windows on top of it.
            for w in list(_get_app().topLevelWidgets()):
                if w is not win and isinstance(w, QDialog):
                    w.close()
                    _get_app().processEvents()

        # Collect actions reachable through the menubar.  Snapshot the
        # action lists up-front so submenu rebuilds during a trigger
        # don't invalidate the iterator and leave us walking a deleted
        # QMenu (the libshiboken "C++ object already deleted" path).
        menubar = win.menuBar()
        for menu_action in list(menubar.actions()):
            menu = menu_action.menu()
            if menu is None:
                continue
            for action in list(menu.actions()):
                sub_menu = action.menu()
                if sub_menu is not None:
                    for sub in list(sub_menu.actions()):
                        _trigger(sub)
                    continue
                _trigger(action)

    win.close()
    assert not errors, f"Menu actions crashed: {errors}"
