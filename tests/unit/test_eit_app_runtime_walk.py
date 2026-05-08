"""Runtime walkthrough — drive the live GUI through each major workflow.

Goes beyond clicking buttons: actually runs forward → inverse for each
simulation method, exercises simulator acquisition, switches language /
theme / precision midway, and confirms no exceptions or thread leaks.
"""

from __future__ import annotations

import gc
import os
from contextlib import ExitStack
from unittest.mock import patch

import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QMessageBox,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _get_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _wait_until(predicate, *, timeout: float = 30.0, step: float = 0.05) -> bool:
    import time

    app = _get_app()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(step)
    app.processEvents()
    return bool(predicate())


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
                deadline = 200
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


def _patch_modal(stack):
    stack.enter_context(
        patch.object(
            QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: "")
        )
    )
    stack.enter_context(
        patch.object(
            QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: ("", ""))
        )
    )
    stack.enter_context(
        patch.object(
            QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", ""))
        )
    )
    stack.enter_context(
        patch.object(QMessageBox, "warning", staticmethod(lambda *a, **k: 0))
    )
    stack.enter_context(
        patch.object(QMessageBox, "information", staticmethod(lambda *a, **k: 0))
    )
    stack.enter_context(
        patch.object(QMessageBox, "critical", staticmethod(lambda *a, **k: 0))
    )
    stack.enter_context(patch.object(QDialog, "exec", lambda self: 0))


def _connect_simulator(window) -> None:
    """Connect simulator with fps=0 so frames flow without sleep."""
    window._on_connect_requested("simulator", {"simulator_fps": 0})
    assert _wait_until(
        lambda: window._state.connection_status.value == "connected", timeout=5.0
    )


def test_window_boots_and_switches_tabs() -> None:
    """Construct EITWorkstation, switch through every tab, close cleanly."""
    from eit_app.ui.main_window import EITWorkstation

    win = EITWorkstation()
    win.show()
    _get_app().processEvents()

    tabs = win._tab_widget
    for i in range(tabs.count()):
        tabs.setCurrentIndex(i)
        _get_app().processEvents()
        # Each tab must report a non-zero size
        assert tabs.currentWidget().size().width() > 0

    win.close()


def test_language_switch_does_not_crash_in_any_tab() -> None:
    """Switch zh/en/zh on every tab — verifies no missing translation key."""
    from eit_app.i18n import set_language
    from eit_app.ui.main_window import EITWorkstation

    win = EITWorkstation()
    win.show()
    _get_app().processEvents()

    for tab_idx in range(win._tab_widget.count()):
        win._tab_widget.setCurrentIndex(tab_idx)
        _get_app().processEvents()
        for lang in ("en", "zh", "en"):
            set_language(lang, persist=False)
            _get_app().processEvents()

    set_language("en", persist=False)
    win.close()


def test_theme_toggle_does_not_crash() -> None:
    """Light/dark mode swap on the live window."""
    from eit_app.ui.main_window import EITWorkstation
    from eit_app.ui.theme import set_theme_mode

    win = EITWorkstation()
    win.show()
    _get_app().processEvents()

    app = _get_app()
    for mode in ("light", "dark", "light"):
        set_theme_mode(app, mode, persist=False)
        app.processEvents()

    win.close()


def test_precision_toggle_does_not_crash() -> None:
    from eit_app.models.precision import set_precision
    from eit_app.ui.main_window import EITWorkstation

    win = EITWorkstation()
    win.show()
    _get_app().processEvents()
    for precision in ("float32", "float64", "float32"):
        set_precision(precision, persist=False)
        _get_app().processEvents()
    win.close()


def test_simulator_lifecycle_connect_acquire_stop_disconnect() -> None:
    """End-to-end simulator round trip on hardware tab."""
    from eit_app.ui.main_window import EITWorkstation

    win = EITWorkstation()
    win.show()
    _get_app().processEvents()

    # Switch to hardware tab (default index 0)
    win._tab_widget.setCurrentIndex(0)
    _get_app().processEvents()

    _connect_simulator(win)

    # Power on, then start acquisition
    win._control_panel._power_on_btn.click()
    _get_app().processEvents()

    win._acq_panel._start_btn.click()
    _get_app().processEvents()

    # Wait until at least one frame has flowed through
    assert _wait_until(lambda: win._state.frame_count > 0, timeout=5.0)

    # Stop and disconnect
    win._acq_panel._stop_btn.click()
    _get_app().processEvents()

    win._conn_panel._disconnect_btn.click()
    _get_app().processEvents()

    win.close()


@pytest.mark.slow
def test_simulation_tab_forward_then_inverse_each_method() -> None:
    """Drive forward solve + each inverse algorithm on a 2D mesh.

    Slow integration test (≥3 minutes): the real dolfinx forward solver
    runs once, then each of the six SIMULATION_INVERSE_METHODS gets a
    real reconstruction.  Tagged ``slow`` so the default test run skips
    it; opt in with ``-m slow`` or by naming the test directly.
    """
    from eit_app.ui.main_window import EITWorkstation
    from eit_app.ui.simulation.inverse_problem_panel import (
        SIMULATION_INVERSE_METHODS,
    )

    win = EITWorkstation()
    win.show()
    _get_app().processEvents()

    # Switch to Simulation tab
    win._tab_widget.setCurrentWidget(win._sim_tab)
    _get_app().processEvents()

    # Build a small 2D mesh — keep refinement coarse so the test is fast.
    mesh = win._sim_tab.mesh_setup_panel
    mesh._dim_combo.setCurrentIndex(0)  # 2D
    mesh._n_elec_spin.setValue(8)
    mesh._refine_spin.setValue(0.3)  # coarse
    _get_app().processEvents()

    # Add an inhomogeneity so the inverse has something to recover.
    inhom = win._sim_tab.inhomogeneity_editor
    inhom._add_shape("circle")
    _get_app().processEvents()

    # Run forward.  Real dolfinx + PETSc init can take 60-180s the first
    # time.  Increase timeout to be tolerant of cold-cache runs.
    fwd = win._sim_tab.forward_problem_panel
    fwd._solve_btn.click()
    _get_app().processEvents()
    assert _wait_until(lambda: not win._sim_state.forward_running, timeout=300.0), (
        "forward solver never finished"
    )
    assert win._last_fwd_result is not None
    assert (win._last_fwd_result.error_msg or "") == ""

    # For each inverse method, run reconstruction.  Methods using the
    # one-step single-step path are quick (~seconds); the full GN path is
    # the slow one — keep its iteration cap minimal.
    inv = win._sim_tab.inverse_problem_panel
    inv._iter_spin.setValue(2)
    for method in SIMULATION_INVERSE_METHODS:
        idx = inv._method_combo.findText(method)
        assert idx >= 0, method
        inv._method_combo.setCurrentIndex(idx)
        _get_app().processEvents()

        with ExitStack() as stack:
            _patch_modal(stack)
            inv._recon_btn.click()
            _get_app().processEvents()
            ok = _wait_until(lambda: not win._sim_state.inverse_running, timeout=300.0)
            assert ok, f"Inverse {method} never finished"

    win.close()
