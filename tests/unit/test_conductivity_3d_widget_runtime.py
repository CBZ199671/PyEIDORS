from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from eit_app.ui.conductivity_3d_widget import (  # noqa: E402
    embedded_vtk_enabled,
    embedded_vtk_status,
)
from eit_app.ui.simulation.simulation_results_widget import (  # noqa: E402
    _ConductivityViewSlot,
)


def _get_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _tetra_payload() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    cells = np.array([[0, 1, 2, 3]], dtype=np.int64)
    sigma = np.array([1.25], dtype=float)
    return sigma, coords, cells


def test_embedded_vtk_disabled_for_offscreen_qt(monkeypatch):
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.delenv("EIT_APP_DISABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    enabled, reason = embedded_vtk_status()

    assert enabled is False
    assert embedded_vtk_enabled() is False
    assert "offscreen" in reason


def test_embedded_vtk_can_be_forced(monkeypatch):
    monkeypatch.delenv("EIT_APP_DISABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("EIT_APP_ENABLE_EMBEDDED_VTK", "1")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    enabled, reason = embedded_vtk_status()

    assert enabled is True
    assert embedded_vtk_enabled() is True
    assert "forced" in reason


def test_3d_payload_uses_matplotlib_projection_when_vtk_disabled(monkeypatch):
    _get_app()
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    slot = _ConductivityViewSlot("Conductivity")
    calls: list[tuple[str, str | None]] = []

    def fake_mpl_update(_sigma, _coords, _cells, title=None):
        calls.append(("mpl", title))

    def unexpected_vtk_update(_sigma, _coords, _cells, title=None):
        raise AssertionError("QtInteractor should not be touched when VTK is disabled")

    monkeypatch.setattr(slot._mpl, "update_image", fake_mpl_update)
    monkeypatch.setattr(slot._three_d, "update_image", unexpected_vtk_update)

    sigma, coords, cells = _tetra_payload()
    slot.update_image(sigma, coords, cells, title="Truth")

    assert calls and calls[0][0] == "mpl"
    assert "Truth" in (calls[0][1] or "")
    assert "3D" in (calls[0][1] or "")
    assert slot._stack.currentWidget() is slot._mpl
    slot.close()


def test_3d_projection_fallback_renders_small_tetra(monkeypatch):
    _get_app()
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    slot = _ConductivityViewSlot("Conductivity")

    sigma, coords, cells = _tetra_payload()
    slot.update_image(sigma, coords, cells, title="Truth")

    assert slot._stack.currentWidget() is slot._mpl
    assert slot._mpl._last_image is not None
    slot.close()


def test_3d_payload_uses_vtk_widget_when_forced(monkeypatch):
    _get_app()
    monkeypatch.setenv("EIT_APP_ENABLE_EMBEDDED_VTK", "1")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    slot = _ConductivityViewSlot("Conductivity")
    calls: list[tuple[str, str | None]] = []

    def unexpected_mpl_update(_sigma, _coords, _cells, title=None):
        raise AssertionError("Matplotlib fallback should not run when VTK is forced")

    def fake_vtk_update(_sigma, _coords, _cells, title=None):
        calls.append(("vtk", title))

    monkeypatch.setattr(slot._mpl, "update_image", unexpected_mpl_update)
    monkeypatch.setattr(slot._three_d, "update_image", fake_vtk_update)

    sigma, coords, cells = _tetra_payload()
    slot.update_image(sigma, coords, cells, title="Truth")

    assert calls == [("vtk", "Truth")]
    assert slot._stack.currentWidget() is slot._three_d
    slot.close()
