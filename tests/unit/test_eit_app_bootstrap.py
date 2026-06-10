"""Unit tests for EIT Workstation bootstrap environment decisions."""

from __future__ import annotations

import os

import pytest

import eit_app.app as app_module
from eit_app.ui.main_window import EITWorkstation


_QT_ENV_NAMES = (
    "QT_QPA_PLATFORM",
    "QT_AUTO_SCREEN_SCALE_FACTOR",
    "QT_ENABLE_HIGHDPI_SCALING",
    "QT_SCALE_FACTOR_ROUNDING_POLICY",
    "QT_X11_NO_MITSHM",
    "EIT_APP_USE_QT_WAYLAND",
    "EIT_APP_USE_QT_XCB",
    "EIT_APP_DISABLE_QT_WAYLAND",
    "WSL_DISTRO_NAME",
    "WSL_INTEROP",
    "WAYLAND_DISPLAY",
    "XDG_RUNTIME_DIR",
    "DISPLAY",
)


def _clear_qt_env(monkeypatch) -> None:
    for name in _QT_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_v51_wslg_defaults_to_wayland_for_crisp_hidpi(monkeypatch):
    _clear_qt_env(monkeypatch)
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-22.04")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(app_module, "_wayland_display_available", lambda: True)

    app_module._configure_qt_platform_for_embedded_vtk()

    assert os.environ["QT_QPA_PLATFORM"].split(";", 1)[0] == "wayland"
    assert os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] == "1"
    assert os.environ["QT_ENABLE_HIGHDPI_SCALING"] == "1"
    assert os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] == "PassThrough"
    assert "QT_X11_NO_MITSHM" not in os.environ


def test_v51_wslg_xcb_requires_explicit_legacy_opt_in(monkeypatch):
    _clear_qt_env(monkeypatch)
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-22.04")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("EIT_APP_USE_QT_XCB", "1")
    monkeypatch.setattr(app_module, "_x11_display_available", lambda: True)

    app_module._configure_qt_platform_for_embedded_vtk()

    assert os.environ["QT_QPA_PLATFORM"] == "xcb"
    assert os.environ["QT_X11_NO_MITSHM"] == "1"


def test_v51_wslg_dead_wayland_falls_back_to_xcb(monkeypatch):
    _clear_qt_env(monkeypatch)
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-22.04")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(app_module, "_wayland_display_available", lambda: False)
    monkeypatch.setattr(app_module, "_x11_display_available", lambda: True)

    app_module._configure_qt_platform_for_embedded_vtk()

    assert os.environ["QT_QPA_PLATFORM"] == "xcb"
    assert os.environ["QT_X11_NO_MITSHM"] == "1"


def test_v51_wslg_dead_display_exits_before_qt_abort(monkeypatch):
    _clear_qt_env(monkeypatch)
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-22.04")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(app_module, "_wayland_display_available", lambda: False)
    monkeypatch.setattr(app_module, "_x11_display_available", lambda: False)

    with pytest.raises(SystemExit) as exc_info:
        app_module._configure_qt_platform_for_embedded_vtk()

    assert "WSLg display is not reachable" in str(exc_info.value)
    assert "wsl.exe --shutdown" in str(exc_info.value)


def test_v51_explicit_qt_platform_is_preserved(monkeypatch):
    _clear_qt_env(monkeypatch)
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-22.04")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    app_module._configure_qt_platform_for_embedded_vtk()

    assert os.environ["QT_QPA_PLATFORM"] == "offscreen"
    assert os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] == "1"
    assert os.environ["QT_ENABLE_HIGHDPI_SCALING"] == "1"


def test_v98_about_to_quit_closes_main_window() -> None:
    class _FakeSignal:
        def __init__(self) -> None:
            self._slot = None

        def connect(self, slot) -> None:  # noqa: ANN001 - Qt signal shape
            self._slot = slot

        def emit(self) -> None:
            assert self._slot is not None
            self._slot()

    class _FakeApp:
        def __init__(self) -> None:
            self.aboutToQuit = _FakeSignal()

    class _FakeWindow:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> bool:
            self.close_calls += 1
            return True

    fake_app = _FakeApp()
    fake_window = _FakeWindow()

    app_module._connect_about_to_quit_cleanup(fake_app, fake_window)
    fake_app.aboutToQuit.emit()

    assert fake_window.close_calls == 1


def test_v99_bogus_wslg_screen_geometry_does_not_collapse_main_window():
    size = EITWorkstation._bounded_initial_size(
        preferred_w=1360,
        preferred_h=840,
        available_w=131072,
        available_h=1,
    )

    assert (size.width(), size.height()) == (1360, 840)
