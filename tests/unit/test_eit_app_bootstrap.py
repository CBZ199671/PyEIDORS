"""Unit tests for EIT Workstation bootstrap environment decisions."""

from __future__ import annotations

import os

import eit_app.app as app_module


_QT_ENV_NAMES = (
    "QT_QPA_PLATFORM",
    "QT_SCALE_FACTOR_ROUNDING_POLICY",
    "QT_X11_NO_MITSHM",
    "EIT_APP_USE_QT_WAYLAND",
    "EIT_APP_USE_QT_XCB",
    "EIT_APP_DISABLE_QT_WAYLAND",
    "WSL_DISTRO_NAME",
    "WSL_INTEROP",
    "WAYLAND_DISPLAY",
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

    app_module._configure_qt_platform_for_embedded_vtk()

    assert os.environ["QT_QPA_PLATFORM"].split(";", 1)[0] == "wayland"
    assert os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] == "PassThrough"
    assert "QT_X11_NO_MITSHM" not in os.environ


def test_v51_wslg_xcb_requires_explicit_legacy_opt_in(monkeypatch):
    _clear_qt_env(monkeypatch)
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-22.04")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("EIT_APP_USE_QT_XCB", "1")

    app_module._configure_qt_platform_for_embedded_vtk()

    assert os.environ["QT_QPA_PLATFORM"] == "xcb"
    assert os.environ["QT_X11_NO_MITSHM"] == "1"


def test_v51_explicit_qt_platform_is_preserved(monkeypatch):
    _clear_qt_env(monkeypatch)
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-22.04")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    app_module._configure_qt_platform_for_embedded_vtk()

    assert os.environ["QT_QPA_PLATFORM"] == "offscreen"
