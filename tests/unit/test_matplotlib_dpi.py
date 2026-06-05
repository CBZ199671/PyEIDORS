from __future__ import annotations

from types import SimpleNamespace

import pytest

from eit_app.ui.matplotlib_dpi import (
    qt_effective_device_pixel_ratio,
    sync_matplotlib_canvas_dpi,
)


def test_v624_qt_effective_device_pixel_ratio_uses_logical_dpi_fallback():
    widget = SimpleNamespace(
        devicePixelRatioF=lambda: 1.0,
        logicalDpiX=lambda: 144.0,
        logicalDpiY=lambda: 96.0,
    )

    assert qt_effective_device_pixel_ratio(widget) == pytest.approx(1.5)


def test_v624_sync_matplotlib_canvas_uses_canvas_device_ratio_hook():
    calls: list[float] = []
    canvas = SimpleNamespace(
        _set_device_pixel_ratio=lambda ratio: calls.append(ratio) or True,
    )
    widget = SimpleNamespace(
        devicePixelRatioF=lambda: 2.0,
        logicalDpiX=lambda: 96.0,
        logicalDpiY=lambda: 96.0,
    )

    assert sync_matplotlib_canvas_dpi(canvas, widget) is True
    assert calls == [pytest.approx(2.0)]
