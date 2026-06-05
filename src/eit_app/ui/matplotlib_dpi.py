"""HiDPI helpers for Matplotlib canvases embedded in Qt widgets."""

from __future__ import annotations

from typing import Any


def qt_effective_device_pixel_ratio(widget: Any) -> float:
    """Return the best available Qt pixel ratio for raster plot canvases."""
    ratios: list[float] = [1.0]
    try:
        ratios.append(float(widget.devicePixelRatioF()))
    except Exception:
        pass
    try:
        ratios.append(float(widget.logicalDpiX()) / 96.0)
    except Exception:
        pass
    try:
        ratios.append(float(widget.logicalDpiY()) / 96.0)
    except Exception:
        pass
    return max(1.0, *[ratio for ratio in ratios if ratio > 0.0])


def sync_matplotlib_canvas_dpi(canvas: Any, widget: Any) -> bool:
    """Synchronize a FigureCanvasQTAgg with Qt's current HiDPI ratio."""
    ratio = qt_effective_device_pixel_ratio(widget)
    setter = getattr(canvas, "_set_device_pixel_ratio", None)
    if callable(setter):
        try:
            return bool(setter(ratio))
        except Exception:
            pass
    figure = getattr(canvas, "figure", None)
    if figure is None:
        return False
    try:
        base_dpi = float(getattr(figure, "_original_dpi", figure.dpi))
        target_dpi = base_dpi * ratio
        if abs(float(figure.dpi) - target_dpi) <= 0.5:
            return False
        figure.set_dpi(target_dpi)
        return True
    except Exception:
        return False
