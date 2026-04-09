"""Runtime font registration helpers for Qt and matplotlib."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication

_SERIF_CANDIDATES = [
    "Times New Roman",
    "Liberation Serif",
    "Nimbus Roman",
    "DejaVu Serif",
]
_WINDOWS_TNR_FILES = [
    Path("/mnt/c/Windows/Fonts/times.ttf"),
    Path("/mnt/c/Windows/Fonts/timesbd.ttf"),
    Path("/mnt/c/Windows/Fonts/timesi.ttf"),
    Path("/mnt/c/Windows/Fonts/timesbi.ttf"),
]
_ACTIVE_SERIF_FAMILY = "Times New Roman"


def configure_runtime_fonts(_app: QApplication) -> str:
    """Register serif fonts for Qt and matplotlib and return the active serif family."""
    global _ACTIVE_SERIF_FAMILY

    for path in _existing_windows_serif_paths():
        QFontDatabase.addApplicationFont(str(path))

    _ACTIVE_SERIF_FAMILY = _pick_available_serif_family()
    _configure_matplotlib_fonts(_ACTIVE_SERIF_FAMILY)
    return _ACTIVE_SERIF_FAMILY


def serif_font_family() -> str:
    """Return the best available serif font for plots and engineering labels."""
    return _ACTIVE_SERIF_FAMILY


def _existing_windows_serif_paths() -> list[Path]:
    return [path for path in _WINDOWS_TNR_FILES if path.exists()]


def _pick_available_serif_family() -> str:
    qt_families = set(QFontDatabase.families())
    for family in _SERIF_CANDIDATES:
        if family in qt_families:
            return family
    return "DejaVu Serif"


def _configure_matplotlib_fonts(family: str) -> None:
    try:
        import matplotlib
        from matplotlib import font_manager
    except Exception:
        return

    for path in _existing_windows_serif_paths():
        try:
            font_manager.fontManager.addfont(str(path))
        except Exception:
            continue

    matplotlib.rcParams["font.family"] = [family]
    matplotlib.rcParams["font.serif"] = _SERIF_CANDIDATES
    matplotlib.rcParams["axes.unicode_minus"] = False
