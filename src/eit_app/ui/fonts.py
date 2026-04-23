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

# CJK-capable families used as per-glyph fallback when the serif face lacks
# Chinese glyphs.  Order matters: matplotlib walks the list left-to-right.
_CJK_CANDIDATES = [
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "Noto Sans CJK SC",
    "Noto Serif CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "SimSun",
    "SimHei",
    "PingFang SC",
]
_WINDOWS_CJK_FILES = [
    Path("/mnt/c/Windows/Fonts/msyh.ttc"),  # Microsoft YaHei
    Path("/mnt/c/Windows/Fonts/msyhbd.ttc"),  # YaHei Bold
    Path("/mnt/c/Windows/Fonts/msyhl.ttc"),  # YaHei Light
    Path("/mnt/c/Windows/Fonts/simhei.ttf"),  # SimHei
    Path("/mnt/c/Windows/Fonts/simsun.ttc"),  # SimSun
    Path("/mnt/c/Windows/Fonts/NSimSun.ttf"),
]

_ACTIVE_SERIF_FAMILY = "Times New Roman"
_ACTIVE_PLOT_FAMILIES: list[str] = [_ACTIVE_SERIF_FAMILY]


def configure_runtime_fonts(_app: QApplication) -> str:
    """Register serif / CJK fonts for Qt and matplotlib.

    Returns the active serif family name used by engineering labels.
    Side effect: populates :func:`plot_font_families` with a fallback list
    so that matplotlib can render CJK glyphs that Times New Roman lacks.
    """
    global _ACTIVE_SERIF_FAMILY, _ACTIVE_PLOT_FAMILIES

    for path in _existing_windows_serif_paths():
        QFontDatabase.addApplicationFont(str(path))
    for path in _existing_windows_cjk_paths():
        QFontDatabase.addApplicationFont(str(path))

    # Register Windows fonts with matplotlib BEFORE we inspect its ttflist
    # in _matplotlib_known_cjk_families() — otherwise the CJK families
    # wouldn't be discovered yet.
    try:
        from matplotlib import font_manager as _font_manager

        for path in (*_existing_windows_serif_paths(), *_existing_windows_cjk_paths()):
            try:
                _font_manager.fontManager.addfont(str(path))
            except Exception:
                continue
    except Exception:
        pass

    _ACTIVE_SERIF_FAMILY = _pick_available_serif_family()
    _ACTIVE_PLOT_FAMILIES = _build_plot_family_list(_ACTIVE_SERIF_FAMILY)
    _configure_matplotlib_fonts(_ACTIVE_PLOT_FAMILIES)
    return _ACTIVE_SERIF_FAMILY


def serif_font_family() -> str:
    """Return the best available serif font for plots and engineering labels."""
    return _ACTIVE_SERIF_FAMILY


def plot_font_families() -> list[str]:
    """Return the ordered font-family list used for matplotlib rendering.

    The first entry is a classic serif (Times New Roman etc.); the rest
    are CJK-capable faces used as per-glyph fallback so Chinese titles
    render correctly without warnings.
    """
    return list(_ACTIVE_PLOT_FAMILIES)


def _existing_windows_serif_paths() -> list[Path]:
    return [path for path in _WINDOWS_TNR_FILES if path.exists()]


def _existing_windows_cjk_paths() -> list[Path]:
    return [path for path in _WINDOWS_CJK_FILES if path.exists()]


def _pick_available_serif_family() -> str:
    qt_families = set(QFontDatabase.families())
    for family in _SERIF_CANDIDATES:
        if family in qt_families:
            return family
    return "DejaVu Serif"


def _build_plot_family_list(serif_family: str) -> list[str]:
    """Serif first, then only the CJK families matplotlib actually knows.

    Matplotlib otherwise logs a noisy ``findfont: Font family 'X' not found``
    for every unknown name each time a glyph is rendered.  We query the
    font manager once and keep only families with a real backing file.
    """
    cjk_families = _matplotlib_known_cjk_families()
    families: list[str] = [serif_family, *cjk_families]
    # Always terminate with a broad Unicode-covering sans so rare fallback
    # paths still resolve on minimal systems that lack all CJK faces.
    families.append("DejaVu Sans")
    return families


def _matplotlib_known_cjk_families() -> list[str]:
    try:
        from matplotlib import font_manager

        known = {font.name for font in font_manager.fontManager.ttflist}
    except Exception:
        return []
    return [family for family in _CJK_CANDIDATES if family in known]


def _configure_matplotlib_fonts(family_list: list[str]) -> None:
    try:
        import matplotlib
    except Exception:
        return

    # Setting font.family as a list enables matplotlib's per-glyph fallback
    # (requires matplotlib >= 3.6): missing glyphs in the first family roll
    # over to subsequent ones instead of emitting warnings and rendering a
    # tofu box.
    matplotlib.rcParams["font.family"] = family_list
    matplotlib.rcParams["font.serif"] = _SERIF_CANDIDATES
    matplotlib.rcParams["axes.unicode_minus"] = False
