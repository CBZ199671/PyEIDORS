"""Plot language and font configuration utilities.

This module centralizes two concerns:
1. Resolve plotting language from explicit argument/env/default.
2. Configure matplotlib fonts without forcing unavailable families.
"""

from __future__ import annotations

from dataclasses import dataclass
import locale
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

logger = logging.getLogger(__name__)

PYEIDORS_PLOT_LANG = "PYEIDORS_PLOT_LANG"
_VALID_LANGUAGES = {"en", "zh", "auto"}
_WARNED_KEYS: set[str] = set()

_EN_FONT_CANDIDATES = [
    "DejaVu Sans",
    "Arial",
    "Helvetica",
    "Liberation Sans",
]

_ZH_FONT_CANDIDATES = [
    "PingFang SC",
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "SimHei",
    "WenQuanYi Zen Hei",
]

_PLOT_TEXTS: dict[str, dict[str, str]] = {
    "en": {
        "mesh_title": "Mesh Structure",
        "axis_x": "X",
        "axis_y": "Y",
        "conductivity_title": "Conductivity Distribution",
        "conductivity_label": "Conductivity (S/m)",
        "measurement_title": "Measurement Data",
        "measurement_sequence": "Measurement Sequence",
        "measurement_index": "Measurement Index",
        "voltage": "Voltage (V)",
        "measurement_distribution": "Measurement Distribution",
        "probability_density": "Probability Density",
        "mean": "Mean: {value:.4f}",
        "std": "±Std: {value:.4f}",
        "recon_comparison": "Reconstruction Comparison",
        "true_distribution": "True Distribution",
        "reconstructed_distribution": "Reconstructed Distribution",
        "absolute_error": "Absolute Error",
        "relative_error": "Relative Error: {value:.4f}",
        "convergence": "Convergence Curve",
        "iteration": "Iteration",
        "error_log": "Error (log scale)",
    },
    "zh": {
        "mesh_title": "网格结构",
        "axis_x": "X",
        "axis_y": "Y",
        "conductivity_title": "电导率分布",
        "conductivity_label": "电导率 (S/m)",
        "measurement_title": "测量数据",
        "measurement_sequence": "测量序列",
        "measurement_index": "测量序号",
        "voltage": "电压 (V)",
        "measurement_distribution": "测量分布",
        "probability_density": "概率密度",
        "mean": "均值: {value:.4f}",
        "std": "±标准差: {value:.4f}",
        "recon_comparison": "重建结果对比",
        "true_distribution": "真实分布",
        "reconstructed_distribution": "重建分布",
        "absolute_error": "绝对误差",
        "relative_error": "相对误差: {value:.4f}",
        "convergence": "收敛曲线",
        "iteration": "迭代次数",
        "error_log": "误差（对数坐标）",
    },
}


@dataclass(frozen=True)
class PlotFontConfigResult:
    """Result of plotting language/font configuration."""

    requested_language: str
    effective_language: str
    selected_fonts: tuple[str, ...]
    fallback_used: bool


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED_KEYS:
        return
    _WARNED_KEYS.add(key)
    logger.warning(message)


def _auto_language() -> str:
    for env_key in ("LC_ALL", "LC_CTYPE", "LANG"):
        value = os.getenv(env_key)
        if value and value.lower().startswith("zh"):
            return "zh"
    try:
        loc = locale.getlocale()[0]
    except Exception:
        loc = None
    if loc and loc.lower().startswith("zh"):
        return "zh"
    return "en"


def resolve_plot_language(language: str | None = None) -> str:
    """Resolve plotting language from arg/env/default.

    Priority: explicit arg > PYEIDORS_PLOT_LANG > default('en').
    """
    raw = language if language is not None else os.getenv(PYEIDORS_PLOT_LANG, "en")
    raw_norm = str(raw).strip().lower() if raw is not None else "en"

    if raw_norm not in _VALID_LANGUAGES:
        _warn_once(
            f"plot-lang-{raw_norm}",
            f"Invalid plot language '{raw_norm}', fallback to 'en'.",
        )
        return "en"

    if raw_norm == "auto":
        return _auto_language()
    return raw_norm


def _register_optional_fonts() -> None:
    candidates = [
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/local/share/fonts/wqy/wqy-zenhei.ttc"),
        Path("/opt/homebrew/share/fonts/wqy-zenhei.ttc"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            font_manager.fontManager.addfont(str(path))
        except Exception as exc:  # pragma: no cover - optional registration
            _warn_once(
                f"font-register-{path}",
                f"Failed to register optional font {path}: {exc}",
            )


def _available_font_names() -> set[str]:
    _register_optional_fonts()
    return {entry.name for entry in font_manager.fontManager.ttflist}


def _pick_existing(candidates: list[str], available: set[str]) -> list[str]:
    return [font for font in candidates if font in available]


def _apply_matplotlib_fonts(selected_fonts: list[str]) -> None:
    plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams["font.sans-serif"] = selected_fonts
    plt.rcParams["axes.unicode_minus"] = False


def configure_plot_fonts(language: str | None = None) -> PlotFontConfigResult:
    """Configure matplotlib fonts for plotting language.

    Chinese mode prefers Chinese-capable fonts; if unavailable it falls back to
    safe English fonts and logs only once.
    """
    requested = resolve_plot_language(language)
    available = _available_font_names()

    english_fonts = _pick_existing(_EN_FONT_CANDIDATES, available)
    if not english_fonts:
        # Defensive fallback for minimal environments.
        english_fonts = ["DejaVu Sans"]

    fallback_used = False
    if requested == "zh":
        zh_fonts = _pick_existing(_ZH_FONT_CANDIDATES, available)
        if zh_fonts:
            selected = zh_fonts + [
                font for font in english_fonts if font not in zh_fonts
            ]
            effective = "zh"
        else:
            selected = english_fonts
            effective = "en"
            fallback_used = True
            _warn_once(
                "plot-zh-fallback",
                "Chinese plotting requested but no Chinese font found; fallback to English fonts.",
            )
    else:
        selected = english_fonts
        effective = "en"

    _apply_matplotlib_fonts(selected)
    return PlotFontConfigResult(
        requested_language=requested,
        effective_language=effective,
        selected_fonts=tuple(selected),
        fallback_used=fallback_used,
    )


def get_plot_texts(language: str) -> dict[str, str]:
    """Get plotting text dictionary for resolved language."""
    return _PLOT_TEXTS.get(language, _PLOT_TEXTS["en"])


__all__ = [
    "PYEIDORS_PLOT_LANG",
    "PlotFontConfigResult",
    "configure_plot_fonts",
    "get_plot_texts",
    "resolve_plot_language",
]
