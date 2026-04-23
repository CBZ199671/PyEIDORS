"""Tests for plotting language and font selection logic."""

from __future__ import annotations

from pyeidors.utils import chinese_font_config
from pyeidors.utils.plot_font_i18n import (
    PlotFontConfigResult,
    configure_plot_fonts,
    resolve_plot_language,
)


def test_resolve_plot_language_priority(monkeypatch):
    monkeypatch.delenv("PYEIDORS_PLOT_LANG", raising=False)
    assert resolve_plot_language(None) == "en"

    monkeypatch.setenv("PYEIDORS_PLOT_LANG", "zh")
    assert resolve_plot_language(None) == "zh"
    assert resolve_plot_language("en") == "en"


def test_resolve_plot_language_auto(monkeypatch):
    monkeypatch.delenv("PYEIDORS_PLOT_LANG", raising=False)
    monkeypatch.setenv("LANG", "zh_CN.UTF-8")
    assert resolve_plot_language("auto") == "zh"


def test_configure_plot_fonts_fallback_to_english(monkeypatch):
    monkeypatch.setattr(
        "pyeidors.utils.plot_font_i18n._available_font_names",
        lambda: {"DejaVu Sans"},
    )
    result = configure_plot_fonts("zh")
    assert result.requested_language == "zh"
    assert result.effective_language == "en"
    assert result.fallback_used is True
    assert result.selected_fonts == ("DejaVu Sans",)


def test_configure_plot_fonts_keep_chinese_when_available(monkeypatch):
    monkeypatch.setattr(
        "pyeidors.utils.plot_font_i18n._available_font_names",
        lambda: {"PingFang SC", "DejaVu Sans"},
    )
    result = configure_plot_fonts("zh")
    assert result.requested_language == "zh"
    assert result.effective_language == "zh"
    assert result.fallback_used is False
    assert result.selected_fonts[0] == "PingFang SC"


def test_chinese_font_config_wrapper_behavior(monkeypatch):
    expected = PlotFontConfigResult(
        requested_language="zh",
        effective_language="en",
        selected_fonts=("DejaVu Sans",),
        fallback_used=True,
    )

    monkeypatch.setattr(
        chinese_font_config, "configure_plot_fonts", lambda language=None: expected
    )
    result = chinese_font_config.configure_chinese_font()
    assert result == expected
