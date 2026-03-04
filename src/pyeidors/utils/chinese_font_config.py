#!/usr/bin/env python3
"""Chinese font configuration helpers.

New code should prefer ``configure_plot_fonts`` from
``pyeidors.utils.plot_font_i18n``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from .plot_font_i18n import PlotFontConfigResult, configure_plot_fonts


def configure_chinese_font() -> PlotFontConfigResult:
    """Wrapper that requests Chinese plotting mode.

    Returns:
        PlotFontConfigResult with requested/effective language and selected fonts.
    """
    return configure_plot_fonts(language="zh")


def reset_font_config() -> None:
    """Reset matplotlib font configuration to defaults."""
    plt.rcdefaults()


__all__ = ["configure_chinese_font", "reset_font_config", "configure_plot_fonts", "PlotFontConfigResult"]
