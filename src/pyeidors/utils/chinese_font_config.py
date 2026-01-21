#!/usr/bin/env python3
"""
Chinese font configuration module.

Configure matplotlib for Chinese font support.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

logger = logging.getLogger(__name__)

def configure_chinese_font():
    """Configure matplotlib Chinese font support.

    This function attempts to set up Chinese fonts to resolve display issues.
    """
    try:
        # Set Chinese font support - try common Chinese fonts
        # Prioritize WenQuanYi font (provided in Docker image)
        wqy_path = Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
        if wqy_path.exists():
            try:
                font_manager.fontManager.addfont(str(wqy_path))
            except Exception as add_err:  # pragma: no cover - log if registration fails
                logger.warning("Failed to register WenQuanYi font: %s", add_err)

        chinese_fonts = [
            'WenQuanYi Zen Hei',    # WenQuanYi Zen Hei
            'Noto Sans CJK SC',     # Google Noto font
            'SimHei',               # SimHei
            'Microsoft YaHei',      # Microsoft YaHei
            'DejaVu Sans'           # Fallback English font
        ]

        plt.rcParams['font.sans-serif'] = chinese_fonts
        plt.rcParams['font.family'] = ['WenQuanYi Zen Hei']

        # Fix minus sign '-' displaying as a box
        plt.rcParams['axes.unicode_minus'] = False

        logger.info("Chinese font configuration successful")

    except Exception as e:
        logger.warning(f"Chinese font configuration failed: {e}")
        # Use default configuration
        plt.rcParams['axes.unicode_minus'] = False

def reset_font_config():
    """Reset font configuration to default values."""
    plt.rcdefaults()

# Convenience imports
__all__ = ['configure_chinese_font', 'reset_font_config']
