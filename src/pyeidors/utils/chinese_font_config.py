#!/usr/bin/env python3
"""
中文字体配置模块
为matplotlib设置中文字体支持
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

logger = logging.getLogger(__name__)

def configure_chinese_font():
    """
    配置matplotlib中文字体支持
    
    这个函数会尝试设置中文字体，解决中文显示问题
    """
    try:
        # 设置中文字体支持 - 尝试常见的中文字体
        # 优先确保文泉驿字体已注册（Docker镜像内提供）
        wqy_path = Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
        if wqy_path.exists():
            try:
                font_manager.fontManager.addfont(str(wqy_path))
            except Exception as add_err:  # pragma: no cover - 注册失败时记录日志
                logger.warning("无法注册文泉驿字体: %s", add_err)

        chinese_fonts = [
            'WenQuanYi Zen Hei',    # 文泉驿正黑
            'Noto Sans CJK SC',     # Google Noto字体
            'SimHei',               # 黑体
            'Microsoft YaHei',      # 微软雅黑
            'DejaVu Sans'           # 备选英文字体
        ]

        plt.rcParams['font.sans-serif'] = chinese_fonts
        plt.rcParams['font.family'] = ['WenQuanYi Zen Hei']
        
        # 解决负号'-'显示为方块的问题
        plt.rcParams['axes.unicode_minus'] = False
        
        logger.info("中文字体配置成功")
        
    except Exception as e:
        logger.warning(f"中文字体配置失败: {e}")
        # 使用默认配置
        plt.rcParams['axes.unicode_minus'] = False

def reset_font_config():
    """重置字体配置为默认值"""
    plt.rcdefaults()

# 便捷导入
__all__ = ['configure_chinese_font', 'reset_font_config']
