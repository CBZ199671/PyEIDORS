#!/usr/bin/env python3
"""
中文字体配置模块
为matplotlib设置中文字体支持
"""

import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def configure_chinese_font():
    """
    配置matplotlib中文字体支持
    
    这个函数会尝试设置中文字体，解决中文显示问题
    """
    try:
        # 设置中文字体支持 - 尝试常见的中文字体
        chinese_fonts = [
            'WenQuanYi Zen Hei',    # 文泉驿正黑
            'Noto Sans CJK SC',     # Google Noto字体
            'SimHei',               # 黑体
            'Microsoft YaHei',      # 微软雅黑
            'DejaVu Sans'           # 备选英文字体
        ]
        
        plt.rcParams['font.sans-serif'] = chinese_fonts
        plt.rcParams['font.family'] = 'sans-serif'
        
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