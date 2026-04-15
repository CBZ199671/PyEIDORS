"""Chinese (Simplified) translation dictionary.

Keys mirror :mod:`eit_app.i18n.en`.  Missing keys fall back to English
automatically (handled by :class:`eit_app.i18n.translator.Translator`).

文案规范
--------
* 终端用户可见的动词短语使用祈使式 ("开始采集" 而非 "开始采集一帧")
* 菜单 / 按钮的快捷键用 ``(&X)`` 后缀, ``X`` 为大写字母
* 避免冗余副词 ("已经完成" -> "已完成")
* Tab 标签统一为 2-3 字的名词, 与英文版并列结构保持对齐
"""

from __future__ import annotations

TRANSLATIONS: dict[str, str] = {
    # ------------------------------------------------------------------
    # Application chrome
    # ------------------------------------------------------------------
    "app.title": "EIT \u5de5\u4f5c\u7ad9",              # EIT 工作站

    # ------------------------------------------------------------------
    # Tab labels
    # ------------------------------------------------------------------
    "tab.hardware": "\u5b9e\u6d4b",                      # 实测
    "tab.simulation": "\u4eff\u771f",                    # 仿真
    "tab.dataset": "\u6570\u636e\u96c6",                 # 数据集
    "tab.database": "\u6570\u636e\u5e93",                # 数据库

    # ------------------------------------------------------------------
    # File menu
    # ------------------------------------------------------------------
    "menu.file": "\u6587\u4ef6(&F)",                     # 文件(&F)
    "menu.file.settings": "\u8bbe\u7f6e(&S)\u2026",      # 设置(&S)…
    "menu.file.exit": "\u9000\u51fa(&X)",                # 退出(&X)

    # ------------------------------------------------------------------
    # Tools menu
    # ------------------------------------------------------------------
    "menu.tools": "\u5de5\u5177(&T)",                    # 工具(&T)
    "menu.tools.interop_hub": "EIDORS \u4e92\u64cd\u4f5c(&I)\u2026",   # EIDORS 互操作(&I)…

    # ------------------------------------------------------------------
    # Language menu
    # ------------------------------------------------------------------
    "menu.language": "\u8bed\u8a00(&L)",                 # 语言(&L)
    "menu.language.zh": "\u4e2d\u6587",                  # 中文
    "menu.language.en": "English",
    "menu.language.tooltip": "\u5728\u4e2d\u6587\u548c\u82f1\u6587\u4e4b\u95f4\u5207\u6362",  # 在中文和英文之间切换
}
