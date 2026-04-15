"""Chinese (Simplified) translation dictionary.

Keys mirror :mod:`eit_app.i18n.en`.  Missing keys fall back to English
automatically (handled by :class:`eit_app.i18n.translator.Translator`).

文案规范:
  * 终端用户可见的动词短语使用祈使式 ("开始采集" 而非 "开始采集一帧")
  * 菜单 / 按钮的快捷键用 ``(&X)`` 后缀, ``X`` 为大写字母
  * 避免冗余副词 ("已经完成" -> "已完成")
  * 保持单行, 不在 dict 值里塞换行
"""

from __future__ import annotations

TRANSLATIONS: dict[str, str] = {
    # ------------------------------------------------------------------
    # Phase 0 validation keys
    # ------------------------------------------------------------------
    "_test.hello": "\u4f60\u597d\uff0c\u4e16\u754c\uff01",               # 你好，世界！
    "_test.greeting": "\u6b22\u8fce\uff0c{name}",                         # 欢迎，{name}
    "_test.plural": "\u4f60\u6709 {n} \u4e2a\u5f85\u529e\u4e8b\u9879",    # 你有 {n} 个待办事项

    # ------------------------------------------------------------------
    # Application chrome skeleton
    # ------------------------------------------------------------------
    "app.title": "EIT \u5de5\u4f5c\u7ad9",    # EIT 工作站
}
