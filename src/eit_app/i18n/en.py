"""English translation dictionary.

Keys follow a dotted scope convention: ``<area>.<component>.<element>``.
Formatting placeholders use :py:meth:`str.format` syntax (e.g. ``{count}``).

This file is a plain Python ``dict`` (not JSON / YAML) on purpose so that:
  * IDEs can autocomplete key references,
  * ``grep`` finds usages directly,
  * static checkers can flag unused keys.
"""

from __future__ import annotations

TRANSLATIONS: dict[str, str] = {
    # ------------------------------------------------------------------
    # Application chrome
    # ------------------------------------------------------------------
    "app.title": "EIT Workstation",

    # ------------------------------------------------------------------
    # Tab labels  (kept as short domain nouns for a tight tab bar)
    # ------------------------------------------------------------------
    "tab.hardware": "Hardware",
    "tab.simulation": "Simulation",
    "tab.dataset": "Dataset",
    "tab.database": "Database",

    # ------------------------------------------------------------------
    # File menu
    # ------------------------------------------------------------------
    "menu.file": "&File",
    "menu.file.settings": "&Settings\u2026",
    "menu.file.exit": "E&xit",

    # ------------------------------------------------------------------
    # Tools menu
    # ------------------------------------------------------------------
    "menu.tools": "&Tools",
    "menu.tools.interop_hub": "EIDORS &Interop Hub\u2026",

    # ------------------------------------------------------------------
    # Language menu
    # ------------------------------------------------------------------
    "menu.language": "&Language",
    "menu.language.zh": "\u4e2d\u6587",            # 中文 — always in native script
    "menu.language.en": "English",                   # always in English
    "menu.language.tooltip": "Switch between Chinese and English",
}
