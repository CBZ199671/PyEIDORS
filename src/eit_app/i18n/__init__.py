"""Runtime i18n package for the EIT Workstation.

Public API — import only from this module (not from ``translator`` /
``en`` / ``zh`` directly)::

    from eit_app.i18n import (
        t,                    # translate a key in the current language
        translator,           # the Translator QObject (for signal connections)
        set_language,         # programmatic language switch
        current_language,     # get the active language code
        init_from_settings,   # load persisted language at startup
    )
"""

from __future__ import annotations

from typing import Any

from eit_app.i18n.translator import Translator, get_translator


def t(key: str, **kwargs: Any) -> str:
    """Translate *key* in the current language."""
    return get_translator().t(key, **kwargs)


def set_language(lang: str, *, persist: bool = True) -> None:
    """Change the active language and notify all subscribers."""
    get_translator().set_language(lang, persist=persist)


def current_language() -> str:
    """Return the active language code (``"zh"`` or ``"en"``)."""
    return get_translator().language


def translator() -> Translator:
    """Return the :class:`Translator` QObject (for signal connections)."""
    return get_translator()


def init_from_settings() -> None:
    """Load the persisted language — call once after ``QApplication`` is created."""
    get_translator().init_from_settings()


__all__ = [
    "Translator",
    "t",
    "set_language",
    "current_language",
    "translator",
    "init_from_settings",
]
