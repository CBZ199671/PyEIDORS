"""Lightweight in-memory i18n translator with runtime language switching.

Design
------
* A single :class:`Translator` QObject holds the current language and the
  English / Chinese dictionaries.
* Widgets subscribe to :attr:`Translator.language_changed` and call
  :func:`t` (or ``translator().t``) to refresh their visible strings — no
  ``QTranslator`` / ``.ts`` file compilation required.
* The current language is persisted through :class:`QSettings` under the
  organisation / application names configured by ``QCoreApplication``.

Usage
-----
Widget side::

    from eit_app.i18n import t, translator

    class MyWidget(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self._button = QPushButton()
            translator().language_changed.connect(self._retranslate)
            self._retranslate()

        def _retranslate(self) -> None:
            self._button.setText(t("my.widget.button"))

Bootstrap side (call once, after ``QApplication`` is created and
``setOrganizationName`` / ``setApplicationName`` have been called)::

    from eit_app.i18n import init_from_settings
    init_from_settings()
"""

from __future__ import annotations

import logging
import os
from typing import Any

from PySide6.QtCore import QObject, QSettings, Signal

from eit_app.i18n.en import TRANSLATIONS as _EN
from eit_app.i18n.zh import TRANSLATIONS as _ZH

log = logging.getLogger(__name__)

_SETTINGS_KEY = "ui/language"
_SUPPORTED: tuple[str, ...] = ("zh", "en")
_DEFAULT_FALLBACK = "en"


def _detect_system_language() -> str:
    """Best-effort detection of the user's preferred language.

    Order of precedence:
      1. POSIX locale env vars (``LC_ALL`` / ``LC_MESSAGES`` / ``LANG`` /
         ``LANGUAGE``).  Most reliable on Linux / WSL.
      2. :func:`locale.getlocale`.  Covers native Windows more robustly.
      3. Windows-specific fallback via ``GetUserDefaultUILanguage``.

    Returns ``"zh"`` on Chinese systems, otherwise ``"en"``.
    """
    for env_var in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        value = os.environ.get(env_var, "")
        if value.lower().startswith("zh"):
            return "zh"

    try:
        import locale as _locale

        lang, _ = _locale.getlocale()
        if lang and lang.lower().startswith(("zh", "chinese")):
            return "zh"
    except Exception:  # pragma: no cover — defensive
        pass

    if os.name == "nt":
        try:
            import ctypes

            lang_id = ctypes.windll.kernel32.GetUserDefaultUILanguage() & 0xFF
            # 0x04 = LANG_CHINESE
            if lang_id == 0x04:
                return "zh"
        except Exception:  # pragma: no cover — defensive
            pass

    return _DEFAULT_FALLBACK


class Translator(QObject):
    """Singleton that holds the current language and translation dicts."""

    language_changed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._dicts: dict[str, dict[str, str]] = {"zh": _ZH, "en": _EN}
        self._language = _detect_system_language()
        self._settings_initialized = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def language(self) -> str:
        """Current active language code (``"zh"`` or ``"en"``)."""
        return self._language

    @property
    def supported_languages(self) -> tuple[str, ...]:
        return _SUPPORTED

    def init_from_settings(self) -> None:
        """Load the saved language preference from :class:`QSettings`.

        Must be called after :class:`QCoreApplication` has been instantiated
        and after ``setOrganizationName`` / ``setApplicationName``.
        Idempotent — safe to call multiple times.
        """
        settings = QSettings()
        saved = settings.value(_SETTINGS_KEY, "", type=str) or ""
        if saved in _SUPPORTED:
            self._language = saved
            log.info("Loaded persisted language preference: %s", saved)
        else:
            log.info(
                "No persisted language preference; using detected: %s",
                self._language,
            )
        self._settings_initialized = True

    def set_language(self, lang: str, *, persist: bool = True) -> None:
        """Switch to *lang* and emit :attr:`language_changed` if different."""
        if lang not in _SUPPORTED:
            log.warning(
                "Unsupported language code %r; keeping %s", lang, self._language
            )
            return
        if lang == self._language:
            return
        self._language = lang
        if persist:
            settings = QSettings()
            settings.setValue(_SETTINGS_KEY, lang)
        log.info("Language switched to: %s", lang)
        self.language_changed.emit(lang)

    def t(self, key: str, **kwargs: Any) -> str:
        """Translate *key* in the current language.

        Falls back to English, then to the literal *key* for dev visibility.
        Supports :py:meth:`str.format` style interpolation via ``**kwargs``.
        """
        value = self._dicts[self._language].get(key)
        if value is None:
            value = self._dicts[_DEFAULT_FALLBACK].get(key)
            if value is None:
                log.debug(
                    "Missing translation for key=%r (lang=%s)",
                    key,
                    self._language,
                )
                value = key
        if kwargs:
            try:
                value = value.format(**kwargs)
            except (KeyError, IndexError, ValueError) as exc:
                log.warning("Format error for key=%r kwargs=%r: %s", key, kwargs, exc)
        return value


# ----------------------------------------------------------------------
# Module-level lazy singleton
# ----------------------------------------------------------------------

_INSTANCE: Translator | None = None


def get_translator() -> Translator:
    """Return (creating on first call) the global :class:`Translator` singleton."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = Translator()
    return _INSTANCE
