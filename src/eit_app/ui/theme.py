"""Global Qt theme helpers for the EIT workstation."""

from __future__ import annotations

import base64
import weakref

from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QWidget


# =====================================================================
# Inline SVG helpers (used for QSpinBox / QDateEdit step arrows)
# =====================================================================
#
# The stylesheet references four triangle arrows (up/down × idle/hover)
# plus a fifth for the date-edit drop-down.  Qt's SVG rasteriser honours
# the device pixel ratio — but only when the SVG root has a viewBox and
# NO intrinsic ``width``/``height`` attributes.  The earlier hardcoded
# payload included ``width="10" height="10"`` which pinned rasterisation
# to 10×10 pixels, so arrows showed up blurry at 2× DPI scaling.
#
# Build the data URLs from code so:
#   * We can flip fill colours per theme mode without hand-editing
#     base64 blobs.
#   * The SVG always comes out DPR-aware (no intrinsic size).


def _arrow_data_url(direction: str, fill: str) -> str:
    """Return a ``data:image/svg+xml;base64,…`` URL for a spinbox arrow.

    ``direction`` is 'up' or 'down'; ``fill`` is a CSS color string
    (e.g. ``"#5b6573"``).  The returned URL can be dropped into any
    ``image: url(...)`` declaration.
    """
    if direction == "up":
        points = "5,2.5 1.5,7 8.5,7"
    else:
        points = "5,7.5 1.5,3 8.5,3"
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        f'<polygon points="{points}" fill="{fill}"/>'
        "</svg>"
    )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'url("data:image/svg+xml;base64,{encoded}")'


# Light-mode fills mirror the existing palette (#5b6573 neutral,
# #1f5d8b accent on hover).  Dark mode brightens both so the arrows
# stay visible against the dark input fill.
_ARROW_URLS_LIGHT = {
    "up_idle": _arrow_data_url("up", "#5b6573"),
    "up_hover": _arrow_data_url("up", "#1f5d8b"),
    "down_idle": _arrow_data_url("down", "#5b6573"),
    "down_hover": _arrow_data_url("down", "#1f5d8b"),
}
_ARROW_URLS_DARK = {
    "up_idle": _arrow_data_url("up", "#a7b2c2"),
    "up_hover": _arrow_data_url("up", "#5ca8e0"),
    "down_idle": _arrow_data_url("down", "#a7b2c2"),
    "down_hover": _arrow_data_url("down", "#5ca8e0"),
}


# =====================================================================
# Light / Dark mode infrastructure
# =====================================================================
#
# The main `_APP_STYLESHEET` string at the bottom of this file is the
# Light variant — hundreds of rules tuned for a high-luminance
# background.  Instead of maintaining a fully independent Dark
# stylesheet (which would double the maintenance burden for every
# future tweak), we append a compact Dark *overlay* that overrides
# only the color tokens: surfaces, borders, fills, and focus rings.
# The overlay is appended AFTER the main stylesheet so Qt's last-rule-
# wins semantics apply — positional priority, not `!important`.
#
# This approach leaves every layout rule (paddings, radii, font sizes,
# selectors) untouched, which is what actually drives consistency
# across the 4 tabs.  The tradeoff is that the Dark variant inherits
# any spacing decisions made for Light — if a Dark-specific rule ever
# needs different geometry, add it to `_DARK_OVERLAY` below.

_MODE_SETTINGS_KEY = "ui/theme_mode"
_DEFAULT_MODE = "light"

# Callers subscribe via theme_mode_changed to update non-stylesheet
# surfaces (e.g. matplotlib plot backgrounds, pyqtgraph axes).  We use
# weak listener refs so closed widgets do not stay alive just because
# they once subscribed to theme changes.
_mode_listeners: list[_ThemeModeListener] = []
_current_mode: str = _DEFAULT_MODE


class _ThemeModeListener:
    def __init__(self, listener) -> None:
        owner = getattr(listener, "__self__", None)
        func = getattr(listener, "__func__", None)
        if owner is not None and func is not None:
            self.key = (id(owner), id(func))
            self._ref = weakref.WeakMethod(listener)
            return

        self.key = (id(listener), None)
        try:
            self._ref = weakref.ref(listener)
        except TypeError:
            self._ref = lambda: listener

    def resolve(self):
        return self._ref()


def current_theme_mode() -> str:
    """Return the active theme mode ('light' or 'dark')."""
    return _current_mode


def set_theme_mode(
    app: QApplication,
    mode: str,
    *,
    persist: bool = True,
    apply_stylesheet: bool = True,
) -> None:
    """Switch between 'light' and 'dark' and re-apply the stylesheet.

    Persistence via QSettings is on by default so the next launch
    restores the user's preference.  Pass ``persist=False`` for
    tests or preview flows that should not mutate the store.  Pass
    ``apply_stylesheet=False`` when a narrow test only needs to exercise
    palette subscribers without forcing a full QApplication repolish.
    """
    global _current_mode
    mode = mode if mode in ("light", "dark") else _DEFAULT_MODE
    _current_mode = mode
    if persist:
        QSettings("PyEIDORS", "EITWorkstation").setValue(_MODE_SETTINGS_KEY, mode)
    if apply_stylesheet:
        # The stylesheet only covers widgets it explicitly selects; plain
        # container QWidgets, QSplitter bodies, and group-box title
        # margins erase themselves with the QPalette Window color.  Keep
        # the palette in lockstep with the mode so those regions don't
        # stay platform-default light (#efefef) and bleed through layout
        # gaps as bright bands in dark mode.
        _apply_mode_palette(app, mode)
        app.setStyleSheet(_build_stylesheet(mode))
    live_listeners: list[_ThemeModeListener] = []
    for listener_ref in list(_mode_listeners):
        listener = listener_ref.resolve()
        if listener is None:
            continue
        try:
            listener(mode)
        except Exception:  # pragma: no cover — best effort
            continue
        live_listeners.append(listener_ref)
    _mode_listeners[:] = live_listeners


# Captured the first time set_theme_mode runs so switching back to light
# restores the exact platform-default palette instead of a hand-rolled one.
_BASE_LIGHT_PALETTE: QPalette | None = None


def _build_dark_palette() -> QPalette:
    """QPalette mirror of the dark QSS color system.

    Covers the paint paths QSS never reaches: plain QWidget erase fills,
    QSplitter bodies, group-box title margins, and any widget that asks
    the palette directly.
    """
    palette = QPalette()
    window = QColor("#1a1f26")
    base = QColor("#222831")
    text = QColor("#dbe1ea")
    disabled = QColor("#5e6876")
    palette.setColor(QPalette.ColorRole.Window, window)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, base)
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#262d38"))
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.PlaceholderText, disabled)
    palette.setColor(QPalette.ColorRole.Button, QColor("#2a313a"))
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, base)
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#1e4870"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ecf4fb"))
    palette.setColor(QPalette.ColorRole.Link, QColor("#5ca8e0"))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, disabled)
    return palette


def _apply_mode_palette(app: QApplication, mode: str) -> None:
    global _BASE_LIGHT_PALETTE
    if _BASE_LIGHT_PALETTE is None:
        _BASE_LIGHT_PALETTE = QPalette(app.palette())
    if mode == "dark":
        app.setPalette(_build_dark_palette())
    else:
        app.setPalette(QPalette(_BASE_LIGHT_PALETTE))


def init_theme_mode_from_settings() -> str:
    """Resolve the persisted mode without applying it.

    apply_app_theme() reads the result and calls set_theme_mode
    during startup.  Kept separate so callers can introspect the
    stored preference before the QApplication exists.
    """
    global _current_mode
    stored = QSettings("PyEIDORS", "EITWorkstation").value(
        _MODE_SETTINGS_KEY, _DEFAULT_MODE
    )
    mode = str(stored) if stored in ("light", "dark") else _DEFAULT_MODE
    _current_mode = mode
    return mode


def subscribe_theme_mode(listener) -> None:
    """Register a ``callable(mode: str)`` invoked on every mode switch.

    Useful for plot widgets that need to update their own non-QSS
    background colors when the app flips to dark.
    """
    listener_ref = _ThemeModeListener(listener)
    _mode_listeners[:] = [
        existing for existing in _mode_listeners if existing.resolve() is not None
    ]
    if all(existing.key != listener_ref.key for existing in _mode_listeners):
        _mode_listeners.append(listener_ref)


def _build_stylesheet(mode: str) -> str:
    """Concatenate the base Light stylesheet with any mode-specific overlay.

    Also performs inline substitution of the __ARROW_*__ placeholders
    so every spinbox / date-edit arrow uses a DPR-aware, theme-matched
    SVG URL.  Light mode uses the neutral/accent fills; Dark mode
    overrides with brighter versions so the arrow stays visible against
    the dark input background.
    """
    urls = _ARROW_URLS_DARK if mode == "dark" else _ARROW_URLS_LIGHT
    base = (
        _APP_STYLESHEET.replace("__ARROW_UP_IDLE__", urls["up_idle"])
        .replace("__ARROW_UP_HOVER__", urls["up_hover"])
        .replace("__ARROW_DOWN_IDLE__", urls["down_idle"])
        .replace("__ARROW_DOWN_HOVER__", urls["down_hover"])
    )
    if mode == "dark":
        # The overlay below re-skins spinbox / date-edit button fills
        # to match the dark panel chrome; arrow SVG URLs are already
        # picked from _ARROW_URLS_DARK above, so the overlay only has
        # to deal with the button backgrounds.
        return base + "\n\n" + _DARK_OVERLAY
    return base


# Latin-first base families: Segoe UI on Windows, Noto Sans / DejaVu Sans as
# Linux fallbacks.  CJK fallbacks are appended at runtime based on what
# eit_app.ui.fonts actually registered with the Qt font database (Microsoft
# YaHei, Noto Sans CJK SC, etc.).  Qt walks the family list left-to-right and
# falls back per-glyph when a character isn't drawable — the same mechanism
# matplotlib uses.  Without this list Qt renders tofu boxes for Chinese
# labels on systems where the primary family lacks CJK coverage.
_LATIN_BASE_FAMILIES = ["Segoe UI", "Noto Sans", "DejaVu Sans"]
_CJK_FALLBACK_CANDIDATES = [
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
    "PingFang SC",
    "WenQuanYi Zen Hei",
    "SimSun",
    "SimHei",
]


def _resolve_ui_font_families() -> list[str]:
    """Pick a Latin-first + CJK-fallback family list for Qt chrome.

    Call this AFTER `configure_runtime_fonts(app)` so any bundled Windows
    fonts already sit in the Qt font database.  If no CJK face is found
    (pure-Linux CI runners), we still return the Latin list — Qt will
    substitute its own last-resort face for missing glyphs.
    """
    known = set(QFontDatabase.families())
    cjk = [name for name in _CJK_FALLBACK_CANDIDATES if name in known]
    return _LATIN_BASE_FAMILIES + cjk


def apply_app_theme(app: QApplication) -> None:
    """Apply a consistent workstation theme to the entire application.

    Reads the persisted theme mode ('light' / 'dark') from QSettings
    and applies the matching stylesheet.  Callers who want to observe
    later mode flips (e.g. plot widgets needing to re-paint their
    background) should register via ``subscribe_theme_mode``.
    """
    font = QFont()
    font.setFamilies(_resolve_ui_font_families())
    font.setPointSize(10)
    app.setFont(font)
    # Read persisted mode once at startup; set_theme_mode handles
    # emission to listeners for every subsequent flip.
    mode = init_theme_mode_from_settings()
    set_theme_mode(app, mode, persist=False)


def set_button_role(widget: QPushButton, role: str) -> None:
    """Tag a button with a semantic role used by the global stylesheet."""
    widget.setProperty("buttonRole", role)
    _repolish(widget)


def set_section_header(widget: QWidget) -> None:
    """Tag a label as a section header."""
    widget.setProperty("uiSectionHeader", True)
    _repolish(widget)


def set_hint_text(widget: QWidget) -> None:
    """Tag a label as supportive hint text."""
    widget.setProperty("uiHintText", True)
    _repolish(widget)


def set_subtle_value(widget: QWidget) -> None:
    """Tag a widget as a subtle informational value."""
    widget.setProperty("uiSubtleValue", True)
    _repolish(widget)


def set_panel_role(widget: QWidget, role: str) -> None:
    """Tag a widget with an instrument-panel role."""
    widget.setProperty("panelRole", role)
    _repolish(widget)


def set_embedded_step_panel(widget: QWidget) -> None:
    """Style a panel for use inside the left-side workflow toolbox."""
    widget.setProperty("embeddedStepPanel", True)
    _repolish(widget)


# =====================================================================
# Plot palette (pyqtgraph + matplotlib backgrounds, axis colors, grids)
# =====================================================================
#
# pyqtgraph and matplotlib paint their own canvas — they don't honour
# QSS — so plot widgets must explicitly read these colors and re-paint
# themselves on theme_mode changes via subscribe_theme_mode().
#
# Each entry is a small dict so callers can read what they need without
# unpacking ordered tuples.  Adding a new color requires updating both
# variants.
_PLOT_PALETTE_LIGHT = {
    "bg": "#f8fbfe",  # main plot canvas
    "panel_bg": "#f4f7fb",  # outer figure / facecolor for matplotlib
    "axes_bg": "#fbfdff",  # matplotlib axes (slightly brighter than panel)
    "text": "#243447",
    "grid": "#d6e1ec",
    "border": "#c7d4e2",
    # Reconstruction-image widget (pyqtgraph) extras — domain outline,
    # electrode arc, and label color.
    "domain": "#8ba0b8",
    "electrode": "#355c7d",
    "label": "#243447",
    # Color for the matplotlib placeholder/error/loading caption text
    "caption": "#5b6573",
    "caption_loading": "#1f5d8b",
    "caption_error": "#8b2f2f",
    # Inhomogeneity-highlight overlay used by the 3D conductivity
    # widget — amber on light, pumpkin on dark for contrast against
    # the viridis cmap underneath.
    "highlight": "#f39c12",
}

_PLOT_PALETTE_DARK = {
    "bg": "#161b22",  # darker than QSS canvas (#1a1f26) for plot contrast
    "panel_bg": "#1a1f26",
    "axes_bg": "#1e242c",
    "text": "#dbe1ea",
    "grid": "#2f3742",
    "border": "#3e4754",
    "domain": "#5d6a7a",
    "electrode": "#7cbeee",
    "label": "#dbe1ea",
    "caption": "#8b97a7",
    "caption_loading": "#5ca8e0",
    "caption_error": "#f09e95",
    "highlight": "#ffa94d",
}


def plot_palette() -> dict:
    """Return the active plot palette (matches current_theme_mode()).

    Returned dict is a *copy* so callers can mutate / compose without
    touching the module-level source.  Keys: bg, panel_bg, axes_bg,
    text, grid, border, domain, electrode, label, caption, caption_loading,
    caption_error.
    """
    palette = _PLOT_PALETTE_DARK if _current_mode == "dark" else _PLOT_PALETTE_LIGHT
    return dict(palette)


# ---------------------------------------------------------------------
# Overlay stylesheet helpers (used by plot widgets' transient overlays)
# ---------------------------------------------------------------------
#
# All four plot widgets (LivePlot, BoundaryVoltage, Reconstruction,
# ConductivityImage) show a centered QLabel on top of their canvas for
# the "empty" / "loading" / "error" states.  Historically those labels
# used `background: transparent`, which made old plot content bleed
# through the transient text — visually messy and semantically
# confusing when the user kicks off a new solve while previous results
# are still on screen.
#
# The three helpers below return theme-aware stylesheets that
# implement a cleaner pattern:
#   - loading / error:  full-canvas opaque scrim in panel_bg + centered
#                       caption in accent/red.  The scrim covers any
#                       stale plot content so the overlay reads as a
#                       clean "we're working" or "something broke"
#                       modal state.
#   - empty:            transparent background + muted caption, because
#                       there's nothing behind to scrim out.
def loading_scrim_stylesheet() -> str:
    p = plot_palette()
    return (
        f"color: {p['caption_loading']}; "
        "font-size: 13px; font-weight: 700; "
        f"background: {p['panel_bg']}; "
        "padding: 20px;"
    )


def error_scrim_stylesheet() -> str:
    p = plot_palette()
    return (
        f"color: {p['caption_error']}; "
        "font-size: 12px; font-weight: 700; "
        f"background: {p['panel_bg']}; "
        "padding: 20px;"
    )


def empty_placeholder_stylesheet() -> str:
    p = plot_palette()
    return (
        f"color: {p['caption']}; "
        "font-size: 12px; font-weight: 600; "
        "background: transparent;"
    )


# ---------------------------------------------------------------------
# Mini-card / inline stylesheet helpers
# ---------------------------------------------------------------------
#
# These widgets paint their own QSS via setStyleSheet() and don't go
# through the global app stylesheet — so they have to call back into
# theme.py to get the right per-mode color triplet.  Each helper
# returns a complete stylesheet string and the caller assigns it via
# widget.setStyleSheet(...).  Re-paint on later mode flips by calling
# subscribe_theme_mode() with a bound method that re-applies the
# helper's output.
_CARD_PALETTE_LIGHT = {
    "info_bg": "#f5f9fd",  # subtle blue-tinted info card
    "info_border": "#dbe4ef",
    "info_text": "#5b6573",
    "info_accent": "#1f5d8b",  # bold-text accent (e.g. count label)
    "info_subtle": "#6a7686",
    "value_bg": "#f7f9fc",  # readonly value boxes (Session/Dataset summary fields)
    "value_border": "#d8dee9",
    "value_text": "#243447",
    "next_action_bg": "#edf4fb",  # accent-bordered "next step" hint banner
    "next_action_border": "#1f5d8b",
    "next_action_text": "#243447",
    "selection_bg": "#f5f9fd",  # Database tab "selection status" line
    "selection_border": "#dbe4ef",
    "selection_text": "#5b6573",
}

_CARD_PALETTE_DARK = {
    "info_bg": "#222831",
    "info_border": "#3e4754",
    "info_text": "#a7b2c2",
    "info_accent": "#9dc9ea",
    "info_subtle": "#8b97a7",
    "value_bg": "#262d38",
    "value_border": "#3e4754",
    "value_text": "#dbe1ea",
    "next_action_bg": "#1f3148",
    "next_action_border": "#5ca8e0",
    "next_action_text": "#dbe1ea",
    "selection_bg": "#222831",
    "selection_border": "#3e4754",
    "selection_text": "#a7b2c2",
}


def card_palette() -> dict:
    """Return the active mini-card palette (matches current_theme_mode())."""
    return dict(_CARD_PALETTE_DARK if _current_mode == "dark" else _CARD_PALETTE_LIGHT)


def info_card_stylesheet() -> str:
    """Stylesheet for soft-tinted info cards (Database stats card etc.)."""
    p = card_palette()
    return (
        f"background: {p['info_bg']}; "
        f"border: 1px solid {p['info_border']}; "
        "border-radius: 8px; padding: 10px;"
    )


def stat_count_stylesheet() -> str:
    """Stylesheet for the bold accent number inside an info card."""
    p = card_palette()
    return (
        f"color: {p['info_accent']}; font-weight: 700; font-size: 15px;"
        " background: transparent; border: none; padding: 0;"
    )


def stat_subtle_stylesheet() -> str:
    """Stylesheet for the small subtitle under an info card stat."""
    p = card_palette()
    return (
        f"color: {p['info_subtle']}; font-size: 11px;"
        " background: transparent; border: none; padding: 0;"
    )


def field_value_stylesheet() -> str:
    """Stylesheet for read-only value boxes (Session/Dataset summary)."""
    p = card_palette()
    return (
        "padding: 4px 6px; "
        f"border: 1px solid {p['value_border']}; "
        "border-radius: 4px; "
        f"background: {p['value_bg']}; "
        f"color: {p['value_text']};"
    )


def next_action_stylesheet() -> str:
    """Stylesheet for the accent-bordered 'next step' hint banner."""
    p = card_palette()
    return (
        "padding: 8px 10px; "
        f"border-left: 4px solid {p['next_action_border']}; "
        f"background: {p['next_action_bg']}; "
        "border-radius: 8px; "
        f"color: {p['next_action_text']};"
    )


def selection_status_stylesheet() -> str:
    """Stylesheet for the Database tab's selection-status line."""
    p = card_palette()
    return (
        f"background: {p['selection_bg']}; border: 1px solid {p['selection_border']};"
        " border-radius: 6px; padding: 6px 10px;"
        f" color: {p['selection_text']}; font-size: 12px;"
    )


_TONE_PALETTE_LIGHT = {
    "idle": ("#5b6573", "#f7f9fc", "#d8dee9"),
    "warn": ("#8a4b08", "#fff4dc", "#f1c27d"),
    "ready": ("#0f5f3d", "#e6f6ee", "#7bc69a"),
    "active": ("#0b4f80", "#e9f4ff", "#7fb2e5"),
    "error": ("#8c1d18", "#fdecec", "#f1a6a1"),
}

# Dark-mode tones: brighter foreground for contrast against dark fills,
# with 18-22% alpha fills that sit on top of the dark panel background.
# Borders are a muted accent of the same hue so the chip outline stays
# legible on a #1f2630 surface.
_TONE_PALETTE_DARK = {
    "idle": ("#c7d0db", "#2a313a", "#3e4754"),
    "warn": ("#f3c97a", "#3a2f16", "#7a5a22"),
    "ready": ("#7bcfa0", "#17321f", "#2a6a42"),
    "active": ("#7cbeee", "#17324c", "#2a5a84"),
    "error": ("#f09e95", "#35201d", "#7a3830"),
}


def tone_palette(tone: str) -> tuple[str, str, str]:
    """Return foreground/background/border colors for a named UI tone.

    Resolves against the active theme mode so chips painted by
    apply_state_chip/banner automatically follow dark-mode flips.
    """
    palette = _TONE_PALETTE_DARK if _current_mode == "dark" else _TONE_PALETTE_LIGHT
    return palette.get(tone, palette["idle"])


def apply_state_chip(
    label: QLabel, *, tone: str, compact: bool = False, emphasized: bool = False
) -> None:
    """Apply the shared state-chip style used across summary cards and status bar."""
    fg, bg, border = tone_palette(tone)
    padding = "2px 10px" if compact else "4px 8px"
    radius = "8px" if compact else "10px"
    font_weight = "700" if emphasized else "600"
    label.setStyleSheet(
        f"padding: {padding}; "
        f"border-radius: {radius}; "
        f"font-weight: {font_weight}; "
        f"color: {fg}; "
        f"background: {bg}; "
        f"border: 1px solid {border};"
    )


def apply_state_banner(label: QLabel, *, tone: str) -> None:
    """Apply the shared banner style for the large summary status header."""
    fg, bg, border = tone_palette(tone)
    label.setStyleSheet(
        "font-weight: 700; "
        "letter-spacing: 1px; "
        "padding: 9px 10px; "
        "border-radius: 10px; "
        f"color: {fg}; "
        f"background: {bg}; "
        f"border: 1px solid {border};"
    )


def _repolish(widget: QWidget) -> None:
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


_APP_STYLESHEET = """
QMainWindow {
    background: #eef3f8;
}

QDockWidget {
    color: #243447;
    font-weight: 600;
}

QDockWidget::title {
    text-align: left;
    padding: 8px 10px;
    background: #dce7f3;
    border-bottom: 1px solid #c3d1df;
}

QToolBox {
    background: transparent;
    border: none;
}

QToolBox::tab {
    background: #dfe9f4;
    border: 1px solid #c5d2df;
    border-radius: 8px;
    padding: 5px 12px;
    margin: 2px 0;
    color: #243447;
    font-weight: 600;
}

QToolBox::tab:hover:!selected {
    /* Subtle hover feedback so users know the tab is clickable. Kept
       quieter than the selected state so it doesn't compete with it. */
    background: #cfdeef;
    border-color: #a8bccf;
    color: #14253a;
}

QToolBox::tab:selected {
    background: #1f3b5b;
    color: #f8fbff;
    border-color: #1f3b5b;
}

QToolBox::tab:disabled {
    color: #8a96a5;
    background: #eff3f8;
    border-color: #d8e0ea;
}

QToolBox#workflowToolbox::tab {
    height: 30px;
    font-size: 12px;
    font-weight: 700;
    padding: 5px 12px;
}

QTabWidget::pane {
    border: none;
    background: #eef3f8;
    top: -1px;
}

QTabBar {
    qproperty-drawBase: 0;
}

QTabBar::tab {
    background: transparent;
    color: #5b6573;
    border: none;
    padding: 10px 22px;
    margin-right: 4px;
    font-weight: 600;
    font-size: 13px;
    min-width: 120px;
}

QTabBar::tab:selected {
    background: #eef3f8;
    color: #1f5d8b;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    border-bottom: 3px solid #1f5d8b;
    font-weight: 700;
}

QTabBar::tab:hover:!selected {
    color: #1f3b5b;
    background: #e3ebf4;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}

QTabBar::tab:focus {
    /* Thicker bottom rule + darker text so keyboard users see which tab
       is about to receive Enter / Space. */
    color: #0f3a5b;
    border-bottom: 3px solid #0f3a5b;
    background: #dde9f4;
}

QTabBar::tab:disabled {
    color: #a2adbb;
    background: transparent;
}

QToolBox#workflowToolbox > QWidget {
    background: #f9fbfe;
}

/* WorkflowShell intermediate QWidgets — explicitly match the canvas
   so platform-palette defaults (#efefef on Linux) don't leak through
   VBoxLayout gaps between the toolbox and any left_footer widget.
   See the matching dark-mode block near the bottom of this file. */
QScrollArea#workflowScroll,
QWidget#workflowScrollViewport,
QWidget#workflowLeftContainer {
    background: #eef3f8;
    border: none;
}

QGroupBox {
    background: #ffffff;
    border: 1px solid #e0e6ee;
    border-radius: 10px;
    margin-top: 14px;
    padding: 12px 12px 10px 12px;
    color: #243447;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 8px;
    color: #1f5d8b;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.3px;
}

QGroupBox[panelRole="summary"] {
    background: #f5f9fd;
    border: 1px solid #bfd1e2;
}

QGroupBox[panelRole="summary"]::title {
    color: #18324d;
}

QGroupBox[summaryHeaderless="true"] {
    margin-top: 0;
    padding-top: 8px;
}

QGroupBox[summaryHeaderless="true"]::title {
    color: transparent;
    height: 0px;
    padding: 0;
    margin: 0;
}

QGroupBox[panelRole="workflow"] {
    background: #f9fbfe;
}

QGroupBox[embeddedStepPanel="true"] {
    background: #fbfdff;
    border: 1px solid #d5dee8;
    border-radius: 12px;
    margin-top: 0;
    padding: 6px 7px 5px 7px;
}

QGroupBox[embeddedStepPanel="true"]::title {
    subcontrol-origin: margin;
    color: transparent;
    padding: 0;
    margin: 0;
    height: 0px;
}

QGroupBox[embeddedStepPanel="true"] QLabel[uiSectionHeader="true"] {
    padding-top: 1px;
    font-size: 11px;
}

QGroupBox[embeddedStepPanel="true"] QLabel[uiHintText="true"] {
    font-size: 11px;
}

QGroupBox[embeddedStepPanel="true"] QLineEdit,
QGroupBox[embeddedStepPanel="true"] QAbstractSpinBox,
QGroupBox[embeddedStepPanel="true"] QPlainTextEdit,
QGroupBox[embeddedStepPanel="true"] QTextEdit,
QGroupBox[embeddedStepPanel="true"] QLineEdit#selectorDisplay {
    padding: 4px 7px;
    min-height: 18px;
    border-radius: 7px;
}

QGroupBox[embeddedStepPanel="true"] QPushButton,
QGroupBox[embeddedStepPanel="true"] QToolButton {
    padding: 4px 8px;
    min-height: 18px;
    border-radius: 7px;
    font-size: 11px;
}

QGroupBox[embeddedStepPanel="true"] QToolButton#selectorButton {
    padding: 0;
    min-width: 22px;
    max-width: 22px;
}

QGroupBox[embeddedStepPanel="true"] QCheckBox {
    spacing: 6px;
    font-size: 11px;
}

QGroupBox[embeddedStepPanel="true"] QCheckBox::indicator {
    width: 16px;
    height: 16px;
}

/* Plain QLabels without a role property inherit Qt's default
   WindowText color (#000000), which is invisible on the dark
   canvas and low-contrast even on the light canvas.  Pin a sane
   default that later role-specific rules can still override. */
QLabel {
    color: #243447;
}

QLabel[uiSectionHeader="true"] {
    color: #1f3b5b;
    font-weight: 700;
    padding-top: 4px;
}

QLabel[uiHintText="true"] {
    color: #5b6573;
}

QLabel[uiSubtleValue="true"] {
    color: #4d5f75;
}

QLineEdit,
QAbstractSpinBox,
QPlainTextEdit,
QTextEdit {
    background: #ffffff;
    color: #243447;
    border: 1px solid #d0d9e3;
    border-radius: 8px;
    padding: 7px 10px;
    selection-background-color: #275d95;
    selection-color: #ffffff;
}

QLineEdit:hover,
QAbstractSpinBox:hover,
QPlainTextEdit:hover,
QTextEdit:hover {
    border-color: #b1c2d3;
}

QLineEdit:focus,
QAbstractSpinBox:focus,
QPlainTextEdit:focus,
QTextEdit:focus {
    /* 2px accent border for obvious focus indication on keyboard nav. */
    border: 2px solid #275d95;
    padding: 6px 9px;
    background: #fbfdff;
}

QLineEdit:disabled,
QAbstractSpinBox:disabled,
QPlainTextEdit:disabled,
QTextEdit:disabled {
    /* Disabled fields need enough contrast that users can still read
       the value they can't edit. */
    color: #6c7a8a;
    background: #eef2f7;
    border-color: #d3dbe4;
}

QComboBox {
    background: #ffffff;
    color: #243447;
    border: 1px solid #c4d0db;
    border-radius: 8px;
    padding: 6px 10px;
    min-height: 20px;
}

QComboBox:hover {
    border-color: #b1c2d3;
    background: #fbfdff;
}

QComboBox:focus {
    border: 2px solid #275d95;
    padding: 5px 9px;
}

QComboBox:disabled {
    color: #6c7a8a;
    background: #eef2f7;
    border-color: #d3dbe4;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: 1px solid #d5dee8;
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
    background: #eef3f8;
}

QComboBox::down-arrow {
    /* Without an explicit image, Qt falls back to its native down-arrow
       which renders as a near-white glyph (#efefef-ish) on platforms
       where QStyle paints it directly — an obvious bright rectangle
       on a dark combobox button.  Use the same DPR-aware SVG as
       QSpinBox / QDateEdit so all dropdowns share one glyph. */
    image: __ARROW_DOWN_IDLE__;
    width: 10px;
    height: 10px;
}
QComboBox::down-arrow:hover {
    image: __ARROW_DOWN_HOVER__;
}

QComboBox QAbstractItemView {
    background: #ffffff;
    color: #243447;
    border: 1px solid #c4d0db;
    border-radius: 6px;
    padding: 4px;
    selection-background-color: #d9e8f7;
    selection-color: #17324c;
    outline: none;
}

QComboBox QAbstractItemView::item {
    padding: 6px 10px;
    min-height: 22px;
}

QComboBox QAbstractItemView::item:hover {
    background: #e4eef9;
}

QScrollArea {
    background: transparent;
    border: none;
}

QScrollBar:vertical {
    background: #eef3f8;
    width: 10px;
    margin: 0;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: #c4d0db;
    min-height: 30px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #a8b8c8;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background: #eef3f8;
    height: 10px;
    margin: 0;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background: #c4d0db;
    min-width: 30px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal:hover {
    background: #a8b8c8;
}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    width: 0;
}

QDialog {
    background: #eef3f8;
    color: #243447;
}

QTreeView,
QListView {
    background: #ffffff;
    color: #243447;
    border: 1px solid #d5dee8;
    border-radius: 6px;
    selection-background-color: #d9e8f7;
    selection-color: #17324c;
    alternate-background-color: #f7fafd;
}

QTreeView::item,
QListView::item {
    padding: 4px 6px;
}

QTreeView::item:hover,
QListView::item:hover {
    background: #e4eef9;
}

QSplitter::handle {
    background: #d5dee8;
}

QSplitter::handle:horizontal {
    width: 3px;
}

QSplitter::handle:vertical {
    height: 3px;
}

QProgressBar {
    background: #dce7f3;
    border: 1px solid #c5d2df;
    border-radius: 6px;
    text-align: center;
    color: #243447;
    height: 18px;
}

QProgressBar::chunk {
    background: #1f5d8b;
    border-radius: 5px;
}

QPushButton {
    background: #ffffff;
    color: #243447;
    border: 1px solid #d0d9e3;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
    min-height: 20px;
    /* Keep Qt's default dotted focus outline suppressed; we paint our own
       focus ring via border styling so keyboard users see a clear ring
       that matches the rest of the theme. */
    outline: none;
}

QPushButton:hover {
    background: #f4f7fa;
    border-color: #b1c2d3;
}

QPushButton:pressed {
    background: #e3ebf4;
    padding-top: 9px;
    padding-bottom: 7px;
}

QPushButton:checked {
    border-width: 2px;
    border-color: #5e7994;
}

QPushButton:focus {
    /* Keyboard focus: 2px accent ring so Tab navigation is visible.
       Uses the same accent as the active tab so the focus cue feels
       consistent across the app. */
    border: 2px solid #1f5d8b;
    padding: 7px 15px;
}

QPushButton:disabled {
    /* Bump contrast to meet WCAG 2.1 AA (4.5:1 for normal text).
       Previous #97a6b5 on #f4f7fa was ~3.6:1 which fails AA. */
    color: #6c7a8a;
    background: #eef2f7;
    border-color: #d3dbe4;
}

QPushButton[buttonRole="primary"] {
    background: #1f5d8b;
    color: #f8fbff;
    border-color: #18496d;
}

QPushButton[buttonRole="primary"]:hover {
    background: #1a5078;
}

QPushButton[buttonRole="primary"]:checked {
    background: #18496d;
    border-color: #123754;
}

QPushButton[buttonRole="primary"]:focus {
    /* Light ring stands out against the dark primary fill. */
    border: 2px solid #9fc8e4;
    padding: 7px 15px;
}

QPushButton[buttonRole="primary"]:disabled {
    background: #7ea3bd;
    color: #f3f7fa;
    border-color: #6b8fa6;
}

QPushButton[buttonRole="success"] {
    background: #1f7a52;
    color: #f8fbff;
    border-color: #17603f;
}

QPushButton[buttonRole="success"]:hover {
    background: #1b6c49;
}

QPushButton[buttonRole="success"]:checked {
    background: #15563a;
    border-color: #0f412c;
}

QPushButton[buttonRole="success"]:focus {
    border: 2px solid #a5d9bf;
    padding: 7px 15px;
}

QPushButton[buttonRole="success"]:disabled {
    background: #89b5a0;
    color: #f3f7fa;
    border-color: #72a087;
}

QPushButton[buttonRole="danger"] {
    background: #8b2f2f;
    color: #fff8f8;
    border-color: #6f2525;
}

QPushButton[buttonRole="danger"]:hover {
    background: #7d2929;
}

QPushButton[buttonRole="danger"]:checked {
    background: #642020;
    border-color: #4d1818;
}

QPushButton[buttonRole="danger"]:focus {
    border: 2px solid #e8b7b4;
    padding: 7px 15px;
}

QPushButton[buttonRole="danger"]:disabled {
    background: #b78585;
    color: #f4eaea;
    border-color: #9d6e6e;
}

QPushButton[buttonRole="subtle"] {
    background: #f7fafc;
}

QPushButton[buttonRole="subtle"]:focus {
    border: 2px solid #1f5d8b;
    padding: 7px 15px;
}

QTableView {
    background: #ffffff;
    alternate-background-color: #f8fbfd;
    border: 1px solid #e0e6ee;
    border-radius: 8px;
    gridline-color: #eef3f8;
    selection-background-color: #d9e8f7;
    selection-color: #17324c;
    outline: 0;
}

QTableView QWidget,
QTreeView QWidget,
QListView QWidget,
QHeaderView {
    background: #ffffff;
}

QTableView::item {
    padding: 4px 6px;
    border: none;
}

QTableView::item:selected {
    background: #d9e8f7;
    color: #17324c;
}

QHeaderView::section {
    background: #f4f7fa;
    color: #5b6573;
    padding: 8px 12px;
    border: none;
    border-bottom: 2px solid #e0e6ee;
    font-weight: 700;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

QHeaderView::section:first {
    border-top-left-radius: 8px;
}

QHeaderView::section:last {
    border-top-right-radius: 8px;
}

QTableCornerButton::section {
    background: #f4f7fa;
    border: none;
    border-bottom: 2px solid #e0e6ee;
}

QAbstractScrollArea::corner {
    background: #f4f7fa;
    border: none;
}

QMenu {
    background: #ffffff;
    border: 1px solid #ccd8e3;
    border-radius: 10px;
    padding: 6px;
}

QMenu::item {
    padding: 7px 12px;
    border-radius: 6px;
}

QMenu::item:selected {
    background: #e4eef9;
    color: #17324c;
}

QToolButton {
    background: #eef3f8;
    color: #243447;
    border: 1px solid #c4d0db;
    border-radius: 8px;
    padding: 4px 10px;
    font-weight: 700;
}

QToolButton:hover {
    background: #e3ebf4;
}

QStatusBar {
    background: #ffffff;
    color: #5b6573;
    border-top: 1px solid #e0e6ee;
    padding: 2px 8px;
    min-height: 26px;
}

QStatusBar::item {
    border: none;
}

QMainWindow::separator {
    background: #e0e6ee;
    width: 1px;
    height: 1px;
}

QCheckBox {
    color: #243447;
    spacing: 8px;
}

QCheckBox:disabled {
    color: #6c7a8a;
}

QCheckBox:focus {
    /* Simple text-only focus cue to avoid fighting with the indicator
       styling (Qt renders a QFocusFrame around the whole check). */
    color: #0f3a5b;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 1.5px solid #c4d0db;
    border-radius: 4px;
    background: #ffffff;
}

QCheckBox::indicator:hover {
    border-color: #1f5d8b;
}

QCheckBox::indicator:focus {
    /* Accent ring around the indicator box for keyboard users. */
    border: 2px solid #1f5d8b;
}

QCheckBox::indicator:checked {
    background: #1f5d8b;
    border-color: #1f5d8b;
    image: none;
}

QCheckBox::indicator:checked:hover {
    background: #2a6fa0;
}

QCheckBox::indicator:disabled {
    background: #eef2f7;
    border-color: #d3dbe4;
}

QCheckBox::indicator:checked:disabled {
    background: #9fbacb;
    border-color: #8aa5b7;
}

/* Spin box buttons with embedded SVG arrows */
QSpinBox::up-button,
QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 20px;
    background: #f4f7fa;
    border-left: 1px solid #e0e6ee;
    border-top-right-radius: 7px;
}

QSpinBox::down-button,
QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 20px;
    background: #f4f7fa;
    border-left: 1px solid #e0e6ee;
    border-top: 1px solid #e0e6ee;
    border-bottom-right-radius: 7px;
}

QSpinBox::up-button:hover,
QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover,
QDoubleSpinBox::down-button:hover {
    background: #d9e8f7;
}

QSpinBox::up-button:pressed,
QDoubleSpinBox::up-button:pressed,
QSpinBox::down-button:pressed,
QDoubleSpinBox::down-button:pressed {
    background: #b6cfe3;
}

QSpinBox::up-arrow,
QDoubleSpinBox::up-arrow {
    image: __ARROW_UP_IDLE__;
    width: 10px;
    height: 10px;
}

QSpinBox::down-arrow,
QDoubleSpinBox::down-arrow {
    image: __ARROW_DOWN_IDLE__;
    width: 10px;
    height: 10px;
}

QSpinBox::up-arrow:hover,
QDoubleSpinBox::up-arrow:hover {
    image: __ARROW_UP_HOVER__;
}

QSpinBox::down-arrow:hover,
QDoubleSpinBox::down-arrow:hover {
    image: __ARROW_DOWN_HOVER__;
}

/* Date edit dropdown */
QDateEdit::drop-down {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 22px;
    background: #f4f7fa;
    border-left: 1px solid #e0e6ee;
    border-top-right-radius: 7px;
    border-bottom-right-radius: 7px;
}

QDateEdit::drop-down:hover {
    background: #d9e8f7;
}

QDateEdit::down-arrow {
    image: __ARROW_DOWN_IDLE__;
    width: 10px;
    height: 10px;
}

QDateEdit::down-arrow:hover {
    image: __ARROW_DOWN_HOVER__;
}
"""


# =====================================================================
# Dark-mode overlay
# =====================================================================
#
# Appended AFTER _APP_STYLESHEET by set_theme_mode('dark').  Only
# overrides color tokens, never layout.  Kept compact and grouped by
# widget so future tweaks are easy to locate.
#
# Palette choices:
#   Canvas:          #1a1f26  (near-black with a blue undertone)
#   Panels:          #222831  (one step up so panels read as "on the canvas")
#   Inputs:          #2a313a  (two steps up; QLineEdit / QComboBox fills)
#   Borders:         #323a45  (low-contrast structure on dark panels)
#   Accent:          #5ca8e0  (brighter than light's #1f5d8b for contrast)
#   Primary text:    #dbe1ea
#   Muted text:      #8b97a7
#   Group titles:    #a7b2c2  (muted, avoids bright section seams)
# Contrast ratios verified for WCAG AA — muted text on panels ~4.6:1,
# primary text on inputs ~9.7:1.
_DARK_OVERLAY = """
/* === Canvas + Dock + Panels === */
QMainWindow, QDialog, QWidget#centralWidget {
    background: #1a1f26;
    color: #dbe1ea;
}
QDockWidget {
    color: #dbe1ea;
}
QDockWidget::title {
    background: #2a313a;
    border-bottom: 1px solid #3e4754;
    color: #dbe1ea;
}
QTabWidget::pane {
    background: #1a1f26;
}

/* === Tab bar === */
QTabBar::tab {
    color: #8b97a7;
}
QTabBar::tab:selected {
    background: #1a1f26;
    color: #8fc8ea;
    border-bottom: 3px solid #5ca8e0;
}
QTabBar::tab:hover:!selected {
    background: #2a313a;
    color: #dbe1ea;
}
QTabBar::tab:focus {
    color: #a8d5f0;
    border-bottom: 3px solid #a8d5f0;
    background: #2a313a;
}

/* === ToolBox (left workflow rail) === */
QToolBox::tab {
    background: #2a313a;
    border: 1px solid #3e4754;
    color: #dbe1ea;
}
QToolBox::tab:hover:!selected {
    background: #353d48;
    border-color: #4d5868;
    color: #eef2f8;
}
QToolBox::tab:selected {
    background: #1e4870;
    color: #ecf4fb;
    border-color: #1e4870;
}
QToolBox::tab:disabled {
    color: #596272;
    background: #23292f;
    border-color: #303640;
}
QToolBox#workflowToolbox > QWidget {
    /* Match the main canvas so the thin strip between a toolbox tab
       header and the GroupBox below doesn't appear as a visibly
       brighter band in dark mode.  (The light theme uses #f9fbfe
       here, which blends seamlessly into the #eef3f8 canvas because
       both colors are near-white; the same 4-point gap is much more
       noticeable at low luminance.) */
    background: #1a1f26;
}

/* WorkflowShell left column — three intermediate QWidgets that would
   otherwise paint Qt's platform-default palette (#efefef near-white
   on Linux) and show through every VBoxLayout gap as a bright
   horizontal band.  The most obvious leak was the strip between the
   last QToolBox tab and the ``left_footer`` widget below it (e.g.
   SessionSummaryPanel on the Hardware tab).  Target them by
   objectName so this doesn't accidentally repaint unrelated widgets. */
QScrollArea#workflowScroll,
QWidget#workflowScrollViewport,
QWidget#workflowLeftContainer {
    background: #1a1f26;
    border: none;
}

/* === GroupBox / section panels === */
QGroupBox {
    background: #222831;
    border: 1px solid #323a45;
    color: #dbe1ea;
}
QGroupBox::title {
    color: #a7b2c2;
    /* The title sits in the GroupBox's 14px margin-top and would
       otherwise show the parent widget's background behind its
       letters.  Give the title its own opaque fill matching the
       GroupBox body so the 'title on an intermediate-brightness
       strip' effect goes away. */
    background: #222831;
}
QGroupBox[panelRole="summary"] {
    /* Flatten summary panels to the same body shade as regular
       group boxes so there's no visible vertical brightness
       cascade.  The accent border is what distinguishes them. */
    background: #222831;
    border: 1px solid #364152;
}
QGroupBox[panelRole="summary"]::title {
    color: #a7b2c2;
    background: #222831;
}
QGroupBox[panelRole="workflow"] {
    background: #1a1f26;
}
QGroupBox[panelRole="workflow"]::title {
    background: #1a1f26;
}
QGroupBox[embeddedStepPanel="true"] {
    background: #222831;
    border: 1px solid #323a45;
}
QGroupBox[embeddedStepPanel="true"]::title {
    background: #222831;
}

/* Dark default for plain QLabels — inherits from canvas color.
   Role-specific selectors below override this where needed. */
QLabel {
    color: #dbe1ea;
}
QLabel[uiSectionHeader="true"] {
    color: #9dc9ea;
}
QLabel[uiHintText="true"] {
    color: #8b97a7;
}
QLabel[uiSubtleValue="true"] {
    color: #a7b2c2;
}

/* === Inputs === */
QLineEdit, QAbstractSpinBox, QPlainTextEdit, QTextEdit {
    background: #2a313a;
    color: #dbe1ea;
    border: 1px solid #3e4754;
    selection-background-color: #5ca8e0;
    selection-color: #0f1419;
}
QLineEdit:hover, QAbstractSpinBox:hover,
QPlainTextEdit:hover, QTextEdit:hover {
    border-color: #5d6a7a;
}
QLineEdit:focus, QAbstractSpinBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {
    border: 2px solid #5ca8e0;
    background: #313a46;
}
QLineEdit:disabled, QAbstractSpinBox:disabled,
QPlainTextEdit:disabled, QTextEdit:disabled {
    color: #5e6876;
    background: #23292f;
    border-color: #2e3540;
}

QComboBox {
    background: #2a313a;
    color: #dbe1ea;
    border: 1px solid #3e4754;
}
QComboBox:hover {
    border-color: #5d6a7a;
    background: #313a46;
}
QComboBox:focus {
    border: 2px solid #5ca8e0;
}
QComboBox:disabled {
    color: #5e6876;
    background: #23292f;
    border-color: #2e3540;
}
QComboBox::drop-down {
    border-left: 1px solid #3e4754;
    background: #23292f;
}
QComboBox QAbstractItemView {
    background: #222831;
    color: #dbe1ea;
    border: 1px solid #3e4754;
    selection-background-color: #1e4870;
    selection-color: #ecf4fb;
}
QComboBox QAbstractItemView::item:hover {
    background: #2d3543;
}

/* === Buttons === */
QPushButton {
    background: #2a313a;
    color: #dbe1ea;
    border: 1px solid #3e4754;
}
QPushButton:hover {
    background: #313a46;
    border-color: #5d6a7a;
}
QPushButton:pressed {
    background: #252b33;
}
QPushButton:disabled {
    color: #5e6876;
    background: #23292f;
    border-color: #2e3540;
}
QPushButton:focus {
    border: 2px solid #5ca8e0;
}
QPushButton[buttonRole="primary"] {
    background: #1e5a87;
    color: #ecf4fb;
    border-color: #154969;
}
QPushButton[buttonRole="primary"]:hover {
    background: #226a9b;
}
QPushButton[buttonRole="primary"]:disabled {
    background: #1a3a54;
    color: #6b8299;
    border-color: #143048;
}
QPushButton[buttonRole="success"] {
    background: #1e7a52;
    color: #ecf4fb;
    border-color: #145c3d;
}
QPushButton[buttonRole="success"]:hover {
    background: #24916a;
}
QPushButton[buttonRole="success"]:disabled {
    background: #1f4b38;
    color: #7ca090;
    border-color: #173c2b;
}
QPushButton[buttonRole="danger"] {
    background: #7a3a3a;
    color: #fff;
    border-color: #5c2a2a;
}
QPushButton[buttonRole="danger"]:hover {
    background: #8e4545;
}
QPushButton[buttonRole="danger"]:disabled {
    background: #4a2828;
    color: #a48383;
    border-color: #3a1f1f;
}
QPushButton[buttonRole="subtle"] {
    background: #262d38;
}

/* === Tables / trees / lists === */
QTableView, QTreeView, QListView {
    background: #222831;
    alternate-background-color: #262d38;
    color: #dbe1ea;
    border: 1px solid #323a45;
    gridline-color: #2a313a;
    selection-background-color: #1e4870;
    selection-color: #ecf4fb;
}
QTableView QWidget,
QTreeView QWidget,
QListView QWidget,
QHeaderView {
    background: #222831;
}
QTableView::item:selected, QTreeView::item:selected, QListView::item:selected {
    background: #1e4870;
    color: #ecf4fb;
}
QTableView::item:hover, QTreeView::item:hover, QListView::item:hover {
    background: #2d3543;
}

QHeaderView::section {
    background: #252c36;
    color: #a7b2c2;
    border-bottom: 1px solid #323a45;
}
QHeaderView::section:first {
    border-top-left-radius: 0px;
}
QHeaderView::section:last {
    border-top-right-radius: 0px;
}

QTableCornerButton::section {
    background: #252c36;
    border: none;
    border-right: 1px solid #323a45;
    border-bottom: 1px solid #323a45;
    border-top-left-radius: 0px;
}

QAbstractScrollArea::corner {
    background: #252c36;
    border: none;
}

/* === Menus === */
QMenuBar {
    background: #1a1f26;
    color: #dbe1ea;
    border-bottom: 1px solid #323a45;
}
QMenuBar::item:selected {
    background: #2d3543;
    color: #ecf4fb;
}
QMenu {
    background: #222831;
    border: 1px solid #323a45;
    color: #dbe1ea;
}
QMenu::item:selected {
    background: #1e4870;
    color: #ecf4fb;
}

/* === Status bar === */
QStatusBar {
    background: #222831;
    color: #a7b2c2;
    border-top: 1px solid #323a45;
}

/* === Scrollbars === */
QScrollBar:vertical, QScrollBar:horizontal {
    background: #1a1f26;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #3e4754;
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background: #5d6a7a;
}

/* === Progress === */
QProgressBar {
    background: #2a313a;
    border: 1px solid #3e4754;
    color: #dbe1ea;
}
QProgressBar::chunk {
    background: #5ca8e0;
}

/* === Checkbox === */
QCheckBox {
    color: #dbe1ea;
}
QCheckBox:disabled {
    color: #5e6876;
}
QCheckBox::indicator {
    background: #2a313a;
    border: 1.5px solid #3e4754;
}
QCheckBox::indicator:hover {
    border-color: #5ca8e0;
}
QCheckBox::indicator:checked {
    background: #5ca8e0;
    border-color: #5ca8e0;
}
QCheckBox::indicator:disabled {
    background: #23292f;
    border-color: #2e3540;
}

/* === Splitter handles === */
QSplitter::handle {
    background: #3e4754;
}

/* === SpinBox arrow backgrounds (keep the SVG arrows from base) === */
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background: #2a313a;
    border-color: #3e4754;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: #1e4870;
}

/* === ToolButton === */
QToolButton {
    background: #2a313a;
    color: #dbe1ea;
    border: 1px solid #3e4754;
}
QToolButton:hover {
    background: #313a46;
}

/* === QDateEdit dropdown button (was forgotten by the first dark pass) === */
QDateEdit::drop-down {
    background: #23292f;
    border-left: 1px solid #3e4754;
}
QDateEdit::drop-down:hover {
    background: #1e4870;
}
"""
