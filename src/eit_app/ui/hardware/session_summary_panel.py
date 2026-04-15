"""Read-only engineering summary of the active session."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QGridLayout, QGroupBox, QLabel, QSizePolicy, QVBoxLayout, QWidget

from eit_app.i18n import t, translator
from eit_app.ui.theme import apply_state_banner, apply_state_chip, set_panel_role


class SessionSummaryPanel(QGroupBox):
    """Compact, read-only summary of the active session configuration."""

    # Field identifier → translation key for the row title.
    _FIELDS = (
        ("identity", "hw.summary.field.identity"),
        ("transport", "hw.summary.field.transport"),
        ("layout", "hw.summary.field.layout"),
        ("drive", "hw.summary.field.drive"),
        ("record", "hw.summary.field.record"),
        ("plan", "hw.summary.field.plan"),
    )
    # Indicator identifier → translation key for the uppercase header.
    _INDICATORS = (
        ("link", "hw.summary.indicator.link"),
        ("power", "hw.summary.indicator.power"),
        ("record", "hw.summary.indicator.record"),
        ("acq", "hw.summary.indicator.acq"),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("", parent)
        self._values: dict[str, QLabel] = {}
        self._field_titles: dict[str, QLabel] = {}
        self._indicator_values: dict[str, QLabel] = {}
        self._indicator_titles: dict[str, QLabel] = {}
        self.setProperty("summaryHeaderless", True)
        set_panel_role(self, "summary")
        self._build_ui()
        translator().language_changed.connect(self._retranslate)
        self._retranslate()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        # State banner text is populated by _retranslate() / set_status_banner
        self._state_badge = QLabel("")
        self._state_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root_layout.addWidget(self._state_badge)

        indicator_grid = QGridLayout()
        indicator_grid.setContentsMargins(0, 0, 0, 0)
        indicator_grid.setHorizontalSpacing(8)
        indicator_grid.setVerticalSpacing(6)

        for index, (key, _title_key) in enumerate(self._INDICATORS):
            grid_row = (index // 2) * 2
            grid_col = index % 2
            title = QLabel("")  # retranslated below
            title.setStyleSheet(
                "font-size: 10px; font-weight: 700; letter-spacing: 0.5px; color: #5b6573;"
            )
            value = QLabel("\u2014")
            value.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value.setMinimumWidth(58)
            self._indicator_values[key] = value
            self._indicator_titles[key] = title
            indicator_grid.addWidget(title, grid_row, grid_col)
            indicator_grid.addWidget(value, grid_row + 1, grid_col)

        root_layout.addLayout(indicator_grid)

        self._next_action = QLabel("")  # retranslated below
        self._next_action.setWordWrap(True)
        self._next_action.setStyleSheet(
            "padding: 8px 10px; "
            "border-left: 4px solid #1f5d8b; "
            "background: #edf4fb; "
            "border-radius: 8px; "
            "color: #243447;"
        )
        root_layout.addWidget(self._next_action)

        field_grid = QGridLayout()
        field_grid.setContentsMargins(0, 0, 0, 0)
        field_grid.setHorizontalSpacing(10)
        field_grid.setVerticalSpacing(8)
        field_grid.setColumnStretch(1, 1)
        root_layout.addLayout(field_grid)

        fixed_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        fixed_font.setPointSize(max(fixed_font.pointSize(), 10))

        for index, (key, _title_key) in enumerate(self._FIELDS):
            title_label = QLabel("")  # retranslated below
            title_label.setStyleSheet(
                "color: #4d5f75; "
                "font-weight: 700; "
                "padding-top: 2px;"
            )
            value = QLabel("\u2014")
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            value.setFont(fixed_font)
            value.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            value.setStyleSheet(
                "padding: 4px 6px; "
                "border: 1px solid #d8dee9; "
                "border-radius: 4px; "
                "background: #f7f9fc; "
                "color: #243447;"
            )
            self._values[key] = value
            self._field_titles[key] = title_label
            field_grid.addWidget(title_label, index, 0)
            field_grid.addWidget(value, index, 1)

        root_layout.addStretch(1)

        # Default idle banner / indicators — re-rendered on language change.
        self._banner_tone = "idle"
        self._banner_active = False  # True once a caller overrides the default
        self._default_indicators: dict[str, tuple[str, str]] = {
            "link": ("hw.summary.state.down", "idle"),
            "power": ("hw.summary.state.unknown", "idle"),
            "record": ("hw.summary.state.off", "idle"),
            "acq": ("hw.summary.state.idle", "idle"),
        }
        self._indicator_cache: dict[str, tuple[str, str]] = {}

    def set_status_banner(
        self,
        *,
        title: str,
        detail: str,
        next_action: str,
        tone: str = "idle",
    ) -> None:
        """Set the banner from pre-translated strings (e.g. ``t('...')``)."""
        self._banner_active = True
        self._banner_tone = tone
        self._state_badge.setText(title)
        self._state_badge.setToolTip(detail)
        self.setToolTip(detail)
        apply_state_banner(self._state_badge, tone=tone)
        self._next_action.setText(next_action)

    def set_indicator_states(self, states: dict[str, tuple[str, str]]) -> None:
        """Update indicator chips from pre-translated (text, tone) pairs."""
        self._indicator_cache = dict(states)
        for key, _title_key in self._INDICATORS:
            text, tone = states.get(key, ("\u2014", "idle"))
            self._indicator_values[key].setText(text)
            apply_state_chip(self._indicator_values[key], tone=tone, emphasized=True)

    def set_summary(self, summary: dict[str, str]) -> None:
        for key, _title_key in self._FIELDS:
            self._values[key].setText(summary.get(key, "\u2014"))

    # ── i18n ──

    def _retranslate(self) -> None:
        """Refresh all owner-provided strings whose source is the i18n dict."""
        # Field title labels (left column)
        for key, title_key in self._FIELDS:
            self._field_titles[key].setText(t(title_key))
        # Indicator column titles (upper-case header row)
        for key, title_key in self._INDICATORS:
            self._indicator_titles[key].setText(t(title_key))

        # Default banner applies only while the owner hasn't overridden it.
        # Once a concrete banner has been pushed via set_status_banner, the
        # owner (main_window) is responsible for refreshing it on language
        # change — this panel has no way to know the owner's current state.
        if not self._banner_active:
            self.set_status_banner(
                title=t("hw.summary.banner.link_down.title"),
                detail=t("hw.summary.banner.link_down.detail"),
                next_action=t("hw.summary.banner.link_down.action"),
                tone="idle",
            )
            self._banner_active = False  # keep "not yet overridden" true
            default_states = {
                key: (t(text_key), tone)
                for key, (text_key, tone) in self._default_indicators.items()
            }
            self.set_indicator_states(default_states)
