"""Read-only engineering summary of the active session."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from eit_app.ui.theme import apply_state_banner, apply_state_chip, set_hint_text, set_panel_role


class SessionSummaryPanel(QGroupBox):
    """Compact, read-only summary of the active session configuration."""

    _FIELDS = (
        ("link", "Link:"),
        ("power", "Power:"),
        ("identity", "Identity:"),
        ("protocol", "Protocol:"),
        ("transport", "Transport:"),
        ("frequency", "Frequency:"),
        ("stim", "Stim:"),
        ("gain", "Gain:"),
        ("record", "Record path:"),
        ("mode", "Acquisition:"),
    )
    _INDICATORS = ("link", "power", "record", "acq")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("4. Current Session Summary", parent)
        self._values: dict[str, QLabel] = {}
        self._indicator_values: dict[str, QLabel] = {}
        set_panel_role(self, "summary")
        self._build_ui()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 10, 8, 8)
        root_layout.setSpacing(8)

        hint = QLabel("Read-only summary of the verified session and next acquisition.")
        hint.setWordWrap(True)
        set_hint_text(hint)
        root_layout.addWidget(hint)

        self._state_badge = QLabel("LINK DOWN")
        self._state_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root_layout.addWidget(self._state_badge)

        self._state_detail = QLabel("Verify a device link to prepare the workstation.")
        self._state_detail.setWordWrap(True)
        self._state_detail.setProperty("uiSubtleValue", True)
        root_layout.addWidget(self._state_detail)

        indicator_row = QHBoxLayout()
        indicator_row.setSpacing(6)
        indicator_grid = QGridLayout()
        indicator_grid.setHorizontalSpacing(8)
        indicator_grid.setVerticalSpacing(6)

        for index, key in enumerate(self._INDICATORS):
            title = QLabel(key.upper())
            title.setProperty("uiHintText", True)
            title.setStyleSheet("font-size: 10px; font-weight: 700; letter-spacing: 0.5px;")
            value = QLabel("—")
            value.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value.setMinimumWidth(74)
            self._indicator_values[key] = value
            indicator_grid.addWidget(title, 0, index)
            indicator_grid.addWidget(value, 1, index)

        indicator_row.addLayout(indicator_grid)
        indicator_row.addStretch()
        root_layout.addLayout(indicator_row)

        self._next_action = QLabel("Next: Select a transport and click Connect & Verify.")
        self._next_action.setWordWrap(True)
        self._next_action.setStyleSheet(
            "padding: 8px 10px; "
            "border-left: 4px solid #1f5d8b; "
            "background: #edf4fb; "
            "border-radius: 8px; "
            "color: #243447;"
        )
        root_layout.addWidget(self._next_action)

        layout = QFormLayout()
        layout.setSpacing(8)
        root_layout.addLayout(layout)

        fixed_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        fixed_font.setPointSize(max(fixed_font.pointSize(), 10))

        for key, title in self._FIELDS:
            value = QLabel("—")
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            value.setFont(fixed_font)
            value.setStyleSheet(
                "padding: 4px 6px; "
                "border: 1px solid #d8dee9; "
                "border-radius: 4px; "
                "background: #f7f9fc; "
                "color: #243447;"
            )
            self._values[key] = value
            layout.addRow(title, value)

        self.set_status_banner(
            title="LINK DOWN",
            detail="Verify a device link to prepare the workstation.",
            next_action="Next: Select a transport and click Connect & Verify.",
            tone="idle",
        )
        self.set_indicator_states(
            {
                "link": ("Down", "idle"),
                "power": ("Unknown", "idle"),
                "record": ("Off", "idle"),
                "acq": ("Idle", "idle"),
            }
        )

    def set_status_banner(
        self,
        *,
        title: str,
        detail: str,
        next_action: str,
        tone: str = "idle",
    ) -> None:
        self._state_badge.setText(title)
        apply_state_banner(self._state_badge, tone=tone)
        self._state_detail.setText(detail)
        self._next_action.setText(next_action)

    def set_indicator_states(self, states: dict[str, tuple[str, str]]) -> None:
        for key in self._INDICATORS:
            text, tone = states.get(key, ("—", "idle"))
            self._indicator_values[key].setText(text)
            apply_state_chip(self._indicator_values[key], tone=tone, emphasized=True)

    def set_summary(self, summary: dict[str, str]) -> None:
        for key, _title in self._FIELDS:
            self._values[key].setText(summary.get(key, "—"))
