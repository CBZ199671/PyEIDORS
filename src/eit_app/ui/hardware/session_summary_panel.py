"""Read-only engineering summary of the active session."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QGridLayout, QGroupBox, QLabel, QSizePolicy, QVBoxLayout, QWidget

from eit_app.ui.theme import apply_state_banner, apply_state_chip, set_panel_role


class SessionSummaryPanel(QGroupBox):
    """Compact, read-only summary of the active session configuration."""

    _FIELDS = (
        ("identity", "Identity:"),
        ("transport", "Transport:"),
        ("layout", "Layout:"),
        ("drive", "Drive:"),
        ("record", "Record path:"),
        ("plan", "Plan:"),
    )
    _INDICATORS = ("link", "power", "record", "acq")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("", parent)
        self._values: dict[str, QLabel] = {}
        self._indicator_values: dict[str, QLabel] = {}
        self.setProperty("summaryHeaderless", True)
        set_panel_role(self, "summary")
        self._build_ui()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        self._state_badge = QLabel("LINK DOWN")
        self._state_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root_layout.addWidget(self._state_badge)

        indicator_grid = QGridLayout()
        indicator_grid.setContentsMargins(0, 0, 0, 0)
        indicator_grid.setHorizontalSpacing(8)
        indicator_grid.setVerticalSpacing(6)

        for index, key in enumerate(self._INDICATORS):
            grid_row = (index // 2) * 2
            grid_col = index % 2
            title = QLabel(key.upper())
            title.setStyleSheet("font-size: 10px; font-weight: 700; letter-spacing: 0.5px; color: #5b6573;")
            value = QLabel("—")
            value.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value.setMinimumWidth(58)
            self._indicator_values[key] = value
            indicator_grid.addWidget(title, grid_row, grid_col)
            indicator_grid.addWidget(value, grid_row + 1, grid_col)

        root_layout.addLayout(indicator_grid)

        self._next_action = QLabel("Select a transport and click Connect & Verify.")
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

        for index, (key, title) in enumerate(self._FIELDS):
            title_label = QLabel(title)
            title_label.setStyleSheet(
                "color: #4d5f75; "
                "font-weight: 700; "
                "padding-top: 2px;"
            )
            value = QLabel("—")
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
            field_grid.addWidget(title_label, index, 0)
            field_grid.addWidget(value, index, 1)

        root_layout.addStretch(1)

        self.set_status_banner(
            title="LINK DOWN",
            detail="Verify a device link to prepare the workstation.",
            next_action="Select a transport and click Connect & Verify.",
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
        self._state_badge.setToolTip(detail)
        self.setToolTip(detail)
        apply_state_banner(self._state_badge, tone=tone)
        self._next_action.setText(next_action)

    def set_indicator_states(self, states: dict[str, tuple[str, str]]) -> None:
        for key in self._INDICATORS:
            text, tone = states.get(key, ("—", "idle"))
            self._indicator_values[key].setText(text)
            apply_state_chip(self._indicator_values[key], tone=tone, emphasized=True)

    def set_summary(self, summary: dict[str, str]) -> None:
        for key, _title in self._FIELDS:
            self._values[key].setText(summary.get(key, "—"))
