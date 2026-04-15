"""Read-only summary and progress panel for dataset generation."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel, QProgressBar, QVBoxLayout, QWidget

from eit_app.ui.theme import apply_state_chip, set_hint_text, set_panel_role, set_subtle_value


class DatasetSummaryPanel(QGroupBox):
    """Compact status and configuration summary for batch generation."""

    _FIELDS = (
        ("output_dir", "Output:"),
        ("samples", "Samples:"),
        ("shapes", "Shapes:"),
        ("mesh", "Mesh:"),
        ("electrodes", "Electrodes:"),
        ("status", "Status:"),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Generation Summary", parent)
        self._values: dict[str, QLabel] = {}
        set_panel_role(self, "summary")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 12, 10, 8)
        layout.setSpacing(8)

        hint = QLabel(
            "Review the active batch configuration here before launching the generator."
        )
        hint.setWordWrap(True)
        set_hint_text(hint)
        layout.addWidget(hint)

        self._state_chip = QLabel("Idle")
        apply_state_chip(self._state_chip, tone="idle", emphasized=True)
        layout.addWidget(self._state_chip)

        self._progress_label = QLabel("Progress: 0 / 0")
        set_subtle_value(self._progress_label)
        layout.addWidget(self._progress_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        form = QFormLayout()
        form.setSpacing(8)
        layout.addLayout(form)

        for key, title in self._FIELDS:
            value = QLabel("—")
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            value.setStyleSheet(
                "padding: 4px 6px; "
                "border: 1px solid #d8dee9; "
                "border-radius: 4px; "
                "background: #f7f9fc; "
                "color: #243447;"
            )
            self._values[key] = value
            form.addRow(title, value)

        self.set_status("Idle", tone="idle")

    def set_status(self, text: str, *, tone: str) -> None:
        self._state_chip.setText(text)
        apply_state_chip(self._state_chip, tone=tone, emphasized=True)
        self._values["status"].setText(text)

    def set_progress(self, current: int, total: int) -> None:
        cap = max(total, 1)
        self._progress_bar.setMaximum(cap)
        self._progress_bar.setValue(min(current, cap))
        self._progress_label.setText(f"Progress: {current} / {total}")

    def set_summary(self, summary: dict[str, str]) -> None:
        for key, _title in self._FIELDS:
            if key == "status":
                continue
            self._values[key].setText(summary.get(key, "—"))
