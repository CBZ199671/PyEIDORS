"""Read-only summary and progress panel for dataset generation."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel, QProgressBar, QVBoxLayout, QWidget

from eit_app.i18n import t, translator
from eit_app.ui.theme import (
    apply_state_chip,
    field_value_stylesheet,
    set_hint_text,
    set_panel_role,
    set_subtle_value,
    subscribe_theme_mode,
)


class DatasetSummaryPanel(QGroupBox):
    """Compact status and configuration summary for batch generation."""

    # Field identifier -> translation key for the row title.
    _FIELDS = (
        ("output_dir", "dataset.summary.field.output"),
        ("samples", "dataset.summary.field.samples"),
        ("shapes", "dataset.summary.field.shapes"),
        ("mesh", "dataset.summary.field.mesh"),
        ("electrodes", "dataset.summary.field.electrodes"),
        ("status", "dataset.summary.field.status"),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        # Title assigned by _retranslate() so it follows the UI language.
        super().__init__("", parent)
        self._values: dict[str, QLabel] = {}
        self._field_titles: dict[str, QLabel] = {}
        self._progress_cache = (0, 0)
        self._chip_tone = "idle"
        self._chip_text_owner_set = False  # True once an owner pushed a state
        set_panel_role(self, "summary")
        self._build_ui()
        translator().language_changed.connect(self._retranslate)
        self._retranslate()
        # Re-apply per-card stylesheets when the user toggles dark mode.
        subscribe_theme_mode(self._on_theme_mode_changed)

    def _on_theme_mode_changed(self, _mode: str) -> None:
        """Re-paint the chip + value boxes whose stylesheets are
        composed by setStyleSheet rather than the global QSS."""
        for value in self._values.values():
            value.setStyleSheet(field_value_stylesheet())
        # State chip uses tone_palette which is already dark-aware;
        # re-apply with the cached tone so it picks up the new
        # palette colors.
        apply_state_chip(self._state_chip, tone=self._chip_tone, emphasized=True)
        # _retranslate will refresh the chip text under the default
        # idle path; explicit owner-set states stay as the owner left
        # them.
        self._retranslate()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 12, 10, 8)
        layout.setSpacing(8)

        self._hint = QLabel("")  # retranslated
        self._hint.setWordWrap(True)
        set_hint_text(self._hint)
        layout.addWidget(self._hint)

        self._state_chip = QLabel("")
        apply_state_chip(self._state_chip, tone="idle", emphasized=True)
        layout.addWidget(self._state_chip)

        self._progress_label = QLabel("")  # retranslated in _retranslate
        set_subtle_value(self._progress_label)
        layout.addWidget(self._progress_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        form = QFormLayout()
        form.setSpacing(8)
        layout.addLayout(form)

        for key, _title_key in self._FIELDS:
            title_label = QLabel("")  # retranslated
            value = QLabel("\u2014")
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            # Style comes from theme.field_value_stylesheet() so the
            # box surface follows dark mode.
            value.setStyleSheet(field_value_stylesheet())
            self._values[key] = value
            self._field_titles[key] = title_label
            form.addRow(title_label, value)

    def set_status(self, text: str, *, tone: str) -> None:
        """Set the status chip from a pre-translated string."""
        self._chip_text_owner_set = True
        self._chip_tone = tone
        self._state_chip.setText(text)
        apply_state_chip(self._state_chip, tone=tone, emphasized=True)
        self._values["status"].setText(text)

    def set_progress(self, current: int, total: int) -> None:
        self._progress_cache = (current, total)
        cap = max(total, 1)
        self._progress_bar.setMaximum(cap)
        self._progress_bar.setValue(min(current, cap))
        self._progress_label.setText(
            t("dataset.summary.progress", current=current, total=total)
        )

    def set_summary(self, summary: dict[str, str]) -> None:
        for key, _title_key in self._FIELDS:
            if key == "status":
                continue
            self._values[key].setText(summary.get(key, "\u2014"))

    # ── i18n ──

    def _retranslate(self) -> None:
        self.setTitle(t("dataset.summary.title"))
        self._hint.setText(t("dataset.summary.hint"))
        for key, title_key in self._FIELDS:
            self._field_titles[key].setText(t(title_key))
        # Refresh the dynamic progress line with the new locale.
        current, total = self._progress_cache
        self._progress_label.setText(
            t("dataset.summary.progress", current=current, total=total)
        )
        # If the owner hasn't pushed a status yet, show the default idle one.
        if not self._chip_text_owner_set:
            default_text = t("dataset.summary.state.idle")
            self._state_chip.setText(default_text)
            self._values["status"].setText(default_text)
            apply_state_chip(self._state_chip, tone=self._chip_tone, emphasized=True)
