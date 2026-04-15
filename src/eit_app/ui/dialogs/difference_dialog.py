"""Dialog for selecting reference and target frames for difference imaging."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QLabel, QVBoxLayout

from eit_app.i18n import t, translator
from eit_app.ui.auto_close_combo_box import AutoCloseComboBox

class DifferenceDialog(QDialog):
    """Modal dialog for configuring difference reconstruction.

    The user selects reference frame, target frame, difference mode,
    and orientation before confirming.

    Signals:
        reconstruction_requested: Emitted with config dict containing
            ref_index, tgt_index, mode, orientation, use_part.
    """

    reconstruction_requested = Signal(dict)

    def __init__(
        self,
        frame_entries: list[dict],
        parent=None,
        *,
        default_ref_index: int = 0,
        default_tgt_index: int | None = None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumWidth(400)

        self._frame_entries = frame_entries
        self._default_ref_index = default_ref_index
        self._default_tgt_index = default_tgt_index
        self._build_ui()
        translator().language_changed.connect(self._retranslate)
        self._retranslate()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Frame selection
        self._frame_group = QGroupBox("")  # retranslated
        frame_layout = QFormLayout(self._frame_group)

        self._ref_combo = AutoCloseComboBox()
        self._tgt_combo = AutoCloseComboBox()
        for entry in self._frame_entries:
            label = f"Frame {entry.get('frame_index', '?')} - {entry.get('file_path', '')}"
            self._ref_combo.addItem(label)
            self._tgt_combo.addItem(label)

        if self._frame_entries:
            ref_index = min(max(self._default_ref_index, 0), len(self._frame_entries) - 1)
            self._ref_combo.setCurrentIndex(ref_index)

        if len(self._frame_entries) > 1:
            tgt_index = self._default_tgt_index
            if tgt_index is None:
                tgt_index = 1 if self._ref_combo.currentIndex() == 0 else 0
            tgt_index = min(max(tgt_index, 0), len(self._frame_entries) - 1)
            self._tgt_combo.setCurrentIndex(tgt_index)

        self._lbl_ref = QLabel("")
        self._lbl_tgt = QLabel("")
        frame_layout.addRow(self._lbl_ref, self._ref_combo)
        frame_layout.addRow(self._lbl_tgt, self._tgt_combo)
        layout.addWidget(self._frame_group)

        # Difference settings
        self._settings_group = QGroupBox("")  # retranslated
        settings_layout = QFormLayout(self._settings_group)

        # ComboBox values are invariant algorithm tags — not localised.
        self._mode_combo = AutoCloseComboBox()
        self._mode_combo.addItems(["raw", "normalized"])
        self._lbl_mode = QLabel("")
        settings_layout.addRow(self._lbl_mode, self._mode_combo)

        self._orient_combo = AutoCloseComboBox()
        self._orient_combo.addItems(["target_minus_reference", "reference_minus_target"])
        self._lbl_orient = QLabel("")
        settings_layout.addRow(self._lbl_orient, self._orient_combo)

        self._part_combo = AutoCloseComboBox()
        self._part_combo.addItems(["real", "imag", "mag"])
        self._lbl_part = QLabel("")
        settings_layout.addRow(self._lbl_part, self._part_combo)

        layout.addWidget(self._settings_group)

        # Info
        self._info_label = QLabel("")
        layout.addWidget(self._info_label)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        ref_idx = self._ref_combo.currentIndex()
        tgt_idx = self._tgt_combo.currentIndex()

        if ref_idx == tgt_idx:
            self._info_label.setText(t("dlg.difference.warn_same_frame"))
            self._info_label.setStyleSheet("color: red;")
            return

        config = {
            "ref_index": ref_idx,
            "tgt_index": tgt_idx,
            "ref_entry": self._frame_entries[ref_idx],
            "tgt_entry": self._frame_entries[tgt_idx],
            "mode": self._mode_combo.currentText(),
            "orientation": self._orient_combo.currentText(),
            "use_part": self._part_combo.currentText(),
        }
        self.reconstruction_requested.emit(config)
        self.accept()

    # ── i18n ──

    def _retranslate(self) -> None:
        self.setWindowTitle(t("dlg.difference.title"))
        self._frame_group.setTitle(t("dlg.difference.frame_group"))
        self._lbl_ref.setText(t("dlg.difference.ref_label"))
        self._lbl_tgt.setText(t("dlg.difference.tgt_label"))
        self._settings_group.setTitle(t("dlg.difference.settings_group"))
        self._lbl_mode.setText(t("dlg.difference.mode_label"))
        self._lbl_orient.setText(t("dlg.difference.orient_label"))
        self._lbl_part.setText(t("dlg.difference.part_label"))
