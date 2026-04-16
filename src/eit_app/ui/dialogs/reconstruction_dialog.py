"""Multi-algorithm reconstruction dialog.

Launched from the Database tab after the user picks frames to reconstruct.
Supports:
- Gauss-Newton difference (single-step) — default
- Gauss-Newton absolute
- Sparse Bayesian difference
- Sparse Bayesian absolute

Optionally saves the conductivity image (PNG) and boundary voltage fit
plot to a chosen output folder.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from eit_app.i18n import t, translator
from eit_app.ui.auto_close_combo_box import AutoCloseComboBox
from eit_app.ui.theme import card_palette, set_button_role, set_hint_text

log = logging.getLogger(__name__)


def _default_results_dir() -> Path:
    """Return the default output directory: <app cwd>/results, created if missing."""
    base = Path.cwd() / "results"
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return base


# Algorithm options: (display_label, method_key, requires_reference)
_ALGORITHMS = [
    ("Gauss-Newton · Difference (single-step)", "gn-difference", True),
    ("Gauss-Newton · Absolute", "gn-absolute", False),
    ("Sparse Bayesian · Difference", "sparse-bayes-difference", True),
    ("Sparse Bayesian · Absolute", "sparse-bayes-absolute", False),
]


class ReconstructionDialog(QDialog):
    """Collects algorithm choice + parameters, then emits the final config.

    Signal:
        run_requested(dict) — emitted on Run with config:
            {
                "method": str,
                "reference_entry": dict | None,
                "target_entry": dict,
                "regularization_alpha": float,
                "max_iterations": int,
                "use_part": str,
                "output_dir": str | None,
                "save_recon_image": bool,
                "save_voltage_fit": bool,
            }
    """

    run_requested = Signal(dict)

    def __init__(
        self,
        *,
        reference_entry: dict | None,
        target_entry: dict | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumWidth(720)
        self.resize(780, 700)
        self._reference_entry = reference_entry
        self._target_entry = target_entry
        self._build_ui()
        self._update_reference_visibility()
        translator().language_changed.connect(self._retranslate)
        self._retranslate()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(14)

        # Header card
        header = QWidget()
        header.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            " stop:0 #1f5d8b, stop:1 #2a6fa0);"
            " border-radius: 10px; padding: 14px 18px;"
        )
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 16, 12)
        header_layout.setSpacing(4)

        self._title_label = QLabel("")
        self._title_label.setStyleSheet(
            "background: transparent; color: #ffffff;"
            " font-size: 17px; font-weight: 700; border: none;"
        )
        header_layout.addWidget(self._title_label)

        self._subtitle_label = QLabel("")
        self._subtitle_label.setWordWrap(True)
        self._subtitle_label.setStyleSheet(
            "background: transparent; color: #dbe8f4;"
            " font-size: 12px; border: none;"
        )
        header_layout.addWidget(self._subtitle_label)
        root.addWidget(header)

        # Frame selection summary
        root.addWidget(self._build_frames_section())

        # Algorithm section
        root.addWidget(self._build_algorithm_section())

        # Output section
        root.addWidget(self._build_output_section())

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 4, 0, 0)
        btn_row.setSpacing(8)
        btn_row.addStretch()

        self._cancel_btn = QPushButton("")
        set_button_role(self._cancel_btn, "subtle")
        self._cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._cancel_btn)

        self._run_btn = QPushButton("")
        set_button_role(self._run_btn, "primary")
        self._run_btn.setMinimumWidth(160)
        self._run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self._run_btn)

        root.addLayout(btn_row)

    def _build_frames_section(self) -> QWidget:
        self._frames_box = QGroupBox("")  # retranslated
        layout = QFormLayout(self._frames_box)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 20, 14, 14)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        # Pull chip surface colors from the palette so dark mode applies.
        # The dialog is short-lived and re-instantiated on each open
        # (single-frame reconstruct flow), so we don't bother
        # subscribing to theme_mode_changed here — colors are correct
        # for whichever mode was active when the dialog spawned.
        p = card_palette()
        chip_style = (
            f"background: {p['info_bg']}; border: 1px solid {p['info_border']};"
            " border-radius: 6px; padding: 7px 12px;"
            f" color: {p['value_text']}; font-family: monospace; font-size: 12px;"
            " min-width: 360px;"
        )

        self._ref_label = QLabel(self._format_entry(self._reference_entry))
        self._ref_label.setWordWrap(True)
        self._ref_label.setStyleSheet(chip_style)

        self._tgt_label = QLabel(self._format_entry(self._target_entry))
        self._tgt_label.setWordWrap(True)
        self._tgt_label.setStyleSheet(chip_style)

        self._ref_row_label = QLabel("")
        self._tgt_row_label = QLabel("")
        layout.addRow(self._ref_row_label, self._ref_label)
        layout.addRow(self._tgt_row_label, self._tgt_label)
        return self._frames_box

    def _build_algorithm_section(self) -> QWidget:
        self._algo_box = QGroupBox("")  # retranslated
        layout = QFormLayout(self._algo_box)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 20, 14, 14)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._algo_combo = AutoCloseComboBox()
        for label, _method, _needs_ref in _ALGORITHMS:
            self._algo_combo.addItem(label)
        self._algo_combo.currentIndexChanged.connect(self._update_reference_visibility)
        self._lbl_method = QLabel("")
        layout.addRow(self._lbl_method, self._algo_combo)

        self._use_part_combo = AutoCloseComboBox()
        self._use_part_combo.addItems(["real", "imag", "mag"])
        self._lbl_part = QLabel("")
        layout.addRow(self._lbl_part, self._use_part_combo)

        self._alpha_spin = QDoubleSpinBox()
        self._alpha_spin.setRange(0.0001, 1000.0)
        self._alpha_spin.setValue(1.0)
        self._alpha_spin.setDecimals(4)
        self._alpha_spin.setSingleStep(0.1)
        self._lbl_alpha = QLabel("")
        layout.addRow(self._lbl_alpha, self._alpha_spin)

        self._iter_spin = QSpinBox()
        self._iter_spin.setRange(1, 200)
        self._iter_spin.setValue(10)
        self._lbl_iter = QLabel("")
        layout.addRow(self._lbl_iter, self._iter_spin)

        return self._algo_box

    def _build_output_section(self) -> QWidget:
        self._output_box = QGroupBox("")  # retranslated
        layout = QVBoxLayout(self._output_box)
        layout.setContentsMargins(14, 20, 14, 14)
        layout.setSpacing(10)

        dir_row = QHBoxLayout()
        dir_row.setSpacing(6)
        self._dir_edit = QLineEdit()
        self._dir_edit.setText(str(_default_results_dir()))
        self._dir_browse_btn = QPushButton("")
        set_button_role(self._dir_browse_btn, "subtle")
        self._dir_browse_btn.setMinimumWidth(90)
        self._dir_browse_btn.clicked.connect(self._on_browse_output_dir)
        dir_row.addWidget(self._dir_edit, 1)
        dir_row.addWidget(self._dir_browse_btn)

        self._lbl_output_folder = QLabel("")
        layout.addWidget(self._lbl_output_folder)
        layout.addLayout(dir_row)

        self._save_recon_check = QCheckBox("")
        self._save_recon_check.setChecked(True)
        layout.addWidget(self._save_recon_check)

        self._save_voltage_check = QCheckBox("")
        self._save_voltage_check.setChecked(True)
        layout.addWidget(self._save_voltage_check)

        return self._output_box

    # ---- Event handlers ----

    def _update_reference_visibility(self, *args) -> None:
        needs_ref = _ALGORITHMS[self._algo_combo.currentIndex()][2]
        self._ref_label.setEnabled(needs_ref)
        self._ref_row_label.setEnabled(needs_ref)
        if not needs_ref:
            self._ref_label.setToolTip(t("dlg.reconstruction.absolute_no_ref_tip"))
        else:
            self._ref_label.setToolTip("")

        # Validation: refresh run button enabled state
        self._update_run_enabled()

    def _update_run_enabled(self) -> None:
        needs_ref = _ALGORITHMS[self._algo_combo.currentIndex()][2]
        has_tgt = self._target_entry is not None
        has_ref = self._reference_entry is not None
        enabled = has_tgt and (not needs_ref or has_ref)
        self._run_btn.setEnabled(enabled)

    def _on_browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, t("hw.acquisition.file_dialog_title"),
            self._dir_edit.text() or str(Path.home()),
        )
        if path:
            self._dir_edit.setText(path)

    def _on_run(self) -> None:
        idx = self._algo_combo.currentIndex()
        label, method, needs_ref = _ALGORITHMS[idx]
        if needs_ref and self._reference_entry is None:
            return
        if self._target_entry is None:
            return

        config: dict[str, Any] = {
            "method": method,
            "method_label": label,
            "reference_entry": self._reference_entry if needs_ref else None,
            "target_entry": self._target_entry,
            "regularization_alpha": self._alpha_spin.value(),
            "max_iterations": self._iter_spin.value(),
            "use_part": self._use_part_combo.currentText(),
            "output_dir": self._dir_edit.text().strip() or None,
            "save_recon_image": self._save_recon_check.isChecked(),
            "save_voltage_fit": self._save_voltage_check.isChecked(),
        }
        self.run_requested.emit(config)
        self.accept()

    # ---- Helpers ----

    @staticmethod
    def _format_entry(entry: dict | None) -> str:
        if entry is None:
            return t("dlg.reconstruction.not_selected")
        idx = entry.get("frame_index", "?")
        path = entry.get("csv_path") or entry.get("file_path", "")
        name = Path(path).name if path else ""
        return f"#{idx}  \u00b7  {name}"

    # ── i18n ──

    def _retranslate(self) -> None:
        self.setWindowTitle(t("dlg.reconstruction.title"))
        self._title_label.setText(t("dlg.reconstruction.heading"))
        self._subtitle_label.setText(t("dlg.reconstruction.subtitle"))
        self._cancel_btn.setText(t("dlg.reconstruction.cancel_button"))
        self._run_btn.setText(t("dlg.reconstruction.run_button"))
        self._frames_box.setTitle(t("dlg.reconstruction.selected_frames_group"))
        self._ref_row_label.setText(t("dlg.reconstruction.ref_label"))
        self._tgt_row_label.setText(t("dlg.reconstruction.tgt_label"))
        self._algo_box.setTitle(t("dlg.reconstruction.algo_params_group"))
        self._lbl_method.setText(t("dlg.reconstruction.method_label"))
        self._lbl_part.setText(t("dlg.reconstruction.part_label"))
        self._lbl_alpha.setText(t("dlg.reconstruction.alpha_label"))
        self._lbl_iter.setText(t("dlg.reconstruction.iter_label"))
        self._output_box.setTitle(t("dlg.reconstruction.output_group"))
        self._dir_edit.setPlaceholderText(t("dlg.reconstruction.output_placeholder"))
        self._dir_browse_btn.setText(t("dlg.reconstruction.browse_button"))
        self._lbl_output_folder.setText(t("dlg.reconstruction.output_folder_label"))
        self._save_recon_check.setText(t("dlg.reconstruction.save_image_check"))
        self._save_voltage_check.setText(t("dlg.reconstruction.save_voltage_check"))
        # Re-render frame chips (they use "<not selected>" placeholder)
        self._ref_label.setText(self._format_entry(self._reference_entry))
        self._tgt_label.setText(self._format_entry(self._target_entry))
