"""Batch reconstruction dialog.

Collects input folder, output folder, algorithm, optional reference
frame, and save options, then delegates to BatchReconstructionController.
Shows live progress via a QProgressBar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from eit_app.ui.auto_close_combo_box import AutoCloseComboBox
from eit_app.ui.theme import set_button_role, set_hint_text


# (display, method_key, requires_reference)
_ALGORITHMS = [
    ("Gauss-Newton · Difference (single-step)", "gn-difference", True),
    ("Gauss-Newton · Absolute", "gn-absolute", False),
    ("Sparse Bayesian · Difference", "sparse-bayes-difference", True),
    ("Sparse Bayesian · Absolute", "sparse-bayes-absolute", False),
]


class BatchReconstructionDialog(QDialog):
    """Dialog that builds a batch reconstruction job and delegates execution.

    Signal:
        start_requested(dict) — emitted when user clicks Run with the full
        configuration. Parent wires this to BatchReconstructionController.
    """

    start_requested = Signal(dict)
    cancel_requested = Signal()

    def __init__(
        self,
        *,
        default_input: Path | None = None,
        default_output: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch Reconstruct")
        self.setMinimumWidth(620)
        self._default_input = default_input
        self._default_output = default_output
        self._is_running = False
        self._build_ui()
        self._connect_signals()
        self._update_reference_requirement()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(12)

        title = QLabel("Batch reconstruction")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #1f3b5b;")
        root.addWidget(title)

        hint = QLabel(
            "Reconstruct every frame CSV in the input folder. For difference "
            "methods, the reference frame is applied to all targets (and "
            "excluded from targets if it lives in the same folder)."
        )
        hint.setWordWrap(True)
        set_hint_text(hint)
        root.addWidget(hint)

        root.addWidget(self._build_folders_section())
        root.addWidget(self._build_algorithm_section())
        root.addWidget(self._build_output_section())
        root.addWidget(self._build_progress_section())

        # Buttons
        self._button_box = QDialogButtonBox(parent=self)
        self._run_btn = QPushButton("Run Batch")
        set_button_role(self._run_btn, "primary")
        self._cancel_btn = QPushButton("Cancel")
        set_button_role(self._cancel_btn, "danger")
        self._cancel_btn.setVisible(False)
        self._close_btn = QPushButton("Close")
        set_button_role(self._close_btn, "subtle")
        self._button_box.addButton(self._run_btn, QDialogButtonBox.ButtonRole.AcceptRole)
        self._button_box.addButton(self._cancel_btn, QDialogButtonBox.ButtonRole.DestructiveRole)
        self._button_box.addButton(self._close_btn, QDialogButtonBox.ButtonRole.RejectRole)
        root.addWidget(self._button_box)

    def _build_folders_section(self) -> QWidget:
        box = QGroupBox("Folders")
        layout = QFormLayout(box)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("Folder containing frame CSV files")
        if self._default_input:
            self._input_edit.setText(str(self._default_input))
        self._input_browse_btn = QPushButton("Browse…")
        set_button_role(self._input_browse_btn, "subtle")
        in_row = QHBoxLayout()
        in_row.setContentsMargins(0, 0, 0, 0)
        in_row.setSpacing(6)
        in_row.addWidget(self._input_edit, 1)
        in_row.addWidget(self._input_browse_btn)
        in_w = QWidget()
        in_w.setLayout(in_row)
        layout.addRow("Input folder:", in_w)

        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Folder to write reconstruction images")
        if self._default_output:
            self._output_edit.setText(str(self._default_output))
        self._output_browse_btn = QPushButton("Browse…")
        set_button_role(self._output_browse_btn, "subtle")
        out_row = QHBoxLayout()
        out_row.setContentsMargins(0, 0, 0, 0)
        out_row.setSpacing(6)
        out_row.addWidget(self._output_edit, 1)
        out_row.addWidget(self._output_browse_btn)
        out_w = QWidget()
        out_w.setLayout(out_row)
        layout.addRow("Output folder:", out_w)

        return box

    def _build_algorithm_section(self) -> QWidget:
        box = QGroupBox("Algorithm")
        layout = QFormLayout(box)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._algo_combo = AutoCloseComboBox()
        for label, _, _ in _ALGORITHMS:
            self._algo_combo.addItem(label)
        layout.addRow("Method:", self._algo_combo)

        self._use_part_combo = AutoCloseComboBox()
        self._use_part_combo.addItems(["real", "imag", "mag"])
        layout.addRow("Use part:", self._use_part_combo)

        self._alpha_spin = QDoubleSpinBox()
        self._alpha_spin.setRange(0.0001, 1000.0)
        self._alpha_spin.setValue(1.0)
        self._alpha_spin.setDecimals(4)
        self._alpha_spin.setSingleStep(0.1)
        layout.addRow("Regularization α:", self._alpha_spin)

        self._iter_spin = QSpinBox()
        self._iter_spin.setRange(1, 200)
        self._iter_spin.setValue(10)
        layout.addRow("Max iterations:", self._iter_spin)

        # Reference frame (only for difference methods)
        self._ref_edit = QLineEdit()
        self._ref_edit.setPlaceholderText(
            "CSV file to use as reference (required for difference methods)"
        )
        self._ref_browse_btn = QPushButton("Browse…")
        set_button_role(self._ref_browse_btn, "subtle")
        ref_row = QHBoxLayout()
        ref_row.setContentsMargins(0, 0, 0, 0)
        ref_row.setSpacing(6)
        ref_row.addWidget(self._ref_edit, 1)
        ref_row.addWidget(self._ref_browse_btn)
        self._ref_row_w = QWidget()
        self._ref_row_w.setLayout(ref_row)
        self._ref_row_label = QLabel("Reference frame:")
        layout.addRow(self._ref_row_label, self._ref_row_w)

        return box

    def _build_output_section(self) -> QWidget:
        box = QGroupBox("Outputs")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(6)

        self._save_recon_check = QCheckBox("Save reconstruction image (PNG)")
        self._save_recon_check.setChecked(True)
        layout.addWidget(self._save_recon_check)

        self._save_voltage_check = QCheckBox(
            "Save boundary voltage fit plot (PNG)"
        )
        self._save_voltage_check.setChecked(True)
        layout.addWidget(self._save_voltage_check)

        return box

    def _build_progress_section(self) -> QWidget:
        box = QGroupBox("Progress")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(6)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("Ready.")
        set_hint_text(self._progress_label)
        self._progress_label.setWordWrap(True)
        layout.addWidget(self._progress_label)

        return box

    # ---- Signals & handlers ----

    def _connect_signals(self) -> None:
        self._input_browse_btn.clicked.connect(self._on_browse_input)
        self._output_browse_btn.clicked.connect(self._on_browse_output)
        self._ref_browse_btn.clicked.connect(self._on_browse_ref)
        self._algo_combo.currentIndexChanged.connect(self._update_reference_requirement)
        self._input_edit.textChanged.connect(self._update_run_enabled)
        self._output_edit.textChanged.connect(self._update_run_enabled)
        self._ref_edit.textChanged.connect(self._update_run_enabled)

        self._run_btn.clicked.connect(self._on_run)
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._close_btn.clicked.connect(self.reject)

    def _update_reference_requirement(self, *args) -> None:
        needs_ref = _ALGORITHMS[self._algo_combo.currentIndex()][2]
        self._ref_row_label.setEnabled(needs_ref)
        self._ref_row_w.setEnabled(needs_ref)
        self._update_run_enabled()

    def _update_run_enabled(self, *args) -> None:
        if self._is_running:
            return
        has_input = bool(self._input_edit.text().strip())
        has_output = bool(self._output_edit.text().strip())
        needs_ref = _ALGORITHMS[self._algo_combo.currentIndex()][2]
        has_ref = bool(self._ref_edit.text().strip())
        ok = has_input and has_output and (not needs_ref or has_ref)
        self._run_btn.setEnabled(ok)

    def _on_browse_input(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select Input Folder", self._input_edit.text() or str(Path.home())
        )
        if path:
            self._input_edit.setText(path)

    def _on_browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", self._output_edit.text() or str(Path.home())
        )
        if path:
            self._output_edit.setText(path)

    def _on_browse_ref(self) -> None:
        start_dir = self._input_edit.text() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Frame CSV",
            start_dir,
            "CSV files (*.csv)",
        )
        if path:
            self._ref_edit.setText(path)

    def _on_run(self) -> None:
        idx = self._algo_combo.currentIndex()
        label, method, needs_ref = _ALGORITHMS[idx]
        ref_csv = self._ref_edit.text().strip() or None
        config: dict[str, Any] = {
            "input_folder": self._input_edit.text().strip(),
            "output_folder": self._output_edit.text().strip(),
            "method": method,
            "method_label": label,
            "reference_csv": ref_csv if needs_ref else None,
            "use_part": self._use_part_combo.currentText(),
            "regularization_alpha": self._alpha_spin.value(),
            "max_iterations": self._iter_spin.value(),
            "save_recon_image": self._save_recon_check.isChecked(),
            "save_voltage_fit": self._save_voltage_check.isChecked(),
        }
        self._set_running(True)
        self.start_requested.emit(config)

    def _on_cancel(self) -> None:
        self._progress_label.setText("Cancelling…")
        self.cancel_requested.emit()

    def _set_running(self, running: bool) -> None:
        self._is_running = running
        self._run_btn.setVisible(not running)
        self._cancel_btn.setVisible(running)
        for w in (
            self._input_edit,
            self._input_browse_btn,
            self._output_edit,
            self._output_browse_btn,
            self._ref_edit,
            self._ref_browse_btn,
            self._algo_combo,
            self._use_part_combo,
            self._alpha_spin,
            self._iter_spin,
            self._save_recon_check,
            self._save_voltage_check,
        ):
            w.setEnabled(not running)

    # ---- Progress API (called by parent wiring controller signals) ----

    def set_progress(self, current: int, total: int, message: str = "") -> None:
        if total > 0:
            self._progress_bar.setRange(0, total)
            self._progress_bar.setValue(current)
        self._progress_label.setText(message or f"{current}/{total}")

    def on_finished(self, succeeded: int, failed: int) -> None:
        self._set_running(False)
        self._progress_label.setText(
            f"Finished. Succeeded: {succeeded}, failed: {failed}"
        )
        self._run_btn.setEnabled(True)

    def on_error(self, message: str) -> None:
        self._set_running(False)
        self._progress_label.setText(f"Error: {message}")
