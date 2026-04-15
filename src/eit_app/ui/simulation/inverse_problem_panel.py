"""Inverse problem reconstruction controls."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

from eit_app.ui.auto_close_combo_box import AutoCloseComboBox
from eit_app.ui.theme import set_button_role, set_hint_text


_METHODS = [
    "eidors_one_step_noser",
    "eidors_abs_gn",
]


class InverseProblemPanel(QGroupBox):
    """Controls for running inverse reconstruction on simulated data."""

    run_inverse_requested = Signal()
    save_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Inverse Problem", parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QFormLayout(self)
        layout.setContentsMargins(10, 14, 10, 8)
        layout.setSpacing(8)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        hint = QLabel("Reconstruct the conductivity distribution from boundary voltages.")
        hint.setWordWrap(True)
        set_hint_text(hint)
        layout.addRow(hint)

        self._method_combo = AutoCloseComboBox()
        self._method_combo.addItems(_METHODS)
        layout.addRow("Method:", self._method_combo)

        self._alpha_spin = QDoubleSpinBox()
        self._alpha_spin.setRange(0.001, 1000.0)
        self._alpha_spin.setValue(1.0)
        self._alpha_spin.setDecimals(4)
        self._alpha_spin.setSingleStep(0.1)
        layout.addRow("Regularization \u03b1:", self._alpha_spin)

        self._iter_spin = QSpinBox()
        self._iter_spin.setRange(1, 200)
        self._iter_spin.setValue(10)
        layout.addRow("Max iterations:", self._iter_spin)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(8)

        self._recon_btn = QPushButton("Reconstruct")
        self._recon_btn.clicked.connect(self.run_inverse_requested)
        set_button_role(self._recon_btn, "primary")
        btn_row.addWidget(self._recon_btn)

        self._save_btn = QPushButton("Save Results")
        self._save_btn.clicked.connect(self.save_requested)
        self._save_btn.setEnabled(False)
        set_button_role(self._save_btn, "subtle")
        btn_row.addWidget(self._save_btn)

        layout.addRow(btn_row)

        self._status_label = QLabel("")
        set_hint_text(self._status_label)
        layout.addRow(self._status_label)

    def get_config(self) -> dict:
        return {
            "method": self._method_combo.currentText(),
            "regularization_alpha": self._alpha_spin.value(),
            "max_iterations": self._iter_spin.value(),
        }

    def set_config(self, config: dict) -> None:
        widgets = (
            self._method_combo,
            self._alpha_spin,
            self._iter_spin,
        )
        blockers = [widget.blockSignals(True) for widget in widgets]
        try:
            method = str(config.get("method", self._method_combo.currentText()))
            index = self._method_combo.findText(method)
            if index >= 0:
                self._method_combo.setCurrentIndex(index)
            self._alpha_spin.setValue(float(config.get("regularization_alpha", 1.0)))
            self._iter_spin.setValue(int(config.get("max_iterations", 10)))
        finally:
            for widget, blocked in zip(widgets, blockers):
                widget.blockSignals(blocked)

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def set_running(self, running: bool) -> None:
        self._recon_btn.setEnabled(not running)
        if running:
            self._status_label.setText("Reconstructing...")

    def set_save_enabled(self, enabled: bool) -> None:
        self._save_btn.setEnabled(enabled)
