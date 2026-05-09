"""Inverse problem reconstruction controls."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QWidget,
)

from eit_app.i18n import t, translator
from eit_app.ui.auto_close_combo_box import AutoCloseComboBox
from eit_app.ui.theme import set_button_role, set_hint_text


SIMULATION_INVERSE_METHODS = [
    "noser_rm",
    "laplace_rm",
    "curvature_rm",
    "greit3d_rm",
    "debug_fine_mesh_noser",
    "debug_full_gn",
]
_LEGACY_METHOD_ALIASES = {
    "eidors_one_step_noser": "debug_fine_mesh_noser",
    "eidors_abs_gn": "debug_full_gn",
    "eidors_demo3d_tv": "debug_full_gn",
}
_METHOD_TOOLTIP_KEYS = {
    "debug_fine_mesh_noser": "sim.inverse.method.debug_fine_mesh_noser.tooltip",
    "noser_rm": "sim.inverse.method.noser_rm.tooltip",
    "laplace_rm": "sim.inverse.method.laplace_rm.tooltip",
    "curvature_rm": "sim.inverse.method.curvature_rm.tooltip",
    "greit3d_rm": "sim.inverse.method.greit3d_rm.tooltip",
    "debug_full_gn": "sim.inverse.method.debug_full_gn.tooltip",
}
CANONICAL_SINGLE_STEP_LAMBDA_EFF = 1.0e-2
_LOCKED_LAMBDA_EFF_METHODS = {
    "noser_rm",
    "laplace_rm",
    "curvature_rm",
    "debug_fine_mesh_noser",
}
_ARTIFACT_HYPERPARAM_METHODS = {"greit3d_rm"}


def normalize_simulation_inverse_method(method: str) -> str:
    """Return the SPEC route label for a GUI simulation inverse method."""

    key = str(method or "").strip().lower()
    return _LEGACY_METHOD_ALIASES.get(key, key)


class InverseProblemPanel(QGroupBox):
    """Controls for running inverse reconstruction on simulated data."""

    run_inverse_requested = Signal()
    save_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        # Title assigned by _retranslate() so it follows the UI language.
        super().__init__("", parent)
        self._editable_alpha_value = 1.0
        self._build_ui()
        translator().language_changed.connect(self._retranslate)
        self._retranslate()

    def _build_ui(self) -> None:
        layout = QFormLayout(self)
        layout.setContentsMargins(10, 14, 10, 8)
        layout.setSpacing(8)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        set_hint_text(self._hint)
        layout.addRow(self._hint)

        self._method_combo = AutoCloseComboBox()
        # Method identifiers are invariant algorithm codes; no translation.
        self._method_combo.addItems(SIMULATION_INVERSE_METHODS)
        self._method_combo.currentIndexChanged.connect(
            lambda _index: self._update_method_state()
        )
        self._lbl_method = QLabel("")
        layout.addRow(self._lbl_method, self._method_combo)

        self._alpha_spin = QDoubleSpinBox()
        self._alpha_spin.setRange(0.001, 1000.0)
        self._alpha_spin.setValue(1.0)
        self._alpha_spin.setDecimals(4)
        self._alpha_spin.setSingleStep(0.1)
        self._alpha_spin.valueChanged.connect(self._remember_editable_alpha)
        self._lbl_alpha = QLabel("")
        layout.addRow(self._lbl_alpha, self._alpha_spin)

        self._iter_spin = QSpinBox()
        self._iter_spin.setRange(1, 200)
        self._iter_spin.setValue(10)
        self._lbl_iter = QLabel("")
        layout.addRow(self._lbl_iter, self._iter_spin)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(8)

        self._recon_btn = QPushButton("")
        self._recon_btn.clicked.connect(self.run_inverse_requested)
        set_button_role(self._recon_btn, "primary")
        btn_row.addWidget(self._recon_btn)

        self._save_btn = QPushButton("")
        self._save_btn.clicked.connect(self.save_requested)
        self._save_btn.setEnabled(False)
        set_button_role(self._save_btn, "subtle")
        btn_row.addWidget(self._save_btn)

        layout.addRow(btn_row)

        # Indeterminate busy bar, shown only while a reconstruction is
        # running.  Matches the pattern in ForwardProblemPanel so both
        # halves of the simulation workflow give the same visual feedback.
        self._busy_bar = QProgressBar()
        self._busy_bar.setRange(0, 0)
        self._busy_bar.setTextVisible(False)
        self._busy_bar.setFixedHeight(6)
        self._busy_bar.setVisible(False)
        layout.addRow(self._busy_bar)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setMinimumWidth(0)
        self._status_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        set_hint_text(self._status_label)
        layout.addRow(self._status_label)

    def get_config(self) -> dict:
        return {
            "method": normalize_simulation_inverse_method(
                self._method_combo.currentText()
            ),
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
            method = normalize_simulation_inverse_method(
                str(config.get("method", self._method_combo.currentText()))
            )
            index = self._method_combo.findText(method)
            if index >= 0:
                self._method_combo.setCurrentIndex(index)
            alpha = float(config.get("regularization_alpha", 1.0))
            if method not in _LOCKED_LAMBDA_EFF_METHODS:
                self._editable_alpha_value = alpha
            self._alpha_spin.setValue(alpha)
            self._iter_spin.setValue(int(config.get("max_iterations", 10)))
        finally:
            for widget, blocked in zip(widgets, blockers):
                widget.blockSignals(blocked)
        self._update_method_state()

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def set_running(self, running: bool) -> None:
        self._recon_btn.setEnabled(not running)
        # Lock adjacent inputs during busy so changing α / method / iters
        # mid-flight doesn't desync the next run's request.
        self._method_combo.setEnabled(not running)
        self._iter_spin.setEnabled(not running)
        if running:
            self._alpha_spin.setEnabled(False)
        else:
            self._update_hyperparameter_control()
        self._busy_bar.setVisible(running)
        if running:
            self._status_label.setText(t("sim.inverse.status_reconstructing"))

    def set_save_enabled(self, enabled: bool) -> None:
        self._save_btn.setEnabled(enabled)

    # ── i18n ──

    def _retranslate(self) -> None:
        self.setTitle(t("sim.inverse.title"))
        self._hint.setText(t("sim.inverse.hint"))
        self._lbl_method.setText(t("sim.inverse.method_label"))
        self._lbl_iter.setText(t("sim.inverse.iterations_label"))
        self._recon_btn.setText(t("sim.inverse.reconstruct_button"))
        self._save_btn.setText(t("sim.inverse.save_button"))
        self._update_method_state()

    def _remember_editable_alpha(self, value: float) -> None:
        method = normalize_simulation_inverse_method(self._method_combo.currentText())
        if method not in _LOCKED_LAMBDA_EFF_METHODS | _ARTIFACT_HYPERPARAM_METHODS:
            self._editable_alpha_value = float(value)

    def _update_method_state(self) -> None:
        self._update_method_tooltip()
        self._update_hyperparameter_control()

    def _update_method_tooltip(self) -> None:
        method = normalize_simulation_inverse_method(self._method_combo.currentText())
        tooltip = t(_METHOD_TOOLTIP_KEYS.get(method, "sim.inverse.method_label"))
        self._method_combo.setToolTip(tooltip)
        self._lbl_method.setToolTip(tooltip)

    def _update_hyperparameter_control(self) -> None:
        method = normalize_simulation_inverse_method(self._method_combo.currentText())
        blocked = self._alpha_spin.blockSignals(True)
        try:
            if method in _LOCKED_LAMBDA_EFF_METHODS:
                self._lbl_alpha.setText(t("sim.inverse.lambda_eff_locked_label"))
                tooltip = t("sim.inverse.lambda_eff_locked_tooltip")
                self._alpha_spin.setValue(CANONICAL_SINGLE_STEP_LAMBDA_EFF)
                self._alpha_spin.setEnabled(False)
            elif method in _ARTIFACT_HYPERPARAM_METHODS:
                self._lbl_alpha.setText(t("sim.inverse.artifact_weight_label"))
                tooltip = t("sim.inverse.artifact_weight_tooltip")
                self._alpha_spin.setValue(self._editable_alpha_value)
                self._alpha_spin.setEnabled(False)
            else:
                self._lbl_alpha.setText(t("sim.inverse.alpha_label"))
                tooltip = t("sim.inverse.alpha_tooltip")
                self._alpha_spin.setValue(self._editable_alpha_value)
                self._alpha_spin.setEnabled(True)
        finally:
            self._alpha_spin.blockSignals(blocked)
        self._lbl_alpha.setToolTip(tooltip)
        self._alpha_spin.setToolTip(tooltip)
