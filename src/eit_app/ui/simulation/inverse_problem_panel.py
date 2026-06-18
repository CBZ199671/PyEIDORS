"""Inverse problem reconstruction controls."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
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
    "pseudo3d_noser_rm",
    "greit",
    "absolute_gn",
]
SIMULATION_DEBUG_INVERSE_METHODS = [
    "debug_fine_mesh_noser",
    "debug_full_gn",
]
_PSEUDO3D_METHODS = {"pseudo3d_noser_rm"}
_GREIT_METHODS = {"greit"}
_LEGACY_METHOD_ALIASES = {
    "eidors_one_step_noser": "noser_rm",
    "eidors_abs_gn": "absolute_gn",
    "gn-absolute": "absolute_gn",
    "gn_absolute": "absolute_gn",
    "eidors_demo3d_tv": "debug_full_gn",
    "greit_rm": "greit",
    "greit2d_rm": "greit",
    "greit3d_rm": "greit",
    "pseudo3d": "pseudo3d_noser_rm",
    "pseudo_3d": "pseudo3d_noser_rm",
    "pseudo3d_noser": "pseudo3d_noser_rm",
    "pseudo_3d_noser": "pseudo3d_noser_rm",
}
_METHOD_TOOLTIP_KEYS = {
    "debug_fine_mesh_noser": "sim.inverse.method.debug_fine_mesh_noser.tooltip",
    "noser_rm": "sim.inverse.method.noser_rm.tooltip",
    "laplace_rm": "sim.inverse.method.laplace_rm.tooltip",
    "curvature_rm": "sim.inverse.method.curvature_rm.tooltip",
    "pseudo3d_noser_rm": "sim.inverse.method.pseudo3d_noser_rm.tooltip",
    "greit": "sim.inverse.method.greit.tooltip",
    "absolute_gn": "sim.inverse.method.absolute_gn.tooltip",
    "debug_full_gn": "sim.inverse.method.debug_full_gn.tooltip",
}
CANONICAL_SINGLE_STEP_LAMBDA_EFF = 1.0e-2
_LOCKED_LAMBDA_EFF_METHODS = {
    "noser_rm",
    "laplace_rm",
    "curvature_rm",
    "pseudo3d_noser_rm",
    "debug_fine_mesh_noser",
}
_ARTIFACT_HYPERPARAM_METHODS = set(_GREIT_METHODS)
_ITERATION_CONTROL_METHODS = {"absolute_gn"}
_CUSTOM_LAMBDA_EFF_METHODS = {
    "noser_rm",
    "laplace_rm",
    "curvature_rm",
    "pseudo3d_noser_rm",
}
_DEFAULT_GREIT_DESIRED_IMAGE_MODE = "gauss"
_DEFAULT_GREIT_WEIGHT_STRATEGY = "fixed"
_DEFAULT_GREIT_TARGET_SIZE = 0.20
_DEFAULT_GREIT_ARTIFACT_WEIGHT = 1.0
_GREIT_WEIGHT_STRATEGIES = (
    ("fixed", "sim.inverse.greit.weight_strategy.fixed"),
    ("eidors_nf1", "sim.inverse.greit.weight_strategy.eidors_nf1"),
)
_GREIT_DESIRED_IMAGE_MODES = (
    ("center", "sim.inverse.greit.desired.center"),
    ("gauss", "sim.inverse.greit.desired.gauss"),
    ("adaptive_gauss", "sim.inverse.greit.desired.adaptive_gauss"),
    ("sobol_qmc", "sim.inverse.greit.desired.sobol_qmc"),
)


def normalize_simulation_inverse_method(method: str) -> str:
    """Return the SPEC route label for a GUI simulation inverse method."""

    key = str(method or "").strip().lower()
    return _LEGACY_METHOD_ALIASES.get(key, key)


def simulation_inverse_methods_for_mesh_dimension(
    mesh_dimension: int, *, include_debug: bool = False
) -> list[str]:
    """Return simulation inverse routes that make sense for the source mesh."""

    source_dimension = 3 if int(mesh_dimension) == 3 else 2
    methods = list(SIMULATION_INVERSE_METHODS)
    if include_debug:
        methods.extend(SIMULATION_DEBUG_INVERSE_METHODS)
    if source_dimension == 3:
        return methods
    return [method for method in methods if method not in _PSEUDO3D_METHODS]


class InverseProblemPanel(QGroupBox):
    """Controls for running inverse reconstruction on simulated data."""

    run_inverse_requested = Signal()
    save_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        # Title assigned by _retranslate() so it follows the UI language.
        super().__init__("", parent)
        self._source_mesh_dimension = 2
        self._editable_alpha_value = 1.0
        self._custom_lambda_eff_value = CANONICAL_SINGLE_STEP_LAMBDA_EFF
        self._running = False
        self._build_ui()
        translator().language_changed.connect(self._retranslate)
        self._retranslate()

    def _build_ui(self) -> None:
        layout = QFormLayout(self)
        layout.setContentsMargins(10, 14, 10, 8)
        layout.setSpacing(8)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        set_hint_text(self._hint)
        layout.addRow(self._hint)

        self._method_combo = AutoCloseComboBox()
        # Method identifiers are invariant algorithm codes; no translation.
        self._populate_method_combo(preferred_method="noser_rm")
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

        self._custom_lambda_check = QCheckBox("")
        self._custom_lambda_check.toggled.connect(self._on_custom_lambda_toggled)
        layout.addRow(self._custom_lambda_check)

        self._greit_group = QGroupBox("")
        self._greit_group.setVisible(False)
        greit_layout = QFormLayout(self._greit_group)
        greit_layout.setContentsMargins(10, 12, 10, 10)
        greit_layout.setSpacing(8)
        greit_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        greit_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._greit_desired_combo = AutoCloseComboBox()
        for mode, key in _GREIT_DESIRED_IMAGE_MODES:
            self._greit_desired_combo.addItem(t(key), mode)
        self._lbl_greit_desired = QLabel("")
        greit_layout.addRow(self._lbl_greit_desired, self._greit_desired_combo)

        self._greit_target_count_spin = QSpinBox()
        self._greit_target_count_spin.setRange(0, 1_000_000)
        self._greit_target_count_spin.setValue(0)
        self._greit_target_count_spin.setSingleStep(64)
        self._lbl_greit_target_count = QLabel("")
        greit_layout.addRow(
            self._lbl_greit_target_count,
            self._greit_target_count_spin,
        )

        self._greit_target_size_spin = QDoubleSpinBox()
        self._greit_target_size_spin.setRange(0.001, 1.0)
        self._greit_target_size_spin.setDecimals(4)
        self._greit_target_size_spin.setSingleStep(0.01)
        self._greit_target_size_spin.setValue(_DEFAULT_GREIT_TARGET_SIZE)
        self._lbl_greit_target_size = QLabel("")
        greit_layout.addRow(self._lbl_greit_target_size, self._greit_target_size_spin)

        self._greit_weight_strategy_combo = AutoCloseComboBox()
        for strategy, key in _GREIT_WEIGHT_STRATEGIES:
            self._greit_weight_strategy_combo.addItem(t(key), strategy)
        self._greit_weight_strategy_combo.currentIndexChanged.connect(
            lambda _index: self._update_method_state()
        )
        self._lbl_greit_weight_strategy = QLabel("")
        greit_layout.addRow(
            self._lbl_greit_weight_strategy,
            self._greit_weight_strategy_combo,
        )

        self._greit_weight_spin = QDoubleSpinBox()
        self._greit_weight_spin.setRange(1.0e-6, 1000.0)
        self._greit_weight_spin.setDecimals(4)
        self._greit_weight_spin.setSingleStep(0.1)
        self._greit_weight_spin.setValue(_DEFAULT_GREIT_ARTIFACT_WEIGHT)
        self._greit_weight_spin.valueChanged.connect(self._remember_greit_weight)
        self._lbl_greit_weight = QLabel("")
        greit_layout.addRow(self._lbl_greit_weight, self._greit_weight_spin)

        self._greit_use_cache_check = QCheckBox("")
        self._greit_use_cache_check.setChecked(True)
        greit_layout.addRow(self._greit_use_cache_check)

        self._greit_rebuild_check = QCheckBox("")
        greit_layout.addRow(self._greit_rebuild_check)

        self._greit_cold_build_hint = QLabel("")
        self._greit_cold_build_hint.setWordWrap(True)
        set_hint_text(self._greit_cold_build_hint)
        greit_layout.addRow(self._greit_cold_build_hint)

        layout.addRow(self._greit_group)

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
        method = normalize_simulation_inverse_method(self._method_combo.currentText())
        custom_lambda_enabled = (
            method in _CUSTOM_LAMBDA_EFF_METHODS
            and self._custom_lambda_check.isChecked()
        )
        greit_weight = float(self._greit_weight_spin.value())
        greit_weight_strategy = self._greit_weight_strategy()
        regularization_alpha = (
            (1.0 if greit_weight_strategy == "eidors_nf1" else greit_weight)
            if method in _GREIT_METHODS
            else float(self._alpha_spin.value())
        )
        return {
            "method": method,
            "regularization_alpha": regularization_alpha,
            "lambda_eff_custom_enabled": custom_lambda_enabled,
            "custom_lambda_eff": self._alpha_spin.value()
            if custom_lambda_enabled
            else CANONICAL_SINGLE_STEP_LAMBDA_EFF,
            "max_iterations": self._iter_spin.value(),
            "greit_desired_image_mode": self._greit_desired_mode(),
            "greit_training_target_count": self._greit_target_count_spin.value(),
            "greit_target_size": self._greit_target_size_spin.value(),
            "greit_weight_strategy": greit_weight_strategy,
            "greit_noise_figure": 1.0
            if greit_weight_strategy == "eidors_nf1"
            else None,
            "greit_weight": greit_weight,
            "greit_use_cached_rm": self._greit_use_cache_check.isChecked(),
            "greit_rebuild_rm": self._greit_rebuild_check.isChecked(),
        }

    def set_config(self, config: dict) -> None:
        widgets = (
            self._method_combo,
            self._alpha_spin,
            self._custom_lambda_check,
            self._greit_desired_combo,
            self._greit_target_count_spin,
            self._greit_target_size_spin,
            self._greit_weight_strategy_combo,
            self._greit_weight_spin,
            self._greit_use_cache_check,
            self._greit_rebuild_check,
            self._iter_spin,
        )
        blockers = [widget.blockSignals(True) for widget in widgets]
        try:
            method = normalize_simulation_inverse_method(
                str(config.get("method", self._method_combo.currentText()))
            )
            include_debug = method in SIMULATION_DEBUG_INVERSE_METHODS
            if method not in self._available_methods(include_debug=include_debug):
                method = self._fallback_method()
                include_debug = False
            if self._method_combo.findText(method) < 0:
                self._populate_method_combo(
                    preferred_method=method,
                    include_debug=include_debug,
                )
            index = self._method_combo.findText(method)
            if index >= 0:
                self._method_combo.setCurrentIndex(index)
            alpha = float(config.get("regularization_alpha", 1.0))
            custom_lambda_enabled = (
                bool(config.get("lambda_eff_custom_enabled", False))
                and method in _CUSTOM_LAMBDA_EFF_METHODS
            )
            custom_lambda = float(
                config.get("custom_lambda_eff", config.get("lambda_eff", alpha))
            )
            self._custom_lambda_check.setChecked(custom_lambda_enabled)
            if custom_lambda_enabled:
                self._custom_lambda_eff_value = custom_lambda
                alpha = custom_lambda
            if method not in _LOCKED_LAMBDA_EFF_METHODS:
                self._editable_alpha_value = alpha
            self._alpha_spin.setValue(alpha)
            greit_weight = float(config.get("greit_weight", alpha))
            self._greit_weight_spin.setValue(greit_weight)
            self._select_greit_desired_mode(
                str(
                    config.get(
                        "greit_desired_image_mode",
                        _DEFAULT_GREIT_DESIRED_IMAGE_MODE,
                    )
                )
            )
            self._greit_target_count_spin.setValue(
                int(config.get("greit_training_target_count", 0))
            )
            self._greit_target_size_spin.setValue(
                float(config.get("greit_target_size", _DEFAULT_GREIT_TARGET_SIZE))
            )
            self._select_greit_weight_strategy(
                str(
                    config.get(
                        "greit_weight_strategy",
                        _DEFAULT_GREIT_WEIGHT_STRATEGY,
                    )
                )
            )
            self._greit_use_cache_check.setChecked(
                bool(config.get("greit_use_cached_rm", True))
            )
            self._greit_rebuild_check.setChecked(
                bool(config.get("greit_rebuild_rm", False))
            )
            self._iter_spin.setValue(int(config.get("max_iterations", 10)))
        finally:
            for widget, blocked in zip(widgets, blockers, strict=True):
                widget.blockSignals(blocked)
        self._update_method_state()

    def set_source_mesh_dimension(self, mesh_dimension: int) -> None:
        """Constrain route choices to the current simulation forward dimension."""

        source_dimension = 3 if int(mesh_dimension) == 3 else 2
        if source_dimension == self._source_mesh_dimension:
            return
        self._source_mesh_dimension = source_dimension
        current_method = normalize_simulation_inverse_method(
            self._method_combo.currentText()
        )
        self._populate_method_combo(preferred_method=current_method)
        self._update_method_state()

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def set_running(self, running: bool) -> None:
        self._running = running
        self._recon_btn.setEnabled(not running)
        # Lock adjacent inputs during busy so changing α / method / iters
        # mid-flight doesn't desync the next run's request.
        self._method_combo.setEnabled(not running)
        if running:
            self._alpha_spin.setEnabled(False)
            self._iter_spin.setEnabled(False)
            self._greit_group.setEnabled(False)
        else:
            self._update_method_state()
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
        self._custom_lambda_check.setText(t("sim.inverse.custom_lambda_check"))
        self._retranslate_greit_controls()
        self._recon_btn.setText(t("sim.inverse.reconstruct_button"))
        self._save_btn.setText(t("sim.inverse.save_button"))
        self._update_method_state()

    def _remember_editable_alpha(self, value: float) -> None:
        method = normalize_simulation_inverse_method(self._method_combo.currentText())
        if (
            method in _CUSTOM_LAMBDA_EFF_METHODS
            and self._custom_lambda_check.isChecked()
        ):
            self._custom_lambda_eff_value = float(value)
        elif method not in _LOCKED_LAMBDA_EFF_METHODS | _ARTIFACT_HYPERPARAM_METHODS:
            self._editable_alpha_value = float(value)

    def _remember_greit_weight(self, value: float) -> None:
        if (
            normalize_simulation_inverse_method(self._method_combo.currentText())
            in _GREIT_METHODS
        ):
            self._editable_alpha_value = float(value)
            blocked = self._alpha_spin.blockSignals(True)
            try:
                self._alpha_spin.setValue(float(value))
            finally:
                self._alpha_spin.blockSignals(blocked)

    def _on_custom_lambda_toggled(self, checked: bool) -> None:
        method = normalize_simulation_inverse_method(self._method_combo.currentText())
        if not checked and method in _CUSTOM_LAMBDA_EFF_METHODS:
            self._custom_lambda_eff_value = float(self._alpha_spin.value())
        self._update_method_state()

    def _available_methods(self, *, include_debug: bool = False) -> list[str]:
        return simulation_inverse_methods_for_mesh_dimension(
            self._source_mesh_dimension,
            include_debug=include_debug,
        )

    def _fallback_method(self) -> str:
        methods = self._available_methods()
        return "noser_rm" if "noser_rm" in methods else methods[0]

    def _populate_method_combo(
        self, *, preferred_method: str, include_debug: bool = False
    ) -> None:
        methods = self._available_methods(include_debug=include_debug)
        selected_method = (
            preferred_method if preferred_method in methods else self._fallback_method()
        )
        blocked = self._method_combo.blockSignals(True)
        try:
            self._method_combo.clear()
            self._method_combo.addItems(methods)
            index = self._method_combo.findText(selected_method)
            self._method_combo.setCurrentIndex(max(index, 0))
        finally:
            self._method_combo.blockSignals(blocked)

    def _update_method_state(self) -> None:
        self._update_method_tooltip()
        self._update_hyperparameter_control()
        self._update_greit_controls()
        self._update_iteration_control()

    def _update_method_tooltip(self) -> None:
        method = normalize_simulation_inverse_method(self._method_combo.currentText())
        tooltip = t(_METHOD_TOOLTIP_KEYS.get(method, "sim.inverse.method_label"))
        self._method_combo.setToolTip(tooltip)
        self._lbl_method.setToolTip(tooltip)

    def _update_hyperparameter_control(self) -> None:
        method = normalize_simulation_inverse_method(self._method_combo.currentText())
        blocked = self._alpha_spin.blockSignals(True)
        alpha_visible = True
        try:
            custom_lambda_available = method in _CUSTOM_LAMBDA_EFF_METHODS
            self._custom_lambda_check.setVisible(custom_lambda_available)
            self._custom_lambda_check.setEnabled(
                custom_lambda_available and not self._running
            )
            custom_lambda_enabled = (
                custom_lambda_available and self._custom_lambda_check.isChecked()
            )
            custom_lambda_tooltip = t("sim.inverse.custom_lambda_tooltip")
            self._custom_lambda_check.setToolTip(custom_lambda_tooltip)
            if custom_lambda_enabled:
                self._lbl_alpha.setText(t("sim.inverse.lambda_eff_custom_label"))
                tooltip = custom_lambda_tooltip
                self._alpha_spin.setValue(self._custom_lambda_eff_value)
                self._alpha_spin.setEnabled(not self._running)
            elif method in _LOCKED_LAMBDA_EFF_METHODS:
                self._lbl_alpha.setText(t("sim.inverse.lambda_eff_locked_label"))
                tooltip = t("sim.inverse.lambda_eff_locked_tooltip")
                self._alpha_spin.setValue(CANONICAL_SINGLE_STEP_LAMBDA_EFF)
                self._alpha_spin.setEnabled(False)
            elif method in _ARTIFACT_HYPERPARAM_METHODS:
                self._custom_lambda_check.setVisible(False)
                alpha_visible = False
                if self._greit_weight_strategy() == "eidors_nf1":
                    tooltip = t("sim.inverse.artifact_nf1_tooltip")
                    self._alpha_spin.setValue(1.0)
                else:
                    tooltip = t("sim.inverse.artifact_weight_tooltip")
                    self._alpha_spin.setValue(self._greit_weight_spin.value())
                self._alpha_spin.setEnabled(False)
            else:
                self._custom_lambda_check.setVisible(False)
                self._lbl_alpha.setText(t("sim.inverse.alpha_label"))
                tooltip = t("sim.inverse.alpha_tooltip")
                self._alpha_spin.setValue(self._editable_alpha_value)
                self._alpha_spin.setEnabled(not self._running)
        finally:
            self._alpha_spin.blockSignals(blocked)
        self._lbl_alpha.setVisible(alpha_visible)
        self._alpha_spin.setVisible(alpha_visible)
        self._lbl_alpha.setToolTip(tooltip)
        self._alpha_spin.setToolTip(tooltip)

    def _update_greit_controls(self) -> None:
        method = normalize_simulation_inverse_method(self._method_combo.currentText())
        visible = method in _GREIT_METHODS
        self._greit_group.setVisible(visible)
        self._greit_group.setEnabled(visible and not self._running)
        self._update_greit_weight_strategy_state(visible=visible)

    def _update_iteration_control(self) -> None:
        method = normalize_simulation_inverse_method(self._method_combo.currentText())
        visible = method in _ITERATION_CONTROL_METHODS
        self._lbl_iter.setVisible(visible)
        self._iter_spin.setVisible(visible)
        self._iter_spin.setEnabled(visible and not self._running)
        tooltip = t("sim.inverse.iterations_tooltip") if visible else ""
        self._lbl_iter.setToolTip(tooltip)
        self._iter_spin.setToolTip(tooltip)

    def _retranslate_greit_controls(self) -> None:
        current_mode = self._greit_desired_mode()
        current_strategy = self._greit_weight_strategy()
        self._greit_group.setTitle(t("sim.inverse.greit.group_title"))
        self._lbl_greit_desired.setText(t("sim.inverse.greit.desired_label"))
        for idx, (_mode, key) in enumerate(_GREIT_DESIRED_IMAGE_MODES):
            self._greit_desired_combo.setItemText(idx, t(key))
        self._select_greit_desired_mode(current_mode)
        self._lbl_greit_weight_strategy.setText(
            t("sim.inverse.greit.weight_strategy_label")
        )
        for idx, (_strategy, key) in enumerate(_GREIT_WEIGHT_STRATEGIES):
            self._greit_weight_strategy_combo.setItemText(idx, t(key))
        self._select_greit_weight_strategy(current_strategy)

        target_count_tooltip = t("sim.inverse.greit.target_count_tooltip")
        target_size_tooltip = t("sim.inverse.greit.target_size_tooltip")
        strategy_tooltip = t("sim.inverse.greit.weight_strategy_tooltip")
        weight_tooltip = t("sim.inverse.greit.weight_tooltip")
        cache_tooltip = t("sim.inverse.greit.cache_tooltip")
        rebuild_tooltip = t("sim.inverse.greit.rebuild_tooltip")

        self._lbl_greit_target_count.setText(t("sim.inverse.greit.target_count_label"))
        self._lbl_greit_target_count.setToolTip(target_count_tooltip)
        self._greit_target_count_spin.setToolTip(target_count_tooltip)
        self._lbl_greit_target_size.setText(t("sim.inverse.greit.target_size_label"))
        self._lbl_greit_target_size.setToolTip(target_size_tooltip)
        self._greit_target_size_spin.setToolTip(target_size_tooltip)
        self._lbl_greit_weight_strategy.setToolTip(strategy_tooltip)
        self._greit_weight_strategy_combo.setToolTip(strategy_tooltip)
        self._lbl_greit_weight.setText(t("sim.inverse.greit.weight_label"))
        self._lbl_greit_weight.setToolTip(weight_tooltip)
        self._greit_weight_spin.setToolTip(weight_tooltip)
        self._greit_use_cache_check.setText(t("sim.inverse.greit.use_cache_check"))
        self._greit_use_cache_check.setToolTip(cache_tooltip)
        self._greit_rebuild_check.setText(t("sim.inverse.greit.rebuild_check"))
        self._greit_rebuild_check.setToolTip(rebuild_tooltip)
        self._greit_cold_build_hint.setText(t("sim.inverse.greit.cold_build_hint"))

    def _greit_desired_mode(self) -> str:
        data = self._greit_desired_combo.currentData()
        mode = str(data or "").strip()
        return mode or _DEFAULT_GREIT_DESIRED_IMAGE_MODE

    def _greit_weight_strategy(self) -> str:
        data = self._greit_weight_strategy_combo.currentData()
        strategy = str(data or "").strip().lower()
        if strategy not in {item[0] for item in _GREIT_WEIGHT_STRATEGIES}:
            return _DEFAULT_GREIT_WEIGHT_STRATEGY
        return strategy

    def _select_greit_desired_mode(self, mode: str) -> None:
        normalized = str(mode or _DEFAULT_GREIT_DESIRED_IMAGE_MODE).strip()
        for idx in range(self._greit_desired_combo.count()):
            if self._greit_desired_combo.itemData(idx) == normalized:
                self._greit_desired_combo.setCurrentIndex(idx)
                return
        self._greit_desired_combo.setCurrentIndex(1)

    def _select_greit_weight_strategy(self, strategy: str) -> None:
        normalized = str(strategy or _DEFAULT_GREIT_WEIGHT_STRATEGY).strip().lower()
        for idx in range(self._greit_weight_strategy_combo.count()):
            if self._greit_weight_strategy_combo.itemData(idx) == normalized:
                self._greit_weight_strategy_combo.setCurrentIndex(idx)
                return
        self._greit_weight_strategy_combo.setCurrentIndex(0)

    def _update_greit_weight_strategy_state(self, *, visible: bool) -> None:
        fixed_weight = self._greit_weight_strategy() == "fixed"
        enabled = visible and fixed_weight and not self._running
        self._lbl_greit_weight.setEnabled(visible and fixed_weight)
        self._greit_weight_spin.setEnabled(enabled)
        self._greit_weight_strategy_combo.setEnabled(visible and not self._running)
