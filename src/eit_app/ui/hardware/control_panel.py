"""Hardware control panel: frequency, amplitude, and diagnostics settings."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from eit_app.hardware.types import VOLTAGE_AMP_LABELS
from eit_app.i18n import t, translator
from eit_app.measurement_layout import measurement_layout_from_config
from eit_app.ui.auto_close_combo_box import AutoCloseComboBox
from eit_app.ui.theme import set_button_role, set_hint_text, set_section_header, set_subtle_value


# Stimulation amplitude level descriptions — kept as numeric/unit labels
# that do not require localisation (engineering standard notation).
_STIM_AMP_LABELS = [
    "0 - 50 uA",
    "1 - 100 uA",
    "2 - 200 uA",
    "3 - 500 uA",
    "4 - 1 mA",
    "5 - 2 mA",
    "6 - 5 mA",
    "7 - 10 mA",
]

_VOLTAGE_AMP_LABELS = list(VOLTAGE_AMP_LABELS)


def _coerce_scalar_float(value: object, default: float) -> float:
    if value in (None, ""):
        return float(default)
    if isinstance(value, (list, tuple)):
        if not value:
            return float(default)
        return _coerce_scalar_float(value[0], default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class ControlPanel(QGroupBox):
    """Hardware parameter controls.

    Signals:
        frequency_changed: Emitted with frequency in Hz.
        stim_amp_changed: Emitted with amplitude level (0-7).
        voltage_amp_changed: Emitted with two voltage amp levels.
        power_toggled: Emitted with True (on) / False (off).
        single_point_requested: Emitted when user clicks single point test.
        impedance_requested: Emitted when user clicks contact impedance test.
    """

    frequency_changed = Signal(int)
    stim_amp_changed = Signal(int)
    voltage_amp_changed = Signal(int)
    power_toggled = Signal(bool)
    measurement_layout_changed = Signal(dict)
    single_point_requested = Signal()
    impedance_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        # Title is assigned by _retranslate() so it follows the UI language.
        super().__init__("", parent)
        # Labels held here so _retranslate can push new text into them.
        self._field_labels: dict[str, QLabel] = {}
        self._grid_labels: dict[str, QLabel] = {}
        self._build_ui()
        from PySide6.QtWidgets import QSizePolicy
        for spin in (
            self._n_elec_spin,
            self._n_rings_spin,
            self._exclude_neighbors_spin,
            self._radius_spin,
            self._electrode_length_spin,
            self._contact_impedance_spin,
        ):
            spin.setMinimumWidth(60)
            spin.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        for edit in (self._stim_pattern_edit, self._meas_pattern_edit):
            edit.setMinimumWidth(60)
            edit.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._mea_mode_combo.setMinimumWidth(60)
        self._mea_mode_combo.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        translator().language_changed.connect(self._retranslate)
        self._retranslate()

    def _build_ui(self) -> None:
        from PySide6.QtWidgets import QSizePolicy

        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 10, 8, 6)
        layout.setSpacing(8)

        self._power_header = QLabel("")
        set_section_header(self._power_header)
        layout.addWidget(self._power_header)

        self._power_on_btn = QPushButton("")
        self._power_on_btn.setCheckable(True)
        self._power_on_btn.clicked.connect(lambda: self._emit_power_toggle(True))
        set_button_role(self._power_on_btn, "success")
        self._power_off_btn = QPushButton("")
        self._power_off_btn.setCheckable(True)
        self._power_off_btn.clicked.connect(lambda: self._emit_power_toggle(False))
        set_button_role(self._power_off_btn, "danger")
        self._power_section = QWidget()
        power_layout = QVBoxLayout(self._power_section)
        power_layout.setContentsMargins(0, 0, 0, 0)
        power_layout.setSpacing(4)
        power_layout.addWidget(self._inline_row(self._power_on_btn, self._power_off_btn))
        self._power_hint = QLabel("")
        self._power_hint.setWordWrap(True)
        set_hint_text(self._power_hint)
        power_layout.addWidget(self._power_hint)
        layout.addWidget(self._power_section)

        self._layout_header = QLabel("")
        set_section_header(self._layout_header)
        layout.addWidget(self._layout_header)

        self._mea_mode_combo = AutoCloseComboBox()
        self._mea_mode_combo.addItems(["2D", "3D"])
        self._mea_mode_combo.setCurrentIndex(0)
        self._mea_mode_combo.currentIndexChanged.connect(lambda _: self._emit_layout_changed())

        self._n_elec_spin = QSpinBox()
        self._n_elec_spin.setRange(4, 128)
        self._n_elec_spin.setValue(16)
        self._n_elec_spin.setSingleStep(4)
        self._n_elec_spin.valueChanged.connect(lambda _: self._emit_layout_changed())

        self._n_rings_spin = QSpinBox()
        self._n_rings_spin.setRange(1, 8)
        self._n_rings_spin.setValue(1)
        self._n_rings_spin.valueChanged.connect(lambda _: self._emit_layout_changed())

        self._stim_pattern_edit = QLineEdit("{ad}")
        self._stim_pattern_edit.setPlaceholderText("{ad}")
        self._stim_pattern_edit.editingFinished.connect(self._emit_layout_changed)

        self._meas_pattern_edit = QLineEdit("{ad}")
        self._meas_pattern_edit.setPlaceholderText("{ad}")
        self._meas_pattern_edit.editingFinished.connect(self._emit_layout_changed)

        self._rotate_meas_check = QCheckBox("")
        self._rotate_meas_check.setChecked(True)
        self._rotate_meas_check.toggled.connect(lambda _checked: self._emit_layout_changed())

        self._use_meas_current_check = QCheckBox("")
        self._use_meas_current_check.setChecked(False)
        self._use_meas_current_check.toggled.connect(lambda _checked: self._emit_layout_changed())

        self._exclude_neighbors_spin = QSpinBox()
        self._exclude_neighbors_spin.setRange(0, 8)
        self._exclude_neighbors_spin.setValue(0)
        self._exclude_neighbors_spin.valueChanged.connect(lambda _: self._emit_layout_changed())

        self._radius_spin = QDoubleSpinBox()
        self._radius_spin.setRange(0.01, 1000.0)
        self._radius_spin.setDecimals(4)
        self._radius_spin.setSingleStep(0.1)
        self._radius_spin.setValue(1.0)
        self._radius_spin.valueChanged.connect(lambda _: self._emit_layout_changed())

        self._electrode_length_spin = QDoubleSpinBox()
        self._electrode_length_spin.setRange(0.000001, 1000.0)
        self._electrode_length_spin.setDecimals(6)
        self._electrode_length_spin.setSingleStep(0.01)
        self._electrode_length_spin.setValue(0.19635)
        self._electrode_length_spin.valueChanged.connect(lambda _: self._emit_layout_changed())

        self._contact_impedance_spin = QDoubleSpinBox()
        self._contact_impedance_spin.setRange(0.0, 1000000.0)
        self._contact_impedance_spin.setDecimals(6)
        self._contact_impedance_spin.setSingleStep(0.001)
        self._contact_impedance_spin.setValue(0.01)
        self._contact_impedance_spin.valueChanged.connect(lambda _: self._emit_layout_changed())

        self._layout_hint = QLabel()
        self._layout_hint.setWordWrap(True)
        set_hint_text(self._layout_hint)
        self._layout_section = self._hardware_layout_block()
        layout.addWidget(self._layout_section)

        self._setup_header = QLabel("")
        set_section_header(self._setup_header)
        layout.addWidget(self._setup_header)

        self._freq_spin = QSpinBox()
        self._freq_spin.setRange(100, 1_000_000)
        self._freq_spin.setValue(1000)
        self._freq_spin.setSuffix(" Hz")
        self._freq_spin.setSingleStep(100)
        self._freq_apply = QPushButton("")
        self._freq_apply.clicked.connect(
            lambda: self.frequency_changed.emit(self._freq_spin.value())
        )
        set_button_role(self._freq_apply, "subtle")
        self._frequency_block = self._field_block(
            "frequency", self._inline_row(self._freq_spin, self._freq_apply)
        )
        layout.addWidget(self._frequency_block)

        self._stim_combo = AutoCloseComboBox()
        self._stim_combo.addItems(_STIM_AMP_LABELS)
        self._stim_apply = QPushButton("")
        self._stim_apply.clicked.connect(
            lambda: self.stim_amp_changed.emit(self._stim_combo.currentIndex())
        )
        set_button_role(self._stim_apply, "subtle")
        self._stim_block = self._field_block(
            "stim_amp", self._inline_row(self._stim_combo, self._stim_apply)
        )
        layout.addWidget(self._stim_block)

        self._vamp_combo = AutoCloseComboBox()
        self._vamp_combo.addItems(_VOLTAGE_AMP_LABELS)
        self._vamp_combo.setCurrentIndex(7)
        self._vamp_apply = QPushButton("")
        self._vamp_apply.clicked.connect(
            lambda: self.voltage_amp_changed.emit(self._vamp_combo.currentIndex())
        )
        set_button_role(self._vamp_apply, "subtle")
        self._voltage_gain_block_w = self._field_block(
            "voltage_gain", self._inline_row(self._vamp_combo, self._vamp_apply)
        )
        layout.addWidget(self._voltage_gain_block_w)

        self._diag_header = QLabel("")
        set_section_header(self._diag_header)
        layout.addWidget(self._diag_header)

        self._spt_btn = QPushButton("")
        self._spt_btn.clicked.connect(self.single_point_requested)
        set_button_role(self._spt_btn, "primary")
        self._imp_btn = QPushButton("")
        self._imp_btn.clicked.connect(self.impedance_requested)
        set_button_role(self._imp_btn, "subtle")
        self._diagnostic_actions = self._inline_row(self._spt_btn, self._imp_btn)
        layout.addWidget(self._diagnostic_actions)

        layout.addStretch(1)

        self.set_power_state("unknown")
        self.set_measurement_layout({})

    def _inline_row(self, *widgets: QWidget) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        for _index, widget in enumerate(widgets):
            stretch = 0 if isinstance(widget, QPushButton) else 1
            row_layout.addWidget(widget, stretch)
        return row

    def _field_block(self, field_key: str, field_widget: QWidget) -> QWidget:
        block = QWidget()
        block_layout = QVBoxLayout(block)
        block_layout.setContentsMargins(0, 0, 0, 0)
        block_layout.setSpacing(4)
        label = QLabel("")  # retranslated via _field_labels[field_key]
        # Use subtle-value role so color follows theme (was #4d5f75 which
        # is invisible on the dark canvas).
        set_subtle_value(label)
        label.setStyleSheet("font-weight: 600;")
        self._field_labels[field_key] = label
        block_layout.addWidget(label)
        block_layout.addWidget(field_widget)
        return block

    def _hardware_layout_block(self) -> QWidget:
        block = QWidget()
        block_layout = QVBoxLayout(block)
        block_layout.setContentsMargins(0, 0, 0, 0)
        block_layout.setSpacing(4)

        top_grid = QGridLayout()
        top_grid.setContentsMargins(0, 0, 0, 0)
        top_grid.setHorizontalSpacing(6)
        top_grid.setVerticalSpacing(4)
        top_grid.setColumnStretch(0, 1)
        top_grid.setColumnStretch(1, 1)
        top_grid.setColumnStretch(2, 1)

        for col, key in enumerate(("mode", "elec_ring", "rings")):
            label = QLabel("")
            # Drop explicit #6a7686 — subtle-value role provides the
            # theme-aware color (invisible on dark canvas otherwise).
            set_subtle_value(label)
            label.setStyleSheet("font-size: 11px; font-weight: 700;")
            self._grid_labels[f"layout.{key}"] = label
            top_grid.addWidget(label, 0, col)
        top_grid.addWidget(self._mea_mode_combo, 1, 0)
        top_grid.addWidget(self._n_elec_spin, 1, 1)
        top_grid.addWidget(self._n_rings_spin, 1, 2)
        block_layout.addLayout(top_grid)

        pattern_grid = QGridLayout()
        pattern_grid.setContentsMargins(0, 0, 0, 0)
        pattern_grid.setHorizontalSpacing(6)
        pattern_grid.setVerticalSpacing(4)
        pattern_grid.setColumnStretch(0, 1)
        pattern_grid.setColumnStretch(1, 1)

        stim_label = QLabel("")
        set_subtle_value(stim_label)
        stim_label.setStyleSheet("font-size: 11px; font-weight: 700;")
        self._grid_labels["layout.stim_pattern"] = stim_label
        meas_label = QLabel("")
        set_subtle_value(meas_label)
        meas_label.setStyleSheet("font-size: 11px; font-weight: 700;")
        self._grid_labels["layout.meas_pattern"] = meas_label
        pattern_grid.addWidget(stim_label, 0, 0)
        pattern_grid.addWidget(meas_label, 0, 1)
        pattern_grid.addWidget(self._stim_pattern_edit, 1, 0)
        pattern_grid.addWidget(self._meas_pattern_edit, 1, 1)
        block_layout.addLayout(pattern_grid)

        options_grid = QGridLayout()
        options_grid.setContentsMargins(0, 0, 0, 0)
        options_grid.setHorizontalSpacing(6)
        options_grid.setVerticalSpacing(4)
        options_grid.setColumnStretch(0, 1)
        options_grid.setColumnStretch(1, 1)
        options_grid.addWidget(self._rotate_meas_check, 0, 0)
        options_grid.addWidget(self._use_meas_current_check, 0, 1)
        extra_label = QLabel("")
        set_subtle_value(extra_label)
        extra_label.setStyleSheet("font-size: 11px; font-weight: 700;")
        self._grid_labels["layout.extra_neighbors"] = extra_label
        options_grid.addWidget(extra_label, 1, 0)
        options_grid.addWidget(self._exclude_neighbors_spin, 1, 1)
        block_layout.addLayout(options_grid)

        cem_grid = QGridLayout()
        cem_grid.setContentsMargins(0, 0, 0, 0)
        cem_grid.setHorizontalSpacing(6)
        cem_grid.setVerticalSpacing(4)
        cem_grid.setColumnStretch(0, 1)
        cem_grid.setColumnStretch(1, 1)
        cem_grid.setColumnStretch(2, 1)
        for col, key in enumerate(("radius", "elec_length", "contact_z")):
            label = QLabel("")
            # Drop explicit #6a7686 — subtle-value role provides the
            # theme-aware color (invisible on dark canvas otherwise).
            set_subtle_value(label)
            label.setStyleSheet("font-size: 11px; font-weight: 700;")
            self._grid_labels[f"cem.{key}"] = label
            cem_grid.addWidget(label, 0, col)
        cem_grid.addWidget(self._radius_spin, 1, 0)
        cem_grid.addWidget(self._electrode_length_spin, 1, 1)
        cem_grid.addWidget(self._contact_impedance_spin, 1, 2)
        block_layout.addLayout(cem_grid)
        block_layout.addWidget(self._layout_hint)
        return block

    def _emit_power_toggle(self, on: bool) -> None:
        self.set_power_state("on" if on else "off")
        self.power_toggled.emit(on)

    def measurement_layout_config(self) -> dict:
        stim_pattern = self._stim_pattern_edit.text().strip() or "{ad}"
        meas_pattern = self._meas_pattern_edit.text().strip() or stim_pattern
        return {
            "mea_mode": 3 if self._mea_mode_combo.currentText() == "3D" else 2,
            "n_elec": self._n_elec_spin.value(),
            "n_rings": self._n_rings_spin.value(),
            "stim_pattern": stim_pattern,
            "meas_pattern": meas_pattern,
            "rotate_meas": self._rotate_meas_check.isChecked(),
            "use_meas_current": self._use_meas_current_check.isChecked(),
            "use_meas_current_next": self._exclude_neighbors_spin.value(),
            "radius": float(self._radius_spin.value()),
            "electrode_length_m_override": float(self._electrode_length_spin.value()),
            "contact_impedance": float(self._contact_impedance_spin.value()),
            "geometry_scale_to_m": 1.0,
        }

    def set_measurement_layout(self, config: dict) -> None:
        layout = measurement_layout_from_config(config)
        mea_mode = int(config.get("mea_mode", 3 if int(layout["n_rings"]) > 1 else 2))
        widgets = (
            self._mea_mode_combo,
            self._n_elec_spin,
            self._n_rings_spin,
            self._stim_pattern_edit,
            self._meas_pattern_edit,
            self._rotate_meas_check,
            self._use_meas_current_check,
            self._exclude_neighbors_spin,
            self._radius_spin,
            self._electrode_length_spin,
            self._contact_impedance_spin,
        )
        blockers = [widget.blockSignals(True) for widget in widgets]
        try:
            self._mea_mode_combo.setCurrentIndex(1 if mea_mode == 3 else 0)
            self._n_elec_spin.setValue(int(layout["n_elec"]))
            self._n_rings_spin.setValue(int(layout["n_rings"]))
            self._stim_pattern_edit.setText(str(layout["stim_pattern"]))
            self._meas_pattern_edit.setText(str(layout["meas_pattern"]))
            self._rotate_meas_check.setChecked(bool(layout["rotate_meas"]))
            self._use_meas_current_check.setChecked(bool(layout["use_meas_current"]))
            self._exclude_neighbors_spin.setValue(int(layout["use_meas_current_next"]))
            self._radius_spin.setValue(float(layout["radius"]))
            self._electrode_length_spin.setValue(
                _coerce_scalar_float(layout.get("electrode_length_m_override"), 0.19635)
            )
            self._contact_impedance_spin.setValue(float(layout["contact_impedance"]))
        finally:
            for widget, blocked in zip(widgets, blockers):
                widget.blockSignals(blocked)
        self._update_layout_hint(layout, mea_mode=mea_mode)

    def _emit_layout_changed(self) -> None:
        config = self.measurement_layout_config()
        layout = measurement_layout_from_config(config)
        self._update_layout_hint(layout, mea_mode=int(config["mea_mode"]))
        self.measurement_layout_changed.emit(config)

    def _update_layout_hint(self, layout: dict, *, mea_mode: int) -> None:
        # Stored so _retranslate() can regenerate the hint if it wants to,
        # but currently the hint is dense engineering debug info that we
        # keep in English.  Regenerating here is a no-op when the layout
        # hasn't changed.
        dimension = "3D" if mea_mode == 3 else "2D"
        rotate = "rotate" if bool(layout["rotate_meas"]) else "fixed"
        drive = "include drive electrodes" if bool(layout["use_meas_current"]) else "exclude drive electrodes"
        electrode_length = _coerce_scalar_float(layout.get("electrode_length_m_override"), 0.0)
        coverage = _coerce_scalar_float(layout.get("electrode_coverage"), 0.5)
        contact_impedance = _coerce_scalar_float(layout.get("contact_impedance"), 0.01)
        self._layout_hint.setText(
            f"{dimension} | {int(layout['n_elec'])} e/ring x {int(layout['n_rings'])} ring(s) | "
            f"{layout['stim_pattern']} / {layout['meas_pattern']} | {rotate} | {drive} | "
            f"+{int(layout['use_meas_current_next'])} extra skip | "
            f"CEM L={electrode_length:.4f} | z={contact_impedance:.4g} | "
            f"cov={coverage * 100.0:.1f}% | "
            f"expected {int(layout['points_per_frame'])} boundary samples"
        )

    def set_power_state(self, status: str) -> None:
        normalized = str(status or "").strip().lower()
        on_checked = normalized == "on"
        off_checked = normalized == "off"
        for button, checked in (
            (self._power_on_btn, on_checked),
            (self._power_off_btn, off_checked),
        ):
            blocked = button.blockSignals(True)
            button.setChecked(checked)
            button.blockSignals(blocked)

    def set_frequency_value(self, hz: int) -> None:
        blocked = self._freq_spin.blockSignals(True)
        self._freq_spin.setValue(int(hz))
        self._freq_spin.blockSignals(blocked)

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable all controls (e.g., when not connected)."""
        for child in self.findChildren(QWidget):
            child.setEnabled(enabled)

    # ── i18n ──

    def _retranslate(self) -> None:
        """Refresh all user-visible strings to the active language."""
        self.setTitle(t("hw.control.title"))

        # Section headers
        self._power_header.setText(t("hw.control.power_header"))
        self._layout_header.setText(t("hw.control.layout_header"))
        self._setup_header.setText(t("hw.control.setup_header"))
        self._diag_header.setText(t("hw.control.diag_header"))

        # Power section
        self._power_on_btn.setText(t("hw.control.power_on_button"))
        self._power_off_btn.setText(t("hw.control.power_off_button"))
        self._power_hint.setText(t("hw.control.power_hint"))

        # Layout section — grid header labels
        self._grid_labels["layout.mode"].setText(t("hw.control.layout_grid.mode"))
        self._grid_labels["layout.elec_ring"].setText(t("hw.control.layout_grid.elec_ring"))
        self._grid_labels["layout.rings"].setText(t("hw.control.layout_grid.rings"))
        self._grid_labels["layout.stim_pattern"].setText(
            t("hw.control.layout_grid.stim_pattern")
        )
        self._grid_labels["layout.meas_pattern"].setText(
            t("hw.control.layout_grid.meas_pattern")
        )
        self._grid_labels["layout.extra_neighbors"].setText(
            t("hw.control.layout_grid.extra_neighbors")
        )
        self._grid_labels["cem.radius"].setText(t("hw.control.cem_grid.radius"))
        self._grid_labels["cem.elec_length"].setText(t("hw.control.cem_grid.elec_length"))
        self._grid_labels["cem.contact_z"].setText(t("hw.control.cem_grid.contact_z"))

        # Checkboxes
        self._rotate_meas_check.setText(t("hw.control.rotate_meas_check"))
        self._use_meas_current_check.setText(t("hw.control.use_meas_current_check"))

        # Measurement-setup field blocks and Set buttons
        # Append ':' once, matching the existing visual pattern.
        self._field_labels["frequency"].setText(t("hw.control.frequency_label") + ":")
        self._field_labels["stim_amp"].setText(t("hw.control.stim_amp_label") + ":")
        self._field_labels["voltage_gain"].setText(t("hw.control.voltage_gain_label") + ":")
        self._freq_apply.setText(t("hw.control.freq_apply_button"))
        self._stim_apply.setText(t("hw.control.stim_apply_button"))
        self._vamp_apply.setText(t("hw.control.vamp_apply_button"))

        # Diagnostics
        self._spt_btn.setText(t("hw.control.spt_button"))
        self._imp_btn.setText(t("hw.control.impedance_button"))
