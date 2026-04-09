"""Hardware control panel: frequency, amplitude, and sweep settings."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QWidget

from eit_app.ui.auto_close_combo_box import AutoCloseComboBox
from eit_app.ui.theme import set_button_role, set_section_header


# Stimulation amplitude level descriptions
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

_VOLTAGE_AMP_LABELS = [
    "0 - 0.097x",
    "1 - 0.175x",
    "2 - 0.327x",
    "3 - 0.623x",
    "4 - 1.238x",
    "5 - 2.460x",
    "6 - 4.880x",
    "7 - 9.000x",
]


class ControlPanel(QGroupBox):
    """Hardware parameter controls.

    Signals:
        frequency_changed: Emitted with frequency in Hz.
        stim_amp_changed: Emitted with amplitude level (0-7).
        voltage_amp_changed: Emitted with two voltage amp levels.
        power_toggled: Emitted with True (on) / False (off).
        single_point_requested: Emitted when user clicks single point test.
        impedance_requested: Emitted when user clicks contact impedance test.
        sweep_requested: Emitted with (start_hz, end_hz, n_points).
    """

    frequency_changed = Signal(int)
    stim_amp_changed = Signal(int)
    voltage_amp_changed = Signal(int, int)
    power_toggled = Signal(bool)
    single_point_requested = Signal()
    impedance_requested = Signal()
    sweep_requested = Signal(int, int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("2. Setup & Diagnostics", parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QFormLayout(self)
        layout.setContentsMargins(10, 14, 10, 8)
        layout.setSpacing(10)

        self._setup_header = QLabel("Measurement setup")
        set_section_header(self._setup_header)
        layout.addRow(self._setup_header)

        # Frequency
        self._freq_spin = QSpinBox()
        self._freq_spin.setRange(100, 1_000_000)
        self._freq_spin.setValue(1000)
        self._freq_spin.setSuffix(" Hz")
        self._freq_spin.setSingleStep(100)
        self._freq_apply = QPushButton("Set")
        self._freq_apply.clicked.connect(
            lambda: self.frequency_changed.emit(self._freq_spin.value())
        )
        set_button_role(self._freq_apply, "subtle")
        freq_row = QHBoxLayout()
        freq_row.setContentsMargins(0, 0, 0, 0)
        freq_row.setSpacing(8)
        freq_row.addWidget(self._freq_spin, 1)
        freq_row.addWidget(self._freq_apply)
        freq_w = QWidget()
        freq_w.setLayout(freq_row)
        layout.addRow("Frequency:", freq_w)

        # Stimulation amplitude
        self._stim_combo = AutoCloseComboBox()
        self._stim_combo.addItems(_STIM_AMP_LABELS)
        self._stim_apply = QPushButton("Set")
        self._stim_apply.clicked.connect(
            lambda: self.stim_amp_changed.emit(self._stim_combo.currentIndex())
        )
        set_button_role(self._stim_apply, "subtle")
        stim_row = QHBoxLayout()
        stim_row.setContentsMargins(0, 0, 0, 0)
        stim_row.setSpacing(8)
        stim_row.addWidget(self._stim_combo, 1)
        stim_row.addWidget(self._stim_apply)
        stim_w = QWidget()
        stim_w.setLayout(stim_row)
        layout.addRow("Stim amplitude:", stim_w)

        # Voltage amplification
        self._vamp_combo_1 = AutoCloseComboBox()
        self._vamp_combo_1.addItems(_VOLTAGE_AMP_LABELS)
        self._vamp_combo_1.setCurrentIndex(3)
        self._vamp_combo_2 = AutoCloseComboBox()
        self._vamp_combo_2.addItems(_VOLTAGE_AMP_LABELS)
        self._vamp_combo_2.setCurrentIndex(5)
        self._vamp_apply = QPushButton("Set")
        self._vamp_apply.clicked.connect(
            lambda: self.voltage_amp_changed.emit(
                self._vamp_combo_1.currentIndex(),
                self._vamp_combo_2.currentIndex(),
            )
        )
        set_button_role(self._vamp_apply, "subtle")
        vamp_row = QHBoxLayout()
        vamp_row.setContentsMargins(0, 0, 0, 0)
        vamp_row.setSpacing(8)
        vamp_row.addWidget(self._vamp_combo_1, 1)
        vamp_row.addWidget(self._vamp_combo_2, 1)
        vamp_row.addWidget(self._vamp_apply)
        vamp_w = QWidget()
        vamp_w.setLayout(vamp_row)
        layout.addRow("Voltage amps:", vamp_w)

        self._diag_header = QLabel("Power & diagnostics")
        set_section_header(self._diag_header)
        layout.addRow(self._diag_header)

        # Power control
        self._power_on_btn = QPushButton("Turn Power ON")
        self._power_on_btn.clicked.connect(lambda: self.power_toggled.emit(True))
        set_button_role(self._power_on_btn, "success")
        self._power_off_btn = QPushButton("Turn Power OFF")
        self._power_off_btn.clicked.connect(lambda: self.power_toggled.emit(False))
        set_button_role(self._power_off_btn, "danger")
        pwr_row = QHBoxLayout()
        pwr_row.setContentsMargins(0, 0, 0, 0)
        pwr_row.setSpacing(8)
        pwr_row.addWidget(self._power_on_btn)
        pwr_row.addWidget(self._power_off_btn)
        pwr_w = QWidget()
        pwr_w.setLayout(pwr_row)
        layout.addRow("Measurement power:", pwr_w)

        # Test buttons
        self._spt_btn = QPushButton("Single Point Test")
        self._spt_btn.clicked.connect(self.single_point_requested)
        set_button_role(self._spt_btn, "primary")
        layout.addRow(self._spt_btn)

        self._imp_btn = QPushButton("Contact Impedance")
        self._imp_btn.clicked.connect(self.impedance_requested)
        set_button_role(self._imp_btn, "subtle")
        layout.addRow(self._imp_btn)

        # Sweep
        sweep_row = QHBoxLayout()
        self._sweep_start = QSpinBox()
        self._sweep_start.setRange(100, 1_000_000)
        self._sweep_start.setValue(1000)
        self._sweep_start.setSuffix(" Hz")
        self._sweep_end = QSpinBox()
        self._sweep_end.setRange(100, 1_000_000)
        self._sweep_end.setValue(100_000)
        self._sweep_end.setSuffix(" Hz")
        self._sweep_points = QSpinBox()
        self._sweep_points.setRange(2, 1000)
        self._sweep_points.setValue(10)
        self._sweep_btn = QPushButton("Sweep")
        self._sweep_btn.clicked.connect(
            lambda: self.sweep_requested.emit(
                self._sweep_start.value(),
                self._sweep_end.value(),
                self._sweep_points.value(),
            )
        )
        set_button_role(self._sweep_btn, "subtle")
        sweep_row.setContentsMargins(0, 0, 0, 0)
        sweep_row.setSpacing(6)
        sweep_row.addWidget(self._sweep_start)
        sweep_row.addWidget(self._sweep_end)
        sweep_row.addWidget(self._sweep_points)
        sweep_row.addWidget(self._sweep_btn)
        sweep_w = QWidget()
        sweep_w.setLayout(sweep_row)
        layout.addRow("Sweep:", sweep_w)

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable all controls (e.g., when not connected)."""
        for child in self.findChildren(QWidget):
            child.setEnabled(enabled)
