"""Forward problem controls: noise level and solve button."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QGroupBox, QLabel, QPushButton, QWidget

from eit_app.ui.theme import set_button_role, set_hint_text


class ForwardProblemPanel(QGroupBox):
    """Controls for running the forward problem solver."""

    run_forward_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Forward Problem", parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QFormLayout(self)
        layout.setContentsMargins(10, 14, 10, 8)
        layout.setSpacing(8)

        hint = QLabel("Compute boundary voltages from the conductivity distribution.")
        hint.setWordWrap(True)
        set_hint_text(hint)
        layout.addRow(hint)

        self._noise_spin = QDoubleSpinBox()
        self._noise_spin.setRange(0.0, 1.0)
        self._noise_spin.setValue(0.0)
        self._noise_spin.setDecimals(4)
        self._noise_spin.setSingleStep(0.005)
        self._noise_spin.setToolTip("Relative noise level (0 = noiseless)")
        layout.addRow("Noise level:", self._noise_spin)

        self._solve_btn = QPushButton("Solve Forward Problem")
        self._solve_btn.clicked.connect(self.run_forward_requested)
        set_button_role(self._solve_btn, "primary")
        layout.addRow(self._solve_btn)

        self._status_label = QLabel("")
        set_hint_text(self._status_label)
        layout.addRow(self._status_label)

    @property
    def noise_level(self) -> float:
        return self._noise_spin.value()

    def set_noise_level(self, value: float) -> None:
        blocked = self._noise_spin.blockSignals(True)
        self._noise_spin.setValue(float(value))
        self._noise_spin.blockSignals(blocked)

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def set_running(self, running: bool) -> None:
        self._solve_btn.setEnabled(not running)
        if running:
            self._status_label.setText("Solving...")
