"""Forward problem controls: noise level and solve button."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from eit_app.i18n import t, translator
from eit_app.ui.theme import set_button_role, set_hint_text


class ForwardProblemPanel(QGroupBox):
    """Controls for running the forward problem solver."""

    run_forward_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        # Title assigned by _retranslate() so it follows the UI language.
        super().__init__("", parent)
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

        self._noise_spin = QDoubleSpinBox()
        self._noise_spin.setRange(0.0, 1.0)
        self._noise_spin.setValue(0.0)
        self._noise_spin.setDecimals(4)
        self._noise_spin.setSingleStep(0.005)
        self._lbl_noise = QLabel("")
        layout.addRow(self._lbl_noise, self._noise_spin)

        self._solve_btn = QPushButton("")
        self._solve_btn.clicked.connect(self.run_forward_requested)
        set_button_role(self._solve_btn, "primary")
        layout.addRow(self._solve_btn)

        # Indeterminate "busy" bar — minRange=maxRange=0 tells Qt to keep
        # sliding the chunk so the user sees the solve is still alive.
        # Hidden by default; revealed by set_running(True).
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
        # Lock adjacent inputs during busy so users don't kick off a
        # second solve with different parameters mid-flight.
        self._noise_spin.setEnabled(not running)
        self._busy_bar.setVisible(running)
        if running:
            self._status_label.setText(t("sim.forward.status_solving"))

    # ── i18n ──

    def _retranslate(self) -> None:
        self.setTitle(t("sim.forward.title"))
        self._hint.setText(t("sim.forward.hint"))
        self._lbl_noise.setText(t("sim.forward.noise_label"))
        self._noise_spin.setToolTip(t("sim.forward.noise_tooltip"))
        self._solve_btn.setText(t("sim.forward.solve_button"))
