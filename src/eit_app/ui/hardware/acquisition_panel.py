"""Acquisition controls: recording plus finite/timed acquisition plans."""

from pathlib import Path

from PySide6.QtCore import QSignalBlocker, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from eit_app.i18n import t, translator
from eit_app.ui.theme import (
    set_button_role,
    set_hint_text,
    set_section_header,
    set_subtle_value,
)


class AcquisitionPanel(QGroupBox):
    """Panel for controlling data acquisition and recording.

    Signals:
        start_requested: Emitted when user clicks Start.
        single_frame_requested: Emitted when user clicks Single Frame.
        stop_requested: Emitted when user clicks Stop.
        recording_toggled: Emitted with (active, output_dir).
        acquisition_plan_changed: Emitted with the current acquisition plan.
    """

    start_requested = Signal()
    single_frame_requested = Signal()
    stop_requested = Signal()
    recording_toggled = Signal(bool, str)
    output_dir_changed = Signal(str)
    acquisition_plan_changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        # Title is assigned by _retranslate() so it follows the UI language.
        super().__init__("", parent)
        self._build_ui()
        translator().language_changed.connect(self._retranslate)
        self._retranslate()

    def _build_ui(self) -> None:
        layout = QFormLayout(self)
        layout.setContentsMargins(8, 10, 8, 6)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._layout = layout

        self._flow_hint = QLabel("")
        self._flow_hint.setWordWrap(True)
        set_hint_text(self._flow_hint)
        layout.addRow(self._flow_hint)

        self._record_header = QLabel("")
        set_section_header(self._record_header)
        layout.addRow(self._record_header)

        dir_row = QHBoxLayout()
        dir_row.setContentsMargins(0, 0, 0, 0)
        dir_row.setSpacing(8)
        self._dir_edit = QLineEdit()
        self._dir_edit.setText(str(self.default_output_dir()))
        self._dir_edit.textChanged.connect(self.output_dir_changed)
        self._dir_browse = QPushButton("")
        self._dir_browse.clicked.connect(self._browse_dir)
        set_button_role(self._dir_browse, "subtle")
        dir_row.addWidget(self._dir_edit, 1)
        dir_row.addWidget(self._dir_browse)
        dir_w = QWidget()
        dir_w.setLayout(dir_row)
        self._lbl_save_to = QLabel("")
        layout.addRow(self._lbl_save_to, dir_w)

        # Recording
        rec_row = QHBoxLayout()
        rec_row.setContentsMargins(0, 0, 0, 0)
        self._rec_check = QCheckBox("")
        self._rec_check.clicked.connect(self._on_recording_clicked)
        rec_row.addWidget(self._rec_check)
        rec_w = QWidget()
        rec_w.setLayout(rec_row)
        layout.addRow(rec_w)

        self._plan_header = QLabel("")
        set_section_header(self._plan_header)
        layout.addRow(self._plan_header)

        self._sched_check = QCheckBox("")
        self._sched_check.toggled.connect(self._on_plan_toggled)
        layout.addRow(self._sched_check)

        self._interval_spin = QDoubleSpinBox()
        self._interval_spin.setRange(0.1, 86400.0)
        self._interval_spin.setValue(5.0)
        self._interval_spin.setSuffix(" s")
        self._interval_spin.setDecimals(2)
        self._interval_spin.valueChanged.connect(lambda _: self._emit_plan_state())
        self._lbl_interval = QLabel("")
        layout.addRow(self._lbl_interval, self._interval_spin)

        self._count_spin = QSpinBox()
        self._count_spin.setRange(0, 10000)
        # Special-value text updated in _retranslate so it follows UI language.
        self._count_spin.setValue(0)
        self._count_spin.valueChanged.connect(lambda _: self._emit_plan_state())
        self._lbl_count = QLabel("")
        layout.addRow(self._lbl_count, self._count_spin)

        self._freq_step_check = QCheckBox("")
        self._freq_step_check.toggled.connect(self._on_plan_toggled)
        layout.addRow(self._freq_step_check)

        self._freq_start_spin = QSpinBox()
        self._freq_start_spin.setRange(100, 1_000_000)
        self._freq_start_spin.setValue(1000)
        self._freq_start_spin.setSuffix(" Hz")
        self._freq_start_spin.valueChanged.connect(lambda _: self._emit_plan_state())
        self._lbl_start_freq = QLabel("")
        layout.addRow(self._lbl_start_freq, self._freq_start_spin)

        self._freq_end_spin = QSpinBox()
        self._freq_end_spin.setRange(100, 1_000_000)
        self._freq_end_spin.setValue(1000)
        self._freq_end_spin.setSuffix(" Hz")
        self._freq_end_spin.valueChanged.connect(lambda _: self._emit_plan_state())
        self._lbl_end_freq = QLabel("")
        layout.addRow(self._lbl_end_freq, self._freq_end_spin)

        self._set_row_visible(self._interval_spin, False)
        self._set_row_visible(self._freq_start_spin, False)
        self._set_row_visible(self._freq_end_spin, False)

        self._plan_hint = QLabel("")
        self._plan_hint.setWordWrap(True)
        set_hint_text(self._plan_hint)
        layout.addRow(self._plan_hint)

        self._action_header = QLabel("")
        set_section_header(self._action_header)
        layout.addRow(self._action_header)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(6)
        self._start_btn = QPushButton("")
        self._start_btn.clicked.connect(self.start_requested)
        set_button_role(self._start_btn, "primary")
        self._single_frame_btn = QPushButton("")
        self._single_frame_btn.clicked.connect(self.single_frame_requested)
        set_button_role(self._single_frame_btn, "success")
        self._stop_btn = QPushButton("")
        self._stop_btn.clicked.connect(self.stop_requested)
        self._stop_btn.setEnabled(False)
        set_button_role(self._stop_btn, "danger")
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._single_frame_btn)
        btn_row.addWidget(self._stop_btn)
        btn_w = QWidget()
        btn_w.setLayout(btn_row)
        layout.addRow(btn_w)

        # Frame counter
        self._frame_label = QLabel("0")
        set_subtle_value(self._frame_label)
        self._lbl_frames_acquired = QLabel("")
        layout.addRow(self._lbl_frames_acquired, self._frame_label)

    def _on_plan_toggled(self, _checked: bool) -> None:
        self._set_row_visible(self._interval_spin, self._sched_check.isChecked())
        self._set_row_visible(self._freq_start_spin, self._freq_step_check.isChecked())
        self._set_row_visible(self._freq_end_spin, self._freq_step_check.isChecked())
        self._emit_plan_state()

    def _on_recording_clicked(self, checked: bool) -> None:
        # Let the checkbox paint its new state before potentially expensive
        # session setup work runs in the main window.
        QTimer.singleShot(
            0,
            lambda checked=checked: self.recording_toggled.emit(
                checked, self._dir_edit.text()
            ),
        )

    def _emit_plan_state(self) -> None:
        self.acquisition_plan_changed.emit(self.acquisition_plan())

    def acquisition_plan(self) -> dict:
        return {
            "timed_enabled": self._sched_check.isChecked(),
            "interval_sec": float(self._interval_spin.value()),
            "acquisition_count": int(self._count_spin.value()),
            "frequency_stepping": self._freq_step_check.isChecked(),
            "start_hz": int(self._freq_start_spin.value()),
            "end_hz": int(self._freq_end_spin.value()),
        }

    def set_acquisition_plan(self, plan: dict) -> None:
        widgets = (
            self._sched_check,
            self._interval_spin,
            self._count_spin,
            self._freq_step_check,
            self._freq_start_spin,
            self._freq_end_spin,
        )
        blockers = [QSignalBlocker(widget) for widget in widgets]
        try:
            self._sched_check.setChecked(
                bool(plan.get("timed_enabled", self._sched_check.isChecked()))
            )
            self._interval_spin.setValue(
                float(plan.get("interval_sec", self._interval_spin.value()))
            )
            self._count_spin.setValue(
                int(plan.get("acquisition_count", self._count_spin.value()))
            )
            self._freq_step_check.setChecked(
                bool(plan.get("frequency_stepping", self._freq_step_check.isChecked()))
            )
            self._freq_start_spin.setValue(
                int(plan.get("start_hz", self._freq_start_spin.value()))
            )
            self._freq_end_spin.setValue(
                int(plan.get("end_hz", self._freq_end_spin.value()))
            )
        finally:
            del blockers
        self._on_plan_toggled(False)

    def _browse_dir(self) -> None:
        current = self._dir_edit.text().strip()
        default_root = self.default_output_dir()
        try:
            default_root.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        start = current if current and Path(current).exists() else str(default_root)
        path = QFileDialog.getExistingDirectory(
            self, t("hw.acquisition.file_dialog_title"), start
        )
        if path:
            self._dir_edit.setText(path)

    @staticmethod
    def default_output_dir() -> Path:
        project_measurements = (Path.cwd() / "data" / "measurements").resolve()
        if project_measurements.parent.exists():
            return project_measurements
        return (Path.cwd() / "eit_recordings").resolve()

    def set_acquiring(self, active: bool) -> None:
        """Update UI state for acquisition running/stopped."""
        self._start_btn.setEnabled(not active)
        self._single_frame_btn.setEnabled(not active)
        self._stop_btn.setEnabled(active)

    def set_frame_count(self, count: int) -> None:
        self._frame_label.setText(str(count))

    def set_recording_active(self, active: bool) -> None:
        blocker = QSignalBlocker(self._rec_check)
        self._rec_check.setChecked(active)
        del blocker

    def output_dir(self) -> str:
        return self._dir_edit.text().strip()

    def set_output_dir(self, path: str) -> None:
        self._dir_edit.setText(path)

    def _set_row_visible(self, field: QWidget, visible: bool) -> None:
        try:
            self._layout.setRowVisible(field, visible)
        except AttributeError:
            field.setVisible(visible)

    # ── i18n ──

    def _retranslate(self) -> None:
        """Refresh all user-visible strings to the active language."""
        self.setTitle(t("hw.acquisition.title"))

        self._flow_hint.setText(t("hw.acquisition.flow_hint"))

        self._record_header.setText(t("hw.acquisition.record_header"))
        self._lbl_save_to.setText(t("hw.acquisition.save_to_label"))
        self._dir_edit.setPlaceholderText(t("hw.acquisition.dir_placeholder"))
        self._dir_browse.setText(t("hw.acquisition.browse_button"))
        self._rec_check.setText(t("hw.acquisition.record_check"))

        self._plan_header.setText(t("hw.acquisition.plan_header"))
        self._sched_check.setText(t("hw.acquisition.timed_interval_check"))
        self._lbl_interval.setText(t("hw.acquisition.interval_label"))
        self._lbl_count.setText(t("hw.acquisition.count_label"))
        self._count_spin.setSpecialValueText(t("hw.acquisition.count_continuous"))
        self._freq_step_check.setText(t("hw.acquisition.freq_step_check"))
        self._lbl_start_freq.setText(t("hw.acquisition.start_freq_label"))
        self._lbl_end_freq.setText(t("hw.acquisition.end_freq_label"))
        self._plan_hint.setText(t("hw.acquisition.plan_hint"))

        self._action_header.setText(t("hw.acquisition.action_header"))
        self._start_btn.setText(t("hw.acquisition.start_button"))
        self._start_btn.setToolTip(t("hw.acquisition.start_button_tooltip"))
        self._single_frame_btn.setText(t("hw.acquisition.single_frame_button"))
        self._single_frame_btn.setToolTip(
            t("hw.acquisition.single_frame_button_tooltip")
        )
        self._stop_btn.setText(t("hw.acquisition.stop_button"))
        self._stop_btn.setToolTip(t("hw.acquisition.stop_button_tooltip"))

        self._lbl_frames_acquired.setText(t("hw.acquisition.frames_acquired_label"))
