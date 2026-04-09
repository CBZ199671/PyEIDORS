"""Acquisition controls: start/stop, scheduled mode, recording toggle."""

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

from eit_app.ui.theme import set_button_role, set_hint_text, set_section_header, set_subtle_value


class AcquisitionPanel(QGroupBox):
    """Panel for controlling data acquisition and recording.

    Signals:
        start_requested: Emitted when user clicks Start.
        single_frame_requested: Emitted when user clicks Single Frame.
        stop_requested: Emitted when user clicks Stop.
        recording_toggled: Emitted with (active, output_dir).
        scheduled_mode_changed: Emitted with (enabled, interval_sec, frames_per_burst).
    """

    start_requested = Signal()
    single_frame_requested = Signal()
    stop_requested = Signal()
    recording_toggled = Signal(bool, str)
    output_dir_changed = Signal(str)
    scheduled_mode_changed = Signal(bool, float, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("3. Acquire & Record", parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QFormLayout(self)
        layout.setContentsMargins(10, 14, 10, 8)
        layout.setSpacing(10)
        self._layout = layout

        self._flow_hint = QLabel("Prepare the save path and plan, then launch the acquisition run.")
        self._flow_hint.setWordWrap(True)
        set_hint_text(self._flow_hint)
        layout.addRow(self._flow_hint)

        self._record_header = QLabel("Recording setup")
        set_section_header(self._record_header)
        layout.addRow(self._record_header)

        dir_row = QHBoxLayout()
        dir_row.setContentsMargins(0, 0, 0, 0)
        dir_row.setSpacing(8)
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("Output directory...")
        self._dir_edit.setText(str(self.default_output_dir()))
        self._dir_edit.textChanged.connect(self.output_dir_changed)
        self._dir_browse = QPushButton("Browse...")
        self._dir_browse.clicked.connect(self._browse_dir)
        set_button_role(self._dir_browse, "subtle")
        dir_row.addWidget(self._dir_edit, 1)
        dir_row.addWidget(self._dir_browse)
        dir_w = QWidget()
        dir_w.setLayout(dir_row)
        layout.addRow("Save to:", dir_w)

        # Recording
        rec_row = QHBoxLayout()
        rec_row.setContentsMargins(0, 0, 0, 0)
        self._rec_check = QCheckBox("Record to disk")
        self._rec_check.clicked.connect(self._on_recording_clicked)
        rec_row.addWidget(self._rec_check)
        rec_w = QWidget()
        rec_w.setLayout(rec_row)
        layout.addRow(rec_w)

        self._plan_header = QLabel("Acquisition plan")
        set_section_header(self._plan_header)
        layout.addRow(self._plan_header)

        # Scheduled mode
        self._sched_check = QCheckBox("Scheduled mode")
        self._sched_check.toggled.connect(self._on_schedule_toggled)
        layout.addRow(self._sched_check)

        self._interval_spin = QDoubleSpinBox()
        self._interval_spin.setRange(1.0, 86400.0)
        self._interval_spin.setValue(300.0)
        self._interval_spin.setSuffix(" s")
        self._interval_spin.setDecimals(1)
        self._interval_spin.valueChanged.connect(lambda _: self._emit_schedule_state())
        layout.addRow("Interval:", self._interval_spin)

        self._burst_spin = QSpinBox()
        self._burst_spin.setRange(1, 10000)
        self._burst_spin.setValue(1)
        self._burst_spin.valueChanged.connect(lambda _: self._emit_schedule_state())
        layout.addRow("Frames/burst:", self._burst_spin)

        self._set_row_visible(self._interval_spin, False)
        self._set_row_visible(self._burst_spin, False)

        self._action_header = QLabel("Acquisition actions")
        set_section_header(self._action_header)
        layout.addRow(self._action_header)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(8)
        self._start_btn = QPushButton("Start Continuous")
        self._start_btn.clicked.connect(self.start_requested)
        set_button_role(self._start_btn, "primary")
        self._single_frame_btn = QPushButton("Acquire One Frame")
        self._single_frame_btn.clicked.connect(self.single_frame_requested)
        set_button_role(self._single_frame_btn, "success")
        self._stop_btn = QPushButton("Stop Acquisition")
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
        layout.addRow("Frames acquired:", self._frame_label)

    def _on_schedule_toggled(self, checked: bool) -> None:
        self._set_row_visible(self._interval_spin, checked)
        self._set_row_visible(self._burst_spin, checked)
        self._emit_schedule_state()

    def _on_recording_clicked(self, checked: bool) -> None:
        # Let the checkbox paint its new state before potentially expensive
        # session setup work runs in the main window.
        QTimer.singleShot(0, lambda checked=checked: self.recording_toggled.emit(checked, self._dir_edit.text()))

    def _emit_schedule_state(self) -> None:
        self.scheduled_mode_changed.emit(
            self._sched_check.isChecked(),
            self._interval_spin.value(),
            self._burst_spin.value(),
        )

    def _browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
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
