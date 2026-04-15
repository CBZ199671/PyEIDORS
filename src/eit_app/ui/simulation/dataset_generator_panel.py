"""Dataset-generation workflow panels and shared control surface."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QCheckBox,
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
    QWidget,
)

from eit_app.ui.theme import set_button_role, set_hint_text, set_section_header, set_subtle_value


class DatasetRandomizationPanel(QGroupBox):
    """Controls the random-generation parameter ranges."""

    config_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Randomization Ranges", parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QFormLayout(self)
        layout.setContentsMargins(10, 14, 10, 8)
        layout.setSpacing(8)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        hint = QLabel(
            "Choose which shapes to sample and the numeric ranges used to paint "
            "synthetic conductivity targets."
        )
        hint.setWordWrap(True)
        set_hint_text(hint)
        layout.addRow(hint)

        shape_header = QLabel("Shape families")
        set_section_header(shape_header)
        layout.addRow(shape_header)

        shape_row = QHBoxLayout()
        shape_row.setContentsMargins(0, 0, 0, 0)
        shape_row.setSpacing(8)
        self._circle_check = QCheckBox("Circle")
        self._circle_check.setChecked(True)
        self._ellipse_check = QCheckBox("Ellipse")
        self._rect_check = QCheckBox("Rectangle")
        for widget in (self._circle_check, self._ellipse_check, self._rect_check):
            widget.toggled.connect(lambda _checked=False: self.config_changed.emit())
            shape_row.addWidget(widget)
        shape_w = QWidget()
        shape_w.setLayout(shape_row)
        layout.addRow("Shapes:", shape_w)

        count_header = QLabel("Target population")
        set_section_header(count_header)
        layout.addRow(count_header)

        n_row = QHBoxLayout()
        n_row.setContentsMargins(0, 0, 0, 0)
        n_row.setSpacing(6)
        self._n_min_spin = QSpinBox()
        self._n_min_spin.setRange(0, 20)
        self._n_min_spin.setValue(1)
        self._n_max_spin = QSpinBox()
        self._n_max_spin.setRange(1, 20)
        self._n_max_spin.setValue(3)
        for widget in (self._n_min_spin, self._n_max_spin):
            widget.valueChanged.connect(lambda _value: self.config_changed.emit())
        n_row.addWidget(self._n_min_spin)
        n_row.addWidget(QLabel("~"))
        n_row.addWidget(self._n_max_spin)
        n_w = QWidget()
        n_w.setLayout(n_row)
        layout.addRow("N inhom.:", n_w)

        position_header = QLabel("Spatial ranges")
        set_section_header(position_header)
        layout.addRow(position_header)

        pos_row = QHBoxLayout()
        pos_row.setContentsMargins(0, 0, 0, 0)
        pos_row.setSpacing(6)
        self._pos_min = QDoubleSpinBox()
        self._pos_min.setRange(-1.0, 1.0)
        self._pos_min.setValue(-0.7)
        self._pos_min.setDecimals(2)
        self._pos_max = QDoubleSpinBox()
        self._pos_max.setRange(-1.0, 1.0)
        self._pos_max.setValue(0.7)
        self._pos_max.setDecimals(2)
        for widget in (self._pos_min, self._pos_max):
            widget.valueChanged.connect(lambda _value: self.config_changed.emit())
        pos_row.addWidget(self._pos_min)
        pos_row.addWidget(QLabel("~"))
        pos_row.addWidget(self._pos_max)
        pos_w = QWidget()
        pos_w.setLayout(pos_row)
        layout.addRow("Position:", pos_w)

        size_row = QHBoxLayout()
        size_row.setContentsMargins(0, 0, 0, 0)
        size_row.setSpacing(6)
        self._size_min = QDoubleSpinBox()
        self._size_min.setRange(0.01, 1.0)
        self._size_min.setValue(0.05)
        self._size_min.setDecimals(3)
        self._size_max = QDoubleSpinBox()
        self._size_max.setRange(0.01, 1.0)
        self._size_max.setValue(0.3)
        self._size_max.setDecimals(3)
        for widget in (self._size_min, self._size_max):
            widget.valueChanged.connect(lambda _value: self.config_changed.emit())
        size_row.addWidget(self._size_min)
        size_row.addWidget(QLabel("~"))
        size_row.addWidget(self._size_max)
        size_w = QWidget()
        size_w.setLayout(size_row)
        layout.addRow("Size:", size_w)

        conductivity_header = QLabel("Conductivity ranges")
        set_section_header(conductivity_header)
        layout.addRow(conductivity_header)

        cond_row = QHBoxLayout()
        cond_row.setContentsMargins(0, 0, 0, 0)
        cond_row.setSpacing(6)
        self._cond_min = QDoubleSpinBox()
        self._cond_min.setRange(0.001, 100.0)
        self._cond_min.setValue(0.5)
        self._cond_min.setDecimals(3)
        self._cond_min.setSuffix(" S/m")
        self._cond_max = QDoubleSpinBox()
        self._cond_max.setRange(0.001, 100.0)
        self._cond_max.setValue(3.0)
        self._cond_max.setDecimals(3)
        self._cond_max.setSuffix(" S/m")
        for widget in (self._cond_min, self._cond_max):
            widget.valueChanged.connect(lambda _value: self.config_changed.emit())
        cond_row.addWidget(self._cond_min)
        cond_row.addWidget(QLabel("~"))
        cond_row.addWidget(self._cond_max)
        cond_w = QWidget()
        cond_w.setLayout(cond_row)
        layout.addRow("\u03c3 range:", cond_w)

        bg_row = QHBoxLayout()
        bg_row.setContentsMargins(0, 0, 0, 0)
        bg_row.setSpacing(6)
        self._bg_min = QDoubleSpinBox()
        self._bg_min.setRange(0.001, 100.0)
        self._bg_min.setValue(0.8)
        self._bg_min.setDecimals(3)
        self._bg_max = QDoubleSpinBox()
        self._bg_max.setRange(0.001, 100.0)
        self._bg_max.setValue(1.2)
        self._bg_max.setDecimals(3)
        for widget in (self._bg_min, self._bg_max):
            widget.valueChanged.connect(lambda _value: self.config_changed.emit())
        bg_row.addWidget(self._bg_min)
        bg_row.addWidget(QLabel("~"))
        bg_row.addWidget(self._bg_max)
        bg_w = QWidget()
        bg_w.setLayout(bg_row)
        layout.addRow("Background \u03c3:", bg_w)

        self._noise_spin = QDoubleSpinBox()
        self._noise_spin.setRange(0.0, 1.0)
        self._noise_spin.setValue(0.0)
        self._noise_spin.setDecimals(4)
        self._noise_spin.valueChanged.connect(lambda _value: self.config_changed.emit())
        layout.addRow("Noise level:", self._noise_spin)

    def get_config(self) -> dict:
        shapes = []
        if self._circle_check.isChecked():
            shapes.append("circle")
        if self._ellipse_check.isChecked():
            shapes.append("ellipse")
        if self._rect_check.isChecked():
            shapes.append("rectangle")
        if not shapes:
            shapes = ["circle"]
        return {
            "n_inhomogeneities_min": self._n_min_spin.value(),
            "n_inhomogeneities_max": self._n_max_spin.value(),
            "shapes": shapes,
            "position_min": self._pos_min.value(),
            "position_max": self._pos_max.value(),
            "size_min": self._size_min.value(),
            "size_max": self._size_max.value(),
            "conductivity_min": self._cond_min.value(),
            "conductivity_max": self._cond_max.value(),
            "background_conductivity_min": self._bg_min.value(),
            "background_conductivity_max": self._bg_max.value(),
            "noise_level": self._noise_spin.value(),
        }

    def set_config(self, config: dict) -> None:
        widgets = (
            self._circle_check,
            self._ellipse_check,
            self._rect_check,
            self._n_min_spin,
            self._n_max_spin,
            self._pos_min,
            self._pos_max,
            self._size_min,
            self._size_max,
            self._cond_min,
            self._cond_max,
            self._bg_min,
            self._bg_max,
            self._noise_spin,
        )
        blockers = [widget.blockSignals(True) for widget in widgets]
        try:
            shapes = {str(shape).lower() for shape in config.get("shapes", ["circle"])}
            self._circle_check.setChecked("circle" in shapes)
            self._ellipse_check.setChecked("ellipse" in shapes)
            self._rect_check.setChecked("rectangle" in shapes)
            self._n_min_spin.setValue(int(config.get("n_inhomogeneities_min", 1)))
            self._n_max_spin.setValue(int(config.get("n_inhomogeneities_max", 3)))
            self._pos_min.setValue(float(config.get("position_min", -0.7)))
            self._pos_max.setValue(float(config.get("position_max", 0.7)))
            self._size_min.setValue(float(config.get("size_min", 0.05)))
            self._size_max.setValue(float(config.get("size_max", 0.3)))
            self._cond_min.setValue(float(config.get("conductivity_min", 0.5)))
            self._cond_max.setValue(float(config.get("conductivity_max", 3.0)))
            self._bg_min.setValue(float(config.get("background_conductivity_min", 0.8)))
            self._bg_max.setValue(float(config.get("background_conductivity_max", 1.2)))
            self._noise_spin.setValue(float(config.get("noise_level", 0.0)))
        finally:
            for widget, blocked in zip(widgets, blockers):
                widget.blockSignals(blocked)
        self.config_changed.emit()


class DatasetRunPanel(QGroupBox):
    """Controls output location and execution state."""

    generate_requested = Signal()
    cancel_requested = Signal()
    config_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Output & Run", parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QFormLayout(self)
        layout.setContentsMargins(10, 14, 10, 8)
        layout.setSpacing(8)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        hint = QLabel(
            "Choose where the dataset should be written, then start the batch job "
            "when the mesh and ranges look right."
        )
        hint.setWordWrap(True)
        set_hint_text(hint)
        layout.addRow(hint)

        self._n_samples_spin = QSpinBox()
        self._n_samples_spin.setRange(1, 1_000_000)
        self._n_samples_spin.setValue(1000)
        self._n_samples_spin.valueChanged.connect(lambda _value: self.config_changed.emit())
        layout.addRow("Samples:", self._n_samples_spin)

        dir_row = QHBoxLayout()
        dir_row.setContentsMargins(0, 0, 0, 0)
        dir_row.setSpacing(6)
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("Output directory...")
        self._dir_edit.textChanged.connect(lambda _text: self.config_changed.emit())
        self._dir_browse = QPushButton("Browse...")
        self._dir_browse.clicked.connect(self._browse_dir)
        set_button_role(self._dir_browse, "subtle")
        dir_row.addWidget(self._dir_edit, 1)
        dir_row.addWidget(self._dir_browse)
        dir_w = QWidget()
        dir_w.setLayout(dir_row)
        layout.addRow("Save to:", dir_w)

        self._progress_header = QLabel("Execution progress")
        set_section_header(self._progress_header)
        layout.addRow(self._progress_header)

        self._status_label = QLabel("Ready to generate.")
        set_subtle_value(self._status_label)
        self._status_label.setWordWrap(True)
        layout.addRow(self._status_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        layout.addRow(self._progress)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(8)
        self._gen_btn = QPushButton("Generate Dataset")
        self._gen_btn.clicked.connect(self.generate_requested)
        set_button_role(self._gen_btn, "primary")
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.cancel_requested)
        self._cancel_btn.setEnabled(False)
        set_button_role(self._cancel_btn, "danger")
        btn_row.addWidget(self._gen_btn)
        btn_row.addWidget(self._cancel_btn)
        layout.addRow(btn_row)

    def _browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self._dir_edit.setText(path)

    def get_config(self) -> dict:
        return {
            "n_samples": self._n_samples_spin.value(),
            "output_dir": self._dir_edit.text().strip(),
        }

    def set_config(self, config: dict) -> None:
        widgets = (
            self._n_samples_spin,
            self._dir_edit,
        )
        blockers = [widget.blockSignals(True) for widget in widgets]
        try:
            self._n_samples_spin.setValue(int(config.get("n_samples", 1000)))
            self._dir_edit.setText(str(config.get("output_dir", "")).strip())
        finally:
            for widget, blocked in zip(widgets, blockers):
                widget.blockSignals(blocked)
        self.config_changed.emit()

    def set_progress(self, current: int, total: int) -> None:
        self._progress.setMaximum(max(total, 1))
        self._progress.setValue(min(current, max(total, 1)))
        self._status_label.setText(f"Generated {current} / {total} samples.")

    def set_generating(self, running: bool) -> None:
        self._gen_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
        if not running:
            self._progress.setValue(0)
            self._status_label.setText("Ready to generate.")


class DatasetGeneratorPanel(QObject):
    """Aggregates the dataset workflow controls used by the dataset tab."""

    generate_requested = Signal()
    cancel_requested = Signal()
    config_changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._random_panel = DatasetRandomizationPanel()
        self._run_panel = DatasetRunPanel()

        self._random_panel.config_changed.connect(self.config_changed.emit)
        self._run_panel.config_changed.connect(self.config_changed.emit)
        self._run_panel.generate_requested.connect(self.generate_requested.emit)
        self._run_panel.cancel_requested.connect(self.cancel_requested.emit)

        # Compatibility proxies used by tests and controller wiring.
        self._circle_check = self._random_panel._circle_check
        self._ellipse_check = self._random_panel._ellipse_check
        self._rect_check = self._random_panel._rect_check
        self._n_min_spin = self._random_panel._n_min_spin
        self._n_max_spin = self._random_panel._n_max_spin
        self._pos_min = self._random_panel._pos_min
        self._pos_max = self._random_panel._pos_max
        self._size_min = self._random_panel._size_min
        self._size_max = self._random_panel._size_max
        self._cond_min = self._random_panel._cond_min
        self._cond_max = self._random_panel._cond_max
        self._bg_min = self._random_panel._bg_min
        self._bg_max = self._random_panel._bg_max
        self._noise_spin = self._random_panel._noise_spin
        self._n_samples_spin = self._run_panel._n_samples_spin
        self._dir_edit = self._run_panel._dir_edit
        self._dir_browse = self._run_panel._dir_browse
        self._progress = self._run_panel._progress
        self._status_label = self._run_panel._status_label
        self._gen_btn = self._run_panel._gen_btn
        self._cancel_btn = self._run_panel._cancel_btn

    @property
    def randomization_panel(self) -> DatasetRandomizationPanel:
        return self._random_panel

    @property
    def run_panel(self) -> DatasetRunPanel:
        return self._run_panel

    def get_config(self) -> dict:
        config = self._random_panel.get_config()
        config.update(self._run_panel.get_config())
        return config

    def set_progress(self, current: int, total: int) -> None:
        self._run_panel.set_progress(current, total)

    def set_generating(self, running: bool) -> None:
        self._run_panel.set_generating(running)

    def set_config(self, config: dict) -> None:
        self._random_panel.set_config(config)
        self._run_panel.set_config(config)

    @staticmethod
    def default_output_dir() -> Path:
        return (Path.cwd() / "data" / "datasets").resolve()
