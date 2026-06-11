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

from eit_app.i18n import t, translator
from eit_app.ui.theme import (
    set_button_role,
    set_hint_text,
    set_section_header,
    set_subtle_value,
)
from pyeidors.runtime_paths import pyeidors_data_path


class DatasetRandomizationPanel(QGroupBox):
    """Controls the random-generation parameter ranges."""

    config_changed = Signal()

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
        # Wrap the field below its label on narrow panels so long English
        # labels never force a horizontal scrollbar (vertical scroll only).
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        set_hint_text(self._hint)
        layout.addRow(self._hint)

        self._shape_header = QLabel("")
        set_section_header(self._shape_header)
        layout.addRow(self._shape_header)

        shape_row = QHBoxLayout()
        shape_row.setContentsMargins(0, 0, 0, 0)
        shape_row.setSpacing(8)
        self._circle_check = QCheckBox("")
        self._circle_check.setChecked(True)
        self._ellipse_check = QCheckBox("")
        self._rect_check = QCheckBox("")
        for widget in (self._circle_check, self._ellipse_check, self._rect_check):
            widget.toggled.connect(lambda _checked=False: self.config_changed.emit())
            shape_row.addWidget(widget)
        shape_w = QWidget()
        shape_w.setLayout(shape_row)
        self._lbl_shapes = QLabel("")
        layout.addRow(self._lbl_shapes, shape_w)

        self._count_header = QLabel("")
        set_section_header(self._count_header)
        layout.addRow(self._count_header)

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
        self._lbl_n = QLabel("")
        layout.addRow(self._lbl_n, n_w)

        self._position_header = QLabel("")
        set_section_header(self._position_header)
        layout.addRow(self._position_header)

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
        self._lbl_position = QLabel("")
        layout.addRow(self._lbl_position, pos_w)

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
        self._lbl_size = QLabel("")
        layout.addRow(self._lbl_size, size_w)

        self._conductivity_header = QLabel("")
        set_section_header(self._conductivity_header)
        layout.addRow(self._conductivity_header)

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
        self._lbl_conductivity = QLabel("")
        layout.addRow(self._lbl_conductivity, cond_w)

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
        self._lbl_background = QLabel("")
        layout.addRow(self._lbl_background, bg_w)

        self._noise_spin = QDoubleSpinBox()
        self._noise_spin.setRange(0.0, 1.0)
        self._noise_spin.setValue(0.0)
        self._noise_spin.setDecimals(4)
        self._noise_spin.valueChanged.connect(lambda _value: self.config_changed.emit())
        self._lbl_noise = QLabel("")
        layout.addRow(self._lbl_noise, self._noise_spin)

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
            for widget, blocked in zip(widgets, blockers, strict=True):
                widget.blockSignals(blocked)
        self.config_changed.emit()

    # ── i18n ──

    def _retranslate(self) -> None:
        self.setTitle(t("dataset.random.title"))
        self._hint.setText(t("dataset.random.hint"))
        self._shape_header.setText(t("dataset.random.header.shapes"))
        self._count_header.setText(t("dataset.random.header.count"))
        self._position_header.setText(t("dataset.random.header.spatial"))
        self._conductivity_header.setText(t("dataset.random.header.conductivity"))
        self._circle_check.setText(t("dataset.random.shape.circle"))
        self._ellipse_check.setText(t("dataset.random.shape.ellipse"))
        self._rect_check.setText(t("dataset.random.shape.rectangle"))
        self._lbl_shapes.setText(t("dataset.random.shapes_label"))
        self._lbl_n.setText(t("dataset.random.n_label"))
        self._lbl_position.setText(t("dataset.random.position_label"))
        self._lbl_size.setText(t("dataset.random.size_label"))
        self._lbl_conductivity.setText(t("dataset.random.conductivity_label"))
        self._lbl_background.setText(t("dataset.random.background_label"))
        self._lbl_noise.setText(t("dataset.random.noise_label"))


class DatasetRunPanel(QGroupBox):
    """Controls output location and execution state."""

    generate_requested = Signal()
    cancel_requested = Signal()
    config_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        # Title assigned by _retranslate() so it follows the UI language.
        super().__init__("", parent)
        self._generating = False
        self._progress_cache = (0, 0)
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

        self._n_samples_spin = QSpinBox()
        self._n_samples_spin.setRange(1, 1_000_000)
        self._n_samples_spin.setValue(1000)
        self._n_samples_spin.valueChanged.connect(
            lambda _value: self.config_changed.emit()
        )
        self._lbl_samples = QLabel("")
        layout.addRow(self._lbl_samples, self._n_samples_spin)

        dir_row = QHBoxLayout()
        dir_row.setContentsMargins(0, 0, 0, 0)
        dir_row.setSpacing(6)
        self._dir_edit = QLineEdit()
        self._dir_edit.textChanged.connect(lambda _text: self.config_changed.emit())
        self._dir_browse = QPushButton("")
        self._dir_browse.clicked.connect(self._browse_dir)
        set_button_role(self._dir_browse, "subtle")
        dir_row.addWidget(self._dir_edit, 1)
        dir_row.addWidget(self._dir_browse)
        dir_w = QWidget()
        dir_w.setLayout(dir_row)
        self._lbl_save_to = QLabel("")
        layout.addRow(self._lbl_save_to, dir_w)

        self._progress_header = QLabel("")
        set_section_header(self._progress_header)
        layout.addRow(self._progress_header)

        self._status_label = QLabel("")
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
        self._gen_btn = QPushButton("")
        self._gen_btn.clicked.connect(self.generate_requested)
        set_button_role(self._gen_btn, "primary")
        self._cancel_btn = QPushButton("")
        self._cancel_btn.clicked.connect(self.cancel_requested)
        self._cancel_btn.setEnabled(False)
        set_button_role(self._cancel_btn, "danger")
        btn_row.addWidget(self._gen_btn)
        btn_row.addWidget(self._cancel_btn)
        layout.addRow(btn_row)

    def _browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, t("dataset.run.file_dialog_title")
        )
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
            for widget, blocked in zip(widgets, blockers, strict=True):
                widget.blockSignals(blocked)
        self.config_changed.emit()

    def set_progress(self, current: int, total: int) -> None:
        self._progress_cache = (current, total)
        self._progress.setMaximum(max(total, 1))
        self._progress.setValue(min(current, max(total, 1)))
        self._status_label.setText(
            t("dataset.run.status.progress", current=current, total=total)
        )

    def set_generating(self, running: bool) -> None:
        self._generating = running
        self._gen_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
        if not running:
            self._progress.setValue(0)
            self._progress_cache = (0, 0)
            self._status_label.setText(t("dataset.run.status.ready"))

    # ── i18n ──

    def _retranslate(self) -> None:
        self.setTitle(t("dataset.run.title"))
        self._hint.setText(t("dataset.run.hint"))
        self._lbl_samples.setText(t("dataset.run.samples_label"))
        self._lbl_save_to.setText(t("dataset.run.save_to_label"))
        self._dir_edit.setPlaceholderText(t("dataset.run.dir_placeholder"))
        self._dir_browse.setText(t("dataset.run.browse_button"))
        self._progress_header.setText(t("dataset.run.progress_header"))
        self._gen_btn.setText(t("dataset.run.generate_button"))
        self._cancel_btn.setText(t("dataset.run.cancel_button"))
        # Re-apply the dynamic status line so the progress string tracks
        # the active language when switched mid-run.
        if self._generating or self._progress_cache[1] > 0:
            current, total = self._progress_cache
            self._status_label.setText(
                t("dataset.run.status.progress", current=current, total=total)
            )
        else:
            self._status_label.setText(t("dataset.run.status.ready"))


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
        return pyeidors_data_path("datasets").resolve()
