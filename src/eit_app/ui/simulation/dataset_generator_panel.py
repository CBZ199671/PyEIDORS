"""Batch dataset generation panel for deep learning training data."""

from pathlib import Path

from PySide6.QtCore import Signal
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
    QVBoxLayout,
    QWidget,
)

from eit_app.ui.theme import set_button_role, set_hint_text, set_section_header


class DatasetGeneratorPanel(QGroupBox):
    """Controls for batch generation of EIT training data."""

    generate_requested = Signal()
    cancel_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Dataset Generator", parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QFormLayout(self)
        layout.setContentsMargins(10, 14, 10, 8)
        layout.setSpacing(8)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        hint = QLabel(
            "Generate ground truth conductivity + boundary voltage pairs "
            "for deep learning training."
        )
        hint.setWordWrap(True)
        set_hint_text(hint)
        layout.addRow(hint)

        self._n_samples_spin = QSpinBox()
        self._n_samples_spin.setRange(1, 1_000_000)
        self._n_samples_spin.setValue(1000)
        layout.addRow("Samples:", self._n_samples_spin)

        # Output directory
        dir_row = QHBoxLayout()
        dir_row.setContentsMargins(0, 0, 0, 0)
        dir_row.setSpacing(6)
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("Output directory...")
        self._dir_browse = QPushButton("...")
        self._dir_browse.setMaximumWidth(32)
        self._dir_browse.clicked.connect(self._browse_dir)
        set_button_role(self._dir_browse, "subtle")
        dir_row.addWidget(self._dir_edit, 1)
        dir_row.addWidget(self._dir_browse)
        dir_w = QWidget()
        dir_w.setLayout(dir_row)
        layout.addRow("Save to:", dir_w)

        # Shapes
        shape_header = QLabel("Random generation ranges")
        set_section_header(shape_header)
        layout.addRow(shape_header)

        shape_row = QHBoxLayout()
        shape_row.setContentsMargins(0, 0, 0, 0)
        shape_row.setSpacing(8)
        self._circle_check = QCheckBox("Circle")
        self._circle_check.setChecked(True)
        self._ellipse_check = QCheckBox("Ellipse")
        self._rect_check = QCheckBox("Rectangle")
        shape_row.addWidget(self._circle_check)
        shape_row.addWidget(self._ellipse_check)
        shape_row.addWidget(self._rect_check)
        shape_w = QWidget()
        shape_w.setLayout(shape_row)
        layout.addRow("Shapes:", shape_w)

        # N inhomogeneities range
        n_row = QHBoxLayout()
        n_row.setContentsMargins(0, 0, 0, 0)
        n_row.setSpacing(6)
        self._n_min_spin = QSpinBox()
        self._n_min_spin.setRange(0, 20)
        self._n_min_spin.setValue(1)
        self._n_max_spin = QSpinBox()
        self._n_max_spin.setRange(1, 20)
        self._n_max_spin.setValue(3)
        n_row.addWidget(self._n_min_spin)
        n_row.addWidget(QLabel("~"))
        n_row.addWidget(self._n_max_spin)
        n_w = QWidget()
        n_w.setLayout(n_row)
        layout.addRow("N inhom.:", n_w)

        # Position range
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
        pos_row.addWidget(self._pos_min)
        pos_row.addWidget(QLabel("~"))
        pos_row.addWidget(self._pos_max)
        pos_w = QWidget()
        pos_w.setLayout(pos_row)
        layout.addRow("Position:", pos_w)

        # Size range
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
        size_row.addWidget(self._size_min)
        size_row.addWidget(QLabel("~"))
        size_row.addWidget(self._size_max)
        size_w = QWidget()
        size_w.setLayout(size_row)
        layout.addRow("Size:", size_w)

        # Conductivity range
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
        cond_row.addWidget(self._cond_min)
        cond_row.addWidget(QLabel("~"))
        cond_row.addWidget(self._cond_max)
        cond_w = QWidget()
        cond_w.setLayout(cond_row)
        layout.addRow("\u03c3 range:", cond_w)

        # Background conductivity range
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
        bg_row.addWidget(self._bg_min)
        bg_row.addWidget(QLabel("~"))
        bg_row.addWidget(self._bg_max)
        bg_w = QWidget()
        bg_w.setLayout(bg_row)
        layout.addRow("Background \u03c3:", bg_w)

        # Noise
        self._noise_spin = QDoubleSpinBox()
        self._noise_spin.setRange(0.0, 1.0)
        self._noise_spin.setValue(0.0)
        self._noise_spin.setDecimals(4)
        layout.addRow("Noise level:", self._noise_spin)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        layout.addRow(self._progress)

        # Action buttons
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
            "n_samples": self._n_samples_spin.value(),
            "output_dir": self._dir_edit.text().strip(),
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

    def set_progress(self, current: int, total: int) -> None:
        self._progress.setMaximum(total)
        self._progress.setValue(current)

    def set_generating(self, running: bool) -> None:
        self._gen_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
        if not running:
            self._progress.setValue(0)
