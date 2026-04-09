"""Settings dialog for reconstruction parameters and application preferences."""

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from eit_app.models.app_state import ReconstructionConfig
from eit_app.ui.auto_close_combo_box import AutoCloseComboBox


class SettingsDialog(QDialog):
    """Application settings dialog.

    Exposes reconstruction parameters, data paths, and device defaults.
    """

    def __init__(self, config: ReconstructionConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(450)
        self._config = config
        self._build_ui()
        self._load_from_config()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Reconstruction settings
        recon_group = QGroupBox("Reconstruction")
        recon_layout = QFormLayout(recon_group)

        self._method_combo = AutoCloseComboBox()
        self._method_combo.addItems(["gn-difference", "gn-absolute", "sparse-bayes"])
        recon_layout.addRow("Method:", self._method_combo)

        self._alpha_spin = QDoubleSpinBox()
        self._alpha_spin.setRange(1e-6, 1e6)
        self._alpha_spin.setDecimals(4)
        self._alpha_spin.setValue(1.0)
        recon_layout.addRow("Regularization alpha:", self._alpha_spin)

        self._iter_spin = QSpinBox()
        self._iter_spin.setRange(1, 1000)
        self._iter_spin.setValue(10)
        recon_layout.addRow("Max iterations:", self._iter_spin)

        self._dim_combo = AutoCloseComboBox()
        self._dim_combo.addItems(["2D", "3D"])
        recon_layout.addRow("Mesh dimension:", self._dim_combo)

        self._refine_spin = QSpinBox()
        self._refine_spin.setRange(1, 10)
        self._refine_spin.setValue(4)
        recon_layout.addRow("Mesh refinement:", self._refine_spin)

        self._part_combo = AutoCloseComboBox()
        self._part_combo.addItems(["real", "imag", "mag"])
        recon_layout.addRow("Use part:", self._part_combo)

        layout.addWidget(recon_group)

        # Data paths
        path_group = QGroupBox("Data Paths")
        path_layout = QFormLayout(path_group)

        dir_row = QHBoxLayout()
        self._output_dir = QLineEdit()
        self._output_dir.setPlaceholderText("Default output directory...")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_output)
        dir_row.addWidget(self._output_dir, 1)
        dir_row.addWidget(browse_btn)
        path_layout.addRow("Output dir:", dir_row)

        layout.addWidget(path_group)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_from_config(self) -> None:
        idx = self._method_combo.findText(self._config.method)
        if idx >= 0:
            self._method_combo.setCurrentIndex(idx)
        self._alpha_spin.setValue(self._config.regularization_alpha)
        self._iter_spin.setValue(self._config.max_iterations)
        self._dim_combo.setCurrentIndex(0 if self._config.mesh_dimension == 2 else 1)
        self._refine_spin.setValue(self._config.mesh_refinement)
        idx = self._part_combo.findText(self._config.use_part)
        if idx >= 0:
            self._part_combo.setCurrentIndex(idx)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self._output_dir.setText(path)

    def _on_accept(self) -> None:
        self._config.method = self._method_combo.currentText()
        self._config.regularization_alpha = self._alpha_spin.value()
        self._config.max_iterations = self._iter_spin.value()
        self._config.mesh_dimension = 2 if self._dim_combo.currentIndex() == 0 else 3
        self._config.mesh_refinement = self._refine_spin.value()
        self._config.use_part = self._part_combo.currentText()
        self.accept()

    def get_config(self) -> ReconstructionConfig:
        return self._config
