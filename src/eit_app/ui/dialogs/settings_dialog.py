"""Settings dialog for reconstruction parameters and application preferences."""

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from eit_app.i18n import t, translator
from eit_app.models.app_state import ReconstructionConfig
from eit_app.ui.auto_close_combo_box import AutoCloseComboBox
from eit_app.ui.theme import set_button_role


class SettingsDialog(QDialog):
    """Application settings dialog.

    Exposes reconstruction parameters, data paths, and device defaults.
    """

    def __init__(self, config: ReconstructionConfig, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(450)
        self._config = config
        self._build_ui()
        self._load_from_config()
        translator().language_changed.connect(self._retranslate)
        self._retranslate()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Reconstruction settings
        self._recon_group = QGroupBox("")  # retranslated
        recon_layout = QFormLayout(self._recon_group)

        # Method / part values are invariant algorithm codes.
        self._method_combo = AutoCloseComboBox()
        self._method_combo.addItems(["gn-difference", "gn-absolute", "sparse-bayes"])
        self._lbl_method = QLabel("")
        recon_layout.addRow(self._lbl_method, self._method_combo)

        self._alpha_spin = QDoubleSpinBox()
        self._alpha_spin.setRange(1e-6, 1e6)
        self._alpha_spin.setDecimals(4)
        self._alpha_spin.setValue(1.0)
        self._lbl_alpha = QLabel("")
        recon_layout.addRow(self._lbl_alpha, self._alpha_spin)

        self._iter_spin = QSpinBox()
        self._iter_spin.setRange(1, 1000)
        self._iter_spin.setValue(10)
        self._lbl_iter = QLabel("")
        recon_layout.addRow(self._lbl_iter, self._iter_spin)

        self._dim_combo = AutoCloseComboBox()
        self._dim_combo.addItems(["2D", "3D"])
        self._lbl_dim = QLabel("")
        recon_layout.addRow(self._lbl_dim, self._dim_combo)

        self._refine_spin = QSpinBox()
        self._refine_spin.setRange(1, 10)
        self._refine_spin.setValue(4)
        self._lbl_refine = QLabel("")
        recon_layout.addRow(self._lbl_refine, self._refine_spin)

        self._part_combo = AutoCloseComboBox()
        self._part_combo.addItems(["real", "imag", "mag"])
        self._lbl_part = QLabel("")
        recon_layout.addRow(self._lbl_part, self._part_combo)

        layout.addWidget(self._recon_group)

        # Data paths
        self._path_group = QGroupBox("")  # retranslated
        path_layout = QFormLayout(self._path_group)

        dir_row = QHBoxLayout()
        self._output_dir = QLineEdit()
        self._browse_btn = QPushButton("")
        # Secondary helper button — matches the "subtle" role used by
        # the other Browse… buttons across the app.
        set_button_role(self._browse_btn, "subtle")
        self._browse_btn.clicked.connect(self._browse_output)
        dir_row.addWidget(self._output_dir, 1)
        dir_row.addWidget(self._browse_btn)
        self._lbl_output = QLabel("")
        path_layout.addRow(self._lbl_output, dir_row)

        layout.addWidget(self._path_group)

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
        path = QFileDialog.getExistingDirectory(self, t("hw.acquisition.file_dialog_title"))
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

    # ── i18n ──

    def _retranslate(self) -> None:
        self.setWindowTitle(t("dlg.settings.title"))
        self._recon_group.setTitle(t("dlg.settings.recon.title"))
        self._lbl_method.setText(t("dlg.settings.recon.method_label"))
        self._lbl_alpha.setText(t("dlg.settings.recon.alpha_label"))
        self._lbl_iter.setText(t("dlg.settings.recon.iter_label"))
        self._lbl_dim.setText(t("dlg.settings.recon.dim_label"))
        self._lbl_refine.setText(t("dlg.settings.recon.refine_label"))
        self._lbl_part.setText(t("dlg.settings.recon.part_label"))
        self._path_group.setTitle(t("dlg.settings.paths.title"))
        self._output_dir.setPlaceholderText(t("dlg.settings.paths.output_placeholder"))
        self._browse_btn.setText(t("dlg.settings.paths.browse_button"))
        self._lbl_output.setText(t("dlg.settings.paths.output_label"))
