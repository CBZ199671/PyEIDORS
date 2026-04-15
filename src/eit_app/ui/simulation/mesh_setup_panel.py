"""Mesh and electrode configuration panel for simulation."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QGroupBox, QLabel, QSpinBox, QWidget

from eit_app.i18n import t, translator
from eit_app.ui.auto_close_combo_box import AutoCloseComboBox
from eit_app.ui.theme import set_hint_text


class MeshSetupPanel(QGroupBox):
    """Configure mesh dimension, refinement, electrodes, and background conductivity."""

    config_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        # Title assigned by _retranslate() below so it follows the UI language.
        super().__init__("", parent)
        self._build_ui()
        translator().language_changed.connect(self._retranslate)
        self._retranslate()

    def _build_ui(self) -> None:
        layout = QFormLayout(self)
        layout.setContentsMargins(10, 14, 10, 8)
        layout.setSpacing(8)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        set_hint_text(self._hint)
        layout.addRow(self._hint)

        self._dim_combo = AutoCloseComboBox()
        self._dim_combo.addItems(["", ""])  # retranslated
        self._dim_combo.currentIndexChanged.connect(lambda _: self.config_changed.emit())
        self._lbl_dim = QLabel("")
        layout.addRow(self._lbl_dim, self._dim_combo)

        self._refine_spin = QDoubleSpinBox()
        self._refine_spin.setRange(0.01, 1.0)
        self._refine_spin.setValue(0.1)
        self._refine_spin.setDecimals(3)
        self._refine_spin.setSingleStep(0.01)
        self._refine_spin.valueChanged.connect(lambda _: self.config_changed.emit())
        self._lbl_size = QLabel("")
        layout.addRow(self._lbl_size, self._refine_spin)

        self._n_elec_spin = QSpinBox()
        self._n_elec_spin.setRange(4, 64)
        self._n_elec_spin.setValue(16)
        self._n_elec_spin.valueChanged.connect(lambda _: self.config_changed.emit())
        self._lbl_electrodes = QLabel("")
        layout.addRow(self._lbl_electrodes, self._n_elec_spin)

        self._bg_cond_spin = QDoubleSpinBox()
        self._bg_cond_spin.setRange(0.001, 100.0)
        self._bg_cond_spin.setValue(1.0)
        self._bg_cond_spin.setDecimals(3)
        self._bg_cond_spin.setSuffix(" S/m")
        self._bg_cond_spin.valueChanged.connect(lambda _: self.config_changed.emit())
        self._lbl_conductivity = QLabel("")
        layout.addRow(self._lbl_conductivity, self._bg_cond_spin)

    # ── i18n ──

    def _retranslate(self) -> None:
        """Refresh all user-visible strings to the active language."""
        self.setTitle(t("sim.mesh.title"))
        self._hint.setText(t("sim.mesh.hint"))
        self._dim_combo.setItemText(0, t("sim.mesh.dim.2d"))
        self._dim_combo.setItemText(1, t("sim.mesh.dim.3d"))
        self._lbl_dim.setText(t("sim.mesh.dimension_label"))
        self._lbl_size.setText(t("sim.mesh.size_label"))
        self._refine_spin.setToolTip(t("sim.mesh.refinement_tooltip"))
        self._lbl_electrodes.setText(t("sim.mesh.electrodes_label"))
        self._lbl_conductivity.setText(t("sim.mesh.conductivity_label"))

    def get_config(self) -> dict:
        return {
            "mesh_dimension": 2 if self._dim_combo.currentIndex() == 0 else 3,
            "mesh_refinement": self._refine_spin.value(),
            "n_electrodes": self._n_elec_spin.value(),
            "background_conductivity": self._bg_cond_spin.value(),
        }

    def set_config(self, config: dict) -> None:
        widgets = (
            self._dim_combo,
            self._refine_spin,
            self._n_elec_spin,
            self._bg_cond_spin,
        )
        blockers = [widget.blockSignals(True) for widget in widgets]
        try:
            mesh_dimension = int(config.get("mesh_dimension", 2))
            self._dim_combo.setCurrentIndex(0 if mesh_dimension == 2 else 1)
            self._refine_spin.setValue(float(config.get("mesh_refinement", 0.1)))
            self._n_elec_spin.setValue(int(config.get("n_electrodes", config.get("n_elec", 16))))
            self._bg_cond_spin.setValue(float(config.get("background_conductivity", 1.0)))
        finally:
            for widget, blocked in zip(widgets, blockers):
                widget.blockSignals(blocked)
        self.config_changed.emit()
