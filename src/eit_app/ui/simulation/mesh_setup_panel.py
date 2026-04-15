"""Mesh and electrode configuration panel for simulation."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QGroupBox, QSpinBox, QWidget

from eit_app.ui.auto_close_combo_box import AutoCloseComboBox
from eit_app.ui.theme import set_hint_text


class MeshSetupPanel(QGroupBox):
    """Configure mesh dimension, refinement, electrodes, and background conductivity."""

    config_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Mesh & Electrodes", parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QFormLayout(self)
        layout.setContentsMargins(10, 14, 10, 8)
        layout.setSpacing(8)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        from PySide6.QtWidgets import QLabel
        hint = QLabel("Configure the simulation mesh and electrode layout.")
        hint.setWordWrap(True)
        set_hint_text(hint)
        layout.addRow(hint)

        self._dim_combo = AutoCloseComboBox()
        self._dim_combo.addItems(["2D", "3D"])
        self._dim_combo.currentIndexChanged.connect(lambda _: self.config_changed.emit())
        layout.addRow("Dimension:", self._dim_combo)

        self._refine_spin = QDoubleSpinBox()
        self._refine_spin.setRange(0.01, 1.0)
        self._refine_spin.setValue(0.1)
        self._refine_spin.setDecimals(3)
        self._refine_spin.setSingleStep(0.01)
        self._refine_spin.setToolTip("Smaller values produce finer meshes (more elements)")
        self._refine_spin.valueChanged.connect(lambda _: self.config_changed.emit())
        layout.addRow("Mesh size:", self._refine_spin)

        self._n_elec_spin = QSpinBox()
        self._n_elec_spin.setRange(4, 64)
        self._n_elec_spin.setValue(16)
        self._n_elec_spin.valueChanged.connect(lambda _: self.config_changed.emit())
        layout.addRow("Electrodes:", self._n_elec_spin)

        self._bg_cond_spin = QDoubleSpinBox()
        self._bg_cond_spin.setRange(0.001, 100.0)
        self._bg_cond_spin.setValue(1.0)
        self._bg_cond_spin.setDecimals(3)
        self._bg_cond_spin.setSuffix(" S/m")
        self._bg_cond_spin.valueChanged.connect(lambda _: self.config_changed.emit())
        layout.addRow("Background \u03c3:", self._bg_cond_spin)

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
