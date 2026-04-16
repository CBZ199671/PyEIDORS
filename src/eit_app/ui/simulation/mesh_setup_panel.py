"""Mesh and electrode configuration panel for simulation."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from eit_app.i18n import t, translator
from eit_app.measurement_layout import measurement_layout_from_config
from eit_app.ui.auto_close_combo_box import AutoCloseComboBox
from eit_app.ui.theme import set_hint_text, set_section_header


class MeshSetupPanel(QGroupBox):
    """Configure mesh dimension, refinement, electrodes, background conductivity,
    and the drive / measurement pattern used by both forward and inverse solvers.
    """

    config_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        # Title assigned by _retranslate() below so it follows the UI language.
        super().__init__("", parent)
        self._point_count_cache: int = 0
        self._build_ui()
        translator().language_changed.connect(self._retranslate)
        self._retranslate()
        # Emit one config_changed to initialise downstream counters.
        self._on_any_change()

    def _build_ui(self) -> None:
        # Container layout: split into "Mesh" and "Patterns" sections so the
        # two conceptual blocks are visually distinct.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 14, 10, 8)
        outer.setSpacing(10)

        # ── Mesh section ───────────────────────────────────────────────
        mesh_widget = QWidget()
        mesh_form = QFormLayout(mesh_widget)
        mesh_form.setContentsMargins(0, 0, 0, 0)
        mesh_form.setSpacing(8)
        mesh_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        set_hint_text(self._hint)
        mesh_form.addRow(self._hint)

        self._dim_combo = AutoCloseComboBox()
        self._dim_combo.addItems(["", ""])  # retranslated
        self._dim_combo.currentIndexChanged.connect(lambda _: self._on_any_change())
        self._lbl_dim = QLabel("")
        mesh_form.addRow(self._lbl_dim, self._dim_combo)

        self._refine_spin = QDoubleSpinBox()
        self._refine_spin.setRange(0.01, 1.0)
        self._refine_spin.setValue(0.1)
        self._refine_spin.setDecimals(3)
        self._refine_spin.setSingleStep(0.01)
        self._refine_spin.valueChanged.connect(lambda _: self._on_any_change())
        self._lbl_size = QLabel("")
        mesh_form.addRow(self._lbl_size, self._refine_spin)

        self._n_elec_spin = QSpinBox()
        self._n_elec_spin.setRange(4, 64)
        self._n_elec_spin.setValue(16)
        self._n_elec_spin.valueChanged.connect(lambda _: self._on_any_change())
        self._lbl_electrodes = QLabel("")
        mesh_form.addRow(self._lbl_electrodes, self._n_elec_spin)

        self._bg_cond_spin = QDoubleSpinBox()
        self._bg_cond_spin.setRange(0.001, 100.0)
        self._bg_cond_spin.setValue(1.0)
        self._bg_cond_spin.setDecimals(3)
        self._bg_cond_spin.setSuffix(" S/m")
        self._bg_cond_spin.valueChanged.connect(lambda _: self._on_any_change())
        self._lbl_conductivity = QLabel("")
        mesh_form.addRow(self._lbl_conductivity, self._bg_cond_spin)

        outer.addWidget(mesh_widget)

        # ── Drive & measurement pattern section ────────────────────────
        self._patterns_header = QLabel("")
        set_section_header(self._patterns_header)
        outer.addWidget(self._patterns_header)

        self._patterns_hint = QLabel("")
        self._patterns_hint.setWordWrap(True)
        set_hint_text(self._patterns_hint)
        outer.addWidget(self._patterns_hint)

        patterns_widget = QWidget()
        patterns_form = QFormLayout(patterns_widget)
        patterns_form.setContentsMargins(0, 0, 0, 0)
        patterns_form.setSpacing(6)
        patterns_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._stim_pattern_edit = QLineEdit("{ad}")
        self._stim_pattern_edit.setPlaceholderText("{ad}")
        self._stim_pattern_edit.editingFinished.connect(self._on_any_change)
        self._lbl_stim_pattern = QLabel("")
        patterns_form.addRow(self._lbl_stim_pattern, self._stim_pattern_edit)

        self._meas_pattern_edit = QLineEdit("{ad}")
        self._meas_pattern_edit.setPlaceholderText("{ad}")
        self._meas_pattern_edit.editingFinished.connect(self._on_any_change)
        self._lbl_meas_pattern = QLabel("")
        patterns_form.addRow(self._lbl_meas_pattern, self._meas_pattern_edit)

        self._rotate_meas_check = QCheckBox("")
        self._rotate_meas_check.setChecked(True)
        self._rotate_meas_check.toggled.connect(lambda _: self._on_any_change())
        patterns_form.addRow("", self._rotate_meas_check)

        self._use_meas_current_check = QCheckBox("")
        self._use_meas_current_check.setChecked(False)
        self._use_meas_current_check.toggled.connect(lambda _: self._on_any_change())
        patterns_form.addRow("", self._use_meas_current_check)

        self._extra_neighbors_spin = QSpinBox()
        self._extra_neighbors_spin.setRange(0, 8)
        self._extra_neighbors_spin.setValue(0)
        self._extra_neighbors_spin.valueChanged.connect(lambda _: self._on_any_change())
        self._lbl_extra_neighbors = QLabel("")
        patterns_form.addRow(self._lbl_extra_neighbors, self._extra_neighbors_spin)

        outer.addWidget(patterns_widget)

        # Live point-count preview so the user sees the effect of pattern
        # tweaks immediately (updated on every config change).
        self._point_count_label = QLabel("")
        set_hint_text(self._point_count_label)
        self._point_count_label.setStyleSheet(
            "padding: 4px 0; color: #1f5d8b; font-weight: 600;"
        )
        outer.addWidget(self._point_count_label)

    # ------------------------------------------------------------------
    # Config interface
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        stim_pattern = self._stim_pattern_edit.text().strip() or "{ad}"
        meas_pattern = self._meas_pattern_edit.text().strip() or stim_pattern
        return {
            "mesh_dimension": 2 if self._dim_combo.currentIndex() == 0 else 3,
            "mesh_refinement": self._refine_spin.value(),
            "n_electrodes": self._n_elec_spin.value(),
            "background_conductivity": self._bg_cond_spin.value(),
            "stim_pattern": stim_pattern,
            "meas_pattern": meas_pattern,
            "rotate_meas": self._rotate_meas_check.isChecked(),
            "use_meas_current": self._use_meas_current_check.isChecked(),
            "use_meas_current_next": int(self._extra_neighbors_spin.value()),
        }

    def set_config(self, config: dict) -> None:
        widgets = (
            self._dim_combo,
            self._refine_spin,
            self._n_elec_spin,
            self._bg_cond_spin,
            self._stim_pattern_edit,
            self._meas_pattern_edit,
            self._rotate_meas_check,
            self._use_meas_current_check,
            self._extra_neighbors_spin,
        )
        blockers = [widget.blockSignals(True) for widget in widgets]
        try:
            mesh_dimension = int(config.get("mesh_dimension", 2))
            self._dim_combo.setCurrentIndex(0 if mesh_dimension == 2 else 1)
            self._refine_spin.setValue(float(config.get("mesh_refinement", 0.1)))
            self._n_elec_spin.setValue(int(config.get("n_electrodes", config.get("n_elec", 16))))
            self._bg_cond_spin.setValue(float(config.get("background_conductivity", 1.0)))
            self._stim_pattern_edit.setText(str(config.get("stim_pattern", "{ad}")))
            self._meas_pattern_edit.setText(str(config.get("meas_pattern", "{ad}")))
            self._rotate_meas_check.setChecked(bool(config.get("rotate_meas", True)))
            self._use_meas_current_check.setChecked(bool(config.get("use_meas_current", False)))
            self._extra_neighbors_spin.setValue(int(config.get("use_meas_current_next", 0)))
        finally:
            for widget, blocked in zip(widgets, blockers):
                widget.blockSignals(blocked)
        self._on_any_change()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_any_change(self) -> None:
        # Recompute and cache the expected measurement point count so the
        # user can see whether their pattern matches their hardware board.
        layout = measurement_layout_from_config(self.get_config())
        self._point_count_cache = int(layout.get("points_per_frame", 0))
        self._refresh_point_count_label()
        self.config_changed.emit()

    def _refresh_point_count_label(self) -> None:
        self._point_count_label.setText(
            t("sim.mesh.point_count_hint", count=self._point_count_cache)
        )

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
        # Pattern section
        self._patterns_header.setText(t("sim.mesh.patterns_header"))
        self._patterns_hint.setText(t("sim.mesh.patterns_hint"))
        self._lbl_stim_pattern.setText(t("sim.mesh.stim_pattern_label"))
        self._lbl_meas_pattern.setText(t("sim.mesh.meas_pattern_label"))
        self._rotate_meas_check.setText(t("sim.mesh.rotate_meas_check"))
        self._use_meas_current_check.setText(t("sim.mesh.use_meas_current_check"))
        self._lbl_extra_neighbors.setText(t("sim.mesh.extra_neighbors_label"))
        self._refresh_point_count_label()
