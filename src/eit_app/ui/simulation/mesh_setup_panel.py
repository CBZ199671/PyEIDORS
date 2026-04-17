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
from eit_app.models.forward_model_config import (
    INTERACTIVE_3D_DEFAULT_ELECTRODES_PER_RING,
    INTERACTIVE_3D_DEFAULT_RINGS,
)
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
        self._dim_combo.currentIndexChanged.connect(lambda _: self._on_dimension_changed())
        self._lbl_dim = QLabel("")
        mesh_form.addRow(self._lbl_dim, self._dim_combo)

        self._mesh_family_combo = AutoCloseComboBox()
        self._mesh_family_combo.addItem("", "tetra")
        self._mesh_family_combo.addItem("", "hex")
        # Keep the interactive 3D default fast while still allowing a
        # deliberate switch to 4-node tetrahedra.
        self._mesh_family_combo.setCurrentIndex(1)
        self._mesh_family_combo.currentIndexChanged.connect(
            lambda _: self._on_any_change()
        )
        self._lbl_mesh_family = QLabel("")
        mesh_form.addRow(self._lbl_mesh_family, self._mesh_family_combo)
        self._refresh_mesh_family_enabled()

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

        self._n_rings_spin = QSpinBox()
        self._n_rings_spin.setRange(1, 8)
        self._n_rings_spin.setValue(1)
        self._n_rings_spin.valueChanged.connect(lambda _: self._on_any_change())
        self._lbl_rings = QLabel("")
        mesh_form.addRow(self._lbl_rings, self._n_rings_spin)

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
        # tweaks immediately (updated on every config change).  Use the
        # uiSectionHeader role so the color follows the theme (was
        # hardcoded #1f5d8b which is invisible on the dark canvas).
        self._point_count_label = QLabel("")
        set_section_header(self._point_count_label)
        self._point_count_label.setStyleSheet("padding: 4px 0;")
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
            "mesh_family": (
                str(self._mesh_family_combo.currentData() or "hex")
                if self._dim_combo.currentIndex() == 1
                else "tetra"
            ),
            "n_electrodes": self._n_elec_spin.value(),
            "n_rings": self._n_rings_spin.value(),
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
            self._mesh_family_combo,
            self._refine_spin,
            self._n_elec_spin,
            self._n_rings_spin,
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
            mesh_family = str(
                config.get("mesh_family", "hex" if mesh_dimension == 3 else "tetra")
            ).strip().lower()
            self._mesh_family_combo.setCurrentIndex(1 if mesh_family == "hex" else 0)
            self._refine_spin.setValue(float(config.get("mesh_refinement", 0.1)))
            self._n_elec_spin.setValue(int(config.get("n_electrodes", config.get("n_elec", 16))))
            default_rings = 2 if mesh_dimension == 3 else 1
            self._n_rings_spin.setValue(int(config.get("n_rings", default_rings)))
            self._bg_cond_spin.setValue(float(config.get("background_conductivity", 1.0)))
            self._stim_pattern_edit.setText(str(config.get("stim_pattern", "{ad}")))
            self._meas_pattern_edit.setText(str(config.get("meas_pattern", "{ad}")))
            self._rotate_meas_check.setChecked(bool(config.get("rotate_meas", True)))
            self._use_meas_current_check.setChecked(bool(config.get("use_meas_current", False)))
            self._extra_neighbors_spin.setValue(int(config.get("use_meas_current_next", 0)))
        finally:
            for widget, blocked in zip(widgets, blockers):
                widget.blockSignals(blocked)
        self._refresh_mesh_family_enabled()
        self._on_any_change()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_dimension_changed(self) -> None:
        # Keep the GUI's default 3D case interactive: 16 total electrodes
        # arranged as 8 per ring x 2 rings, matching the fast 3D benchmark
        # geometry.  Custom electrode counts are preserved; only the default
        # 16x1/8x2 pair is auto-migrated when the user flips dimensions.
        is_3d = self._dim_combo.currentIndex() == 1
        current_elec = self._n_elec_spin.value()
        current_rings = self._n_rings_spin.value()

        target_elec = current_elec
        target_rings = current_rings
        if is_3d:
            if current_elec == 16 and current_rings == 1:
                target_elec = INTERACTIVE_3D_DEFAULT_ELECTRODES_PER_RING
                target_rings = INTERACTIVE_3D_DEFAULT_RINGS
            elif current_rings == 1:
                target_rings = INTERACTIVE_3D_DEFAULT_RINGS
        else:
            if (
                current_elec == INTERACTIVE_3D_DEFAULT_ELECTRODES_PER_RING
                and current_rings == INTERACTIVE_3D_DEFAULT_RINGS
            ):
                target_elec = 16
                target_rings = 1
            elif current_rings != 1:
                target_rings = 1

        if target_elec != current_elec or target_rings != current_rings:
            widgets = (self._n_elec_spin, self._n_rings_spin)
            blockers = [widget.blockSignals(True) for widget in widgets]
            try:
                self._n_elec_spin.setValue(target_elec)
                self._n_rings_spin.setValue(target_rings)
            finally:
                for widget, blocked in zip(widgets, blockers):
                    widget.blockSignals(blocked)
        self._refresh_mesh_family_enabled()
        self._on_any_change()

    def _on_any_change(self) -> None:
        # Recompute and cache the expected measurement point count so the
        # user can see whether their pattern matches their hardware board.
        layout = measurement_layout_from_config(self.get_config())
        self._point_count_cache = int(layout.get("points_per_frame", 0))
        self._refresh_point_count_label()
        self.config_changed.emit()

    def _refresh_mesh_family_enabled(self) -> None:
        enabled = self._dim_combo.currentIndex() == 1
        self._lbl_mesh_family.setEnabled(enabled)
        self._mesh_family_combo.setEnabled(enabled)

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
        self._lbl_mesh_family.setText(t("sim.mesh.family_label"))
        self._mesh_family_combo.setItemText(0, t("sim.mesh.family.tetra"))
        self._mesh_family_combo.setItemText(1, t("sim.mesh.family.hex"))
        self._lbl_size.setText(t("sim.mesh.size_label"))
        self._refine_spin.setToolTip(t("sim.mesh.refinement_tooltip"))
        self._lbl_electrodes.setText(t("sim.mesh.electrodes_label"))
        self._lbl_rings.setText(t("sim.mesh.rings_label"))
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
