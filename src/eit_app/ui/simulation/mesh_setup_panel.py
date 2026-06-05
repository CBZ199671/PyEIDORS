"""Mesh and electrode configuration panel for simulation."""

import math

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from eit_app.i18n import t, translator
from eit_app.measurement_layout import measurement_layout_from_config
from eit_app.models.forward_model_config import (
    INTERACTIVE_3D_DEFAULT_ELECTRODES_PER_RING,
    INTERACTIVE_3D_DEFAULT_HEIGHT,
    INTERACTIVE_3D_DEFAULT_RADIUS,
    INTERACTIVE_3D_DEFAULT_RINGS,
    format_complex_scalar,
    max_electrode_height_ratio_for_rings,
    parse_complex_scalar,
    parse_complex_scalar_list,
)
from eit_app.ui.auto_close_combo_box import AutoCloseComboBox
from eit_app.ui.theme import set_hint_text, set_section_header


DEFAULT_ELECTRODE_COVERAGE = 0.5
DEFAULT_3D_ELECTRODE_HEIGHT_RATIO = 0.2
DEFAULT_2D_DRIVE_VALUE = 1.0
DEFAULT_3D_DRIVE_VALUE_UA = 100.0
MIN_2D_DRIVE_VALUE = 1.0e-6
MIN_3D_DRIVE_VALUE_UA = 1.0e-6
UAMP_TO_AMP = 1.0e-6


class MeshSetupPanel(QGroupBox):
    """Configure mesh dimension, refinement, electrodes, background conductivity,
    and the drive / measurement pattern used by both forward and inverse solvers.
    """

    config_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        # Title assigned by _retranslate() below so it follows the UI language.
        super().__init__("", parent)
        self._point_count_cache: int = 0
        self._electrode_length_user_overridden = False
        self._default_electrode_coverage = DEFAULT_ELECTRODE_COVERAGE
        self._last_drive_dimension = 2
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
        mesh_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        # Wrap the field below its label when the panel is too narrow to
        # fit them side by side, so long English labels never force a
        # horizontal scrollbar — vertical scrolling alone stays enough.
        mesh_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        set_hint_text(self._hint)
        mesh_form.addRow(self._hint)

        self._dim_combo = AutoCloseComboBox()
        self._dim_combo.addItems(["", ""])  # retranslated
        self._dim_combo.currentIndexChanged.connect(
            lambda _: self._on_dimension_changed()
        )
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

        # Domain geometry — 2D circle radius and 3D cylinder height.
        # Both are stored in metres; the suffix on each spin makes the
        # unit explicit and keeps the mesh, electrode positions, and
        # the inhomogeneity coordinates all in the same coordinate
        # system.  The height row is greyed out for 2D meshes since
        # there is no Z extent there.
        self._radius_spin = QDoubleSpinBox()
        self._radius_spin.setRange(0.01, 10.0)
        self._radius_spin.setDecimals(4)
        self._radius_spin.setSingleStep(0.01)
        self._radius_spin.setValue(1.0)
        self._radius_spin.setSuffix(" m")
        self._radius_spin.valueChanged.connect(lambda _: self._on_any_change())
        self._lbl_radius = QLabel("")
        mesh_form.addRow(self._lbl_radius, self._radius_spin)

        self._height_spin = QDoubleSpinBox()
        self._height_spin.setRange(0.01, 10.0)
        self._height_spin.setDecimals(4)
        self._height_spin.setSingleStep(0.01)
        self._height_spin.setValue(INTERACTIVE_3D_DEFAULT_HEIGHT)
        self._height_spin.setSuffix(" m")
        self._height_spin.valueChanged.connect(lambda _: self._on_any_change())
        self._lbl_height = QLabel("")
        mesh_form.addRow(self._lbl_height, self._height_spin)

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

        self._electrode_length_spin = QDoubleSpinBox()
        self._electrode_length_spin.setRange(1.0e-6, 10.0)
        self._electrode_length_spin.setDecimals(6)
        self._electrode_length_spin.setSingleStep(0.001)
        self._electrode_length_spin.setValue(
            2.0 * math.pi * DEFAULT_ELECTRODE_COVERAGE / 16.0
        )
        self._electrode_length_spin.setSuffix(" m")
        self._electrode_length_spin.valueChanged.connect(
            self._on_electrode_length_changed
        )
        self._lbl_electrode_length = QLabel("")
        mesh_form.addRow(self._lbl_electrode_length, self._electrode_length_spin)

        self._electrode_area_spin = QDoubleSpinBox()
        self._electrode_area_spin.setRange(1.0e-8, 10.0)
        self._electrode_area_spin.setDecimals(8)
        self._electrode_area_spin.setSingleStep(0.0001)
        default_3d_length = (
            2.0
            * math.pi
            * INTERACTIVE_3D_DEFAULT_RADIUS
            * DEFAULT_ELECTRODE_COVERAGE
            / INTERACTIVE_3D_DEFAULT_ELECTRODES_PER_RING
        )
        self._electrode_area_spin.setValue(
            default_3d_length
            * INTERACTIVE_3D_DEFAULT_HEIGHT
            * DEFAULT_3D_ELECTRODE_HEIGHT_RATIO
        )
        self._electrode_area_spin.setSuffix(" m^2")
        self._electrode_area_spin.valueChanged.connect(lambda _: self._on_any_change())
        self._lbl_electrode_area = QLabel("")
        mesh_form.addRow(self._lbl_electrode_area, self._electrode_area_spin)

        self._electrode_layout_combo = AutoCloseComboBox()
        self._electrode_layout_combo.addItem("", "ring_major")
        self._electrode_layout_combo.addItem("", "zigzag")
        self._electrode_layout_combo.currentIndexChanged.connect(
            lambda _: self._on_any_change()
        )
        self._lbl_electrode_layout = QLabel("")
        mesh_form.addRow(self._lbl_electrode_layout, self._electrode_layout_combo)

        self._bg_cond_edit = QLineEdit("1")
        self._bg_cond_edit.setPlaceholderText("1 or 1+0.25j")
        self._bg_cond_edit.editingFinished.connect(self._on_any_change)
        self._lbl_conductivity = QLabel("")
        mesh_form.addRow(self._lbl_conductivity, self._bg_cond_edit)

        self._contact_impedance_edit = QLineEdit("0.01")
        self._contact_impedance_edit.setPlaceholderText("0.01 or 0.01+0.002j")
        self._contact_impedance_edit.editingFinished.connect(self._on_any_change)
        self._lbl_contact_impedance = QLabel("")
        mesh_form.addRow(self._lbl_contact_impedance, self._contact_impedance_edit)

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
        patterns_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        patterns_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._drive_value_spin = QDoubleSpinBox()
        self._drive_value_spin.setRange(MIN_2D_DRIVE_VALUE, 1.0e9)
        self._drive_value_spin.setDecimals(6)
        self._drive_value_spin.setSingleStep(1.0)
        self._drive_value_spin.setValue(DEFAULT_2D_DRIVE_VALUE)
        self._drive_value_spin.valueChanged.connect(lambda _: self._on_any_change())
        self._lbl_drive_value = QLabel("")
        patterns_form.addRow(self._lbl_drive_value, self._drive_value_spin)

        self._measurement_protocol_combo = AutoCloseComboBox()
        self._measurement_protocol_combo.addItem("", "eidors_full_3d")
        self._measurement_protocol_combo.addItem("", "layer_local_2p5d")
        self._measurement_protocol_combo.addItem("", "cross_layer_full")
        self._measurement_protocol_combo.addItem("", "hybrid_full_3d")
        self._measurement_protocol_combo.addItem("", "custom")
        self._measurement_protocol_combo.currentIndexChanged.connect(
            lambda _: self._on_protocol_changed()
        )
        self._lbl_measurement_protocol = QLabel("")
        patterns_form.addRow(
            self._lbl_measurement_protocol, self._measurement_protocol_combo
        )

        self._measurement_protocol_hint = QLabel("")
        self._measurement_protocol_hint.setWordWrap(True)
        set_hint_text(self._measurement_protocol_hint)
        patterns_form.addRow("", self._measurement_protocol_hint)

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

        self._custom_pattern_edit = QPlainTextEdit()
        self._custom_pattern_edit.setMaximumHeight(82)
        self._custom_pattern_edit.textChanged.connect(self._on_any_change)
        self._lbl_custom_pattern = QLabel("")
        patterns_form.addRow(self._lbl_custom_pattern, self._custom_pattern_edit)

        outer.addWidget(patterns_widget)

        # Live point-count preview so the user sees the effect of pattern
        # tweaks immediately (updated on every config change).  Use the
        # uiSectionHeader role so the color follows the theme (was
        # hardcoded #1f5d8b which is invisible on the dark canvas).
        self._point_count_label = QLabel("")
        set_section_header(self._point_count_label)
        self._point_count_label.setStyleSheet("padding: 4px 0;")
        outer.addWidget(self._point_count_label)

        # Re-run the enabled-state pass now that every gated widget
        # (height spin, electrode-layout combo, etc.) exists — the
        # earlier call inside the loop above runs before those widgets
        # are constructed.
        self._refresh_mesh_family_enabled()

    # ------------------------------------------------------------------
    # Config interface
    # ------------------------------------------------------------------

    @staticmethod
    def _first_float(value: object, default: float) -> float:
        if value in (None, ""):
            return float(default)
        if isinstance(value, (list, tuple)):
            if not value:
                return float(default)
            return MeshSetupPanel._first_float(value[0], default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _parse_complex_edit(
        edit: QLineEdit,
        *,
        default: complex | float,
    ) -> complex | float:
        try:
            value = parse_complex_scalar(edit.text(), default=default)
        except (TypeError, ValueError):
            edit.setProperty("invalid", True)
            edit.setStyleSheet("border: 1px solid #b00020;")
            return default
        edit.setProperty("invalid", False)
        edit.setStyleSheet("")
        return value

    @staticmethod
    def _format_complex_edit_value(
        value: object,
        *,
        default: complex | float,
    ) -> str:
        if isinstance(value, (list, tuple)):
            return "; ".join(
                format_complex_scalar(item, default=default) for item in value
            )
        return format_complex_scalar(value, default=default)

    @staticmethod
    def _parse_complex_list_edit(
        edit: QLineEdit,
        *,
        default: complex | float,
    ) -> complex | float | list[complex | float]:
        try:
            value = parse_complex_scalar_list(edit.text(), default=default)
        except (TypeError, ValueError):
            edit.setProperty("invalid", True)
            edit.setStyleSheet("border: 1px solid #b00020;")
            return default
        edit.setProperty("invalid", False)
        edit.setStyleSheet("")
        return default if value is None else value

    @staticmethod
    def _electrodes_per_circumference(
        *,
        mesh_dimension: int,
        n_electrodes: int,
        n_rings: int,
        electrode_layout: str,
    ) -> int:
        per_ring = max(int(n_electrodes), 1)
        if (
            int(mesh_dimension) == 3
            and int(n_rings) > 1
            and str(electrode_layout).strip().lower() == "zigzag"
        ):
            return per_ring * max(int(n_rings), 1)
        return per_ring

    @classmethod
    def _electrode_pitch_m(
        cls,
        *,
        radius: float,
        mesh_dimension: int,
        n_electrodes: int,
        n_rings: int,
        electrode_layout: str,
    ) -> float:
        circumference_count = cls._electrodes_per_circumference(
            mesh_dimension=mesh_dimension,
            n_electrodes=n_electrodes,
            n_rings=n_rings,
            electrode_layout=electrode_layout,
        )
        return 2.0 * math.pi * max(float(radius), 1.0e-9) / max(circumference_count, 1)

    @classmethod
    def _max_3d_electrode_area_m2(
        cls,
        *,
        radius: float,
        height: float,
        n_electrodes: int,
        n_rings: int,
        electrode_layout: str,
    ) -> float:
        pitch_m = cls._electrode_pitch_m(
            radius=radius,
            mesh_dimension=3,
            n_electrodes=n_electrodes,
            n_rings=n_rings,
            electrode_layout=electrode_layout,
        )
        electrode_length_m = pitch_m * DEFAULT_ELECTRODE_COVERAGE
        max_ratio = max_electrode_height_ratio_for_rings(n_rings)
        return max(
            electrode_length_m * max(float(height), 1.0e-12) * max_ratio, 1.0e-12
        )

    def get_config(self) -> dict:
        stim_pattern = self._stim_pattern_edit.text().strip() or "{ad}"
        meas_pattern = self._meas_pattern_edit.text().strip() or stim_pattern
        is_3d = self._dim_combo.currentIndex() == 1
        mesh_dimension = 3 if is_3d else 2
        radius = float(self._radius_spin.value())
        height = float(self._height_spin.value()) if is_3d else 0.0
        n_electrodes = int(self._n_elec_spin.value())
        n_rings = int(self._n_rings_spin.value())
        electrode_layout = str(
            self._electrode_layout_combo.currentData() or "ring_major"
        )
        pitch_m = self._electrode_pitch_m(
            radius=radius,
            mesh_dimension=mesh_dimension,
            n_electrodes=n_electrodes,
            n_rings=n_rings,
            electrode_layout=electrode_layout,
        )
        if is_3d:
            electrode_coverage = DEFAULT_ELECTRODE_COVERAGE
            electrode_length_m = pitch_m * electrode_coverage
            max_area_m2 = self._max_3d_electrode_area_m2(
                radius=radius,
                height=height,
                n_electrodes=n_electrodes,
                n_rings=n_rings,
                electrode_layout=electrode_layout,
            )
            electrode_area_m2 = min(
                max(float(self._electrode_area_spin.value()), 1.0e-12),
                max_area_m2,
            )
            electrode_height_ratio = min(
                max(
                    electrode_area_m2 / max(electrode_length_m * height, 1.0e-12),
                    1.0e-6,
                ),
                1.0,
            )
        else:
            displayed_length_m = max(float(self._electrode_length_spin.value()), 1.0e-9)
            if self._electrode_length_user_overridden:
                electrode_length_m = displayed_length_m
                electrode_coverage = min(
                    max(electrode_length_m / max(pitch_m, 1.0e-12), 1.0e-6),
                    1.0,
                )
            else:
                electrode_length_m = None
                electrode_coverage = min(
                    max(float(self._default_electrode_coverage), 1.0e-6),
                    1.0,
                )
            electrode_area_m2 = None
            electrode_height_ratio = DEFAULT_3D_ELECTRODE_HEIGHT_RATIO
        drive_mode = "total_current" if is_3d else "line_current_density"
        drive_display_value = self._clamped_drive_display_value(
            self._drive_value_spin.value(),
            mesh_dimension=mesh_dimension,
        )
        drive_value = drive_display_value
        if is_3d:
            drive_value *= UAMP_TO_AMP
        return {
            "mesh_dimension": mesh_dimension,
            "mesh_refinement": self._refine_spin.value(),
            "mesh_family": (
                str(self._mesh_family_combo.currentData() or "hex")
                if is_3d
                else "tetra"
            ),
            "radius": radius,
            # Height is meaningful only for the 3D cylinder; for 2D
            # the field is greyed out and we report 0 so downstream
            # consumers don't accidentally treat it as a real Z extent.
            "height": height,
            "n_electrodes": n_electrodes,
            "n_rings": n_rings,
            "electrode_layout": electrode_layout,
            "electrode_length_m_override": electrode_length_m,
            "electrode_coverage": electrode_coverage,
            "electrode_area_m2_override": electrode_area_m2,
            "electrode_height_ratio": electrode_height_ratio,
            "background_conductivity": self._parse_complex_edit(
                self._bg_cond_edit,
                default=1.0,
            ),
            "contact_impedance": self._parse_complex_list_edit(
                self._contact_impedance_edit,
                default=0.01,
            ),
            "measurement_protocol": str(
                self._measurement_protocol_combo.currentData() or "eidors_full_3d"
            ),
            "custom_pattern_json": self._custom_pattern_edit.toPlainText().strip(),
            "stim_pattern": stim_pattern,
            "meas_pattern": meas_pattern,
            "rotate_meas": self._rotate_meas_check.isChecked(),
            "use_meas_current": self._use_meas_current_check.isChecked(),
            "use_meas_current_next": int(self._extra_neighbors_spin.value()),
            "drive_mode": drive_mode,
            "drive_value": drive_value,
        }

    def set_config(self, config: dict) -> None:
        widgets = (
            self._dim_combo,
            self._mesh_family_combo,
            self._radius_spin,
            self._height_spin,
            self._refine_spin,
            self._n_elec_spin,
            self._n_rings_spin,
            self._electrode_length_spin,
            self._electrode_area_spin,
            self._electrode_layout_combo,
            self._bg_cond_edit,
            self._contact_impedance_edit,
            self._drive_value_spin,
            self._measurement_protocol_combo,
            self._custom_pattern_edit,
            self._stim_pattern_edit,
            self._meas_pattern_edit,
            self._rotate_meas_check,
            self._use_meas_current_check,
            self._extra_neighbors_spin,
        )
        blockers = [widget.blockSignals(True) for widget in widgets]
        try:
            mesh_dimension = int(config.get("mesh_dimension", 2))
            explicit_2d_length = (
                mesh_dimension == 2
                and "electrode_length_m_override" in config
                and config.get("electrode_length_m_override") not in (None, "")
            )
            self._electrode_length_user_overridden = explicit_2d_length
            self._default_electrode_coverage = min(
                max(
                    self._first_float(
                        config.get("electrode_coverage"),
                        DEFAULT_ELECTRODE_COVERAGE,
                    ),
                    1.0e-6,
                ),
                1.0,
            )
            self._dim_combo.setCurrentIndex(0 if mesh_dimension == 2 else 1)
            mesh_family = (
                str(
                    config.get("mesh_family", "hex" if mesh_dimension == 3 else "tetra")
                )
                .strip()
                .lower()
            )
            self._mesh_family_combo.setCurrentIndex(1 if mesh_family == "hex" else 0)
            default_radius = (
                INTERACTIVE_3D_DEFAULT_RADIUS if mesh_dimension == 3 else 1.0
            )
            self._radius_spin.setValue(float(config.get("radius", default_radius)))
            self._height_spin.setValue(
                float(config.get("height", INTERACTIVE_3D_DEFAULT_HEIGHT))
            )
            self._refine_spin.setValue(float(config.get("mesh_refinement", 0.1)))
            self._n_elec_spin.setValue(
                int(config.get("n_electrodes", config.get("n_elec", 16)))
            )
            default_rings = 2 if mesh_dimension == 3 else 1
            self._n_rings_spin.setValue(int(config.get("n_rings", default_rings)))
            self._select_combo_data(
                self._electrode_layout_combo,
                str(config.get("electrode_layout", "ring_major")),
            )
            layout = measurement_layout_from_config(
                {
                    **config,
                    "mesh_dimension": mesh_dimension,
                    "radius": float(self._radius_spin.value()),
                    "height": float(self._height_spin.value()),
                    "n_elec": int(self._n_elec_spin.value()),
                    "n_rings": int(self._n_rings_spin.value()),
                    "electrode_layout": str(
                        self._electrode_layout_combo.currentData() or "ring_major"
                    ),
                }
            )
            default_length = float(layout.get("electrode_length_m_override", 0.0))
            electrode_length = self._first_float(
                config.get("electrode_length_m_override"), default_length
            )
            self._electrode_length_spin.setValue(max(electrode_length, 1.0e-6))
            default_area = (
                default_length
                * max(float(self._height_spin.value()), 1.0e-9)
                * self._first_float(
                    config.get("electrode_height_ratio"),
                    DEFAULT_3D_ELECTRODE_HEIGHT_RATIO,
                )
            )
            electrode_area = self._first_float(
                config.get("electrode_area_m2_override"), default_area
            )
            self._electrode_area_spin.setValue(max(electrode_area, 1.0e-8))
            self._bg_cond_edit.setText(
                self._format_complex_edit_value(
                    config.get("background_conductivity", 1.0),
                    default=1.0,
                )
            )
            self._contact_impedance_edit.setText(
                self._format_complex_edit_value(
                    config.get("contact_impedance", 0.01),
                    default=0.01,
                )
            )
            self._drive_value_spin.setValue(
                self._drive_display_value_from_config(
                    config, mesh_dimension=mesh_dimension
                )
            )
            self._select_combo_data(
                self._measurement_protocol_combo,
                str(config.get("measurement_protocol", "eidors_full_3d")),
            )
            self._custom_pattern_edit.setPlainText(
                str(config.get("custom_pattern_json", ""))
            )
            self._stim_pattern_edit.setText(str(config.get("stim_pattern", "{ad}")))
            self._meas_pattern_edit.setText(str(config.get("meas_pattern", "{ad}")))
            self._rotate_meas_check.setChecked(bool(config.get("rotate_meas", True)))
            self._use_meas_current_check.setChecked(
                bool(config.get("use_meas_current", False))
            )
            self._extra_neighbors_spin.setValue(
                int(config.get("use_meas_current_next", 0))
            )
            self._last_drive_dimension = mesh_dimension
        finally:
            for widget, blocked in zip(widgets, blockers):
                widget.blockSignals(blocked)
        self._refresh_mesh_family_enabled()
        self._refresh_drive_value_units()
        self._refresh_protocol_enabled()
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
        self._sync_drive_value_for_dimension_change(
            previous_dimension=self._last_drive_dimension,
            current_dimension=3 if is_3d else 2,
        )
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
        # Auto-swap the radius default when the user toggles dimensions
        # between the canonical 1.0 m unit-disc and the 3D tank radius.
        # Custom radii (anything else) are preserved.
        current_radius = float(self._radius_spin.value())
        if is_3d and abs(current_radius - 1.0) < 1.0e-6:
            self._radius_spin.blockSignals(True)
            try:
                self._radius_spin.setValue(INTERACTIVE_3D_DEFAULT_RADIUS)
            finally:
                self._radius_spin.blockSignals(False)
        elif (not is_3d) and abs(
            current_radius - INTERACTIVE_3D_DEFAULT_RADIUS
        ) < 1.0e-6:
            self._radius_spin.blockSignals(True)
            try:
                self._radius_spin.setValue(1.0)
            finally:
                self._radius_spin.blockSignals(False)
        self._refresh_mesh_family_enabled()
        self._refresh_protocol_enabled()
        self._on_any_change()

    def _on_protocol_changed(self) -> None:
        self._refresh_protocol_enabled()
        self._refresh_protocol_hint()
        self._on_any_change()

    def _on_electrode_length_changed(self, _value: float) -> None:
        if self._dim_combo.currentIndex() != 1:
            self._electrode_length_user_overridden = True
        self._on_any_change()

    def _on_any_change(self) -> None:
        self._sync_default_2d_electrode_length_spin()
        self._refresh_protocol_enabled()
        self._clamp_3d_electrode_area_spin()
        # Recompute and cache the expected measurement point count so the
        # user can see whether their pattern matches their hardware board.
        layout = measurement_layout_from_config(self.get_config())
        self._point_count_cache = int(layout.get("points_per_frame", 0))
        self._refresh_point_count_label()
        self.config_changed.emit()

    def _sync_default_2d_electrode_length_spin(self) -> None:
        if (
            self._dim_combo.currentIndex() == 1
            or self._electrode_length_user_overridden
        ):
            return
        if not hasattr(self, "_electrode_length_spin"):
            return
        pitch_m = self._electrode_pitch_m(
            radius=float(self._radius_spin.value()),
            mesh_dimension=2,
            n_electrodes=int(self._n_elec_spin.value()),
            n_rings=1,
            electrode_layout=str(
                self._electrode_layout_combo.currentData() or "ring_major"
            ),
        )
        target = max(
            pitch_m * min(max(float(self._default_electrode_coverage), 1.0e-6), 1.0),
            self._electrode_length_spin.minimum(),
        )
        if abs(float(self._electrode_length_spin.value()) - target) <= 1.0e-9:
            return
        blocked = self._electrode_length_spin.blockSignals(True)
        try:
            self._electrode_length_spin.setValue(target)
        finally:
            self._electrode_length_spin.blockSignals(blocked)

    def _clamp_3d_electrode_area_spin(self) -> None:
        if self._dim_combo.currentIndex() != 1:
            return
        max_area = self._max_3d_electrode_area_m2(
            radius=float(self._radius_spin.value()),
            height=float(self._height_spin.value()),
            n_electrodes=int(self._n_elec_spin.value()),
            n_rings=int(self._n_rings_spin.value()),
            electrode_layout=str(
                self._electrode_layout_combo.currentData() or "ring_major"
            ),
        )
        if float(self._electrode_area_spin.value()) <= max_area + 1.0e-12:
            return
        blocked = self._electrode_area_spin.blockSignals(True)
        try:
            self._electrode_area_spin.setValue(
                max(max_area, self._electrode_area_spin.minimum())
            )
        finally:
            self._electrode_area_spin.blockSignals(blocked)

    def _refresh_mesh_family_enabled(self) -> None:
        enabled = self._dim_combo.currentIndex() == 1
        self._lbl_mesh_family.setEnabled(enabled)
        self._mesh_family_combo.setEnabled(enabled)
        # Cylinder height is meaningless for a 2D circle — grey it out
        # there to keep the form honest.
        if hasattr(self, "_lbl_height"):
            self._lbl_height.setEnabled(enabled)
            self._height_spin.setEnabled(enabled)
        if hasattr(self, "_lbl_electrode_length"):
            self._lbl_electrode_length.setEnabled(not enabled)
            self._electrode_length_spin.setEnabled(not enabled)
            self._lbl_electrode_area.setEnabled(enabled)
            self._electrode_area_spin.setEnabled(enabled)
        if not hasattr(self, "_lbl_electrode_layout"):
            return
        self._lbl_electrode_layout.setEnabled(enabled)
        self._electrode_layout_combo.setEnabled(enabled)
        # The measurement-protocol combo only exposes 3D-specific
        # patterns (eidors_full_3d / layer_local_2p5d / cross_layer_full
        # / hybrid_full_3d / custom) so it must follow the same 3D-only
        # gating as the cell-type combo above — picking 2D should
        # leave the protocol greyed out and unselectable.
        self._refresh_protocol_enabled()

    def _sync_drive_value_for_dimension_change(
        self,
        *,
        previous_dimension: int,
        current_dimension: int,
    ) -> None:
        previous = 3 if int(previous_dimension) == 3 else 2
        current = 3 if int(current_dimension) == 3 else 2
        if previous == current:
            self._refresh_drive_value_units()
            return

        old_default = (
            DEFAULT_3D_DRIVE_VALUE_UA if previous == 3 else DEFAULT_2D_DRIVE_VALUE
        )
        new_default = (
            DEFAULT_3D_DRIVE_VALUE_UA if current == 3 else DEFAULT_2D_DRIVE_VALUE
        )
        if abs(float(self._drive_value_spin.value()) - old_default) <= 1.0e-9:
            blocked = self._drive_value_spin.blockSignals(True)
            try:
                self._drive_value_spin.setValue(new_default)
            finally:
                self._drive_value_spin.blockSignals(blocked)
        self._last_drive_dimension = current
        self._refresh_drive_value_units()

    @staticmethod
    def _default_drive_display_value(*, mesh_dimension: int) -> float:
        return (
            DEFAULT_3D_DRIVE_VALUE_UA
            if int(mesh_dimension) == 3
            else DEFAULT_2D_DRIVE_VALUE
        )

    @staticmethod
    def _minimum_drive_display_value(*, mesh_dimension: int) -> float:
        return MIN_3D_DRIVE_VALUE_UA if int(mesh_dimension) == 3 else MIN_2D_DRIVE_VALUE

    @staticmethod
    def _clamped_drive_display_value(
        value: float,
        *,
        mesh_dimension: int,
    ) -> float:
        default_value = MeshSetupPanel._default_drive_display_value(
            mesh_dimension=mesh_dimension
        )
        try:
            display_value = float(value)
        except (TypeError, ValueError):
            return default_value
        if not math.isfinite(display_value) or display_value <= 0.0:
            return default_value
        return max(
            display_value,
            MeshSetupPanel._minimum_drive_display_value(mesh_dimension=mesh_dimension),
        )

    def _refresh_drive_value_units(self) -> None:
        if not hasattr(self, "_drive_value_spin"):
            return
        if self._dim_combo.currentIndex() == 1:
            self._lbl_drive_value.setText(t("sim.mesh.drive_value_3d_label"))
            self._drive_value_spin.setSuffix(" uA")
            self._drive_value_spin.setToolTip(t("sim.mesh.drive_value_3d_tooltip"))
        else:
            self._lbl_drive_value.setText(t("sim.mesh.drive_value_2d_label"))
            self._drive_value_spin.setSuffix(" A/m")
            self._drive_value_spin.setToolTip(t("sim.mesh.drive_value_2d_tooltip"))

    @staticmethod
    def _drive_display_value_from_config(
        config: dict,
        *,
        mesh_dimension: int,
    ) -> float:
        if "drive_value" not in config:
            return MeshSetupPanel._default_drive_display_value(
                mesh_dimension=mesh_dimension
            )
        try:
            drive_value = float(config.get("drive_value", DEFAULT_2D_DRIVE_VALUE))
        except (TypeError, ValueError):
            return MeshSetupPanel._default_drive_display_value(
                mesh_dimension=mesh_dimension
            )
        if not math.isfinite(drive_value) or drive_value <= 0.0:
            return MeshSetupPanel._default_drive_display_value(
                mesh_dimension=mesh_dimension
            )
        if int(mesh_dimension) == 3:
            drive_value /= UAMP_TO_AMP
        return MeshSetupPanel._clamped_drive_display_value(
            drive_value,
            mesh_dimension=mesh_dimension,
        )

    def _refresh_protocol_enabled(self) -> None:
        is_3d = self._dim_combo.currentIndex() == 1
        # Disable the protocol combo entirely in 2D mode — none of the
        # protocols apply, so leaving them clickable is misleading.
        self._lbl_measurement_protocol.setEnabled(is_3d)
        self._measurement_protocol_combo.setEnabled(is_3d)
        self._measurement_protocol_hint.setEnabled(is_3d)
        is_custom = (
            is_3d
            and str(self._measurement_protocol_combo.currentData() or "") == "custom"
        )
        self._lbl_custom_pattern.setEnabled(is_custom)
        self._custom_pattern_edit.setEnabled(is_custom)
        self._refresh_protocol_hint()

    def _refresh_protocol_hint(self) -> None:
        protocol = str(
            self._measurement_protocol_combo.currentData() or "eidors_full_3d"
        )
        self._measurement_protocol_hint.setText(
            t(f"sim.mesh.measurement_protocol_hint.{protocol}")
        )

    @staticmethod
    def _select_combo_data(combo: AutoCloseComboBox, value: str) -> None:
        target = str(value).strip().lower()
        for idx in range(combo.count()):
            if str(combo.itemData(idx)).strip().lower() == target:
                combo.setCurrentIndex(idx)
                return

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
        self._lbl_radius.setText(t("sim.mesh.radius_label"))
        self._radius_spin.setToolTip(t("sim.mesh.radius_tooltip"))
        self._lbl_height.setText(t("sim.mesh.height_label"))
        self._height_spin.setToolTip(t("sim.mesh.height_tooltip"))
        self._lbl_size.setText(t("sim.mesh.size_label"))
        self._refine_spin.setToolTip(t("sim.mesh.refinement_tooltip"))
        self._lbl_electrodes.setText(t("sim.mesh.electrodes_label"))
        self._lbl_rings.setText(t("sim.mesh.rings_label"))
        self._lbl_electrode_length.setText(t("sim.mesh.electrode_length_label"))
        self._electrode_length_spin.setToolTip(t("sim.mesh.electrode_length_tooltip"))
        self._lbl_electrode_area.setText(t("sim.mesh.electrode_area_label"))
        self._electrode_area_spin.setToolTip(t("sim.mesh.electrode_area_tooltip"))
        self._lbl_electrode_layout.setText(t("sim.mesh.electrode_layout_label"))
        self._electrode_layout_combo.setItemText(
            0, t("sim.mesh.electrode_layout.ring_major")
        )
        self._electrode_layout_combo.setItemText(
            1, t("sim.mesh.electrode_layout.zigzag")
        )
        self._lbl_conductivity.setText(t("sim.mesh.conductivity_label"))
        self._bg_cond_edit.setToolTip(t("sim.mesh.complex_admittivity_tooltip"))
        self._lbl_contact_impedance.setText(t("sim.mesh.contact_impedance_label"))
        self._contact_impedance_edit.setToolTip(t("sim.mesh.complex_impedance_tooltip"))
        # Pattern section
        self._patterns_header.setText(t("sim.mesh.patterns_header"))
        self._patterns_hint.setText(t("sim.mesh.patterns_hint"))
        self._refresh_drive_value_units()
        self._lbl_measurement_protocol.setText(t("sim.mesh.measurement_protocol_label"))
        self._measurement_protocol_combo.setItemText(
            0, t("sim.mesh.measurement_protocol.eidors_full_3d")
        )
        self._measurement_protocol_combo.setItemText(
            1, t("sim.mesh.measurement_protocol.layer_local_2p5d")
        )
        self._measurement_protocol_combo.setItemText(
            2, t("sim.mesh.measurement_protocol.cross_layer_full")
        )
        self._measurement_protocol_combo.setItemText(
            3, t("sim.mesh.measurement_protocol.hybrid_full_3d")
        )
        self._measurement_protocol_combo.setItemText(
            4, t("sim.mesh.measurement_protocol.custom")
        )
        self._refresh_protocol_hint()
        self._lbl_stim_pattern.setText(t("sim.mesh.stim_pattern_label"))
        self._lbl_meas_pattern.setText(t("sim.mesh.meas_pattern_label"))
        self._rotate_meas_check.setText(t("sim.mesh.rotate_meas_check"))
        self._use_meas_current_check.setText(t("sim.mesh.use_meas_current_check"))
        self._lbl_extra_neighbors.setText(t("sim.mesh.extra_neighbors_label"))
        self._lbl_custom_pattern.setText(t("sim.mesh.custom_pattern_label"))
        self._custom_pattern_edit.setPlaceholderText(
            t("sim.mesh.custom_pattern_placeholder")
        )
        self._refresh_point_count_label()
