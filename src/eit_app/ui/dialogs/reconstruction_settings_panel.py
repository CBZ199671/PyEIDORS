"""Shared advanced forward/inverse settings for database reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from eit_app.i18n import t
from eit_app.ui.auto_close_combo_box import AutoCloseComboBox
from eit_app.ui.theme import set_button_role


_OPTIONAL_FLOAT_EPS = 1.0e-15
FORWARD_SOLVER_PRESET_CHOICES = [
    "auto",
    "direct",
    "3d_gamg",
    "spd_gamg",
    "cuda_amgx",
    "complex_block_real_amgx",
]


def _decode_json_mapping(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def metadata_from_frame_entries(*entries: dict[str, Any] | None) -> dict[str, Any]:
    """Merge reconstruction-relevant metadata from selected DB frame rows."""
    merged: dict[str, Any] = {}
    row_keys = (
        "n_elec",
        "n_rings",
        "stim_pattern",
        "meas_pattern",
        "frequency_hz",
        "stim_amp_uA",
        "voltage_amp_level",
        "transport_type",
        "mea_mode",
    )
    payload_keys = ("session_metadata_json", "metadata_json", "frame_metadata_json")
    for entry in entries:
        if not entry:
            continue
        for key in payload_keys:
            merged.update(_decode_json_mapping(entry.get(key)))
        for key in row_keys:
            value = entry.get(key)
            if value not in (None, ""):
                merged[key] = value
    return merged


def metadata_from_session_folder(folder: Path | str | None) -> dict[str, Any]:
    """Read ``session_metadata.yaml`` from a batch input folder when present."""
    if folder in (None, ""):
        return {}
    root = Path(folder)
    metadata_path = root / "session_metadata.yaml"
    if not metadata_path.exists():
        return {}
    try:
        from pyeidors.data.frame_io import read_session_metadata

        data = read_session_metadata(metadata_path)
    except Exception:
        return {}
    return dict(data) if isinstance(data, dict) else {}


def _first_present(meta: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = meta.get(key)
        if value not in (None, ""):
            return value
    return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
    return bool(default)


def _optional_float(value: float) -> float | None:
    value = float(value)
    return value if value > _OPTIONAL_FLOAT_EPS else None


class ReconstructionSettingsPanel(QWidget):
    """Forward/inverse settings editor used by the database settings dialog."""

    def __init__(
        self,
        *,
        initial_metadata: dict[str, Any] | None = None,
        show_toggle: bool = True,
        expanded: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._show_toggle = bool(show_toggle)
        self._metadata: dict[str, Any] = {}
        self._build_ui()
        self.load_metadata(initial_metadata or {})
        self._toggle_btn.setVisible(self._show_toggle)
        self._set_expanded(bool(expanded) or not self._show_toggle)
        self._retranslate()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        self._toggle_btn = QPushButton("")
        self._toggle_btn.setCheckable(True)
        set_button_role(self._toggle_btn, "subtle")
        self._toggle_btn.toggled.connect(self._set_expanded)
        root.addWidget(self._toggle_btn)

        self._body = QWidget()
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(8)

        self._tabs = QTabWidget()
        self._tabs.setMinimumHeight(320)
        self._tabs.addTab(self._scroll_tab(self._build_mesh_tab()), "")
        self._tabs.addTab(self._scroll_tab(self._build_protocol_tab()), "")
        self._tabs.addTab(self._scroll_tab(self._build_solver_tab()), "")
        body_layout.addWidget(self._tabs)
        root.addWidget(self._body)

    @staticmethod
    def _scroll_tab(widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(widget)
        return scroll

    def _build_mesh_tab(self) -> QWidget:
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._mesh_dimension = AutoCloseComboBox()
        self._mesh_dimension.addItem("2D", 2)
        self._mesh_dimension.addItem("3D", 3)
        self._mesh_dimension_label = QLabel("")
        layout.addRow(self._mesh_dimension_label, self._mesh_dimension)

        self._mesh_refinement = self._double_spin(0.0001, 1000.0, 4, 0.1)
        self._mesh_refinement_label = QLabel("")
        layout.addRow(self._mesh_refinement_label, self._mesh_refinement)

        self._rm_inverse_mesh = self._double_spin(0.0, 1000.0, 4, 0.01)
        self._rm_inverse_mesh.setSpecialValueText("Auto")
        self._rm_inverse_mesh_label = QLabel("")
        layout.addRow(self._rm_inverse_mesh_label, self._rm_inverse_mesh)

        self._n_elec = QSpinBox()
        self._n_elec.setRange(1, 512)
        self._n_elec_label = QLabel("")
        layout.addRow(self._n_elec_label, self._n_elec)

        self._n_rings = QSpinBox()
        self._n_rings.setRange(1, 32)
        self._n_rings_label = QLabel("")
        layout.addRow(self._n_rings_label, self._n_rings)

        self._electrode_layout = self._editable_combo(
            ["ring_major", "layer_major", "legacy"]
        )
        self._electrode_layout_label = QLabel("")
        layout.addRow(self._electrode_layout_label, self._electrode_layout)

        self._radius = self._double_spin(1.0e-6, 1000.0, 6, 0.01)
        self._radius_label = QLabel("")
        layout.addRow(self._radius_label, self._radius)

        self._height = self._double_spin(0.0, 1000.0, 6, 0.01)
        self._height.setSpecialValueText("Auto")
        self._height_label = QLabel("")
        layout.addRow(self._height_label, self._height)

        self._geometry_scale = self._double_spin(1.0e-12, 1.0e12, 6, 0.1)
        self._geometry_scale_label = QLabel("")
        layout.addRow(self._geometry_scale_label, self._geometry_scale)

        self._electrode_coverage = self._double_spin(0.0, 1.0, 4, 0.05)
        self._electrode_coverage_label = QLabel("")
        layout.addRow(self._electrode_coverage_label, self._electrode_coverage)

        self._electrode_length = self._double_spin(0.0, 1000.0, 6, 0.001)
        self._electrode_length.setSpecialValueText("Auto")
        self._electrode_length_label = QLabel("")
        layout.addRow(self._electrode_length_label, self._electrode_length)

        self._electrode_area = self._double_spin(0.0, 1000.0, 8, 0.0001)
        self._electrode_area.setSpecialValueText("Auto")
        self._electrode_area_label = QLabel("")
        layout.addRow(self._electrode_area_label, self._electrode_area)

        self._electrode_height_ratio = self._double_spin(0.0, 1.0, 4, 0.01)
        self._electrode_height_ratio_label = QLabel("")
        layout.addRow(self._electrode_height_ratio_label, self._electrode_height_ratio)

        return tab

    def _build_protocol_tab(self) -> QWidget:
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._stim_pattern = self._editable_combo(["{ad}", "{op}", "{mono}"])
        self._stim_pattern_label = QLabel("")
        layout.addRow(self._stim_pattern_label, self._stim_pattern)

        self._meas_pattern = self._editable_combo(["{ad}", "{op}", "{mono}"])
        self._meas_pattern_label = QLabel("")
        layout.addRow(self._meas_pattern_label, self._meas_pattern)

        self._measurement_protocol = self._editable_combo(
            ["eidors_full_3d", "adjacent", "custom"]
        )
        self._measurement_protocol_label = QLabel("")
        layout.addRow(self._measurement_protocol_label, self._measurement_protocol)

        self._rotate_meas = QCheckBox("")
        layout.addRow(self._rotate_meas)

        self._use_meas_current = QCheckBox("")
        layout.addRow(self._use_meas_current)

        self._use_meas_current_next = QSpinBox()
        self._use_meas_current_next.setRange(0, 16)
        self._use_meas_current_next_label = QLabel("")
        layout.addRow(self._use_meas_current_next_label, self._use_meas_current_next)

        self._stim_direction = self._editable_combo(["ccw", "cw"])
        self._stim_direction_label = QLabel("")
        layout.addRow(self._stim_direction_label, self._stim_direction)

        self._meas_direction = self._editable_combo(["ccw", "cw"])
        self._meas_direction_label = QLabel("")
        layout.addRow(self._meas_direction_label, self._meas_direction)

        self._stim_first_positive = QCheckBox("")
        layout.addRow(self._stim_first_positive)

        self._drive_mode = self._editable_combo(
            ["auto", "total_current", "line_current_density", "normalized"]
        )
        self._drive_mode_label = QLabel("")
        layout.addRow(self._drive_mode_label, self._drive_mode)

        self._drive_value = self._double_spin(1.0e-15, 1.0e3, 12, 1.0e-6)
        self._drive_value_label = QLabel("")
        layout.addRow(self._drive_value_label, self._drive_value)

        self._contact_impedance = self._double_spin(0.0, 1.0e6, 8, 0.001)
        self._contact_impedance_label = QLabel("")
        layout.addRow(self._contact_impedance_label, self._contact_impedance)

        return tab

    def _build_solver_tab(self) -> QWidget:
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._solver_mode = self._editable_combo(["auto", "fast", "strict"])
        self._solver_mode_label = QLabel("")
        layout.addRow(self._solver_mode_label, self._solver_mode)

        self._linear_solver = self._editable_combo(
            ["auto", "dense", "pcg", "cg", "woodbury"]
        )
        self._linear_solver_label = QLabel("")
        layout.addRow(self._linear_solver_label, self._linear_solver)

        self._preconditioner = self._editable_combo(
            [
                "auto",
                "diag",
                "noser",
                "prior",
                "pmat",
                "coarse",
                "petsc-gamg",
                "cholmod",
                "pyamg",
            ]
        )
        self._preconditioner_label = QLabel("")
        layout.addRow(self._preconditioner_label, self._preconditioner)

        self._jacobian_representation = self._editable_combo(
            ["auto", "dense", "linearized", "lazy"]
        )
        self._jacobian_representation_label = QLabel("")
        layout.addRow(
            self._jacobian_representation_label, self._jacobian_representation
        )

        self._forward_solver_preset = self._editable_combo(
            FORWARD_SOLVER_PRESET_CHOICES
        )
        self._forward_solver_preset_label = QLabel("")
        layout.addRow(self._forward_solver_preset_label, self._forward_solver_preset)

        self._forward_mat_solve = self._editable_combo(["auto", "on", "off"])
        self._forward_mat_solve_label = QLabel("")
        layout.addRow(self._forward_mat_solve_label, self._forward_mat_solve)

        self._petsc_device = self._editable_combo(["auto", "cpu", "cuda"])
        self._petsc_device_label = QLabel("")
        layout.addRow(self._petsc_device_label, self._petsc_device)

        self._runtime_device = self._editable_combo(["auto", "cpu", "cuda"])
        self._runtime_device_label = QLabel("")
        layout.addRow(self._runtime_device_label, self._runtime_device)

        self._acceleration_profile = self._editable_combo(["default", "cpu", "cuda"])
        self._acceleration_profile_label = QLabel("")
        layout.addRow(self._acceleration_profile_label, self._acceleration_profile)

        return tab

    @staticmethod
    def _double_spin(
        minimum: float, maximum: float, decimals: int, step: float
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(float(minimum), float(maximum))
        spin.setDecimals(int(decimals))
        spin.setSingleStep(float(step))
        return spin

    @staticmethod
    def _editable_combo(values: list[str]) -> AutoCloseComboBox:
        combo = AutoCloseComboBox()
        combo.setEditable(True)
        combo.addItems(values)
        return combo

    def _set_expanded(self, expanded: bool) -> None:
        self._body.setVisible(bool(expanded))
        self._toggle_btn.setChecked(bool(expanded))
        if not self._show_toggle:
            return
        self._toggle_btn.setText(
            t("dlg.recon_settings.toggle_hide")
            if expanded
            else t("dlg.recon_settings.toggle_show")
        )

    def load_metadata(self, metadata: dict[str, Any]) -> None:
        self._metadata = dict(metadata or {})
        meta = self._metadata

        mesh_dim = _as_int(_first_present(meta, "mesh_dimension", "mea_mode"), 2)
        self._set_combo_data(self._mesh_dimension, 3 if mesh_dim == 3 else 2)
        self._mesh_refinement.setValue(
            _as_float(_first_present(meta, "mesh_size", "mesh_refinement"), 4.0)
        )
        self._rm_inverse_mesh.setValue(
            _as_float(_first_present(meta, "rm_inverse_mesh_size"), 0.0)
        )
        self._n_elec.setValue(_as_int(_first_present(meta, "n_elec"), 16))
        self._n_rings.setValue(_as_int(_first_present(meta, "n_rings"), 1))
        self._set_combo_text(
            self._electrode_layout,
            str(_first_present(meta, "electrode_layout", default="ring_major")),
        )
        self._radius.setValue(_as_float(_first_present(meta, "radius"), 1.0))
        self._height.setValue(
            _as_float(_first_present(meta, "mesh_height", "height"), 0.0)
        )
        self._geometry_scale.setValue(
            _as_float(_first_present(meta, "geometry_scale_to_m"), 1.0)
        )
        self._electrode_coverage.setValue(
            _as_float(_first_present(meta, "electrode_coverage"), 0.5)
        )
        self._electrode_length.setValue(
            _as_float(_first_present(meta, "electrode_length_m_override"), 0.0)
        )
        self._electrode_area.setValue(
            _as_float(_first_present(meta, "electrode_area_m2_override"), 0.0)
        )
        self._electrode_height_ratio.setValue(
            _as_float(_first_present(meta, "electrode_height_ratio"), 0.2)
        )
        self._set_combo_text(
            self._stim_pattern,
            str(_first_present(meta, "stim_pattern", default="{ad}")),
        )
        self._set_combo_text(
            self._meas_pattern,
            str(_first_present(meta, "meas_pattern", default="{ad}")),
        )
        self._set_combo_text(
            self._measurement_protocol,
            str(_first_present(meta, "measurement_protocol", default="eidors_full_3d")),
        )
        self._rotate_meas.setChecked(_as_bool(meta.get("rotate_meas"), True))
        self._use_meas_current.setChecked(_as_bool(meta.get("use_meas_current"), False))
        self._use_meas_current_next.setValue(
            _as_int(_first_present(meta, "use_meas_current_next"), 0)
        )
        self._set_combo_text(
            self._stim_direction,
            str(_first_present(meta, "stim_direction", default="ccw")),
        )
        self._set_combo_text(
            self._meas_direction,
            str(_first_present(meta, "meas_direction", default="ccw")),
        )
        self._stim_first_positive.setChecked(
            _as_bool(meta.get("stim_first_positive"), False)
        )
        self._set_combo_text(
            self._drive_mode, str(_first_present(meta, "drive_mode", default="auto"))
        )
        self._drive_value.setValue(_as_float(_first_present(meta, "drive_value"), 1.0))
        self._contact_impedance.setValue(
            _as_float(_first_present(meta, "contact_impedance"), 0.01)
        )
        self._set_combo_text(
            self._solver_mode, str(_first_present(meta, "solver_mode", default="auto"))
        )
        self._set_combo_text(
            self._linear_solver,
            str(_first_present(meta, "linear_solver", default="auto")),
        )
        self._set_combo_text(
            self._preconditioner,
            str(_first_present(meta, "preconditioner", default="auto")),
        )
        self._set_combo_text(
            self._jacobian_representation,
            str(_first_present(meta, "jacobian_representation", default="auto")),
        )
        self._set_combo_text(
            self._forward_solver_preset,
            str(_first_present(meta, "forward_solver_preset", default="auto")),
        )
        self._set_combo_text(
            self._forward_mat_solve,
            str(_first_present(meta, "forward_mat_solve", default="auto")),
        )
        self._set_combo_text(
            self._petsc_device,
            str(_first_present(meta, "petsc_device", default="auto")),
        )
        self._set_combo_text(
            self._runtime_device, str(_first_present(meta, "device", default="auto"))
        )
        self._set_combo_text(
            self._acceleration_profile,
            str(_first_present(meta, "acceleration_profile", default="default")),
        )

    def metadata(self) -> dict[str, Any]:
        mesh_refinement = float(self._mesh_refinement.value())
        height = _optional_float(self._height.value())
        rm_inverse_mesh = _optional_float(self._rm_inverse_mesh.value())
        metadata: dict[str, Any] = {
            "mesh_dimension": self.mesh_dimension(),
            "mesh_refinement": mesh_refinement,
            "mesh_size": mesh_refinement,
            "n_elec": int(self._n_elec.value()),
            "n_rings": int(self._n_rings.value()),
            "electrode_layout": self._combo_text(self._electrode_layout),
            "radius": float(self._radius.value()),
            "height": height if height is not None else 1.0,
            "mesh_height": height if height is not None else 1.0,
            "geometry_scale_to_m": float(self._geometry_scale.value()),
            "electrode_coverage": float(self._electrode_coverage.value()),
            "electrode_length_m_override": _optional_float(
                self._electrode_length.value()
            ),
            "electrode_area_m2_override": _optional_float(self._electrode_area.value()),
            "electrode_height_ratio": float(self._electrode_height_ratio.value()),
            "stim_pattern": self._combo_text(self._stim_pattern),
            "meas_pattern": self._combo_text(self._meas_pattern),
            "measurement_protocol": self._combo_text(self._measurement_protocol),
            "rotate_meas": bool(self._rotate_meas.isChecked()),
            "use_meas_current": bool(self._use_meas_current.isChecked()),
            "use_meas_current_next": int(self._use_meas_current_next.value()),
            "stim_direction": self._combo_text(self._stim_direction),
            "meas_direction": self._combo_text(self._meas_direction),
            "stim_first_positive": bool(self._stim_first_positive.isChecked()),
            "drive_mode": self._combo_text(self._drive_mode),
            "drive_value": float(self._drive_value.value()),
            "contact_impedance": float(self._contact_impedance.value()),
            "solver_mode": self._combo_text(self._solver_mode),
            "linear_solver": self._combo_text(self._linear_solver),
            "preconditioner": self._combo_text(self._preconditioner),
            "jacobian_representation": self._combo_text(self._jacobian_representation),
            "forward_solver_preset": self._combo_text(self._forward_solver_preset),
            "forward_mat_solve": self._combo_text(self._forward_mat_solve),
            "petsc_device": self._combo_text(self._petsc_device),
            "device": self._combo_text(self._runtime_device),
            "acceleration_profile": self._combo_text(self._acceleration_profile),
        }
        if rm_inverse_mesh is not None:
            metadata["rm_inverse_mesh_size"] = rm_inverse_mesh
        return metadata

    def mesh_dimension(self) -> int:
        data = self._mesh_dimension.currentData()
        return _as_int(data, 2)

    def mesh_refinement(self) -> float:
        return float(self._mesh_refinement.value())

    @staticmethod
    def _set_combo_data(combo: AutoCloseComboBox, data: Any) -> None:
        for index in range(combo.count()):
            if combo.itemData(index) == data:
                combo.setCurrentIndex(index)
                return

    @staticmethod
    def _set_combo_text(combo: AutoCloseComboBox, text: str) -> None:
        text = str(text)
        for index in range(combo.count()):
            if combo.itemText(index) == text:
                combo.setCurrentIndex(index)
                return
        combo.setCurrentText(text)

    @staticmethod
    def _combo_text(combo: AutoCloseComboBox) -> str:
        return str(combo.currentText()).strip()

    def _retranslate(self) -> None:
        self._toggle_btn.setText(
            t("dlg.recon_settings.toggle_hide")
            if self._toggle_btn.isChecked()
            else t("dlg.recon_settings.toggle_show")
        )
        self._tabs.setTabText(0, t("dlg.recon_settings.tab_mesh"))
        self._tabs.setTabText(1, t("dlg.recon_settings.tab_protocol"))
        self._tabs.setTabText(2, t("dlg.recon_settings.tab_solver"))
        self._mesh_dimension_label.setText(t("dlg.recon_settings.mesh_dimension"))
        self._mesh_refinement_label.setText(t("dlg.recon_settings.mesh_refinement"))
        self._rm_inverse_mesh_label.setText(t("dlg.recon_settings.rm_inverse_mesh"))
        self._n_elec_label.setText(t("dlg.recon_settings.n_elec"))
        self._n_rings_label.setText(t("dlg.recon_settings.n_rings"))
        self._electrode_layout_label.setText(t("dlg.recon_settings.electrode_layout"))
        self._radius_label.setText(t("dlg.recon_settings.radius"))
        self._height_label.setText(t("dlg.recon_settings.height"))
        self._geometry_scale_label.setText(t("dlg.recon_settings.geometry_scale"))
        self._electrode_coverage_label.setText(
            t("dlg.recon_settings.electrode_coverage")
        )
        self._electrode_length_label.setText(t("dlg.recon_settings.electrode_length"))
        self._electrode_area_label.setText(t("dlg.recon_settings.electrode_area"))
        self._electrode_height_ratio_label.setText(
            t("dlg.recon_settings.electrode_height_ratio")
        )
        self._stim_pattern_label.setText(t("dlg.recon_settings.stim_pattern"))
        self._meas_pattern_label.setText(t("dlg.recon_settings.meas_pattern"))
        self._measurement_protocol_label.setText(
            t("dlg.recon_settings.measurement_protocol")
        )
        self._rotate_meas.setText(t("dlg.recon_settings.rotate_meas"))
        self._use_meas_current.setText(t("dlg.recon_settings.use_meas_current"))
        self._use_meas_current_next_label.setText(
            t("dlg.recon_settings.use_meas_current_next")
        )
        self._stim_direction_label.setText(t("dlg.recon_settings.stim_direction"))
        self._meas_direction_label.setText(t("dlg.recon_settings.meas_direction"))
        self._stim_first_positive.setText(t("dlg.recon_settings.stim_first_positive"))
        self._drive_mode_label.setText(t("dlg.recon_settings.drive_mode"))
        self._drive_value_label.setText(t("dlg.recon_settings.drive_value"))
        self._contact_impedance_label.setText(t("dlg.recon_settings.contact_impedance"))
        self._solver_mode_label.setText(t("dlg.recon_settings.solver_mode"))
        self._linear_solver_label.setText(t("dlg.recon_settings.linear_solver"))
        self._preconditioner_label.setText(t("dlg.recon_settings.preconditioner"))
        self._jacobian_representation_label.setText(
            t("dlg.recon_settings.jacobian_representation")
        )
        self._forward_solver_preset_label.setText(
            t("dlg.recon_settings.forward_solver_preset")
        )
        self._forward_mat_solve_label.setText(t("dlg.recon_settings.forward_mat_solve"))
        self._petsc_device_label.setText(t("dlg.recon_settings.petsc_device"))
        self._runtime_device_label.setText(t("dlg.recon_settings.runtime_device"))
        self._acceleration_profile_label.setText(
            t("dlg.recon_settings.acceleration_profile")
        )


class ReconstructionSettingsDialog(QDialog):
    """Standalone editor for database reconstruction forward/inverse settings."""

    def __init__(
        self,
        *,
        initial_metadata: dict[str, Any] | None = None,
        reset_metadata: dict[str, Any] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumWidth(760)
        self.resize(840, 620)
        self._reset_metadata = dict(reset_metadata or initial_metadata or {})
        self._build_ui(initial_metadata or {})
        self._retranslate()

    def _build_ui(self, initial_metadata: dict[str, Any]) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 16)
        root.setSpacing(12)

        self._title_label = QLabel("")
        self._title_label.setStyleSheet("font-size: 16px; font-weight: 700;")
        root.addWidget(self._title_label)

        self._panel = ReconstructionSettingsPanel(
            initial_metadata=initial_metadata,
            show_toggle=False,
            expanded=True,
            parent=self,
        )
        root.addWidget(self._panel, 1)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 4, 0, 0)
        btn_row.setSpacing(8)

        self._reset_btn = QPushButton("")
        set_button_role(self._reset_btn, "subtle")
        self._reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(self._reset_btn)
        btn_row.addStretch()

        self._cancel_btn = QPushButton("")
        set_button_role(self._cancel_btn, "subtle")
        self._cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._cancel_btn)

        self._apply_btn = QPushButton("")
        set_button_role(self._apply_btn, "primary")
        self._apply_btn.setMinimumWidth(120)
        self._apply_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._apply_btn)

        root.addLayout(btn_row)

    def _on_reset(self) -> None:
        self._panel.load_metadata(self._reset_metadata)

    def metadata(self) -> dict[str, Any]:
        return self._panel.metadata()

    def mesh_dimension(self) -> int:
        return self._panel.mesh_dimension()

    def mesh_refinement(self) -> float:
        return self._panel.mesh_refinement()

    def _retranslate(self) -> None:
        self.setWindowTitle(t("dlg.recon_settings.dialog_title"))
        self._title_label.setText(t("dlg.recon_settings.dialog_heading"))
        self._reset_btn.setText(t("dlg.recon_settings.reset_button"))
        self._cancel_btn.setText(t("dlg.recon_settings.cancel_button"))
        self._apply_btn.setText(t("dlg.recon_settings.apply_button"))
