"""Table-based editor for defining inhomogeneity (anomaly) shapes."""

from __future__ import annotations

import math

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from eit_app.i18n import t, translator
from eit_app.models.simulation_state import InhomogeneitySpec
from eit_app.ui.theme import set_button_role, set_hint_text


_COLUMN_KEYS = (
    "sim.inhom.col.shape",
    "sim.inhom.col.x",
    "sim.inhom.col.y",
    "sim.inhom.col.z",
    "sim.inhom.col.sizex",
    "sim.inhom.col.sizey",
    "sim.inhom.col.sizez",
    "sim.inhom.col.conductivity",
)
_SHAPES = ("circle", "ellipse", "rectangle")
_SHAPE_BUTTON_KEYS_2D = {
    "circle": "sim.inhom.add_circle",
    "ellipse": "sim.inhom.add_ellipse",
    "rectangle": "sim.inhom.add_rectangle",
}
_SHAPE_BUTTON_KEYS_3D = {
    "circle": "sim.inhom.add_sphere",
    "ellipse": "sim.inhom.add_ellipsoid",
    "rectangle": "sim.inhom.add_box",
}
_Z_COLUMN = 3
_SIZE_Z_COLUMN = 6
_COLUMN_BASE_WIDTHS_2D = {
    0: 112,  # shape
    1: 78,  # X
    2: 78,  # Y
    4: 82,  # length
    5: 82,  # width
    7: 92,  # conductivity
}
_COLUMN_BASE_WIDTHS_3D = {
    0: 108,  # shape
    1: 68,  # X
    2: 68,  # Y
    3: 68,  # Z
    4: 70,  # length
    5: 70,  # width
    6: 70,  # height
    7: 88,  # conductivity
}
_COLUMN_EXTRA_WEIGHTS = {
    0: 1.2,
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 1.0,
    6: 1.0,
    7: 0.8,
}


def _sphere_radius(spec: InhomogeneitySpec) -> float:
    values = [
        abs(float(value))
        for value in (spec.size_x, spec.size_y, spec.size_z)
        if abs(float(value)) > 0.0
    ]
    return min(values) if values else 0.0


def _ellipse_inside_circle(
    *,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    domain_radius: float,
) -> bool:
    if radius_x <= 0.0 or radius_y <= 0.0:
        return True
    samples = 96
    limit = max(float(domain_radius), 0.0)
    for idx in range(samples):
        theta = 2.0 * math.pi * idx / samples
        x = float(center_x) + float(radius_x) * math.cos(theta)
        y = float(center_y) + float(radius_y) * math.sin(theta)
        if math.hypot(x, y) > limit + 1.0e-9:
            return False
    return True


def _box_xy_inside_circle(
    *,
    center_x: float,
    center_y: float,
    half_x: float,
    half_y: float,
    domain_radius: float,
) -> bool:
    limit = max(float(domain_radius), 0.0)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            x = float(center_x) + sx * float(half_x)
            y = float(center_y) + sy * float(half_y)
            if math.hypot(x, y) > limit + 1.0e-9:
                return False
    return True


def _z_extent_inside_domain(
    spec: InhomogeneitySpec,
    *,
    height: float,
    z_center: float,
) -> bool:
    half_height = abs(float(height)) * 0.5
    z_min = float(z_center) - half_height
    z_max = float(z_center) + half_height
    if spec.shape == "circle":
        half_z = _sphere_radius(spec)
    else:
        half_z = abs(float(spec.size_z))
    cz = float(spec.center_z)
    return cz - half_z >= z_min - 1.0e-9 and cz + half_z <= z_max + 1.0e-9


def inhomogeneity_boundary_violations(
    specs: list[InhomogeneitySpec],
    *,
    mesh_dimension: int,
    radius: float,
    height: float = 1.0,
    z_center: float = 0.0,
) -> list[int]:
    """Return 1-based rows whose full-size inclusions exceed the domain."""
    domain_radius = max(abs(float(radius)), 1.0e-12)
    rows: list[int] = []
    is_3d = int(mesh_dimension) == 3
    for row, spec in enumerate(specs, start=1):
        cx = float(spec.center_x)
        cy = float(spec.center_y)
        if spec.shape == "circle":
            sphere_radius = _sphere_radius(spec)
            xy_inside = math.hypot(cx, cy) + sphere_radius <= domain_radius + 1.0e-9
        elif spec.shape == "ellipse":
            xy_inside = _ellipse_inside_circle(
                center_x=cx,
                center_y=cy,
                radius_x=abs(float(spec.size_x)),
                radius_y=abs(float(spec.size_y)),
                domain_radius=domain_radius,
            )
        else:
            xy_inside = _box_xy_inside_circle(
                center_x=cx,
                center_y=cy,
                half_x=abs(float(spec.size_x)),
                half_y=abs(float(spec.size_y)),
                domain_radius=domain_radius,
            )
        z_inside = True
        if is_3d:
            z_inside = _z_extent_inside_domain(
                spec, height=float(height), z_center=float(z_center)
            )
        if not (xy_inside and z_inside):
            rows.append(row)
    return rows


class _InhomogeneityTableModel(QAbstractTableModel):
    """Editable table model for InhomogeneitySpec entries."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._data: list[InhomogeneitySpec] = []

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._data)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(_COLUMN_KEYS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if (
            role == Qt.ItemDataRole.DisplayRole
            and orientation == Qt.Orientation.Horizontal
        ):
            return t(_COLUMN_KEYS[section])
        return None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role not in (
            Qt.ItemDataRole.DisplayRole,
            Qt.ItemDataRole.EditRole,
        ):
            return None
        spec = self._data[index.row()]
        col = index.column()
        if col == 0:
            return spec.shape
        if col == 1:
            return spec.center_x
        if col == 2:
            return spec.center_y
        if col == 3:
            return spec.center_z
        if spec.shape == "circle" and col in (4, 5, 6):
            return _sphere_radius(spec) * 2.0
        if col == 4:
            return spec.size_x * 2.0
        if col == 5:
            return spec.size_y * 2.0
        if col == 6:
            return spec.size_z * 2.0
        if col == 7:
            return spec.conductivity
        return None

    def setData(self, index: QModelIndex, value, role=Qt.ItemDataRole.EditRole) -> bool:
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False
        spec = self._data[index.row()]
        col = index.column()
        changed_left = index
        changed_right = index
        try:
            if col == 0:
                if str(value).lower() in _SHAPES:
                    spec.shape = str(value).lower()
                    if spec.shape == "circle":
                        radius = _sphere_radius(spec)
                        spec.size_x = radius
                        spec.size_y = radius
                        spec.size_z = radius
                        changed_right = self.index(index.row(), 6)
            elif col == 1:
                spec.center_x = float(value)
            elif col == 2:
                spec.center_y = float(value)
            elif col == 3:
                spec.center_z = float(value)
            elif col == 4:
                size = abs(float(value)) * 0.5
                if spec.shape == "circle":
                    spec.size_x = size
                    spec.size_y = size
                    spec.size_z = size
                    changed_left = self.index(index.row(), 4)
                    changed_right = self.index(index.row(), 6)
                else:
                    spec.size_x = size
            elif col == 5:
                size = abs(float(value)) * 0.5
                if spec.shape == "circle":
                    spec.size_x = size
                    spec.size_y = size
                    spec.size_z = size
                    changed_left = self.index(index.row(), 4)
                    changed_right = self.index(index.row(), 6)
                else:
                    spec.size_y = size
            elif col == 6:
                size = abs(float(value)) * 0.5
                if spec.shape == "circle":
                    spec.size_x = size
                    spec.size_y = size
                    spec.size_z = size
                    changed_left = self.index(index.row(), 4)
                    changed_right = self.index(index.row(), 6)
                else:
                    spec.size_z = size
            elif col == 7:
                spec.conductivity = float(value)
            else:
                return False
        except (ValueError, TypeError):
            return False
        self.dataChanged.emit(changed_left, changed_right)
        return True

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        base = super().flags(index)
        return base | Qt.ItemFlag.ItemIsEditable

    def add_spec(self, spec: InhomogeneitySpec) -> None:
        row = len(self._data)
        self.beginInsertRows(QModelIndex(), row, row)
        self._data.append(spec)
        self.endInsertRows()

    def remove_row(self, row: int) -> None:
        if 0 <= row < len(self._data):
            self.beginRemoveRows(QModelIndex(), row, row)
            self._data.pop(row)
            self.endRemoveRows()

    def get_specs(self) -> list[InhomogeneitySpec]:
        return list(self._data)

    def set_specs(self, specs: list[InhomogeneitySpec]) -> None:
        self.beginResetModel()
        self._data = list(specs)
        self.endResetModel()


class InhomogeneityEditor(QGroupBox):
    """Editor for defining anomaly shapes in the simulation domain."""

    inhomogeneities_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        # Title assigned by _retranslate() so it follows the UI language.
        super().__init__("", parent)
        self._shape_buttons: dict[str, QPushButton] = {}
        self._mesh_dimension = 2
        self._domain_radius = 1.0
        self._domain_height = 1.0
        self._domain_z_center = 0.0
        self._build_ui()
        translator().language_changed.connect(self._retranslate)
        self._retranslate()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(6)

        self._model = _InhomogeneityTableModel(self)
        self._model.dataChanged.connect(lambda *_: self._on_model_changed())
        self._model.rowsInserted.connect(lambda *_: self._on_model_changed())
        self._model.rowsRemoved.connect(lambda *_: self._on_model_changed())

        # Caption above the table — single line that consolidates the
        # unit info that previously lived in every column header.  This
        # frees ~20 px per column so headers no longer clip in the
        # narrow context pane.
        self._units_hint = QLabel("")
        set_hint_text(self._units_hint)
        self._units_hint.setStyleSheet(
            (self._units_hint.styleSheet() or "") + " padding: 0 0 4px 0;"
        )
        layout.addWidget(self._units_hint)

        self._boundary_warning = QLabel("")
        self._boundary_warning.setWordWrap(True)
        self._boundary_warning.setVisible(False)
        set_hint_text(self._boundary_warning)
        self._boundary_warning.setStyleSheet(
            (self._boundary_warning.styleSheet() or "")
            + " color: #9a4f00; padding: 3px 6px; "
            + "border: 1px solid #f1c27d; border-radius: 4px; "
            + "background: #fff7e6;"
        )
        layout.addWidget(self._boundary_warning)

        self._table = QTableView()
        self._table.setModel(self._model)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(28)
        header = self._table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setMinimumSectionSize(56)
        self._table.setHorizontalScrollMode(QTableView.ScrollMode.ScrollPerPixel)
        self._apply_column_visibility()
        layout.addWidget(self._table, 1)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(6)

        for shape in _SHAPES:
            btn = QPushButton("")  # retranslated
            btn.clicked.connect(lambda checked=False, s=shape: self._add_shape(s))
            set_button_role(btn, "subtle")
            self._shape_buttons[shape] = btn
            btn_row.addWidget(btn)

        self._remove_btn = QPushButton("")  # retranslated
        self._remove_btn.clicked.connect(self._remove_selected)
        set_button_role(self._remove_btn, "danger")
        btn_row.addWidget(self._remove_btn)

        layout.addLayout(btn_row)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._resize_table_columns()

    # ── i18n ──

    def _retranslate(self) -> None:
        """Refresh all user-visible strings to the active language."""
        title_key = (
            "sim.inhom.title_3d" if self._mesh_dimension == 3 else "sim.inhom.title_2d"
        )
        self.setTitle(t(title_key))
        self._units_hint.setText(t("sim.inhom.units_hint"))
        button_keys = (
            _SHAPE_BUTTON_KEYS_3D
            if self._mesh_dimension == 3
            else _SHAPE_BUTTON_KEYS_2D
        )
        for shape, btn in self._shape_buttons.items():
            btn.setText(t(button_keys[shape]))
        self._remove_btn.setText(t("sim.inhom.remove_button"))
        # Notify the view that column header labels need a repaint.
        self._model.headerDataChanged.emit(
            Qt.Orientation.Horizontal, 0, self._model.columnCount() - 1
        )
        self._update_boundary_warning()

    def _add_shape(self, shape: str) -> None:
        spec = self._default_spec(shape)
        self._model.add_spec(spec)

    def _on_model_changed(self) -> None:
        self._update_boundary_warning()
        self.inhomogeneities_changed.emit()

    def _default_spec(self, shape: str) -> InhomogeneitySpec:
        """Create a domain-scaled default inclusion for the active dimension."""
        radius = max(abs(float(self._domain_radius)), 1.0e-9)
        height = max(abs(float(self._domain_height)), 1.0e-9)
        spec = InhomogeneitySpec(shape=shape, center_z=float(self._domain_z_center))

        if self._mesh_dimension == 3:
            sphere_radius = min(radius * 0.35, height * 0.40)
            if shape == "rectangle":
                spec.size_x = radius * 0.28
                spec.size_y = radius * 0.22
                spec.size_z = height * 0.25
            elif shape == "ellipse":
                spec.size_x = sphere_radius
                spec.size_y = radius * 0.24
                spec.size_z = height * 0.28
            else:
                spec.size_x = sphere_radius
                spec.size_y = sphere_radius
                spec.size_z = sphere_radius
            return spec

        if shape == "rectangle":
            spec.size_x = radius * 0.20
            spec.size_y = radius * 0.14
        elif shape == "ellipse":
            spec.size_x = radius * 0.25
            spec.size_y = radius * 0.16
        else:
            spec.size_x = radius * 0.25
            spec.size_y = radius * 0.25
        spec.size_z = spec.size_x
        return spec

    def _remove_selected(self) -> None:
        indexes = self._table.selectionModel().selectedRows()
        if indexes:
            self._model.remove_row(indexes[0].row())

    def get_inhomogeneities(self) -> list[InhomogeneitySpec]:
        return self._model.get_specs()

    def set_inhomogeneities(self, specs: list[InhomogeneitySpec]) -> None:
        self._model.set_specs(specs)
        self._update_boundary_warning()

    def set_domain_context(
        self,
        *,
        mesh_dimension: int,
        radius: float | None = None,
        height: float | None = None,
        z_center: float | None = None,
    ) -> None:
        """Switch between 2D area inclusions and 3D volume inclusions."""
        old_dimension = self._mesh_dimension
        self._mesh_dimension = 3 if int(mesh_dimension) == 3 else 2
        if radius is not None:
            self._domain_radius = max(abs(float(radius)), 1.0e-9)
        if height is not None:
            self._domain_height = max(abs(float(height)), 1.0e-9)
        if z_center is not None:
            self._domain_z_center = float(z_center)
        self._apply_column_visibility()
        if self._mesh_dimension != old_dimension:
            self._retranslate()
        self._update_boundary_warning()

    def _apply_column_visibility(self) -> None:
        show_z = self._mesh_dimension == 3
        self._table.setColumnHidden(_Z_COLUMN, not show_z)
        self._table.setColumnHidden(_SIZE_Z_COLUMN, not show_z)
        self._resize_table_columns()

    def _update_boundary_warning(self) -> None:
        if not hasattr(self, "_boundary_warning"):
            return
        rows = inhomogeneity_boundary_violations(
            self._model.get_specs(),
            mesh_dimension=self._mesh_dimension,
            radius=self._domain_radius,
            height=self._domain_height,
            z_center=self._domain_z_center,
        )
        if not rows:
            self._boundary_warning.clear()
            self._boundary_warning.setVisible(False)
            return
        row_text = ", ".join(str(row) for row in rows[:4])
        if len(rows) > 4:
            row_text += "..."
        self._boundary_warning.setText(
            t(
                "sim.inhom.boundary_warning",
                count=len(rows),
                rows=row_text,
            )
        )
        self._boundary_warning.setVisible(True)

    def _resize_table_columns(self) -> None:
        """Keep Step 2 columns readable without letting sigma dominate."""
        if not hasattr(self, "_table"):
            return
        base_widths = (
            _COLUMN_BASE_WIDTHS_3D
            if self._mesh_dimension == 3
            else _COLUMN_BASE_WIDTHS_2D
        )
        visible_columns = [
            col
            for col in range(self._model.columnCount())
            if not self._table.isColumnHidden(col)
        ]
        if not visible_columns:
            return

        widths = {col: int(base_widths.get(col, 70)) for col in visible_columns}
        viewport_width = max(int(self._table.viewport().width()) - 4, 0)
        natural_width = sum(widths.values())
        extra = max(0, viewport_width - natural_width)
        if extra > 0:
            weight_sum = sum(
                _COLUMN_EXTRA_WEIGHTS.get(col, 1.0) for col in visible_columns
            )
            for col in visible_columns:
                share = extra * (_COLUMN_EXTRA_WEIGHTS.get(col, 1.0) / weight_sum)
                widths[col] += int(round(share))

        header = self._table.horizontalHeader()
        for col in visible_columns:
            header.resizeSection(col, widths[col])
