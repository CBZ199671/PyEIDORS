"""Table-based editor for defining inhomogeneity (anomaly) shapes."""

from __future__ import annotations

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
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return t(_COLUMN_KEYS[section])
        return None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
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
        if col == 4:
            return spec.size_x
        if col == 5:
            return spec.size_y
        if col == 6:
            return spec.size_z
        if col == 7:
            return spec.conductivity
        return None

    def setData(self, index: QModelIndex, value, role=Qt.ItemDataRole.EditRole) -> bool:
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False
        spec = self._data[index.row()]
        col = index.column()
        try:
            if col == 0:
                if str(value).lower() in _SHAPES:
                    spec.shape = str(value).lower()
            elif col == 1:
                spec.center_x = float(value)
            elif col == 2:
                spec.center_y = float(value)
            elif col == 3:
                spec.center_z = float(value)
            elif col == 4:
                spec.size_x = float(value)
            elif col == 5:
                spec.size_y = float(value)
            elif col == 6:
                spec.size_z = float(value)
            elif col == 7:
                spec.conductivity = float(value)
            else:
                return False
        except (ValueError, TypeError):
            return False
        self.dataChanged.emit(index, index)
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
        self._model.dataChanged.connect(lambda *_: self.inhomogeneities_changed.emit())
        self._model.rowsInserted.connect(lambda *_: self.inhomogeneities_changed.emit())
        self._model.rowsRemoved.connect(lambda *_: self.inhomogeneities_changed.emit())

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

        self._table = QTableView()
        self._table.setModel(self._model)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self._table.verticalHeader().setDefaultSectionSize(28)
        # Column-width strategy:
        # - The 8 columns sit inside the ~280px right context pane.
        #   Header labels are single-char (X / Y / Z / 长 / 宽 / 高 / σ)
        #   with the unit hint moved to a caption line above; that lets
        #   each numeric column shrink to ~44 px without clipping.
        # - Last column (σ) stretches to absorb leftover width.
        # - Horizontal scroll falls back when the user squeezes the
        #   panel below the natural sum.
        header = self._table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        # (col_index, pixel_width)
        for col, width in (
            (0, 72),  # Shape — short word + small dropdown indicator
            (1, 44),  # X
            (2, 44),  # Y
            (3, 44),  # Z (3D only)
            (4, 44),  # 长 / L (size X)
            (5, 44),  # 宽 / W (size Y)
            (6, 44),  # 高 / H (3D only)
            # Last column (σ) is left to stretch via setStretchLastSection.
        ):
            header.resizeSection(col, width)
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

    # ── i18n ──

    def _retranslate(self) -> None:
        """Refresh all user-visible strings to the active language."""
        title_key = "sim.inhom.title_3d" if self._mesh_dimension == 3 else "sim.inhom.title_2d"
        self.setTitle(t(title_key))
        self._units_hint.setText(t("sim.inhom.units_hint"))
        button_keys = (
            _SHAPE_BUTTON_KEYS_3D if self._mesh_dimension == 3 else _SHAPE_BUTTON_KEYS_2D
        )
        for shape, btn in self._shape_buttons.items():
            btn.setText(t(button_keys[shape]))
        self._remove_btn.setText(t("sim.inhom.remove_button"))
        # Notify the view that column header labels need a repaint.
        self._model.headerDataChanged.emit(
            Qt.Orientation.Horizontal, 0, self._model.columnCount() - 1
        )

    def _add_shape(self, shape: str) -> None:
        spec = self._default_spec(shape)
        self._model.add_spec(spec)

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

    def _apply_column_visibility(self) -> None:
        show_z = self._mesh_dimension == 3
        self._table.setColumnHidden(_Z_COLUMN, not show_z)
        self._table.setColumnHidden(_SIZE_Z_COLUMN, not show_z)
