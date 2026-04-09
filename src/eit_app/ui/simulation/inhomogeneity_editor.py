"""Table-based editor for defining inhomogeneity (anomaly) shapes."""

from __future__ import annotations

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from eit_app.models.simulation_state import InhomogeneitySpec
from eit_app.ui.theme import set_button_role


_COLUMNS = ["Shape", "X", "Y", "Size X", "Size Y", "\u03c3 (S/m)"]
_SHAPES = ("circle", "ellipse", "rectangle")


class _InhomogeneityTableModel(QAbstractTableModel):
    """Editable table model for InhomogeneitySpec entries."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._data: list[InhomogeneitySpec] = []

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._data)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(_COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return _COLUMNS[section]
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
            return spec.size_x
        if col == 4:
            return spec.size_y
        if col == 5:
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
                spec.size_x = float(value)
            elif col == 4:
                spec.size_y = float(value)
            elif col == 5:
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
        super().__init__("Inhomogeneities", parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(6)

        self._model = _InhomogeneityTableModel(self)
        self._model.dataChanged.connect(lambda *_: self.inhomogeneities_changed.emit())
        self._model.rowsInserted.connect(lambda *_: self.inhomogeneities_changed.emit())
        self._model.rowsRemoved.connect(lambda *_: self.inhomogeneities_changed.emit())

        self._table = QTableView()
        self._table.setModel(self._model)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self._table.verticalHeader().setDefaultSectionSize(28)
        header = self._table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._table, 1)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(6)

        for shape in _SHAPES:
            btn = QPushButton(f"+ {shape.capitalize()}")
            btn.clicked.connect(lambda checked=False, s=shape: self._add_shape(s))
            set_button_role(btn, "subtle")
            btn_row.addWidget(btn)

        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._remove_selected)
        set_button_role(self._remove_btn, "danger")
        btn_row.addWidget(self._remove_btn)

        layout.addLayout(btn_row)

    def _add_shape(self, shape: str) -> None:
        spec = InhomogeneitySpec(shape=shape)
        if shape == "rectangle":
            spec.size_x = 0.15
            spec.size_y = 0.1
        elif shape == "ellipse":
            spec.size_x = 0.2
            spec.size_y = 0.15
        self._model.add_spec(spec)

    def _remove_selected(self) -> None:
        indexes = self._table.selectionModel().selectedRows()
        if indexes:
            self._model.remove_row(indexes[0].row())

    def get_inhomogeneities(self) -> list[InhomogeneitySpec]:
        return self._model.get_specs()

    def set_inhomogeneities(self, specs: list[InhomogeneitySpec]) -> None:
        self._model.set_specs(specs)
