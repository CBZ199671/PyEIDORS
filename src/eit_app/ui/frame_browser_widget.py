"""Table view of recorded frames with double-click to select as ref/target."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    Qt,
    Signal,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from eit_app.ui.theme import set_button_role, set_hint_text, set_subtle_value


class _FrameTableModel(QAbstractTableModel):
    """Backing model for the frame browser table."""

    _COLUMNS = ("Index", "Timestamp", "File")

    def __init__(self) -> None:
        super().__init__()
        self._entries: list[dict[str, Any]] = []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._entries)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._COLUMNS)

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._COLUMNS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        entry = self._entries[index.row()]
        col = index.column()
        if col == 0:
            return entry.get("frame_index", index.row())
        if col == 1:
            ts = entry.get("timestamp", 0.0)
            if isinstance(ts, (int, float)) and ts > 0:
                return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3]
            return str(ts)
        if col == 2:
            return entry.get("file_path", "")
        return None

    def add_entry(self, entry: dict[str, Any]) -> None:
        row = len(self._entries)
        self.beginInsertRows(QModelIndex(), row, row)
        self._entries.append(entry)
        self.endInsertRows()

    def clear(self) -> None:
        self.beginResetModel()
        self._entries.clear()
        self.endResetModel()

    def get_entry(self, row: int) -> dict[str, Any] | None:
        if 0 <= row < len(self._entries):
            return self._entries[row]
        return None


class FrameBrowserWidget(QGroupBox):
    """Browse recorded frames and select reference/target for reconstruction.

    Signals:
        reference_selected: Emitted with entry dict when user picks reference.
        target_selected: Emitted with entry dict when user picks target.
    """

    reference_selected = Signal(dict)
    target_selected = Signal(dict)
    cleared = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Recorded Frames", parent)
        self._model = _FrameTableModel()
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.setSpacing(8)

        self._hint = QLabel("Select a recorded frame and mark it as reference or target for difference imaging.")
        self._hint.setWordWrap(True)
        set_hint_text(self._hint)
        layout.addWidget(self._hint)

        self._table = QTableView()
        self._table.setModel(self._model)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setSortingEnabled(False)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.selectionModel().selectionChanged.connect(self._update_action_state)
        layout.addWidget(self._table)

        self._count_label = QLabel("Recorded frames: 0")
        set_subtle_value(self._count_label)
        layout.addWidget(self._count_label)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(8)
        self._ref_btn = QPushButton("Set as Reference")
        self._ref_btn.clicked.connect(self._on_set_reference)
        set_button_role(self._ref_btn, "primary")
        self._tgt_btn = QPushButton("Set as Target")
        self._tgt_btn.clicked.connect(self._on_set_target)
        set_button_role(self._tgt_btn, "subtle")
        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.clicked.connect(self._on_clear)
        set_button_role(self._clear_btn, "danger")
        btn_row.addWidget(self._ref_btn)
        btn_row.addWidget(self._tgt_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._clear_btn)
        layout.addLayout(btn_row)
        self._update_action_state()

    def add_frame_entry(self, frame_index: int, timestamp: float, file_path: str) -> None:
        """Add a recorded frame to the browser."""
        self._model.add_entry(
            {"frame_index": frame_index, "timestamp": timestamp, "file_path": file_path}
        )
        self._update_action_state()

    def _selected_entry(self) -> dict[str, Any] | None:
        indexes = self._table.selectionModel().selectedRows()
        if not indexes:
            return None
        return self._model.get_entry(indexes[0].row())

    def _on_set_reference(self) -> None:
        entry = self._selected_entry()
        if entry:
            self.reference_selected.emit(entry)

    def _on_set_target(self) -> None:
        entry = self._selected_entry()
        if entry:
            self.target_selected.emit(entry)

    def _on_clear(self) -> None:
        self._model.clear()
        self._update_action_state()
        self.cleared.emit()

    def _update_action_state(self) -> None:
        has_selection = self._selected_entry() is not None
        has_rows = self._model.rowCount() > 0
        self._ref_btn.setEnabled(has_selection)
        self._tgt_btn.setEnabled(has_selection)
        self._clear_btn.setEnabled(has_rows)
        self._count_label.setText(f"Recorded frames: {self._model.rowCount()}")
