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
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from eit_app.i18n import t, translator
from eit_app.ui.theme import set_button_role, set_hint_text, set_subtle_value


class _FrameTableModel(QAbstractTableModel):
    """Backing model for the frame browser table."""

    # Column translation keys — resolved dynamically in :meth:`headerData`
    # so the header text follows the current UI language.
    _COLUMN_KEYS = (
        "hw.frame_browser.column.index",
        "hw.frame_browser.column.timestamp",
        "hw.frame_browser.column.file",
    )

    def __init__(self) -> None:
        super().__init__()
        self._entries: list[dict[str, Any]] = []
        self._reference_row: int = -1

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._entries)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._COLUMN_KEYS)

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return t(self._COLUMN_KEYS[section])
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.BackgroundRole and index.row() == self._reference_row:
            from PySide6.QtGui import QColor
            return QColor("#d9e8f7")
        if role != Qt.ItemDataRole.DisplayRole:
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

    def set_reference_row(self, row: int) -> None:
        old = self._reference_row
        self._reference_row = row
        if old >= 0 and old < self.rowCount():
            self.dataChanged.emit(
                self.index(old, 0), self.index(old, self.columnCount() - 1)
            )
        if row >= 0 and row < self.rowCount():
            self.dataChanged.emit(
                self.index(row, 0), self.index(row, self.columnCount() - 1)
            )


class FrameBrowserWidget(QGroupBox):
    """Browse recorded frames and pick the reference for real-time imaging.

    During live acquisition the newest frame is always the target, so only
    the reference needs to be selected manually. If the user never clicks
    "Set as Reference", the very first captured frame of the run becomes
    the reference automatically.

    Signals:
        reference_selected: Emitted with entry dict when user picks reference.
        target_selected: Emitted with entry dict (kept for API back-compat;
            not driven by any UI button).
        frame_clicked: Emitted when a row is clicked (for quick preview).
        cleared: Emitted when the list is cleared.
    """

    reference_selected = Signal(dict)
    target_selected = Signal(dict)  # retained so existing slots still connect
    frame_clicked = Signal(dict)
    cleared = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        # Title is populated by _retranslate() so it follows the UI language.
        super().__init__("", parent)
        self._model = _FrameTableModel()
        self._build_ui()
        translator().language_changed.connect(self._retranslate)
        self._retranslate()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.setSpacing(8)

        self._hint = QLabel("")  # retranslated below
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
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._table.selectionModel().selectionChanged.connect(self._update_action_state)
        self._table.clicked.connect(self._on_row_clicked)
        layout.addWidget(self._table, 1)

        self._count_label = QLabel("")  # retranslated below
        set_subtle_value(self._count_label)
        layout.addWidget(self._count_label)

        btn_grid = QGridLayout()
        btn_grid.setContentsMargins(0, 0, 0, 0)
        btn_grid.setHorizontalSpacing(6)
        btn_grid.setVerticalSpacing(6)
        btn_grid.setColumnStretch(0, 1)
        btn_grid.setColumnStretch(1, 1)
        self._ref_btn = QPushButton("")
        self._ref_btn.clicked.connect(self._on_set_reference)
        set_button_role(self._ref_btn, "primary")
        self._clear_btn = QPushButton("")
        self._clear_btn.clicked.connect(self._on_clear)
        set_button_role(self._clear_btn, "danger")
        for button in (self._ref_btn, self._clear_btn):
            button.setMinimumWidth(0)
        btn_grid.addWidget(self._ref_btn, 0, 0, 1, 2)
        btn_grid.addWidget(self._clear_btn, 1, 0, 1, 2)
        layout.addLayout(btn_grid)
        self._update_action_state()

    def add_frame_entry(self, frame_index: int, timestamp: float, file_path: str) -> None:
        """Add a recorded frame to the browser."""
        self._model.add_entry(
            {"frame_index": frame_index, "timestamp": timestamp, "file_path": file_path}
        )
        self._update_action_state()

    def set_reference_highlight(self, row: int) -> None:
        """Highlight the given row as the current reference frame."""
        self._model.set_reference_row(row)

    def _selected_entry(self) -> dict[str, Any] | None:
        indexes = self._table.selectionModel().selectedRows()
        if not indexes:
            return None
        return self._model.get_entry(indexes[0].row())

    def _on_set_reference(self) -> None:
        entry = self._selected_entry()
        if entry:
            self.reference_selected.emit(entry)

    def _on_row_clicked(self, index: QModelIndex) -> None:
        entry = self._model.get_entry(index.row())
        if entry:
            self.frame_clicked.emit(entry)

    def _on_clear(self) -> None:
        self._model.clear()
        self._update_action_state()
        self.cleared.emit()

    def _update_action_state(self) -> None:
        has_selection = self._selected_entry() is not None
        has_rows = self._model.rowCount() > 0
        self._ref_btn.setEnabled(has_selection)
        self._clear_btn.setEnabled(has_rows)
        self._count_label.setText(
            t("hw.frame_browser.count_label", count=self._model.rowCount())
        )
        # Auto-hide the "how this works" hint once the user actually has
        # frames in the list.  New users still see the explanation when
        # the browser is empty; seasoned users don't stare at a block of
        # static teaching copy they've already read.
        self._hint.setVisible(not has_rows)

    # ── i18n ──

    def _retranslate(self) -> None:
        """Refresh all user-visible strings to the active language."""
        self.setTitle(t("hw.frame_browser.title"))
        self._hint.setText(t("hw.frame_browser.hint"))
        self._ref_btn.setText(t("hw.frame_browser.set_ref_button"))
        self._clear_btn.setText(t("hw.frame_browser.clear_button"))
        self._update_action_state()
        # Notify the view that column header labels need to be repainted.
        self._model.headerDataChanged.emit(
            Qt.Orientation.Horizontal, 0, self._model.columnCount() - 1
        )
