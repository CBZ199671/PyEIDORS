"""Database (数据库) tab — browse and filter historical EIT sessions and frames."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDateEdit,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from eit_app.ui.hardware.live_plot_widget import LivePlotWidget
from eit_app.ui.theme import set_button_role, set_hint_text

log = logging.getLogger(__name__)


def _format_timestamp(ts_iso: str) -> str:
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts_iso


def _format_unix(ts: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    except Exception:
        return str(ts)


class _SessionTableModel(QAbstractTableModel):
    _COLUMNS = (
        "ID",
        "Name",
        "Started",
        "N_elec",
        "Frequency",
        "Stim (uA)",
        "Gain",
        "Frames",
    )

    def __init__(self) -> None:
        super().__init__()
        self._rows: list[dict[str, Any]] = []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._COLUMNS)

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._COLUMNS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        row = self._rows[index.row()]
        col = index.column()
        if col == 0:
            return row.get("id", "")
        if col == 1:
            return row.get("name", "")
        if col == 2:
            return _format_timestamp(row.get("started_at", ""))
        if col == 3:
            return row.get("n_elec", "") or ""
        if col == 4:
            hz = row.get("frequency_hz")
            return f"{hz} Hz" if hz else ""
        if col == 5:
            return row.get("stim_amp_uA", "") or ""
        if col == 6:
            return row.get("voltage_amp_level", "") or ""
        if col == 7:
            return row.get("frame_count", 0)
        return ""

    def set_rows(self, rows: list[dict[str, Any]]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def row_at(self, row: int) -> dict[str, Any] | None:
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None

    def upsert(self, row: dict[str, Any]) -> None:
        session_id = row.get("id")
        for i, existing in enumerate(self._rows):
            if existing.get("id") == session_id:
                self._rows[i] = {**existing, **row}
                self.dataChanged.emit(
                    self.index(i, 0), self.index(i, self.columnCount() - 1)
                )
                return
        self.beginInsertRows(QModelIndex(), 0, 0)
        self._rows.insert(0, row)
        self.endInsertRows()


class _FrameTableModel(QAbstractTableModel):
    _COLUMNS = ("Index", "Timestamp", "File")

    def __init__(self) -> None:
        super().__init__()
        self._rows: list[dict[str, Any]] = []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._COLUMNS)

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._COLUMNS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        row = self._rows[index.row()]
        col = index.column()
        if col == 0:
            return row.get("frame_index", "")
        if col == 1:
            return _format_unix(row.get("timestamp", 0.0))
        if col == 2:
            return Path(row.get("csv_path", "")).name
        return ""

    def set_rows(self, rows: list[dict[str, Any]]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def row_at(self, row: int) -> dict[str, Any] | None:
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None

    def append(self, row: dict[str, Any]) -> None:
        pos = len(self._rows)
        self.beginInsertRows(QModelIndex(), pos, pos)
        self._rows.append(row)
        self.endInsertRows()


class DatabaseTab(QWidget):
    """Historical data browser driven by DatabaseController."""

    load_as_reference_requested = Signal(dict)
    load_as_target_requested = Signal(dict)
    open_containing_folder_requested = Signal(str)

    def __init__(self, db_controller, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db_ctrl = db_controller
        self._current_session_id: int | None = None
        self._build_ui()
        self._connect_signals()
        self.refresh_sessions()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_filter_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_preview_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([280, 760, 420])
        root.addWidget(splitter)

    def _build_filter_panel(self) -> QWidget:
        box = QGroupBox("Filters")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.setSpacing(8)

        hint = QLabel("Search and filter historical sessions.")
        hint.setWordWrap(True)
        set_hint_text(hint)
        layout.addWidget(hint)

        form = QFormLayout()
        form.setSpacing(8)

        self._filter_name = QLineEdit()
        self._filter_name.setPlaceholderText("e.g. tank, test_for_gui")
        form.addRow("Name contains:", self._filter_name)

        self._filter_freq = QLineEdit()
        self._filter_freq.setPlaceholderText("e.g. 1000")
        form.addRow("Frequency (Hz):", self._filter_freq)

        self._filter_date_from = QDateEdit()
        self._filter_date_from.setCalendarPopup(True)
        self._filter_date_from.setSpecialValueText(" ")
        self._filter_date_from.setDate(self._filter_date_from.minimumDate())
        form.addRow("Date from:", self._filter_date_from)

        self._filter_date_to = QDateEdit()
        self._filter_date_to.setCalendarPopup(True)
        self._filter_date_to.setSpecialValueText(" ")
        self._filter_date_to.setDate(self._filter_date_to.minimumDate())
        form.addRow("Date to:", self._filter_date_to)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self._apply_btn = QPushButton("Apply")
        set_button_role(self._apply_btn, "primary")
        self._clear_btn = QPushButton("Clear")
        set_button_role(self._clear_btn, "subtle")
        self._refresh_btn = QPushButton("Refresh")
        set_button_role(self._refresh_btn, "subtle")
        btn_row.addWidget(self._apply_btn)
        btn_row.addWidget(self._clear_btn)
        btn_row.addWidget(self._refresh_btn)
        layout.addLayout(btn_row)

        layout.addStretch()

        self._count_label = QLabel("Sessions: 0")
        set_hint_text(self._count_label)
        layout.addWidget(self._count_label)

        self._backfill_status = QLabel("")
        set_hint_text(self._backfill_status)
        self._backfill_status.setWordWrap(True)
        layout.addWidget(self._backfill_status)

        box.setMinimumWidth(260)
        box.setMaximumWidth(360)
        return box

    def _build_center_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)

        sessions_box = QGroupBox("Sessions")
        sessions_layout = QVBoxLayout(sessions_box)
        sessions_layout.setContentsMargins(10, 14, 10, 10)
        self._session_model = _SessionTableModel()
        self._session_table = QTableView()
        self._session_table.setModel(self._session_model)
        self._session_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._session_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._session_table.setAlternatingRowColors(True)
        self._session_table.verticalHeader().setVisible(False)
        self._session_table.horizontalHeader().setStretchLastSection(True)
        self._session_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        sessions_layout.addWidget(self._session_table)

        session_actions = QHBoxLayout()
        self._open_folder_btn = QPushButton("Open folder")
        set_button_role(self._open_folder_btn, "subtle")
        self._open_folder_btn.setEnabled(False)
        session_actions.addWidget(self._open_folder_btn)
        session_actions.addStretch()
        sessions_layout.addLayout(session_actions)

        splitter.addWidget(sessions_box)

        frames_box = QGroupBox("Frames")
        frames_layout = QVBoxLayout(frames_box)
        frames_layout.setContentsMargins(10, 14, 10, 10)
        self._frame_model = _FrameTableModel()
        self._frame_table = QTableView()
        self._frame_table.setModel(self._frame_model)
        self._frame_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._frame_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._frame_table.setAlternatingRowColors(True)
        self._frame_table.verticalHeader().setVisible(False)
        self._frame_table.horizontalHeader().setStretchLastSection(True)
        frames_layout.addWidget(self._frame_table)

        frame_actions = QHBoxLayout()
        self._as_ref_btn = QPushButton("Load as Reference")
        set_button_role(self._as_ref_btn, "primary")
        self._as_ref_btn.setEnabled(False)
        self._as_tgt_btn = QPushButton("Load as Target")
        set_button_role(self._as_tgt_btn, "subtle")
        self._as_tgt_btn.setEnabled(False)
        frame_actions.addWidget(self._as_ref_btn)
        frame_actions.addWidget(self._as_tgt_btn)
        frame_actions.addStretch()
        frames_layout.addLayout(frame_actions)

        splitter.addWidget(frames_box)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)
        return container

    def _build_preview_panel(self) -> QWidget:
        box = QGroupBox("Frame preview")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(6)

        hint = QLabel("Click a frame to preview its waveform.")
        hint.setWordWrap(True)
        set_hint_text(hint)
        layout.addWidget(hint)

        self._preview_plot = LivePlotWidget()
        layout.addWidget(self._preview_plot, 1)

        box.setMinimumWidth(360)
        return box

    def _connect_signals(self) -> None:
        self._apply_btn.clicked.connect(self.refresh_sessions)
        self._clear_btn.clicked.connect(self._clear_filters)
        self._refresh_btn.clicked.connect(self.refresh_sessions)

        self._session_table.selectionModel().selectionChanged.connect(
            self._on_session_selection_changed
        )
        self._frame_table.selectionModel().selectionChanged.connect(
            self._on_frame_selection_changed
        )
        self._frame_table.clicked.connect(self._on_frame_clicked)

        self._open_folder_btn.clicked.connect(self._on_open_folder)
        self._as_ref_btn.clicked.connect(self._on_load_as_reference)
        self._as_tgt_btn.clicked.connect(self._on_load_as_target)

        self._db_ctrl.session_added.connect(self._on_session_added)
        self._db_ctrl.frame_added.connect(self._on_frame_added)
        self._db_ctrl.backfill_progress.connect(self._on_backfill_progress)
        self._db_ctrl.backfill_done.connect(self._on_backfill_done)

    def refresh_sessions(self) -> None:
        filters: dict[str, Any] = {}
        name = self._filter_name.text().strip()
        if name:
            filters["name_like"] = name
        freq_text = self._filter_freq.text().strip()
        if freq_text:
            try:
                filters["frequency_hz"] = int(freq_text)
            except ValueError:
                pass
        date_from = self._filter_date_from.date()
        if date_from != self._filter_date_from.minimumDate():
            filters["started_after"] = date_from.toString("yyyy-MM-dd")
        date_to = self._filter_date_to.date()
        if date_to != self._filter_date_to.minimumDate():
            filters["started_before"] = date_to.toString("yyyy-MM-dd") + "T23:59:59"

        sessions = self._db_ctrl.query_sessions(**filters)
        self._session_model.set_rows(sessions)
        self._count_label.setText(f"Sessions: {len(sessions)}")
        self._frame_model.set_rows([])
        self._current_session_id = None
        self._as_ref_btn.setEnabled(False)
        self._as_tgt_btn.setEnabled(False)
        self._open_folder_btn.setEnabled(False)

    def _clear_filters(self) -> None:
        self._filter_name.clear()
        self._filter_freq.clear()
        self._filter_date_from.setDate(self._filter_date_from.minimumDate())
        self._filter_date_to.setDate(self._filter_date_to.minimumDate())
        self.refresh_sessions()

    def _selected_session(self) -> dict[str, Any] | None:
        idx = self._session_table.currentIndex()
        if not idx.isValid():
            return None
        return self._session_model.row_at(idx.row())

    def _selected_frame(self) -> dict[str, Any] | None:
        idx = self._frame_table.currentIndex()
        if not idx.isValid():
            return None
        return self._frame_model.row_at(idx.row())

    def _on_session_selection_changed(self, *args) -> None:
        session = self._selected_session()
        if session is None:
            self._current_session_id = None
            self._frame_model.set_rows([])
            self._open_folder_btn.setEnabled(False)
            return
        self._current_session_id = int(session["id"])
        self._open_folder_btn.setEnabled(True)
        frames = self._db_ctrl.query_frames(self._current_session_id)
        self._frame_model.set_rows(frames)

    def _on_frame_selection_changed(self, *args) -> None:
        frame = self._selected_frame()
        enabled = frame is not None
        self._as_ref_btn.setEnabled(enabled)
        self._as_tgt_btn.setEnabled(enabled)

    def _on_frame_clicked(self, index: QModelIndex) -> None:
        row = self._frame_model.row_at(index.row())
        if row is None:
            return
        csv_path = row.get("csv_path", "")
        if not csv_path:
            return
        try:
            from pyeidors.data.frame_io import read_frame_csv
            from eit_app.models.frame_model import FrameData

            real, imag = read_frame_csv(csv_path)
            frame = FrameData(
                real=real,
                imag=imag,
                timestamp=float(row.get("timestamp", 0.0)),
                frame_index=int(row.get("frame_index", 0)),
            )
            self._preview_plot.update_frame(frame)
        except Exception as exc:
            log.warning("Failed to preview frame %s: %s", csv_path, exc)

    def _on_open_folder(self) -> None:
        session = self._selected_session()
        if session is None:
            return
        folder = str(session.get("session_dir", ""))
        if folder:
            self.open_containing_folder_requested.emit(folder)

    def _on_load_as_reference(self) -> None:
        frame = self._selected_frame()
        if frame is not None:
            self.load_as_reference_requested.emit(dict(frame))

    def _on_load_as_target(self) -> None:
        frame = self._selected_frame()
        if frame is not None:
            self.load_as_target_requested.emit(dict(frame))

    def _on_session_added(self, session_id: int, row: dict) -> None:
        row = dict(row)
        row.setdefault("frame_count", 0)
        self._session_model.upsert(row)
        self._count_label.setText(f"Sessions: {self._session_model.rowCount()}")

    def _on_frame_added(self, frame_id: int, row: dict) -> None:
        session_id = row.get("session_id")
        if session_id is not None:
            for i in range(self._session_model.rowCount()):
                srow = self._session_model.row_at(i)
                if srow and srow.get("id") == session_id:
                    new_row = dict(srow)
                    new_row["frame_count"] = int(srow.get("frame_count", 0)) + 1
                    self._session_model.upsert(new_row)
                    break
        if session_id == self._current_session_id:
            self._frame_model.append(dict(row))

    def _on_backfill_progress(self, current: int, total: int) -> None:
        self._backfill_status.setText(f"Backfill: {current}/{total}")

    def _on_backfill_done(self, count: int) -> None:
        self._backfill_status.setText(f"Backfill complete: {count} sessions imported.")
        self.refresh_sessions()
