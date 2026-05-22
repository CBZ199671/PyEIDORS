"""Database (数据库) tab — browse and filter historical EIT sessions and frames."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from eit_app.hardware.types import voltage_amp_label

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QIntValidator
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

from eit_app.i18n import t, translator
from eit_app.ui.hardware.live_plot_widget import LivePlotWidget
from eit_app.ui.theme import (
    info_card_stylesheet,
    selection_status_stylesheet,
    set_button_role,
    set_hint_text,
    stat_count_stylesheet,
    stat_subtle_stylesheet,
    subscribe_theme_mode,
)

log = logging.getLogger(__name__)


def _format_timestamp(ts_iso: str) -> str:
    """Render an ISO-8601 string in local time.

    UTC timestamps stored by older sessions get converted to the
    operator's local timezone before display; naive (already-local)
    timestamps are shown as-is.
    """
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone()
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts_iso


def _format_unix(ts: float) -> str:
    """Render a Unix-epoch timestamp in the operator's local timezone."""
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def _format_voltage_amp_level(value: Any) -> str:
    """Render the stored 0–7 voltage-amp level as ``10.00x`` etc.

    Blank cells for missing values; otherwise defers to the shared
    :func:`eit_app.hardware.types.voltage_amp_label` so the database
    tab and the hardware status bar stay in lock-step.
    """
    if value in (None, ""):
        return ""
    return voltage_amp_label(value, fallback=str(value))


def _format_session_frequency(row: dict[str, Any]) -> str:
    """Show ``min-max Hz`` when a session spans multiple frequencies.

    Falls back to a single-value display for sessions that ran at
    only one frequency, and to a blank string when no frequency was
    recorded.  Reads ``frequency_hz_min`` / ``frequency_hz_max`` if
    present (newer schema) and uses the legacy ``frequency_hz``
    column otherwise.
    """
    raw_min = row.get("frequency_hz_min")
    raw_max = row.get("frequency_hz_max")
    if raw_min in (None, "") and raw_max in (None, ""):
        legacy = row.get("frequency_hz")
        return f"{legacy} Hz" if legacy not in (None, "") else ""
    try:
        lo = int(raw_min) if raw_min not in (None, "") else None
        hi = int(raw_max) if raw_max not in (None, "") else None
    except (TypeError, ValueError):
        return ""
    if lo is None and hi is None:
        return ""
    if lo is None:
        return f"{hi} Hz"
    if hi is None or lo == hi:
        return f"{lo} Hz"
    return f"{lo} – {hi} Hz"


class _SessionTableModel(QAbstractTableModel):
    _COLUMN_KEYS = (
        "db.sessions.col.id",
        "db.sessions.col.name",
        "db.sessions.col.started",
        "db.sessions.col.n_elec",
        "db.sessions.col.frequency",
        "db.sessions.col.stim",
        "db.sessions.col.gain",
        "db.sessions.col.frames",
    )

    def __init__(self) -> None:
        super().__init__()
        self._rows: list[dict[str, Any]] = []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._COLUMN_KEYS)

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if (
            role == Qt.ItemDataRole.DisplayRole
            and orientation == Qt.Orientation.Horizontal
        ):
            return t(self._COLUMN_KEYS[section])
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
            return _format_session_frequency(row)
        if col == 5:
            return row.get("stim_amp_uA", "") or ""
        if col == 6:
            return _format_voltage_amp_level(row.get("voltage_amp_level"))
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
    _COLUMN_KEYS = (
        "db.frames.col.index",
        "db.frames.col.timestamp",
        "db.frames.col.file",
    )

    def __init__(self) -> None:
        super().__init__()
        self._rows: list[dict[str, Any]] = []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._COLUMN_KEYS)

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if (
            role == Qt.ItemDataRole.DisplayRole
            and orientation == Qt.Orientation.Horizontal
        ):
            return t(self._COLUMN_KEYS[section])
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
    reconstruct_requested = Signal(dict)  # config dict from ReconstructionDialog
    batch_reconstruct_requested = Signal(str)  # session_dir

    def __init__(self, db_controller, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db_ctrl = db_controller
        self._current_session_id: int | None = None
        self._selected_reference: dict | None = None
        self._selected_target: dict | None = None
        self._is_shutting_down = False
        # Cache the most recent dynamic strings so _retranslate can rebuild
        # them in the active language without re-querying the controller.
        self._session_count_cache: int = 0
        self._backfill_status_mode: str = "ready"  # "ready" | "progress" | "done"
        self._backfill_cache: tuple[int, int] = (0, 0)
        self._build_ui()
        self._connect_signals()
        self.refresh_sessions()
        translator().language_changed.connect(self._retranslate)
        self._retranslate()
        # Re-apply mini-card stylesheets when the user toggles dark mode.
        subscribe_theme_mode(self._on_theme_mode_changed)

    def _on_theme_mode_changed(self, _mode: str) -> None:
        """Re-apply per-card stylesheets that don't go through global QSS."""
        self._stats_card.setStyleSheet(info_card_stylesheet())
        self._count_label.setStyleSheet(stat_count_stylesheet())
        self._backfill_status.setStyleSheet(stat_subtle_stylesheet())
        self._selection_status.setStyleSheet(selection_status_stylesheet())

    def prepare_for_shutdown(self) -> None:
        self._is_shutting_down = True

    def _should_skip_database_refresh(self) -> bool:
        return self._is_shutting_down or bool(
            getattr(self._db_ctrl, "is_shutting_down", False)
        )

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        splitter.addWidget(self._build_filter_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_preview_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([240, 900, 340])
        root.addWidget(splitter)

    def _build_filter_panel(self) -> QWidget:
        self._filter_box = QGroupBox("")  # title set by _retranslate
        layout = QVBoxLayout(self._filter_box)
        layout.setContentsMargins(14, 20, 14, 14)
        layout.setSpacing(12)

        self._filter_hint = QLabel("")
        self._filter_hint.setWordWrap(True)
        set_hint_text(self._filter_hint)
        layout.addWidget(self._filter_hint)

        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._filter_name = QLineEdit()
        self._lbl_name = QLabel("")
        form.addRow(self._lbl_name, self._filter_name)

        # Frequency range (Hz) — same layout pattern as electrode-count
        # below.  Sessions that swept multiple frequencies match when
        # their swept envelope overlaps the requested envelope, not
        # only when their frequency exactly equals one number.
        freq_row = QHBoxLayout()
        freq_row.setSpacing(4)
        self._filter_freq_min = QLineEdit()
        self._filter_freq_min.setValidator(QIntValidator(0, 10_000_000, self))
        self._filter_freq_max = QLineEdit()
        self._filter_freq_max.setValidator(QIntValidator(0, 10_000_000, self))
        self._freq_range_dash = QLabel("\u2013")
        self._freq_range_dash.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._freq_range_dash.setMinimumWidth(12)
        freq_row.addWidget(self._filter_freq_min, 1)
        freq_row.addWidget(self._freq_range_dash, 0)
        freq_row.addWidget(self._filter_freq_max, 1)
        self._lbl_freq = QLabel("")
        form.addRow(self._lbl_freq, freq_row)

        self._filter_date_from = QDateEdit()
        self._filter_date_from.setCalendarPopup(True)
        self._filter_date_from.setDate(self._filter_date_from.minimumDate())
        self._filter_date_from.setDisplayFormat("yyyy-MM-dd")
        self._lbl_date_from = QLabel("")
        form.addRow(self._lbl_date_from, self._filter_date_from)

        self._filter_date_to = QDateEdit()
        self._filter_date_to.setCalendarPopup(True)
        self._filter_date_to.setDate(self._filter_date_to.minimumDate())
        self._filter_date_to.setDisplayFormat("yyyy-MM-dd")
        self._lbl_date_to = QLabel("")
        form.addRow(self._lbl_date_to, self._filter_date_to)

        # Electrode-count range (min + max on one row).  Blank = unbounded.
        n_elec_row = QHBoxLayout()
        n_elec_row.setSpacing(4)
        self._filter_n_elec_min = QLineEdit()
        self._filter_n_elec_min.setValidator(QIntValidator(0, 1024, self))
        self._filter_n_elec_max = QLineEdit()
        self._filter_n_elec_max.setValidator(QIntValidator(0, 1024, self))
        self._n_elec_range_dash = QLabel("\u2013")
        self._n_elec_range_dash.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._n_elec_range_dash.setMinimumWidth(12)
        n_elec_row.addWidget(self._filter_n_elec_min, 1)
        n_elec_row.addWidget(self._n_elec_range_dash, 0)
        n_elec_row.addWidget(self._filter_n_elec_max, 1)
        self._lbl_n_elec = QLabel("")
        form.addRow(self._lbl_n_elec, n_elec_row)

        # Stim-amp range (µA) — same layout as electrode count.
        stim_row = QHBoxLayout()
        stim_row.setSpacing(4)
        self._filter_stim_min = QLineEdit()
        self._filter_stim_min.setValidator(QIntValidator(0, 100_000, self))
        self._filter_stim_max = QLineEdit()
        self._filter_stim_max.setValidator(QIntValidator(0, 100_000, self))
        self._stim_range_dash = QLabel("\u2013")
        self._stim_range_dash.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stim_range_dash.setMinimumWidth(12)
        stim_row.addWidget(self._filter_stim_min, 1)
        stim_row.addWidget(self._stim_range_dash, 0)
        stim_row.addWidget(self._filter_stim_max, 1)
        self._lbl_stim_amp = QLabel("")
        form.addRow(self._lbl_stim_amp, stim_row)

        layout.addLayout(form)

        self._apply_btn = QPushButton("")
        set_button_role(self._apply_btn, "primary")
        layout.addWidget(self._apply_btn)

        sub_btn_row = QHBoxLayout()
        sub_btn_row.setSpacing(6)
        # "Clear filters" resets the filter form inputs only — nothing
        # on disk is touched — so "subtle" is the correct role and not
        # an inconsistency with Hardware's Clear List (which *does*
        # drop the in-memory frame list and is correctly tagged danger).
        self._clear_btn = QPushButton("")
        set_button_role(self._clear_btn, "subtle")
        self._refresh_btn = QPushButton("")
        set_button_role(self._refresh_btn, "subtle")
        sub_btn_row.addWidget(self._clear_btn)
        sub_btn_row.addWidget(self._refresh_btn)
        layout.addLayout(sub_btn_row)

        layout.addStretch()

        # Stats card at the bottom — styles come from theme helpers so
        # they re-paint on dark mode toggles (see _on_theme_mode_changed).
        self._stats_card = QWidget()
        self._stats_card.setStyleSheet(info_card_stylesheet())
        stats_layout = QVBoxLayout(self._stats_card)
        stats_layout.setContentsMargins(12, 10, 12, 10)
        stats_layout.setSpacing(4)

        self._count_label = QLabel("")
        self._count_label.setStyleSheet(stat_count_stylesheet())
        stats_layout.addWidget(self._count_label)

        self._backfill_status = QLabel("")
        self._backfill_status.setStyleSheet(stat_subtle_stylesheet())
        self._backfill_status.setWordWrap(True)
        stats_layout.addWidget(self._backfill_status)

        layout.addWidget(self._stats_card)

        self._filter_box.setMinimumWidth(200)
        self._filter_box.setMaximumWidth(330)
        return self._filter_box

    def _build_center_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)

        # ---- Sessions section ----
        self._sessions_box = QGroupBox("")  # title set by _retranslate
        sessions_layout = QVBoxLayout(self._sessions_box)
        sessions_layout.setContentsMargins(14, 20, 14, 14)
        sessions_layout.setSpacing(10)

        self._session_model = _SessionTableModel()
        self._session_table = QTableView()
        self._session_table.setModel(self._session_model)
        self._session_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._session_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._session_table.setAlternatingRowColors(True)
        self._session_table.verticalHeader().setVisible(False)
        self._session_table.setShowGrid(False)
        self._session_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._session_table.horizontalHeader().setHighlightSections(False)
        self._session_table.verticalHeader().setDefaultSectionSize(26)

        hdr = self._session_table.horizontalHeader()
        hdr.setStretchLastSection(False)
        # Column-specific sizing
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # ID
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # Name
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Started
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)  # N_elec
        hdr.setSectionResizeMode(
            4, QHeaderView.ResizeMode.ResizeToContents
        )  # Frequency
        hdr.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)  # Stim
        hdr.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)  # Gain
        hdr.setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)  # Frames

        sessions_layout.addWidget(self._session_table, 1)

        session_actions = QHBoxLayout()
        session_actions.setSpacing(6)
        self._open_folder_btn = QPushButton("")
        set_button_role(self._open_folder_btn, "subtle")
        self._open_folder_btn.setEnabled(False)
        self._batch_recon_btn = QPushButton("")
        # Reconstruction is an affirmative action, not destructive.
        # Previously tagged "danger" which rendered it red and
        # misrepresented the button's purpose.
        set_button_role(self._batch_recon_btn, "primary")
        self._batch_recon_btn.setEnabled(False)
        session_actions.addStretch()
        session_actions.addWidget(self._open_folder_btn)
        session_actions.addWidget(self._batch_recon_btn)
        sessions_layout.addLayout(session_actions)

        splitter.addWidget(self._sessions_box)

        # ---- Frames section ----
        self._frames_box = QGroupBox("")  # title set by _retranslate
        frames_layout = QVBoxLayout(self._frames_box)
        frames_layout.setContentsMargins(14, 20, 14, 14)
        frames_layout.setSpacing(10)

        self._frame_model = _FrameTableModel()
        self._frame_table = QTableView()
        self._frame_table.setModel(self._frame_model)
        self._frame_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._frame_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._frame_table.setAlternatingRowColors(True)
        self._frame_table.verticalHeader().setVisible(False)
        self._frame_table.setShowGrid(False)
        self._frame_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._frame_table.horizontalHeader().setHighlightSections(False)
        self._frame_table.verticalHeader().setDefaultSectionSize(26)

        frame_hdr = self._frame_table.horizontalHeader()
        frame_hdr.setStretchLastSection(True)
        frame_hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        frame_hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)

        frames_layout.addWidget(self._frame_table, 1)

        # Selection status line — text assigned by _update_selection_status().
        # Styling comes from theme.selection_status_stylesheet() so it
        # follows dark mode (was hardcoded #f5f9fd / #5b6573 before).
        self._selection_status = QLabel("")
        self._selection_status.setStyleSheet(selection_status_stylesheet())
        self._selection_status.setWordWrap(True)
        frames_layout.addWidget(self._selection_status)

        frame_actions = QHBoxLayout()
        frame_actions.setSpacing(6)
        self._as_ref_btn = QPushButton("")
        set_button_role(self._as_ref_btn, "primary")
        self._as_ref_btn.setEnabled(False)
        self._as_tgt_btn = QPushButton("")
        set_button_role(self._as_tgt_btn, "success")
        self._as_tgt_btn.setEnabled(False)
        self._reconstruct_btn = QPushButton("")
        # Reconstruction opens the reconstruct dialog — primary CTA,
        # not destructive.  The previous "danger" tag turned a
        # confirmation action red.
        set_button_role(self._reconstruct_btn, "primary")
        self._reconstruct_btn.setEnabled(False)
        self._clear_sel_btn = QPushButton("")
        set_button_role(self._clear_sel_btn, "subtle")
        frame_actions.addWidget(self._as_ref_btn)
        frame_actions.addWidget(self._as_tgt_btn)
        frame_actions.addWidget(self._reconstruct_btn)
        frame_actions.addStretch()
        frame_actions.addWidget(self._clear_sel_btn)
        frames_layout.addLayout(frame_actions)

        splitter.addWidget(self._frames_box)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([420, 280])

        layout.addWidget(splitter)
        return container

    def _build_preview_panel(self) -> QWidget:
        self._preview_box = QGroupBox("")  # title set by _retranslate
        layout = QVBoxLayout(self._preview_box)
        layout.setContentsMargins(14, 20, 14, 14)
        layout.setSpacing(10)

        self._preview_hint = QLabel("")
        self._preview_hint.setWordWrap(True)
        set_hint_text(self._preview_hint)
        layout.addWidget(self._preview_hint)

        self._preview_plot = LivePlotWidget()
        self._preview_plot.setMinimumHeight(280)
        layout.addWidget(self._preview_plot, 1)

        self._preview_box.setMinimumWidth(220)
        self._preview_box.setMaximumWidth(420)
        return self._preview_box

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
        self._batch_recon_btn.clicked.connect(self._on_batch_reconstruct)
        self._as_ref_btn.clicked.connect(self._on_set_reference)
        self._as_tgt_btn.clicked.connect(self._on_set_target)
        self._reconstruct_btn.clicked.connect(self._on_open_reconstruct_dialog)
        self._clear_sel_btn.clicked.connect(self._on_clear_selection)

        self._db_ctrl.session_added.connect(self._on_session_added)
        self._db_ctrl.frame_added.connect(self._on_frame_added)
        self._db_ctrl.backfill_progress.connect(self._on_backfill_progress)
        self._db_ctrl.backfill_done.connect(self._on_backfill_done)

    def refresh_sessions(self) -> None:
        if self._should_skip_database_refresh():
            return
        filters: dict[str, Any] = {}
        name = self._filter_name.text().strip()
        if name:
            filters["name_like"] = name
        date_from = self._filter_date_from.date()
        if date_from != self._filter_date_from.minimumDate():
            filters["started_after"] = date_from.toString("yyyy-MM-dd")
        date_to = self._filter_date_to.date()
        if date_to != self._filter_date_to.minimumDate():
            filters["started_before"] = date_to.toString("yyyy-MM-dd") + "T23:59:59"

        # Range filters: each bound is optional.  Invalid / empty
        # values fall through silently — the QIntValidator already
        # prevents non-digit characters from being entered.
        for key, widget in (
            ("n_elec_min", self._filter_n_elec_min),
            ("n_elec_max", self._filter_n_elec_max),
            ("stim_amp_ua_min", self._filter_stim_min),
            ("stim_amp_ua_max", self._filter_stim_max),
            ("frequency_hz_min", self._filter_freq_min),
            ("frequency_hz_max", self._filter_freq_max),
        ):
            raw = widget.text().strip()
            if not raw:
                continue
            try:
                filters[key] = int(raw)
            except ValueError:
                continue

        sessions = self._db_ctrl.query_sessions(**filters)
        self._session_model.set_rows(sessions)
        self._session_count_cache = len(sessions)
        self._count_label.setText(t("db.stats.count", count=len(sessions)))
        self._frame_model.set_rows([])
        self._current_session_id = None
        self._selected_reference = None
        self._selected_target = None
        self._as_ref_btn.setEnabled(False)
        self._as_tgt_btn.setEnabled(False)
        self._open_folder_btn.setEnabled(False)
        self._batch_recon_btn.setEnabled(False)
        self._update_selection_status()

    def _clear_filters(self) -> None:
        self._filter_name.clear()
        self._filter_freq_min.clear()
        self._filter_freq_max.clear()
        self._filter_date_from.setDate(self._filter_date_from.minimumDate())
        self._filter_date_to.setDate(self._filter_date_to.minimumDate())
        self._filter_n_elec_min.clear()
        self._filter_n_elec_max.clear()
        self._filter_stim_min.clear()
        self._filter_stim_max.clear()
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
        self._selected_reference = None
        self._selected_target = None
        self._update_selection_status()
        if session is None:
            self._current_session_id = None
            self._frame_model.set_rows([])
            self._open_folder_btn.setEnabled(False)
            self._batch_recon_btn.setEnabled(False)
            return
        self._current_session_id = int(session["id"])
        self._open_folder_btn.setEnabled(True)
        self._batch_recon_btn.setEnabled(True)
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

    def _on_set_reference(self) -> None:
        frame = self._selected_frame()
        if frame is not None:
            self._selected_reference = dict(frame)
            self._update_selection_status()
            self.load_as_reference_requested.emit(dict(frame))

    def _on_set_target(self) -> None:
        frame = self._selected_frame()
        if frame is not None:
            self._selected_target = dict(frame)
            self._update_selection_status()
            self.load_as_target_requested.emit(dict(frame))

    def _on_clear_selection(self) -> None:
        self._selected_reference = None
        self._selected_target = None
        self._update_selection_status()

    def _on_batch_reconstruct(self) -> None:
        session = self._selected_session()
        if session is None:
            return
        session_dir = str(session.get("session_dir", ""))
        if session_dir:
            self.batch_reconstruct_requested.emit(session_dir)

    def _on_open_reconstruct_dialog(self) -> None:
        from eit_app.ui.dialogs.reconstruction_dialog import ReconstructionDialog

        dialog = ReconstructionDialog(
            reference_entry=self._selected_reference,
            target_entry=self._selected_target,
            parent=self,
        )
        dialog.run_requested.connect(self.reconstruct_requested)
        dialog.exec()

    def _update_selection_status(self) -> None:
        ref_role = t("db.frames.selection_role.reference")
        tgt_role = t("db.frames.selection_role.target")
        ref_txt = self._format_selection(self._selected_reference, ref_role)
        tgt_txt = self._format_selection(self._selected_target, tgt_role)
        # No hint yet + no selection → show the one-off helper hint; once
        # either slot is populated, switch to the status line.
        if self._selected_reference is None and self._selected_target is None:
            self._selection_status.setText(t("db.frames.selection_hint"))
        else:
            self._selection_status.setText(f"{ref_txt}   |   {tgt_txt}")
        self._reconstruct_btn.setEnabled(self._selected_target is not None)

    @staticmethod
    def _format_selection(entry: dict | None, role: str) -> str:
        if entry is None:
            return t("db.frames.selection_unset", role=role)
        idx = entry.get("frame_index", "?")
        return t("db.frames.selection_set", role=role, index=idx)

    def _on_session_added(self, session_id: int, row: dict) -> None:
        if self._should_skip_database_refresh():
            return
        row = dict(row)
        row.setdefault("frame_count", 0)
        self._session_model.upsert(row)
        self._session_count_cache = self._session_model.rowCount()
        self._count_label.setText(t("db.stats.count", count=self._session_count_cache))

    def _on_frame_added(self, frame_id: int, row: dict) -> None:
        if self._should_skip_database_refresh():
            return
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
        if self._should_skip_database_refresh():
            return
        self._backfill_status_mode = "progress"
        self._backfill_cache = (current, total)
        self._backfill_status.setText(
            t("db.stats.backfill_progress", current=current, total=total)
        )

    def _on_backfill_done(self, count: int) -> None:
        if self._should_skip_database_refresh():
            return
        self._backfill_status_mode = "done"
        self._backfill_cache = (count, count)
        self._backfill_status.setText(t("db.stats.backfill_done", count=count))
        self.refresh_sessions()

    # ── i18n ──

    def _retranslate(self) -> None:
        """Refresh every owned string to the active language."""
        # Filter panel
        self._filter_box.setTitle(t("db.filters.title"))
        self._filter_hint.setText(t("db.filters.hint"))
        self._filter_name.setPlaceholderText(t("db.filters.name_placeholder"))
        self._filter_freq_min.setPlaceholderText(t("db.filters.freq_min_placeholder"))
        self._filter_freq_max.setPlaceholderText(t("db.filters.freq_max_placeholder"))
        self._filter_date_from.setSpecialValueText(t("db.filters.date_any"))
        self._filter_date_to.setSpecialValueText(t("db.filters.date_any"))
        self._lbl_name.setText(t("db.filters.name_label"))
        self._lbl_freq.setText(t("db.filters.freq_label"))
        self._lbl_date_from.setText(t("db.filters.date_from_label"))
        self._lbl_date_to.setText(t("db.filters.date_to_label"))
        self._lbl_n_elec.setText(t("db.filters.n_elec_label"))
        self._filter_n_elec_min.setPlaceholderText(
            t("db.filters.n_elec_min_placeholder")
        )
        self._filter_n_elec_max.setPlaceholderText(
            t("db.filters.n_elec_max_placeholder")
        )
        self._lbl_stim_amp.setText(t("db.filters.stim_amp_label"))
        self._filter_stim_min.setPlaceholderText(
            t("db.filters.stim_amp_min_placeholder")
        )
        self._filter_stim_max.setPlaceholderText(
            t("db.filters.stim_amp_max_placeholder")
        )
        self._apply_btn.setText(t("db.filters.apply_button"))
        self._clear_btn.setText(t("db.filters.clear_button"))
        self._refresh_btn.setText(t("db.filters.refresh_button"))

        # Stats card (dynamic — re-render from caches)
        self._count_label.setText(t("db.stats.count", count=self._session_count_cache))
        if self._backfill_status_mode == "progress":
            current, total = self._backfill_cache
            self._backfill_status.setText(
                t("db.stats.backfill_progress", current=current, total=total)
            )
        elif self._backfill_status_mode == "done":
            self._backfill_status.setText(
                t("db.stats.backfill_done", count=self._backfill_cache[0])
            )
        else:
            self._backfill_status.setText(t("db.stats.ready"))

        # Sessions + Frames sections
        self._sessions_box.setTitle(t("db.sessions.title"))
        self._frames_box.setTitle(t("db.frames.title"))
        self._open_folder_btn.setText(t("db.sessions.open_folder_button"))
        self._batch_recon_btn.setText(t("db.sessions.batch_recon_button"))
        self._as_ref_btn.setText(t("db.frames.set_ref_button"))
        self._as_tgt_btn.setText(t("db.frames.set_tgt_button"))
        self._reconstruct_btn.setText(t("db.frames.reconstruct_button"))
        self._clear_sel_btn.setText(t("db.frames.clear_button"))

        # Selection status line — refreshes role text in current language
        self._update_selection_status()

        # Preview panel
        self._preview_box.setTitle(t("db.preview.title"))
        self._preview_hint.setText(t("db.preview.hint"))

        # Table header labels refresh
        self._session_model.headerDataChanged.emit(
            Qt.Orientation.Horizontal, 0, self._session_model.columnCount() - 1
        )
        self._frame_model.headerDataChanged.emit(
            Qt.Orientation.Horizontal, 0, self._frame_model.columnCount() - 1
        )
