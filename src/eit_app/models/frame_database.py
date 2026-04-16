"""SQLite-backed persistent index of recorded EIT frames.

Pure stdlib — no Qt dependency. The DatabaseController wraps this class
with Qt signals for GUI integration.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    session_dir TEXT NOT NULL UNIQUE,
    started_at TEXT NOT NULL,
    n_elec INTEGER,
    stim_pattern TEXT,
    meas_pattern TEXT,
    frequency_hz INTEGER,
    stim_amp_uA INTEGER,
    voltage_amp_level INTEGER,
    transport_type TEXT,
    mea_mode INTEGER,
    notes TEXT,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_sessions_freq ON sessions(frequency_hz);

CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    frame_index INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    csv_path TEXT NOT NULL UNIQUE,
    yaml_path TEXT,
    frame_metadata_json TEXT,
    UNIQUE(session_id, frame_index)
);

CREATE INDEX IF NOT EXISTS idx_frames_session ON frames(session_id);
CREATE INDEX IF NOT EXISTS idx_frames_timestamp ON frames(timestamp);
"""


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


class FrameDatabase:
    """SQLite wrapper for the EIT frame index."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @property
    def path(self) -> Path:
        return self._db_path

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # ---- Sessions ----

    def add_session(
        self,
        session_dir: Path | str,
        metadata: dict[str, Any],
        *,
        name: str | None = None,
        started_at: str | None = None,
    ) -> int:
        """Add or return id of existing session. Uses session_dir as unique key."""
        session_dir_str = str(Path(session_dir).resolve())
        existing = self.find_session_by_dir(session_dir_str)
        if existing:
            return int(existing["id"])

        resolved_name = name or Path(session_dir_str).name
        resolved_started = started_at or metadata.get("session_start") or datetime.now(
            timezone.utc
        ).isoformat()

        row = {
            "name": resolved_name,
            "session_dir": session_dir_str,
            "started_at": str(resolved_started),
            "n_elec": _to_int(metadata.get("n_elec")),
            "stim_pattern": _to_str(metadata.get("stim_pattern")),
            "meas_pattern": _to_str(metadata.get("meas_pattern")),
            "frequency_hz": _to_int(metadata.get("frequency_hz")),
            "stim_amp_uA": _to_int(metadata.get("stim_amp_uA")),
            "voltage_amp_level": _to_int(
                metadata.get("voltage_amp_level")
                or metadata.get("voltage_amp_level_1")
            ),
            "transport_type": _to_str(metadata.get("transport_type")),
            "mea_mode": _to_int(metadata.get("mea_mode")),
            "notes": _to_str(metadata.get("notes")),
            "metadata_json": json.dumps(metadata, default=str),
        }

        cur = self._conn.execute(
            """
            INSERT INTO sessions (
                name, session_dir, started_at, n_elec, stim_pattern, meas_pattern,
                frequency_hz, stim_amp_uA, voltage_amp_level, transport_type,
                mea_mode, notes, metadata_json
            ) VALUES (
                :name, :session_dir, :started_at, :n_elec, :stim_pattern, :meas_pattern,
                :frequency_hz, :stim_amp_uA, :voltage_amp_level, :transport_type,
                :mea_mode, :notes, :metadata_json
            )
            """,
            row,
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def find_session_by_dir(self, session_dir: str) -> dict[str, Any] | None:
        session_dir = str(Path(session_dir).resolve())
        cur = self._conn.execute(
            "SELECT * FROM sessions WHERE session_dir = ?", (session_dir,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_session(self, session_id: int) -> dict[str, Any] | None:
        cur = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (int(session_id),)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def query_sessions(
        self,
        *,
        frequency_hz: int | None = None,
        started_after: str | None = None,
        started_before: str | None = None,
        name_like: str | None = None,
        n_elec_min: int | None = None,
        n_elec_max: int | None = None,
        stim_amp_ua_min: int | None = None,
        stim_amp_ua_max: int | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Return sessions matching optional filters, newest first.

        Range filters (``*_min`` / ``*_max``) are inclusive on both ends.
        Any bound that is None is skipped, so you can ask for an
        open-ended range by supplying only one side.
        """
        sql = "SELECT s.*, (SELECT COUNT(*) FROM frames f WHERE f.session_id = s.id) AS frame_count FROM sessions s WHERE 1=1"
        params: list[Any] = []
        if frequency_hz is not None:
            sql += " AND s.frequency_hz = ?"
            params.append(int(frequency_hz))
        if started_after:
            sql += " AND s.started_at >= ?"
            params.append(started_after)
        if started_before:
            sql += " AND s.started_at <= ?"
            params.append(started_before)
        if name_like:
            sql += " AND (s.name LIKE ? OR s.session_dir LIKE ?)"
            like = f"%{name_like}%"
            params.extend([like, like])
        if n_elec_min is not None:
            sql += " AND s.n_elec >= ?"
            params.append(int(n_elec_min))
        if n_elec_max is not None:
            sql += " AND s.n_elec <= ?"
            params.append(int(n_elec_max))
        if stim_amp_ua_min is not None:
            sql += " AND s.stim_amp_uA >= ?"
            params.append(int(stim_amp_ua_min))
        if stim_amp_ua_max is not None:
            sql += " AND s.stim_amp_uA <= ?"
            params.append(int(stim_amp_ua_max))
        sql += " ORDER BY s.started_at DESC LIMIT ?"
        params.append(int(limit))

        cur = self._conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]

    def delete_session(self, session_id: int) -> None:
        self._conn.execute("DELETE FROM sessions WHERE id = ?", (int(session_id),))
        self._conn.commit()

    # ---- Frames ----

    def add_frame(
        self,
        session_id: int,
        frame_index: int,
        timestamp: float,
        csv_path: Path | str,
        yaml_path: Path | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add or return id of existing frame. csv_path is the unique key."""
        csv_str = str(Path(csv_path).resolve())
        existing = self.find_frame_by_path(csv_str)
        if existing:
            return int(existing["id"])

        row = {
            "session_id": int(session_id),
            "frame_index": int(frame_index),
            "timestamp": float(timestamp),
            "csv_path": csv_str,
            "yaml_path": str(Path(yaml_path).resolve()) if yaml_path else None,
            "frame_metadata_json": json.dumps(metadata or {}, default=str),
        }
        try:
            cur = self._conn.execute(
                """
                INSERT INTO frames (
                    session_id, frame_index, timestamp, csv_path, yaml_path,
                    frame_metadata_json
                ) VALUES (
                    :session_id, :frame_index, :timestamp, :csv_path, :yaml_path,
                    :frame_metadata_json
                )
                """,
                row,
            )
            self._conn.commit()
            return int(cur.lastrowid)
        except sqlite3.IntegrityError as exc:
            # Duplicate (session_id, frame_index) — return existing row id
            log.debug("Frame insert skipped (already present): %s", exc)
            cur = self._conn.execute(
                "SELECT id FROM frames WHERE session_id = ? AND frame_index = ?",
                (int(session_id), int(frame_index)),
            )
            hit = cur.fetchone()
            if hit:
                return int(hit["id"])
            raise

    def find_frame_by_path(self, csv_path: str) -> dict[str, Any] | None:
        csv_path = str(Path(csv_path).resolve())
        cur = self._conn.execute(
            "SELECT * FROM frames WHERE csv_path = ?", (csv_path,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def query_frames(self, session_id: int) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM frames WHERE session_id = ? ORDER BY frame_index ASC",
            (int(session_id),),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_frame(self, frame_id: int) -> dict[str, Any] | None:
        cur = self._conn.execute(
            "SELECT * FROM frames WHERE id = ?", (int(frame_id),)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def count_sessions(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) AS c FROM sessions")
        row = cur.fetchone()
        return int(row["c"]) if row else 0

    def count_frames(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) AS c FROM frames")
        row = cur.fetchone()
        return int(row["c"]) if row else 0
