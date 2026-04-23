"""SQLite-backed persistent index of recorded EIT frames.

Pure stdlib — no Qt dependency. The DatabaseController wraps this class
with Qt signals for GUI integration.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
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
    frequency_hz_min INTEGER,
    frequency_hz_max INTEGER,
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

    # Schema version 1 = baseline shape.  Bumped to 2 once the
    # frequency-range columns + frame-driven back-fill have been
    # applied, so we don't redo the (potentially many-thousand-row)
    # rebuild on every app start.  Bump again whenever a future
    # schema migration needs to fire exactly once.
    _SCHEMA_VERSION = 2

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._run_migrations()
        self._conn.commit()

    def _run_migrations(self) -> None:
        """Apply pending schema migrations gated by ``PRAGMA user_version``."""
        current = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if current < 2:
            self._migrate_frequency_range_columns()
        # Every future migration adds another `if current < N:` branch
        # before the version bump below.
        if current < self._SCHEMA_VERSION:
            self._conn.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")

    def _migrate_frequency_range_columns(self) -> None:
        """Add ``frequency_hz_min`` / ``_max`` columns and back-fill them.

        Runs at most once per database file (gated by
        ``PRAGMA user_version`` in :meth:`_run_migrations`).  Steps:

        1.  ALTER TABLE adds the new columns when missing (no-op on
            fresh databases — the column is already in ``_SCHEMA``).
        2.  Seed ``min`` / ``max`` from the legacy single-value column
            so rows with no per-frame metadata at all still display
            sensibly.
        3.  Re-scan every frame's ``frame_metadata_json`` and widen the
            session's range from any per-frame frequency it finds —
            this is what makes a sweep session that pre-dates the new
            range columns show its full envelope after the upgrade.
        """
        cur = self._conn.execute("PRAGMA table_info(sessions)")
        existing = {str(row["name"]) for row in cur.fetchall()}
        if "frequency_hz_min" not in existing:
            self._conn.execute(
                "ALTER TABLE sessions ADD COLUMN frequency_hz_min INTEGER"
            )
        if "frequency_hz_max" not in existing:
            self._conn.execute(
                "ALTER TABLE sessions ADD COLUMN frequency_hz_max INTEGER"
            )
        self._conn.execute(
            """
            UPDATE sessions
               SET frequency_hz_min = frequency_hz
             WHERE frequency_hz_min IS NULL AND frequency_hz IS NOT NULL
            """
        )
        self._conn.execute(
            """
            UPDATE sessions
               SET frequency_hz_max = frequency_hz
             WHERE frequency_hz_max IS NULL AND frequency_hz IS NOT NULL
            """
        )
        self._rebuild_session_frequency_ranges_from_frames()

    def _rebuild_session_frequency_ranges_from_frames(self) -> None:
        """Re-derive frequency ranges for sessions that don't have one yet.

        Walks the frames table once per session, parses
        ``frame_metadata_json`` for ``frequency_hz`` values, and
        widens / shrinks the session's stored range to match.  The
        gating SELECT only returns sessions whose ``min`` and ``max``
        are still equal — i.e. those whose range was seeded only
        from the legacy single-value column.  After ``add_frame``
        keeps the bounds live for any new sessions, so this catches
        only the historical pre-migration ones.
        """
        try:
            cur = self._conn.execute(
                "SELECT id FROM sessions WHERE frequency_hz_min IS frequency_hz_max"
            )
            session_ids = [int(row["id"]) for row in cur.fetchall()]
        except sqlite3.OperationalError:
            return
        for sid in session_ids:
            frames = self._conn.execute(
                """
                SELECT frame_metadata_json
                  FROM frames
                 WHERE session_id = ?
                   AND frame_metadata_json IS NOT NULL
                """,
                (sid,),
            ).fetchall()
            seen: list[int] = []
            for row in frames:
                payload = row["frame_metadata_json"]
                if not payload:
                    continue
                try:
                    meta = json.loads(payload)
                except (TypeError, ValueError):
                    continue
                hz = _to_int(meta.get("frequency_hz"))
                if hz is not None:
                    seen.append(hz)
            if not seen:
                continue
            self._conn.execute(
                """
                UPDATE sessions
                   SET frequency_hz_min = ?, frequency_hz_max = ?
                 WHERE id = ?
                """,
                (min(seen), max(seen), sid),
            )

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
        # Default to local-system time so timestamps line up with the
        # operator's wall clock rather than UTC.
        resolved_started = (
            started_at or metadata.get("session_start") or datetime.now().isoformat()
        )

        seed_freq = _to_int(metadata.get("frequency_hz"))
        row = {
            "name": resolved_name,
            "session_dir": session_dir_str,
            "started_at": str(resolved_started),
            "n_elec": _to_int(metadata.get("n_elec")),
            "stim_pattern": _to_str(metadata.get("stim_pattern")),
            "meas_pattern": _to_str(metadata.get("meas_pattern")),
            "frequency_hz": seed_freq,
            # Seed the range with the session's initial frequency.
            # update_session_frequency_range() widens these bounds as
            # subsequent frames arrive at other frequencies.
            "frequency_hz_min": seed_freq,
            "frequency_hz_max": seed_freq,
            "stim_amp_uA": _to_int(metadata.get("stim_amp_uA")),
            "voltage_amp_level": _to_int(
                metadata.get("voltage_amp_level") or metadata.get("voltage_amp_level_1")
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
                frequency_hz, frequency_hz_min, frequency_hz_max,
                stim_amp_uA, voltage_amp_level, transport_type,
                mea_mode, notes, metadata_json
            ) VALUES (
                :name, :session_dir, :started_at, :n_elec, :stim_pattern, :meas_pattern,
                :frequency_hz, :frequency_hz_min, :frequency_hz_max,
                :stim_amp_uA, :voltage_amp_level, :transport_type,
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
        frequency_hz_min: int | None = None,
        frequency_hz_max: int | None = None,
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

        ``frequency_hz_min`` / ``frequency_hz_max`` filter on the
        session's own swept range — i.e. a session that ran from
        1000 to 3000 Hz matches a query of (1500, 2500) because its
        sweep envelope overlaps the query envelope.  This is the
        natural reading of "show me sessions in this band".
        ``frequency_hz`` (the legacy single-value filter) is still
        accepted for backward compatibility.
        """
        sql = "SELECT s.*, (SELECT COUNT(*) FROM frames f WHERE f.session_id = s.id) AS frame_count FROM sessions s WHERE 1=1"
        params: list[Any] = []
        if frequency_hz is not None:
            sql += " AND (s.frequency_hz = ? OR (s.frequency_hz_min <= ? AND s.frequency_hz_max >= ?))"
            params.extend([int(frequency_hz)] * 3)
        if frequency_hz_min is not None:
            # The session's max must reach at least the requested
            # lower bound for the bands to overlap.
            sql += " AND COALESCE(s.frequency_hz_max, s.frequency_hz) >= ?"
            params.append(int(frequency_hz_min))
        if frequency_hz_max is not None:
            sql += " AND COALESCE(s.frequency_hz_min, s.frequency_hz) <= ?"
            params.append(int(frequency_hz_max))
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

    def _update_session_frequency_range(
        self, session_id: int, frame_frequency_hz: Any
    ) -> None:
        """Widen ``frequency_hz_min`` / ``frequency_hz_max`` for a session.

        Called whenever a frame is added so the session row reflects
        the full sweep range as it grows.  Silently no-ops when the
        new frame doesn't carry a frequency.
        """
        freq = _to_int(frame_frequency_hz)
        if freq is None:
            return
        self._conn.execute(
            """
            UPDATE sessions
               SET frequency_hz_min = MIN(COALESCE(frequency_hz_min, ?), ?),
                   frequency_hz_max = MAX(COALESCE(frequency_hz_max, ?), ?)
             WHERE id = ?
            """,
            (freq, freq, freq, freq, int(session_id)),
        )

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
            # Widen the session's frequency range from this frame's
            # metadata so a session that swept multiple frequencies
            # ends up showing them all in the UI.
            self._update_session_frequency_range(
                int(session_id), (metadata or {}).get("frequency_hz")
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
        cur = self._conn.execute("SELECT * FROM frames WHERE csv_path = ?", (csv_path,))
        row = cur.fetchone()
        return dict(row) if row else None

    def query_frames(self, session_id: int) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM frames WHERE session_id = ? ORDER BY frame_index ASC",
            (int(session_id),),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_frame(self, frame_id: int) -> dict[str, Any] | None:
        cur = self._conn.execute("SELECT * FROM frames WHERE id = ?", (int(frame_id),))
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
