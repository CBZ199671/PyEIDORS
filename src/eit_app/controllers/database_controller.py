"""Qt-aware wrapper around FrameDatabase with background backfill support."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

from eit_app.models.frame_database import FrameDatabase

log = logging.getLogger(__name__)


_FRAME_CSV_RE = re.compile(r"_frame_(\d+)\.csv$", re.IGNORECASE)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        log.debug("Failed to read YAML %s: %s", path, exc)
        return {}


class _BackfillWorker(QObject):
    """Scans a root directory for session folders and inserts them into the DB."""

    session_found = Signal(int, dict)  # (session_id, session_row)
    progress = Signal(int, int)  # (current, total)
    finished = Signal(int)  # total sessions imported
    error = Signal(str)

    def __init__(self, db: FrameDatabase, root_dir: Path) -> None:
        super().__init__()
        self._db = db
        self._root = Path(root_dir)

    def run(self) -> None:
        try:
            sessions = self._discover_sessions(self._root)
            total = len(sessions)
            imported = 0
            for idx, (session_dir, metadata_path) in enumerate(sessions):
                try:
                    self._ingest_session(session_dir, metadata_path)
                    imported += 1
                except Exception as exc:
                    log.warning("Backfill skipped %s: %s", session_dir, exc)
                self.progress.emit(idx + 1, total)
            self.finished.emit(imported)
        except Exception as exc:
            log.exception("Backfill failed")
            self.error.emit(str(exc))
            self.finished.emit(0)

    def _discover_sessions(self, root: Path) -> list[tuple[Path, Path | None]]:
        """Return (session_dir, metadata_yaml_path_or_None) pairs."""
        results: list[tuple[Path, Path | None]] = []
        seen: set[Path] = set()

        if not root.exists():
            return results

        # 1) Per-frame sessions (have session_metadata.yaml)
        for meta in root.rglob("session_metadata.yaml"):
            session_dir = meta.parent.resolve()
            if session_dir not in seen:
                results.append((session_dir, meta))
                seen.add(session_dir)

        # 2) Legacy aggregated format — any directory that has a top-level
        #    *.csv without session_metadata.yaml is treated as a session
        for csv_file in root.rglob("*.csv"):
            if csv_file.name.endswith("_AD.csv"):
                continue
            parent = csv_file.parent.resolve()
            if parent in seen:
                continue
            # Skip if this is a per-frame CSV inside an already-registered session
            if _FRAME_CSV_RE.search(csv_file.name):
                continue
            # Treat this parent as a legacy session once
            yaml_sibling = csv_file.with_suffix(".yaml")
            results.append((parent, yaml_sibling if yaml_sibling.exists() else None))
            seen.add(parent)

        return results

    def _ingest_session(
        self, session_dir: Path, metadata_path: Path | None
    ) -> None:
        metadata: dict[str, Any] = {}
        if metadata_path and metadata_path.exists():
            metadata = _read_yaml(metadata_path)

        # Derive started_at from dir name if missing
        started_at = metadata.get("session_start")
        if not started_at:
            started_at = self._derive_timestamp(session_dir.name)

        session_id = self._db.add_session(
            session_dir,
            metadata,
            name=session_dir.name,
            started_at=started_at,
        )
        row = self._db.get_session(session_id)
        if row is None:
            return

        # Ingest frames
        self._ingest_frames(session_id, session_dir)

        self.session_found.emit(session_id, row)

    def _ingest_frames(self, session_id: int, session_dir: Path) -> None:
        # Per-frame CSVs
        for csv_file in sorted(session_dir.glob("*_frame_*.csv")):
            match = _FRAME_CSV_RE.search(csv_file.name)
            if not match:
                continue
            frame_index = int(match.group(1))
            yaml_file = csv_file.with_suffix(".yaml")
            frame_meta = _read_yaml(yaml_file) if yaml_file.exists() else {}
            timestamp = float(
                frame_meta.get("timestamp", csv_file.stat().st_mtime)
            )
            self._db.add_frame(
                session_id=session_id,
                frame_index=frame_index,
                timestamp=timestamp,
                csv_path=csv_file,
                yaml_path=yaml_file if yaml_file.exists() else None,
                metadata=frame_meta,
            )

    @staticmethod
    def _derive_timestamp(name: str) -> str:
        """Try to parse YYYYMMDD_HHMMSS or YYYY-MM-DD-HH-MM-SS from a name."""
        patterns = [
            r"(\d{8})_(\d{6})",
            r"(\d{4}-\d{2}-\d{2})[-_](\d{2}-\d{2}-\d{2})",
        ]
        for pat in patterns:
            m = re.search(pat, name)
            if m:
                date_part, time_part = m.group(1), m.group(2)
                try:
                    if "-" in date_part:
                        dt = datetime.strptime(
                            f"{date_part} {time_part.replace('-', ':')}",
                            "%Y-%m-%d %H:%M:%S",
                        )
                    else:
                        dt = datetime.strptime(
                            f"{date_part} {time_part}", "%Y%m%d %H%M%S"
                        )
                    return dt.replace(tzinfo=timezone.utc).isoformat()
                except ValueError:
                    continue
        return datetime.now(timezone.utc).isoformat()


class DatabaseController(QObject):
    """Controller managing the FrameDatabase with Qt signals."""

    session_added = Signal(int, dict)
    frame_added = Signal(int, dict)
    backfill_progress = Signal(int, int)
    backfill_done = Signal(int)
    error = Signal(str)

    def __init__(self, db_path: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._db = FrameDatabase(db_path)
        self._backfill_thread: QThread | None = None
        self._backfill_worker: _BackfillWorker | None = None

    @property
    def db(self) -> FrameDatabase:
        return self._db

    def shutdown(self) -> None:
        if self._backfill_thread is not None:
            self._backfill_thread.quit()
            self._backfill_thread.wait(3000)
        self._db.close()

    # ---- Live recording integration ----

    def register_session(
        self, session_dir: Path, metadata: dict[str, Any]
    ) -> int:
        try:
            session_id = self._db.add_session(session_dir, metadata)
            row = self._db.get_session(session_id)
            if row is not None:
                self.session_added.emit(session_id, row)
            return session_id
        except Exception as exc:
            log.exception("Failed to register session")
            self.error.emit(str(exc))
            return -1

    def register_frame(
        self,
        session_id: int,
        frame_index: int,
        timestamp: float,
        csv_path: Path,
        yaml_path: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        try:
            frame_id = self._db.add_frame(
                session_id=session_id,
                frame_index=frame_index,
                timestamp=timestamp,
                csv_path=csv_path,
                yaml_path=yaml_path,
                metadata=metadata,
            )
            row = self._db.get_frame(frame_id)
            if row is not None:
                self.frame_added.emit(frame_id, row)
            return frame_id
        except Exception as exc:
            log.exception("Failed to register frame")
            self.error.emit(str(exc))
            return -1

    # ---- Queries ----

    def query_sessions(
        self,
        *,
        frequency_hz: int | None = None,
        started_after: str | None = None,
        started_before: str | None = None,
        name_like: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._db.query_sessions(
            frequency_hz=frequency_hz,
            started_after=started_after,
            started_before=started_before,
            name_like=name_like,
        )

    def query_frames(self, session_id: int) -> list[dict[str, Any]]:
        return self._db.query_frames(session_id)

    # ---- Backfill ----

    def start_backfill(self, root_dir: Path) -> None:
        if self._backfill_thread is not None and self._backfill_thread.isRunning():
            log.info("Backfill already in progress, skipping")
            return

        self._backfill_thread = QThread(self)
        self._backfill_worker = _BackfillWorker(self._db, root_dir)
        self._backfill_worker.moveToThread(self._backfill_thread)

        self._backfill_thread.started.connect(self._backfill_worker.run)
        self._backfill_worker.session_found.connect(self.session_added)
        self._backfill_worker.progress.connect(self.backfill_progress)
        self._backfill_worker.error.connect(self.error)
        self._backfill_worker.finished.connect(self._on_backfill_finished)

        self._backfill_thread.start()

    def _on_backfill_finished(self, count: int) -> None:
        self.backfill_done.emit(count)
        if self._backfill_thread is not None:
            self._backfill_thread.quit()
            self._backfill_thread.wait(3000)
            self._backfill_thread.deleteLater()
            self._backfill_thread = None
        if self._backfill_worker is not None:
            self._backfill_worker.deleteLater()
            self._backfill_worker = None
