"""Records acquired frames to disk in the per-frame CSV + YAML format.

Manages session directories and delegates actual I/O to pyeidors.data.frame_io.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal

from eit_app.models.frame_model import FrameData

log = logging.getLogger(__name__)


class RecordingController(QObject):
    """Writes incoming FrameData to per-frame CSV + YAML files.

    Signals:
        recording_started: Emitted with the session directory path.
        recording_stopped: Emitted with total frames recorded.
        frame_saved: Emitted with (frame_index, timestamp, file_path) after each frame write.
        error: Emitted on I/O errors.
    """

    recording_started = Signal(str)
    recording_stopped = Signal(int)
    frame_saved = Signal(int, float, str)
    error = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._session_dir: Path | None = None
        self._session_prefix: str = ""
        self._frames_recorded: int = 0
        self._is_recording: bool = False
        self._session_metadata: dict[str, Any] = {}
        self._db_controller = None
        self._db_session_id: int | None = None

    def set_database_controller(self, db_controller) -> None:
        """Attach a DatabaseController so recordings are indexed in SQLite."""
        self._db_controller = db_controller

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def frames_recorded(self) -> int:
        return self._frames_recorded

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    def start_recording(
        self,
        output_dir: str | Path,
        session_metadata: dict[str, Any] | None = None,
        session_name: str | None = None,
    ) -> bool:
        """Begin a new recording session.

        Creates a session directory under output_dir and writes
        session_metadata.yaml with the shared PatternConfig fields.

        Args:
            output_dir: Parent directory for session folders.
            session_metadata: Shared metadata (n_elec, stim_pattern, etc.).
            session_name: Optional custom session name. Defaults to timestamp.
        """
        if self._is_recording:
            self.error.emit("Already recording")
            return False

        try:
            from pyeidors.data.frame_io import write_session_metadata
        except ImportError:
            self.error.emit("pyeidors.data.frame_io not available")
            return False

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%d_%H%M%S")
        name = session_name or f"session_{ts}"
        self._session_dir = output_path / name
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._session_prefix = ts

        self._session_metadata = session_metadata or {}
        self._session_metadata.setdefault("session_start", now.isoformat())

        try:
            meta_path = self._session_dir / "session_metadata.yaml"
            write_session_metadata(meta_path, self._session_metadata)
        except Exception as exc:
            self.error.emit(f"Failed to write session metadata: {exc}")
            self._session_dir = None
            self._session_prefix = ""
            self._session_metadata = {}
            return False

        self._frames_recorded = 0
        self._is_recording = True

        # Register session with the frame database (non-fatal if it fails)
        self._db_session_id = None
        if self._db_controller is not None:
            try:
                self._db_session_id = self._db_controller.register_session(
                    self._session_dir, self._session_metadata
                )
            except Exception as exc:
                log.warning("Failed to register session in DB: %s", exc)

        self.recording_started.emit(str(self._session_dir))
        log.info("Recording started: %s", self._session_dir)
        return True

    def stop_recording(self) -> bool:
        """Stop the current recording session."""
        if not self._is_recording:
            return False
        self._is_recording = False
        count = self._frames_recorded
        self.recording_stopped.emit(count)
        log.info("Recording stopped: %d frames in %s", count, self._session_dir)
        return True

    def save_frame(self, frame: FrameData) -> None:
        """Write a single frame to disk. Call from the new_frame signal handler."""
        if not self._is_recording or self._session_dir is None:
            return

        try:
            from pyeidors.data.frame_io import write_frame_csv, write_frame_yaml
        except ImportError:
            self.error.emit("pyeidors.data.frame_io not available")
            self._is_recording = False
            return

        idx = self._frames_recorded
        base = f"{self._session_prefix}_frame_{idx:04d}"
        csv_path = self._session_dir / f"{base}.csv"
        yaml_path = self._session_dir / f"{base}.yaml"

        try:
            write_frame_csv(csv_path, frame.real, frame.imag)
            write_frame_yaml(yaml_path, frame.to_dict())
            self._frames_recorded += 1
            self.frame_saved.emit(idx, frame.timestamp, str(csv_path))

            # Index the frame in the SQLite database if attached
            if self._db_controller is not None and self._db_session_id is not None:
                try:
                    self._db_controller.register_frame(
                        session_id=self._db_session_id,
                        frame_index=idx,
                        timestamp=frame.timestamp,
                        csv_path=csv_path,
                        yaml_path=yaml_path,
                        metadata=frame.to_dict(),
                    )
                except Exception as exc:
                    log.debug("Failed to index frame %d in DB: %s", idx, exc)
        except Exception as exc:
            self.error.emit(f"Failed to save frame {idx}: {exc}")
