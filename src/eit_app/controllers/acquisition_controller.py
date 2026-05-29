"""Manages the acquisition process lifecycle and polls the ring buffer.

The main GUI thread uses a QTimer to poll the shared-memory ring buffer
at ~30fps, converting raw data into FrameData signals for the UI layer.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, QTimer, Signal

from eit_app.models.frame_model import FrameData

if TYPE_CHECKING:
    from eit_app.acquisition.acquisition_process import AcquisitionProcess
    from eit_app.acquisition.ring_buffer import FrameRingBuffer

log = logging.getLogger(__name__)

# Poll interval in milliseconds (~30 fps)
_POLL_INTERVAL_MS = 33


class AcquisitionController(QObject):
    """Controls the acquisition process and emits frames for the UI.

    Signals:
        new_frame: Emitted for each new frame read from the ring buffer.
        status_changed: Emitted when acquisition status changes.
        error: Emitted on acquisition errors.
        fps_updated: Emitted with current frames-per-second.
    """

    new_frame = Signal(object)  # FrameData
    status_changed = Signal(str)
    error = Signal(str)
    fps_updated = Signal(float)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process: AcquisitionProcess | None = None
        self._ring_buffer: FrameRingBuffer | None = None
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_buffer)
        self._is_active = False

        # FPS tracking
        self._last_write_count: int = 0
        self._fps_timestamps: list[float] = []
        self._total_frames: int = 0
        self._frame_metadata: dict[str, Any] = {}

    def configure(
        self,
        process: AcquisitionProcess,
        ring_buffer: FrameRingBuffer,
        frame_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Attach an acquisition process and its ring buffer."""
        self._process = process
        self._ring_buffer = ring_buffer
        self._frame_metadata = dict(frame_metadata or {})

    def start(self, *, activate_device: bool = True) -> None:
        """Start the acquisition process and begin polling."""
        if self._process is None or self._ring_buffer is None:
            self.error.emit("Acquisition not configured")
            return

        if not self._process.is_alive():
            self._process.start()

        if activate_device:
            from eit_app.acquisition.ipc_protocol import AcquisitionCommand

            self._process.send_command(AcquisitionCommand.START)
        self._last_write_count = self._ring_buffer.write_count
        self._total_frames = 0
        self._fps_timestamps.clear()
        self.fps_updated.emit(0.0)
        self._poll_timer.start(_POLL_INTERVAL_MS)
        self._is_active = True
        self.status_changed.emit("running")
        log.info("Acquisition started, polling at %d ms", _POLL_INTERVAL_MS)

    def capture_one(self) -> None:
        """Capture exactly one frame and begin polling for the result."""
        if self._process is None or self._ring_buffer is None:
            self.error.emit("Acquisition not configured")
            return

        if not self._process.is_alive():
            self._process.start()

        from eit_app.acquisition.ipc_protocol import AcquisitionCommand

        self._process.send_command(AcquisitionCommand.CAPTURE_ONE)
        self._last_write_count = self._ring_buffer.write_count
        self._total_frames = 0
        self._fps_timestamps.clear()
        self.fps_updated.emit(0.0)
        self._poll_timer.start(_POLL_INTERVAL_MS)
        self._is_active = True
        self.status_changed.emit("single_shot")
        log.info(
            "Single-frame acquisition started, polling at %d ms", _POLL_INTERVAL_MS
        )

    def stop(self, *, deactivate_device: bool = True) -> None:
        """Stop acquisition and polling."""
        if not self._is_active and not self._poll_timer.isActive():
            return
        self._poll_timer.stop()
        self._fps_timestamps.clear()
        self.fps_updated.emit(0.0)
        self._is_active = False

        if deactivate_device and self._process is not None:
            from eit_app.acquisition.ipc_protocol import AcquisitionCommand

            self._process.send_command(AcquisitionCommand.STOP)

        self.status_changed.emit("idle")
        log.info("Acquisition stopped")

    def shutdown(self) -> None:
        """Fully shut down the acquisition process."""
        self.stop()
        if self._process is not None:
            if self._process.pid is not None:
                from eit_app.acquisition.ipc_protocol import AcquisitionCommand

                # Send STOP first so the inner read loop can break,
                # then SHUTDOWN so the outer loop exits cleanly.
                try:
                    self._process.send_command(AcquisitionCommand.STOP)
                except Exception:
                    pass
                try:
                    self._process.send_command(AcquisitionCommand.SHUTDOWN)
                except Exception:
                    pass
                self._process.join(timeout=5.0)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=2.0)
                    log.debug("Acquisition process terminated on shutdown")

        if self._ring_buffer is not None:
            self._ring_buffer.close()

    def _poll_buffer(self) -> None:
        """Called by QTimer. Reads new frames from the ring buffer."""
        if self._ring_buffer is None:
            return

        current_count = self._ring_buffer.write_count

        # Check for errors from the process
        if self._process is not None:
            for err_msg in self._process.get_errors():
                self.error.emit(err_msg)

        # Read all new frames since last poll
        if current_count <= self._last_write_count:
            return

        # Only emit the latest frame for display (skip intermediate ones)
        # but count all for FPS calculation
        n_new = current_count - self._last_write_count
        self._last_write_count = current_count

        result = self._ring_buffer.read_latest()
        if result is None:
            return

        real, imag, timestamp, frame_index = result
        frame = FrameData(
            real=real,
            imag=imag,
            timestamp=timestamp,
            frame_index=frame_index,
            metadata=dict(self._frame_metadata),
        )
        self._total_frames += n_new
        self.new_frame.emit(frame)

        # Update FPS
        now = time.monotonic()
        self._fps_timestamps.extend([now] * n_new)
        # Keep only timestamps from the last second
        cutoff = now - 1.0
        self._fps_timestamps = [t for t in self._fps_timestamps if t > cutoff]
        fps = len(self._fps_timestamps)
        self.fps_updated.emit(float(fps))

    @property
    def total_frames(self) -> int:
        return self._total_frames
