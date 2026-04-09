"""Periodic burst-mode acquisition scheduler.

Supports modes like *acquire 1 frame every 5 minutes* or
*acquire 10 frames every 30 seconds*.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from .acquisition_process import AcquisitionProcess
from .ipc_protocol import AcquisitionCommand

log = logging.getLogger(__name__)


class BurstScheduler:
    """Schedule periodic acquisition bursts.

    Args:
        acquisition_process: The process to control.
        interval_seconds: Time between burst starts.
        frames_per_burst: Frames to acquire per burst (0 = continuous).
        callback: Called with ``(burst_index, frames_acquired)`` after each burst.
    """

    def __init__(
        self,
        acquisition_process: AcquisitionProcess,
        interval_seconds: float = 300.0,
        frames_per_burst: int = 1,
        callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self._process = acquisition_process
        self._interval = interval_seconds
        self._frames_per_burst = frames_per_burst
        self._callback = callback
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._burst_count = 0
        self._next_burst_time: float | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._burst_count = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._next_burst_time = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def next_burst_time(self) -> float | None:
        return self._next_burst_time

    @property
    def burst_count(self) -> int:
        return self._burst_count

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._next_burst_time = time.time() + self._interval

            # Start acquisition
            self._process.send_command(AcquisitionCommand.START)
            initial = self._process.frame_count

            if self._frames_per_burst > 0:
                # Wait until enough frames are collected
                deadline = time.time() + self._interval
                while not self._stop_event.is_set():
                    acquired = self._process.frame_count - initial
                    if acquired >= self._frames_per_burst:
                        break
                    if time.time() > deadline:
                        log.warning("Burst timed out after %.1f s", self._interval)
                        break
                    self._stop_event.wait(0.05)

                self._process.send_command(AcquisitionCommand.STOP)
            else:
                # Continuous: just wait for interval
                self._stop_event.wait(self._interval)
                self._process.send_command(AcquisitionCommand.STOP)

            frames = self._process.frame_count - initial
            self._burst_count += 1

            if self._callback is not None:
                self._callback(self._burst_count, frames)

            # Wait for next interval
            remaining = (self._next_burst_time or 0) - time.time()
            if remaining > 0 and not self._stop_event.is_set():
                self._stop_event.wait(remaining)
