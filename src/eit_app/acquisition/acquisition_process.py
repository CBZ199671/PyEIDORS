"""Dedicated acquisition process that reads frames from hardware into ring buffer."""

from __future__ import annotations

import ctypes
import logging
import multiprocessing as mp
import traceback
from multiprocessing import Queue, Value
from typing import Any, Callable

from .ipc_protocol import AcquisitionCommand, AcquisitionStatus
from .ring_buffer import FrameRingBuffer

log = logging.getLogger(__name__)


class AcquisitionProcess(mp.Process):
    """Acquisition worker process.

    Owns a HardwareDevice instance (created via *device_factory* inside the
    worker to avoid pickling serial ports), reads frames in a loop, and
    writes them to a :class:`FrameRingBuffer`.

    Communication:
        - ``cmd_queue``:  main -> worker (AcquisitionCommand)
        - ``status``:     worker -> main (AcquisitionStatus as int)
        - ``error_queue``: worker -> main (error strings)
        - ring buffer:    worker -> main (frame data via shared memory)
    """

    def __init__(
        self,
        device_factory: Callable[..., Any],
        device_config: dict[str, Any],
        buffer_name: str,
        buffer_capacity: int = 256,
        n_meas: int = 208,
    ) -> None:
        super().__init__(daemon=True)
        self._device_factory = device_factory
        self._device_config = device_config
        self._buffer_name = buffer_name
        self._buffer_capacity = buffer_capacity
        self._n_meas = n_meas
        self._cmd_queue: Queue[AcquisitionCommand] = Queue()
        self._error_queue: Queue[str] = Queue()
        self._status = Value(ctypes.c_int, AcquisitionStatus.IDLE.value)
        self._frame_count = Value(ctypes.c_long, 0)

    @property
    def status(self) -> AcquisitionStatus:
        return AcquisitionStatus(self._status.value)

    @property
    def frame_count(self) -> int:
        return self._frame_count.value

    def send_command(
        self, cmd: AcquisitionCommand, payload: dict | None = None
    ) -> None:
        self._cmd_queue.put(cmd)

    def get_errors(self) -> list[str]:
        errors: list[str] = []
        while not self._error_queue.empty():
            try:
                errors.append(self._error_queue.get_nowait())
            except Exception:
                break
        return errors

    def run(self) -> None:
        """Main loop in the worker process."""
        ring = FrameRingBuffer(
            capacity=self._buffer_capacity,
            n_meas=self._n_meas,
            name=self._buffer_name,
            create=False,
        )
        device = None
        device_connected = False

        try:
            device = self._device_factory(**self._device_config)
            self._status.value = AcquisitionStatus.IDLE.value

            while True:
                # Wait for a command; blocks up to 0.1s to avoid busy-spin
                cmd = self._drain_command(timeout=0.1)

                if cmd == AcquisitionCommand.SHUTDOWN:
                    self._status.value = AcquisitionStatus.SHUTDOWN.value
                    break

                if cmd == AcquisitionCommand.START:
                    if not device_connected:
                        self._status.value = AcquisitionStatus.CONNECTING.value
                        device.connect()
                        device_connected = True
                    device.start_measurement()
                    self._status.value = AcquisitionStatus.RUNNING.value

                    # Acquisition loop
                    while True:
                        inner_cmd = self._drain_command()
                        if inner_cmd in (
                            AcquisitionCommand.STOP,
                            AcquisitionCommand.SHUTDOWN,
                        ):
                            device.stop_measurement()
                            if device_connected:
                                device.disconnect()
                                device_connected = False
                            self._status.value = AcquisitionStatus.STOPPING.value
                            if inner_cmd == AcquisitionCommand.SHUTDOWN:
                                self._status.value = AcquisitionStatus.SHUTDOWN.value
                                return
                            self._status.value = AcquisitionStatus.IDLE.value
                            break

                        frame = device.read_frame()
                        idx = self._frame_count.value
                        ring.write(frame.real, frame.imag, frame.timestamp, idx)
                        self._frame_count.value = idx + 1

                if cmd == AcquisitionCommand.CAPTURE_ONE:
                    if not device_connected:
                        self._status.value = AcquisitionStatus.CONNECTING.value
                        device.connect()
                        device_connected = True
                    device.start_measurement()
                    self._status.value = AcquisitionStatus.RUNNING.value
                    frame = device.read_frame()
                    idx = self._frame_count.value
                    ring.write(frame.real, frame.imag, frame.timestamp, idx)
                    self._frame_count.value = idx + 1
                    device.stop_measurement()
                    if device_connected:
                        device.disconnect()
                        device_connected = False
                    self._status.value = AcquisitionStatus.IDLE.value

                if cmd == AcquisitionCommand.STOP:
                    if device_connected:
                        try:
                            device.stop_measurement()
                        except Exception:
                            pass
                        try:
                            device.disconnect()
                        except Exception:
                            pass
                        device_connected = False
                    self._status.value = AcquisitionStatus.IDLE.value

        except Exception:
            self._status.value = AcquisitionStatus.ERROR.value
            self._error_queue.put(traceback.format_exc())
        finally:
            if device is not None and device_connected:
                try:
                    device.disconnect()
                except Exception:
                    pass
            ring.close()

    def _drain_command(self, timeout: float | None = None) -> AcquisitionCommand | None:
        """Read the next command from the queue.

        Args:
            timeout: If *None* (default), non-blocking.  If a float,
                     block for at most *timeout* seconds.
        """
        try:
            if timeout is None:
                return self._cmd_queue.get_nowait()
            return self._cmd_queue.get(timeout=timeout)
        except Exception:
            return None
