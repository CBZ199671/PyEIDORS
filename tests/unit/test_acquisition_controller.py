from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QApplication

from eit_app.controllers import acquisition_controller as ac


def _get_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_acquisition_stop_is_idempotent_and_logs_once(
    monkeypatch,
) -> None:
    _get_app()
    controller = ac.AcquisitionController()
    statuses: list[str] = []
    logs: list[str] = []
    controller.status_changed.connect(statuses.append)
    monkeypatch.setattr(
        ac.log,
        "info",
        lambda message, *args: logs.append(message % args if args else message),
    )

    controller._is_active = True

    controller.stop()
    controller.stop()

    assert statuses == ["idle"]
    assert logs == ["Acquisition stopped"]


def test_v560_poll_buffer_reuses_ring_buffer_arrays_without_extra_copy() -> None:
    _get_app()
    controller = ac.AcquisitionController()
    emitted = []
    controller.new_frame.connect(emitted.append)

    real = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    imag = np.array([0.5, 0.25, 0.0], dtype=np.float64)

    class _RingBuffer:
        write_count = 1

        def read_latest(self):
            return real, imag, 12.5, 9

    controller._ring_buffer = _RingBuffer()
    controller._frame_metadata = {"source": "test"}

    controller._poll_buffer()

    assert len(emitted) == 1
    frame = emitted[0]
    assert frame.real is real
    assert frame.imag is imag
    assert frame.timestamp == 12.5
    assert frame.frame_index == 9
    assert frame.metadata == {"source": "test"}
