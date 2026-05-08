from __future__ import annotations

from PySide6.QtWidgets import QApplication

from eit_app.controllers.forward_solver_controller import ForwardSolverController
from eit_app.controllers.reconstruction_controller import ReconstructionController
from eit_app.ui.conductivity_image_widget import ConductivityImageWidget


def _get_app() -> QApplication:
    return QApplication.instance() or QApplication([])


class _UnremovableArtist:
    def __init__(self) -> None:
        self.visible: bool | None = None

    def remove(self) -> None:
        raise NotImplementedError("cannot remove artist")

    def set_visible(self, visible: bool) -> None:
        self.visible = visible


class _UnresponsiveThread:
    def __init__(self) -> None:
        self.running = True
        self.interrupted = False
        self.quit_called = False
        self.terminated = False
        self.deleted = False
        self.wait_calls: list[int] = []

    def isRunning(self) -> bool:  # noqa: N802 - Qt API shape
        return self.running

    def requestInterruption(self) -> None:  # noqa: N802 - Qt API shape
        self.interrupted = True

    def quit(self) -> None:
        self.quit_called = True

    def wait(self, timeout_ms: int) -> bool:
        self.wait_calls.append(timeout_ms)
        return not self.running

    def terminate(self) -> None:
        self.terminated = True
        self.running = False

    def deleteLater(self) -> None:  # noqa: N802 - Qt API shape
        self.deleted = True


class _CancellableWorker:
    def __init__(self) -> None:
        self.cancelled = False
        self.deleted = False

    def cancel(self) -> None:
        self.cancelled = True

    def deleteLater(self) -> None:  # noqa: N802 - Qt API shape
        self.deleted = True


def test_conductivity_widget_hides_unremovable_electrode_artist() -> None:
    _get_app()
    widget = ConductivityImageWidget()
    artist = _UnremovableArtist()
    try:
        widget._electrode_collection = artist

        widget._discard_electrode_collection()

        assert widget._electrode_collection is None
        assert artist.visible is False
    finally:
        widget.close()


def test_forward_controller_shutdown_terminates_unresponsive_worker_thread() -> None:
    controller = ForwardSolverController()
    thread = _UnresponsiveThread()
    worker = _CancellableWorker()
    controller._thread = thread
    controller._worker = worker

    controller.shutdown()

    assert worker.cancelled
    assert thread.interrupted
    assert thread.quit_called
    assert thread.terminated
    assert thread.deleted
    assert worker.deleted
    assert controller._thread is None
    assert controller._worker is None


def test_reconstruction_controller_shutdown_terminates_unresponsive_worker_thread() -> (
    None
):
    controller = ReconstructionController()
    thread = _UnresponsiveThread()
    worker = _CancellableWorker()
    controller._busy = True
    controller._thread = thread
    controller._worker = worker

    controller.shutdown()

    assert worker.cancelled
    assert thread.interrupted
    assert thread.quit_called
    assert thread.terminated
    assert thread.deleted
    assert worker.deleted
    assert not controller.is_busy
    assert controller._thread is None
    assert controller._worker is None
