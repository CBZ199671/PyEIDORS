from __future__ import annotations

import threading
import time

from PySide6.QtWidgets import QApplication

from eit_app.controllers.forward_solver_controller import (
    ForwardSolverController,
    ForwardSolverRequest,
    _ForwardSolverWorker,
)
from eit_app.controllers.reconstruction_controller import (
    ReconstructionController,
    _ReconstructionWorker,
)
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


def test_v146_forward_worker_cancel_stops_external_persistent_backend(
    monkeypatch,
    tmp_path,
) -> None:
    worker = _ForwardSolverWorker(ForwardSolverRequest())
    worker._backend_profile = "cuda"
    calls: list[tuple[object, str]] = []

    monkeypatch.setattr(
        "eit_app.controllers.forward_solver_controller._repo_root",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "eit_app.backend_worker_pool.stop_persistent_backend_worker",
        lambda *, repo, profile: calls.append((repo, profile)) or True,
    )

    worker.cancel()

    assert worker._cancel_requested
    assert calls == [(tmp_path, "cuda")]


def test_v146_reconstruction_worker_cancel_stops_external_persistent_backend(
    monkeypatch,
    tmp_path,
) -> None:
    worker = _ReconstructionWorker()
    worker._backend_profile = "cuda"
    calls: list[tuple[object, str]] = []

    monkeypatch.setattr(
        "eit_app.controllers.reconstruction_controller._repo_root",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "eit_app.backend_worker_pool.stop_persistent_backend_worker",
        lambda *, repo, profile: calls.append((repo, profile)) or True,
    )

    worker.cancel()

    assert worker._cancel_requested
    assert calls == [(tmp_path, "cuda")]


def test_v146_backend_worker_stop_does_not_wait_for_request_lock(tmp_path) -> None:
    import eit_app.backend_worker_pool as pool

    worker = pool._PersistentBackendWorker(repo=tmp_path, profile="cuda")
    key = (str(tmp_path.resolve()), "cuda")
    lock_acquired = threading.Event()
    release_lock = threading.Event()

    def hold_request_lock() -> None:
        with worker._lock:
            lock_acquired.set()
            release_lock.wait(timeout=1.0)

    thread = threading.Thread(target=hold_request_lock)
    thread.start()
    assert lock_acquired.wait(timeout=1.0)
    with pool._POOL_LOCK:
        pool._POOL[key] = worker
    try:
        start = time.monotonic()

        assert pool.stop_persistent_backend_worker(repo=tmp_path, profile="cuda")

        assert time.monotonic() - start < 0.2
        with pool._POOL_LOCK:
            assert key not in pool._POOL
    finally:
        release_lock.set()
        thread.join(timeout=1.0)
        with pool._POOL_LOCK:
            pool._POOL.pop(key, None)


def test_forward_controller_rejects_second_solve_while_busy() -> None:
    _get_app()
    controller = ForwardSolverController()
    thread = _UnresponsiveThread()
    controller._thread = thread
    messages: list[str] = []
    controller.error.connect(messages.append)

    accepted = controller.solve(ForwardSolverRequest())

    assert controller.is_busy
    assert accepted is False
    assert messages == ["A forward solve is already running."]


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
