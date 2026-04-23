from __future__ import annotations

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
