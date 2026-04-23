from __future__ import annotations

import eit_app.hardware.windows_serial_transport as bridge_module
from eit_app.hardware.windows_serial_transport import WindowsSerialTransport


def test_windows_serial_transport_open_retries_transient_access_denied(
    monkeypatch,
) -> None:
    transport = WindowsSerialTransport("COM4", 115200)
    attempts = {"count": 0}
    sleeps: list[float] = []
    closes: list[str] = []

    def _fake_open_once() -> None:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError(
                "Windows serial bridge failed: ERROR 使用“0”个参数调用“Open”时发生异常:“对端口“COM4”的访问被拒绝。”"
            )

    monkeypatch.setattr(transport, "_open_once", _fake_open_once)
    monkeypatch.setattr(transport, "close", lambda: closes.append("close"))
    monkeypatch.setattr(
        bridge_module.time, "sleep", lambda seconds: sleeps.append(seconds)
    )

    transport.open()

    assert attempts["count"] == 3
    assert sleeps == [0.25, 0.55]
    assert closes == ["close", "close", "close", "close", "close"]


def test_windows_serial_transport_open_does_not_retry_non_retryable_error(
    monkeypatch,
) -> None:
    transport = WindowsSerialTransport("COM4", 115200)
    attempts = {"count": 0}

    def _fake_open_once() -> None:
        attempts["count"] += 1
        raise RuntimeError(
            "Windows serial bridge failed: ERROR Cannot find the file specified."
        )

    monkeypatch.setattr(transport, "_open_once", _fake_open_once)
    monkeypatch.setattr(transport, "close", lambda: None)

    try:
        transport.open()
    except RuntimeError as exc:
        assert "Cannot find the file specified" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")

    assert attempts["count"] == 1
