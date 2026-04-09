"""Status bar with explicit engineering state indicators."""

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QLabel, QStatusBar, QWidget

from eit_app.ui.theme import apply_state_chip


class EITStatusBar(QStatusBar):
    """Bottom status bar showing link, power, acquisition, and recording state."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._conn_label = self._make_pill("Link: Down")
        self._power_label = self._make_pill("Power: Unknown")
        self._acq_label = self._make_pill("Acq: Idle")
        self._rec_label = self._make_pill("Record: Off")
        self._fps_label = QLabel("FPS: --")
        self._frame_label = QLabel("Frames: 0")

        for label in (self._fps_label, self._frame_label):
            label.setStyleSheet("padding: 0 10px; color: #d9e2f2; font-weight: 600;")

        apply_state_chip(self._conn_label, tone="error", compact=True)
        apply_state_chip(self._power_label, tone="warn", compact=True)
        apply_state_chip(self._acq_label, tone="idle", compact=True)
        apply_state_chip(self._rec_label, tone="idle", compact=True)

        self.addPermanentWidget(self._conn_label)
        self.addPermanentWidget(self._power_label)
        self.addPermanentWidget(self._acq_label)
        self.addPermanentWidget(self._rec_label)
        self.addPermanentWidget(self._fps_label)
        self.addPermanentWidget(self._frame_label)

    @staticmethod
    def _make_pill(text: str) -> QLabel:
        label = QLabel(text)
        return label

    @Slot(str)
    def on_connection_changed(self, status: str) -> None:
        colors = {
            "connected": ("#0b6b2d", "#dff6e6", "Link: Verified"),
            "connecting": ("#9a6700", "#fff1c2", "Link: Connecting"),
            "disconnected": ("#a94442", "#fdecea", "Link: Down"),
            "error": ("#a94442", "#fdecea", "Link: Error"),
        }
        _fg, _bg, text = colors.get(status, ("#666666", "#eceff4", f"Link: {status}"))
        self._conn_label.setText(text)
        apply_state_chip(
            self._conn_label,
            tone={
                "connected": "ready",
                "connecting": "warn",
                "disconnected": "error",
                "error": "error",
            }.get(status, "idle"),
            compact=True,
            emphasized=True,
        )

    @Slot(str)
    def on_power_status_changed(self, status: str) -> None:
        colors = {
            "on": ("#0b6b2d", "#dff6e6", "Power: ON"),
            "off": ("#666666", "#eceff4", "Power: OFF"),
            "unknown": ("#7a6a00", "#fff7cc", "Power: Unknown"),
        }
        _fg, _bg, text = colors.get(status, ("#666666", "#eceff4", f"Power: {status}"))
        self._power_label.setText(text)
        apply_state_chip(
            self._power_label,
            tone={"on": "ready", "off": "idle", "unknown": "warn"}.get(status, "idle"),
            compact=True,
            emphasized=True,
        )

    @Slot(str)
    def on_acquisition_mode_changed(self, mode: str) -> None:
        mapping = {
            "idle": ("#666666", "#eceff4", "Acq: Idle"),
            "continuous": ("#005f99", "#d8efff", "Acq: Continuous"),
            "scheduled": ("#5a3e9d", "#ece3ff", "Acq: Scheduled"),
            "single_shot": ("#9a4d00", "#ffe6cc", "Acq: Single Frame"),
        }
        _fg, _bg, text = mapping.get(mode, ("#666666", "#eceff4", f"Acq: {mode}"))
        self._acq_label.setText(text)
        apply_state_chip(
            self._acq_label,
            tone={
                "idle": "idle",
                "continuous": "active",
                "scheduled": "active",
                "single_shot": "active",
            }.get(mode, "idle"),
            compact=True,
            emphasized=True,
        )

    @Slot(bool)
    def on_recording_changed(self, active: bool) -> None:
        if active:
            self.on_recording_status_changed("recording")
        elif self._rec_label.text() == "Record: Writing":
            self.on_recording_status_changed("off")

    @Slot(str)
    def on_recording_status_changed(self, status: str) -> None:
        mapping = {
            "off": ("#666666", "#eceff4", "Record: Off"),
            "armed": ("#9a6700", "#fff1c2", "Record: Armed"),
            "recording": ("#b42318", "#fde2e1", "Record: Writing"),
        }
        _fg, _bg, text = mapping.get(status, ("#666666", "#eceff4", f"Record: {status}"))
        self._rec_label.setText(text)
        apply_state_chip(
            self._rec_label,
            tone={"off": "idle", "armed": "warn", "recording": "error"}.get(status, "idle"),
            compact=True,
            emphasized=True,
        )

    @Slot(float)
    def on_fps_updated(self, fps: float) -> None:
        self._fps_label.setText(f"FPS: {fps:.1f}")

    @Slot(int)
    def on_frame_count_changed(self, count: int) -> None:
        self._frame_label.setText(f"Frames: {count}")
