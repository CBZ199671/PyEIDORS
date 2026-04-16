"""Status bar with explicit engineering state indicators."""

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QLabel, QStatusBar, QWidget

from eit_app.i18n import t, translator
from eit_app.ui.theme import apply_state_chip


# Static maps: state code -> translation key  (stored once at module scope so
# _retranslate() can re-render without re-querying QApplication state).
_LINK_KEYS = {
    "connected": ("status.link.connected", "ready"),
    "connecting": ("status.link.connecting", "warn"),
    "disconnected": ("status.link.disconnected", "error"),
    "error": ("status.link.error", "error"),
}
_POWER_KEYS = {
    "on": ("status.power.on", "ready"),
    "off": ("status.power.off", "idle"),
    "unknown": ("status.power.unknown", "warn"),
}
_ACQ_KEYS = {
    "idle": ("status.acq.idle", "idle"),
    "continuous": ("status.acq.continuous", "active"),
    "scheduled": ("status.acq.scheduled", "active"),
    "finite_run": ("status.acq.finite_run", "active"),
    "stepped_run": ("status.acq.stepped_run", "active"),
    "single_shot": ("status.acq.single_shot", "active"),
}
# Tone semantics (see eit_app.ui.theme.tone_palette):
#   idle    neutral grey     — inactive / nothing to report
#   ready   green             — healthy, connected, warmed up
#   active  blue              — currently doing productive work
#   warn    amber             — transitional / attention needed
#   error   red               — failure / broken
_RECORD_KEYS = {
    "off": ("status.record.off", "idle"),
    "armed": ("status.record.armed", "warn"),
    # Recording IS working correctly — it's an active mode, not an
    # error.  Previous "error" tone painted the chip red during normal
    # captures which misled users into thinking something had gone
    # wrong.
    "recording": ("status.record.recording", "active"),
}
# Mode chip uses a single "active" tone across every tab so users can
# read the chip as "app is live" regardless of which tab they're on.
# Previously each tab had a distinct tone (hardware=active, sim=ready,
# dataset=warn, database=idle) which implied the dataset tab carried a
# warning state and the database tab was dormant — neither is true.
_MODE_KEYS = {
    0: ("status.mode.hardware", "active"),
    1: ("status.mode.simulation", "active"),
    2: ("status.mode.dataset", "active"),
    3: ("status.mode.database", "active"),
}


class EITStatusBar(QStatusBar):
    """Bottom status bar showing link, power, acquisition, and recording state."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._conn_label = self._make_pill("")
        self._power_label = self._make_pill("")
        self._acq_label = self._make_pill("")
        self._rec_label = self._make_pill("")
        self._fps_label = QLabel("")
        self._frame_label = QLabel("")

        for label in (self._fps_label, self._frame_label):
            label.setStyleSheet("padding: 0 10px; color: #243447; font-weight: 700;")

        apply_state_chip(self._conn_label, tone="error", compact=True)
        apply_state_chip(self._power_label, tone="warn", compact=True)
        apply_state_chip(self._acq_label, tone="idle", compact=True)
        apply_state_chip(self._rec_label, tone="idle", compact=True)

        self._mode_label = self._make_pill("")
        apply_state_chip(self._mode_label, tone="active", compact=True)

        self.addPermanentWidget(self._mode_label)
        self.addPermanentWidget(self._conn_label)
        self.addPermanentWidget(self._power_label)
        self.addPermanentWidget(self._acq_label)
        self.addPermanentWidget(self._rec_label)
        self.addPermanentWidget(self._fps_label)
        self.addPermanentWidget(self._frame_label)

        # Cache last known state so _retranslate() can re-render chip text
        # in the new language without asking every upstream slot to re-fire.
        self._state_cache = {
            "connection": "disconnected",
            "power": "unknown",
            "acquisition": "idle",
            "recording": "off",
            "mode": 0,
            "fps": None,           # None -> "status.fps" placeholder
            "frames": None,        # None -> "status.frames" placeholder
        }

        translator().language_changed.connect(self._retranslate)
        self._retranslate()

    @staticmethod
    def _make_pill(text: str) -> QLabel:
        label = QLabel(text)
        return label

    # ------------------------------------------------------------------
    # Slot API — unchanged external signature; internal text is translated.
    # ------------------------------------------------------------------

    @Slot(str)
    def on_connection_changed(self, status: str) -> None:
        self._state_cache["connection"] = status
        self._apply_chip(
            self._conn_label,
            _LINK_KEYS,
            status,
            fallback_key="status.link.other",
            fallback_kwargs={"status": status},
        )

    @Slot(str)
    def on_power_status_changed(self, status: str) -> None:
        self._state_cache["power"] = status
        self._apply_chip(
            self._power_label,
            _POWER_KEYS,
            status,
            fallback_key="status.power.other",
            fallback_kwargs={"status": status},
        )

    @Slot(str)
    def on_acquisition_mode_changed(self, mode: str) -> None:
        self._state_cache["acquisition"] = mode
        self._apply_chip(
            self._acq_label,
            _ACQ_KEYS,
            mode,
            fallback_key="status.acq.other",
            fallback_kwargs={"mode": mode},
        )

    @Slot(bool)
    def on_recording_changed(self, active: bool) -> None:
        if active:
            self.on_recording_status_changed("recording")
        elif self._state_cache["recording"] == "recording":
            self.on_recording_status_changed("off")

    @Slot(str)
    def on_recording_status_changed(self, status: str) -> None:
        self._state_cache["recording"] = status
        self._apply_chip(
            self._rec_label,
            _RECORD_KEYS,
            status,
            fallback_key="status.record.other",
            fallback_kwargs={"status": status},
        )

    @Slot(float)
    def on_fps_updated(self, fps: float) -> None:
        self._state_cache["fps"] = float(fps)
        self._fps_label.setText(t("status.fps_value", value=fps))

    @Slot(int)
    def on_frame_count_changed(self, count: int) -> None:
        self._state_cache["frames"] = int(count)
        self._frame_label.setText(t("status.frames_value", count=count))

    @Slot(int)
    def on_tab_changed(self, index: int) -> None:
        """Update mode indicator when the user switches tabs."""
        self._state_cache["mode"] = index
        key, tone = _MODE_KEYS.get(index, ("status.mode.other", "idle"))
        self._mode_label.setText(t(key, index=index))
        apply_state_chip(self._mode_label, tone=tone, compact=True, emphasized=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_chip(
        self,
        label: QLabel,
        state_map: dict[str, tuple[str, str]],
        state: str,
        *,
        fallback_key: str,
        fallback_kwargs: dict,
    ) -> None:
        if state in state_map:
            key, tone = state_map[state]
            label.setText(t(key))
        else:
            tone = "idle"
            label.setText(t(fallback_key, **fallback_kwargs))
        apply_state_chip(label, tone=tone, compact=True, emphasized=True)

    # ── i18n ──

    def _retranslate(self) -> None:
        """Rerender every chip + FPS / frame counter in the active language."""
        self.on_connection_changed(self._state_cache["connection"])
        self.on_power_status_changed(self._state_cache["power"])
        self.on_acquisition_mode_changed(self._state_cache["acquisition"])
        self.on_recording_status_changed(self._state_cache["recording"])
        self.on_tab_changed(self._state_cache["mode"])

        fps = self._state_cache["fps"]
        if fps is None:
            self._fps_label.setText(t("status.fps"))
        else:
            self._fps_label.setText(t("status.fps_value", value=fps))

        frames = self._state_cache["frames"]
        if frames is None:
            self._frame_label.setText(t("status.frames"))
        else:
            self._frame_label.setText(t("status.frames_value", count=frames))
