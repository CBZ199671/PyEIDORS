"""Cache telemetry dialog for runtime cache/JIT/worker visibility."""

from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
)

from eit_app.i18n import t, translator


class CacheTelemetryDialog(QDialog):
    """Small modeless panel for cache health and maintenance actions."""

    refresh_requested = Signal()
    gc_requested = Signal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.resize(860, 620)

        self._summary = QLabel()
        self._summary.setWordWrap(True)

        self._text = QTextEdit()
        self._text.setReadOnly(True)

        self._refresh_btn = QPushButton()
        self._refresh_btn.clicked.connect(self.refresh_requested.emit)

        self._gc_btn = QPushButton()
        self._gc_btn.clicked.connect(self._emit_gc_requested)

        self._max_gib = QSpinBox()
        self._max_gib.setRange(1, 1024)
        self._max_gib.setValue(8)
        self._max_gib.setSuffix(" GiB")

        self._include_worker = QCheckBox()
        self._include_worker.setChecked(False)

        self._include_legacy = QCheckBox()
        self._include_legacy.setChecked(True)

        controls = QHBoxLayout()
        controls.addWidget(self._refresh_btn)
        controls.addWidget(self._gc_btn)
        controls.addWidget(QLabel(t("cache.telemetry.max_size")))
        controls.addWidget(self._max_gib)
        controls.addWidget(self._include_worker)
        controls.addWidget(self._include_legacy)
        controls.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(self._summary)
        layout.addLayout(controls)
        layout.addWidget(self._text, 1)

        translator().language_changed.connect(self._retranslate)
        self._retranslate()

    def set_busy(self, message: str) -> None:
        self._summary.setText(message)
        self._refresh_btn.setEnabled(False)
        self._gc_btn.setEnabled(False)

    def set_report(self, report: dict[str, Any]) -> None:
        self._refresh_btn.setEnabled(True)
        self._gc_btn.setEnabled(True)
        self._summary.setText(_summary_text(report))
        self._text.setPlainText(json.dumps(report, indent=2, ensure_ascii=False))

    def set_error(self, message: str) -> None:
        self._refresh_btn.setEnabled(True)
        self._gc_btn.setEnabled(True)
        self._summary.setText(message)

    def _emit_gc_requested(self) -> None:
        self.gc_requested.emit(
            {
                "max_bytes": int(self._max_gib.value()) * 1024**3,
                "include_worker_cache": bool(self._include_worker.isChecked()),
                "include_legacy_arrays": bool(self._include_legacy.isChecked()),
            }
        )

    def _retranslate(self) -> None:
        self.setWindowTitle(t("cache.telemetry.title"))
        self._refresh_btn.setText(t("cache.telemetry.refresh"))
        self._gc_btn.setText(t("cache.telemetry.gc"))
        self._include_worker.setText(t("cache.telemetry.include_worker"))
        self._include_legacy.setText(t("cache.telemetry.include_legacy"))
        if not self._summary.text():
            self._summary.setText(t("cache.telemetry.idle"))


def _summary_text(report: dict[str, Any]) -> str:
    doctor = report.get("doctor", report)
    manager = doctor.get("cache_manager", {}) if isinstance(doctor, dict) else {}
    stats = manager.get("stats", {}) if isinstance(manager, dict) else {}
    index = manager.get("index", {}) if isinstance(manager, dict) else {}
    workers = doctor.get("backend_workers", {}) if isinstance(doctor, dict) else {}
    scheduler = report.get("background_scheduler", {})
    return t(
        "cache.telemetry.summary",
        disk_items=int(stats.get("disk_items", 0) or 0),
        disk_mib=float(stats.get("disk_bytes", 0) or 0) / (1024.0 * 1024.0),
        indexed=int(index.get("indexed_entry_count", 0) or 0),
        workers=int(workers.get("profile_count", 0) or 0),
        active=int(scheduler.get("active", 0) or 0),
        pending=int(scheduler.get("pending", 0) or 0),
    )
