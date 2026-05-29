from __future__ import annotations

import os

from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from eit_app.ui.cache_telemetry_panel import CacheTelemetryDialog


def _get_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_cache_telemetry_dialog_renders_cache_summary() -> None:
    _get_app()
    dialog = CacheTelemetryDialog()
    dialog.set_report(
        {
            "doctor": {
                "cache_manager": {
                    "stats": {"disk_items": 2, "disk_bytes": 3 * 1024 * 1024},
                    "index": {"indexed_entry_count": 1},
                },
                "backend_workers": {"profile_count": 4},
            },
            "background_scheduler": {"active": 1, "pending": 2},
        }
    )

    assert "2" in dialog._summary.text()
    assert "cache_manager" in dialog._text.toPlainText()
    dialog.close()
