"""QApplication bootstrap and entry point for the EIT Workstation."""

import logging
import os
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from eit_app.ui.fonts import configure_runtime_fonts
from eit_app.ui.main_window import EITWorkstation
from eit_app.ui.theme import apply_app_theme


def main() -> int:
    """Launch the EIT Workstation application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app = QApplication(sys.argv)
    app.setApplicationName("EIT Workstation")
    app.setOrganizationName("PyEIDORS")
    configure_runtime_fonts(app)
    apply_app_theme(app)

    window = EITWorkstation()
    window.show()

    auto_quit_ms = os.getenv("EIT_APP_AUTO_QUIT_MS", "").strip()
    if auto_quit_ms:
        try:
            delay_ms = max(0, int(auto_quit_ms))
        except ValueError:
            logging.getLogger(__name__).warning(
                "Ignoring invalid EIT_APP_AUTO_QUIT_MS=%r",
                auto_quit_ms,
            )
        else:
            QTimer.singleShot(delay_ms, app.quit)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
