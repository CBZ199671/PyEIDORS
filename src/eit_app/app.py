"""QApplication bootstrap and entry point for the EIT Workstation."""

import logging
import os
import sys

from eit_app.runtime_threads import (
    configure_realtime_compute_threads,
    configure_realtime_thread_env,
)

configure_realtime_thread_env()

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from eit_app.i18n import init_from_settings as init_i18n_from_settings
from eit_app.ui.fonts import configure_runtime_fonts
from eit_app.ui.main_window import EITWorkstation
from eit_app.ui.theme import apply_app_theme


def main() -> int:
    """Launch the EIT Workstation application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    thread_info = configure_realtime_compute_threads()
    logging.getLogger(__name__).info(
        "GUI realtime runtime threads configured: %s",
        thread_info,
    )

    app = QApplication(sys.argv)
    app.setApplicationName("EIT Workstation")
    app.setOrganizationName("PyEIDORS")

    # Load the persisted UI language preference.  Must run AFTER
    # setOrganizationName / setApplicationName so QSettings resolves the
    # correct config store.  Falls back to the system locale on first launch.
    init_i18n_from_settings()

    # Pre-initialise gmsh on the main thread.  gmsh.initialize() calls
    # signal.signal(SIGINT, …) internally, and signal.signal() refuses to
    # run from any non-main thread — so if the first invocation happens in
    # a worker (e.g. ForwardSolverController running in a QThread), it
    # crashes with "ValueError: signal only works in main thread".
    #
    # The mesh-generator code path (pyeidors/geometry/optimized_mesh_
    # generator.py) already guards its initialize() call behind
    # `if not gmsh.isInitialized():`, so making sure gmsh is already
    # initialised here lets every subsequent worker-side call skip the
    # signal-handler setup cleanly.
    try:
        import gmsh  # type: ignore[import-not-found]

        if not gmsh.isInitialized():
            gmsh.initialize()
            logging.getLogger(__name__).info(
                "gmsh initialised on main thread for worker-side mesh generation"
            )

        import atexit

        def _finalize_gmsh() -> None:
            try:
                if gmsh.isInitialized():
                    gmsh.finalize()
            except Exception:  # pragma: no cover — best-effort shutdown
                pass

        atexit.register(_finalize_gmsh)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "gmsh pre-initialisation skipped (mesh generation may fail later): %s",
            exc,
        )

    # -----------------------------------------------------------------
    # Tolerant signal.signal monkey-patch
    # -----------------------------------------------------------------
    # gmshio.read_from_msh (and a few other pyeidors / dolfinx paths)
    # re-call gmsh.initialize() unconditionally on mesh load, even when
    # gmsh has already been initialised.  That inner call tries to
    # register a SIGINT handler via signal.signal(), which Python refuses
    # to execute from any non-main thread and raises:
    #     ValueError: signal only works in main thread of the main interpreter
    #
    # Our reconstruction / forward workers run in QThreads, so the
    # cached-mesh load path crashes, cache is marked as failed, and the
    # mesh is regenerated from scratch every time (noticeably slow).
    #
    # Wrap signal.signal so worker threads get a graceful no-op instead
    # of a hard crash.  The main thread keeps the real behaviour so Ctrl+C
    # during development continues to work normally.
    import signal as _signal
    import threading as _threading

    _MAIN_THREAD = _threading.main_thread()
    _original_signal = _signal.signal

    def _tolerant_signal(signum, handler):  # noqa: ANN001
        if _threading.current_thread() is _MAIN_THREAD:
            return _original_signal(signum, handler)
        # On worker threads, refuse to register global signal handlers.
        # Returning SIG_DFL matches what gmsh.initialize() would have
        # captured as the "previous" handler on a freshly-created thread.
        return _signal.SIG_DFL

    _signal.signal = _tolerant_signal  # type: ignore[assignment]
    logging.getLogger(__name__).info(
        "signal.signal wrapped: worker threads now no-op (was: raise ValueError)"
    )

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
