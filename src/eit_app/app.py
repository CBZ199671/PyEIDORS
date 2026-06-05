"""QApplication bootstrap and entry point for the EIT Workstation."""

import logging
import os
import sys
from pathlib import Path

from eit_app.runtime_threads import (
    configure_realtime_compute_threads,
    configure_realtime_thread_env,
)

configure_realtime_thread_env()

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from eit_app.i18n import init_from_settings as init_i18n_from_settings
from eit_app.models.precision import init_precision_from_settings
from eit_app.ui.fonts import configure_runtime_fonts
from eit_app.ui.main_window import EITWorkstation
from eit_app.ui.theme import apply_app_theme


_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_ENV_VALUES


def _running_under_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(errors="ignore").lower()
    except OSError:
        return False


def _wayland_display_available() -> bool:
    display = os.environ.get("WAYLAND_DISPLAY", "").strip()
    if not display:
        return False

    candidates: list[Path] = []
    display_path = Path(display)
    if display_path.is_absolute():
        candidates.append(display_path)
    else:
        runtime_dir = os.environ.get("XDG_RUNTIME_DIR", "").strip()
        if runtime_dir:
            candidates.append(Path(runtime_dir) / display)
        candidates.append(Path("/mnt/wslg/runtime-dir") / display)

    if candidates:
        return any(path.exists() for path in candidates)

    return True


def _x11_display_available() -> bool:
    display = os.environ.get("DISPLAY", "").strip()
    if not display:
        return False
    if not display.startswith(":"):
        return True

    display_id = display[1:].split(".", 1)[0]
    if not display_id:
        return True
    candidates = (
        Path("/tmp/.X11-unix") / f"X{display_id}",
        Path("/mnt/wslg/.X11-unix") / f"X{display_id}",
    )
    return any(path.exists() for path in candidates)


def _wslg_display_unavailable(reason: str) -> SystemExit:
    wayland_display = os.environ.get("WAYLAND_DISPLAY", "<unset>")
    display = os.environ.get("DISPLAY", "<unset>")
    return SystemExit(
        "WSLg display is not reachable; cannot start the interactive Qt GUI. "
        f"{reason} "
        f"(WAYLAND_DISPLAY={wayland_display!r}, DISPLAY={display!r}). "
        "Close WSL GUI apps, run `wsl.exe --shutdown` from Windows, then reopen "
        "Ubuntu/WSL. For headless smoke checks only, run with "
        "`QT_QPA_PLATFORM=offscreen`."
    )


def _configure_qt_hidpi_defaults() -> None:
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_SCALE_FACTOR_ROUNDING_POLICY", "PassThrough")


def _configure_qt_platform_for_embedded_vtk() -> None:
    """Prefer native Wayland on WSLg for crisp HiDPI rendering.

    XWayland/XCB is stable for embedded VTK, but it is visibly soft on
    HiDPI Windows displays because WSLg has to scale the X11 surface.
    The 3D widget already keeps the unsafe embedded-VTK path disabled
    on WSLg unless XCB is explicitly requested, so the main window can
    stay on Qt/Wayland by default.

    ``QT_QPA_PLATFORM`` is still honoured when the caller pins it.  For
    legacy VTK/X11 experiments set ``EIT_APP_USE_QT_XCB=1``; the older
    ``EIT_APP_USE_QT_WAYLAND=1`` remains accepted but is now the default
    when WSLg exposes ``WAYLAND_DISPLAY``.

    Outside WSL, Linux defaults follow whatever PySide6 picks (no env is
    set).  macOS / Windows are unaffected.
    """
    _configure_qt_hidpi_defaults()
    # Honour anything the user has already pinned — we only fill the
    # blanks here, never override an explicit choice.
    if os.environ.get("QT_QPA_PLATFORM"):
        return
    if not _running_under_wsl():
        return

    if _env_flag("EIT_APP_USE_QT_XCB") or _env_flag("EIT_APP_DISABLE_QT_WAYLAND"):
        if not _x11_display_available():
            raise _wslg_display_unavailable(
                "Qt XCB was requested, but the X11 socket is unavailable."
            )
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        # Disable MIT-SHM: WSLg's XWayland doesn't support shared-memory
        # pixmaps and Qt's BadAccess complaints flood the journal otherwise.
        os.environ.setdefault("QT_X11_NO_MITSHM", "1")
        logging.getLogger(__name__).info(
            "WSLg detected; using Qt XCB platform because XCB was requested"
        )
        return

    wayland_available = _wayland_display_available()
    x11_available = _x11_display_available()

    if wayland_available:
        # The semicolon form lets Qt fall back to XCB if the Wayland plugin
        # is unavailable, while keeping Wayland first for crisp text.
        os.environ["QT_QPA_PLATFORM"] = "wayland;xcb"
        logging.getLogger(__name__).info(
            "WSLg detected; using Qt Wayland platform for crisp HiDPI rendering "
            "(set EIT_APP_USE_QT_XCB=1 to force XCB)"
        )
        return

    if os.environ.get("WAYLAND_DISPLAY"):
        logging.getLogger(__name__).warning(
            "WSLg WAYLAND_DISPLAY is set but no Wayland socket is reachable; "
            "falling back to XCB if available"
        )

    if x11_available:
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        os.environ.setdefault("QT_X11_NO_MITSHM", "1")
        logging.getLogger(__name__).info(
            "WSLg detected without WAYLAND_DISPLAY; falling back to Qt XCB platform"
        )
        return

    raise _wslg_display_unavailable(
        "Neither the Wayland nor the X11 display socket is available."
    )


def main() -> int:
    """Launch the EIT Workstation application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _configure_qt_platform_for_embedded_vtk()
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
    init_precision_from_settings()

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

        def _quiet_gmsh_terminal() -> None:
            try:
                if gmsh.isInitialized():
                    gmsh.option.setNumber("General.Terminal", 0)
            except Exception:  # pragma: no cover — best-effort noise control
                pass

        if not getattr(gmsh, "_eit_app_quiet_initialize_wrapped", False):
            _original_gmsh_initialize = gmsh.initialize

            def _quiet_gmsh_initialize(*args, **kwargs):  # noqa: ANN002, ANN003
                result = _original_gmsh_initialize(*args, **kwargs)
                _quiet_gmsh_terminal()
                return result

            gmsh.initialize = _quiet_gmsh_initialize  # type: ignore[assignment]
            gmsh._eit_app_quiet_initialize_wrapped = True  # type: ignore[attr-defined]

        if not gmsh.isInitialized():
            gmsh.initialize()
            logging.getLogger(__name__).info(
                "gmsh initialised on main thread for worker-side mesh generation"
            )
        _quiet_gmsh_terminal()

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

    logger = logging.getLogger(__name__)
    window = EITWorkstation()
    logger.info(
        "Main window constructed: size=%s geometry=%s",
        (window.width(), window.height()),
        window.geometry().getRect(),
    )
    window.showNormal()
    window.raise_()
    window.activateWindow()
    QTimer.singleShot(0, window.raise_)
    QTimer.singleShot(0, window.activateWindow)
    logger.info(
        "Main window show requested: visible=%s geometry=%s",
        window.isVisible(),
        window.geometry().getRect(),
    )

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
