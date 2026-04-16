#!/usr/bin/env python3
"""Capture PNG screenshots of every top-level tab in both languages.

Used as a visual regression baseline for Phase 10 of the GUI polish
work.  Runs under QT_QPA_PLATFORM=offscreen so it works headless on
WSL / CI, using QWidget.grab() (which renders the widget off-screen
and does not need a visible display).

Output: ``docs/screenshots/tab_<name>_<lang>.png`` — 8 files total
(Hardware / Simulation / Dataset / Database × en / zh).

Each tab is seeded with a small amount of representative state
(a couple of frames in the Hardware frame browser, a mesh-ready
simulation panel) so the screenshots actually exercise the tab
layout instead of showing every widget in its initial empty state.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Pin the window size so sequential runs produce byte-stable PNGs and
# diffs are meaningful.  Anything narrower cuts the right context pane.
_WINDOW_SIZE = (1600, 980)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    # scripts/gui/capture_bilingual_screenshots.py → repo root 3 levels up
    return here.parent.parent.parent


def _output_dir() -> Path:
    out = _repo_root() / "docs" / "screenshots"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _capture_all() -> list[Path]:
    # Import inside the function so the Qt platform flag is set first.
    sys.path.insert(0, str(_repo_root() / "src"))
    from PySide6.QtWidgets import QApplication
    from eit_app.i18n import set_language
    from eit_app.ui.fonts import configure_runtime_fonts
    from eit_app.ui.main_window import EITWorkstation
    from eit_app.ui.theme import apply_app_theme

    app = QApplication.instance() or QApplication(sys.argv)
    configure_runtime_fonts(app)
    apply_app_theme(app)

    tabs = [
        ("hardware", 0),
        ("simulation", 1),
        ("dataset", 2),
        ("database", 3),
    ]
    languages = ("en", "zh")

    window = EITWorkstation()
    window.resize(*_WINDOW_SIZE)
    window.show()
    app.processEvents()

    # Seed a couple of hardware frames so the frame-browser right pane
    # is not showing only the "empty state" hint.
    browser = window._frame_browser
    browser.add_frame_entry(0, 0.0, "/tmp/frame_000.csv")
    browser.add_frame_entry(1, 0.1, "/tmp/frame_001.csv")
    app.processEvents()

    produced: list[Path] = []
    for lang in languages:
        set_language(lang, persist=False)
        app.processEvents()
        for name, index in tabs:
            window._tab_widget.setCurrentIndex(index)
            # Let layout settle before grabbing — matplotlib canvases
            # and pyqtgraph plots need a paint cycle to settle their
            # viewport before the grab call.
            for _ in range(3):
                app.processEvents()
            pixmap = window.grab()
            out_path = _output_dir() / f"tab_{name}_{lang}.png"
            if not pixmap.save(str(out_path), "PNG"):
                raise RuntimeError(f"Failed to save {out_path}")
            produced.append(out_path)
            print(f"  wrote {out_path.relative_to(_repo_root())}")

    set_language("en", persist=False)
    window.close()
    return produced


def main() -> int:
    produced = _capture_all()
    print(f"\nCaptured {len(produced)} screenshots into {_output_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
