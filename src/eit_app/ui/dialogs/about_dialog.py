"""About dialog — shows the app brand mark, version, and credit lines.

Brand surface lifted from the EIT Workstation design system handoff
(``docs/design/DESIGN_SYSTEM_README.md``).  The dialog is intentionally
minimal: logo, headline, meta, body paragraph, dismiss button — no
icons elsewhere, matching the design's "no icons / no emoji" rule.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from eit_app.i18n import t, translator
from eit_app.ui.theme import set_hint_text, set_section_header


_LOGO_RESOURCE = "logo.svg"


def _logo_path() -> Path | None:
    """Resolve the bundled brand SVG.

    Tries the importlib.resources path first (works in installed wheels
    + editable installs) then falls back to a worktree-relative search
    so this also works when the package is run straight out of ``src/``
    without ``pip install -e``.
    """
    try:
        files = resources.files("eit_app.assets")
        candidate = files.joinpath(_LOGO_RESOURCE)
        path = Path(str(candidate))
        if path.exists():
            return path
    except Exception:  # pragma: no cover — resources fallback below
        pass

    here = Path(__file__).resolve().parent
    fallback = here.parent.parent / "assets" / _LOGO_RESOURCE
    return fallback if fallback.exists() else None


def _render_logo_pixmap(size: tuple[int, int] = (320, 96)) -> QPixmap | None:
    """Rasterise ``logo.svg`` to a HiDPI-aware QPixmap.

    Returns ``None`` if the asset cannot be located so callers can
    swap in a text-only header without crashing the dialog.
    """
    path = _logo_path()
    if path is None:
        return None
    renderer = QSvgRenderer(str(path))
    if not renderer.isValid():
        return None
    width, height = size
    pixmap = QPixmap(width * 2, height * 2)
    pixmap.fill(Qt.GlobalColor.transparent)
    from PySide6.QtGui import QPainter

    painter = QPainter(pixmap)
    try:
        renderer.render(painter)
    finally:
        painter.end()
    pixmap.setDevicePixelRatio(2.0)
    return pixmap


class AboutDialog(QDialog):
    """Modal "About" panel surfaced from Help → About.

    The window stays small and shadowless to match the rest of the
    workstation chrome (no drop shadows are used anywhere else).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.MSWindowsFixedSizeDialogHint, True)
        self.setSizeGripEnabled(False)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 18, 20, 14)
        outer.setSpacing(12)

        # Brand mark — try the SVG; if it fails, fall back to the
        # wordmark text alone.  No icons / no emoji anywhere.
        self._logo_label = QLabel()
        self._logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = _render_logo_pixmap()
        if pixmap is not None:
            self._logo_label.setPixmap(pixmap)
        else:
            self._logo_label.setText("EIT Workstation")
            set_section_header(self._logo_label)
        outer.addWidget(self._logo_label)

        # Headline + version line
        self._title_label = QLabel("")
        set_section_header(self._title_label)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self._title_label)

        self._version_label = QLabel("")
        set_hint_text(self._version_label)
        self._version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self._version_label)

        # Body — bilingual brand description, wraps to dialog width.
        self._body_label = QLabel("")
        self._body_label.setWordWrap(True)
        self._body_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._body_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        outer.addWidget(self._body_label)

        # Credit / acknowledgement line — kept subtle on purpose.
        self._credit_label = QLabel("")
        set_hint_text(self._credit_label)
        self._credit_label.setWordWrap(True)
        self._credit_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        outer.addWidget(self._credit_label)

        # Dismiss button — single Close, native style.
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self._buttons.rejected.connect(self.reject)
        button_row.addWidget(self._buttons)
        outer.addLayout(button_row)

        translator().language_changed.connect(self._retranslate)
        self._retranslate()

    def _retranslate(self) -> None:
        self.setWindowTitle(t("about.title"))
        self._title_label.setText(t("about.brand_headline"))
        self._version_label.setText(t("about.version_line"))
        self._body_label.setText(t("about.body"))
        self._credit_label.setText(t("about.credit"))
        self._buttons.button(QDialogButtonBox.StandardButton.Close).setText(
            t("about.close")
        )
