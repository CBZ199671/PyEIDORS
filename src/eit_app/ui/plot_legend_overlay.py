"""Compact floating legend widgets for plot overlays."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QEvent, QPoint, QSize, Qt
from PySide6.QtGui import QColor, QCursor, QFont, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from eit_app.i18n import t, translator
from eit_app.ui.theme import current_theme_mode, subscribe_theme_mode


@dataclass(frozen=True)
class LegendEntry:
    key: str
    label: str
    color: str
    width: float
    style: Qt.PenStyle = Qt.PenStyle.SolidLine
    checked: bool = True


def _line_icon(
    *,
    color: str,
    width: float,
    style: Qt.PenStyle,
    compact: bool,
    active: bool,
) -> QIcon:
    icon_width = 34 if not compact else 26
    icon_height = 14 if not compact else 10
    pixmap = QPixmap(icon_width, icon_height)
    pixmap.fill(Qt.GlobalColor.transparent)

    paint = QPainter(pixmap)
    paint.setRenderHint(QPainter.RenderHint.Antialiasing)
    line_color = QColor(color)
    if not active:
        line_color.setAlpha(95)
    pen = QPen(line_color)
    pen.setWidthF(width if active else max(1.0, width - 0.2))
    pen.setStyle(style)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    paint.setPen(pen)
    y = icon_height // 2
    paint.drawLine(2, y, icon_width - 2, y)
    paint.end()
    return QIcon(pixmap)


def _eye_icon(*, visible: bool, compact: bool) -> QIcon:
    icon_size = 16 if not compact else 14
    pixmap = QPixmap(icon_size, icon_size)
    pixmap.fill(Qt.GlobalColor.transparent)

    paint = QPainter(pixmap)
    paint.setRenderHint(QPainter.RenderHint.Antialiasing)
    stroke = QColor("#4d5f75" if visible else "#93a1b2")
    pen = QPen(stroke)
    pen.setWidthF(1.5 if not compact else 1.3)
    paint.setPen(pen)

    rect_x = 2
    rect_y = 4 if not compact else 3
    rect_w = icon_size - 4
    rect_h = icon_size - 8 if not compact else icon_size - 6
    paint.drawEllipse(rect_x, rect_y, rect_w, rect_h)

    iris = QColor("#4d5f75" if visible else "#93a1b2")
    iris.setAlpha(230 if visible else 150)
    paint.setBrush(iris)
    center_x = icon_size / 2.0
    center_y = icon_size / 2.0
    radius = 2.2 if not compact else 1.9
    paint.drawEllipse(
        int(center_x - radius),
        int(center_y - radius),
        int(radius * 2),
        int(radius * 2),
    )

    if not visible:
        strike_pen = QPen(QColor("#c65f58"))
        strike_pen.setWidthF(1.6 if not compact else 1.4)
        strike_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        paint.setPen(strike_pen)
        paint.drawLine(3, icon_size - 3, icon_size - 3, 3)

    paint.end()
    return QIcon(pixmap)


class LegendToggleButton(QToolButton):
    """Checkable legend item with an eye indicator."""

    def __init__(
        self,
        entry: LegendEntry,
        *,
        compact: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._entry = entry
        self._compact = compact
        self.setText(entry.label)
        self.setCheckable(True)
        self.setChecked(entry.checked)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setAutoRaise(True)
        self.setIconSize(QSize(16 if not compact else 14, 16 if not compact else 14))
        self.toggled.connect(self._refresh_appearance)
        self._refresh_appearance(self.isChecked())

    def _refresh_appearance(self, checked: bool) -> None:
        self.setIcon(_eye_icon(visible=checked, compact=self._compact))
        font_size = 10 if self._compact else 11
        padding = "1px 2px" if self._compact else "2px 4px"
        # Text color follows the theme: dark mode uses bright text on
        # the translucent-dark legend card; light mode uses navy on
        # the translucent-white card.  Disabled / unchecked state
        # dims the active color in both modes.
        if current_theme_mode() == "dark":
            active, dim, hover = "#dbe1ea", "#6a7686", "#9dc9ea"
        else:
            active, dim, hover = "#243447", "#8a98a8", "#1f4b78"
        text_color = active if checked else dim
        self.setStyleSheet(
            f"""
            QToolButton {{
                border: none;
                background: transparent;
                color: {text_color};
                font-size: {font_size}pt;
                font-weight: 600;
                padding: {padding};
                text-align: left;
            }}
            QToolButton:hover {{
                color: {hover};
            }}
            """
        )


class _LegendInteractiveRow(QWidget):
    def __init__(
        self,
        entry: LegendEntry,
        *,
        compact: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6 if not compact else 5)

        sample = QLabel()
        sample.setPixmap(
            _line_icon(
                color=entry.color,
                width=entry.width,
                style=entry.style,
                compact=compact,
                active=True,
            ).pixmap(34 if not compact else 26, 14 if not compact else 10)
        )
        layout.addWidget(sample, 0, Qt.AlignmentFlag.AlignVCenter)

        self.button = LegendToggleButton(entry, compact=compact, parent=self)
        layout.addWidget(self.button, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addStretch(1)


class _LegendIndicatorRow(QWidget):
    def __init__(
        self,
        entry: LegendEntry,
        *,
        compact: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6 if not compact else 5)

        sample = QLabel()
        sample.setPixmap(
            _line_icon(
                color=entry.color,
                width=entry.width,
                style=entry.style,
                compact=compact,
                active=True,
            ).pixmap(34 if not compact else 26, 14 if not compact else 10)
        )
        layout.addWidget(sample, 0, Qt.AlignmentFlag.AlignVCenter)

        self.text_label = QLabel(entry.label)
        font = QFont()
        font.setPointSize(10 if compact else 12)
        font.setWeight(QFont.Weight.DemiBold)
        self.text_label.setFont(font)
        # Text color follows the theme.  Light mode uses the navy
        # #243447 that reads cleanly on the translucent-white legend
        # card; dark mode uses #dbe1ea on the translucent-dark card.
        # apply_legend_text_color() is invoked on every mode flip by
        # the PlotLegendOverlay parent's subscribe_theme_mode hook.
        self._apply_legend_text_color()
        layout.addWidget(self.text_label, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addStretch(1)

    def _apply_legend_text_color(self) -> None:
        color = "#dbe1ea" if current_theme_mode() == "dark" else "#243447"
        self.text_label.setStyleSheet(f"color: {color};")


class PlotLegendOverlay(QFrame):
    """Floating legend card that can optionally expose checkable entries."""

    def __init__(
        self,
        entries: list[LegendEntry],
        *,
        interactive: bool,
        compact: bool = False,
        draggable: bool = True,
        background_alpha: int | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._buttons: dict[str, LegendToggleButton] = {}
        self._indicator_labels: dict[str, QLabel] = {}
        self._draggable = draggable
        self._drag_offset: QPoint | None = None
        self._drag_margin = 8
        self._alpha = (
            background_alpha
            if background_alpha is not None
            else (196 if compact else 204)
        )
        self.setObjectName("plotLegendOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._apply_card_stylesheet()
        # Tooltip is assigned by _retranslate() below so it follows UI language.

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            8 if not compact else 6,
            8 if not compact else 6,
            8 if not compact else 6,
            8 if not compact else 6,
        )
        layout.setSpacing(4 if not compact else 2)

        for entry in entries:
            if interactive:
                row = _LegendInteractiveRow(entry, compact=compact, parent=self)
                self._buttons[entry.key] = row.button
                layout.addWidget(row)
            else:
                row = _LegendIndicatorRow(entry, compact=compact, parent=self)
                self._indicator_labels[entry.key] = row.text_label
                layout.addWidget(row)

        self.adjustSize()
        self._install_drag_handles()
        translator().language_changed.connect(self._retranslate)
        self._retranslate()
        # Re-paint card + label colors when the user toggles dark mode.
        subscribe_theme_mode(self._on_theme_mode_changed)

    def _apply_card_stylesheet(self) -> None:
        """Paint the legend card background + border according to mode.

        Light mode uses translucent-white on the cream plot canvas.
        Dark mode uses translucent-dark (#222831 base with alpha) on
        the near-black plot canvas — keeps the legend readable while
        still hinting at the plot underneath.
        """
        if current_theme_mode() == "dark":
            bg = f"rgba(34, 40, 49, {self._alpha})"  # #222831 with alpha
            border = "#3e4754"
        else:
            bg = f"rgba(255, 255, 255, {self._alpha})"
            border = "#c7d4e2"
        self.setStyleSheet(
            f"""
            QFrame#plotLegendOverlay {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 8px;
            }}
            """
        )

    def _on_theme_mode_changed(self, _mode: str) -> None:
        """Re-apply card + child-row colors on theme-mode flip."""
        self._apply_card_stylesheet()
        # Indicator rows own _apply_legend_text_color; button rows refresh
        # through their checked-state appearance helper.
        for row in self.findChildren(_LegendIndicatorRow):
            row._apply_legend_text_color()
        for button in self._buttons.values():
            button._refresh_appearance(button.isChecked())

    def button(self, key: str) -> LegendToggleButton:
        return self._buttons[key]

    def update_labels(self, labels: dict[str, str]) -> None:
        """Re-label any entry (interactive button or indicator-only row).

        The owner widget calls this during its own ``_retranslate()`` pass
        to push newly translated strings into the legend without having to
        rebuild it.
        """
        for key, text in labels.items():
            if key in self._buttons:
                self._buttons[key].setText(text)
            if key in self._indicator_labels:
                self._indicator_labels[key].setText(text)
        self.adjustSize()

    def _retranslate(self) -> None:
        """Refresh the legend's own chrome (drag tooltip)."""
        if self._draggable:
            self.setToolTip(t("plot_legend.drag_tooltip"))

    def eventFilter(self, watched: object, event: object) -> bool:
        if not self._draggable or not isinstance(watched, QWidget):
            return super().eventFilter(watched, event)
        if watched is not self and isinstance(watched, LegendToggleButton):
            return super().eventFilter(watched, event)
        if not hasattr(event, "type"):
            return super().eventFilter(watched, event)

        event_type = event.type()
        if (
            event_type == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._drag_offset = event.globalPosition().toPoint() - self.pos()
            self.raise_()
            self._apply_drag_cursor(closed=True)
            event.accept()
            return True

        if (
            event_type == QEvent.Type.MouseMove
            and self._drag_offset is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            new_pos = event.globalPosition().toPoint() - self._drag_offset
            self.move(self._clamp_to_parent(new_pos))
            event.accept()
            return True

        if (
            event_type == QEvent.Type.MouseButtonRelease
            and self._drag_offset is not None
        ):
            self._drag_offset = None
            self._apply_drag_cursor(closed=False)
            event.accept()
            return True

        return super().eventFilter(watched, event)

    def _install_drag_handles(self) -> None:
        if not self._draggable:
            return
        self.installEventFilter(self)
        self._apply_drag_cursor(closed=False)
        for child in self.findChildren(QWidget):
            if isinstance(child, LegendToggleButton):
                continue
            child.installEventFilter(self)
            child.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

    def _apply_drag_cursor(self, *, closed: bool) -> None:
        if not self._draggable:
            return
        cursor_shape = (
            Qt.CursorShape.ClosedHandCursor if closed else Qt.CursorShape.OpenHandCursor
        )
        self.setCursor(QCursor(cursor_shape))

    def _clamp_to_parent(self, pos: QPoint) -> QPoint:
        parent = self.parentWidget()
        if parent is None:
            return pos
        min_x = self._drag_margin
        min_y = self._drag_margin
        max_x = max(min_x, parent.width() - self.width() - self._drag_margin)
        max_y = max(min_y, parent.height() - self.height() - self._drag_margin)
        return QPoint(
            min(max(pos.x(), min_x), max_x),
            min(max(pos.y(), min_y), max_y),
        )
