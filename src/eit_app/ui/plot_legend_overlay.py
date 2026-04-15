"""Compact floating legend widgets for plot overlays."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QEvent, QPoint, QSize, Qt
from PySide6.QtGui import QColor, QCursor, QFont, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget


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

    def __init__(self, entry: LegendEntry, *, compact: bool = False, parent: QWidget | None = None) -> None:
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
        text_color = "#243447" if checked else "#8a98a8"
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
                color: #1f4b78;
            }}
            """
        )


class _LegendInteractiveRow(QWidget):
    def __init__(self, entry: LegendEntry, *, compact: bool = False, parent: QWidget | None = None) -> None:
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
    def __init__(self, entry: LegendEntry, *, compact: bool = False, parent: QWidget | None = None) -> None:
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

        text = QLabel(entry.label)
        font = QFont()
        font.setPointSize(10 if compact else 12)
        font.setWeight(QFont.Weight.DemiBold)
        text.setFont(font)
        text.setStyleSheet("color: #243447;")
        layout.addWidget(text, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addStretch(1)


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
        self._draggable = draggable
        self._drag_offset: QPoint | None = None
        self._drag_margin = 8
        alpha = background_alpha if background_alpha is not None else (196 if compact else 204)
        self.setObjectName("plotLegendOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"""
            QFrame#plotLegendOverlay {{
                background: rgba(255, 255, 255, {alpha});
                border: 1px solid #c7d4e2;
                border-radius: 8px;
            }}
            """
        )
        if self._draggable:
            self.setToolTip("可拖拽调整图例位置")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8 if not compact else 6, 8 if not compact else 6, 8 if not compact else 6, 8 if not compact else 6)
        layout.setSpacing(4 if not compact else 2)

        for entry in entries:
            if interactive:
                row = _LegendInteractiveRow(entry, compact=compact, parent=self)
                self._buttons[entry.key] = row.button
                layout.addWidget(row)
            else:
                layout.addWidget(_LegendIndicatorRow(entry, compact=compact, parent=self))

        self.adjustSize()
        self._install_drag_handles()

    def button(self, key: str) -> LegendToggleButton:
        return self._buttons[key]

    def eventFilter(self, watched: object, event: object) -> bool:
        if not self._draggable or not isinstance(watched, QWidget):
            return super().eventFilter(watched, event)
        if watched is not self and isinstance(watched, LegendToggleButton):
            return super().eventFilter(watched, event)
        if not hasattr(event, "type"):
            return super().eventFilter(watched, event)

        event_type = event.type()
        if event_type == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
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

        if event_type == QEvent.Type.MouseButtonRelease and self._drag_offset is not None:
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
        cursor_shape = Qt.CursorShape.ClosedHandCursor if closed else Qt.CursorShape.OpenHandCursor
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
