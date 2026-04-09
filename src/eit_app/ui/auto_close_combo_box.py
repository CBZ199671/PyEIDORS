"""Menu-backed selector widget with combo-like API."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QMenu,
    QSizePolicy,
    QToolButton,
    QWidget,
)


class AutoCloseComboBox(QWidget):
    """Combo-like selector that uses a QMenu instead of the native popup."""

    currentIndexChanged = Signal(int)
    activated = Signal(int)
    textActivated = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._items: list[tuple[str, Any]] = []
        self._current_index = -1
        self._editable = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._line_edit = QLineEdit(self)
        self._line_edit.setObjectName("selectorDisplay")
        self._line_edit.setReadOnly(True)
        self._line_edit.textEdited.connect(self._on_text_edited)
        self._line_edit.setMinimumHeight(34)
        layout.addWidget(self._line_edit, 1)

        self._button = QToolButton(self)
        self._button.setObjectName("selectorButton")
        self._button.setText("▾")
        self._button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._button.setToolButtonStyle(self._button.toolButtonStyle())
        self._button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self._button.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self._button)

        self._menu = QMenu(self)
        self._menu.setObjectName("selectorMenu")
        self._button.setMenu(self._menu)
        self.setFocusProxy(self._line_edit)
        self.setMinimumHeight(34)

    def addItem(self, text: str, user_data: Any = None) -> None:
        self._items.append((text, user_data))
        self._rebuild_menu()
        if self._current_index < 0:
            self._set_index(0, emit_changed=False, emit_activated=False)

    def addItems(self, texts: list[str]) -> None:
        for text in texts:
            self.addItem(text)

    def clear(self) -> None:
        self._items.clear()
        self._current_index = -1
        self._line_edit.clear()
        self._menu.clear()

    def count(self) -> int:
        return len(self._items)

    def currentIndex(self) -> int:
        return self._current_index

    def setCurrentIndex(self, index: int) -> None:
        self._set_index(index, emit_changed=True, emit_activated=False)

    def currentText(self) -> str:
        return self._line_edit.text()

    def setCurrentText(self, text: str) -> None:
        index = self.findText(text)
        if index >= 0:
            self._set_index(index, emit_changed=True, emit_activated=False)
            return
        self._line_edit.setText(text)
        self._current_index = -1
        self.currentIndexChanged.emit(-1)

    def itemText(self, index: int) -> str:
        if 0 <= index < len(self._items):
            return self._items[index][0]
        return ""

    def currentData(self) -> Any:
        if 0 <= self._current_index < len(self._items):
            return self._items[self._current_index][1]
        return None

    def findText(self, text: str) -> int:
        for index, (item_text, _data) in enumerate(self._items):
            if item_text == text:
                return index
        return -1

    def setEditable(self, editable: bool) -> None:
        self._editable = bool(editable)
        self._line_edit.setReadOnly(not self._editable)

    def showPopup(self) -> None:
        self._menu.popup(self._button.mapToGlobal(self._button.rect().bottomLeft()))

    def hidePopup(self) -> None:
        self._menu.hide()

    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(enabled)
        self._line_edit.setEnabled(enabled)
        self._button.setEnabled(enabled)

    def _rebuild_menu(self) -> None:
        self._menu.clear()
        for index, (text, _user_data) in enumerate(self._items):
            action = self._menu.addAction(text)
            action.triggered.connect(
                lambda _checked=False, row=index: self._set_index(
                    row,
                    emit_changed=True,
                    emit_activated=True,
                )
            )

    def _set_index(self, index: int, *, emit_changed: bool, emit_activated: bool) -> None:
        if not (0 <= index < len(self._items)):
            return
        changed = index != self._current_index
        self._current_index = index
        text, _user_data = self._items[index]
        self._line_edit.setText(text)
        if changed and emit_changed:
            self.currentIndexChanged.emit(index)
        if emit_activated:
            self.activated.emit(index)
            self.textActivated.emit(text)
            self.hidePopup()

    def _on_text_edited(self, text: str) -> None:
        if not self._editable:
            return
        index = self.findText(text)
        if index != self._current_index:
            self._current_index = index
            self.currentIndexChanged.emit(index)
