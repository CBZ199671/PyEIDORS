"""Menu-backed selector widget with combo-like API."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
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
    _CONTROL_HEIGHT = 28

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._items: list[tuple[str, Any]] = []
        self._current_index = -1
        self._editable = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._line_edit = QLineEdit(self)
        self._line_edit.setObjectName("selectorDisplay")
        self._line_edit.setReadOnly(True)
        self._line_edit.textEdited.connect(self._on_text_edited)
        self._line_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._line_edit.setFixedHeight(self._CONTROL_HEIGHT)
        layout.addWidget(self._line_edit, 1)

        self._button = QToolButton(self)
        self._button.setObjectName("selectorButton")
        self._button.setText("▾")
        self._button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._button.setToolButtonStyle(self._button.toolButtonStyle())
        self._button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._button.setFixedWidth(24)
        self._button.setFixedHeight(self._CONTROL_HEIGHT)
        self._button.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self._button)

        self._menu = QMenu(self)
        self._menu.setObjectName("selectorMenu")
        self._button.setMenu(self._menu)
        self.setFocusProxy(self._line_edit)
        self.setFixedHeight(self._CONTROL_HEIGHT)

        # -----------------------------------------------------------------
        # Defensive hide paths for the dropdown menu.
        #
        # QMenu is normally a Qt.Popup which auto-dismisses on focus
        # loss, but three edge cases can leave it stuck on screen:
        #   1. Disabling the widget programmatically while the menu is
        #      visible (Qt does not auto-close popups owned by a
        #      disabled parent).
        #   2. Reassigning a new item list via clear()/addItems() while
        #      a user is actively hovering the old menu.
        #   3. Composite-widget focus loss when the user Tab's away
        #      from the line edit before ever clicking the arrow
        #      button — the menu wasn't open, but an app-level
        #      focusChanged observer is useful for future-proofing.
        #
        # Install a global focus-change watcher that hides the menu if
        # focus lands on a widget that is NOT inside this composite or
        # the menu itself.  This is a belt-and-suspenders measure on
        # top of Qt's built-in Popup semantics.
        qapp = QApplication.instance()
        if qapp is not None:
            qapp.focusChanged.connect(self._on_app_focus_changed)

    def sizeHint(self) -> QSize:
        return QSize(180, self._CONTROL_HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return QSize(120, self._CONTROL_HEIGHT)

    def addItem(self, text: str, user_data: Any = None) -> None:
        self._items.append((text, user_data))
        self._rebuild_menu()
        if self._current_index < 0:
            self._set_index(0, emit_changed=False, emit_activated=False)

    def addItems(self, texts: list[str]) -> None:
        for text in texts:
            self.addItem(text)

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

    def setItemText(self, index: int, text: str) -> None:
        """Replace the display text of an existing item (matches QComboBox API).

        Keeps the item's associated ``userData`` intact and refreshes the
        popup menu entries so they show the new text.  Also repaints the
        QLineEdit when the affected row is the currently selected one.
        """
        if not (0 <= index < len(self._items)):
            return
        _old_text, user_data = self._items[index]
        self._items[index] = (text, user_data)
        self._rebuild_menu()
        if self._current_index == index:
            self._line_edit.setText(text)

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
        # Disabling the widget while the popup is visible leaves the
        # menu orphaned on-screen until the user clicks elsewhere —
        # force-close it here so the disabled state looks consistent.
        if not enabled and self._menu.isVisible():
            self._menu.hide()

    def clear(self) -> None:
        # Also close the popup if the item list is being replaced from
        # under the user's cursor — otherwise the menu shows stale
        # entries until the next click outside.
        if self._menu.isVisible():
            self._menu.hide()
        self._items.clear()
        self._current_index = -1
        self._line_edit.clear()
        self._menu.clear()

    def _on_app_focus_changed(self, old: QWidget | None, new: QWidget | None) -> None:
        """Force-close the popup when focus leaves this composite widget.

        Qt.Popup already dismisses on click-outside, but keyboard
        navigation (Tab, Ctrl+Tab) can move focus without producing
        a mouse click; this handler catches that path too.  The menu
        stays open when focus lands on any descendant of the combo
        (line edit, button, the menu itself) so normal interactions
        aren't disrupted.
        """
        if not self._menu.isVisible():
            return
        if new is None:
            # Focus went to another top-level (Alt-Tab, etc.).  Close.
            self._menu.hide()
            return
        cursor = new
        while cursor is not None:
            if cursor is self or cursor is self._menu:
                return
            cursor = cursor.parent()
        self._menu.hide()

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
