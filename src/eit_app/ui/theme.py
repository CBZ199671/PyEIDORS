"""Global Qt theme helpers for the EIT workstation."""

from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QWidget


def apply_app_theme(app: QApplication) -> None:
    """Apply a consistent workstation theme to the entire application."""
    font = QFont()
    font.setFamilies(["Segoe UI", "Noto Sans", "DejaVu Sans"])
    font.setPointSize(10)
    app.setFont(font)
    app.setStyleSheet(_APP_STYLESHEET)


def set_button_role(widget: QPushButton, role: str) -> None:
    """Tag a button with a semantic role used by the global stylesheet."""
    widget.setProperty("buttonRole", role)
    _repolish(widget)


def set_section_header(widget: QWidget) -> None:
    """Tag a label as a section header."""
    widget.setProperty("uiSectionHeader", True)
    _repolish(widget)


def set_hint_text(widget: QWidget) -> None:
    """Tag a label as supportive hint text."""
    widget.setProperty("uiHintText", True)
    _repolish(widget)


def set_subtle_value(widget: QWidget) -> None:
    """Tag a widget as a subtle informational value."""
    widget.setProperty("uiSubtleValue", True)
    _repolish(widget)


def set_panel_role(widget: QWidget, role: str) -> None:
    """Tag a widget with an instrument-panel role."""
    widget.setProperty("panelRole", role)
    _repolish(widget)


def set_embedded_step_panel(widget: QWidget) -> None:
    """Style a panel for use inside the left-side workflow toolbox."""
    widget.setProperty("embeddedStepPanel", True)
    _repolish(widget)


def tone_palette(tone: str) -> tuple[str, str, str]:
    """Return foreground/background/border colors for a named UI tone."""
    palette = {
        "idle": ("#5b6573", "#f7f9fc", "#d8dee9"),
        "warn": ("#8a4b08", "#fff4dc", "#f1c27d"),
        "ready": ("#0f5f3d", "#e6f6ee", "#7bc69a"),
        "active": ("#0b4f80", "#e9f4ff", "#7fb2e5"),
        "error": ("#8c1d18", "#fdecec", "#f1a6a1"),
    }
    return palette.get(tone, palette["idle"])


def apply_state_chip(label: QLabel, *, tone: str, compact: bool = False, emphasized: bool = False) -> None:
    """Apply the shared state-chip style used across summary cards and status bar."""
    fg, bg, border = tone_palette(tone)
    padding = "2px 10px" if compact else "4px 8px"
    radius = "8px" if compact else "10px"
    font_weight = "700" if emphasized else "600"
    label.setStyleSheet(
        f"padding: {padding}; "
        f"border-radius: {radius}; "
        f"font-weight: {font_weight}; "
        f"color: {fg}; "
        f"background: {bg}; "
        f"border: 1px solid {border};"
    )


def apply_state_banner(label: QLabel, *, tone: str) -> None:
    """Apply the shared banner style for the large summary status header."""
    fg, bg, border = tone_palette(tone)
    label.setStyleSheet(
        "font-weight: 700; "
        "letter-spacing: 1px; "
        "padding: 9px 10px; "
        "border-radius: 10px; "
        f"color: {fg}; "
        f"background: {bg}; "
        f"border: 1px solid {border};"
    )


def _repolish(widget: QWidget) -> None:
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


_APP_STYLESHEET = """
QMainWindow {
    background: #eef3f8;
}

QDockWidget {
    color: #243447;
    font-weight: 600;
}

QDockWidget::title {
    text-align: left;
    padding: 8px 10px;
    background: #dce7f3;
    border-bottom: 1px solid #c3d1df;
}

QToolBox {
    background: transparent;
    border: none;
}

QToolBox::tab {
    background: #dfe9f4;
    border: 1px solid #c5d2df;
    border-radius: 8px;
    padding: 5px 12px;
    margin: 2px 0;
    color: #243447;
    font-weight: 600;
}

QToolBox::tab:selected {
    background: #1f3b5b;
    color: #f8fbff;
    border-color: #1f3b5b;
}

QToolBox#workflowToolbox::tab {
    height: 30px;
    font-size: 12px;
    font-weight: 700;
    padding: 5px 12px;
}

QTabWidget::pane {
    border: none;
    background: #eef3f8;
    top: -1px;
}

QTabBar {
    qproperty-drawBase: 0;
}

QTabBar::tab {
    background: transparent;
    color: #5b6573;
    border: none;
    padding: 10px 22px;
    margin-right: 4px;
    font-weight: 600;
    font-size: 13px;
    min-width: 120px;
}

QTabBar::tab:selected {
    background: #eef3f8;
    color: #1f5d8b;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    border-bottom: 3px solid #1f5d8b;
    font-weight: 700;
}

QTabBar::tab:hover:!selected {
    color: #1f3b5b;
    background: #e3ebf4;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}

QToolBox#workflowToolbox > QWidget {
    background: #f9fbfe;
}

QGroupBox {
    background: #ffffff;
    border: 1px solid #e0e6ee;
    border-radius: 10px;
    margin-top: 14px;
    padding: 12px 12px 10px 12px;
    color: #243447;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 8px;
    color: #1f5d8b;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.3px;
}

QGroupBox[panelRole="summary"] {
    background: #f5f9fd;
    border: 1px solid #bfd1e2;
}

QGroupBox[panelRole="summary"]::title {
    color: #18324d;
}

QGroupBox[summaryHeaderless="true"] {
    margin-top: 0;
    padding-top: 8px;
}

QGroupBox[summaryHeaderless="true"]::title {
    color: transparent;
    height: 0px;
    padding: 0;
    margin: 0;
}

QGroupBox[panelRole="workflow"] {
    background: #f9fbfe;
}

QGroupBox[embeddedStepPanel="true"] {
    background: #fbfdff;
    border: 1px solid #d5dee8;
    border-radius: 12px;
    margin-top: 0;
    padding: 6px 7px 5px 7px;
}

QGroupBox[embeddedStepPanel="true"]::title {
    subcontrol-origin: margin;
    color: transparent;
    padding: 0;
    margin: 0;
    height: 0px;
}

QGroupBox[embeddedStepPanel="true"] QLabel[uiSectionHeader="true"] {
    padding-top: 1px;
    font-size: 11px;
}

QGroupBox[embeddedStepPanel="true"] QLabel[uiHintText="true"] {
    font-size: 11px;
}

QGroupBox[embeddedStepPanel="true"] QLineEdit,
QGroupBox[embeddedStepPanel="true"] QAbstractSpinBox,
QGroupBox[embeddedStepPanel="true"] QPlainTextEdit,
QGroupBox[embeddedStepPanel="true"] QTextEdit,
QGroupBox[embeddedStepPanel="true"] QLineEdit#selectorDisplay {
    padding: 4px 7px;
    min-height: 18px;
    border-radius: 7px;
}

QGroupBox[embeddedStepPanel="true"] QPushButton,
QGroupBox[embeddedStepPanel="true"] QToolButton {
    padding: 4px 8px;
    min-height: 18px;
    border-radius: 7px;
    font-size: 11px;
}

QGroupBox[embeddedStepPanel="true"] QToolButton#selectorButton {
    padding: 0;
    min-width: 22px;
    max-width: 22px;
}

QGroupBox[embeddedStepPanel="true"] QCheckBox {
    spacing: 6px;
    font-size: 11px;
}

QGroupBox[embeddedStepPanel="true"] QCheckBox::indicator {
    width: 16px;
    height: 16px;
}

QLabel[uiSectionHeader="true"] {
    color: #1f3b5b;
    font-weight: 700;
    padding-top: 4px;
}

QLabel[uiHintText="true"] {
    color: #5b6573;
}

QLabel[uiSubtleValue="true"] {
    color: #4d5f75;
}

QLineEdit,
QAbstractSpinBox,
QPlainTextEdit,
QTextEdit {
    background: #ffffff;
    color: #243447;
    border: 1px solid #d0d9e3;
    border-radius: 8px;
    padding: 7px 10px;
    selection-background-color: #275d95;
    selection-color: #ffffff;
}

QLineEdit:hover,
QAbstractSpinBox:hover,
QPlainTextEdit:hover,
QTextEdit:hover {
    border-color: #b1c2d3;
}

QLineEdit:focus,
QAbstractSpinBox:focus,
QPlainTextEdit:focus,
QTextEdit:focus {
    border: 1px solid #275d95;
    background: #fbfdff;
}

QComboBox {
    background: #ffffff;
    color: #243447;
    border: 1px solid #c4d0db;
    border-radius: 8px;
    padding: 6px 10px;
    min-height: 20px;
}

QComboBox:hover {
    border-color: #b1c2d3;
    background: #fbfdff;
}

QComboBox:focus {
    border: 1px solid #275d95;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: 1px solid #d5dee8;
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
    background: #eef3f8;
}

QComboBox::down-arrow {
    width: 10px;
    height: 10px;
}

QComboBox QAbstractItemView {
    background: #ffffff;
    color: #243447;
    border: 1px solid #c4d0db;
    border-radius: 6px;
    padding: 4px;
    selection-background-color: #d9e8f7;
    selection-color: #17324c;
    outline: none;
}

QComboBox QAbstractItemView::item {
    padding: 6px 10px;
    min-height: 22px;
}

QComboBox QAbstractItemView::item:hover {
    background: #e4eef9;
}

QScrollArea {
    background: transparent;
    border: none;
}

QScrollBar:vertical {
    background: #eef3f8;
    width: 10px;
    margin: 0;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: #c4d0db;
    min-height: 30px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #a8b8c8;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background: #eef3f8;
    height: 10px;
    margin: 0;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background: #c4d0db;
    min-width: 30px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal:hover {
    background: #a8b8c8;
}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    width: 0;
}

QDialog {
    background: #eef3f8;
    color: #243447;
}

QTreeView,
QListView {
    background: #ffffff;
    color: #243447;
    border: 1px solid #d5dee8;
    border-radius: 6px;
    selection-background-color: #d9e8f7;
    selection-color: #17324c;
    alternate-background-color: #f7fafd;
}

QTreeView::item,
QListView::item {
    padding: 4px 6px;
}

QTreeView::item:hover,
QListView::item:hover {
    background: #e4eef9;
}

QSplitter::handle {
    background: #d5dee8;
}

QSplitter::handle:horizontal {
    width: 3px;
}

QSplitter::handle:vertical {
    height: 3px;
}

QProgressBar {
    background: #dce7f3;
    border: 1px solid #c5d2df;
    border-radius: 6px;
    text-align: center;
    color: #243447;
    height: 18px;
}

QProgressBar::chunk {
    background: #1f5d8b;
    border-radius: 5px;
}

QPushButton {
    background: #ffffff;
    color: #243447;
    border: 1px solid #d0d9e3;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
    min-height: 20px;
}

QPushButton:hover {
    background: #f4f7fa;
    border-color: #b1c2d3;
}

QPushButton:pressed {
    background: #e3ebf4;
    padding-top: 9px;
    padding-bottom: 7px;
}

QPushButton:checked {
    border-width: 2px;
    border-color: #5e7994;
}

QPushButton:disabled {
    color: #97a6b5;
    background: #f4f7fa;
    border-color: #dde5ed;
}

QPushButton[buttonRole="primary"] {
    background: #1f5d8b;
    color: #f8fbff;
    border-color: #18496d;
}

QPushButton[buttonRole="primary"]:hover {
    background: #1a5078;
}

QPushButton[buttonRole="primary"]:checked {
    background: #18496d;
    border-color: #123754;
}

QPushButton[buttonRole="success"] {
    background: #1f7a52;
    color: #f8fbff;
    border-color: #17603f;
}

QPushButton[buttonRole="success"]:hover {
    background: #1b6c49;
}

QPushButton[buttonRole="success"]:checked {
    background: #15563a;
    border-color: #0f412c;
}

QPushButton[buttonRole="danger"] {
    background: #8b2f2f;
    color: #fff8f8;
    border-color: #6f2525;
}

QPushButton[buttonRole="danger"]:hover {
    background: #7d2929;
}

QPushButton[buttonRole="danger"]:checked {
    background: #642020;
    border-color: #4d1818;
}

QPushButton[buttonRole="subtle"] {
    background: #f7fafc;
}

QTableView {
    background: #ffffff;
    alternate-background-color: #f8fbfd;
    border: 1px solid #e0e6ee;
    border-radius: 8px;
    gridline-color: #eef3f8;
    selection-background-color: #d9e8f7;
    selection-color: #17324c;
    outline: 0;
}

QTableView::item {
    padding: 4px 6px;
    border: none;
}

QTableView::item:selected {
    background: #d9e8f7;
    color: #17324c;
}

QHeaderView::section {
    background: #f4f7fa;
    color: #5b6573;
    padding: 8px 12px;
    border: none;
    border-bottom: 2px solid #e0e6ee;
    font-weight: 700;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

QHeaderView::section:first {
    border-top-left-radius: 8px;
}

QHeaderView::section:last {
    border-top-right-radius: 8px;
}

QMenu {
    background: #ffffff;
    border: 1px solid #ccd8e3;
    border-radius: 10px;
    padding: 6px;
}

QMenu::item {
    padding: 7px 12px;
    border-radius: 6px;
}

QMenu::item:selected {
    background: #e4eef9;
    color: #17324c;
}

QToolButton {
    background: #eef3f8;
    color: #243447;
    border: 1px solid #c4d0db;
    border-radius: 8px;
    padding: 4px 10px;
    font-weight: 700;
}

QToolButton:hover {
    background: #e3ebf4;
}

QStatusBar {
    background: #ffffff;
    color: #5b6573;
    border-top: 1px solid #e0e6ee;
    padding: 2px 8px;
    min-height: 26px;
}

QStatusBar::item {
    border: none;
}

QMainWindow::separator {
    background: #e0e6ee;
    width: 1px;
    height: 1px;
}

QCheckBox {
    color: #243447;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 1.5px solid #c4d0db;
    border-radius: 4px;
    background: #ffffff;
}

QCheckBox::indicator:hover {
    border-color: #1f5d8b;
}

QCheckBox::indicator:checked {
    background: #1f5d8b;
    border-color: #1f5d8b;
    image: none;
}

QCheckBox::indicator:checked:hover {
    background: #2a6fa0;
}

/* Spin box buttons with embedded SVG arrows */
QSpinBox::up-button,
QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 20px;
    background: #f4f7fa;
    border-left: 1px solid #e0e6ee;
    border-top-right-radius: 7px;
}

QSpinBox::down-button,
QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 20px;
    background: #f4f7fa;
    border-left: 1px solid #e0e6ee;
    border-top: 1px solid #e0e6ee;
    border-bottom-right-radius: 7px;
}

QSpinBox::up-button:hover,
QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover,
QDoubleSpinBox::down-button:hover {
    background: #d9e8f7;
}

QSpinBox::up-button:pressed,
QDoubleSpinBox::up-button:pressed,
QSpinBox::down-button:pressed,
QDoubleSpinBox::down-button:pressed {
    background: #b6cfe3;
}

QSpinBox::up-arrow,
QDoubleSpinBox::up-arrow {
    image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMCIgaGVpZ2h0PSIxMCIgdmlld0JveD0iMCAwIDEwIDEwIj48cG9seWdvbiBwb2ludHM9IjUsMi41IDEuNSw3IDguNSw3IiBmaWxsPSIjNWI2NTczIi8+PC9zdmc+");
    width: 10px;
    height: 10px;
}

QSpinBox::down-arrow,
QDoubleSpinBox::down-arrow {
    image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMCIgaGVpZ2h0PSIxMCIgdmlld0JveD0iMCAwIDEwIDEwIj48cG9seWdvbiBwb2ludHM9IjUsNy41IDEuNSwzIDguNSwzIiBmaWxsPSIjNWI2NTczIi8+PC9zdmc+");
    width: 10px;
    height: 10px;
}

QSpinBox::up-arrow:hover,
QDoubleSpinBox::up-arrow:hover {
    image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMCIgaGVpZ2h0PSIxMCIgdmlld0JveD0iMCAwIDEwIDEwIj48cG9seWdvbiBwb2ludHM9IjUsMi41IDEuNSw3IDguNSw3IiBmaWxsPSIjMWY1ZDhiIi8+PC9zdmc+");
}

QSpinBox::down-arrow:hover,
QDoubleSpinBox::down-arrow:hover {
    image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMCIgaGVpZ2h0PSIxMCIgdmlld0JveD0iMCAwIDEwIDEwIj48cG9seWdvbiBwb2ludHM9IjUsNy41IDEuNSwzIDguNSwzIiBmaWxsPSIjMWY1ZDhiIi8+PC9zdmc+");
}

/* Date edit dropdown */
QDateEdit::drop-down {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 22px;
    background: #f4f7fa;
    border-left: 1px solid #e0e6ee;
    border-top-right-radius: 7px;
    border-bottom-right-radius: 7px;
}

QDateEdit::drop-down:hover {
    background: #d9e8f7;
}

QDateEdit::down-arrow {
    image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMCIgaGVpZ2h0PSIxMCIgdmlld0JveD0iMCAwIDEwIDEwIj48cG9seWdvbiBwb2ludHM9IjUsNy41IDEuNSwzIDguNSwzIiBmaWxsPSIjNWI2NTczIi8+PC9zdmc+");
    width: 10px;
    height: 10px;
}
"""
