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
    padding: 10px 14px;
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
    min-height: 24px;
    font-size: 13px;
}

QTabWidget::pane {
    border: none;
    background: #eef3f8;
}

QTabBar::tab {
    background: #dfe9f4;
    color: #243447;
    border: 1px solid #c5d2df;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 8px 20px;
    margin-right: 2px;
    font-weight: 600;
    font-size: 13px;
}

QTabBar::tab:selected {
    background: #eef3f8;
    color: #1f3b5b;
    border-bottom: 2px solid #1f5d8b;
}

QTabBar::tab:hover:!selected {
    background: #e3ebf4;
}

QToolBox#workflowToolbox > QWidget {
    background: #f9fbfe;
}

QGroupBox {
    background: #f9fbfe;
    border: 1px solid #d5dee8;
    border-radius: 12px;
    margin-top: 12px;
    padding: 10px 10px 8px 10px;
    color: #243447;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: #1f3b5b;
}

QGroupBox[panelRole="summary"] {
    background: #f5f9fd;
    border: 1px solid #bfd1e2;
}

QGroupBox[panelRole="summary"]::title {
    color: #18324d;
}

QGroupBox[panelRole="workflow"] {
    background: #f9fbfe;
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
    border: 1px solid #c4d0db;
    border-radius: 8px;
    padding: 7px 9px;
    selection-background-color: #275d95;
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
    background: #eef3f8;
    color: #243447;
    border: 1px solid #c4d0db;
    border-radius: 8px;
    padding: 7px 12px;
    font-weight: 600;
}

QPushButton:hover {
    background: #e3ebf4;
    border-color: #b1c2d3;
}

QPushButton:pressed {
    background: #d9e5f1;
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

QPushButton[buttonRole="success"] {
    background: #1f7a52;
    color: #f8fbff;
    border-color: #17603f;
}

QPushButton[buttonRole="success"]:hover {
    background: #1b6c49;
}

QPushButton[buttonRole="danger"] {
    background: #8b2f2f;
    color: #fff8f8;
    border-color: #6f2525;
}

QPushButton[buttonRole="danger"]:hover {
    background: #7d2929;
}

QPushButton[buttonRole="subtle"] {
    background: #f7fafc;
}

QCheckBox {
    spacing: 8px;
    color: #243447;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 5px;
    border: 1px solid #b9c7d4;
    background: #ffffff;
}

QCheckBox::indicator:checked {
    background: #1f5d8b;
    border-color: #1f5d8b;
}

QTableView {
    background: #ffffff;
    alternate-background-color: #f7fafd;
    border: 1px solid #d5dee8;
    border-radius: 10px;
    gridline-color: #e5edf5;
    selection-background-color: #d9e8f7;
    selection-color: #17324c;
}

QHeaderView::section {
    background: #eef3f8;
    color: #1f3b5b;
    padding: 7px 8px;
    border: none;
    border-bottom: 1px solid #d5dee8;
    font-weight: 700;
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
    background: #102033;
    color: #d9e2f2;
}

QStatusBar::item {
    border: none;
}
"""
