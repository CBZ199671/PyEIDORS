"""Shared desktop workflow shell used by the main workstation tabs."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractButton,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolBox,
    QVBoxLayout,
    QWidget,
)

from eit_app.ui.theme import set_embedded_step_panel


class WorkflowShell(QWidget):
    """Reusable left-steps / center-workspace / optional right-context layout."""

    _TAB_BUTTON_HEIGHT = 30

    def __init__(
        self,
        *,
        steps: Sequence[tuple[str, QWidget]],
        center_widget: QWidget,
        context_widget: QWidget | None,
        left_footer: QWidget | None = None,
        left_footer_stretch: int = 1,
        compact_toolbox: bool = False,
        step_min_width: int = 220,
        context_min_width: int = 200,
        center_min_width: int = 200,
        splitter_sizes: Sequence[int] = (360, 840, 320),
        toolbox_name: str = "workflowToolbox",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._splitter_sizes = list(splitter_sizes)
        self._build_ui(
            steps=steps,
            center_widget=center_widget,
            context_widget=context_widget,
            left_footer=left_footer,
            left_footer_stretch=left_footer_stretch,
            compact_toolbox=compact_toolbox,
            step_min_width=step_min_width,
            context_min_width=context_min_width,
            center_min_width=center_min_width,
            toolbox_name=toolbox_name,
        )

    def _build_ui(
        self,
        *,
        steps: Sequence[tuple[str, QWidget]],
        center_widget: QWidget,
        context_widget: QWidget | None,
        left_footer: QWidget | None,
        left_footer_stretch: int,
        compact_toolbox: bool,
        step_min_width: int,
        context_min_width: int,
        center_min_width: int,
        toolbox_name: str,
    ) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setChildrenCollapsible(False)

        self._toolbox = QToolBox()
        self._toolbox.setObjectName(toolbox_name)
        toolbox_v_policy = (
            QSizePolicy.Policy.Maximum
            if compact_toolbox
            else QSizePolicy.Policy.Expanding
        )
        self._toolbox.setSizePolicy(QSizePolicy.Policy.Preferred, toolbox_v_policy)
        for title, panel in steps:
            set_embedded_step_panel(panel)
            self._toolbox.addItem(panel, title)
        self._sync_toolbox_tab_buttons()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        # Allow a horizontal scroll if the user shrinks the window
        # past the comfortable width — the form fields stay usable
        # rather than getting clipped or squeezed unreadable.
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll.setMinimumWidth(step_min_width)

        left_container = QWidget()
        # Assign objectNames so the dark theme can specifically target
        # these intermediate QWidgets.  Without this, the QScrollArea
        # viewport and the container both paint with Qt's platform-
        # default QPalette — which on Linux renders as #efefef (near-
        # pure-white) and shows through every VBoxLayout gap as a
        # visibly bright band.  We saw this most prominently as a
        # white strip between the last QToolBox tab and any
        # ``left_footer`` widget (e.g. SessionSummaryPanel in the
        # Hardware tab).
        left_scroll.setObjectName("workflowScroll")
        left_scroll.viewport().setObjectName("workflowScrollViewport")
        left_container.setObjectName("workflowLeftContainer")
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(6)
        if compact_toolbox:
            left_layout.addWidget(self._toolbox, 0, Qt.AlignmentFlag.AlignTop)
        else:
            left_layout.addWidget(self._toolbox, 1)
        if left_footer is not None:
            left_layout.addWidget(left_footer, left_footer_stretch)
        elif compact_toolbox:
            left_layout.addStretch(1)
        left_scroll.setWidget(left_container)
        self._left_scroll = left_scroll

        center_widget.setMinimumWidth(center_min_width)

        self._main_splitter.addWidget(left_scroll)
        self._main_splitter.addWidget(center_widget)
        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        if context_widget is not None:
            context_widget.setMinimumWidth(context_min_width)
            self._main_splitter.addWidget(context_widget)
            self._main_splitter.setStretchFactor(2, 0)
            self._main_splitter.setSizes(self._splitter_sizes)
        else:
            self._main_splitter.setSizes(self._splitter_sizes[:2])

        root.addWidget(self._main_splitter)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._sync_toolbox_tab_buttons()

    def _sync_toolbox_tab_buttons(self) -> None:
        for child in self._toolbox.children():
            if not isinstance(child, QAbstractButton):
                continue
            if not child.text():
                continue
            child.setMinimumHeight(self._TAB_BUTTON_HEIGHT)
            child.setMaximumHeight(self._TAB_BUTTON_HEIGHT)
            child.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    @property
    def toolbox(self) -> QToolBox:
        return self._toolbox

    @property
    def main_splitter(self) -> QSplitter:
        return self._main_splitter
