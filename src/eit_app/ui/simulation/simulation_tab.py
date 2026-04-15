"""Container widget for the Simulation tab."""

from __future__ import annotations

from PySide6.QtWidgets import QGroupBox, QLabel, QVBoxLayout, QWidget

from eit_app.measurement_layout import measurement_layout_from_config
from eit_app.ui.simulation.forward_problem_panel import ForwardProblemPanel
from eit_app.ui.simulation.inhomogeneity_editor import InhomogeneityEditor
from eit_app.ui.simulation.inverse_problem_panel import InverseProblemPanel
from eit_app.ui.simulation.mesh_setup_panel import MeshSetupPanel
from eit_app.ui.simulation.metrics_panel import MetricsPanel
from eit_app.ui.simulation.simulation_results_widget import SimulationResultsWidget
from eit_app.ui.theme import set_hint_text, set_subtle_value
from eit_app.ui.workflow_shell import WorkflowShell


class SimulationTab(QWidget):
    """Top-level container for the simulation workflow.

    Assembles left control panels and central visualization into a
    single QWidget for embedding in a QTabWidget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        self._mesh_panel = MeshSetupPanel()
        self._inhom_editor = InhomogeneityEditor()
        self._forward_panel = ForwardProblemPanel()
        self._inverse_panel = InverseProblemPanel()
        self._results_widget = SimulationResultsWidget()
        self._metrics_panel = MetricsPanel()

        context_widget = QWidget()
        context_layout = QVBoxLayout(context_widget)
        context_layout.setContentsMargins(0, 0, 0, 0)
        context_layout.setSpacing(6)
        context_layout.addWidget(self._metrics_panel)

        run_guide = QGroupBox("Run Guide")
        guide_layout = QVBoxLayout(run_guide)
        guide_layout.setContentsMargins(12, 14, 12, 12)
        guide_layout.setSpacing(6)
        for text in (
            "先配置网格与电极，再维护异常体列表。",
            "运行 Forward 后查看边界电压与 Ground Truth。",
            "运行 Inverse 后在右侧查看误差指标。",
        ):
            label = QLabel(text)
            label.setWordWrap(True)
            set_hint_text(label)
            guide_layout.addWidget(label)
        status_hint = QLabel("中央区域优先用于图像与曲线对照。")
        status_hint.setWordWrap(True)
        set_subtle_value(status_hint)
        guide_layout.addWidget(status_hint)
        guide_layout.addStretch()
        context_layout.addWidget(run_guide)
        context_layout.addStretch()

        self._shell = WorkflowShell(
            steps=[
                ("Step 1 \u00b7 Mesh & Electrodes", self._mesh_panel),
                ("Step 2 \u00b7 Inhomogeneities", self._inhom_editor),
                ("Step 3 \u00b7 Forward Problem", self._forward_panel),
                ("Step 4 \u00b7 Inverse Problem", self._inverse_panel),
            ],
            center_widget=self._results_widget,
            context_widget=context_widget,
            step_min_width=320,
            context_min_width=290,
            splitter_sizes=(350, 940, 300),
            parent=self,
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._shell)
        self._mesh_panel.config_changed.connect(self._sync_expected_point_count)
        self._sync_expected_point_count()

    def _sync_expected_point_count(self) -> None:
        mesh_cfg = self._mesh_panel.get_config()
        point_count = int(
            measurement_layout_from_config({"n_electrodes": mesh_cfg["n_electrodes"]})[
                "points_per_frame"
            ]
        )
        self._results_widget.set_expected_point_count(point_count)

    # --- Property accessors for signal wiring ---

    @property
    def mesh_setup_panel(self) -> MeshSetupPanel:
        return self._mesh_panel

    @property
    def inhomogeneity_editor(self) -> InhomogeneityEditor:
        return self._inhom_editor

    @property
    def forward_problem_panel(self) -> ForwardProblemPanel:
        return self._forward_panel

    @property
    def inverse_problem_panel(self) -> InverseProblemPanel:
        return self._inverse_panel

    @property
    def results_widget(self) -> SimulationResultsWidget:
        return self._results_widget

    @property
    def metrics_panel(self) -> MetricsPanel:
        return self._metrics_panel

    @property
    def workflow_toolbox(self):
        return self._shell.toolbox

    @property
    def main_splitter(self):
        return self._shell.main_splitter
