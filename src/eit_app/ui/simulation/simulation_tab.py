"""Container widget for the Simulation tab."""

from __future__ import annotations

from PySide6.QtWidgets import QGroupBox, QLabel, QVBoxLayout, QWidget

from eit_app.i18n import t, translator
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
        translator().language_changed.connect(self._retranslate)
        self._retranslate()

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

        self._run_guide_box = QGroupBox("")  # title assigned by _retranslate
        guide_layout = QVBoxLayout(self._run_guide_box)
        guide_layout.setContentsMargins(12, 14, 12, 12)
        guide_layout.setSpacing(6)
        # Three instructional steps — handles stored for retranslation.
        self._runguide_step_labels: list[QLabel] = []
        for _key in ("sim.runguide.step1", "sim.runguide.step2", "sim.runguide.step3"):
            label = QLabel("")
            label.setWordWrap(True)
            set_hint_text(label)
            guide_layout.addWidget(label)
            self._runguide_step_labels.append(label)
        self._runguide_hint = QLabel("")
        self._runguide_hint.setWordWrap(True)
        set_subtle_value(self._runguide_hint)
        guide_layout.addWidget(self._runguide_hint)
        guide_layout.addStretch()
        context_layout.addWidget(self._run_guide_box)
        context_layout.addStretch()

        # Step titles filled in by _retranslate below.
        # context_min_width unified to 300px across all WorkflowShell
        # tabs (see Phase 7 in TASKS.md).
        self._shell = WorkflowShell(
            steps=[
                ("", self._mesh_panel),
                ("", self._inhom_editor),
                ("", self._forward_panel),
                ("", self._inverse_panel),
            ],
            center_widget=self._results_widget,
            context_widget=context_widget,
            step_min_width=240,
            context_min_width=220,
            splitter_sizes=(350, 940, 300),
            parent=self,
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._shell)
        self._mesh_panel.config_changed.connect(self._sync_expected_point_count)
        self._sync_expected_point_count()

    # ── i18n ──

    def _retranslate(self) -> None:
        """Refresh the tab's own chrome (Step titles + Run Guide)."""
        toolbox = self._shell.toolbox
        toolbox.setItemText(0, t("sim.step.mesh"))
        toolbox.setItemText(1, t("sim.step.inhom"))
        toolbox.setItemText(2, t("sim.step.forward"))
        toolbox.setItemText(3, t("sim.step.inverse"))
        self._run_guide_box.setTitle(t("sim.runguide.title"))
        step_keys = (
            "sim.runguide.step1",
            "sim.runguide.step2",
            "sim.runguide.step3",
        )
        for label, key in zip(self._runguide_step_labels, step_keys):
            label.setText(t(key))
        self._runguide_hint.setText(t("sim.runguide.hint"))

    def _sync_expected_point_count(self) -> None:
        mesh_cfg = self._mesh_panel.get_config()
        point_count = int(measurement_layout_from_config(mesh_cfg)["points_per_frame"])
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
