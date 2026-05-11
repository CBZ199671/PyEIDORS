"""Container widget for the Simulation tab."""

from __future__ import annotations

from PySide6.QtWidgets import QGroupBox, QLabel, QVBoxLayout, QWidget

from eit_app.i18n import t, translator
from eit_app.measurement_layout import measurement_layout_from_config
from eit_app.models.forward_model_config import INTERACTIVE_3D_DEFAULT_HEIGHT
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

        self._left_status_panel = QWidget()
        status_layout = QVBoxLayout(self._left_status_panel)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(6)
        status_layout.addWidget(self._metrics_panel)

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
        status_layout.addWidget(self._run_guide_box)

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
            # Simulation keeps quality metrics + flow hints in the left
            # rail so the visualization workspace receives the width
            # formerly consumed by the right context column.
            context_widget=None,
            left_footer=self._left_status_panel,
            left_footer_stretch=0,
            # step_min_width sized to the densest simulation step
            # panel (mesh setup ≈ 480 px) — see Hardware tab for the
            # rationale.
            step_min_width=460,
            # Total opens at ~1180 px so the simulation tab fits a
            # 1280-px laptop while giving the result plots the space
            # previously reserved for the right context column.
            splitter_sizes=(460, 720),
            parent=self,
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._shell)
        self._mesh_panel.config_changed.connect(self._sync_expected_point_count)
        self._mesh_panel.config_changed.connect(self._sync_inhomogeneity_context)
        self._sync_expected_point_count()
        self._sync_inhomogeneity_context()

    # ── i18n ──

    def _retranslate(self) -> None:
        """Refresh the tab's own chrome (Step titles + Run Guide)."""
        self._refresh_step_labels()
        self._run_guide_box.setTitle(t("sim.runguide.title"))
        step_keys = (
            "sim.runguide.step1",
            "sim.runguide.step2",
            "sim.runguide.step3",
        )
        for label, key in zip(self._runguide_step_labels, step_keys):
            label.setText(t(key))
        self._runguide_hint.setText(t("sim.runguide.hint"))

    def _refresh_step_labels(self) -> None:
        toolbox = self._shell.toolbox
        toolbox.setItemText(0, t("sim.step.mesh"))
        mesh_dim = int(self._mesh_panel.get_config().get("mesh_dimension", 2))
        inhom_key = "sim.step.inhom_3d" if mesh_dim == 3 else "sim.step.inhom_2d"
        toolbox.setItemText(1, t(inhom_key))
        toolbox.setItemText(2, t("sim.step.forward"))
        toolbox.setItemText(3, t("sim.step.inverse"))

    def _sync_expected_point_count(self) -> None:
        mesh_cfg = self._mesh_panel.get_config()
        point_count = int(measurement_layout_from_config(mesh_cfg)["points_per_frame"])
        self._results_widget.set_expected_point_count(point_count)

    def _sync_inhomogeneity_context(self) -> None:
        mesh_cfg = self._mesh_panel.get_config()
        mesh_dim = int(mesh_cfg.get("mesh_dimension", 2))
        self._inhom_editor.set_domain_context(
            mesh_dimension=mesh_dim,
            radius=float(mesh_cfg.get("radius", 1.0)),
            height=float(mesh_cfg.get("height", INTERACTIVE_3D_DEFAULT_HEIGHT))
            if mesh_dim == 3
            else 1.0,
            z_center=0.0,
        )
        self._refresh_step_labels()

    def set_inhomogeneity_domain(
        self,
        *,
        mesh_dimension: int,
        radius: float,
        height: float,
        z_center: float = 0.0,
    ) -> None:
        self._inhom_editor.set_domain_context(
            mesh_dimension=mesh_dimension,
            radius=radius,
            height=height,
            z_center=z_center,
        )
        self._refresh_step_labels()

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
