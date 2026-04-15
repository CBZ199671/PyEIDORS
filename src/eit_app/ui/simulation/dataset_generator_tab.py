"""Top-level tab for the batch dataset generation workflow."""

from __future__ import annotations

from PySide6.QtWidgets import QGroupBox, QLabel, QVBoxLayout, QWidget

from eit_app.ui.simulation.dataset_generator_panel import DatasetGeneratorPanel
from eit_app.ui.simulation.dataset_summary_panel import DatasetSummaryPanel
from eit_app.ui.simulation.mesh_setup_panel import MeshSetupPanel
from eit_app.ui.theme import set_hint_text, set_subtle_value
from eit_app.ui.workflow_shell import WorkflowShell


class _DatasetWorkspaceWidget(QWidget):
    """Central workspace for the dataset tab."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        hero = QGroupBox("Batch Dataset Pipeline")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(12, 14, 12, 12)
        hero_layout.setSpacing(8)
        title = QLabel(
            "Generate mesh-aware conductivity targets and boundary-voltage pairs "
            "with a cleaner, step-by-step workflow."
        )
        title.setWordWrap(True)
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #1f3b5b;")
        hero_layout.addWidget(title)

        hint = QLabel(
            "Use the left-side steps to define mesh, randomization ranges, and the "
            "batch output target. The summary panel on the right mirrors the active run."
        )
        hint.setWordWrap(True)
        set_hint_text(hint)
        hero_layout.addWidget(hint)
        layout.addWidget(hero)

        artifacts = QGroupBox("Generated Artifacts")
        artifacts_layout = QVBoxLayout(artifacts)
        artifacts_layout.setContentsMargins(12, 14, 12, 12)
        artifacts_layout.setSpacing(6)
        for text in (
            "mesh_info.npz with node coordinates, cell connectivity, and homogeneous voltages",
            "sample_000000.npz style per-sample conductivity and boundary-voltage pairs",
            "The configured output directory becomes a self-contained dataset package",
        ):
            label = QLabel(text)
            label.setWordWrap(True)
            set_subtle_value(label)
            artifacts_layout.addWidget(label)
        layout.addWidget(artifacts)

        notes = QGroupBox("Operating Notes")
        notes_layout = QVBoxLayout(notes)
        notes_layout.setContentsMargins(12, 14, 12, 12)
        notes_layout.setSpacing(6)
        for text in (
            "Mesh settings here are independent from the interactive Simulation tab.",
            "Shape toggles define the random family pool; if none are checked, circle is used by default.",
            "Noise is applied after the forward solve, so voltage perturbations match the configured batch range.",
        ):
            label = QLabel(text)
            label.setWordWrap(True)
            set_hint_text(label)
            notes_layout.addWidget(label)
        notes_layout.addStretch()
        layout.addWidget(notes, 1)


class DatasetGeneratorTab(QWidget):
    """Dedicated top-level workflow for dataset generation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._progress = (0, 0)
        self._generating = False
        self._build_ui()
        self._dataset_panel._dir_edit.setText(str(self._dataset_panel.default_output_dir()))
        self._refresh_summary()

    def _build_ui(self) -> None:
        self._mesh_panel = MeshSetupPanel()
        self._dataset_panel = DatasetGeneratorPanel(self)
        self._summary_panel = DatasetSummaryPanel()
        self._workspace = _DatasetWorkspaceWidget()

        self._shell = WorkflowShell(
            steps=[
                ("Step 1 \u00b7 Mesh & Electrodes", self._mesh_panel),
                (
                    "Step 2 \u00b7 Randomization Ranges",
                    self._dataset_panel.randomization_panel,
                ),
                ("Step 3 \u00b7 Output & Run", self._dataset_panel.run_panel),
            ],
            center_widget=self._workspace,
            context_widget=self._summary_panel,
            step_min_width=330,
            context_min_width=310,
            splitter_sizes=(360, 820, 320),
            parent=self,
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._shell)

        self._mesh_panel.config_changed.connect(self._refresh_summary)
        self._dataset_panel.config_changed.connect(self._refresh_summary)

    def _refresh_summary(self) -> None:
        mesh_cfg = self._mesh_panel.get_config()
        dataset_cfg = self._dataset_panel.get_config()
        shapes = ", ".join(shape.title() for shape in dataset_cfg["shapes"])
        output_dir = dataset_cfg["output_dir"] or str(self._dataset_panel.default_output_dir())
        self._summary_panel.set_summary(
            {
                "output_dir": output_dir,
                "samples": str(dataset_cfg["n_samples"]),
                "shapes": shapes,
                "mesh": (
                    f"{mesh_cfg['mesh_dimension']}D | "
                    f"mesh size {mesh_cfg['mesh_refinement']:.3f}"
                ),
                "electrodes": str(mesh_cfg["n_electrodes"]),
            }
        )
        self._summary_panel.set_progress(*self._progress)
        if self._generating:
            self._summary_panel.set_status("Generating", tone="active")
        elif self._progress[1] > 0 and self._progress[0] == self._progress[1]:
            self._summary_panel.set_status("Complete", tone="ready")
        else:
            self._summary_panel.set_status("Idle", tone="idle")

    def set_generating(self, running: bool) -> None:
        self._generating = running
        self._dataset_panel.set_generating(running)
        if running:
            self._progress = (0, 0)
        self._refresh_summary()

    def set_progress(self, current: int, total: int) -> None:
        self._progress = (current, total)
        self._dataset_panel.set_progress(current, total)
        self._refresh_summary()

    @property
    def mesh_setup_panel(self) -> MeshSetupPanel:
        return self._mesh_panel

    @property
    def dataset_generator_panel(self) -> DatasetGeneratorPanel:
        return self._dataset_panel

    @property
    def summary_panel(self) -> DatasetSummaryPanel:
        return self._summary_panel

    @property
    def workflow_toolbox(self):
        return self._shell.toolbox

    @property
    def main_splitter(self):
        return self._shell.main_splitter
