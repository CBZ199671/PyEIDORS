"""Top-level tab for the batch dataset generation workflow."""

from __future__ import annotations

from PySide6.QtWidgets import QGroupBox, QLabel, QVBoxLayout, QWidget

from eit_app.i18n import t, translator
from eit_app.ui.simulation.dataset_generator_panel import DatasetGeneratorPanel
from eit_app.ui.simulation.dataset_summary_panel import DatasetSummaryPanel
from eit_app.ui.simulation.mesh_setup_panel import MeshSetupPanel
from eit_app.ui.theme import set_hint_text, set_section_header, set_subtle_value
from eit_app.ui.workflow_shell import WorkflowShell


class _DatasetWorkspaceWidget(QWidget):
    """Central workspace for the dataset tab — hero + artifacts + notes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        translator().language_changed.connect(self._retranslate)
        self._retranslate()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._hero_box = QGroupBox("")  # retranslated
        hero_layout = QVBoxLayout(self._hero_box)
        hero_layout.setContentsMargins(12, 14, 12, 12)
        hero_layout.setSpacing(8)
        self._hero_title = QLabel("")
        self._hero_title.setWordWrap(True)
        # Use the section-header role so the color tracks dark mode
        # (was hardcoded #1f3b5b which becomes unreadable on a dark
        # canvas).
        set_section_header(self._hero_title)
        self._hero_title.setStyleSheet("font-size: 16px; font-weight: 700;")
        hero_layout.addWidget(self._hero_title)
        self._hero_hint = QLabel("")
        self._hero_hint.setWordWrap(True)
        set_hint_text(self._hero_hint)
        hero_layout.addWidget(self._hero_hint)
        layout.addWidget(self._hero_box)

        self._artifacts_box = QGroupBox("")
        artifacts_layout = QVBoxLayout(self._artifacts_box)
        artifacts_layout.setContentsMargins(12, 14, 12, 12)
        artifacts_layout.setSpacing(6)
        self._artifact_labels: list[QLabel] = []
        for _key in (
            "dataset.artifacts.item1",
            "dataset.artifacts.item2",
            "dataset.artifacts.item3",
        ):
            label = QLabel("")
            label.setWordWrap(True)
            set_subtle_value(label)
            artifacts_layout.addWidget(label)
            self._artifact_labels.append(label)
        layout.addWidget(self._artifacts_box)

        self._notes_box = QGroupBox("")
        notes_layout = QVBoxLayout(self._notes_box)
        notes_layout.setContentsMargins(12, 14, 12, 12)
        notes_layout.setSpacing(6)
        self._note_labels: list[QLabel] = []
        for _key in (
            "dataset.notes.item1",
            "dataset.notes.item2",
            "dataset.notes.item3",
        ):
            label = QLabel("")
            label.setWordWrap(True)
            set_hint_text(label)
            notes_layout.addWidget(label)
            self._note_labels.append(label)
        layout.addWidget(self._notes_box)
        # Collapse the leftover vertical space into a single stretch at
        # the bottom instead of letting the notes group-box absorb it
        # (stretch=1 + an inner addStretch previously inflated notes into
        # a large half-empty panel).  The three info boxes now sit
        # compactly at the top with the slack pooled below.
        layout.addStretch(1)

    # ── i18n ──

    def _retranslate(self) -> None:
        self._hero_box.setTitle(t("dataset.hero.title"))
        self._hero_title.setText(t("dataset.hero.title_text"))
        self._hero_hint.setText(t("dataset.hero.hint"))
        self._artifacts_box.setTitle(t("dataset.artifacts.title"))
        for label, key in zip(
            self._artifact_labels,
            (
                "dataset.artifacts.item1",
                "dataset.artifacts.item2",
                "dataset.artifacts.item3",
            ),
        ):
            label.setText(t(key))
        self._notes_box.setTitle(t("dataset.notes.title"))
        for label, key in zip(
            self._note_labels,
            ("dataset.notes.item1", "dataset.notes.item2", "dataset.notes.item3"),
        ):
            label.setText(t(key))


class DatasetGeneratorTab(QWidget):
    """Dedicated top-level workflow for dataset generation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._progress = (0, 0)
        self._generating = False
        self._build_ui()
        self._dataset_panel._dir_edit.setText(
            str(self._dataset_panel.default_output_dir())
        )
        translator().language_changed.connect(self._retranslate)
        self._retranslate()
        self._refresh_summary()

    def _build_ui(self) -> None:
        self._mesh_panel = MeshSetupPanel()
        self._dataset_panel = DatasetGeneratorPanel(self)
        self._summary_panel = DatasetSummaryPanel()
        self._workspace = _DatasetWorkspaceWidget()

        # Step titles filled in by _retranslate below.
        # context_min_width unified to 300px across all WorkflowShell
        # tabs (see Phase 7 in TASKS.md); the dataset summary panel
        # previously used 310/320 which made it visibly wider than the
        # other two tabs' right panes.
        self._shell = WorkflowShell(
            steps=[
                ("", self._mesh_panel),
                ("", self._dataset_panel.randomization_panel),
                ("", self._dataset_panel.run_panel),
            ],
            center_widget=self._workspace,
            context_widget=self._summary_panel,
            step_min_width=420,
            context_min_width=220,
            splitter_sizes=(420, 480, 240),
            parent=self,
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._shell)

        self._mesh_panel.config_changed.connect(self._refresh_summary)
        self._dataset_panel.config_changed.connect(self._refresh_summary)

    # ── i18n ──

    def _retranslate(self) -> None:
        toolbox = self._shell.toolbox
        toolbox.setItemText(0, t("dataset.step.mesh"))
        toolbox.setItemText(1, t("dataset.step.ranges"))
        toolbox.setItemText(2, t("dataset.step.run"))
        # Re-apply the summary so state chip / progress label pick up
        # translated tone strings after a language switch.
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        mesh_cfg = self._mesh_panel.get_config()
        dataset_cfg = self._dataset_panel.get_config()
        shapes = ", ".join(shape.title() for shape in dataset_cfg["shapes"])
        output_dir = dataset_cfg["output_dir"] or str(
            self._dataset_panel.default_output_dir()
        )
        self._summary_panel.set_summary(
            {
                "output_dir": output_dir,
                "samples": str(dataset_cfg["n_samples"]),
                "shapes": shapes,
                "mesh": (
                    f"{mesh_cfg['mesh_dimension']}D | "
                    f"mesh size {mesh_cfg['mesh_refinement']:.3f}"
                ),
                "electrodes": (
                    f"{mesh_cfg['n_electrodes']} e/ring x {mesh_cfg.get('n_rings', 1)} ring(s)"
                ),
            }
        )
        self._summary_panel.set_progress(*self._progress)
        if self._generating:
            self._summary_panel.set_status(
                t("dataset.summary.state.generating"), tone="active"
            )
        elif self._progress[1] > 0 and self._progress[0] == self._progress[1]:
            self._summary_panel.set_status(
                t("dataset.summary.state.complete"), tone="ready"
            )
        else:
            self._summary_panel.set_status(t("dataset.summary.state.idle"), tone="idle")

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
