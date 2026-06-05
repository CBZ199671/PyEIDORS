"""Visual audit index generation for historical EIT precision experiments."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
import textwrap
from typing import Iterable

from pyeidors.runtime_paths import pyeidors_output_path


AUDIT_INDEX_FIELDS = [
    "task_id",
    "slug",
    "title",
    "audit_status",
    "confidence_level",
    "missing_required_visuals",
    "present_required_visuals",
    "present_files",
    "audit_plot",
    "claim_scope",
    "confidence_note",
]


@dataclass(frozen=True)
class VisualAuditArtifact:
    """One file used by a visual audit panel."""

    key: str
    path: str
    kind: str
    label: str


@dataclass(frozen=True)
class VisualAuditExperiment:
    """Audit manifest entry for one conclusion-bearing experiment."""

    task_id: str
    slug: str
    title: str
    claim_scope: str
    artifacts: tuple[VisualAuditArtifact, ...]
    required_visual_keys: tuple[str, ...]
    trusted_note: str
    smoke_note: str


@dataclass(frozen=True)
class VisualAuditRow:
    """Resolved audit status for one experiment."""

    task_id: str
    slug: str
    title: str
    audit_status: str
    confidence_level: str
    missing_required_visuals: tuple[str, ...]
    present_required_visuals: tuple[str, ...]
    present_files: tuple[str, ...]
    audit_plot: str
    claim_scope: str
    confidence_note: str

    def as_csv_row(self) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "slug": self.slug,
            "title": self.title,
            "audit_status": self.audit_status,
            "confidence_level": self.confidence_level,
            "missing_required_visuals": ";".join(self.missing_required_visuals),
            "present_required_visuals": ";".join(self.present_required_visuals),
            "present_files": ";".join(self.present_files),
            "audit_plot": self.audit_plot,
            "claim_scope": self.claim_scope,
            "confidence_note": self.confidence_note,
        }


@dataclass(frozen=True)
class VisualAuditRun:
    """Output bundle for a visual audit run."""

    rows: list[VisualAuditRow]
    csv_path: Path
    md_path: Path
    index_plot_path: Path
    experiment_plot_paths: list[Path]


def default_visual_audit_manifest() -> tuple[VisualAuditExperiment, ...]:
    """Return the T24 historical-experiment audit manifest."""

    return (
        VisualAuditExperiment(
            task_id="T13",
            slug="t13_16e_fem_digits",
            title="T13 16e FEM digit replay",
            claim_scope="16e {ad}/{ad} FEM+RM digit metrics and 8e/16e comparison.",
            artifacts=(
                VisualAuditArtifact(
                    "metrics_csv",
                    "eit_digits_pyeidors_fem_16e.csv",
                    "csv",
                    "16e metrics CSV",
                ),
                VisualAuditArtifact(
                    "comparison_plot",
                    "eit_digit_plot_16e_compare.png",
                    "plot",
                    "8e/16e metric plot",
                ),
                VisualAuditArtifact(
                    "report_md",
                    "eit_digit_report_16e_compare.md",
                    "md",
                    "comparison report",
                ),
            ),
            required_visual_keys=("comparison_plot", "field_map", "point_audit"),
            trusted_note="Has comparison plot plus field and point audit.",
            smoke_note=(
                "Legacy metric-only result; lacks conductivity field map or "
                "measurement point audit, so it is smoke evidence only."
            ),
        ),
        VisualAuditExperiment(
            task_id="T14",
            slug="t14_voltage_digits",
            title="T14 voltage digit sweep",
            claim_scope="Target voltage digits {4,5,6,7} versus conductivity metrics.",
            artifacts=(
                VisualAuditArtifact(
                    "summary_csv",
                    "eit_voltage_digit_sweep_16e.csv",
                    "csv",
                    "digit sweep summary",
                ),
                VisualAuditArtifact(
                    "field_csv",
                    "eit_voltage_digit_fields_16e.csv",
                    "csv",
                    "per-cell field CSV",
                ),
                VisualAuditArtifact(
                    "metric_plot",
                    "eit_voltage_digit_sweep_16e.png",
                    "plot",
                    "digit metric plot",
                ),
            ),
            required_visual_keys=("metric_plot", "field_map"),
            trusted_note="Has digit metric plot and conductivity field map.",
            smoke_note=(
                "Has metric plot and per-cell CSV, but no visual "
                "sigma_true/sigma_recon/error field audit for this sweep."
            ),
        ),
        VisualAuditExperiment(
            task_id="T15",
            slug="t15_factor_sweep",
            title="T15 factor sweep",
            claim_scope="Controlled factor ranking and grid x ridge interaction.",
            artifacts=(
                VisualAuditArtifact(
                    "summary_csv",
                    "eit_factor_sweep_16e.csv",
                    "csv",
                    "factor sweep CSV",
                ),
                VisualAuditArtifact(
                    "ranking_report",
                    "eit_factor_sweep_16e.md",
                    "md",
                    "factor ranking report",
                ),
                VisualAuditArtifact(
                    "factor_plot",
                    "eit_factor_sweep_16e.png",
                    "plot",
                    "factor ranking plot",
                ),
            ),
            required_visual_keys=("factor_plot", "field_map"),
            trusted_note="Has factor plot and matching field-level audit.",
            smoke_note=(
                "Ranking plot exists, but conclusion lacks field-level visual audit; "
                "treat factor ordering as smoke until dense visual rerun."
            ),
        ),
        VisualAuditExperiment(
            task_id="T17",
            slug="t17_factor_sweep_extended",
            title="T17 extended factor sweep",
            claim_scope="Extended factor levels: full-scale, noise, RM mode, anomalies.",
            artifacts=(
                VisualAuditArtifact(
                    "summary_csv",
                    "eit_factor_sweep_t17_16e.csv",
                    "csv",
                    "extended factor CSV",
                ),
                VisualAuditArtifact(
                    "ranking_report",
                    "eit_factor_sweep_t17_16e.md",
                    "md",
                    "extended ranking report",
                ),
                VisualAuditArtifact(
                    "factor_plot",
                    "eit_factor_sweep_t17_16e.png",
                    "plot",
                    "extended factor plot",
                ),
                VisualAuditArtifact(
                    "noser_plot",
                    "eit_factor_sweep_t17_noser_exponent_levels_16e.png",
                    "plot",
                    "NOSER exponent plot",
                ),
            ),
            required_visual_keys=("factor_plot", "field_map"),
            trusted_note="Has extended factor plot and field-level audit.",
            smoke_note=(
                "Extended ranking has plots, but no matching truth/recon/error "
                "field audit per changed factor; keep as smoke."
            ),
        ),
    )


def _selected_manifest(
    experiments: Iterable[VisualAuditExperiment],
    slugs: Iterable[str] | None,
) -> list[VisualAuditExperiment]:
    experiment_list = list(experiments)
    if slugs is None:
        return experiment_list
    requested = {str(slug).strip() for slug in slugs if str(slug).strip()}
    if not requested:
        return experiment_list
    by_slug = {experiment.slug: experiment for experiment in experiment_list}
    missing = sorted(requested - set(by_slug))
    if missing:
        raise ValueError(f"unknown visual audit slug(s): {', '.join(missing)}")
    return [
        experiment for experiment in experiment_list if experiment.slug in requested
    ]


def _artifact_path(base_dir: Path, artifact: VisualAuditArtifact) -> Path:
    return Path(base_dir) / artifact.path


def evaluate_visual_audit(
    *,
    output_dir: Path,
    audit_output_dir: Path | None = None,
    experiments: Iterable[VisualAuditExperiment] | None = None,
    slugs: Iterable[str] | None = None,
) -> list[VisualAuditRow]:
    """Evaluate audit status for configured experiments."""

    base = Path(output_dir)
    audit_dir = base if audit_output_dir is None else Path(audit_output_dir)
    selected = _selected_manifest(
        experiments or default_visual_audit_manifest(),
        slugs,
    )
    rows: list[VisualAuditRow] = []
    for experiment in selected:
        artifact_by_key = {artifact.key: artifact for artifact in experiment.artifacts}
        present_files = []
        for artifact in experiment.artifacts:
            path = _artifact_path(base, artifact)
            if path.exists():
                present_files.append(artifact.path)

        present_required = []
        missing_required = []
        for key in experiment.required_visual_keys:
            artifact = artifact_by_key.get(key)
            if artifact is not None and _artifact_path(base, artifact).exists():
                present_required.append(key)
            else:
                missing_required.append(key)

        audited = not missing_required
        rows.append(
            VisualAuditRow(
                task_id=experiment.task_id,
                slug=experiment.slug,
                title=experiment.title,
                audit_status="trusted/audited" if audited else "untrusted/smoke",
                confidence_level="visual-audited" if audited else "smoke-only",
                missing_required_visuals=tuple(missing_required),
                present_required_visuals=tuple(present_required),
                present_files=tuple(present_files),
                audit_plot=str(audit_dir / f"eit_visual_audit_{experiment.slug}.png"),
                claim_scope=experiment.claim_scope,
                confidence_note=experiment.trusted_note
                if audited
                else experiment.smoke_note,
            )
        )
    return rows


def write_visual_audit_csv(rows: Iterable[VisualAuditRow], output_path: Path) -> Path:
    """Write the audit status table."""

    row_list = list(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_INDEX_FIELDS)
        writer.writeheader()
        for row in row_list:
            writer.writerow(row.as_csv_row())
    return output


def format_visual_audit_markdown(rows: Iterable[VisualAuditRow]) -> str:
    """Format a Chinese visual-audit confidence report."""

    row_list = list(rows)
    lines = [
        "# T24 历史实验 visual audit 索引",
        "",
        "本索引只评价历史实验能否支撑结论，不改变原始数值。`untrusted/smoke` "
        "表示只能当冒烟或线索，不能作为最终物理结论。",
        "",
        "| task | experiment | status | missing required visuals | confidence note |",
        "|---|---|---|---|---|",
    ]
    for row in row_list:
        missing = (
            ", ".join(f"`{item}`" for item in row.missing_required_visuals)
            if row.missing_required_visuals
            else "-"
        )
        lines.append(
            f"| {row.task_id} | `{row.slug}` | `{row.audit_status}` | "
            f"{missing} | {row.confidence_note} |"
        )

    trusted = [row for row in row_list if row.audit_status == "trusted/audited"]
    smoke = [row for row in row_list if row.audit_status != "trusted/audited"]
    lines.extend(
        [
            "",
            "## 可信度口径",
            "",
            f"- visual-audited：{', '.join(row.task_id for row in trusted) or '-'}。",
            f"- untrusted/smoke：{', '.join(row.task_id for row in smoke) or '-'}。",
            "- T13/T14/T15/T17 的旧结论若缺真值/重建/误差或点位审计图，均只能作为筛查线索。",
            "- 粗方形历史实验已移除；密集圆桶结论应看 T23。",
            "",
            "## 审计图",
            "",
        ]
    )
    for row in row_list:
        lines.append(f"- `{row.slug}` → `{row.audit_plot}`")
    return "\n".join(lines) + "\n"


def write_visual_audit_markdown(
    rows: Iterable[VisualAuditRow], output_path: Path
) -> Path:
    """Write the Chinese audit report."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(format_visual_audit_markdown(rows), encoding="utf-8")
    return output


def _wrap_text(text: str, width: int = 64) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def _draw_text_panel(ax, title: str, lines: Iterable[str]) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=12, loc="left")
    ax.text(
        0.0,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=9,
        transform=ax.transAxes,
    )


def _draw_image_or_placeholder(ax, path: Path, *, label: str) -> None:
    ax.set_title(label, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    if not path.exists():
        ax.text(
            0.5,
            0.5,
            f"Missing\n{path.name}",
            ha="center",
            va="center",
            fontsize=9,
            transform=ax.transAxes,
        )
        for spine in ax.spines.values():
            spine.set_color("#d62728")
            spine.set_linewidth(1.4)
        return
    try:
        import matplotlib.image as mpimg

        image = mpimg.imread(path)
    except Exception as exc:  # pragma: no cover - defensive rendering guard.
        ax.text(
            0.5,
            0.5,
            f"Unreadable\n{path.name}\n{exc}",
            ha="center",
            va="center",
            fontsize=8,
            transform=ax.transAxes,
        )
        return
    ax.imshow(image)


def plot_visual_audit_experiment(
    *,
    experiment: VisualAuditExperiment,
    row: VisualAuditRow,
    output_dir: Path,
    output_path: Path,
    dpi: int = 160,
) -> Path:
    """Render one experiment's visual audit panel."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .digit_plot import configure_times_new_roman

    configure_times_new_roman()
    output = Path(output_path).with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    plot_artifacts = [
        artifact for artifact in experiment.artifacts if artifact.kind == "plot"
    ]
    plot_keys = {artifact.key for artifact in plot_artifacts}
    for key in experiment.required_visual_keys:
        if key not in plot_keys:
            plot_artifacts.append(
                VisualAuditArtifact(
                    key=key,
                    path=f"missing_{key}.png",
                    kind="plot",
                    label=f"missing {key}",
                )
            )

    max_panels = max(1, min(4, len(plot_artifacts)))
    fig, axes = plt.subplots(
        2,
        max_panels,
        figsize=(4.2 * max_panels, 6.4),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(f"{experiment.task_id} visual audit: {experiment.title}", fontsize=14)

    text_lines = [
        f"status: {row.audit_status}",
        f"confidence: {row.confidence_level}",
        f"scope: {_wrap_text(row.claim_scope, 58)}",
        f"missing: {', '.join(row.missing_required_visuals) or '-'}",
        f"note: {_wrap_text(row.confidence_note, 58)}",
        "present files:",
        _wrap_text(", ".join(row.present_files) or "-", 58),
    ]
    _draw_text_panel(axes[0, 0], "Audit status", text_lines)
    for ax in axes[0, 1:]:
        ax.axis("off")

    for ax, artifact in zip(axes[1], plot_artifacts[:max_panels], strict=False):
        _draw_image_or_placeholder(
            ax,
            _artifact_path(Path(output_dir), artifact),
            label=artifact.label,
        )
    for ax in axes[1, len(plot_artifacts[:max_panels]) :]:
        ax.axis("off")

    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output


def plot_visual_audit_index(
    rows: Iterable[VisualAuditRow],
    output_path: Path,
    *,
    dpi: int = 160,
) -> Path:
    """Render a compact all-experiment audit status table."""

    row_list = list(rows)
    if not row_list:
        raise ValueError("rows must not be empty")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .digit_plot import configure_times_new_roman

    configure_times_new_roman()
    output = Path(output_path).with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.5, max(4.5, 0.72 * len(row_list) + 2.4)))
    ax.axis("off")
    fig.suptitle("T24 historical EIT visual audit index", fontsize=15)
    columns = ["Task", "Experiment", "Status", "Missing required visuals", "Confidence"]
    cell_text = []
    for row in row_list:
        confidence = (
            "visual audit present"
            if row.audit_status == "trusted/audited"
            else "smoke only; missing audit panels"
        )
        cell_text.append(
            [
                row.task_id,
                row.slug,
                row.audit_status,
                ", ".join(row.missing_required_visuals) or "-",
                confidence,
            ]
        )
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.07, 0.19, 0.16, 0.23, 0.35],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.45)
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#eaeaea")
            cell.set_text_props(weight="bold")
        elif col_idx == 2:
            value = cell.get_text().get_text()
            cell.set_facecolor("#e8f4ea" if value == "trusted/audited" else "#fbe8e8")
    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output


def run_visual_audit(
    *,
    output_dir: Path | None = None,
    audit_output_dir: Path | None = None,
    slugs: Iterable[str] | None = None,
    dpi: int = 160,
) -> VisualAuditRun:
    """Generate all T24 visual audit CSV, Markdown, and PNG outputs."""

    base = (
        pyeidors_output_path("visual_audit") if output_dir is None else Path(output_dir)
    )
    audit_dir = base if audit_output_dir is None else Path(audit_output_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    experiments = _selected_manifest(default_visual_audit_manifest(), slugs)
    rows = evaluate_visual_audit(
        output_dir=base,
        audit_output_dir=audit_dir,
        experiments=experiments,
    )
    by_slug = {experiment.slug: experiment for experiment in experiments}
    experiment_plot_paths = []
    for row in rows:
        plot_path = plot_visual_audit_experiment(
            experiment=by_slug[row.slug],
            row=row,
            output_dir=base,
            output_path=Path(row.audit_plot),
            dpi=dpi,
        )
        experiment_plot_paths.append(plot_path)

    csv_path = write_visual_audit_csv(rows, audit_dir / "eit_visual_audit_index.csv")
    md_path = write_visual_audit_markdown(rows, audit_dir / "eit_visual_audit_index.md")
    index_plot = plot_visual_audit_index(
        rows,
        audit_dir / "eit_visual_audit_index.png",
        dpi=dpi,
    )
    return VisualAuditRun(
        rows=rows,
        csv_path=csv_path,
        md_path=md_path,
        index_plot_path=index_plot,
        experiment_plot_paths=experiment_plot_paths,
    )
