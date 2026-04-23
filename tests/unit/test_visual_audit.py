from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys

from pyeidors.data.visual_audit import (
    default_visual_audit_manifest,
    evaluate_visual_audit,
    run_visual_audit,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_tiny_png(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(0.7, 0.5))
    ax.plot([0, 1], [0, 1])
    ax.set_axis_off()
    fig.savefig(path, dpi=40)
    plt.close(fig)


def test_visual_audit_manifest_covers_t24_history_tasks() -> None:
    manifest = default_visual_audit_manifest()

    assert {experiment.task_id for experiment in manifest} == {
        "T13",
        "T14",
        "T15",
        "T17",
    }
    assert all(experiment.required_visual_keys for experiment in manifest)


def test_visual_audit_marks_missing_required_visuals_as_smoke(tmp_path) -> None:
    (tmp_path / "eit_digits_pyeidors_fem_16e.csv").write_text(
        "bit,voltage_rmse\n16,0.0\n",
        encoding="utf-8",
    )

    rows = evaluate_visual_audit(output_dir=tmp_path)
    t13 = next(row for row in rows if row.task_id == "T13")

    assert t13.audit_status == "untrusted/smoke"
    assert "field_map" in t13.missing_required_visuals
    assert "point_audit" in t13.missing_required_visuals


def test_run_visual_audit_writes_index_without_removed_square_tasks(tmp_path) -> None:
    _write_tiny_png(tmp_path / "eit_factor_sweep_16e.png")
    (tmp_path / "eit_factor_sweep_16e.csv").write_text(
        "sweep,changed_factor,level\nsingle_factor,ridge,0.01\n",
        encoding="utf-8",
    )

    run = run_visual_audit(
        output_dir=tmp_path,
        audit_output_dir=tmp_path / "audit",
        slugs=["t15_factor_sweep"],
        dpi=70,
    )

    assert run.csv_path.read_text(encoding="utf-8").startswith("task_id,slug")
    assert run.md_path.read_text(encoding="utf-8").startswith(
        "# T24 历史实验 visual audit 索引"
    )
    assert run.index_plot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert run.experiment_plot_paths[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with run.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["task_id"] == "T15"
    assert rows[0]["audit_status"] == "untrusted/smoke"
    assert rows[0]["missing_required_visuals"] == "field_map"


def test_eit_visual_audit_index_cli_writes_outputs(tmp_path) -> None:
    _write_tiny_png(tmp_path / "eit_voltage_digit_sweep_16e.png")
    (tmp_path / "eit_voltage_digit_sweep_16e.csv").write_text(
        "target_voltage_digits,sigma_relative_rmse\n5,0.1\n",
        encoding="utf-8",
    )

    audit_dir = tmp_path / "audit"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/eit_visual_audit_index.py",
            "--output-dir",
            str(tmp_path),
            "--audit-output-dir",
            str(audit_dir),
            "--experiments",
            "t14_voltage_digits",
            "--dpi",
            "70",
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert "t14_voltage_digits" in completed.stdout
    assert "untrusted/smoke" in completed.stdout
    assert (audit_dir / "eit_visual_audit_index.csv").exists()
    assert (audit_dir / "eit_visual_audit_t14_voltage_digits.png").exists()
