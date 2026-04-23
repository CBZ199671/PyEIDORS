from __future__ import annotations

import csv
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from pyeidors.data.bucket_domain_audit import (
    BUCKET_DOMAIN_AUDIT_FIELDS,
    build_bucket_domain_audit_rows,
    build_circle_bucket_domain,
    format_bucket_domain_report,
    plot_bucket_domain_audit,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_circle_bucket_mesh_is_dense_disk_not_square_clip() -> None:
    bucket = build_circle_bucket_domain(mesh_h=0.08, n_elec=16)

    node_radii = np.linalg.norm(bucket.nodes, axis=1)
    center_radii = np.linalg.norm(bucket.cell_centers, axis=1)
    assert bucket.domain == "circle_bucket"
    assert bucket.n_elec == 16
    assert bucket.n_measurements == 208
    assert bucket.n_cells >= 800
    assert bucket.n_dofs == bucket.n_cells
    assert bucket.dense_threshold_passed
    assert np.max(node_radii) <= bucket.bucket_radius + 1e-10
    assert np.max(center_radii) <= bucket.bucket_radius + 1e-10
    assert np.max(bucket.nodes[:, 0]) > 0.99 * bucket.bucket_radius
    assert np.min(bucket.nodes[:, 0]) < -0.99 * bucket.bucket_radius
    assert np.max(bucket.nodes[:, 1]) > 0.99 * bucket.bucket_radius
    assert np.min(bucket.nodes[:, 1]) < -0.99 * bucket.bucket_radius


def test_circle_bucket_electrodes_are_equal_angle_and_audited() -> None:
    bucket = build_circle_bucket_domain(
        mesh_h=0.08,
        n_elec=16,
        electrode_coverage=0.5,
    )
    rows = build_bucket_domain_audit_rows(bucket)

    assert len(rows) == 16
    assert rows[0].as_csv_row().keys() == set(BUCKET_DOMAIN_AUDIT_FIELDS)
    assert {row.n_measurements for row in rows} == {208}
    assert {row.stim_pattern for row in rows} == {"{ad}"}
    assert {row.meas_pattern for row in rows} == {"{ad}"}
    assert {row.n_cells for row in rows} == {bucket.n_cells}

    angles = [row.electrode_center_angle for row in rows]
    expected = [idx * 360.0 / 16.0 for idx in range(16)]
    np.testing.assert_allclose(angles, expected, atol=1e-12)
    expected_arc = bucket.bucket_radius * (2.0 * math.pi / 16.0) * 0.5
    expected_width = (
        2.0 * bucket.bucket_radius * math.sin((2.0 * math.pi / 16.0 * 0.5) / 2.0)
    )
    assert math.isclose(rows[0].electrode_arc_length, expected_arc)
    assert math.isclose(rows[0].electrode_width, expected_width)


def test_circle_bucket_truth_and_plot_outputs(tmp_path) -> None:
    bucket = build_circle_bucket_domain(mesh_h=0.08, n_elec=16)
    report = format_bucket_domain_report(bucket)
    output = plot_bucket_domain_audit(bucket, tmp_path / "bucket.png", dpi=80)

    assert "圆形小水桶域审计" in report
    assert "n_measurements=208" in report
    assert (
        np.count_nonzero(np.isclose(bucket.sigma_true, bucket.anomaly_conductivity)) > 0
    )
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 1000


def test_eit_bucket_domain_audit_cli_writes_expected_outputs(tmp_path) -> None:
    csv_output = tmp_path / "audit.csv"
    plot_output = tmp_path / "audit.png"
    report_output = tmp_path / "audit.md"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/eit_bucket_domain_audit.py",
            "--domain",
            "circle_bucket",
            "--bucket-radius",
            "1",
            "--n-elec",
            "16",
            "--mesh-h",
            "0.08",
            "--plot",
            "--output",
            str(csv_output),
            "--plot-output",
            str(plot_output),
            "--report-output",
            str(report_output),
            "--dpi",
            "80",
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert "domain          = circle_bucket" in completed.stdout
    assert "n_measurements  = 208" in completed.stdout
    with csv_output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 16
    assert list(rows[0].keys()) == BUCKET_DOMAIN_AUDIT_FIELDS
    assert {row["domain"] for row in rows} == {"circle_bucket"}
    assert {row["n_measurements"] for row in rows} == {"208"}
    assert int(rows[0]["n_cells"]) >= 800
    assert report_output.read_text(encoding="utf-8").startswith(
        "# T22 圆形小水桶域审计"
    )
    assert plot_output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
