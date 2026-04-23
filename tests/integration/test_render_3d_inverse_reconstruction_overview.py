"""Integration tests for 3D reconstruction diagnostics export."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pyeidors.geometry.mesh3d_generator import GMSH_AVAILABLE
from scripts.common.hdf5_outputs import read_output_bundle

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "diagnostics"
    / "render_3d_inverse_reconstruction_overview.py"
)


def _run(
    args: list[str], *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    merged_env = dict(os.environ)
    merged_env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    merged_env.setdefault("OMP_NUM_THREADS", "1")
    if env:
        merged_env.update(env)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        env=merged_env,
    )


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
def test_render_3d_overview_exports_difference_artifacts(tmp_path: Path):
    output_dir = tmp_path / "render_3d"
    run = _run(
        [
            "--output-dir",
            str(output_dir),
            "--refinement",
            "1",
            "--max-iterations",
            "1",
            "--inverse-mode",
            "difference",
            "--difference-mode",
            "normalized",
            "--difference-orientation",
            "target_minus_reference",
            "--electrode-level-fractions",
            "0.25,0.75",
        ]
    )
    assert run.returncode == 0, run.stderr

    metrics_path = output_dir / "inverse_3d_overview_metrics.json"
    data_path = output_dir / "inverse_3d_overview_data.h5"
    assert metrics_path.exists()
    assert data_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["inverse_mode"] == "difference"
    assert metrics["difference_mode"] == "normalized"
    assert metrics["difference_orientation"] == "target_minus_reference"
    assert metrics["measurement_space"]["type"] == "difference"
    assert metrics["inverse_target"] == "eidors_3d_difference_one_step_gn_noser"
    assert metrics["preset_name"] == "eidors_one_step_noser"
    assert "hyperparameter" in metrics
    assert "lambda_eff" in metrics
    assert "contrast_recovery" in metrics
    assert "difference_step_size" in metrics
    assert "shape_metrics" in metrics
    assert "wall_time_breakdown" in metrics
    for key in (
        "setup_elapsed_sec",
        "solve_elapsed_sec",
        "postprocess_elapsed_sec",
        "save_elapsed_sec",
    ):
        assert key in metrics["wall_time_breakdown"]

    payload = read_output_bundle(data_path)
    for key in (
        "coords",
        "truth_sigma",
        "recon_sigma",
        "vh",
        "vi",
        "dv_raw",
        "dv_norm",
        "dv_measurement_space",
        "pred_vi",
        "pred_dv_measurement_space",
        "measurement_vector",
        "prediction_vector",
        "residual_vector",
    ):
        assert key in payload


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
def test_render_3d_overview_exports_absolute_artifacts(tmp_path: Path):
    output_dir = tmp_path / "render_3d_absolute"
    run = _run(
        [
            "--output-dir",
            str(output_dir),
            "--refinement",
            "1",
            "--max-iterations",
            "1",
            "--inverse-mode",
            "absolute",
            "--difference-mode",
            "normalized",
            "--difference-orientation",
            "target_minus_reference",
            "--electrode-level-fractions",
            "0.25,0.75",
        ]
    )
    assert run.returncode == 0, run.stderr

    metrics = json.loads(
        (output_dir / "inverse_3d_overview_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["inverse_mode"] == "absolute"
    assert metrics["measurement_space"]["type"] == "real"
    assert metrics["inverse_target"] == "eidors_abs_gn_prior"
    assert metrics["preset_name"] == "eidors_abs_gn"
    assert "best_homog" in metrics
    assert "wall_time_breakdown" in metrics

    payload = read_output_bundle(output_dir / "inverse_3d_overview_data.h5")
    for key in (
        "coords",
        "truth_sigma",
        "recon_sigma",
        "vh",
        "vi",
        "measurement_vector",
        "prediction_vector",
        "residual_vector",
        "target_mask",
        "background_mask",
    ):
        assert key in payload


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
def test_render_3d_overview_supports_no_plot_and_no_save_data(tmp_path: Path):
    output_dir = tmp_path / "render_3d_no_artifacts"
    run = _run(
        [
            "--output-dir",
            str(output_dir),
            "--refinement",
            "1",
            "--max-iterations",
            "1",
            "--inverse-mode",
            "difference",
            "--no-plot",
            "--no-save-data",
        ]
    )
    assert run.returncode == 0, run.stderr
    assert '"inverse_mode": "difference"' in run.stdout
    assert '"wall_time_breakdown"' in run.stdout
    assert not (output_dir / "inverse_3d_overview.png").exists()
    assert not (output_dir / "inverse_3d_overview.svg").exists()
