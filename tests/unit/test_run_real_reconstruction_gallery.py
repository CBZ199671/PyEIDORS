"""Contract tests for the real reconstruction gallery diagnostics script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def _load_module():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "diagnostics"
        / "run_real_reconstruction_gallery.py"
    )
    spec = importlib.util.spec_from_file_location("run_real_reconstruction_gallery", script)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise AssertionError("failed to load run_real_reconstruction_gallery.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_safe_pearson_handles_constant_equal_arrays():
    module = _load_module()
    values = np.ones(5, dtype=float)
    assert module._safe_pearson(values, values.copy()) == 1.0


def test_consistency_metrics_apply_existing_dim_thresholds():
    module = _load_module()
    metrics_2d = module._consistency_metrics(
        dim=2,
        baseline_cpu_meas=np.array([1.0, 2.0]),
        baseline_gpu_meas=np.array([1.0, 2.0 + 1e-9]),
        target_cpu_meas=np.array([2.0, 3.0]),
        target_gpu_meas=np.array([2.0, 3.0 + 1e-9]),
        cpu_recon=np.array([1.0, 1.1]),
        gpu_recon=np.array([1.0, 1.1 + 1e-9]),
    )
    metrics_3d = module._consistency_metrics(
        dim=3,
        baseline_cpu_meas=np.array([1.0, 2.0]),
        baseline_gpu_meas=np.array([1.0, 2.0 + 1e-9]),
        target_cpu_meas=np.array([2.0, 3.0]),
        target_gpu_meas=np.array([2.0, 3.0 + 1e-9]),
        cpu_recon=np.array([1.0, 1.1]),
        gpu_recon=np.array([1.0, 1.1 + 1e-9]),
    )

    assert metrics_2d["measurement_pass"] is True
    assert metrics_2d["baseline_measurement_pass"] is True
    assert metrics_2d["target_measurement_pass"] is True
    assert metrics_2d["image_pass"] is True
    assert metrics_2d["image_rmse_threshold"] == 1e-6
    assert metrics_3d["image_rmse_threshold"] == 1.25e-6


def test_write_report_embeds_expected_figure_links(tmp_path: Path):
    module = _load_module()
    report_path = tmp_path / "report.md"
    module._write_report(
        output_path=report_path,
        title="Gallery",
        figures={"2d_overview": "figures/2d_overview.png", "3d_overview": "figures/3d_overview.png"},
        config={"refinement_3d": 3},
        case_rows=[
            {"case": "2D CPU", "forward_backend": "dolfinx", "forward_sec": 1.0, "inverse_sec": 2.0},
        ],
        consistency_rows=[
            {"passed": True},
            {"passed": False},
        ],
        fairness_order_rows=[
            {
                "dimension": "3D",
                "order": "CPU->GPU",
                "cold_forward_speedup_x": 2.0,
                "hot_forward_speedup_x": 3.0,
                "cold_inverse_speedup_x": 1.5,
                "hot_inverse_speedup_x": 1.6,
            }
        ],
        fairness_backend_rows=[
            {"dimension": "3D", "backend": "CPU", "report_only": False, "passed": True},
            {"dimension": "3D", "backend": "GPU", "report_only": False, "passed": True},
        ],
        all_passed=False,
    )
    text = report_path.read_text(encoding="utf-8")
    assert "![2D overview](figures/2d_overview.png)" in text
    assert "![3D overview](figures/3d_overview.png)" in text
    assert "2D consistency: PASS" in text
    assert "3D consistency: FAIL" in text
    assert "3D fairness: PASS" in text
    assert "Speed summary:" in text


def test_worker_command_uses_separate_worker_script(tmp_path: Path):
    module = _load_module()
    args = module._parse_args.__globals__["argparse"].Namespace(
        output_dir=tmp_path,
        n_elec=16,
        mesh_size_2d=0.08,
        radius_2d=1.0,
        radius_3d=0.18,
        height_3d=0.16,
        refinement_3d=3,
        electrode_height_ratio=0.2,
        electrode_coverage=0.5,
        contact_impedance=1e-5,
        max_iterations=2,
        slice_resolution=220,
        report_title="Gallery",
    )
    cmd = module._worker_command(
        args,
        dim=3,
        output_dir=tmp_path / "worker-run",
        worker_output_json=tmp_path / "worker.json",
        run_kind="fairness",
        backend_order="gpu-first",
    )
    assert cmd[1].endswith("run_real_reconstruction_gallery_worker.py")
    assert "--worker-dim" not in cmd
    assert "--run-kind" in cmd
    assert "--backend-order" in cmd
