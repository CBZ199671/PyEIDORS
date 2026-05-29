"""Tests for the 48e/5936 dual-model RM benchmark report contract."""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np


def _load_script_module(*parts: str):
    script = Path(__file__).resolve().parents[2].joinpath(*parts)
    spec = importlib.util.spec_from_file_location("_".join(parts), script)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise AssertionError(f"failed to load script: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dual_model_rm_benchmark_writes_t36_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script_module(
        "scripts", "benchmarks", "benchmark_dual_model_rm_v1.py"
    )
    out_dir = tmp_path / "bench"
    forward_ref = tmp_path / "forward.json"
    lazy_ref = tmp_path / "lazy.json"
    previous_ref = tmp_path / "previous_greit.json"
    forward_ref.write_text(
        json.dumps(
            {
                "forward_solver_benchmark": {
                    "solver_preset": "spd_gamg",
                    "ksp_type": "cg",
                    "pc_type": "gamg",
                    "mat_solve_effective": "vec-loop",
                    "petsc_device_effective": "cuda",
                    "setup_seconds": 0.1,
                    "solve_seconds": 0.2,
                    "n_patterns": 48,
                },
                "mesh_info": {
                    "nodes": 10,
                    "elements": 20,
                    "potential_dofs": 10,
                    "sigma_dofs": 20,
                    "mesh_family": "tetra",
                    "geometry_version": "geomv2",
                },
            }
        ),
        encoding="utf-8",
    )
    lazy_ref.write_text(
        json.dumps(
            {
                "cold_context": {
                    "context_build_seconds": 3.0,
                    "mesh_cache_hit": False,
                    "mesh_cache_layer": "generated",
                    "jacobian_shape": [12, 8],
                    "n_meas_total": 12,
                    "torch_device": "cuda",
                    "cache_build_seconds": {
                        "mesh": 0.1,
                        "base_meas": 0.2,
                        "jacobian": 0.3,
                        "operator_noser": 0.01,
                        "operator_precond": 0.02,
                    },
                    "petsc_backend_info": {
                        "solver_preset": "spd_gamg",
                        "petsc_device_effective": "cuda",
                        "forward_mat_solve_effective": "vec-loop",
                        "pc_type": "gamg",
                        "pc_gamg_type": "agg",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    previous_ref.write_text(
        json.dumps(
            {
                "warm_seconds": {
                    "apply_cpu_1_frame": 0.2,
                    "apply_cpu_512_frames": 10.0,
                    "apply_auto_1_frame": 0.3,
                    "apply_auto_512_frames": 12.0,
                    "artifact_load": 0.1,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_dual_model_rm_v1.py",
            "--output-dir",
            str(out_dir),
            "--coarse-shape",
            "2,2,2",
            "--fine-per-coarse",
            "2",
            "--n-measurements",
            "12",
            "--n-elec",
            "6",
            "--n-rings",
            "2",
            "--n-frames",
            "4",
            "--devices",
            "cpu",
            "--forward-reference",
            str(forward_ref),
            "--lazy-reference",
            str(lazy_ref),
            "--previous-greit-reference",
            str(previous_ref),
        ],
    )

    assert module.main() == 0

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == "pyeidors-dual-model-rm-v1-benchmark"
    assert summary["sizes"]["measurements"] == 12
    assert summary["sizes"]["frames"] == 4
    assert summary["forward_reference"]["forward_solver"]["solver_preset"] == "spd_gamg"
    assert summary["forward_reference"]["lazy_context"]["n_meas_total"] == 12
    assert set(summary["rm_builds"]) == {"noser", "laplace"}
    assert set(summary["artifact_load"]) == {"noser", "laplace", "greit"}
    assert summary["artifacts"]["greit_rm"].endswith(".h5")
    assert summary["artifacts"]["one_step_noser_rm"].endswith(".h5")
    assert summary["artifacts"]["one_step_laplace_rm"].endswith(".h5")
    assert set(summary["online_apply"]) == {"noser", "laplace", "greit"}
    assert set(summary["greit"]["metric_keys"]) == {"AR", "PE", "RES", "SD", "RNG"}
    assert summary["previous_greit_reference"]["found"] is True
    assert "cpu" in summary["previous_greit_reference"]["comparisons"]

    noser_cpu = summary["online_apply"]["noser"]["cpu"]
    assert noser_cpu["apply_batch_n_frames"] == 4
    assert noser_cpu["metadata_batch"]["online_hot_path"] == "rm_matmul"
    assert noser_cpu["metadata_batch"]["forward_solve_count"] == 0
    assert noser_cpu["metadata_batch"]["ksp_solve_count"] == 0
    assert noser_cpu["metadata_batch"]["rm_prepare_mode"] == "reused_handle"

    report = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "48e/5936 Dual-Model RM Runtime Report" in report
    assert "RM Build And Load" in report
    assert "Online Apply" in report
    assert "Previous GREIT Baseline" in report


def test_v536_dual_model_benchmark_direct_fills_fine_mesh_and_jacobian() -> None:
    module = _load_script_module(
        "scripts", "benchmarks", "benchmark_dual_model_rm_v1.py"
    )

    fine_source = inspect.getsource(module._build_fine_mesh)
    jac_source = inspect.getsource(module._build_synthetic_coarse_j)
    projection_source = inspect.getsource(module._coarse_j_to_fine_j)
    assert "np.vstack" not in fine_source
    assert "np.vstack" not in jac_source
    assert "np.diag" not in projection_source
    assert "coarse_j * inv_counts.reshape(1, -1)" in projection_source

    coarse = module.VoxelGrid.from_bounds(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        shape=(2, 1, 1),
        name="test-coarse",
    )
    fine = module._build_fine_mesh(coarse, fine_per_coarse=2)
    assert fine.num_cells() == coarse.num_cells() * 2

    centers = coarse.cell_centers()
    jacobian = module._build_synthetic_coarse_j(
        centers,
        n_measurements=4,
        n_elec=6,
        n_rings=2,
    )
    assert jacobian.shape == (4, centers.shape[0])
    np.testing.assert_allclose(np.linalg.norm(jacobian, axis=1), 1.0)
