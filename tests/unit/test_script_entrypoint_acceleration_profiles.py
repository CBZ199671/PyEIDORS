"""Lightweight contract tests for script entrypoints using acceleration profiles."""

from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


def _load_script_module(*parts: str):
    script = Path(__file__).resolve().parents[2].joinpath(*parts)
    spec = importlib.util.spec_from_file_location("_".join(parts), script)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise AssertionError(f"failed to load script: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_3d_runtime_parser_accepts_acceleration_profile(monkeypatch):
    module = _load_script_module("scripts", "benchmarks", "benchmark_3d_runtime.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_3d_runtime.py",
            "--acceleration-profile",
            "gpu3d_fused",
        ],
    )

    args = module._parse_args()

    assert args.acceleration_profile == "gpu3d_fused"


def test_benchmark_3d_runtime_parser_accepts_forward_solver_artifact_options(
    monkeypatch,
):
    module = _load_script_module("scripts", "benchmarks", "benchmark_3d_runtime.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_3d_runtime.py",
            "--forward-only",
            "on",
            "--forward-solver-preset",
            "3d_gamg",
        ],
    )

    args = module._parse_args()

    assert args.forward_only == "on"
    assert args.forward_solver_preset == "3d_gamg"


def test_benchmark_3d_runtime_builds_forward_solver_artifact():
    module = _load_script_module("scripts", "benchmarks", "benchmark_3d_runtime.py")
    args = Namespace(
        n_elec=8,
        forward_solver_preset="3d_gamg",
        forward_backend="dolfinx",
    )

    artifact = module._build_forward_solver_benchmark_artifact(
        args=args,
        mesh_info={
            "mesh_dim": 3,
            "elements": 42,
            "potential_dofs": 100,
        },
        backend_info={
            "forward_rhs_count": 8,
            "solver_preset": "3d_gamg",
            "ksp_type": "fgmres",
            "pc_type": "gamg",
            "pc_gamg_type": "agg",
            "petsc_mat_type": "seqaij",
            "petsc_vec_type": "seq",
            "forward_setup_seconds": 0.25,
            "forward_ksp_setup_count": 1,
            "forward_ksp_setup_attempts": 1,
            "forward_factor_cache_hit": False,
            "forward_reuse_preconditioner_requested": True,
            "forward_reuse_preconditioner_applied": True,
            "forward_solve_seconds": 0.5,
            "forward_ksp_iterations_per_rhs": [3, 4],
            "forward_ksp_iterations_total": 7,
            "forward_ksp_converged_reason": 2,
            "forward_ksp_converged": True,
            "forward_mat_solve_effective": "vec-loop",
            "petsc_device_requested": "auto",
            "petsc_device_effective": "cpu",
            "gpu_fallback_reason": "petsc_cuda_not_available",
            "capability": {
                "petsc_cuda": False,
                "petsc_cuda_mat": False,
                "petsc_cuda_vec": False,
                "petsc_cuda_dense": False,
                "petsc_hypre": True,
                "petsc_amgx": False,
                "petsc_amgx_cuda_candidate": False,
                "errors": {"mat": "Unknown type"},
            },
            "mpi_size": 1,
            "mpi_rank": 0,
            "mpi_parallel": False,
            "mpi_size_supported": True,
            "mpi_fallback_reason": None,
            "forward_backend_effective": "dolfinx",
            "jacobian_backend_effective": "matrix-free",
            "forward_ksp_solve_count": 8,
            "forward_ksp_mat_solve_count": 0,
        },
        timing={},
    )

    assert artifact["mesh_dim"] == 3
    assert artifact["n_cells"] == 42
    assert artifact["n_dofs"] == 109
    assert artifact["n_patterns"] == 8
    assert artifact["solver_preset"] == "3d_gamg"
    assert artifact["ksp_type"] == "fgmres"
    assert artifact["pc_type"] == "gamg"
    assert artifact["pc_subtype"] == "agg"
    assert artifact["forward_amg_backend"] == "gamg-agg"
    assert artifact["mat_type"] == "seqaij"
    assert artifact["vec_type"] == "seq"
    assert artifact["setup_seconds"] == 0.25
    assert artifact["ksp_setup_count"] == 1
    assert artifact["ksp_setup_attempts"] == 1
    assert artifact["forward_factor_cache_hit"] is False
    assert artifact["reuse_preconditioner_requested"] is True
    assert artifact["reuse_preconditioner_applied"] is True
    assert artifact["solve_seconds"] == 0.5
    assert artifact["iterations_per_rhs"] == [3, 4]
    assert artifact["converged_reason"] == 2
    assert artifact["converged"] is True
    assert artifact["mat_solve_effective"] == "vec-loop"
    assert artifact["petsc_device_effective"] == "cpu"
    assert artifact["petsc_cuda_available"] is False
    assert artifact["petsc_hypre_available"] is True
    assert artifact["petsc_amgx_available"] is False
    assert artifact["petsc_amgx_cuda_candidate"] is False
    assert artifact["petsc_cuda_errors"] == {"mat": "Unknown type"}
    assert artifact["mpi_size"] == 1
    assert artifact["mpi_size_supported"] is True
    assert artifact["mpi_fallback_reason"] is None
    assert artifact["fallback_reason"] == "petsc_cuda_not_available"
    assert artifact["forward_backend"] == "dolfinx"
    assert artifact["jacobian_backend"] == "matrix-free"


def test_benchmark_3d_runtime_forward_artifact_reports_amgx_cuda_capability():
    module = _load_script_module("scripts", "benchmarks", "benchmark_3d_runtime.py")
    args = Namespace(
        n_elec=8,
        forward_solver_preset="cuda_amgx",
        forward_backend="dolfinx",
    )

    artifact = module._build_forward_solver_benchmark_artifact(
        args=args,
        mesh_info={"mesh_dim": 3, "elements": 42, "potential_dofs": 100},
        backend_info={
            "forward_rhs_count": 8,
            "solver_preset": "cuda_amgx",
            "ksp_type": "cg",
            "pc_type": "amgx",
            "petsc_mat_type": "aijcusparse",
            "petsc_vec_type": "cuda",
            "petsc_dense_mat_type": "densecuda",
            "petsc_device_requested": "cuda",
            "petsc_device_effective": "cuda",
            "capability": {
                "petsc_cuda": True,
                "petsc_cuda_mat": True,
                "petsc_cuda_vec": True,
                "petsc_cuda_dense": True,
                "petsc_hypre": True,
                "petsc_amgx": True,
                "petsc_amgx_cuda_candidate": True,
                "errors": {},
            },
            "mpi_size": 1,
            "mpi_rank": 0,
            "mpi_parallel": False,
            "mpi_size_supported": True,
            "forward_ksp_solve_count": 8,
            "forward_ksp_mat_solve_count": 0,
        },
        timing={},
    )

    assert artifact["solver_preset"] == "cuda_amgx"
    assert artifact["ksp_type"] == "cg"
    assert artifact["pc_type"] == "amgx"
    assert artifact["forward_amg_backend"] == "amgx"
    assert artifact["petsc_device_effective"] == "cuda"
    assert artifact["petsc_cuda_available"] is True
    assert artifact["petsc_hypre_available"] is True
    assert artifact["petsc_amgx_available"] is True
    assert artifact["petsc_amgx_cuda_candidate"] is True
    assert artifact["mat_type"] == "aijcusparse"
    assert artifact["vec_type"] == "cuda"


def test_probe_petsc_cuda_script_includes_mpi_diagnostics(monkeypatch, capsys):
    module = _load_script_module("scripts", "diagnostics", "probe_petsc_cuda.py")
    monkeypatch.setattr(
        module,
        "probe_petsc_cuda_runtime",
        lambda: {"petsc_cuda": False, "errors": {"mat": "missing"}},
    )
    monkeypatch.setattr(
        module,
        "probe_mpi_runtime",
        lambda: {"mpi_size": 1, "mpi_size_supported": True},
    )
    monkeypatch.setattr(sys, "argv", ["probe_petsc_cuda.py"])

    module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["petsc_cuda"] is False
    assert payload["mpi"]["mpi_size"] == 1
    assert payload["mpi"]["mpi_size_supported"] is True


def test_benchmark_3d_fair_compare_forwards_acceleration_profile(
    monkeypatch, tmp_path: Path
):
    module = _load_script_module(
        "scripts", "benchmarks", "benchmark_3d_fair_compare.py"
    )
    report_path = tmp_path / "runtime_report.json"
    report_path.write_text(json.dumps({"stages": []}), encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_run(cmd, capture_output, text, check):  # noqa: ANN001
        captured["cmd"] = list(cmd)

        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Result()

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    payload = module._run_runtime_report(
        Path("/tmp/benchmark_3d_runtime.py"),
        output_json=report_path,
        cache_dir=tmp_path / "cache",
        mesh_dir=tmp_path / "mesh",
        refinement=2,
        profile_label="demo",
        preconditioner="auto",
        jacobian_block_tune="auto",
        jacobian_block_size=0,
        fast_linear_path="auto",
        rom_mode="off",
        rom_rank_global=32,
        rom_rank_adaptive=16,
        rom_refresh_every=2,
        rom_snapshot_source="hybrid",
        inexact_mode="off",
        inexact_forcing="eisenstat-walker",
        inexact_eta0=0.2,
        inexact_eta_min=1e-3,
        inexact_eta_max=0.5,
        lowrank_mode="off",
        lowrank_rank=16,
        lowrank_method="tsvd",
        lowrank_energy=0.995,
        cholmod_max_n=50000,
        cholmod_max_memory_gib=4.0,
        absolute_startup_cache="on",
        run_diff="on",
        run_absolute="off",
        acceleration_profile="gpu3d",
    )

    assert payload == {"stages": []}
    assert "--acceleration-profile" in captured["cmd"]
    index = captured["cmd"].index("--acceleration-profile")
    assert captured["cmd"][index + 1] == "gpu3d"


def test_benchmark_difference_runtime_parser_accepts_acceleration_profile(monkeypatch):
    module = _load_script_module(
        "scripts", "benchmarks", "benchmark_difference_runtime.py"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_difference_runtime.py",
            "--acceleration-profile",
            "gpu3d",
        ],
    )

    args = module.parse_args()

    assert args.acceleration_profile == "gpu3d"


def test_profile_reconstruction_pipeline_parser_accepts_acceleration_profile(
    monkeypatch,
):
    module = _load_script_module(
        "scripts", "benchmarks", "profile_reconstruction_pipeline.py"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_reconstruction_pipeline.py",
            "--acceleration-profile",
            "gpu3d",
        ],
    )

    args = module.parse_args()

    assert args.acceleration_profile == "gpu3d"


def test_run_synthetic_parity_forwards_acceleration_profile(
    monkeypatch, tmp_path: Path
):
    module = _load_script_module("scripts", "run_synthetic_parity.py")
    captured: dict[str, object] = {}

    class _FakeSystem:
        def __init__(self, **kwargs):
            captured["kwargs"] = dict(kwargs)

        def setup(self, *, mesh):  # noqa: ANN003
            captured["mesh"] = mesh

    monkeypatch.setattr(module, "EITSystem", _FakeSystem)
    monkeypatch.setattr(module, "load_or_create_mesh", lambda **_: "mesh")
    args = Namespace(
        n_elec=16,
        mesh_dir=tmp_path / "meshes",
        mesh_name=None,
        refinement=4,
        mesh_radius=1.0,
        electrode_coverage=0.5,
        petsc_device="cuda",
        device="cuda",
        acceleration_profile="gpu3d",
    )

    system, mesh = module.setup_eit_system(args)

    assert isinstance(system, _FakeSystem)
    assert mesh == "mesh"
    assert captured["kwargs"]["acceleration_profile"] == "gpu3d"


def test_gallery_worker_backend_settings_preserve_requested_gpu_profile():
    module = _load_script_module(
        "scripts",
        "diagnostics",
        "run_real_reconstruction_gallery_worker.py",
    )

    settings = module._backend_settings(
        dim=3,
        backend_key="gpu",
        gpu_acceleration_profile="gpu3d_fused",
    )

    assert settings["label"] == "3D GPU"
    assert settings["acceleration_profile"] == "gpu3d_fused"


def test_check_unit_consistency_parser_accepts_acceleration_profile(monkeypatch):
    module = _load_script_module("scripts", "diagnostics", "check_unit_consistency.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_unit_consistency.py",
            "--acceleration-profile",
            "gpu3d",
        ],
    )

    args = module.parse_args()

    assert args.acceleration_profile == "gpu3d"


def test_run_cem_square_parser_accepts_acceleration_profile(monkeypatch):
    module = _load_script_module("scripts", "run_cem_16e_square_test.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cem_16e_square_test.py",
            "--acceleration-profile",
            "gpu3d",
        ],
    )

    args = module._parse_args()

    assert args.acceleration_profile == "gpu3d"
