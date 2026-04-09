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


def test_benchmark_3d_fair_compare_forwards_acceleration_profile(monkeypatch, tmp_path: Path):
    module = _load_script_module("scripts", "benchmarks", "benchmark_3d_fair_compare.py")
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
    module = _load_script_module("scripts", "benchmarks", "benchmark_difference_runtime.py")
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


def test_profile_reconstruction_pipeline_parser_accepts_acceleration_profile(monkeypatch):
    module = _load_script_module("scripts", "benchmarks", "profile_reconstruction_pipeline.py")
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


def test_run_synthetic_parity_forwards_acceleration_profile(monkeypatch, tmp_path: Path):
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
