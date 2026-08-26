"""Fair multi-FEM Robin CEM experiment guardrails."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

from scripts.benchmarks.cem_multifem_accuracy import (
    ENVIRONMENT_SCHEMA,
    build_environment_report,
    runtime_environment,
    runtime_paths,
)


def test_v804_runtime_paths_are_isolated_and_deterministic(tmp_path: Path) -> None:
    paths = runtime_paths(tmp_path)
    assert paths.prefix == tmp_path.resolve()
    assert paths.mfem_prefix == tmp_path.resolve() / "mfem-4.9"
    assert paths.freefem == (tmp_path.resolve() / "ubuntu-jammy/usr/bin/FreeFem++-nw")
    assert paths.getfem_pythonpath == (
        tmp_path.resolve() / "ubuntu-jammy/usr/lib/python3/dist-packages"
    )

    env = runtime_environment(paths)
    assert env is not os.environ
    assert env["PATH"].split(os.pathsep)[0] == str(paths.mfem_prefix / "bin")
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(paths.getfem_pythonpath)
    assert str(paths.deb_root / "usr/lib/freefem++") in env["FF_LOADPATH"]


def test_v804_doctor_fails_closed_when_prefix_is_missing(tmp_path: Path) -> None:
    report = build_environment_report(tmp_path / "missing")
    assert report["schema"] == ENVIRONMENT_SCHEMA
    assert report["ok"] is False
    assert report["checks"]["metadata_schema"] is False
    assert report["checks"]["mfem_library"] is False
    assert report["checks"]["freefem"] is False
    assert report["checks"]["getfem"] is False


def test_v807_setup_rejects_invalid_mfem_build_jobs(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts/benchmarks/setup_cem_multifem_env.sh"
    env = dict(os.environ)
    env["PYEIDORS_CEM_MULTIFEM_PREFIX"] = str(tmp_path / "runtime")
    env["PYEIDORS_CEM_MFEM_BUILD_JOBS"] = "0"
    completed = subprocess.run(
        [str(script), "install"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 2
    assert "must be a positive integer" in completed.stderr
    assert not (tmp_path / "runtime").exists()


def test_v808_runtime_environment_drops_nix_and_python_contamination(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/nix/store/fake-lib")
    monkeypatch.setenv("LD_PRELOAD", "/nix/store/fake-preload.so")
    monkeypatch.setenv("PYTHONPATH", "/nix/store/fake-python")
    monkeypatch.setenv("PYTHONHOME", "/nix/store/fake-home")
    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/fake-venv")
    monkeypatch.setenv("CONDA_PREFIX", "/tmp/fake-conda")
    env = runtime_environment(runtime_paths(tmp_path))
    assert "/nix/store" not in env["PATH"]
    assert "/nix/store" not in env["LD_LIBRARY_PATH"]
    assert "/nix/store" not in env["PYTHONPATH"]
    assert "LD_PRELOAD" not in env
    assert "PYTHONHOME" not in env
    assert "VIRTUAL_ENV" not in env
    assert "CONDA_PREFIX" not in env
