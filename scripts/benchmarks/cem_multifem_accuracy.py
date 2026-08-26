#!/usr/bin/env python3
"""Accuracy-only orchestration for MFEM, FreeFEM, and GetFEM CEM adapters."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.cem_block_audit import (
    assemble_analytic_blocks,
    build_nonuniform_fixture,
    prepare_fixture,
)
from scripts.benchmarks.cem_multifem_common import validate_native_report


ENVIRONMENT_SCHEMA = "cem-multifem-environment-v1"
REPORT_SCHEMA = "cem-multifem-accuracy-v1"
BLOCK_TOLERANCE = 5.0e-12


def default_environment_prefix() -> Path:
    """Return the isolated multi-FEM runtime prefix."""

    override = os.environ.get("PYEIDORS_CEM_MULTIFEM_PREFIX")
    if override:
        return Path(override).expanduser().resolve()
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    return (data_home / "pyeidors-cem-multifem").resolve()


@dataclass(frozen=True)
class RuntimePaths:
    prefix: Path
    mfem_prefix: Path
    deb_root: Path
    freefem: Path
    getfem_python: Path
    getfem_pythonpath: Path
    mfem_library: Path


def runtime_paths(prefix: Path | None = None) -> RuntimePaths:
    resolved = (prefix or default_environment_prefix()).resolve()
    mfem_prefix = resolved / "mfem-4.9"
    deb_root = resolved / "ubuntu-jammy"
    return RuntimePaths(
        prefix=resolved,
        mfem_prefix=mfem_prefix,
        deb_root=deb_root,
        freefem=deb_root / "usr/bin/FreeFem++-nw",
        getfem_python=Path("/usr/bin/python3"),
        getfem_pythonpath=deb_root / "usr/lib/python3/dist-packages",
        mfem_library=mfem_prefix / "lib/libmfem.so",
    )


def runtime_environment(paths: RuntimePaths) -> dict[str, str]:
    """Build a subprocess environment without mutating the caller."""

    env = dict(os.environ)
    for key in (
        "LD_PRELOAD",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "VIRTUAL_ENV",
    ):
        env.pop(key, None)
    for key in tuple(env):
        if key == "CONDA_PREFIX" or key.startswith("CONDA_"):
            env.pop(key, None)
    path_entries = [
        str(paths.mfem_prefix / "bin"),
        str(paths.deb_root / "usr/bin"),
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    ]
    python_entries = [str(paths.getfem_pythonpath)]
    library_entries = [
        str(paths.mfem_prefix / "lib"),
        str(paths.deb_root / "usr/lib/x86_64-linux-gnu"),
        str(paths.deb_root / "usr/lib/freefem++"),
    ]
    env["PATH"] = os.pathsep.join(item for item in path_entries if item)
    env["PYTHONPATH"] = os.pathsep.join(item for item in python_entries if item)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(item for item in library_entries if item)
    env["FF_LOADPATH"] = os.pathsep.join(
        item
        for item in [
            str(paths.deb_root / "usr/lib/freefem++"),
            env.get("FF_LOADPATH", ""),
        ]
        if item
    )
    return env


def _command_version(command: list[str], env: dict[str, str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=env,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc), "command": command}
    output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "output": output,
        "command": command,
    }


def _metadata(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, value = line.split("\t", maxsplit=1)
        result[key] = value
    return result


def _freefem_probe(paths: RuntimePaths, env: dict[str, str]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="cem-freefem-doctor-") as directory:
        script = Path(directory) / "doctor.edp"
        script.write_text(
            'load "gmsh"\ncout << "freefem-gmsh-ok" << endl;\n',
            encoding="utf-8",
        )
        return _command_version(
            [str(paths.freefem), "-v", "0", str(script)],
            env,
        )


def build_environment_report(prefix: Path | None = None) -> dict[str, Any]:
    paths = runtime_paths(prefix)
    env = runtime_environment(paths)
    metadata = _metadata(paths.prefix / "environment.tsv")
    freefem_probe = _freefem_probe(paths, env)
    getfem_probe = _command_version(
        [
            str(paths.getfem_python),
            "-c",
            ("import getfem; print(getattr(getfem, '__version__', 'import-ok'))"),
        ],
        env,
    )
    gmsh_probe = _command_version(["gmsh", "--version"], env)
    compiler_probe = _command_version(["g++", "--version"], env)
    mfem_header = paths.mfem_prefix / "lib/cmake/mfem/MFEMConfigVersion.cmake"
    mfem_version = None
    if mfem_header.is_file():
        match = re.search(
            r"set\(PACKAGE_VERSION\s+\"([^\"]+)\"\)",
            mfem_header.read_text(encoding="utf-8", errors="replace"),
        )
        if match:
            mfem_version = match.group(1)
    checks = {
        "metadata_schema": metadata.get("schema") == ENVIRONMENT_SCHEMA,
        "mfem_library": paths.mfem_library.is_file(),
        "mfem_version": mfem_version == "4.9.0",
        "freefem": bool(freefem_probe["ok"]),
        "getfem": bool(getfem_probe["ok"]),
        "gmsh": bool(gmsh_probe["ok"]),
        "compiler": bool(compiler_probe["ok"]),
    }
    return {
        "schema": ENVIRONMENT_SCHEMA,
        "ok": all(checks.values()),
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "paths": {key: str(value) for key, value in asdict(paths).items()},
        "metadata": metadata,
        "checks": checks,
        "probes": {
            "mfem_version": mfem_version,
            "freefem": freefem_probe,
            "getfem": getfem_probe,
            "gmsh": gmsh_probe,
            "compiler": compiler_probe,
        },
    }


def _write_json(path: Path | None, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path is None:
        sys.stdout.write(rendered)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _run_checked(
    command: list[str], *, env: dict[str, str], context: str
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        detail = "\n".join(
            part.strip()
            for part in (completed.stdout, completed.stderr)
            if part.strip()
        )
        raise RuntimeError(
            f"{context} failed with code {completed.returncode}: {detail}"
        )
    return completed


def _relative_frobenius(actual: Any, expected: Any) -> float:
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    numerator = float(np.linalg.norm(actual_array - expected_array, ord="fro"))
    denominator = max(
        float(np.linalg.norm(expected_array, ord="fro")), np.finfo(float).tiny
    )
    return numerator / denominator


def build_mfem_adapter(
    build_dir: Path, *, prefix: Path | None = None
) -> tuple[Path, RuntimePaths, dict[str, str]]:
    """Compile the native MFEM adapter against the isolated runtime."""

    paths = runtime_paths(prefix)
    env = runtime_environment(paths)
    source = ROOT / "scripts/benchmarks/mfem_cem_robin.cpp"
    executable = build_dir / "mfem_cem_robin"
    build_dir.mkdir(parents=True, exist_ok=True)
    cmake_source = build_dir / "cmake_source"
    cmake_build = build_dir / "cmake_build"
    cmake_source.mkdir(parents=True, exist_ok=True)
    (cmake_source / "CMakeLists.txt").write_text(
        "\n".join(
            (
                "cmake_minimum_required(VERSION 3.22)",
                "project(mfem_cem_robin LANGUAGES CXX)",
                "find_package(MFEM 4.9.0 EXACT CONFIG REQUIRED)",
                f'add_executable(mfem_cem_robin "{source}")',
                "target_compile_features(mfem_cem_robin PRIVATE cxx_std_17)",
                "target_compile_options(mfem_cem_robin PRIVATE -Wall -Wextra)",
                "target_link_libraries(mfem_cem_robin PRIVATE mfem)",
                "",
            )
        ),
        encoding="utf-8",
    )
    _run_checked(
        [
            "cmake",
            "-S",
            str(cmake_source),
            "-B",
            str(cmake_build),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DMFEM_DIR={paths.mfem_prefix / 'lib/cmake/mfem'}",
        ],
        env=env,
        context="MFEM adapter CMake configuration",
    )
    _run_checked(
        ["cmake", "--build", str(cmake_build), "--parallel", "2"],
        env=env,
        context="MFEM adapter compilation",
    )
    built_executable = cmake_build / "mfem_cem_robin"
    if not built_executable.is_file():
        raise RuntimeError("MFEM adapter build did not produce the expected executable")
    executable.write_bytes(built_executable.read_bytes())
    executable.chmod(0o755)
    return executable, paths, env


def run_mfem_fixture(output_dir: Path, *, prefix: Path | None = None) -> dict[str, Any]:
    """Run and independently validate MFEM on the shared nonuniform P1 fixture."""

    output_dir = output_dir.resolve()
    metadata = prepare_fixture(output_dir)
    fixture = build_nonuniform_fixture()
    analytic_blocks, _ = assemble_analytic_blocks(fixture)
    current_path = output_dir / "common_mesh/cem_block_audit_currents.csv"
    np.savetxt(current_path, fixture.currents, delimiter=",", fmt="%.17g")
    report_path = output_dir / "MFEM_native_report.json"
    executable, _, env = build_mfem_adapter(
        output_dir / "native_build/mfem", prefix=prefix
    )
    command = [
        str(executable),
        str(metadata["common_msh"]),
        str(report_path),
        fixture.mesh_fingerprint,
        f"{fixture.conductivity:.17g}",
        ",".join(f"{value:.17g}" for value in fixture.contact_impedance),
        str(current_path),
    ]
    _run_checked(command, env=env, context="MFEM native Robin solve")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    fixture_contract = {
        "mesh_fingerprint": fixture.mesh_fingerprint,
        "currents": fixture.currents,
    }
    identity_metrics = validate_native_report(
        report, fixture_contract, expected_solver="MFEM"
    )
    block_metrics = {
        key: _relative_frobenius(report["blocks"][key], analytic_blocks[key])
        for key in ("K", "B", "C_plus", "D", "A_R")
    }
    failures = {
        key: value for key, value in block_metrics.items() if value > BLOCK_TOLERANCE
    }
    if failures:
        raise RuntimeError(f"MFEM analytic P1 block comparison failed: {failures}")
    validation = {
        "schema": REPORT_SCHEMA,
        "solver": "MFEM",
        "mesh_fingerprint": fixture.mesh_fingerprint,
        "native_identity_metrics": identity_metrics,
        "analytic_block_relative_frobenius": block_metrics,
        "all_pass": True,
        "native_report": str(report_path),
    }
    _write_json(output_dir / "MFEM_validation.json", validation)
    return validation


def run_freefem_fixture(
    output_dir: Path, *, prefix: Path | None = None
) -> dict[str, Any]:
    """Run and independently validate FreeFEM on the shared nonuniform fixture."""

    output_dir = output_dir.resolve()
    paths = runtime_paths(prefix)
    env = runtime_environment(paths)
    metadata = prepare_fixture(output_dir)
    fixture = build_nonuniform_fixture()
    analytic_blocks, _ = assemble_analytic_blocks(fixture)
    current_path = output_dir / "common_mesh/cem_block_audit_currents.txt"
    impedance_path = output_dir / "common_mesh/cem_block_audit_impedance.txt"
    np.savetxt(current_path, fixture.currents, fmt="%.17g")
    np.savetxt(impedance_path, fixture.contact_impedance, fmt="%.17g")
    report_path = output_dir / "FreeFEM_native_report.json"
    script = ROOT / "scripts/benchmarks/freefem_cem_robin.edp"
    env.update(
        {
            "PYEIDORS_CEM_MESH": str(metadata["common_msh"]),
            "PYEIDORS_CEM_OUTPUT": str(report_path),
            "PYEIDORS_CEM_FINGERPRINT": fixture.mesh_fingerprint,
            "PYEIDORS_CEM_CONDUCTIVITY": f"{fixture.conductivity:.17g}",
            "PYEIDORS_CEM_ELECTRODES": str(fixture.contact_impedance.size),
            "PYEIDORS_CEM_DRIVES": str(fixture.currents.shape[1]),
            "PYEIDORS_CEM_IMPEDANCE": str(impedance_path),
            "PYEIDORS_CEM_CURRENTS": str(current_path),
        }
    )
    command = [
        str(paths.freefem),
        "-v",
        "0",
        str(script),
    ]
    _run_checked(command, env=env, context="FreeFEM native Robin solve")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    fixture_contract = {
        "mesh_fingerprint": fixture.mesh_fingerprint,
        "currents": fixture.currents,
    }
    identity_metrics = validate_native_report(
        report, fixture_contract, expected_solver="FreeFEM"
    )
    block_metrics = {
        key: _relative_frobenius(report["blocks"][key], analytic_blocks[key])
        for key in ("K", "B", "C_plus", "D", "A_R")
    }
    failures = {
        key: value for key, value in block_metrics.items() if value > BLOCK_TOLERANCE
    }
    if failures:
        raise RuntimeError(f"FreeFEM analytic P1 block comparison failed: {failures}")
    validation = {
        "schema": REPORT_SCHEMA,
        "solver": "FreeFEM",
        "mesh_fingerprint": fixture.mesh_fingerprint,
        "native_identity_metrics": identity_metrics,
        "analytic_block_relative_frobenius": block_metrics,
        "all_pass": True,
        "native_report": str(report_path),
    }
    _write_json(output_dir / "FreeFEM_validation.json", validation)
    return validation


def _doctor(args: argparse.Namespace) -> int:
    report = build_environment_report(args.prefix)
    _write_json(args.output_json, report)
    return 0 if report["ok"] or not args.strict else 1


def _run(args: argparse.Namespace) -> int:
    if args.solver == "MFEM":
        report = run_mfem_fixture(args.output_dir, prefix=args.prefix)
    elif args.solver == "FreeFEM":
        report = run_freefem_fixture(args.output_dir, prefix=args.prefix)
    else:
        raise NotImplementedError(f"{args.solver} adapter is not implemented yet")
    _write_json(None, report)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="verify isolated FEM runtimes")
    doctor.add_argument("--prefix", type=Path)
    doctor.add_argument("--output-json", type=Path)
    doctor.add_argument("--strict", action="store_true")
    doctor.set_defaults(handler=_doctor)
    run = subparsers.add_parser("run", help="run one native solver on the P1 fixture")
    run.add_argument("--solver", choices=("MFEM", "FreeFEM", "GetFEM"), required=True)
    run.add_argument("--prefix", type=Path)
    run.add_argument("--output-dir", type=Path, required=True)
    run.set_defaults(handler=_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
