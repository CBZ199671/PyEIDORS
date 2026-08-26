#!/usr/bin/env python3
"""Accuracy-only orchestration for MFEM, FreeFEM, and GetFEM CEM adapters."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from typing import Any

from mpmath import mp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.cem_block_audit import (
    assemble_analytic_blocks,
    build_nonuniform_fixture,
    prepare_fixture,
)
from scripts.benchmarks.cem_multifem_common import (
    PRIMARY_METHODS,
    validate_native_report,
)


ENVIRONMENT_SCHEMA = "cem-multifem-environment-v1"
REPORT_SCHEMA = "cem-multifem-accuracy-v1"
BLOCK_TOLERANCE = 5.0e-12
SIX_METHOD_REPORT_SCHEMA = "cem-six-method-accuracy-v1"
X01_CASE_DIR = (
    ROOT / "output/cem_exact_extension/cases/"
    "X01_range_q0_uniform_sigma_1_8_to_1_8_z_1_adjacent"
)
H0_CASE_DIR = ROOT / "output/cem_continuum_accuracy/cases/C1_baseline/H0"
H0_MESH_DIR = ROOT / "output/cem_continuum_accuracy/mesh_sequence/H0"
BLOCK_AUDIT_REPORT = (
    ROOT / "output/cem_professor_share_package/work/cem_block_audit/results/"
    "cem_block_audit_report.json"
)


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


def _relative_l2(actual: Any, expected: Any) -> float:
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    numerator = float(np.linalg.norm(actual_array - expected_array))
    denominator = max(float(np.linalg.norm(expected_array)), np.finfo(float).tiny)
    return numerator / denominator


def _max_abs(actual: Any, expected: Any) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(actual, dtype=np.float64)
                - np.asarray(expected, dtype=np.float64)
            )
        )
    )


def _exact_fraction_metrics(
    candidate: np.ndarray, fraction_strings: list[list[str]]
) -> dict[str, float]:
    values = np.asarray(candidate, dtype=np.float64)
    if values.shape != (
        len(fraction_strings),
        len(fraction_strings[0]),
    ):
        raise ValueError("exact voltage shape mismatch")
    with mp.workdps(100):
        squared_error = []
        squared_truth = []
        absolute_error = []
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                candidate_fraction = Fraction.from_float(float(values[row, column]))
                candidate_mp = mp.mpf(candidate_fraction.numerator) / mp.mpf(
                    candidate_fraction.denominator
                )
                truth_fraction = Fraction(fraction_strings[row][column])
                truth_mp = mp.mpf(truth_fraction.numerator) / mp.mpf(
                    truth_fraction.denominator
                )
                delta = candidate_mp - truth_mp
                squared_error.append(abs(delta) ** 2)
                squared_truth.append(abs(truth_mp) ** 2)
                absolute_error.append(abs(delta))
        return {
            "reference_relative_l2": float(
                mp.sqrt(mp.fsum(squared_error)) / mp.sqrt(mp.fsum(squared_truth))
            ),
            "reference_max_abs": float(max(absolute_error, default=mp.mpf("0"))),
        }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required accuracy artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _balanced_adjacent_currents(n_electrodes: int) -> np.ndarray:
    currents = np.zeros((n_electrodes, n_electrodes), dtype=np.float64)
    for drive in range(n_electrodes):
        currents[drive, drive] = 1.0
        currents[(drive + 1) % n_electrodes, drive] = -1.0
    return currents


def _validate_existing_report(
    report: dict[str, Any],
    *,
    expected_solver: str,
    mesh_fingerprint: str,
    n_electrodes: int,
) -> None:
    if report.get("solver") != expected_solver:
        raise ValueError(f"expected {expected_solver} report")
    discretization = report.get("discretization", {})
    if discretization.get("mesh_fingerprint") != mesh_fingerprint:
        raise ValueError(f"{expected_solver} did not use the selected common mesh")
    if int(discretization.get("potential_order", -1)) != 1:
        raise ValueError(f"{expected_solver} did not use P1")
    if discretization.get("scalar_dtype", "float64") != "float64":
        raise ValueError(f"{expected_solver} did not use float64")
    if int(report.get("physical_config", {}).get("n_electrodes", n_electrodes)) != int(
        n_electrodes
    ):
        raise ValueError(f"{expected_solver} electrode count mismatch")


def run_native_case(
    solver: str,
    output_dir: Path,
    *,
    mesh_path: Path,
    mesh_fingerprint: str,
    conductivity: float,
    contact_impedance: np.ndarray,
    currents: np.ndarray,
    prefix: Path | None = None,
    mfem_executable: Path | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Run one native general-FEM adapter on an arbitrary common P1 mesh."""

    if solver not in {"MFEM", "FreeFEM", "GetFEM"}:
        raise ValueError(f"unsupported native solver: {solver}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = mesh_path.resolve()
    impedance = np.asarray(contact_impedance, dtype=np.float64)
    current_matrix = np.asarray(currents, dtype=np.float64)
    if impedance.ndim != 1 or current_matrix.ndim != 2:
        raise ValueError("contact impedance and currents must be vector/matrix")
    if current_matrix.shape[0] != impedance.size:
        raise ValueError("current rows must equal electrode count")
    if not np.array_equal(
        np.sum(current_matrix, axis=0), np.zeros(current_matrix.shape[1])
    ):
        raise ValueError("native case currents must be exactly balanced")

    paths = runtime_paths(prefix)
    env = runtime_environment(paths)
    report_path = output_dir / f"{solver}_native_report.json"
    current_csv = output_dir / "currents.csv"
    current_txt = output_dir / "currents.txt"
    impedance_txt = output_dir / "contact_impedance.txt"
    np.savetxt(current_csv, current_matrix, delimiter=",", fmt="%.17g")
    np.savetxt(current_txt, current_matrix, fmt="%.17g")
    np.savetxt(impedance_txt, impedance, fmt="%.17g")

    if solver == "MFEM":
        executable = mfem_executable
        if executable is None:
            executable, _, env = build_mfem_adapter(
                output_dir / "native_build/mfem", prefix=prefix
            )
        command = [
            str(executable),
            str(mesh_path),
            str(report_path),
            mesh_fingerprint,
            f"{conductivity:.17g}",
            ",".join(f"{value:.17g}" for value in impedance),
            str(current_csv),
        ]
    elif solver == "FreeFEM":
        env.update(
            {
                "PYEIDORS_CEM_MESH": str(mesh_path),
                "PYEIDORS_CEM_OUTPUT": str(report_path),
                "PYEIDORS_CEM_FINGERPRINT": mesh_fingerprint,
                "PYEIDORS_CEM_CONDUCTIVITY": f"{conductivity:.17g}",
                "PYEIDORS_CEM_ELECTRODES": str(impedance.size),
                "PYEIDORS_CEM_DRIVES": str(current_matrix.shape[1]),
                "PYEIDORS_CEM_IMPEDANCE": str(impedance_txt),
                "PYEIDORS_CEM_CURRENTS": str(current_txt),
            }
        )
        command = [
            str(paths.freefem),
            "-v",
            "0",
            str(ROOT / "scripts/benchmarks/freefem_cem_robin.edp"),
        ]
    else:
        config_path = output_dir / "GetFEM_config.json"
        _write_json(
            config_path,
            {
                "mesh": str(mesh_path),
                "mesh_fingerprint": mesh_fingerprint,
                "conductivity": float(conductivity),
                "contact_impedance": impedance.tolist(),
                "currents": current_matrix.tolist(),
            },
        )
        command = [
            str(paths.getfem_python),
            str(ROOT / "scripts/benchmarks/getfem_cem_robin.py"),
            str(config_path),
            str(report_path),
        ]

    _run_checked(command, env=env, context=f"{solver} native Robin solve")
    report = _load_json(report_path)
    identity_metrics = validate_native_report(
        report,
        {"mesh_fingerprint": mesh_fingerprint, "currents": current_matrix},
        expected_solver=solver,
    )
    return report, identity_metrics


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


def run_getfem_fixture(
    output_dir: Path, *, prefix: Path | None = None
) -> dict[str, Any]:
    """Run and independently validate GetFEM on the shared nonuniform fixture."""

    output_dir = output_dir.resolve()
    paths = runtime_paths(prefix)
    env = runtime_environment(paths)
    metadata = prepare_fixture(output_dir)
    fixture = build_nonuniform_fixture()
    analytic_blocks, _ = assemble_analytic_blocks(fixture)
    config_path = output_dir / "common_mesh/GetFEM_fixture_config.json"
    report_path = output_dir / "GetFEM_native_report.json"
    _write_json(
        config_path,
        {
            "mesh": str(metadata["common_msh"]),
            "mesh_fingerprint": fixture.mesh_fingerprint,
            "conductivity": fixture.conductivity,
            "contact_impedance": fixture.contact_impedance.tolist(),
            "currents": fixture.currents.tolist(),
        },
    )
    script = ROOT / "scripts/benchmarks/getfem_cem_robin.py"
    command = [
        str(paths.getfem_python),
        str(script),
        str(config_path),
        str(report_path),
    ]
    _run_checked(command, env=env, context="GetFEM native Robin solve")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    fixture_contract = {
        "mesh_fingerprint": fixture.mesh_fingerprint,
        "currents": fixture.currents,
    }
    identity_metrics = validate_native_report(
        report, fixture_contract, expected_solver="GetFEM"
    )
    block_metrics = {
        key: _relative_frobenius(report["blocks"][key], analytic_blocks[key])
        for key in ("K", "B", "C_plus", "D", "A_R")
    }
    failures = {
        key: value for key, value in block_metrics.items() if value > BLOCK_TOLERANCE
    }
    if failures:
        raise RuntimeError(f"GetFEM analytic P1 block comparison failed: {failures}")
    validation = {
        "schema": REPORT_SCHEMA,
        "solver": "GetFEM",
        "mesh_fingerprint": fixture.mesh_fingerprint,
        "native_identity_metrics": identity_metrics,
        "analytic_block_relative_frobenius": block_metrics,
        "all_pass": True,
        "native_report": str(report_path),
    }
    _write_json(output_dir / "GetFEM_validation.json", validation)
    return validation


def _native_method_result(
    report: dict[str, Any],
    *,
    truth_electrode_rhs: np.ndarray,
    eidors_electrode_rhs: np.ndarray,
) -> dict[str, Any]:
    voltage = np.asarray(report["solution"]["electrode_voltage"], dtype=np.float64)
    return {
        "reference_relative_l2": _relative_l2(voltage, truth_electrode_rhs),
        "reference_max_abs": _max_abs(voltage, truth_electrode_rhs),
        "vs_eidors_standard_relative_l2": _relative_l2(voltage, eidors_electrode_rhs),
    }


def _existing_method_result(
    report: dict[str, Any],
    *,
    voltage_key: str,
    truth_electrode_rhs: np.ndarray,
    eidors_electrode_rhs: np.ndarray,
) -> dict[str, Any]:
    voltage = np.asarray(
        report["raw_electrode_voltages"][voltage_key], dtype=np.float64
    )
    return {
        "reference_relative_l2": _relative_l2(voltage, truth_electrode_rhs),
        "reference_max_abs": _max_abs(voltage, truth_electrode_rhs),
        "vs_eidors_standard_relative_l2": _relative_l2(voltage, eidors_electrode_rhs),
    }


def _method_descriptor(solver: str) -> dict[str, str]:
    method = next(item for item in PRIMARY_METHODS if item.solver == solver)
    return {
        "solver": method.solver,
        "formulation": method.formulation,
        "role": method.role,
    }


def _old_block_metrics(block_report: dict[str, Any], solver: str) -> dict[str, float]:
    exported_name = "PyEIDORS" if solver == "PyEIDORS-DOLFINx" else solver
    values = {
        row["quantity"]: float(row["relative_frobenius"])
        for row in block_report["metrics"]
        if row.get("stage") == "blocks"
        and row.get("left") == exported_name
        and row.get("right") == "Analytic"
        and row.get("quantity") in {"K", "B", "C_plus", "D", "A_R"}
    }
    if set(values) != {"K", "B", "C_plus", "D", "A_R"}:
        raise ValueError(f"incomplete analytic block metrics for {solver}")
    return values


def _run_block_layer(
    output_dir: Path,
    *,
    prefix: Path | None,
    mfem_executable: Path,
) -> dict[str, Any]:
    block_report = _load_json(BLOCK_AUDIT_REPORT)
    fixture = build_nonuniform_fixture()
    analytic_blocks, _ = assemble_analytic_blocks(fixture)
    common_dir = output_dir / "common"
    metadata = prepare_fixture(common_dir)
    methods: list[dict[str, Any]] = []
    for solver in ("EIDORS", "PyEIDORS-DOLFINx", "NGSolve"):
        metrics = _old_block_metrics(block_report, solver)
        methods.append(
            {
                **_method_descriptor(solver),
                "analytic_block_relative_frobenius": metrics,
                "analytic_block_max_relative_frobenius": max(metrics.values()),
                "pass": max(metrics.values()) <= BLOCK_TOLERANCE,
            }
        )

    native_fingerprints: dict[str, str] = {}
    native_identity_max: dict[str, float] = {}
    for solver in ("MFEM", "FreeFEM", "GetFEM"):
        report, identity = run_native_case(
            solver,
            output_dir / solver,
            mesh_path=Path(metadata["common_msh"]),
            mesh_fingerprint=fixture.mesh_fingerprint,
            conductivity=fixture.conductivity,
            contact_impedance=fixture.contact_impedance,
            currents=fixture.currents,
            prefix=prefix,
            mfem_executable=mfem_executable,
        )
        metrics = {
            key: _relative_frobenius(report["blocks"][key], analytic_blocks[key])
            for key in ("K", "B", "C_plus", "D", "A_R")
        }
        native_fingerprints[solver] = report["discretization"]["mesh_fingerprint"]
        native_identity_max[solver] = max(identity.values())
        methods.append(
            {
                **_method_descriptor(solver),
                "analytic_block_relative_frobenius": metrics,
                "analytic_block_max_relative_frobenius": max(metrics.values()),
                "native_identity_max": max(identity.values()),
                "pass": max(metrics.values()) <= BLOCK_TOLERANCE,
            }
        )

    fingerprints = {
        solver: fixture.mesh_fingerprint
        for solver in ("EIDORS", "PyEIDORS-DOLFINx", "NGSolve")
    }
    fingerprints.update(native_fingerprints)
    return {
        "layer": "analytic_nonuniform_p1_block_fixture",
        "reference": {
            "type": "independently assembled analytic P1 element blocks",
            "candidate_solver_matrix_used": False,
        },
        "fairness": {
            "mesh_fingerprint": fixture.mesh_fingerprint,
            "mesh_fingerprints_by_method": fingerprints,
            "all_mesh_fingerprints_identical": len(set(fingerprints.values())) == 1,
            "potential_order_by_method": {
                method.solver: 1 for method in PRIMARY_METHODS
            },
            "all_methods_p1": True,
            "scalar_dtype_by_method": {
                method.solver: "float64" for method in PRIMARY_METHODS
            },
            "identical_conductivity_contact_impedance_and_currents": True,
        },
        "methods": methods,
        "native_identity_max_by_method": native_identity_max,
        "all_pass": all(item["pass"] for item in methods),
    }


def _run_x01_layer(
    output_dir: Path,
    *,
    prefix: Path | None,
    mfem_executable: Path,
) -> dict[str, Any]:
    mesh_dir = X01_CASE_DIR / "common_mesh"
    metadata = _load_json(mesh_dir / "cem_exact_extension_p1.json")
    mesh_fingerprint = str(metadata["mesh_fingerprint"])
    n_electrodes = int(metadata["n_electrodes"])
    currents = np.asarray(metadata["current_patterns"], dtype=np.float64)
    conductivity = float(metadata["material_conductivities"][0])
    impedance = np.full(
        n_electrodes, float(metadata["contact_impedance"]), dtype=np.float64
    )
    existing_reports = {
        "EIDORS": _load_json(X01_CASE_DIR / "eidors_report.json"),
        "PyEIDORS-DOLFINx": _load_json(X01_CASE_DIR / "pyeidors_report.json"),
        "NGSolve": _load_json(X01_CASE_DIR / "ngsolve_report.json"),
    }
    for solver, report in existing_reports.items():
        expected = "PyEIDORS/DOLFINx" if solver == "PyEIDORS-DOLFINx" else solver
        _validate_existing_report(
            report,
            expected_solver=expected,
            mesh_fingerprint=mesh_fingerprint,
            n_electrodes=n_electrodes,
        )

    exact_accuracy = _load_json(
        ROOT / "output/cem_exact_extension/cem_exact_extension_accuracy.json"
    )
    exact_certificate = exact_accuracy["truth"]["X01"]
    truth_fractions = exact_certificate["electrode_voltage_fractions"]
    eidors_voltage = np.asarray(
        existing_reports["EIDORS"]["raw_electrode_voltages"]["classic"],
        dtype=np.float64,
    )
    voltage_key = {
        "EIDORS": "classic",
        "PyEIDORS-DOLFINx": "robin_transconductance",
        "NGSolve": "robin_transconductance",
    }
    methods = []
    fingerprints: dict[str, str] = {}
    for solver, report in existing_reports.items():
        voltage = np.asarray(
            report["raw_electrode_voltages"][voltage_key[solver]], dtype=np.float64
        )
        methods.append(
            {
                **_method_descriptor(solver),
                **_exact_fraction_metrics(voltage, truth_fractions),
                "vs_eidors_standard_relative_l2": _relative_l2(voltage, eidors_voltage),
            }
        )
        fingerprints[solver] = report["discretization"]["mesh_fingerprint"]

    native_identity_max: dict[str, float] = {}
    for solver in ("MFEM", "FreeFEM", "GetFEM"):
        report, identity = run_native_case(
            solver,
            output_dir / solver,
            mesh_path=mesh_dir / "cem_exact_extension_p1.msh",
            mesh_fingerprint=mesh_fingerprint,
            conductivity=conductivity,
            contact_impedance=impedance,
            currents=currents,
            prefix=prefix,
            mfem_executable=mfem_executable,
        )
        voltage = np.asarray(report["solution"]["electrode_voltage"], dtype=np.float64)
        methods.append(
            {
                **_method_descriptor(solver),
                **_exact_fraction_metrics(voltage, truth_fractions),
                "vs_eidors_standard_relative_l2": _relative_l2(voltage, eidors_voltage),
            }
        )
        fingerprints[solver] = report["discretization"]["mesh_fingerprint"]
        native_identity_max[solver] = max(identity.values())

    for item in methods:
        item["pass"] = item["reference_relative_l2"] <= BLOCK_TOLERANCE
    return {
        "layer": "X01_exact_finite_dimensional_QQ",
        "reference": {
            "type": "exact rational discrete P1 CEM solution over QQ",
            "solver": exact_certificate["exact_linear_solver"],
            "truth_sha256": exact_certificate["truth_sha256"],
            "exact_classic_residual_zero": exact_certificate[
                "exact_classic_residual_zero"
            ],
            "exact_robin_residual_zero": exact_certificate["exact_robin_residual_zero"],
            "candidate_solver_matrix_used": False,
        },
        "fairness": {
            "mesh_fingerprint": mesh_fingerprint,
            "mesh_fingerprints_by_method": fingerprints,
            "all_mesh_fingerprints_identical": len(set(fingerprints.values())) == 1,
            "potential_order_by_method": {
                method.solver: 1 for method in PRIMARY_METHODS
            },
            "all_methods_p1": True,
            "scalar_dtype_by_method": {
                method.solver: "float64" for method in PRIMARY_METHODS
            },
            "identical_conductivity_contact_impedance_and_currents": True,
        },
        "methods": methods,
        "native_identity_max_by_method": native_identity_max,
        "all_pass": all(item["pass"] for item in methods),
    }


def _run_h0_layer(
    output_dir: Path,
    *,
    prefix: Path | None,
    mfem_executable: Path,
) -> dict[str, Any]:
    metadata = _load_json(H0_MESH_DIR / "cem_continuum_common_p1.json")
    suite = _load_json(ROOT / "output/cem_continuum_accuracy/suite_manifest.json")
    case = next(item for item in suite["cases"] if item["case_id"] == "C1")
    reference = _load_json(ROOT / "output/cem_continuum_accuracy/references/C1.json")
    mesh_fingerprint = str(metadata["mesh_fingerprint"])
    n_electrodes = int(metadata["n_electrodes"])
    currents = _balanced_adjacent_currents(n_electrodes)
    impedance = np.full(
        n_electrodes, float(case["contact_impedance"]), dtype=np.float64
    )
    truth = np.asarray(reference["reference_voltages"], dtype=np.float64)
    existing_reports = {
        "EIDORS": _load_json(H0_CASE_DIR / "eidors_report.json"),
        "PyEIDORS-DOLFINx": _load_json(H0_CASE_DIR / "pyeidors_report.json"),
        "NGSolve": _load_json(H0_CASE_DIR / "ngsolve_report.json"),
    }
    for solver, report in existing_reports.items():
        expected = "PyEIDORS/DOLFINx" if solver == "PyEIDORS-DOLFINx" else solver
        _validate_existing_report(
            report,
            expected_solver=expected,
            mesh_fingerprint=mesh_fingerprint,
            n_electrodes=n_electrodes,
        )
    eidors_voltage = np.asarray(
        existing_reports["EIDORS"]["raw_electrode_voltages"]["classic"],
        dtype=np.float64,
    )
    voltage_key = {
        "EIDORS": "classic",
        "PyEIDORS-DOLFINx": "robin_transconductance",
        "NGSolve": "robin_transconductance",
    }
    methods = []
    fingerprints: dict[str, str] = {}
    for solver, report in existing_reports.items():
        methods.append(
            {
                **_method_descriptor(solver),
                **_existing_method_result(
                    report,
                    voltage_key=voltage_key[solver],
                    truth_electrode_rhs=truth,
                    eidors_electrode_rhs=eidors_voltage,
                ),
            }
        )
        fingerprints[solver] = report["discretization"]["mesh_fingerprint"]

    native_identity_max: dict[str, float] = {}
    for solver in ("MFEM", "FreeFEM", "GetFEM"):
        report, identity = run_native_case(
            solver,
            output_dir / solver,
            mesh_path=H0_MESH_DIR / "cem_continuum_common_p1.msh",
            mesh_fingerprint=mesh_fingerprint,
            conductivity=float(case["conductivity"]),
            contact_impedance=impedance,
            currents=currents,
            prefix=prefix,
            mfem_executable=mfem_executable,
        )
        methods.append(
            {
                **_method_descriptor(solver),
                **_native_method_result(
                    report,
                    truth_electrode_rhs=truth,
                    eidors_electrode_rhs=eidors_voltage,
                ),
            }
        )
        fingerprints[solver] = report["discretization"]["mesh_fingerprint"]
        native_identity_max[solver] = max(identity.values())

    finite = all(
        np.isfinite(item["reference_relative_l2"])
        and np.isfinite(item["reference_max_abs"])
        for item in methods
    )
    return {
        "layer": "C1_H0_independent_disk_continuum",
        "reference": {
            "type": reference["method"],
            "certified": bool(reference["certified"]),
            "uses_interior_fem_mesh": bool(reference["uses_interior_fem_mesh"]),
            "uses_candidate_solver_matrix": bool(
                reference["uses_candidate_solver_matrix"]
            ),
            "relative_uncertainty": float(reference["reference_relative_uncertainty"]),
        },
        "fairness": {
            "mesh_fingerprint": mesh_fingerprint,
            "mesh_fingerprints_by_method": fingerprints,
            "all_mesh_fingerprints_identical": len(set(fingerprints.values())) == 1,
            "potential_order_by_method": {
                method.solver: 1 for method in PRIMARY_METHODS
            },
            "all_methods_p1": True,
            "scalar_dtype_by_method": {
                method.solver: "float64" for method in PRIMARY_METHODS
            },
            "identical_conductivity_contact_impedance_and_currents": True,
        },
        "methods": methods,
        "native_identity_max_by_method": native_identity_max,
        "all_pass": bool(reference["certified"]) and finite,
    }


def _accuracy_only_guard(value: Any) -> None:
    forbidden = {"timing", "elapsed_seconds", "runtime_seconds", "speedup"}
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in forbidden:
                raise ValueError(
                    f"accuracy-only report contains forbidden field: {key}"
                )
            _accuracy_only_guard(item)
    elif isinstance(value, list):
        for item in value:
            _accuracy_only_guard(item)


def _render_six_method_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Six-method CEM accuracy comparison",
        "",
        "All methods use the identical imported straight-sided mesh, P1 potential "
        "space, float64 scalars, conductivity, contact impedance, electrode tags, "
        "and current patterns within each layer.",
        "",
        "EIDORS classic augmented CEM is the standard implementation comparator. "
        "The numerical truth is instead the independent analytic block reference, "
        "the exact QQ solution, or the independent disk continuum reference, as "
        "appropriate.",
        "",
    ]
    for layer in report["layers"]:
        lines.extend(
            [
                f"## {layer['layer']}",
                "",
                "| Method | Formulation | Reference relative error | Reference max abs | Versus EIDORS |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for method in layer["methods"]:
            relative = method.get(
                "reference_relative_l2",
                method.get("analytic_block_max_relative_frobenius"),
            )
            max_abs = method.get("reference_max_abs")
            vs_eidors = method.get("vs_eidors_standard_relative_l2")
            lines.append(
                "| {solver} | {formulation} | {relative:.6e} | {maximum} | {eidors} |".format(
                    solver=method["solver"],
                    formulation=method["formulation"],
                    relative=float(relative),
                    maximum="—" if max_abs is None else f"{float(max_abs):.6e}",
                    eidors="—" if vs_eidors is None else f"{float(vs_eidors):.6e}",
                )
            )
        lines.extend(["", f"Layer gate: {'PASS' if layer['all_pass'] else 'FAIL'}", ""])
    lines.extend(
        [
            "## Interpretation",
            "",
            "The QQ layer isolates floating-point implementation error on one exact "
            "finite-dimensional P1 problem. The H0 layer measures discretization "
            "error against a continuum disk solution; therefore its error is much "
            "larger even when all FEM implementations agree closely.",
            "",
        ]
    )
    return "\n".join(lines)


def run_six_method_accuracy(
    output_dir: Path, *, prefix: Path | None = None
) -> dict[str, Any]:
    """Execute and validate the registered six-method accuracy-only comparison."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    executable, _, _ = build_mfem_adapter(
        output_dir / "native_build/mfem", prefix=prefix
    )
    layers = [
        _run_block_layer(
            output_dir / "block_fixture",
            prefix=prefix,
            mfem_executable=executable,
        ),
        _run_x01_layer(
            output_dir / "X01",
            prefix=prefix,
            mfem_executable=executable,
        ),
        _run_h0_layer(
            output_dir / "H0",
            prefix=prefix,
            mfem_executable=executable,
        ),
    ]
    registered_methods = [asdict(method) for method in PRIMARY_METHODS]
    expected_names = [method.solver for method in PRIMARY_METHODS]
    for layer in layers:
        names = [item["solver"] for item in layer["methods"]]
        if names != expected_names:
            raise ValueError(
                f"six-method set/order mismatch in {layer['layer']}: {names}"
            )
        fairness = layer["fairness"]
        if not fairness["all_mesh_fingerprints_identical"]:
            raise ValueError(f"mesh mismatch in {layer['layer']}")
        if not fairness["all_methods_p1"]:
            raise ValueError(f"non-P1 method in {layer['layer']}")
    report = {
        "schema": SIX_METHOD_REPORT_SCHEMA,
        "scope": "accuracy only",
        "registered_methods": registered_methods,
        "reference_policy": {
            "standard_implementation_comparator": "EIDORS classic augmented CEM",
            "discrete_truth": "exact rational QQ solution",
            "continuum_truth": "independent disk NtD Fourier-Nystrom reference",
            "eidors_is_numerical_truth": False,
        },
        "layers": layers,
        "all_pass": all(layer["all_pass"] for layer in layers),
    }
    _accuracy_only_guard(report)
    _write_json(output_dir / "six_method_accuracy.json", report)
    (output_dir / "six_method_accuracy_report.md").write_text(
        _render_six_method_markdown(report), encoding="utf-8"
    )
    return report


def _doctor(args: argparse.Namespace) -> int:
    report = build_environment_report(args.prefix)
    _write_json(args.output_json, report)
    return 0 if report["ok"] or not args.strict else 1


def _run(args: argparse.Namespace) -> int:
    if args.solver == "MFEM":
        report = run_mfem_fixture(args.output_dir, prefix=args.prefix)
    elif args.solver == "FreeFEM":
        report = run_freefem_fixture(args.output_dir, prefix=args.prefix)
    elif args.solver == "GetFEM":
        report = run_getfem_fixture(args.output_dir, prefix=args.prefix)
    else:
        raise NotImplementedError(f"{args.solver} adapter is not implemented yet")
    _write_json(None, report)
    return 0


def _run_six(args: argparse.Namespace) -> int:
    report = run_six_method_accuracy(args.output_dir, prefix=args.prefix)
    _write_json(None, report)
    return 0 if report["all_pass"] else 1


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
    run_six = subparsers.add_parser(
        "run-six", help="run the six-method block, QQ, and continuum accuracy gates"
    )
    run_six.add_argument("--prefix", type=Path)
    run_six.add_argument("--output-dir", type=Path, required=True)
    run_six.set_defaults(handler=_run_six)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
