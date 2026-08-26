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


ENVIRONMENT_SCHEMA = "cem-multifem-environment-v1"
REPORT_SCHEMA = "cem-multifem-accuracy-v1"


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
        script.write_text('cout << "freefem-ok" << endl;\n', encoding="utf-8")
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


def _doctor(args: argparse.Namespace) -> int:
    report = build_environment_report(args.prefix)
    _write_json(args.output_json, report)
    return 0 if report["ok"] or not args.strict else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="verify isolated FEM runtimes")
    doctor.add_argument("--prefix", type=Path)
    doctor.add_argument("--output-json", type=Path)
    doctor.add_argument("--strict", action="store_true")
    doctor.set_defaults(handler=_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
