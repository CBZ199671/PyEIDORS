"""PyEIDORS backend runtime doctor.

This command is intended for external GUI hosts.  It checks the packaged
worker runtime without entering a development shell, and reports enough
environment detail to make field installation failures actionable.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any


CUDA_TOOLKIT_VERSION = "12.8.1"
CUDA_12_8_MIN_LINUX_DRIVER = "570.124.06"
CUDA_12_8_MIN_WINDOWS_DRIVER = "572.61"
CUDA_DRIVER_REQUIREMENT_SOURCE = (
    "https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html"
)
DEFAULT_PROFILE = "complex64"
DEFAULT_WORKER_COMMAND = "eit-backend-worker"
PROFILE_PACKAGE_ATTRS = {
    "default": "pyeidors",
    "complex": "pyeidors-complex",
    "complex64": "pyeidors-complex64",
    "cuda": "pyeidors-cuda",
    "cuda-amgx": "pyeidors-cuda-amgx",
    "complex-cuda": "pyeidors-complex-cuda",
    "complex-cuda-amgx": "pyeidors-complex-cuda-amgx",
    "complex64-cuda": "pyeidors-complex64-cuda",
    "cuda-sm61": "pyeidors-cuda-sm61",
    "complex64-cuda-sm61": "pyeidors-complex64-cuda-sm61",
}


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", value))


def driver_meets_requirement(driver_version: str, required: str) -> bool:
    """Return true when driver_version is >= required."""

    driver = _version_tuple(driver_version)
    minimum = _version_tuple(required)
    length = max(len(driver), len(minimum))
    driver = driver + (0,) * (length - len(driver))
    minimum = minimum + (0,) * (length - len(minimum))
    return driver >= minimum


def _check(
    checks: list[dict[str, Any]], check_id: str, status: str, message: str, **extra: Any
) -> None:
    item: dict[str, Any] = {
        "id": check_id,
        "status": status,
        "message": message,
    }
    item.update(extra)
    checks.append(item)


def _run_command(
    args: list[str], timeout: float = 10.0, input_text: str | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def _read_project_version() -> str:
    try:
        return importlib.metadata.version("pyeidors")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _import_check(checks: list[dict[str, Any]]) -> None:
    modules = [
        "pyeidors",
        "eit_app",
        "eit_app.backend_worker",
        "eit_app.backend_worker_protocol",
        "eit_app.controllers.reconstruction_controller",
        "dolfinx",
        "petsc4py",
        "h5py",
        "numpy",
        "scipy",
    ]
    imported: list[str] = []
    errors: dict[str, str] = {}
    for name in modules:
        try:
            importlib.import_module(name)
            imported.append(name)
        except Exception as exc:  # pragma: no cover - depends on packaged runtime
            errors[name] = str(exc)
    if errors:
        _check(
            checks,
            "python-imports",
            "error",
            "required runtime imports failed",
            imported=imported,
            errors=errors,
        )
    else:
        _check(
            checks,
            "python-imports",
            "ok",
            "required runtime imports succeeded",
            imported=imported,
        )


def _worker_check(
    checks: list[dict[str, Any]], worker_command: str, run_protocol_smoke: bool
) -> None:
    worker_path = shutil.which(worker_command)
    if not worker_path:
        _check(
            checks, "worker-command", "error", f"{worker_command!r} not found on PATH"
        )
        return
    _check(checks, "worker-command", "ok", "worker command found", path=worker_path)

    help_result = _run_command([worker_path, "--help"], timeout=10.0)
    if help_result.returncode != 0:
        _check(
            checks,
            "worker-help",
            "error",
            "worker --help failed",
            returncode=help_result.returncode,
            stderr=help_result.stderr[-2000:],
        )
        return
    if "serve" not in help_result.stdout:
        _check(
            checks,
            "worker-help",
            "error",
            "worker help does not advertise serve command",
        )
        return
    _check(checks, "worker-help", "ok", "worker help advertises persistent serve mode")

    if not run_protocol_smoke:
        _check(checks, "worker-protocol-smoke", "skip", "protocol smoke disabled")
        return

    smoke = _run_command(
        [worker_path, "serve"],
        timeout=20.0,
        input_text='{"id":"doctor-smoke","command":"shutdown"}\n',
    )
    if smoke.returncode != 0:
        _check(
            checks,
            "worker-protocol-smoke",
            "error",
            "worker serve shutdown smoke failed",
            returncode=smoke.returncode,
            stdout=smoke.stdout[-2000:],
            stderr=smoke.stderr[-2000:],
        )
        return
    try:
        lines = [json.loads(line) for line in smoke.stdout.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        _check(
            checks,
            "worker-protocol-smoke",
            "error",
            "worker smoke returned non-JSON output",
            error=str(exc),
            stdout=smoke.stdout[-2000:],
        )
        return
    ok = any(
        item.get("id") == "doctor-smoke"
        and item.get("type") == "done"
        and item.get("status") == "ok"
        for item in lines
    )
    if ok:
        _check(
            checks,
            "worker-protocol-smoke",
            "ok",
            "worker JSON-lines shutdown smoke passed",
        )
    else:
        _check(
            checks,
            "worker-protocol-smoke",
            "error",
            "worker smoke did not return expected done/ok message",
            messages=lines,
        )


def _nix_check(checks: list[dict[str, Any]]) -> None:
    nix_path = shutil.which("nix")
    if not nix_path:
        for candidate in (
            "/nix/var/nix/profiles/default/bin/nix",
            "/run/current-system/sw/bin/nix",
        ):
            if Path(candidate).exists():
                nix_path = candidate
                break
    if not nix_path:
        _check(checks, "nix", "warning", "nix not found on PATH inside current process")
        return
    result = _run_command([nix_path, "--version"], timeout=5.0)
    if result.returncode == 0:
        _check(checks, "nix", "ok", result.stdout.strip(), path=nix_path)
    else:
        _check(
            checks,
            "nix",
            "warning",
            "nix --version failed",
            stderr=result.stderr[-1000:],
        )


def _parse_nvidia_smi_table(stdout: str) -> list[dict[str, str]]:
    gpus: list[dict[str, str]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3:
            continue
        gpus.append(
            {
                "name": fields[0],
                "driver_version": fields[1],
                "compute_capability": fields[2],
            }
        )
    return gpus


def _nvidia_smi_summary() -> tuple[list[dict[str, str]], str | None, str | None]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return [], None, "nvidia-smi not found"
    query = _run_command(
        [
            smi,
            "--query-gpu=name,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        timeout=10.0,
    )
    if query.returncode != 0:
        return (
            [],
            None,
            query.stderr.strip() or query.stdout.strip() or "nvidia-smi query failed",
        )
    raw = _run_command([smi], timeout=10.0)
    cuda_version: str | None = None
    match = re.search(r"CUDA Version:\s*([0-9.]+)", raw.stdout)
    if match:
        cuda_version = match.group(1)
    return _parse_nvidia_smi_table(query.stdout), cuda_version, None


def _gpu_check(
    checks: list[dict[str, Any]], profile: str, require_gpu: bool, require_amgx: bool
) -> None:
    gpu_profile = "cuda" in profile or require_gpu or require_amgx
    gpus, reported_cuda, error = _nvidia_smi_summary()
    if error:
        status = "error" if gpu_profile else "skip"
        _check(checks, "nvidia-smi", status, error, required=gpu_profile)
        return
    _check(
        checks,
        "nvidia-smi",
        "ok",
        "nvidia-smi detected GPU runtime",
        cuda_reported=reported_cuda,
        gpus=gpus,
    )

    required_driver = CUDA_12_8_MIN_LINUX_DRIVER
    if platform.system().lower().startswith("win"):
        required_driver = CUDA_12_8_MIN_WINDOWS_DRIVER
    failing = [
        gpu
        for gpu in gpus
        if not driver_meets_requirement(gpu["driver_version"], required_driver)
    ]
    driver_status = (
        "error" if failing and gpu_profile else "warning" if failing else "ok"
    )
    _check(
        checks,
        "cuda-driver-minimum",
        driver_status,
        f"CUDA {CUDA_TOOLKIT_VERSION} requires NVIDIA driver >= {required_driver}",
        required_driver=required_driver,
        cuda_toolkit=CUDA_TOOLKIT_VERSION,
        source=CUDA_DRIVER_REQUIREMENT_SOURCE,
        failing_gpus=failing,
    )

    sm61 = [gpu for gpu in gpus if gpu.get("compute_capability") == "6.1"]
    if sm61 and profile in {
        "cuda",
        "cuda-amgx",
        "complex-cuda",
        "complex-cuda-amgx",
        "complex64-cuda",
    }:
        _check(
            checks,
            "gpu-architecture-profile",
            "error" if gpu_profile else "warning",
            "sm_61 GPU detected; use cuda-sm61 or complex64-cuda-sm61 package/app",
            gpus=sm61,
        )
    elif profile.endswith("sm61") and not sm61:
        _check(
            checks,
            "gpu-architecture-profile",
            "warning",
            "sm61 profile selected but no compute capability 6.1 GPU detected",
        )
    else:
        _check(
            checks,
            "gpu-architecture-profile",
            "ok",
            "GPU architecture matches selected profile",
        )

    if require_amgx or "amgx" in profile:
        _check(
            checks,
            "amgx-driver",
            driver_status,
            f"AMGX route uses CUDA {CUDA_TOOLKIT_VERSION}; driver must satisfy CUDA minimum",
            required_driver=required_driver,
        )


def run_doctor(
    *,
    profile: str,
    require_gpu: bool = False,
    require_amgx: bool = False,
    worker_command: str = DEFAULT_WORKER_COMMAND,
    run_protocol_smoke: bool = True,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    normalized_profile = (profile or DEFAULT_PROFILE).strip() or DEFAULT_PROFILE
    package_attr = PROFILE_PACKAGE_ATTRS.get(normalized_profile, "")
    if not package_attr:
        _check(checks, "profile", "error", f"unknown profile: {normalized_profile}")
    else:
        _check(
            checks,
            "profile",
            "ok",
            f"profile {normalized_profile} maps to {package_attr}",
            package_attr=package_attr,
        )

    _nix_check(checks)
    _import_check(checks)
    _worker_check(checks, worker_command, run_protocol_smoke)
    _gpu_check(checks, normalized_profile, require_gpu, require_amgx)

    has_error = any(item["status"] == "error" for item in checks)
    has_warning = any(item["status"] == "warning" for item in checks)
    status = "error" if has_error else "warning" if has_warning else "ok"
    return {
        "schemaVersion": 1,
        "backendName": "PyEIDORS",
        "backendVersion": _read_project_version(),
        "status": status,
        "profile": normalized_profile,
        "packageAttr": package_attr,
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "prefix": sys.prefix,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "environment": {
            "PYEIDORS_ENV_PROFILE": os.environ.get("PYEIDORS_ENV_PROFILE"),
            "EIT_APP_GUI_RUNTIME_PROFILE": os.environ.get(
                "EIT_APP_GUI_RUNTIME_PROFILE"
            ),
            "PYEIDORS_PETSC_SCALAR_TYPE": os.environ.get("PYEIDORS_PETSC_SCALAR_TYPE"),
            "EIT_APP_GUI_PROFILE": os.environ.get("EIT_APP_GUI_PROFILE"),
        },
        "cuda": {
            "toolkitVersion": CUDA_TOOLKIT_VERSION,
            "minLinuxDriver": CUDA_12_8_MIN_LINUX_DRIVER,
            "minWindowsDriver": CUDA_12_8_MIN_WINDOWS_DRIVER,
            "requirementSource": CUDA_DRIVER_REQUIREMENT_SOURCE,
        },
        "checks": checks,
    }


def _print_human(report: dict[str, Any]) -> None:
    print(f"PyEIDORS backend doctor: {report['status']}")
    print(f"version={report['backendVersion']} profile={report['profile']}")
    for check in report["checks"]:
        print(f"[{check['status']}] {check['id']}: {check['message']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--worker-command", default=DEFAULT_WORKER_COMMAND)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--require-amgx", action="store_true")
    parser.add_argument("--skip-protocol-smoke", action="store_true")
    parser.add_argument("--format", choices=("human", "json"), default="human")
    parser.add_argument(
        "--no-fail", action="store_true", help="Always exit 0 after printing report."
    )
    args = parser.parse_args(argv)

    report = run_doctor(
        profile=args.profile,
        require_gpu=args.require_gpu,
        require_amgx=args.require_amgx,
        worker_command=args.worker_command,
        run_protocol_smoke=not args.skip_protocol_smoke,
    )
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_human(report)
    if args.no_fail:
        return 0
    return 1 if report["status"] == "error" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
