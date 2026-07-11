from __future__ import annotations

import json
from pathlib import Path
import subprocess

from eit_app import backend_doctor


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_backend_manifest_defaults_to_pure_nix_backend_worker_app():
    manifest = json.loads(
        (REPO_ROOT / "pyeidors.backend.json").read_text(encoding="utf-8")
    )

    assert manifest["schemaVersion"] == 1
    assert manifest["defaultProfile"] == "complex64"
    assert manifest["developmentRuntimeAllowed"] is False
    default_profile = manifest["profiles"]["complex64"]
    assert default_profile["workerLaunchCommand"] == (
        "nix run .#eit-backend-worker-complex64 -- serve"
    )
    assert default_profile["doctorCommand"] == (
        "nix run .#eit-backend-doctor-complex64 -- --profile complex64 --format json"
    )
    assert "nix develop" not in default_profile["workerLaunchCommand"]
    assert "doctor" in manifest["capabilities"]


def test_backend_manifest_exposes_all_packaged_backend_profiles():
    manifest = json.loads(
        (REPO_ROOT / "pyeidors.backend.json").read_text(encoding="utf-8")
    )
    profiles = manifest["profiles"]

    assert list(profiles) == [
        "default",
        "complex64",
        "complex",
        "cuda",
        "cuda-amgx",
        "complex64-cuda",
        "complex-cuda",
        "complex-cuda-amgx",
        "cuda-sm61",
        "complex64-cuda-sm61",
    ]
    for name, profile in profiles.items():
        assert profile["displayName"]
        assert profile["packageAttr"]
        assert profile["workerApp"]
        assert profile["doctorApp"]
        assert (
            profile["workerLaunchCommand"]
            == f"nix run .#{profile['workerApp']} -- serve"
        )
        assert (
            f"nix run .#{profile['doctorApp']} -- --profile {name}"
            in profile["doctorCommand"]
        )
        assert "nix develop" not in profile["workerLaunchCommand"]


def test_v672_pure_nix_worker_import_checks_include_ecd_cwr_simulation():
    flake = (REPO_ROOT / "flake.nix").read_text(encoding="utf-8")

    assert '"eit_app.ecd_cwr_simulation"' in flake


def test_v673_pure_nix_runtime_exposes_fenics_jit_toolchain():
    flake = (REPO_ROOT / "flake.nix").read_text(encoding="utf-8")
    wrapper = flake.split("makeWrapperArgs = [", 1)[1].split("] ++ lib.concatLists", 1)[
        0
    ]

    assert "pkgsFor.stdenv.cc" in wrapper
    assert (
        '"--set"\n                "CC"\n'
        '                "${pkgsFor.stdenv.cc}/bin/cc"' in wrapper
    )
    assert (
        '"--set"\n                "CXX"\n'
        '                "${pkgsFor.stdenv.cc}/bin/c++"' in wrapper
    )


def test_v680_pure_nix_wrapper_init_bundles_core_runtime_commands():
    flake = (REPO_ROOT / "flake.nix").read_text(encoding="utf-8")
    wrapper = flake.split("makeWrapperArgs = [", 1)[1].split("] ++ lib.concatLists", 1)[
        0
    ]

    assert "pkgsFor.coreutils" in wrapper


def test_backend_manifest_records_cuda_amgx_driver_requirement():
    manifest = json.loads(
        (REPO_ROOT / "pyeidors.backend.json").read_text(encoding="utf-8")
    )

    amgx = manifest["profiles"]["cuda-amgx"]
    assert amgx["requiresGpu"] is True
    assert amgx["requiresAmgx"] is True
    assert amgx["cuda"]["toolkitVersion"] == "12.8.1"
    assert amgx["cuda"]["minLinuxDriver"] == "570.124.06"


def test_backend_doctor_driver_version_comparison():
    assert backend_doctor.driver_meets_requirement("570.124.06", "570.124.06")
    assert backend_doctor.driver_meets_requirement("575.57.08", "570.124.06")
    assert not backend_doctor.driver_meets_requirement("550.144.03", "570.124.06")


def test_backend_doctor_parses_nvidia_smi_query_rows():
    rows = backend_doctor._parse_nvidia_smi_table(
        "NVIDIA GeForce GTX 1050 Ti, 570.124.06, 6.1\n"
        "NVIDIA GeForce RTX 4090, 575.57.08, 8.9\n"
    )

    assert rows == [
        {
            "name": "NVIDIA GeForce GTX 1050 Ti",
            "driver_version": "570.124.06",
            "compute_capability": "6.1",
        },
        {
            "name": "NVIDIA GeForce RTX 4090",
            "driver_version": "575.57.08",
            "compute_capability": "8.9",
        },
    ]


def _check_by_id(checks, check_id: str):
    return next(item for item in checks if item["id"] == check_id)


def test_v673_backend_doctor_reports_missing_fenics_jit_toolchain(
    monkeypatch,
):
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.setattr(backend_doctor.shutil, "which", lambda _command: None)
    checks = []

    backend_doctor._compiler_check(checks)

    check = _check_by_id(checks, "fenics-jit-toolchain")
    assert check["status"] == "error"
    assert "C/C++ compiler toolchain" in check["message"]


def test_v673_backend_doctor_compiles_and_links_fenics_jit_probes(monkeypatch):
    monkeypatch.setenv("CC", "nix-cc")
    monkeypatch.setenv("CXX", "nix-cxx")
    compiler_paths = {
        "nix-cc": "/nix/store/toolchain/bin/cc",
        "nix-cxx": "/nix/store/toolchain/bin/c++",
    }
    monkeypatch.setattr(
        backend_doctor.shutil,
        "which",
        lambda command: compiler_paths.get(command),
    )
    commands = []

    def _succeed(args, timeout=10.0, input_text=None):
        _ = timeout, input_text
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(backend_doctor, "_run_command_safely", _succeed)
    checks = []

    backend_doctor._compiler_check(checks)

    check = _check_by_id(checks, "fenics-jit-toolchain")
    assert check["status"] == "ok"
    assert check["cc"] == "/nix/store/toolchain/bin/cc"
    assert check["cxx"] == "/nix/store/toolchain/bin/c++"
    assert [command[0] for command in commands] == [
        "/nix/store/toolchain/bin/cc",
        "/nix/store/toolchain/bin/c++",
    ]


def test_backend_doctor_run_command_converts_timeout(monkeypatch):
    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(backend_doctor.subprocess, "run", _raise_timeout)

    result = backend_doctor._run_command(["slow-command"], timeout=3.0)

    assert result.returncode == backend_doctor.COMMAND_TIMEOUT_RETURNCODE
    assert backend_doctor._command_timed_out(result)
    assert "timed out after 3 seconds" in result.stderr


def test_backend_doctor_worker_help_timeout_becomes_error_check(monkeypatch):
    monkeypatch.setattr(backend_doctor.shutil, "which", lambda _command: "/bin/worker")

    def _raise_timeout(args, timeout=10.0, input_text=None):
        _ = input_text
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)

    monkeypatch.setattr(backend_doctor, "_run_command", _raise_timeout)
    checks = []

    backend_doctor._worker_check(checks, "eit-backend-worker", True)

    check = _check_by_id(checks, "worker-help")
    assert check["status"] == "error"
    assert check["returncode"] == backend_doctor.COMMAND_TIMEOUT_RETURNCODE
    assert "timed out after 10 seconds" in check["message"]


def test_backend_doctor_worker_smoke_timeout_uses_configured_timeout(monkeypatch):
    monkeypatch.setattr(backend_doctor.shutil, "which", lambda _command: "/bin/worker")

    def _run_or_timeout(args, timeout=10.0, input_text=None):
        _ = input_text
        if args == ["/bin/worker", "--help"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout="usage: worker\nserve\n",
                stderr="",
            )
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)

    monkeypatch.setattr(backend_doctor, "_run_command", _run_or_timeout)
    checks = []

    backend_doctor._worker_check(
        checks,
        "eit-backend-worker",
        True,
        smoke_timeout=7.0,
    )

    check = _check_by_id(checks, "worker-protocol-smoke")
    assert check["status"] == "error"
    assert check["returncode"] == backend_doctor.COMMAND_TIMEOUT_RETURNCODE
    assert "timed out after 7 seconds" in check["message"]


def test_backend_doctor_nix_timeout_becomes_error_check(monkeypatch):
    monkeypatch.setattr(backend_doctor.shutil, "which", lambda name: "/bin/nix")

    def _raise_timeout(args, timeout=10.0, input_text=None):
        _ = input_text
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)

    monkeypatch.setattr(backend_doctor, "_run_command", _raise_timeout)
    checks = []

    backend_doctor._nix_check(checks)

    check = _check_by_id(checks, "nix")
    assert check["status"] == "error"
    assert "timed out after 5 seconds" in check["message"]


def test_backend_doctor_nvidia_smi_timeout_becomes_error_check(monkeypatch):
    monkeypatch.setattr(
        backend_doctor.shutil,
        "which",
        lambda name: "/bin/nvidia-smi" if name == "nvidia-smi" else None,
    )

    def _raise_timeout(args, timeout=10.0, input_text=None):
        _ = input_text
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)

    monkeypatch.setattr(backend_doctor, "_run_command", _raise_timeout)
    checks = []

    backend_doctor._gpu_check(checks, "cuda-amgx", require_gpu=True, require_amgx=True)

    check = _check_by_id(checks, "nvidia-smi")
    assert check["status"] == "error"
    assert "timed out after 10 seconds" in check["message"]


def test_backend_doctor_json_output_survives_timeout(monkeypatch, capsys):
    def _which(name):
        if name == "nix":
            return "/bin/nix"
        return None

    def _raise_timeout(args, timeout=10.0, input_text=None):
        _ = input_text
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)

    monkeypatch.setattr(backend_doctor.shutil, "which", _which)
    monkeypatch.setattr(backend_doctor, "_run_command", _raise_timeout)

    exit_code = backend_doctor.main(["--format", "json"])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert report["status"] == "error"
    assert _check_by_id(report["checks"], "nix")["status"] == "error"


def test_flake_exposes_backend_worker_and_doctor_apps():
    text = (REPO_ROOT / "flake.nix").read_text(encoding="utf-8")
    manifest = json.loads(
        (REPO_ROOT / "pyeidors.backend.json").read_text(encoding="utf-8")
    )

    assert 'rel == "pyeidors.backend.json"' in text
    for profile in manifest["profiles"].values():
        assert (
            f'{profile["workerApp"]} = mkApp "{profile["packageAttr"]}" "eit-backend-worker"'
            in text
        )
        assert (
            f'{profile["doctorApp"]} = mkApp "{profile["packageAttr"]}" "eit-backend-doctor"'
            in text
        )


def test_release_source_zip_stages_backend_manifest():
    text = (
        REPO_ROOT / "scripts" / "release" / "build_private_distribution.sh"
    ).read_text(encoding="utf-8")

    assert '"pyeidors.backend.json",' in text
