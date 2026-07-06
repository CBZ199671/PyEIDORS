from __future__ import annotations

import json
from pathlib import Path

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
