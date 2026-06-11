"""Unit tests for environment manifest export/verify helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_SCRIPTS_DIR = REPO_ROOT / "scripts" / "env"
if str(ENV_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_SCRIPTS_DIR))

import export_env_manifest as exporter
import verify_env_manifest as verifier


def _manifest_base() -> dict:
    return {
        "schema_version": 1,
        "project": "pyeidors",
        "profile": {
            "extras": ["torch", "cuqi", "dev"],
            "sync_flags": ["--frozen"],
            "lock_check": "uv lock --check",
        },
        "platform": {
            "id": "macos-aarch64",
            "system": "darwin",
            "machine": "arm64",
            "runtime_context": {"kind": "native"},
        },
        "python": {
            "version": "3.13.2",
            "implementation": "CPython",
        },
        "locks": {
            "nixpkgs_rev": "abc",
            "flake_lock_sha256": "h1",
            "uv_lock_sha256": "h2",
            "pyproject_sha256": "h3",
        },
        "packages": {
            "dolfinx": "0.9.0",
            "torch": "2.10.0",
            "cuqi": "1.5.0",
            "numpy": "2.2.3",
            "scipy": "1.16.2",
            "pyeidors": "1.0.0",
        },
    }


def test_compare_manifests_reports_no_diff_for_identical_payload():
    expected = _manifest_base()
    actual = _manifest_base()
    assert verifier.compare_manifests(actual, expected) == []


def test_compare_manifests_reports_platform_and_version_mismatch():
    expected = _manifest_base()
    actual = _manifest_base()
    actual["platform"]["id"] = "linux-x86_64"
    actual["platform"]["system"] = "linux"
    actual["packages"]["scipy"] = "1.15.0"

    diffs = verifier.compare_manifests(actual, expected)
    assert any("platform.id" in diff for diff in diffs)
    assert any("platform.system" in diff for diff in diffs)
    assert any("packages.scipy" in diff for diff in diffs)


def test_default_manifest_path_uses_detected_platform(monkeypatch):
    monkeypatch.setattr(verifier, "current_platform_id", lambda: "linux-x86_64")
    path = verifier.default_manifest_path(Path("/repo"))
    assert path == Path("/repo/env/manifests/linux-x86_64-complex64-cuda.lock.json")


def test_platform_details_uses_override_mapping(monkeypatch):
    monkeypatch.setattr(exporter, "runtime_context_kind", lambda: None)
    details = exporter.platform_details("linux-x86_64")
    assert details == {
        "id": "linux-x86_64",
        "system": "linux",
        "machine": "x86_64",
    }


def test_compare_manifests_ignores_runtime_context_differences():
    expected = _manifest_base()
    actual = _manifest_base()
    actual["platform"]["runtime_context"] = {"kind": "wsl2"}

    assert verifier.compare_manifests(actual, expected) == []


def test_build_manifest_collects_lock_and_profile_fields(tmp_path: Path, monkeypatch):
    (tmp_path / "flake.lock").write_text(
        json.dumps({"nodes": {"nixpkgs": {"locked": {"rev": "deadbeef"}}}}),
        encoding="utf-8",
    )
    (tmp_path / "uv.lock").write_text("uv-lock", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='pyeidors'\n", encoding="utf-8"
    )
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='pyeidors'\n", encoding="utf-8"
    )

    versions = {
        "dolfinx": "0.9.0",
        "torch": "2.10.0",
        "cuqi": "1.5.0",
        "numpy": "2.2.3",
        "scipy": "1.16.2",
        "pyeidors": "1.0.0",
    }

    def _fake_package_version(module_name: str, dist_name: str | None = None) -> str:
        return versions[module_name]

    monkeypatch.setattr(exporter, "package_version", _fake_package_version)
    monkeypatch.setattr(exporter, "runtime_context_kind", lambda: "wsl2")

    manifest = exporter.build_manifest(tmp_path, platform_id="linux-x86_64")
    assert manifest["profile"]["extras"] == ["torch", "cuqi", "dev"]
    assert manifest["profile"]["sync_flags"] == ["--frozen"]
    assert manifest["profile"]["lock_check"] == "uv lock --check"
    assert manifest["platform"]["id"] == "linux-x86_64"
    assert manifest["platform"]["system"] == "linux"
    assert manifest["platform"]["runtime_context"]["kind"] == "wsl2"
    assert manifest["locks"]["nixpkgs_rev"] == "deadbeef"
    assert manifest["packages"]["cuqi"] == "1.5.0"


def test_default_manifest_path_uses_profile_suffix(monkeypatch):
    monkeypatch.setattr(verifier, "current_platform_id", lambda: "linux-x86_64")
    path = verifier.default_manifest_path(Path("/repo"), profile_name="cuda")
    assert path == Path("/repo/env/manifests/linux-x86_64-cuda.lock.json")


def test_build_manifest_adds_profile_name_for_nondefault(tmp_path: Path, monkeypatch):
    (tmp_path / "flake.lock").write_text(
        json.dumps({"nodes": {"nixpkgs": {"locked": {"rev": "deadbeef"}}}}),
        encoding="utf-8",
    )
    (tmp_path / "uv.lock").write_text("uv-lock", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='pyeidors'\n", encoding="utf-8"
    )

    versions = {
        "dolfinx": "0.9.0",
        "torch": "2.10.0",
        "cuqi": "1.5.0",
        "numpy": "2.2.3",
        "scipy": "1.16.2",
        "pyeidors": "1.0.0",
    }

    monkeypatch.setattr(
        exporter,
        "package_version",
        lambda module_name, dist_name=None: versions[module_name],
    )
    monkeypatch.setattr(exporter, "runtime_context_kind", lambda: "wsl2")

    manifest = exporter.build_manifest(
        tmp_path, platform_id="linux-x86_64", profile_name="cuda"
    )
    assert manifest["profile"]["name"] == "cuda"
