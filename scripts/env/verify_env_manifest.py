#!/usr/bin/env python3
"""Verify current environment against locked manifest for this platform."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from export_env_manifest import build_manifest, current_platform_id, repo_root


def default_manifest_path(root: Path) -> Path:
    return root / "env" / "manifests" / f"{current_platform_id()}.lock.json"


def load_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cmp_field(diffs: List[str], key: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        diffs.append(f"{key}: expected={expected!r}, actual={actual!r}")


def compare_manifests(actual: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
    diffs: List[str] = []

    _cmp_field(diffs, "schema_version", actual.get("schema_version"), expected.get("schema_version"))
    _cmp_field(diffs, "project", actual.get("project"), expected.get("project"))

    _cmp_field(
        diffs,
        "profile.extras",
        actual.get("profile", {}).get("extras"),
        expected.get("profile", {}).get("extras"),
    )
    _cmp_field(
        diffs,
        "profile.sync_flags",
        actual.get("profile", {}).get("sync_flags"),
        expected.get("profile", {}).get("sync_flags"),
    )
    _cmp_field(
        diffs,
        "profile.lock_check",
        actual.get("profile", {}).get("lock_check"),
        expected.get("profile", {}).get("lock_check"),
    )

    _cmp_field(
        diffs,
        "platform.id",
        actual.get("platform", {}).get("id"),
        expected.get("platform", {}).get("id"),
    )
    _cmp_field(
        diffs,
        "platform.system",
        actual.get("platform", {}).get("system"),
        expected.get("platform", {}).get("system"),
    )
    _cmp_field(
        diffs,
        "platform.machine",
        actual.get("platform", {}).get("machine"),
        expected.get("platform", {}).get("machine"),
    )
    _cmp_field(
        diffs,
        "python.version",
        actual.get("python", {}).get("version"),
        expected.get("python", {}).get("version"),
    )

    for lock_key in ("nixpkgs_rev", "flake_lock_sha256", "uv_lock_sha256", "pyproject_sha256"):
        _cmp_field(
            diffs,
            f"locks.{lock_key}",
            actual.get("locks", {}).get(lock_key),
            expected.get("locks", {}).get(lock_key),
        )

    expected_packages = expected.get("packages", {})
    actual_packages = actual.get("packages", {})
    for name, expected_version in expected_packages.items():
        _cmp_field(
            diffs,
            f"packages.{name}",
            actual_packages.get(name),
            expected_version,
        )

    return diffs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify current environment against lock manifest")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to manifest JSON (default: env/manifests/<platform>.lock.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    manifest_path = args.manifest if args.manifest is not None else default_manifest_path(root)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. "
            "Generate it with scripts/env/export_env_manifest.py."
        )

    expected = load_manifest(manifest_path)
    actual = build_manifest(root, platform_id=current_platform_id())
    diffs = compare_manifests(actual, expected)

    if diffs:
        print("[env-verify] environment mismatch detected:", file=sys.stderr)
        for item in diffs:
            print(f"  - {item}", file=sys.stderr)
        print("[env-verify] repair command: scripts/env/sync_locked_env.sh --repair", file=sys.stderr)
        raise SystemExit(1)

    print(f"[env-verify] OK: {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"[env-verify] ERROR: {exc}", file=sys.stderr)
        raise
