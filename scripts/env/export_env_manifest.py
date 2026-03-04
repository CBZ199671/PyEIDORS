#!/usr/bin/env python3
"""Export reproducible environment manifest for PyEIDORS."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as ilm
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


PROFILE_EXTRAS = ["torch", "cuqi", "dev"]
PROFILE_SYNC_FLAGS = ["--frozen"]
PROFILE_LOCK_CHECK = "uv lock --check"
PLATFORM_MAP = {
    "macos-aarch64": ("darwin", "arm64"),
    "macos-x86_64": ("darwin", "x86_64"),
    "linux-x86_64": ("linux", "x86_64"),
    "linux-aarch64": ("linux", "aarch64"),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_nixpkgs_rev(flake_lock_path: Path) -> str:
    payload = json.loads(flake_lock_path.read_text(encoding="utf-8"))
    return payload["nodes"]["nixpkgs"]["locked"]["rev"]


def current_platform_id() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return "macos-aarch64"
    if system == "darwin" and machine in {"x86_64", "amd64"}:
        return "macos-x86_64"
    if system == "linux" and machine in {"x86_64", "amd64"}:
        return "linux-x86_64"
    if system == "linux" and machine in {"arm64", "aarch64"}:
        return "linux-aarch64"
    return f"{system}-{machine}"


def platform_details(platform_id: Optional[str] = None) -> Dict[str, str]:
    selected_id = platform_id or current_platform_id()
    mapped = PLATFORM_MAP.get(selected_id)
    if mapped is None:
        return {
            "id": selected_id,
            "system": platform.system().lower(),
            "machine": platform.machine().lower(),
        }
    system_name, machine_name = mapped
    return {"id": selected_id, "system": system_name, "machine": machine_name}


def package_version(module_name: str, dist_name: Optional[str] = None) -> str:
    mod = importlib.import_module(module_name)
    version = getattr(mod, "__version__", None)
    if isinstance(version, str) and version:
        return version
    if dist_name is not None:
        return ilm.version(dist_name)
    return ilm.version(module_name)


def build_manifest(root: Path, platform_id: Optional[str] = None) -> Dict[str, Any]:
    flake_lock = root / "flake.lock"
    uv_lock = root / "uv.lock"
    pyproject = root / "pyproject.toml"

    selected_platform = platform_details(platform_id)

    packages = {
        "dolfinx": package_version("dolfinx"),
        "torch": package_version("torch"),
        "cuqi": package_version("cuqi", "CUQIpy"),
        "numpy": package_version("numpy"),
        "scipy": package_version("scipy"),
        "pyeidors": package_version("pyeidors"),
    }

    return {
        "schema_version": 1,
        "project": "pyeidors",
        "profile": {
            "extras": PROFILE_EXTRAS,
            "sync_flags": PROFILE_SYNC_FLAGS,
            "lock_check": PROFILE_LOCK_CHECK,
        },
        "platform": selected_platform,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "locks": {
            "nixpkgs_rev": read_nixpkgs_rev(flake_lock),
            "flake_lock_sha256": sha256_file(flake_lock),
            "uv_lock_sha256": sha256_file(uv_lock),
            "pyproject_sha256": sha256_file(pyproject),
        },
        "packages": packages,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PyEIDORS environment manifest")
    parser.add_argument("--output", type=Path, required=True, help="Output manifest JSON path")
    parser.add_argument(
        "--platform-id",
        type=str,
        default=None,
        help="Override platform id (default: detect from current runtime)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    manifest = build_manifest(root, platform_id=args.platform_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[env-manifest] wrote {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"[env-manifest] ERROR: {exc}", file=sys.stderr)
        raise
