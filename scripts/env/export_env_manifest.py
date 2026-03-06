#!/usr/bin/env python3
"""Export reproducible environment manifest for PyEIDORS."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as ilm
import json
import os
import platform
import sys
import warnings
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


class MissingRequiredPackagesError(RuntimeError):
    """Raised when the locked runtime packages are not importable."""


def apply_known_cuqi_warning_filters() -> None:
    """Suppress known CUQI import warnings while collecting manifest info."""

    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message=r"pkg_resources is deprecated as an API",
        module=r"(pkg_resources(\..*)?|setuptools\._vendor\.pkg_resources(\..*)?|cuqi(\..*)?)",
    )
    warnings.filterwarnings(
        action="ignore",
        category=PendingDeprecationWarning,
        message=r"Importing from numpy\.matlib is deprecated",
        module=r"(numpy\.matlib(\..*)?|cuqi(\..*)?)",
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_repo_src_on_path(root: Path) -> None:
    src_path = root / "src"
    src_str = str(src_path)
    if src_path.is_dir() and src_str not in sys.path:
        sys.path.insert(0, src_str)


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


def runtime_context_kind() -> Optional[str]:
    if platform.system().lower() != "linux":
        return None

    if os.environ.get("WSL_DISTRO_NAME"):
        return "wsl2"

    try:
        version_text = Path("/proc/version").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    return "wsl2" if "microsoft" in version_text.lower() else None


def platform_details(platform_id: Optional[str] = None) -> Dict[str, Any]:
    selected_id = platform_id or current_platform_id()
    mapped = PLATFORM_MAP.get(selected_id)
    runtime_kind = runtime_context_kind()

    if mapped is None:
        details: Dict[str, Any] = {
            "id": selected_id,
            "system": platform.system().lower(),
            "machine": platform.machine().lower(),
        }
    else:
        system_name, machine_name = mapped
        details = {"id": selected_id, "system": system_name, "machine": machine_name}

    if runtime_kind is not None:
        details["runtime_context"] = {"kind": runtime_kind}

    return details


def package_version(module_name: str, dist_name: Optional[str] = None) -> str:
    apply_known_cuqi_warning_filters()
    mod = importlib.import_module(module_name)
    version = getattr(mod, "__version__", None)
    if isinstance(version, str) and version:
        return version
    if dist_name is not None:
        return ilm.version(dist_name)
    return ilm.version(module_name)


def collect_package_versions() -> Dict[str, str]:
    required = (
        ("dolfinx", None),
        ("torch", None),
        ("cuqi", "CUQIpy"),
        ("numpy", None),
        ("scipy", None),
        ("pyeidors", None),
    )
    packages: Dict[str, str] = {}
    missing: list[str] = []
    for module_name, dist_name in required:
        try:
            packages[module_name] = package_version(module_name, dist_name)
        except Exception as exc:
            missing.append(f"{module_name}: {exc}")
    if missing:
        raise MissingRequiredPackagesError(
            "missing required imports for locked manifest verification: "
            + "; ".join(missing)
            + ". Enter the supported dev shell with `nix develop` and retry."
        )
    return packages


def build_manifest(root: Path, platform_id: Optional[str] = None) -> Dict[str, Any]:
    flake_lock = root / "flake.lock"
    uv_lock = root / "uv.lock"
    pyproject = root / "pyproject.toml"

    ensure_repo_src_on_path(root)

    selected_platform = platform_details(platform_id)

    packages = collect_package_versions()

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
    except MissingRequiredPackagesError as exc:  # pragma: no cover
        print(f"[env-manifest] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception as exc:  # pragma: no cover
        print(f"[env-manifest] ERROR: {exc}", file=sys.stderr)
        raise
