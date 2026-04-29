#!/usr/bin/env python3
"""Migrate legacy NumPy numeric artifacts to HDF5 packages."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pyeidors.io.hdf5_artifacts import migrate_npz_to_hdf5


MANIFEST_SCHEMA = "pyeidors-hdf5-migration-manifest-v1"
LEGACY_SUFFIXES = {".npz", ".npy"}
DEFAULT_EXCLUDE_DIRS = (
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv-cuda",
    ".venv-mfem-cuda",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
)


def discover_legacy_artifacts(
    root: str | Path,
    *,
    exclude_dirs: Iterable[str] = DEFAULT_EXCLUDE_DIRS,
) -> list[Path]:
    """Return legacy ``.npz`` / ``.npy`` artifacts under ``root``."""

    base = Path(root)
    excluded = {str(name) for name in exclude_dirs}
    paths: list[Path] = []
    for current, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(name for name in dirnames if name not in excluded)
        current_path = Path(current)
        for filename in sorted(filenames):
            path = current_path / filename
            if path.suffix.lower() in LEGACY_SUFFIXES:
                paths.append(path)
    return paths


def build_migration_manifest(
    *,
    root: str | Path,
    apply: bool,
    overwrite: bool = False,
    exclude_dirs: Iterable[str] = DEFAULT_EXCLUDE_DIRS,
) -> dict[str, Any]:
    """Build a migration manifest and optionally write HDF5 destinations."""

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"--root does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"--root must be a directory: {root_path}")

    items: list[dict[str, Any]] = []
    for source in discover_legacy_artifacts(root_path, exclude_dirs=exclude_dirs):
        target = source.with_suffix(".h5")
        item: dict[str, Any] = {
            "source": source.as_posix(),
            "target": target.as_posix(),
            "legacy_format": source.suffix.lower().lstrip("."),
            "status": "planned",
            "reason": None,
        }
        if target.exists() and not overwrite:
            item["status"] = "skipped"
            item["reason"] = "target_exists"
        elif apply:
            try:
                migrated = migrate_npz_to_hdf5(
                    source,
                    target,
                    metadata={
                        "migration_cli": "scripts/cache/migrate_artifacts_to_hdf5.py",
                        "migration_root": root_path.as_posix(),
                    },
                )
            except Exception as exc:  # pragma: no cover - reported in manifest
                item["status"] = "error"
                item["reason"] = f"{type(exc).__name__}: {exc}"
            else:
                item["target"] = migrated.as_posix()
                item["status"] = "migrated"
        items.append(item)

    counts = {
        "planned": sum(1 for item in items if item["status"] == "planned"),
        "migrated": sum(1 for item in items if item["status"] == "migrated"),
        "skipped": sum(1 for item in items if item["status"] == "skipped"),
        "error": sum(1 for item in items if item["status"] == "error"),
    }
    return {
        "schema": MANIFEST_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": root_path.as_posix(),
        "mode": "apply" if apply else "dry-run",
        "overwrite": bool(overwrite),
        "legacy_suffixes": sorted(LEGACY_SUFFIXES),
        "counts": counts,
        "items": items,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Plan only")
    mode.add_argument("--apply", action="store_true", help="Write HDF5 targets")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .h5 targets when applying.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional path for the JSON manifest. The manifest is always printed.",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Directory basename to skip during recursive discovery. Repeatable.",
    )
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    exclude_dirs = tuple(DEFAULT_EXCLUDE_DIRS) + tuple(args.exclude_dir or ())
    payload = build_migration_manifest(
        root=args.root,
        apply=bool(args.apply),
        overwrite=bool(args.overwrite),
        exclude_dirs=exclude_dirs,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(text + "\n", encoding="utf-8")
    return 1 if payload["counts"]["error"] else 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
