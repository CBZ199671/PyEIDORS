#!/usr/bin/env python3
"""Precompute or load common 3D GREIT HDF5 warmup artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyeidors.inverse.greit_warmup import (
    GREIT_COMMON_CONFIG_WARMUP_SCHEMA,
    greit_common_config_ids,
    load_greit_common_config,
    precompute_greit_common_config,
    register_greit_common_config_artifact,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config_ids = _selected_config_ids(args.config)
    if args.source_artifact is not None and len(config_ids) != 1:
        raise SystemExit("--source-artifact requires one --config, not all.")

    results: list[dict[str, Any]] = []
    for config_id in config_ids:
        if args.source_artifact is not None:
            result = register_greit_common_config_artifact(
                config_id,
                args.source_artifact,
                artifact_dir=args.artifact_dir,
                overwrite=args.overwrite,
                prepare_online=args.prepare_online,
                device=args.device,
                dtype=args.dtype,
                strict_shape=not args.allow_shape_mismatch,
            )
        elif args.load_only:
            result = load_greit_common_config(
                config_id,
                artifact_dir=args.artifact_dir,
                prepare_online=args.prepare_online,
                device=args.device,
                dtype=args.dtype,
            )
        else:
            result = precompute_greit_common_config(
                config_id,
                artifact_dir=args.artifact_dir,
                overwrite=args.overwrite,
                prepare_online=args.prepare_online,
                device=args.device,
                dtype=args.dtype,
            )
        results.append(result.as_json())

    manifest = {
        "schema": GREIT_COMMON_CONFIG_WARMUP_SCHEMA,
        "artifact_dir": str(Path(args.artifact_dir).expanduser())
        if args.artifact_dir is not None
        else None,
        "config_count": len(results),
        "configs": results,
        "online_contract": "load_hdf5_then_rm_matmul",
    }
    encoded = json.dumps(manifest, indent=2, sort_keys=True)
    if args.manifest_out is not None:
        output = Path(args.manifest_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    choices = ("all", *greit_common_config_ids())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        choices=choices,
        default="all",
        help="Common hardware config to precompute/load.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Directory for greit3d_common_<config>.h5 artifacts.",
    )
    parser.add_argument(
        "--source-artifact",
        type=Path,
        default=None,
        help="Externally built GREIT .h5 artifact to register under --config.",
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Only load an existing artifact; never build a fixture.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing common-config artifact.",
    )
    parser.add_argument(
        "--prepare-online",
        action="store_true",
        help="Prepare the RM matmul handle after loading.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Matmul device used with --prepare-online.",
    )
    parser.add_argument(
        "--dtype",
        default="float64",
        help="Matmul dtype used with --prepare-online.",
    )
    parser.add_argument(
        "--allow-shape-mismatch",
        action="store_true",
        help="Allow source artifacts whose RM shape differs from the config contract.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Optional JSON manifest output path.",
    )
    return parser.parse_args(argv)


def _selected_config_ids(config: str) -> tuple[str, ...]:
    if config == "all":
        return greit_common_config_ids()
    return (config,)


if __name__ == "__main__":
    raise SystemExit(main())
