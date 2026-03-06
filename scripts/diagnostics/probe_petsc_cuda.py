#!/usr/bin/env python3
"""Probe whether PETSc in the current runtime truly supports CUDA backends."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pyeidors.perf.capabilities import probe_petsc_cuda_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require",
        choices=["cuda", "none"],
        default="none",
        help="Fail if the requested PETSc CUDA capability is not available.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = probe_petsc_cuda_runtime()
    print(json.dumps(payload, indent=2 if args.pretty else None, ensure_ascii=False))
    if args.require == "cuda" and not bool(payload.get("petsc_cuda", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
