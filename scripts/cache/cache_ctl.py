#!/usr/bin/env python3
"""Cache control utility for PyEIDORS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pyeidors.cache import CacheManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["status", "list", "clear-all", "clear-name", "clear-max", "collect-recent"])
    parser.add_argument("--cache-scope", choices=["off", "process", "both"], default="both")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / ".pyeidors_cache" / "v2")
    parser.add_argument("--name", action="append", default=[], help="Cache family name")
    parser.add_argument("--namespace", type=str, default=None)
    parser.add_argument("--max-bytes", type=int, default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--limit-per-name", type=int, default=1)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    manager = CacheManager(
        scope=str(args.cache_scope),
        cache_dir=args.cache_dir,
    )

    if args.command == "status":
        print(json.dumps(manager.stats(), indent=2, ensure_ascii=False))
        return

    if args.command == "list":
        name = args.name[0] if args.name else None
        entries = manager.list_entries(
            name=name,
            namespace=args.namespace,
            limit=args.limit,
        )
        print(json.dumps(entries, indent=2, ensure_ascii=False, default=str))
        return

    if args.command == "clear-all":
        manager.clear(scope="both")
        print(json.dumps({"removed": "all"}, ensure_ascii=False))
        return

    if args.command == "clear-name":
        if not args.name:
            raise SystemExit("clear-name requires at least one --name")
        removed = 0
        for name in args.name:
            removed += manager.clear_name(name=name, namespace=args.namespace)
        print(json.dumps({"removed": removed}, ensure_ascii=False))
        return

    if args.command == "clear-max":
        if args.max_bytes is None:
            raise SystemExit("clear-max requires --max-bytes")
        removed = manager.clear_max(max_bytes=int(args.max_bytes))
        print(json.dumps({"removed": removed}, ensure_ascii=False))
        return

    if args.command == "collect-recent":
        if not args.name:
            raise SystemExit("collect-recent requires at least one --name")
        collected = manager.collect_recent(
            names=list(args.name),
            limit_per_name=max(1, int(args.limit_per_name)),
            namespace=args.namespace,
        )
        print(json.dumps(collected, indent=2, ensure_ascii=False, default=str))
        return


if __name__ == "__main__":
    main()

