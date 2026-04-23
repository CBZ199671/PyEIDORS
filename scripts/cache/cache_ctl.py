#!/usr/bin/env python3
"""Cache control utility for PyEIDORS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pyeidors.cache import CacheManager, CachePolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "status",
            "list",
            "on",
            "off",
            "debug-on",
            "debug-off",
            "boost-priority",
            "clear-all",
            "clear-name",
            "clear-max",
            "clear-old",
            "clear-new",
            "collect-recent",
            "install-to-cache",
        ],
    )
    parser.add_argument(
        "--cache-scope", choices=["off", "process", "both"], default="both"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / ".pyeidors_cache" / "v2",
        help="Persistent cache root to inspect/manage. Runtime session caches live under <cache-dir>/.sessions/.",
    )
    parser.add_argument("--name", action="append", default=[], help="Cache family name")
    parser.add_argument("--namespace", type=str, default=None)
    parser.add_argument("--max-bytes", type=int, default=None)
    parser.add_argument("--timestamp", type=float, default=None)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--limit-per-name", type=int, default=1)
    parser.add_argument(
        "--with-values", action="store_true", help="Include cached values"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Write JSON output to path"
    )
    parser.add_argument(
        "--input", type=Path, default=None, help="Read snapshot JSON from path"
    )
    parser.add_argument(
        "--target-layers", choices=["process", "disk", "both"], default="both"
    )
    return parser


def _print_or_write(payload: Any, output: Path | None) -> None:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(json.dumps({"written": str(output)}, ensure_ascii=False))
        return
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    manager = CacheManager(
        scope=str(args.cache_scope),
        cache_dir=args.cache_dir,
        policy=CachePolicy(disk_lifecycle="persistent", cleanup_on_exit=False),
    )

    if args.command == "status":
        payload: dict[str, Any] = {
            "cache_status": manager.status(),
            "debug_status": manager.debug_status(),
            "stats": manager.stats(),
        }
        if args.name:
            payload["names"] = {
                name: {
                    "cache_status": manager.status(name),
                    "debug_status": manager.debug_status(name),
                }
                for name in args.name
            }
        _print_or_write(payload, args.output)
        return

    if args.command == "list":
        name = args.name[0] if args.name else None
        entries = manager.list_entries(
            name=name,
            namespace=args.namespace,
            limit=args.limit,
        )
        _print_or_write(entries, args.output)
        return

    if args.command == "on":
        if args.name:
            status = {name: manager.set_enabled(True, name) for name in args.name}
            _print_or_write({"status": status, "global": manager.status()}, args.output)
            return
        _print_or_write({"global": manager.set_enabled(True)}, args.output)
        return

    if args.command == "off":
        if args.name:
            status = {name: manager.set_enabled(False, name) for name in args.name}
            _print_or_write({"status": status, "global": manager.status()}, args.output)
            return
        _print_or_write({"global": manager.set_enabled(False)}, args.output)
        return

    if args.command == "debug-on":
        if args.name:
            status = {name: manager.set_debug(True, name) for name in args.name}
            _print_or_write(
                {"status": status, "global": manager.debug_status()}, args.output
            )
            return
        _print_or_write({"global": manager.set_debug(True)}, args.output)
        return

    if args.command == "debug-off":
        if args.name:
            status = {name: manager.set_debug(False, name) for name in args.name}
            _print_or_write(
                {"status": status, "global": manager.debug_status()}, args.output
            )
            return
        _print_or_write({"global": manager.set_debug(False)}, args.output)
        return

    if args.command == "boost-priority":
        _print_or_write(
            {"priority_boost": manager.boost_priority(float(args.delta))}, args.output
        )
        return

    if args.command == "clear-all":
        manager.clear(scope="both")
        _print_or_write({"removed": "all"}, args.output)
        return

    if args.command == "clear-name":
        if not args.name:
            raise SystemExit("clear-name requires at least one --name")
        removed = 0
        for name in args.name:
            removed += manager.clear_name(name=name, namespace=args.namespace)
        _print_or_write({"removed": removed}, args.output)
        return

    if args.command == "clear-max":
        if args.max_bytes is None:
            raise SystemExit("clear-max requires --max-bytes")
        removed = manager.clear_max(max_bytes=int(args.max_bytes))
        _print_or_write({"removed": removed}, args.output)
        return

    if args.command == "clear-old":
        if args.timestamp is None:
            raise SystemExit("clear-old requires --timestamp")
        removed = manager.clear_old(float(args.timestamp))
        _print_or_write({"removed": removed}, args.output)
        return

    if args.command == "clear-new":
        if args.timestamp is None:
            raise SystemExit("clear-new requires --timestamp")
        removed = manager.clear_new(float(args.timestamp))
        _print_or_write({"removed": removed}, args.output)
        return

    if args.command == "collect-recent":
        if not args.name:
            raise SystemExit("collect-recent requires at least one --name")
        collected = manager.collect_recent(
            names=list(args.name),
            limit_per_name=max(1, int(args.limit_per_name)),
            namespace=args.namespace,
            include_value=bool(args.with_values),
        )
        _print_or_write(collected, args.output)
        return

    if args.command == "install-to-cache":
        if args.input is None:
            raise SystemExit("install-to-cache requires --input")
        snapshot = json.loads(args.input.read_text(encoding="utf-8"))
        installed = manager.install_to_cache(snapshot, target_layers=args.target_layers)
        _print_or_write({"installed": int(installed)}, args.output)
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
