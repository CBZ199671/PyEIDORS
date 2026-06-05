"""Command-line cache operations for PyEIDORS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any, Sequence

from . import CacheManager, CachePolicy
from .ops import (
    build_forward_setup_warm_request,
    cache_manager_status,
    doctor_cache,
    gc_cache,
    parse_size_bytes,
    summarize_gui_array_geometry_cache,
    summarize_import_health,
    summarize_backend_worker_caches,
    warm_backend_worker,
)
from pyeidors.runtime_paths import pyeidors_cache_path


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "doctor",
            "stats",
            "gc",
            "warm",
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
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument(
        "--cache-scope", choices=["off", "process", "both"], default="both"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=pyeidors_cache_path("v2"),
        help="Persistent cache root to inspect/manage.",
    )
    parser.add_argument("--name", action="append", default=[], help="Cache family name")
    parser.add_argument("--namespace", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--n-elec", type=int, default=None)
    parser.add_argument("--mesh-hash", type=str, default=None)
    parser.add_argument("--max-bytes", type=int, default=None)
    parser.add_argument("--max-size", type=str, default=None)
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
    parser.add_argument(
        "--repair-jit",
        action="store_true",
        help="Repair stale FFCx JIT lock files in backend worker caches.",
    )
    parser.add_argument(
        "--include-worker-cache",
        action="store_true",
        help="Allow gc to trim profile-scoped GUI backend worker caches.",
    )
    parser.add_argument(
        "--include-legacy-arrays",
        action="store_true",
        help="Allow gc to remove legacy .npz/.npy files under .pyeidors_cache.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--profile",
        default="default",
        help="Backend worker profile for warm, e.g. default, cuda, complex64-cuda.",
    )
    parser.add_argument(
        "--forward-request",
        type=Path,
        default=None,
        help=(
            "Forward request HDF5/JSON to setup-prime in the backend worker "
            "instead of import-only warm."
        ),
    )
    parser.add_argument(
        "--forward-setup",
        action="store_true",
        help="Generate a GUI-style forward setup request for warmup.",
    )
    parser.add_argument("--mesh-refinement", type=float, default=None)
    parser.add_argument("--n-rings", type=int, default=None)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument("--electrode-coverage", type=float, default=0.5)
    parser.add_argument("--electrode-height-ratio", type=float, default=0.2)
    parser.add_argument(
        "--electrode-level-fractions",
        default=None,
        help="Comma-separated 3D electrode level fractions.",
    )
    parser.add_argument("--electrode-layout", default=None)
    parser.add_argument("--measurement-protocol", default=None)
    parser.add_argument("--stim-pattern", default="{ad}")
    parser.add_argument("--meas-pattern", default="{ad}")
    parser.add_argument("--mesh-family", default=None)
    parser.add_argument("--geometry-version", default=None)
    parser.add_argument("--forward-backend", default="dolfinx")
    parser.add_argument("--acceleration-profile", default=None)
    parser.add_argument("--forward-solver-preset", default="auto")
    parser.add_argument("--forward-mat-solve", default="auto")
    parser.add_argument("--petsc-device", default="auto")
    parser.add_argument("--background-conductivity", type=float, default=1.0)
    parser.add_argument("--noise-level", type=float, default=0.0)
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


def _manager_from_args(args: argparse.Namespace) -> CacheManager:
    return CacheManager(
        scope=str(args.cache_scope),
        cache_dir=args.cache_dir,
        policy=CachePolicy(disk_lifecycle="persistent", cleanup_on_exit=False),
    )


def _float_list(text: str | None) -> list[float] | None:
    if text is None:
        return None
    values = [item.strip() for item in str(text or "").split(",")]
    parsed = [float(item) for item in values if item]
    return parsed or None


def _warm_generates_forward_setup(args: argparse.Namespace) -> bool:
    if args.command != "warm" or args.forward_request is not None:
        return False
    return bool(
        args.forward_setup
        or args.dim is not None
        or args.n_elec is not None
        or args.mesh_refinement is not None
    )


def _warm_generated_forward_setup(args: argparse.Namespace) -> dict[str, Any]:
    from eit_app.backend_worker_protocol import (
        forward_request_to_payload,
        write_forward_request,
    )

    request = build_forward_setup_warm_request(
        dim=int(args.dim if args.dim is not None else 3),
        mesh_refinement=args.mesh_refinement,
        n_elec=int(args.n_elec if args.n_elec is not None else 16),
        n_rings=args.n_rings,
        radius=float(args.radius),
        height=args.height,
        electrode_coverage=float(args.electrode_coverage),
        electrode_height_ratio=float(args.electrode_height_ratio),
        electrode_level_fractions=_float_list(args.electrode_level_fractions),
        electrode_layout=args.electrode_layout,
        measurement_protocol=args.measurement_protocol,
        stim_pattern=str(args.stim_pattern),
        meas_pattern=str(args.meas_pattern),
        mesh_family=args.mesh_family,
        geometry_version=args.geometry_version,
        forward_backend=str(args.forward_backend),
        acceleration_profile=args.acceleration_profile,
        forward_solver_preset=str(args.forward_solver_preset),
        forward_mat_solve=str(args.forward_mat_solve),
        petsc_device=str(args.petsc_device),
        background_conductivity=float(args.background_conductivity),
        noise_level=float(args.noise_level),
    )
    with tempfile.TemporaryDirectory(prefix="pyeidors-cache-warm-forward-") as tmp:
        request_path = Path(tmp) / "forward_request.h5"
        write_forward_request(request_path, request)
        report = warm_backend_worker(
            repo=args.repo,
            profile=str(args.profile),
            repair_jit=bool(args.repair_jit),
            forward_request=request_path,
        )
    report["generated_forward_request"] = True
    report["generated_forward_request_payload"] = forward_request_to_payload(request)
    return report


def _resolved_max_bytes(args: argparse.Namespace) -> int:
    if args.max_size is not None:
        return parse_size_bytes(args.max_size)
    if args.max_bytes is not None:
        return parse_size_bytes(args.max_bytes)
    raise SystemExit("gc/clear-max requires --max-size or --max-bytes")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        _print_or_write(
            doctor_cache(
                repo=args.repo,
                cache_dir=args.cache_dir,
                repair_jit=bool(args.repair_jit),
            ),
            args.output,
        )
        return

    if args.command == "stats":
        _print_or_write(
            {
                "cache_manager": cache_manager_status(cache_dir=args.cache_dir),
                "backend_workers": summarize_backend_worker_caches(repo=args.repo),
                "import_health": summarize_import_health(repo=args.repo),
                "gui_array_geometry_cache": summarize_gui_array_geometry_cache(),
            },
            args.output,
        )
        return

    if args.command == "gc":
        _print_or_write(
            gc_cache(
                repo=args.repo,
                cache_dir=args.cache_dir,
                max_bytes=_resolved_max_bytes(args),
                include_worker_cache=bool(args.include_worker_cache),
                include_legacy_arrays=bool(args.include_legacy_arrays),
                dry_run=bool(args.dry_run),
            ),
            args.output,
        )
        return

    if args.command == "warm":
        if _warm_generates_forward_setup(args):
            _print_or_write(_warm_generated_forward_setup(args), args.output)
            return
        _print_or_write(
            warm_backend_worker(
                repo=args.repo,
                profile=str(args.profile),
                repair_jit=bool(args.repair_jit),
                forward_request=args.forward_request,
            ),
            args.output,
        )
        return

    manager = _manager_from_args(args)

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
            dtype=args.dtype,
            backend=args.backend,
            device=args.device,
            dim=args.dim,
            n_elec=args.n_elec,
            mesh_hash=args.mesh_hash,
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
        removed = manager.clear_max(max_bytes=_resolved_max_bytes(args))
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
