#!/usr/bin/env python3
"""Benchmark GUI-style 3D forward setup-prime and first-load solve timings."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SCHEMA = "eit_gui_forward_first_load_benchmark_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["setup", "solve", "both"],
        default="setup",
        help="Run setup-prime only, full forward solve only, or both.",
    )
    parser.add_argument(
        "--profile",
        default="cuda",
        help="Backend worker profile to use, e.g. cuda or complex64-cuda.",
    )
    parser.add_argument("--mesh-dimension", type=int, default=3)
    parser.add_argument("--mesh-refinement", type=float, default=0.25)
    parser.add_argument("--n-electrodes", type=int, default=16)
    parser.add_argument("--n-rings", type=int, default=2)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--height", type=float, default=1.0)
    parser.add_argument("--z-center", type=float, default=0.0)
    parser.add_argument("--electrode-coverage", type=float, default=0.5)
    parser.add_argument("--electrode-height-ratio", type=float, default=0.2)
    parser.add_argument(
        "--electrode-level-fractions",
        default="0.25,0.75",
        help="Comma-separated 3D electrode level fractions.",
    )
    parser.add_argument("--electrode-layout", default="ring_major")
    parser.add_argument("--measurement-protocol", default="eidors_full_3d")
    parser.add_argument("--stim-pattern", default="{ad}")
    parser.add_argument("--meas-pattern", default="{ad}")
    parser.add_argument("--mesh-family", default="tetra")
    parser.add_argument("--geometry-version", default="geomv2")
    parser.add_argument("--forward-backend", default="dolfinx")
    parser.add_argument("--acceleration-profile", default="gpu3d")
    parser.add_argument("--forward-solver-preset", default="auto")
    parser.add_argument("--forward-mat-solve", default="auto")
    parser.add_argument("--petsc-device", default="cuda")
    parser.add_argument("--background-conductivity", type=float, default=1.0)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--request-out", type=Path, default=None)
    parser.add_argument("--repair-jit", action="store_true")
    parser.add_argument(
        "--prewarm-worker",
        action="store_true",
        help=(
            "Run import/capability worker warm before setup/solve so the "
            "measurement reflects a GUI worker-prewarmed click."
        ),
    )
    parser.add_argument(
        "--progress-message-limit",
        type=int,
        default=200,
        help="Maximum progress messages to keep in the JSON report.",
    )
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def _float_list(text: str) -> list[float]:
    values = [item.strip() for item in str(text or "").split(",")]
    parsed = [float(item) for item in values if item]
    return parsed or [0.25, 0.75]


def build_forward_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mesh_dimension": int(args.mesh_dimension),
        "mesh_refinement": float(args.mesh_refinement),
        "n_elec": int(args.n_electrodes),
        "n_electrodes": int(args.n_electrodes),
        "n_rings": int(args.n_rings),
        "electrode_layout": str(args.electrode_layout),
        "measurement_protocol": str(args.measurement_protocol),
        "stim_pattern": str(args.stim_pattern),
        "meas_pattern": str(args.meas_pattern),
        "rotate_meas": True,
        "use_meas_current": False,
        "use_meas_current_next": 0,
        "background_conductivity": float(args.background_conductivity),
        "noise_level": float(args.noise_level),
        "radius": float(args.radius),
        "height": float(args.height),
        "electrode_coverage": float(args.electrode_coverage),
        "electrode_height_ratio": float(args.electrode_height_ratio),
        "electrode_level_fractions": _float_list(args.electrode_level_fractions),
        "z_center": float(args.z_center),
        "mesh_family": str(args.mesh_family),
        "geometry_version": str(args.geometry_version),
        "forward_backend": str(args.forward_backend),
        "acceleration_profile": str(args.acceleration_profile),
        "forward_solver_preset": str(args.forward_solver_preset),
        "forward_mat_solve": str(args.forward_mat_solve),
        "petsc_device": str(args.petsc_device),
    }


def build_forward_request(args: argparse.Namespace):
    from eit_app.controllers.forward_solver_controller import ForwardSolverRequest

    return ForwardSolverRequest(
        mesh_dimension=int(args.mesh_dimension),
        mesh_refinement=float(args.mesh_refinement),
        n_electrodes=int(args.n_electrodes),
        background_conductivity=float(args.background_conductivity),
        inhomogeneities=[],
        noise_level=float(args.noise_level),
        forward_model_config=build_forward_config(args),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            pass
    return value


def _timing_from_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    timings = dict(mapping.get("forward_timing_ms") or {})
    return {
        "timing_schema": str(mapping.get("forward_timing_schema", "")),
        "timing_ms": timings,
        "phase_order": list(mapping.get("forward_timing_phase_order") or []),
        "total_ms": float(mapping.get("forward_timing_total_ms", 0.0) or 0.0),
    }


def _run_setup_prime(
    *,
    args: argparse.Namespace,
    request_path: Path,
    repair_jit: bool | None = None,
) -> dict[str, Any]:
    from pyeidors.cache.ops import warm_backend_worker

    started = time.perf_counter()
    report = warm_backend_worker(
        repo=REPO_ROOT,
        profile=str(args.profile),
        repair_jit=bool(args.repair_jit if repair_jit is None else repair_jit),
        forward_request=request_path,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    prime_metadata = dict(report.get("prime_metadata") or {})
    return {
        "elapsed_ms": elapsed_ms,
        "report": report,
        "timing": _timing_from_mapping(prime_metadata),
    }


def _run_worker_prewarm(*, args: argparse.Namespace) -> dict[str, Any]:
    from pyeidors.cache.ops import warm_backend_worker

    started = time.perf_counter()
    report = warm_backend_worker(
        repo=REPO_ROOT,
        profile=str(args.profile),
        repair_jit=bool(args.repair_jit),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "elapsed_ms": elapsed_ms,
        "report": report,
    }


def _run_solve(*, args: argparse.Namespace, request: Any) -> dict[str, Any]:
    from eit_app.controllers.forward_solver_controller import (
        execute_forward_request_in_backend,
    )
    from pyeidors.cache.ops import BoundedProgressMessageCollector

    messages = BoundedProgressMessageCollector(
        limit=int(args.progress_message_limit),
    )
    started = time.perf_counter()
    result = execute_forward_request_in_backend(
        request,
        profile=str(args.profile),
        route_reason="diagnostic_forward_first_load",
        progress_cb=messages.append,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    config = dict(getattr(result, "forward_model_config", {}) or {})
    return {
        "elapsed_ms": elapsed_ms,
        **messages.report_fields(),
        "n_elements": int(getattr(result, "n_elements", 0) or 0),
        "n_measurements": int(getattr(result, "n_measurements", 0) or 0),
        "timing": _timing_from_mapping(config),
        "backend": {
            "profile": config.get("backend_worker_profile", str(args.profile)),
            "pid": config.get("backend_worker_pid", 0),
            "reused_process": config.get("backend_worker_reused_process", False),
            "rss_bytes": config.get("backend_worker_rss_bytes", 0),
            "request_duration_ms": config.get("backend_worker_request_duration_ms", 0),
            "result_read_ms": config.get("backend_worker_result_read_ms", 0),
        },
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from eit_app.backend_worker_protocol import write_forward_request

    request = build_forward_request(args)
    with tempfile.TemporaryDirectory(prefix="pyeidors-forward-first-load-") as tmp:
        request_path = args.request_out or Path(tmp) / "forward_request.h5"
        request_path.parent.mkdir(parents=True, exist_ok=True)
        write_forward_request(request_path, request)

        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "mode": str(args.mode),
            "profile": str(args.profile),
            "prewarm_worker": bool(args.prewarm_worker),
            "request_path": str(request_path),
            "request": build_forward_config(args),
        }
        if args.prewarm_worker:
            payload["worker_prewarm"] = _run_worker_prewarm(args=args)
        if args.mode in {"setup", "both"}:
            payload["setup_prime"] = _run_setup_prime(
                args=args,
                request_path=request_path,
                repair_jit=False if args.prewarm_worker else None,
            )
        if args.mode in {"solve", "both"}:
            payload["solve"] = _run_solve(args=args, request=request)
        return _json_safe(payload)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_benchmark(args)
    text = json.dumps(payload, indent=2 if args.pretty else None, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(json.dumps({"written": str(args.output)}, ensure_ascii=False))
        return
    print(text)


if __name__ == "__main__":
    main()
