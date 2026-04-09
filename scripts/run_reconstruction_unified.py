#!/usr/bin/env python3
"""Unified reconstruction runner for GN absolute/difference and sparse Bayesian."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
SCRIPTS_PATH = REPO_ROOT / "scripts"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

from common.recon_cli_models import InputMode, ReconstructionMethod
from common.acceleration_profiles import (
    add_acceleration_profile_argument,
    apply_acceleration_profile_overrides,
)
from pyeidors.perf import (
    DEFAULT_ACCELERATION_PROFILE,
    DEFAULT_3D_GEOMETRY_VERSION,
    DEFAULT_CHOLMOD_MAX_MEMORY_GIB,
    DEFAULT_CHOLMOD_MAX_N,
    DEFAULT_FORWARD_BACKEND,
    DEFAULT_MESH_FAMILY,
    DEFAULT_INEXACT_ETA0,
    DEFAULT_INEXACT_ETA_MAX,
    DEFAULT_INEXACT_ETA_MIN,
    DEFAULT_INEXACT_FORCING,
    DEFAULT_INEXACT_MODE,
    DEFAULT_JACOBIAN_BLOCK_CANDIDATES,
    DEFAULT_JACOBIAN_BLOCK_SIZE,
    DEFAULT_JACOBIAN_BLOCK_TUNE,
    DEFAULT_LOWRANK_ENERGY,
    DEFAULT_LOWRANK_METHOD,
    DEFAULT_LOWRANK_MODE,
    DEFAULT_LOWRANK_RANK,
    DEFAULT_PETSC_DEVICE,
    DEFAULT_PRECONDITIONER,
    DEFAULT_ROM_MODE,
    DEFAULT_ROM_RANK_ADAPTIVE,
    DEFAULT_ROM_RANK_GLOBAL,
    DEFAULT_ROM_REFRESH_EVERY,
    DEFAULT_ROM_SNAPSHOT_SOURCE,
    FORWARD_BACKEND_VALUES,
    MESH_FAMILY_VALUES,
    normalize_forward_backend,
    normalize_mesh_family,
    normalize_petsc_device,
    parse_block_size_candidates,
    resolve_experimental_mode,
    resolve_forward_mat_solve,
    resolve_line_search_mode,
    resolve_solver_mode,
)

LOGGER = logging.getLogger("reconstruction_unified")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified PyEIDORS reconstruction CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=[m.value for m in ReconstructionMethod],
        help="Reconstruction method",
    )

    parser.add_argument("--input-dir", type=Path, default=None, help="Directory of CSV files")
    parser.add_argument("--glob", type=str, default="*.csv", help="Glob pattern under input-dir")
    parser.add_argument("--csv", type=Path, action="append", default=None, help="Explicit CSV path(s)")
    parser.add_argument("--include-ad", action="store_true", help="Include *_AD.csv files")

    parser.add_argument(
        "--input-mode",
        choices=[m.value for m in InputMode],
        default=InputMode.PAIRED.value,
        help="paired: reference/target in one CSV, frame: one CSV per frame",
    )
    parser.add_argument("--reference-csv", type=Path, default=None, help="Reference frame CSV (frame mode)")
    parser.add_argument("--reference-index", type=int, default=None, help="Reference index in discovered files")
    parser.add_argument(
        "--frame-layout",
        choices=["auto", "stim-meas", "meas-stim", "vector"],
        default="auto",
        help="Frame CSV layout interpretation in frame mode",
    )

    parser.add_argument("--output-root", type=Path, required=True, help="Output root directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing case outputs")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue batch if a case fails")
    parser.add_argument("--dry-run", action="store_true", help="List resolved cases and exit")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation when supported")

    parser.add_argument("--metadata", type=Path, default=None, help="Metadata YAML/JSON path")
    parser.add_argument("--measurement-gain", type=float, default=10.0, help="Measurement gain divisor")
    parser.add_argument("--use-part", choices=["real", "imag", "mag"], default="real")

    parser.add_argument("--absolute-col", type=int, default=2, help="Column index for absolute reconstruction")
    parser.add_argument("--reference-col", type=int, default=0, help="Reference column index in paired CSV")
    parser.add_argument("--target-col", type=int, default=2, help="Target column index in paired CSV")

    parser.add_argument("--lambda", dest="lam", type=float, default=None, help="Regularization lambda")
    parser.add_argument("--max-iter", type=int, default=None, help="Maximum GN iterations (absolute)")
    parser.add_argument("--background-sigma", type=float, default=None, help="Background conductivity")
    parser.add_argument("--contact-impedance", type=float, default=None, help="Contact impedance")
    parser.add_argument("--refinement", type=int, default=None, help="Mesh refinement level")
    parser.add_argument("--mesh-radius", type=float, default=None, help="Mesh radius used by method")

    parser.add_argument("--mesh-dir", type=Path, default=REPO_ROOT / "eit_meshes", help="Mesh cache directory")
    parser.add_argument("--mesh-name", type=str, default="mesh_16e_r0p025_ref10_cov0p5", help="Mesh cache name")
    parser.add_argument("--mesh-dim", type=int, choices=[2, 3], default=2, help="Mesh geometric dimension")
    parser.add_argument("--mesh-height", type=float, default=1.0, help="3D cylinder height")
    parser.add_argument(
        "--electrode-height-ratio",
        type=float,
        default=0.2,
        help="3D electrode height ratio (reserved for mesh signature)",
    )
    parser.add_argument("--z-center", type=float, default=0.0, help="3D cylinder z-center")
    add_acceleration_profile_argument(
        parser,
        default=DEFAULT_ACCELERATION_PROFILE,
        help_suffix="Only affects 3D runs; low-level flags remain available for advanced overrides.",
    )
    parser.add_argument(
        "--solver-mode",
        choices=["auto", "strict", "fast"],
        default="auto",
        help="Solver mode (auto => 3D fast, 2D strict)",
    )
    parser.add_argument(
        "--linear-solver",
        choices=["auto", "petsc-ksp", "scipy-lsmr", "pyamg-cg", "cholmod"],
        default="auto",
        help="Linear solver backend for fast mode",
    )
    parser.add_argument(
        "--preconditioner",
        choices=["auto", "diag", "pyamg", "cholmod", "petsc-gamg"],
        default=DEFAULT_PRECONDITIONER,
        help="Preconditioner selection for fast iterative paths",
    )
    parser.add_argument(
        "--fast-linear-path",
        choices=["auto", "woodbury", "pcg", "cholmod-direct", "strict"],
        default="auto",
        help="Fast-mode linear path strategy (3D absolute/difference acceleration)",
    )
    parser.add_argument(
        "--rom-mode",
        choices=["off", "auto", "on"],
        default=DEFAULT_ROM_MODE,
        help="Experimental reduced-order model acceleration mode for 3D fast paths",
    )
    parser.add_argument("--rom-rank-global", type=int, default=DEFAULT_ROM_RANK_GLOBAL, help="Global POD basis rank cap")
    parser.add_argument("--rom-rank-adaptive", type=int, default=DEFAULT_ROM_RANK_ADAPTIVE, help="Adaptive low-rank basis rank cap")
    parser.add_argument("--rom-refresh-every", type=int, default=DEFAULT_ROM_REFRESH_EVERY, help="Adaptive ROM refresh interval")
    parser.add_argument(
        "--rom-snapshot-source",
        choices=["cache", "synthetic", "hybrid"],
        default=DEFAULT_ROM_SNAPSHOT_SOURCE,
        help="Source policy for ROM snapshots",
    )
    parser.add_argument(
        "--inexact-mode",
        choices=["off", "auto", "on"],
        default=DEFAULT_INEXACT_MODE,
        help="Experimental inexact GN inner-solve control mode for 3D fast paths",
    )
    parser.add_argument(
        "--inexact-forcing",
        choices=["fixed", "eisenstat-walker"],
        default=DEFAULT_INEXACT_FORCING,
        help="Inexact GN forcing-term policy",
    )
    parser.add_argument("--inexact-eta0", type=float, default=DEFAULT_INEXACT_ETA0, help="Initial inexact forcing eta")
    parser.add_argument("--inexact-eta-min", type=float, default=DEFAULT_INEXACT_ETA_MIN, help="Minimum inexact forcing eta")
    parser.add_argument("--inexact-eta-max", type=float, default=DEFAULT_INEXACT_ETA_MAX, help="Maximum inexact forcing eta")
    parser.add_argument(
        "--lowrank-mode",
        choices=["off", "auto", "on"],
        default=DEFAULT_LOWRANK_MODE,
        help="Experimental low-rank Jacobian subspace mode for fused acceleration",
    )
    parser.add_argument("--lowrank-rank", type=int, default=DEFAULT_LOWRANK_RANK, help="Low-rank subspace rank cap")
    parser.add_argument(
        "--lowrank-method",
        choices=["tsvd", "randomized"],
        default=DEFAULT_LOWRANK_METHOD,
        help="Low-rank subspace extraction method",
    )
    parser.add_argument("--lowrank-energy", type=float, default=DEFAULT_LOWRANK_ENERGY, help="Energy threshold for low-rank subspace")
    parser.add_argument(
        "--cholmod-max-n",
        type=int,
        default=DEFAULT_CHOLMOD_MAX_N,
        help="Max parameter size allowed for CHOLMOD fast path",
    )
    parser.add_argument(
        "--cholmod-max-memory-gib",
        type=float,
        default=DEFAULT_CHOLMOD_MAX_MEMORY_GIB,
        help="Estimated memory guard (GiB) for CHOLMOD fast path",
    )
    parser.add_argument(
        "--jacobian-update-every",
        type=int,
        default=2,
        help="Fast mode Jacobian refresh interval (absolute GN)",
    )
    parser.add_argument(
        "--jacobian-reuse-tol",
        type=float,
        default=1e-3,
        help="Fast mode Jacobian reuse tolerance",
    )
    parser.add_argument(
        "--line-search-mode",
        choices=["auto", "full", "fast"],
        default="auto",
        help="Line-search mode (auto => 3D fast uses fast)",
    )
    parser.add_argument(
        "--jacobian-block-tune",
        choices=["auto", "off"],
        default="auto",
        help="Auto-tune Jacobian element block size",
    )
    parser.add_argument(
        "--jacobian-block-size",
        type=int,
        default=0,
        help="Fixed Jacobian element block size (0 means auto)",
    )
    parser.add_argument(
        "--jacobian-block-candidates",
        type=str,
        default="64,128,256,512",
        help="Comma-separated candidate block sizes for auto-tuning",
    )
    parser.add_argument(
        "--absolute-startup-cache",
        choices=["on", "off"],
        default="on",
        help="Enable startup Jacobian cache for absolute fast mode",
    )
    parser.add_argument(
        "--forward-mat-solve",
        choices=["auto", "off", "on"],
        default="auto",
        help="Control PETSc matSolve multi-RHS path",
    )
    parser.add_argument(
        "--petsc-device",
        choices=["auto", "cpu", "cuda"],
        default=DEFAULT_PETSC_DEVICE,
        help="PETSc/DOLFINx FEM device policy for forward and adjoint solves",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Torch/GN inverse runtime device policy",
    )
    parser.add_argument(
        "--forward-backend",
        choices=list(FORWARD_BACKEND_VALUES),
        default=DEFAULT_FORWARD_BACKEND,
        help=(
            "Forward discretization backend. `cuda_structured` is the single-rank "
            "3D hex CUDA fast path; `dolfinx` remains the default/reference backend."
        ),
    )
    parser.add_argument(
        "--mesh-family",
        choices=list(MESH_FAMILY_VALUES),
        default=DEFAULT_MESH_FAMILY,
        help="3D cell family for generated/cached meshes. GPU-only forward backends require `hex`.",
    )
    parser.add_argument(
        "--geometry-version",
        type=str,
        default=DEFAULT_3D_GEOMETRY_VERSION,
        help="3D generated mesh contract. `geomv2` enables finite-height electrode patches.",
    )
    parser.add_argument(
        "--perf-report",
        type=Path,
        default=None,
        help="Write lightweight performance report JSON after run",
    )
    parser.add_argument(
        "--perf-gate",
        choices=["off", "warn", "strict"],
        default="warn",
        help="Performance gate behavior when --perf-report is enabled",
    )
    parser.add_argument("--n-elec", type=int, default=16, help="Number of electrodes (GN diff)")
    parser.add_argument("--radius", type=float, default=0.025, help="Mesh radius for GN diff")
    parser.add_argument("--drive-value", type=float, default=None, help="Drive value override (GN diff)")
    parser.add_argument("--step-size-calibration", action="store_true", help="Enable one-step alpha search")
    parser.add_argument("--step-size-min", type=float, default=1e-3, help="Lower bound for alpha search")
    parser.add_argument("--step-size-max", type=float, default=1e1, help="Upper bound for alpha search")
    parser.add_argument("--step-size-maxiter", type=int, default=50, help="Max iterations for alpha search")
    parser.add_argument("--colormap", type=str, default="viridis", help="Colormap for reconstruction")
    parser.add_argument("--colorbar-scientific", action="store_true", help="Scientific notation colorbar")
    parser.add_argument(
        "--colorbar-format",
        type=str,
        default=None,
        choices=["plain", "scientific", "matlab_short"],
        help="Colorbar format",
    )
    parser.add_argument("--transparent", action="store_true", help="Save plots with transparent background")

    parser.add_argument("--difference-calibration", choices=["before", "after", "none"], default="after")
    parser.add_argument("--calibration-col", type=int, default=-1)
    parser.add_argument("--prior-scale", type=float, default=None)
    parser.add_argument("--noise-std", type=float, default=None)
    parser.add_argument("--subspace-rank", type=int, default=None)
    parser.add_argument("--coarse-group-size", type=int, default=None)
    parser.add_argument("--coarse-levels", type=int, nargs="+", default=None)
    parser.add_argument("--linear-warm-start", action="store_true")
    parser.add_argument("--solver", choices=["map", "fista", "irls"], default="map")
    parser.add_argument("--linear-max-iters", type=int, default=200)
    parser.add_argument("--linear-tol", type=float, default=1e-6)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--gpu-dtype", type=str, default="float32")
    parser.add_argument("--block-iterations", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--refinement-gradient-tol", type=float, default=1e-8)
    parser.add_argument("--coarse-iterations", type=int, default=20)
    parser.add_argument("--coarse-relaxation", type=float, default=0.7)
    parser.add_argument("--jacobian-cache", action="store_true")
    parser.add_argument("--electrode-coverage", type=float, default=0.5)
    parser.add_argument(
        "--cache-scope",
        choices=["off", "process", "both"],
        default="both",
        help="Cache scope for reconstruction kernels",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / ".pyeidors_cache" / "v2",
        help="Cache root. Disk cache defaults to terminal-session lifecycle and is cleaned when the active dev shell exits.",
    )
    parser.add_argument(
        "--cache-clear-name",
        action="append",
        default=[],
        help="Cache family name to clear before run (repeatable)",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    method = ReconstructionMethod(args.method)
    input_mode = InputMode(args.input_mode)

    if args.input_dir is None and not args.csv:
        parser.error("Provide --input-dir or --csv.")

    if method == ReconstructionMethod.GN_ABSOLUTE:
        if args.metadata is None:
            parser.error("gn-absolute requires --metadata.")
        if args.reference_csv is not None or args.reference_index is not None:
            parser.error("gn-absolute does not accept --reference-csv or --reference-index.")

    if method in {ReconstructionMethod.GN_DIFFERENCE, ReconstructionMethod.SPARSE_BAYES}:
        if input_mode == InputMode.PAIRED and args.reference_col == args.target_col:
            parser.error(
                "In paired mode, --reference-col and --target-col must be different."
            )
        if input_mode == InputMode.FRAME:
            if args.reference_csv is not None and args.reference_index is not None:
                parser.error("Use only one of --reference-csv or --reference-index.")
            if args.reference_csv is None and args.reference_index is None:
                parser.error(
                    "Frame mode for difference/sparse requires --reference-csv or --reference-index."
                )

    mesh_dim = int(args.mesh_dim)
    if mesh_dim == 3 and method == ReconstructionMethod.SPARSE_BAYES:
        parser.error("3D mesh mode currently does not support --method sparse-bayes.")

    apply_acceleration_profile_overrides(args, mesh_dim=mesh_dim)
    args.solver_mode = resolve_solver_mode(args.solver_mode, mesh_dim=mesh_dim)
    args.line_search_mode = resolve_line_search_mode(args.line_search_mode, mesh_dim=mesh_dim)
    args.rom_mode = resolve_experimental_mode(args.rom_mode)
    args.inexact_mode = resolve_experimental_mode(args.inexact_mode)
    args.lowrank_mode = resolve_experimental_mode(args.lowrank_mode)
    args.forward_mat_solve = resolve_forward_mat_solve(
        args.forward_mat_solve,
        mesh_dim=mesh_dim,
        solver_mode=args.solver_mode,
    )
    args.petsc_device = normalize_petsc_device(args.petsc_device, default=DEFAULT_PETSC_DEVICE)
    args.forward_backend = normalize_forward_backend(
        args.forward_backend,
        default=DEFAULT_FORWARD_BACKEND,
    )
    args.mesh_family = normalize_mesh_family(
        args.mesh_family,
        default=DEFAULT_MESH_FAMILY,
    )
    args.geometry_version = str(args.geometry_version).strip().lower() or DEFAULT_3D_GEOMETRY_VERSION
    if int(args.cholmod_max_n) <= 0:
        parser.error("--cholmod-max-n must be positive.")
    if float(args.cholmod_max_memory_gib) <= 0:
        parser.error("--cholmod-max-memory-gib must be positive.")
    if int(args.rom_rank_global) <= 0:
        parser.error("--rom-rank-global must be positive.")
    if int(args.rom_rank_adaptive) < 0:
        parser.error("--rom-rank-adaptive must be >= 0.")
    if int(args.rom_refresh_every) <= 0:
        parser.error("--rom-refresh-every must be positive.")
    if float(args.inexact_eta0) <= 0:
        parser.error("--inexact-eta0 must be positive.")
    if float(args.inexact_eta_min) <= 0 or float(args.inexact_eta_max) <= 0:
        parser.error("--inexact-eta-min/--inexact-eta-max must be positive.")
    if float(args.inexact_eta_min) > float(args.inexact_eta_max):
        parser.error("--inexact-eta-min must be <= --inexact-eta-max.")
    if int(args.lowrank_rank) <= 0:
        parser.error("--lowrank-rank must be positive.")
    if not (0.0 < float(args.lowrank_energy) <= 1.0):
        parser.error("--lowrank-energy must be in (0, 1].")
    if int(args.jacobian_block_size) < 0:
        parser.error("--jacobian-block-size must be >= 0.")
    try:
        args.jacobian_block_candidates = parse_block_size_candidates(args.jacobian_block_candidates)
    except ValueError as exc:
        parser.error(f"--jacobian-block-candidates {exc}")


def _config_snapshot(args: argparse.Namespace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, list) and value and isinstance(value[0], Path):
            payload[key] = [str(v) for v in value]
        else:
            payload[key] = value
    return payload


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _collect_perf_metrics(results) -> dict[str, Any]:
    stage_totals: dict[str, float] = {}
    cache_hit_layers: dict[str, int] = {}
    for result in results:
        metrics = getattr(result, "metrics", None)
        if not isinstance(metrics, dict):
            continue
        stage = metrics.get("stage_timings")
        if isinstance(stage, dict):
            for key, value in stage.items():
                numeric = _safe_float(value)
                if numeric is None:
                    continue
                stage_totals[key] = stage_totals.get(key, 0.0) + numeric
        lookups = metrics.get("cache_lookups")
        if isinstance(lookups, dict):
            context = lookups.get("context")
            if isinstance(context, dict):
                for value in context.values():
                    if isinstance(value, dict):
                        layer = value.get("layer")
                        if isinstance(layer, str):
                            cache_hit_layers[layer] = cache_hit_layers.get(layer, 0) + 1
    return {
        "stage_totals": stage_totals,
        "cache_hit_layers": cache_hit_layers,
    }


def _write_perf_report(path: Path, *, method: ReconstructionMethod, summary_path: Path, results, args) -> dict[str, Any]:
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": method.value,
        "summary_path": str(summary_path),
        "perf_gate": str(args.perf_gate),
        "metrics": _collect_perf_metrics(results),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def run(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )

    validate_args(parser, args)

    method = ReconstructionMethod(args.method)
    input_mode = InputMode(args.input_mode)

    from common.case_discovery import build_cases, collect_csv_files
    from common.output_writer import format_dry_run

    input_files = collect_csv_files(
        input_dir=args.input_dir,
        glob_pattern=args.glob,
        csv_files=args.csv,
        include_ad=bool(args.include_ad),
    )
    if not input_files:
        raise ValueError("No CSV files found for reconstruction.")

    require_reference = method in {
        ReconstructionMethod.GN_DIFFERENCE,
        ReconstructionMethod.SPARSE_BAYES,
    }
    cases = build_cases(
        input_mode=input_mode,
        input_files=input_files,
        require_reference=require_reference,
        reference_csv=args.reference_csv,
        reference_index=args.reference_index,
    )
    if not cases:
        raise ValueError("No reconstruction cases resolved from provided inputs.")

    method_output_root = args.output_root / method.value

    if args.dry_run:
        print(format_dry_run(cases))
        return 0

    from common.method_runners import get_method_runner
    from common.output_writer import write_batch_summary

    if require_reference and input_mode == InputMode.FRAME and cases[0].reference_csv:
        method_output_root.mkdir(parents=True, exist_ok=True)
        (method_output_root / "reference_frame.txt").write_text(
            str(cases[0].reference_csv.resolve()) + "\n",
            encoding="utf-8",
        )

    runner = get_method_runner(method)
    results = runner(cases=cases, output_root=method_output_root, args=args)

    summary_path = write_batch_summary(
        method=method,
        output_root=method_output_root,
        cases=cases,
        results=results,
        config=_config_snapshot(args),
    )

    if args.perf_report is not None:
        perf_report = _write_perf_report(
            args.perf_report,
            method=method,
            summary_path=summary_path,
            results=results,
            args=args,
        )
        stage_totals = perf_report.get("metrics", {}).get("stage_totals", {})
        linear_total = _safe_float(stage_totals.get("linear_solve"))
        if linear_total is not None and linear_total > 0 and args.perf_gate in {"warn", "strict"}:
            threshold = 120.0
            if linear_total > threshold:
                message = (
                    f"performance gate exceeded: linear_solve total {linear_total:.3f}s > {threshold:.3f}s"
                )
                if args.perf_gate == "strict":
                    raise RuntimeError(message)
                LOGGER.warning(message)

    processed = sum(1 for result in results if result.status == "success")
    skipped = sum(1 for result in results if result.status == "skipped")
    failed = sum(1 for result in results if result.status == "failed")
    LOGGER.info(
        "Completed method=%s processed=%d skipped=%d failed=%d summary=%s",
        method.value,
        processed,
        skipped,
        failed,
        summary_path,
    )

    return 1 if failed else 0


def main() -> None:
    try:
        code = run()
    except Exception as exc:  # pragma: no cover - user-facing CLI behavior
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    raise SystemExit(code)


if __name__ == "__main__":
    main()
