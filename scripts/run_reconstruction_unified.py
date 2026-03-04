#!/usr/bin/env python3
"""Unified reconstruction runner for GN absolute/difference and sparse Bayesian."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
SCRIPTS_PATH = REPO_ROOT / "scripts"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

from common.case_discovery import build_cases, collect_csv_files
from common.method_runners import get_method_runner
from common.output_writer import format_dry_run, write_batch_summary
from common.recon_cli_models import InputMode, ReconstructionMethod

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
