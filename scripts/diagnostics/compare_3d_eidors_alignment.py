#!/usr/bin/env python3
"""Compare 3D PyEIDORS reconstructions against EIDORS-aligned layouts/modes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for candidate in (str(REPO_ROOT), str(SCRIPTS_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from pyeidors.perf import DEFAULT_ACCELERATION_PROFILE

from render_3d_inverse_reconstruction_overview import (
    DEFAULT_ABSOLUTE_PRESET,
    DEFAULT_DIFFERENCE_PRESET,
    DEFAULT_OUTPUT_DIR,
    _configure_times_new_roman,
    run_case,
)
from common.acceleration_profiles import add_acceleration_profile_argument

DEFAULT_LEVEL_FRACTIONS = (0.25, 0.75)


def _shape_score(metrics: dict[str, object]) -> float:
    shape_metrics = metrics.get("shape_metrics", {})
    if not isinstance(shape_metrics, dict):
        return float("inf")
    truth = shape_metrics.get("truth", {})
    recon = shape_metrics.get("reconstruction", {})
    if not isinstance(truth, dict) or not isinstance(recon, dict):
        return float("inf")
    truth_z_ratio = float(truth.get("z_to_xy_mean_ratio", float("nan")))
    recon_z_ratio = float(recon.get("z_to_xy_mean_ratio", float("nan")))
    truth_xy_ratio = float(truth.get("xy_aspect_ratio", float("nan")))
    recon_xy_ratio = float(recon.get("xy_aspect_ratio", float("nan")))
    return abs(recon_z_ratio - truth_z_ratio) + abs(recon_xy_ratio - truth_xy_ratio)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare 3D EIDORS-aligned reconstruction settings")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "eidors_alignment_suite")
    parser.add_argument("--refinement", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--radius", type=float, default=0.22)
    parser.add_argument("--height", type=float, default=0.16)
    add_acceleration_profile_argument(
        parser,
        default=DEFAULT_ACCELERATION_PROFILE,
        help_suffix="Forwarded to each 3D reconstruction case.",
    )
    parser.add_argument("--difference-mode", choices=["raw", "normalized"], default="normalized")
    parser.add_argument(
        "--difference-orientation",
        choices=["target_minus_reference", "reference_minus_target"],
        default="target_minus_reference",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip per-case 3D plot construction/export and only keep numerical comparisons.",
    )
    parser.add_argument(
        "--no-save-data",
        action="store_true",
        help="Skip writing per-case JSON/NPZ outputs and final alignment_summary.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_times_new_roman()

    cases = [
        (
            "difference_eidors_one_step_noser",
            "difference",
            DEFAULT_LEVEL_FRACTIONS,
            DEFAULT_DIFFERENCE_PRESET,
            DEFAULT_ABSOLUTE_PRESET,
            args.max_iterations,
        ),
        (
            "difference_sphere_multistep_noser",
            "difference",
            DEFAULT_LEVEL_FRACTIONS,
            "sphere_multistep_noser",
            DEFAULT_ABSOLUTE_PRESET,
            3 if args.max_iterations is None else args.max_iterations,
        ),
        (
            "difference_eidors_demo3d_tv",
            "difference",
            DEFAULT_LEVEL_FRACTIONS,
            "eidors_demo3d_tv",
            DEFAULT_ABSOLUTE_PRESET,
            args.max_iterations,
        ),
        (
            "absolute_eidors_abs_gn",
            "absolute",
            DEFAULT_LEVEL_FRACTIONS,
            DEFAULT_DIFFERENCE_PRESET,
            DEFAULT_ABSOLUTE_PRESET,
            3 if args.max_iterations is None else args.max_iterations,
        ),
    ]

    summary: list[dict[str, object]] = []
    for (
        case_name,
        inverse_mode,
        level_fractions,
        difference_preset,
        absolute_preset,
        max_iterations,
    ) in cases:
        case_output = args.output_dir / case_name
        metrics = run_case(
            output_dir=case_output,
            refinement=args.refinement,
            max_iterations=max_iterations,
            radius=args.radius,
            height=args.height,
            inverse_mode=inverse_mode,
            difference_mode=args.difference_mode,
            difference_orientation=args.difference_orientation,
            electrode_level_fractions=level_fractions,
            difference_preset=difference_preset,
            absolute_preset=absolute_preset,
            hyperparameter=None,
            difference_step_size_mode=None,
            best_homog_mode=None,
            acceleration_profile=args.acceleration_profile,
            render_plot=not args.no_plot,
            save_data=not args.no_save_data,
        )
        summary.append(
            {
                "case": case_name,
                "inverse_mode": inverse_mode,
                "preset_name": metrics.get("preset_name"),
                "conductivity_rmse": metrics.get("conductivity_rmse"),
                "conductivity_correlation": metrics.get("conductivity_correlation"),
                "voltage_rmse": metrics.get("voltage_rmse"),
                "residual_l2": metrics.get("residual_l2"),
                "contrast_recovery": metrics.get("contrast_recovery"),
                "target_mean": metrics.get("target_mean"),
                "background_mean": metrics.get("background_mean"),
                "peak_conductivity": metrics.get("peak_conductivity"),
                "step_size": metrics.get("step_size"),
                "shape_score": _shape_score(metrics),
                "wall_time_breakdown": metrics.get("wall_time_breakdown", {}),
                "output_dir": str(case_output),
            }
        )

    print(json.dumps(summary, indent=2))
    if not args.no_save_data:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.output_dir / "alignment_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
