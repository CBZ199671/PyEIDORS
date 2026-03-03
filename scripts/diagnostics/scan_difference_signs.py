#!/usr/bin/env python3
"""Scan reference/target sign combinations for difference reconstruction."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

SIGN_OPTIONS = ["keep", "flip"]
ORIENT_OPTIONS = ["target_minus_reference", "reference_minus_target"]


def run_case(args, ref_sign, tgt_sign, orientation, idx):
    output_dir = args.output_root / f"case_{idx:02d}_ref{ref_sign}_tgt{tgt_sign}_{orientation}"
    cmd = [
        "python", str(REPO_ROOT / "scripts" / "run_difference_single_step.py"),
        "--csv", str(args.csv),
        "--metadata", str(args.metadata),
        "--use-cols", str(args.use_cols[0]), str(args.use_cols[1]),
        "--reference-frame", str(args.reference_frame),
        "--target-frame", str(args.target_frame),
        "--reference-sign", ref_sign,
        "--target-sign", tgt_sign,
        "--difference-mode", args.difference_mode,
        "--diff-orientation", orientation,
        "--mesh-name", args.mesh_name,
        "--mesh-radius", str(args.mesh_radius),
        "--electrode-coverage", str(args.electrode_coverage),
        "--figure-dpi", str(args.figure_dpi),
        "--output-dir", str(output_dir),
    ]
    if args.strict_eidors:
        cmd.append("--strict-eidors")
    if args.step_size_calibration:
        cmd.append("--step-size-calibration")
    if args.stim_direction:
        cmd.extend(["--stim-direction", args.stim_direction])
    if args.meas_direction:
        cmd.extend(["--meas-direction", args.meas_direction])
    if args.stim_first == "positive":
        cmd.append("--stim-first-positive")
    elif args.stim_first == "negative":
        cmd.append("--stim-first-negative")
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        print(f"Case {idx} failed", file=sys.stderr)
    return output_dir


def read_metrics(path: Path, run_stem: str) -> float:
    metrics_path = path / run_stem / "metrics.json"
    if not metrics_path.exists():
        return float("inf")
    data = json.loads(metrics_path.read_text())
    try:
        return float(data["difference"]["rmse"])
    except Exception:
        return float("inf")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--use-cols", type=int, nargs=2, default=[0, 2])
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument("--target-frame", type=int, default=1)
    parser.add_argument("--difference-mode", choices=["difference", "normalized"], default="difference")
    parser.add_argument("--mesh-name", type=str, default="tank_60mm")
    parser.add_argument("--mesh-radius", type=float, default=0.03)
    parser.add_argument("--electrode-coverage", type=float, default=0.2)
    parser.add_argument("--figure-dpi", type=int, default=300)
    parser.add_argument("--strict-eidors", action="store_true")
    parser.add_argument("--step-size-calibration", action="store_true")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stim-direction", choices=["ccw", "cw"], help="Override stimulation direction")
    parser.add_argument("--meas-direction", choices=["ccw", "cw"], help="Override measurement direction")
    parser.add_argument("--stim-first", choices=["positive", "negative"], help="Set polarity order for first electrode")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    results = []
    idx = 0
    for ref_sign, tgt_sign, orientation in itertools.product(SIGN_OPTIONS, SIGN_OPTIONS, ORIENT_OPTIONS):
        idx += 1
        out_dir = run_case(args, ref_sign, tgt_sign, orientation, idx)
        rmse = read_metrics(out_dir, args.csv.stem)
        results.append((rmse, ref_sign, tgt_sign, orientation, out_dir))

    results.sort(key=lambda x: x[0])
    summary = [
        {
            "rmse": r,
            "reference_sign": ref,
            "target_sign": tgt,
            "orientation": orient,
            "output_dir": str(out_dir),
        }
        for r, ref, tgt, orient, out_dir in results
    ]
    (args.output_root / "scan_summary.json").write_text(json.dumps(summary, indent=2))
    print("Scan summary saved to", args.output_root / "scan_summary.json")


if __name__ == "__main__":
    main()
