#!/usr/bin/env python3
"""Build a PyEIDORS mesh-matched control table against the EIDORS-style mesh size."""

from __future__ import annotations

import argparse
import csv
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pyeidors import EITSystem  # noqa: E402
from pyeidors.data.structures import EITImage, PatternConfig  # noqa: E402
from pyeidors.data.synthetic_data import create_custom_phantom  # noqa: E402
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh  # noqa: E402

from benchmark_difference_runtime import (  # noqa: E402
    build_single_step_namespace,
    compute_difference,
    run_single_step_benchmark,
)
from benchmark_reviewer_case import conductivity_metrics, get_git_commit, voltage_metrics  # noqa: E402


SCENARIO_CONFIG = {
    "low_z": {
        "contact_impedance": 1e-6,
        "background": 1.0,
        "phantom_conductivity": 2.0,
        "phantom_center": (0.30, 0.20),
        "phantom_radius": 0.20,
    },
    "high_z": {
        "contact_impedance": 1e-2,
        "background": 1.0,
        "phantom_conductivity": 2.0,
        "phantom_center": (0.25, -0.22),
        "phantom_radius": 0.18,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--mesh-dir", type=Path, default=Path("eit_meshes"))
    parser.add_argument("--scenario", choices=["low_z", "high_z"], default="low_z")
    parser.add_argument("--matched-refinement", type=int, default=5)
    parser.add_argument("--finer-refinement", type=int, default=10)
    parser.add_argument("--reference-eidors-elements", type=int, default=2130)
    parser.add_argument("--n-elec", type=int, default=16)
    return parser.parse_args()


def get_peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def make_pattern_config(n_elec: int) -> PatternConfig:
    return PatternConfig(
        n_elec=n_elec,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        amplitude=1.0,
        rotate_meas=True,
    )


def run_case(args: argparse.Namespace, refinement: int, label: str) -> dict[str, object]:
    cfg = SCENARIO_CONFIG[args.scenario]
    pattern = make_pattern_config(args.n_elec)
    mesh = load_or_create_mesh(
        mesh_dir=str(args.mesh_dir),
        n_elec=args.n_elec,
        refinement=refinement,
        radius=1.0,
        electrode_coverage=0.5,
    )
    system = EITSystem(
        n_elec=args.n_elec,
        pattern_config=pattern,
        contact_impedance=np.ones(args.n_elec) * cfg["contact_impedance"],
        base_conductivity=cfg["background"],
        regularization_type="noser",
        regularization_alpha=1.0,
        noser_exponent=0.5,
    )
    system.setup(mesh=mesh)
    baseline_image = system.create_homogeneous_image(conductivity=cfg["background"])
    sigma = create_custom_phantom(
        system.fwd_model,
        background_conductivity=cfg["background"],
        anomalies=[{
            "center": tuple(cfg["phantom_center"]),
            "radius": cfg["phantom_radius"],
            "conductivity": cfg["phantom_conductivity"],
        }],
    )
    truth_image = EITImage(elem_data=sigma.vector()[:], fwd_model=system.fwd_model)
    truth_sigma = truth_image.elem_data.copy()
    baseline_meas, _ = system.fwd_model.fwd_solve(baseline_image)
    target_meas, _ = system.fwd_model.fwd_solve(truth_image)
    diff_vector = compute_difference(target_meas.meas, baseline_meas.meas, False)
    single_step_args = build_single_step_namespace(
        system,
    )
    t0 = time.perf_counter()
    recon_image, predicted_diff, step_size = run_single_step_benchmark(
        system,
        baseline_image,
        baseline_meas.meas,
        target_meas.meas,
        single_step_args,
        target_diff=diff_vector,
    )
    runtime = time.perf_counter() - t0
    row: dict[str, object] = {
        "study": "mesh_matched_control",
        "label": label,
        "framework": "pyeidors",
        "task": "difference",
        "mesh_name": getattr(mesh, "mesh_name", f"ref{refinement}"),
        "refinement": refinement,
        "scenario": args.scenario,
        "n_nodes": int(mesh.num_vertices()),
        "n_elements": int(len(system.fwd_model.V_sigma.dofmap().dofs())),
        "reference_eidors_elements": int(args.reference_eidors_elements),
        "element_gap": int(len(system.fwd_model.V_sigma.dofmap().dofs()) - args.reference_eidors_elements),
        "runtime_sec": float(runtime),
        "peak_rss_mb": get_peak_rss_mb(),
        "optimal_step_size": float(step_size),
        "commit": get_git_commit(),
    }
    row.update(voltage_metrics(diff_vector, predicted_diff))
    row.update(conductivity_metrics(truth_sigma, recon_image.elem_data))
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "study",
        "label",
        "framework",
        "task",
        "mesh_name",
        "refinement",
        "scenario",
        "n_nodes",
        "n_elements",
        "reference_eidors_elements",
        "element_gap",
        "runtime_sec",
        "peak_rss_mb",
        "optimal_step_size",
        "voltage_rmse",
        "voltage_mae",
        "voltage_relative_error_pct",
        "conductivity_mae",
        "conductivity_rmse",
        "conductivity_relative_error_pct",
        "commit",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = [
        run_case(args, args.matched_refinement, "matched"),
        run_case(args, args.finer_refinement, "finer"),
    ]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()
