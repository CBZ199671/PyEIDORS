#!/usr/bin/env python3
"""Export a PyEIDORS native geometry in the standard interop format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
BENCHMARK_SCRIPT_DIR = REPO_ROOT / "scripts" / "benchmarks"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(BENCHMARK_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_SCRIPT_DIR))

from pyeidors import EITSystem
from pyeidors.data.structures import EITImage, PatternConfig
from pyeidors.data.synthetic_data import create_custom_phantom
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.interop import (
    STANDARD_INTEROP_FORMAT,
    build_boundary_edges,
    build_electrode_arrays,
    export_forward_csv,
    save_exchange_mat,
)

from benchmark_reviewer_case import PYEIDORS_REFINEMENTS, SCENARIO_CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-mat", type=Path, required=True)
    parser.add_argument("--forward-export-csv", type=Path, default=None)
    parser.add_argument("--mesh-level", choices=["coarse", "medium", "fine"], default="medium")
    parser.add_argument("--scenario", choices=["low_z", "high_z"], default="low_z")
    parser.add_argument("--n-elec", type=int, default=16)
    parser.add_argument("--mesh-dir", type=Path, default=Path("eit_meshes"))
    parser.add_argument("--electrode-coverage", type=float, default=0.5)
    return parser.parse_args()


def make_pattern_config(n_elec: int) -> PatternConfig:
    return PatternConfig(
        n_elec=n_elec,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        amplitude=1.0,
        rotate_meas=True,
    )


def main() -> None:
    args = parse_args()
    args.output_mat.parent.mkdir(parents=True, exist_ok=True)

    cfg = SCENARIO_CONFIG[args.scenario]
    mesh = load_or_create_mesh(
        mesh_dir=str(args.mesh_dir),
        n_elec=args.n_elec,
        refinement=PYEIDORS_REFINEMENTS[args.mesh_level],
        radius=1.0,
        electrode_coverage=args.electrode_coverage,
    )
    system = EITSystem(
        n_elec=args.n_elec,
        pattern_config=make_pattern_config(args.n_elec),
        contact_impedance=np.ones(args.n_elec, dtype=float) * cfg["contact_impedance"],
        base_conductivity=cfg["background"],
        regularization_type="noser",
        regularization_alpha=1.0,
        noser_exponent=0.5,
    )
    system.setup(mesh=mesh, initialize_default_reconstructor=False)

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

    baseline_data = system.forward_solve(baseline_image)
    phantom_data = system.forward_solve(truth_image)
    baseline = np.asarray(baseline_data.meas, dtype=float).reshape(-1)
    phantom = np.asarray(phantom_data.meas, dtype=float).reshape(-1)

    if args.forward_export_csv is not None:
        export_forward_csv(args.forward_export_csv, baseline, phantom)

    nodes = np.asarray(mesh.coordinates(), dtype=float)
    elems = np.asarray(mesh.cells(), dtype=np.int64) + 1
    boundary_edges = build_boundary_edges(mesh)
    electrode_nodes, electrode_counts = build_electrode_arrays(mesh)

    payload = {
        "exchange_format": STANDARD_INTEROP_FORMAT,
        "source_framework": "pyeidors",
        "nodes": nodes,
        "elems": elems,
        "boundary_edges": boundary_edges,
        "electrode_nodes": electrode_nodes,
        "electrode_node_counts": electrode_counts,
        "n_elec": int(args.n_elec),
        "background": float(cfg["background"]),
        "truth_elem_data": np.asarray(truth_image.elem_data, dtype=float).reshape(-1),
        "contact_impedance": float(cfg["contact_impedance"]),
        "mesh_level": args.mesh_level,
        "scenario_name": args.scenario,
        "electrode_coverage": float(args.electrode_coverage),
        "mesh_name": getattr(mesh, "mesh_name", f"ref{PYEIDORS_REFINEMENTS[args.mesh_level]}"),
    }
    save_exchange_mat(args.output_mat, payload)
    print(f"Wrote {args.output_mat}")


if __name__ == "__main__":
    main()
