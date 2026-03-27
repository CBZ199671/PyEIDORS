#!/usr/bin/env python3
"""Import a standard EIDORS geometry payload and reconstruct it in PyEIDORS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
BENCHMARK_SCRIPT_DIR = REPO_ROOT / "scripts" / "benchmarks"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(BENCHMARK_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_SCRIPT_DIR))

from pyeidors import EITSystem
from pyeidors.data.structures import EITImage, PatternConfig
from pyeidors.interop import STANDARD_INTEROP_FORMAT, build_mesh_from_exchange_mat, load_forward_csv

from benchmark_difference_runtime import build_single_step_namespace, run_single_step_benchmark
from benchmark_reviewer_case import conductivity_metrics, voltage_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-mat", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--details-mat", type=Path, default=None)
    parser.add_argument("--n-elec", type=int, default=16)
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
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    baseline, phantom, target_diff = load_forward_csv(args.input_csv)
    imported_mesh, payload = build_mesh_from_exchange_mat(args.mesh_mat)
    source_framework = (
        str(np.asarray(payload.get("source_framework", "eidors")).reshape(-1)[0])
        if "source_framework" in payload
        else "eidors"
    )

    background = float(np.asarray(payload["background"]).reshape(-1)[0])
    contact_impedance = float(np.asarray(payload["contact_impedance"]).reshape(-1)[0])
    truth_elem_data = np.asarray(payload["truth_elem_data"], dtype=float).reshape(-1)

    system = EITSystem(
        n_elec=args.n_elec,
        pattern_config=make_pattern_config(args.n_elec),
        contact_impedance=np.ones(args.n_elec, dtype=float) * contact_impedance,
        base_conductivity=background,
        regularization_type="noser",
        regularization_alpha=1.0,
        noser_exponent=0.5,
    )
    system.setup(mesh=imported_mesh)

    baseline_image = system.create_homogeneous_image(conductivity=background)
    truth_image = EITImage(elem_data=truth_elem_data, fwd_model=system.fwd_model)

    single_step_args = build_single_step_namespace(
        system,
        jacobian_backend="eidors_adjoint",
        meas_weight_strategy="none",
        single_step_space="parameter",
        eidors_compatible_step_search=True,
    )
    recon_image, predicted_diff, step_size = run_single_step_benchmark(
        system,
        baseline_image,
        baseline,
        phantom,
        single_step_args,
        target_diff=target_diff,
    )

    result = {
        "study": "same_geometry_cross_generation",
        "source_framework": source_framework,
        "framework": "pyeidors",
        "exchange_format": str(getattr(imported_mesh, "exchange_format", STANDARD_INTEROP_FORMAT)),
        "mesh_name": imported_mesh.mesh_name,
        "n_nodes": int(imported_mesh.num_vertices()),
        "n_elements": int(len(system.fwd_model.V_sigma.dofmap().dofs())),
        "electrode_coverage": None,
        "jacobian_backend": "eidors_adjoint",
        "meas_weight_strategy": "none",
        "single_step_space": "parameter",
        "eidors_compatible_step_search": True,
        "optimal_step_size": float(step_size),
        "imported_same_geometry": True,
    }
    result.update(voltage_metrics(target_diff, predicted_diff))
    result.update(conductivity_metrics(truth_elem_data, recon_image.elem_data))

    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.details_mat is not None:
        args.details_mat.parent.mkdir(parents=True, exist_ok=True)
        savemat(
            args.details_mat,
            {
                "exchange_format": getattr(imported_mesh, "exchange_format", STANDARD_INTEROP_FORMAT),
                "source_framework": source_framework,
                "framework": "pyeidors",
                "mesh_name": imported_mesh.mesh_name,
                "truth_elem_data": np.asarray(truth_elem_data, dtype=float).reshape(-1),
                "recon_elem_data": np.asarray(recon_image.elem_data, dtype=float).reshape(-1),
                "target_diff": np.asarray(target_diff, dtype=float).reshape(-1),
                "predicted_diff": np.asarray(predicted_diff, dtype=float).reshape(-1),
                "voltage_rmse": float(result["voltage_rmse"]),
                "conductivity_rmse": float(result["conductivity_rmse"]),
            },
        )


if __name__ == "__main__":
    main()
