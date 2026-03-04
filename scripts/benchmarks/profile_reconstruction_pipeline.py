#!/usr/bin/env python3
"""Profile forward/Jacobian/GN stages with timing and peak memory snapshots."""

from __future__ import annotations

import argparse
import json
import os
import time
import tracemalloc
from pathlib import Path
from typing import Callable

import numpy as np
from dolfinx import fem

# Keep runtime consistent with test environment on macOS mixed PETSc/Torch stack.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pyeidors import EITSystem
from pyeidors.data.structures import EITImage, PatternConfig
from pyeidors.data.synthetic_data import create_custom_phantom
from pyeidors.femx import function_get_array
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-dir", type=Path, default=Path("eit_meshes"))
    parser.add_argument("--refinement", type=int, default=12)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--electrode-coverage", type=float, default=0.5)
    parser.add_argument("--background", type=float, default=1.0)
    parser.add_argument("--contact-impedance", type=float, default=1e-5)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _profile_block(name: str, fn: Callable[[], object]) -> tuple[object, dict[str, float]]:
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, {
        "stage": name,
        "elapsed_sec": float(elapsed),
        "peak_mib": float(peak / (1024 * 1024)),
        "current_mib": float(current / (1024 * 1024)),
    }


def _build_system(args: argparse.Namespace) -> EITSystem:
    pattern = PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="normalized",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
        rotate_meas=True,
    )
    system = EITSystem(
        n_elec=16,
        pattern_config=pattern,
        contact_impedance=np.full(16, float(args.contact_impedance), dtype=float),
        regularization_type="noser",
        regularization_alpha=1.0,
        cache_scope="off",
    )
    mesh = load_or_create_mesh(
        mesh_dir=str(args.mesh_dir),
        n_elec=16,
        refinement=int(args.refinement),
        radius=float(args.radius),
        electrode_coverage=float(args.electrode_coverage),
    )
    system.setup(mesh=mesh)
    system.reconstructor.max_iterations = int(args.max_iterations)
    system.reconstructor.min_iterations = 1
    system.reconstructor.verbose = False
    return system


def main() -> None:
    args = parse_args()
    system = _build_system(args)

    baseline_image = system.create_homogeneous_image(conductivity=float(args.background))
    sigma = create_custom_phantom(
        system.fwd_model,
        background_conductivity=float(args.background),
        anomalies=[{"center": (0.3, 0.2), "radius": 0.18, "conductivity": float(args.background) * 1.8}],
    )
    phantom_image = EITImage(elem_data=function_get_array(sigma).copy(), fwd_model=system.fwd_model)

    (_, stage_forward_baseline) = _profile_block(
        "forward_baseline",
        lambda: system.fwd_model.fwd_solve(baseline_image),
    )
    (target_data, _), stage_forward_target = _profile_block(
        "forward_target",
        lambda: system.fwd_model.fwd_solve(phantom_image),
    )

    sigma_fn = fem.Function(system.fwd_model.V_sigma)
    sigma_fn.x.array[:] = baseline_image.elem_data
    (_, stage_jacobian) = _profile_block(
        "jacobian",
        lambda: system.reconstructor.jacobian_calculator.calculate(sigma_fn, method="efficient"),
    )

    (_, stage_gn) = _profile_block(
        "gauss_newton",
        lambda: system.reconstructor.reconstruct(
            measured_data=target_data,
            initial_conductivity=float(args.background),
            jacobian_method="efficient",
        ),
    )

    rows = [
        stage_forward_baseline,
        stage_forward_target,
        stage_jacobian,
        stage_gn,
    ]
    for row in rows:
        print(
            f"{row['stage']:>16}  time={row['elapsed_sec']:.4f}s  "
            f"peak={row['peak_mib']:.2f} MiB"
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "refinement": int(args.refinement),
                "background": float(args.background),
                "max_iterations": int(args.max_iterations),
            },
            "stages": rows,
        }
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved profile report: {args.output_json}")


if __name__ == "__main__":
    main()
