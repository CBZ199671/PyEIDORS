#!/usr/bin/env python3
"""3D cylinder CEM forward + difference-inverse smoke test."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Runtime stability guard for PETSc/Torch mixed workloads.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI

from pyeidors import EITSystem
from pyeidors.data.structures import EITImage, PatternConfig
from pyeidors.femx import function_get_array
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.perf import DEFAULT_ACCELERATION_PROFILE
from common.acceleration_profiles import (
    add_acceleration_profile_argument,
    resolve_3d_mesh_contract,
)

try:  # pragma: no cover - runtime tuning only
    import torch

    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
except Exception:
    pass


def _build_3d_phantom(
    eit_system: EITSystem,
    *,
    base_conductivity: float,
    phantom_conductivity: float,
    center: tuple[float, float, float],
    radius: float,
) -> EITImage:
    image = eit_system.create_homogeneous_image(conductivity=base_conductivity)
    sigma = np.asarray(image.elem_data, dtype=float).copy()
    coords = eit_system.fwd_model.V_sigma.tabulate_dof_coordinates()
    distances = np.linalg.norm(coords[:, :3] - np.asarray(center, dtype=float)[None, :], axis=1)
    sigma[distances <= float(radius)] = float(phantom_conductivity)
    return EITImage(elem_data=sigma, fwd_model=eit_system.fwd_model)


def run_test(
    *,
    skip_inverse: bool = False,
    acceleration_profile: str = DEFAULT_ACCELERATION_PROFILE,
) -> None:
    n_elec = 16
    radius = 0.22
    mesh_family, geometry_version, generator_revision = resolve_3d_mesh_contract(
        acceleration_profile=acceleration_profile,
    )
    mesh = load_or_create_mesh(
        mesh_dir=str(Path("eit_meshes")),
        n_elec=n_elec,
        dimension=3,
        radius=radius,
        height=0.16,
        refinement=1,
        electrode_coverage=0.5,
        electrode_height_ratio=0.2,
        z_center=0.0,
        mesh_family=mesh_family,
        geometry_version=geometry_version,
        generator_revision=generator_revision,
    )

    pattern_config = PatternConfig(
        n_elec=n_elec,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="total_current",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
    )
    contact_impedance = np.ones(n_elec, dtype=float) * 1e-5

    eit_system = EITSystem(
        n_elec=n_elec,
        pattern_config=pattern_config,
        contact_impedance=contact_impedance,
        base_conductivity=1.0,
        regularization_type="noser",
        regularization_alpha=1.0,
        linear_backend="scipy",
        performance_mode="safe",
        solver_mode="fast",
        linear_solver="auto",
        jacobian_update_every=2,
        jacobian_reuse_tol=1e-3,
        line_search_mode="fast",
        acceleration_profile=str(acceleration_profile),
    )
    eit_system.setup(mesh=mesh)

    ds = ufl.Measure("ds", domain=mesh.mesh, subdomain_data=mesh.facet_tags)
    one = fem.Constant(mesh.mesh, 1.0)
    electrode_measures = [
        float(mesh.comm.allreduce(fem.assemble_scalar(fem.form(one * ds(tag))), op=MPI.SUM))
        for tag in range(2, 2 + n_elec)
    ]
    print(f"Electrode boundary measures min/max: {min(electrode_measures):.6e} / {max(electrode_measures):.6e}")

    reference_img = eit_system.create_homogeneous_image(conductivity=1.0)
    reference_data = eit_system.forward_solve(reference_img)

    phantom_img = _build_3d_phantom(
        eit_system,
        base_conductivity=1.0,
        phantom_conductivity=2.0,
        center=(radius * 0.35, 0.0, 0.0),
        radius=radius * 0.22,
    )
    phantom_data = eit_system.forward_solve(phantom_img)

    print(f"Reference meas range: [{reference_data.meas.min():.6e}, {reference_data.meas.max():.6e}]")
    print(f"Phantom meas range:   [{phantom_data.meas.min():.6e}, {phantom_data.meas.max():.6e}]")

    if skip_inverse:
        print("Inverse reconstruction is skipped (requested by --skip-inverse).")
        return

    if os.getenv("PYEIDORS_TEST_FORCE_CEM_FAIL", "0") == "1":
        raise RuntimeError("Forced 3D CEM failure via PYEIDORS_TEST_FORCE_CEM_FAIL.")

    if eit_system.reconstructor is None:
        raise RuntimeError("EIT reconstructor is not initialized after setup().")
    try:
        eit_system.reconstructor.ensure_regularization_ready()
    except Exception as exc:
        raise RuntimeError(f"regularization warmup failed in 3D run_cem: {exc}") from exc

    # 3D smoke keeps single update step for deterministic runtime.
    eit_system.reconstructor.max_iterations = 1
    eit_system.reconstructor.step_schedule = [0.25]

    recon_result = eit_system.inverse_solve(
        data=phantom_data,
        reference_data=reference_data,
        initial_guess=None,
    )
    recon_sigma = function_get_array(recon_result.conductivity).copy()
    true_sigma = np.asarray(phantom_img.elem_data, dtype=float)
    rel_err = np.linalg.norm(recon_sigma - true_sigma) / np.linalg.norm(true_sigma)

    print(f"Reconstruction range: [{recon_sigma.min():.6f}, {recon_sigma.max():.6f}]")
    print(f"Relative error (L2): {rel_err:.4f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D CEM cylinder forward/inverse smoke test")
    add_acceleration_profile_argument(
        parser,
        default=DEFAULT_ACCELERATION_PROFILE,
        help_suffix="Use `gpu3d` to simplify the 3D GPU path when the runtime supports it.",
    )
    parser.add_argument(
        "--skip-inverse",
        action="store_true",
        help="Skip inverse reconstruction and run forward checks only.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        run_test(
            skip_inverse=bool(args.skip_inverse),
            acceleration_profile=str(args.acceleration_profile),
        )
    except Exception as exc:
        print(f"[ERROR] 3D CEM cylinder test failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
