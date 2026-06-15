#!/usr/bin/env python3
"""
16-electrode Complete Electrode Model (CEM) end-to-end simulation and reconstruction test script.

Steps:
1) Create a unit-square mesh and tag 16 boundary electrodes.
2) Construct EITSystem (using contact impedance and adjacent stimulation/measurement patterns).
3) Generate homogeneous reference and phantom with circular inclusion, perform forward solve.
4) Use difference reconstruction (Gauss-Newton + NOSER) to estimate conductivity.
5) Print electrode measures, measurement ranges, reconstruction range and relative error for quick CEM verification.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Runtime stability guard for mixed PETSc/Torch execution on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from mpi4py import MPI

from pyeidors import EITSystem
from pyeidors.data.structures import PatternConfig
from pyeidors.femx import build_eit_mesh, function_get_array
from pyeidors.forward.complex_support import petsc_scalar_dtype
from pyeidors.perf import DEFAULT_ACCELERATION_PROFILE
from pyeidors.utils.numeric_ops import real_array_if_zero_imaginary
from scripts.common.acceleration_profiles import add_acceleration_profile_argument

try:  # pragma: no cover - thread cap is a runtime stability measure
    import torch

    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
except Exception:
    pass


def _edge_parameter(points: np.ndarray) -> np.ndarray:
    """Map boundary points to parameter t in [0, 4) on square perimeter."""
    x = points[:, 0]
    y = points[:, 1]
    eps = 1e-10

    t = np.zeros_like(x)
    left = np.isclose(x, 0.0, atol=eps)
    top = (~left) & np.isclose(y, 1.0, atol=eps)
    right = (~left) & (~top) & np.isclose(x, 1.0, atol=eps)
    bottom = (~left) & (~top) & (~right) & np.isclose(y, 0.0, atol=eps)

    t[left] = y[left]
    t[top] = 1.0 + x[top]
    t[right] = 2.0 + (1.0 - y[right])
    t[bottom] = 3.0 + (1.0 - x[bottom])
    return np.clip(t, 0.0, 4.0 - eps)


def create_square_eit_mesh(n_elec: int = 16, nx: int = 64, ny: int = 64):
    """Create a unit-square mesh with 16 tagged electrodes on the exterior boundary."""
    square_mesh = dmesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
    tdim = square_mesh.topology.dim
    fdim = tdim - 1

    # Exterior facets
    boundary_facets = dmesh.locate_entities_boundary(
        square_mesh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
    ).astype(np.int32)

    # Compute facet centroids using facet->vertex connectivity
    square_mesh.topology.create_connectivity(fdim, 0)
    f2v = square_mesh.topology.connectivity(fdim, 0)
    if f2v is None:
        raise RuntimeError("Failed to build facet->vertex connectivity")
    coords = square_mesh.geometry.x[:, :2]
    centroids = np.zeros((boundary_facets.size, 2), dtype=float)
    for i, facet in enumerate(boundary_facets):
        vertices = f2v.links(int(facet))
        centroids[i, :] = coords[vertices].mean(axis=0)

    seg_len = 4.0 / n_elec
    tags = (np.floor(_edge_parameter(centroids) / seg_len).astype(np.int32) + 2).astype(
        np.int32
    )

    # meshtags require sorted entity indices
    order = np.argsort(boundary_facets)
    boundary_facets = boundary_facets[order]
    tags = tags[order]
    facet_tags = dmesh.meshtags(square_mesh, fdim, boundary_facets, tags)

    association_table = {f"electrode_{i + 1}": i + 2 for i in range(n_elec)}
    eit_mesh = build_eit_mesh(
        square_mesh,
        facet_tags=facet_tags,
        association_table=association_table,
    )
    return eit_mesh


def _fem_unit_constant(domain):
    return fem.Constant(domain, np.asarray(1.0, dtype=petsc_scalar_dtype())[()])


def _real_scalar(value, *, name: str) -> float:
    return float(real_array_if_zero_imaginary(value, name=name).reshape(()))


def run_test(
    *,
    skip_inverse: bool = False,
    acceleration_profile: str = DEFAULT_ACCELERATION_PROFILE,
):
    n_elec = 16
    # Keep the mesh moderately fine for signal quality while avoiding
    # unnecessary solver pressure in local/CI smoke runs.
    mesh = create_square_eit_mesh(n_elec=n_elec, nx=48, ny=48)

    pattern_config = PatternConfig(
        n_elec=n_elec,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="normalized",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
    )

    contact_impedance = np.ones(n_elec) * 1e-5

    eit_system = EITSystem(
        n_elec=n_elec,
        pattern_config=pattern_config,
        contact_impedance=contact_impedance,
        base_conductivity=1.0,
        regularization_type="noser",
        regularization_alpha=1.0,
        linear_backend="scipy",
        performance_mode="safe",
        acceleration_profile=str(acceleration_profile),
    )
    eit_system.setup(mesh=mesh)

    ds = ufl.Measure("ds", domain=mesh.mesh, subdomain_data=mesh.facet_tags)
    one = _fem_unit_constant(mesh.mesh)
    electrode_measures = [
        _real_scalar(
            mesh.comm.allreduce(
                fem.assemble_scalar(fem.form(one * ds(tag))), op=MPI.SUM
            ),
            name=f"electrode {tag} measure",
        )
        for tag in range(2, 2 + n_elec)
    ]
    print(
        f"Electrode boundary measures min/max: {min(electrode_measures):.6f} / {max(electrode_measures):.6f}"
    )

    reference_img = eit_system.create_homogeneous_image(conductivity=1.0)
    reference_data = eit_system.forward_solve(reference_img)

    phantom_img = eit_system.add_phantom(
        base_conductivity=1.0,
        phantom_conductivity=2.5,
        phantom_center=(0.35, 0.35),
        phantom_radius=0.12,
    )
    phantom_data = eit_system.forward_solve(phantom_img)

    print(
        f"Reference meas range: [{reference_data.meas.min():.6e}, {reference_data.meas.max():.6e}]"
    )
    print(
        f"Phantom meas range:   [{phantom_data.meas.min():.6e}, {phantom_data.meas.max():.6e}]"
    )

    if skip_inverse:
        print("Inverse reconstruction is skipped (requested by --skip-inverse).")
        return

    if os.getenv("PYEIDORS_TEST_FORCE_CEM_FAIL", "0") == "1":
        raise RuntimeError("Forced CEM failure via PYEIDORS_TEST_FORCE_CEM_FAIL.")

    if eit_system.reconstructor is None:
        raise RuntimeError("EIT reconstructor is not initialized after setup().")
    try:
        eit_system.reconstructor.ensure_regularization_ready()
    except Exception as exc:
        raise RuntimeError(f"regularization warmup failed in run_cem: {exc}") from exc

    # Use a fixed step schedule for deterministic script behavior.
    eit_system.reconstructor.step_schedule = [
        0.25
    ] * eit_system.reconstructor.max_iterations

    recon_result = eit_system.inverse_solve(
        data=phantom_data,
        reference_data=reference_data,
        initial_guess=None,
    )
    recon_sigma = function_get_array(recon_result.conductivity).copy()

    true_sigma = real_array_if_zero_imaginary(
        phantom_img.elem_data, name="phantom conductivity"
    )
    rel_err = np.linalg.norm(recon_sigma - true_sigma) / np.linalg.norm(true_sigma)

    print(f"Reconstruction range: [{recon_sigma.min():.6f}, {recon_sigma.max():.6f}]")
    print(f"Relative error (L2): {rel_err:.4f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CEM square forward/inverse smoke test"
    )
    parser.add_argument(
        "--skip-inverse",
        action="store_true",
        help="Skip inverse reconstruction and run forward checks only.",
    )
    add_acceleration_profile_argument(
        parser,
        default=DEFAULT_ACCELERATION_PROFILE,
        help_suffix="For this 2D smoke script the profile is accepted mainly for CLI consistency.",
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
        print(
            f"[ERROR] CEM square test failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
