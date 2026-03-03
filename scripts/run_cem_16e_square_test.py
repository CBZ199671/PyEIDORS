#!/usr/bin/env python3
"""
16-electrode Complete Electrode Model (CEM) end-to-end simulation and reconstruction test script.

Steps:
1) Create a mesh on unit square boundary with 16 uniformly distributed electrodes and boundary markers.
2) Construct EITSystem (using contact impedance and adjacent stimulation/measurement patterns).
3) Generate homogeneous reference and phantom with circular inclusion, perform forward solve.
4) Use difference reconstruction (Gauss-Newton + NOSER) to estimate conductivity.
5) Print electrode measures, measurement ranges, reconstruction range and relative error for quick CEM verification.
"""

import numpy as np
from fenics import MeshFunction, SubDomain, UnitSquareMesh, Measure, near, assemble

from pyeidors import EITSystem
from pyeidors.data.structures import PatternConfig


def create_square_eit_mesh(n_elec: int = 16, nx: int = 64, ny: int = 64):
    """
    Create a mesh on unit square with 16 uniformly distributed electrode markers.
    Boundary parameter t in [0, 4): bottom 3->4, left 0->1, top 1->2, right 2->3.
    """
    mesh = UnitSquareMesh(nx, ny)
    facet_function = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

    seg_len = 4.0 / n_elec
    eps = 1e-10

    class ElectrodeMarker(SubDomain):
        def __init__(self, seg_idx: int):
            super().__init__()
            self.seg_idx = seg_idx

        def inside(self, x, on_boundary):
            if not on_boundary:
                return False

            if near(x[0], 0.0):
                t = x[1]
            elif near(x[1], 1.0):
                t = 1.0 + x[0]
            elif near(x[0], 1.0):
                t = 2.0 + (1.0 - x[1])
            elif near(x[1], 0.0):
                t = 3.0 + (1.0 - x[0])
            else:
                return False

            t = min(max(t, 0.0), 4.0 - eps)
            idx = int(t / seg_len)
            return idx == self.seg_idx

    association_table = {}
    for i in range(n_elec):
        tag = i + 2  # Convention: domain is 1, electrodes start from 2
        ElectrodeMarker(i).mark(facet_function, tag)
        association_table[f"electrode_{i + 1}"] = tag

    # Attach EIT-required attributes
    mesh.boundaries_mf = facet_function
    mesh.association_table = association_table
    mesh.n_electrodes = n_elec

    return mesh


def run_test():
    n_elec = 16
    mesh = create_square_eit_mesh(n_elec=n_elec, nx=64, ny=64)

    # Use adjacent stimulation/measurement, current amplitude 1mA (adjustable)
    pattern_config = PatternConfig(
        n_elec=n_elec,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        amplitude=1.0,
    )

    # Contact impedance 1e-5 ohm*m^2
    contact_impedance = np.ones(n_elec) * 1e-5

    eit_system = EITSystem(
        n_elec=n_elec,
        pattern_config=pattern_config,
        contact_impedance=contact_impedance,
        base_conductivity=1.0,
        regularization_type="noser",
        regularization_alpha=1.0,
    )
    eit_system.setup(mesh=mesh)

    # Check electrode boundary measures
    ds = Measure("ds", domain=mesh, subdomain_data=mesh.boundaries_mf)
    electrode_measures = [assemble(1 * ds(tag)) for tag in range(2, 2 + n_elec)]
    print(f"Electrode boundary measures min/max: {min(electrode_measures):.6f} / {max(electrode_measures):.6f}")

    # Reference and phantom
    reference_img = eit_system.create_homogeneous_image(conductivity=1.0)
    reference_data = eit_system.forward_solve(reference_img)

    phantom_img = eit_system.add_phantom(
        base_conductivity=1.0,
        phantom_conductivity=2.5,
        phantom_center=(0.35, 0.35),
        phantom_radius=0.12,
    )
    phantom_data = eit_system.forward_solve(phantom_img)

    print(f"Reference meas range: [{reference_data.meas.min():.6e}, {reference_data.meas.max():.6e}]")
    print(f"Phantom meas range:   [{phantom_data.meas.min():.6e}, {phantom_data.meas.max():.6e}]")

    # Difference reconstruction
    recon_result = eit_system.inverse_solve(
        data=phantom_data,
        reference_data=reference_data,
        initial_guess=None,
    )
    recon_sigma = recon_result["conductivity"].vector()[:]

    # Error metrics
    true_sigma = phantom_img.elem_data
    rel_err = np.linalg.norm(recon_sigma - true_sigma) / np.linalg.norm(true_sigma)

    print(f"Reconstruction range: [{recon_sigma.min():.6f}, {recon_sigma.max():.6f}]")
    print(f"Relative error (L2): {rel_err:.4f}")


if __name__ == "__main__":
    run_test()
