#!/usr/bin/env python3
"""Minimal demo: Random target placement on DOLFINx mesh for EIT forward/inverse simulation.

Workflow:
1) Load cached 16-electrode circular domain mesh (eit_meshes/mesh_102070*), avoiding gmsh dependency.
2) Construct adjacent drive/measurement patterns, contact impedance matching MATLAB example (1e-6).
3) Generate random circular anomaly (random position, radius, contrast), forward solve for baseline/target voltages.
4) Use modular Gauss-Newton reconstruction (NOSER regularization, default settings) to estimate conductivity.
5) Compute simple metrics against ground truth/measurements, write results to results/demo_random_dolfinx/*.h5.

Run:
    python scripts/demos/demo_dolfinx_random_sim.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
for candidate in (str(REPO_ROOT), str(SRC_PATH)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from pyeidors.core_system import EITSystem
from pyeidors.data.structures import PatternConfig, EITImage
from pyeidors.data.synthetic_data import create_custom_phantom
from pyeidors.femx import function_get_array
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.visualization import create_visualizer
from scripts.common.hdf5_outputs import DEMO_ARRAYS_SCHEMA, write_output_bundle
from scripts.demos._shared import (
    cell_to_node,
    make_random_anomaly,
    save_voltage_comparison_figure,
)


def main() -> None:
    rng = np.random.default_rng(20241116)

    # 1) Load cached mesh (16 electrodes), avoiding gmsh dependency
    mesh = load_or_create_mesh(
        mesh_dir="eit_meshes", mesh_name="mesh_102070", n_elec=16
    )

    # 2) Build system (adjacent drive/measurement, normalized drive=1; contact impedance 1e-6)
    pattern_cfg = PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="normalized",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
        use_meas_current=False,
        rotate_meas=True,
    )
    contact_impedance = np.full(16, 1e-6, dtype=float)

    system = EITSystem(
        n_elec=16,
        pattern_config=pattern_cfg,
        contact_impedance=contact_impedance,
        base_conductivity=1.0,
        regularization_type="noser",
        regularization_alpha=1.0,
    )
    system.setup(mesh=mesh)

    n_elem = int(
        system.fwd_model.V_sigma.dofmap.index_map.size_local
        * system.fwd_model.V_sigma.dofmap.index_map_bs
    )

    # 3) Construct baseline and random anomaly
    sigma_bg = np.ones(n_elem, dtype=float)
    anomaly = make_random_anomaly(rng)
    sigma_true_fn = create_custom_phantom(
        system.fwd_model,
        background_conductivity=1.0,
        anomalies=[anomaly],
    )
    sigma_true = function_get_array(sigma_true_fn).copy()

    img_bg = EITImage(elem_data=sigma_bg, fwd_model=system.fwd_model)
    img_true = EITImage(elem_data=sigma_true, fwd_model=system.fwd_model)

    # 4) Forward solve: baseline and target
    data_bg, _ = system.fwd_model.fwd_solve(img_bg)
    data_true, _ = system.fwd_model.fwd_solve(img_true)
    diff_meas = data_true.meas - data_bg.meas

    # 5) Single-step Gauss-Newton reconstruction (absolute reconstruction, initial conductivity=1)
    recon = system.reconstructor.reconstruct(
        data_true, initial_conductivity=1.0, jacobian_method="efficient"
    )
    sigma_est = function_get_array(recon.conductivity).copy()
    img_est = EITImage(elem_data=sigma_est, fwd_model=system.fwd_model)
    data_est, _ = system.fwd_model.fwd_solve(img_est)

    # 6) Error metrics
    meas_rmse = float(np.sqrt(np.mean((data_est.meas - data_true.meas) ** 2)))
    sigma_rmse = float(np.sqrt(np.mean((sigma_est - sigma_true) ** 2)))
    sigma_mae = float(np.mean(np.abs(sigma_est - sigma_true)))
    metrics: Dict[str, float] = {
        "meas_rmse": meas_rmse,
        "sigma_rmse": sigma_rmse,
        "sigma_mae": sigma_mae,
        "residual_final": float(recon.final_residual),
    }

    # 7) Save results
    out_dir = Path("results/demo_random_dolfinx")
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = write_output_bundle(
        out_dir / "simulation_outputs.h5",
        {
            "sigma_bg": sigma_bg,
            "sigma_true": sigma_true,
            "sigma_est": sigma_est,
            "anomaly_center": np.array(anomaly["center"]),
            "anomaly_radius": anomaly["radius"],
            "anomaly_conductivity": anomaly["conductivity"],
            "meas_bg": data_bg.meas,
            "meas_true": data_true.meas,
            "meas_est": data_est.meas,
            "diff_meas": diff_meas,
            "metrics": np.array(list(metrics.values())),
            "metric_names": np.array(list(metrics.keys())),
        },
        {"package_role": "demo_random_simulation_outputs"},
        schema=DEMO_ARRAYS_SCHEMA,
    )

    # 8) Visualization: ground truth vs reconstructed conductivity, measurement comparison
    viz = create_visualizer()
    sigma_true_nodes = (
        cell_to_node(mesh, sigma_true)
        if len(sigma_true) == mesh.num_cells()
        else sigma_true
    )
    sigma_est_nodes = (
        cell_to_node(mesh, sigma_est)
        if len(sigma_est) == mesh.num_cells()
        else sigma_est
    )
    fig_cmp = viz.plot_reconstruction_comparison(
        mesh,
        sigma_true_nodes,
        sigma_est_nodes,
        title="Ground Truth vs Reconstructed Conductivity",
    )
    fig_cmp.savefig(
        out_dir / "conductivity_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig_cmp)

    # Boundary voltage comparison: true target vs reconstructed prediction (in measurement space)
    save_voltage_comparison_figure(
        output_path=out_dir / "voltage_comparison.png",
        measured=data_true.meas,
        predicted=data_est.meas,
        suptitle="Target/Predicted Boundary Voltage Comparison",
    )

    print("Random anomaly:", anomaly)
    print("Metrics:", metrics)
    print(f"Results saved to {outputs_path}")
    print(f"Conductivity comparison: {out_dir / 'conductivity_comparison.png'}")
    print(f"Voltage comparison: {out_dir / 'voltage_comparison.png'}")


if __name__ == "__main__":
    main()
