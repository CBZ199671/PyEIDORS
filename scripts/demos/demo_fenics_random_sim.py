#!/usr/bin/env python3
"""Minimal demo: Random target placement on FEniCS mesh for EIT forward/inverse simulation.

Workflow:
1) Load cached 16-electrode circular domain mesh (eit_meshes/mesh_102070*), avoiding gmsh dependency.
2) Construct adjacent drive/measurement patterns, contact impedance matching MATLAB example (1e-6).
3) Generate random circular anomaly (random position, radius, contrast), forward solve for baseline/target voltages.
4) Use modular Gauss-Newton reconstruction (NOSER regularization, default settings) to estimate conductivity.
5) Compute simple metrics against ground truth/measurements, write results to results/demo_random_fenics/*.npz.

Run:
    python scripts/demo_fenics_random_sim.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pyeidors.core_system import EITSystem
from pyeidors.data.structures import PatternConfig, EITImage
from pyeidors.data.synthetic_data import create_custom_phantom
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.visualization import create_visualizer

def cell_to_node(mesh, cell_values: np.ndarray) -> np.ndarray:
    """Interpolate cell values to nodes by averaging, for plotting convenience."""
    node_values = np.zeros(mesh.num_vertices())
    node_counts = np.zeros(mesh.num_vertices())
    for cell_idx, cell in enumerate(mesh.cells()):
        for v_idx in cell:
            node_values[v_idx] += cell_values[cell_idx]
            node_counts[v_idx] += 1
    node_counts[node_counts == 0] = 1
    node_values /= node_counts
    return node_values


def make_random_anomaly(rng: np.random.Generator) -> Dict:
    """Randomly generate a circular anomaly description."""
    # Radius in [0.08, 0.18], center distance from origin <= 0.5
    radius = float(rng.uniform(0.08, 0.18))
    angle = float(rng.uniform(0, 2 * math.pi))
    dist = float(rng.uniform(0.0, 0.5))
    center = (dist * math.cos(angle), dist * math.sin(angle))
    # Contrast: increase or decrease conductivity
    contrast = float(rng.uniform(1.5, 3.0))
    if rng.random() < 0.35:  # 35% chance to make resistive (low conductivity)
        contrast = float(rng.uniform(0.2, 0.8))
    return {"center": center, "radius": radius, "conductivity": contrast}


def main() -> None:
    rng = np.random.default_rng(20241116)

    # 1) Load cached mesh (16 electrodes), avoiding gmsh dependency
    mesh = load_or_create_mesh(mesh_dir="eit_meshes", mesh_name="mesh_102070", n_elec=16)

    # 2) Build system (adjacent drive/measurement, amplitude 1; contact impedance 1e-6)
    pattern_cfg = PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        amplitude=1.0,
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

    n_elem = len(system.fwd_model.V_sigma.dofmap().dofs())

    # 3) Construct baseline and random anomaly
    sigma_bg = np.ones(n_elem, dtype=float)
    anomaly = make_random_anomaly(rng)
    sigma_true_fn = create_custom_phantom(
        system.fwd_model,
        background_conductivity=1.0,
        anomalies=[anomaly],
    )
    sigma_true = sigma_true_fn.vector()[:]

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
    sigma_est = recon["conductivity"].vector()[:]
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
        "residual_final": float(recon["final_residual"]),
    }

    # 7) Save results
    out_dir = Path("results/demo_random_fenics")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "simulation_outputs.npz",
        sigma_bg=sigma_bg,
        sigma_true=sigma_true,
        sigma_est=sigma_est,
        anomaly_center=np.array(anomaly["center"]),
        anomaly_radius=anomaly["radius"],
        anomaly_conductivity=anomaly["conductivity"],
        meas_bg=data_bg.meas,
        meas_true=data_true.meas,
        meas_est=data_est.meas,
        diff_meas=diff_meas,
        metrics=np.array(list(metrics.values())),
        metric_names=np.array(list(metrics.keys())),
    )

    # 8) Visualization: ground truth vs reconstructed conductivity, measurement comparison
    viz = create_visualizer()
    sigma_true_nodes = cell_to_node(mesh, sigma_true) if len(sigma_true) == mesh.num_cells() else sigma_true
    sigma_est_nodes = cell_to_node(mesh, sigma_est) if len(sigma_est) == mesh.num_cells() else sigma_est
    fig_cmp = viz.plot_reconstruction_comparison(
        mesh,
        sigma_true_nodes,
        sigma_est_nodes,
        title="Ground Truth vs Reconstructed Conductivity"
    )
    fig_cmp.savefig(out_dir / "conductivity_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig_cmp)

    # Boundary voltage comparison: true target vs reconstructed prediction (in measurement space)
    fig_v = plt.figure(figsize=(10, 4))
    ax1 = fig_v.add_subplot(1, 2, 1)
    ax1.scatter(data_true.meas, data_est.meas, s=14, alpha=0.7, label="Predicted vs Ground Truth")
    vmin = min(data_true.meas.min(), data_est.meas.min())
    vmax = max(data_true.meas.max(), data_est.meas.max())
    ax1.plot([vmin, vmax], [vmin, vmax], "r--", label="y = x")
    ax1.set_title("Boundary Voltage Scatter")
    ax1.set_xlabel("Ground Truth")
    ax1.set_ylabel("Predicted")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = fig_v.add_subplot(1, 2, 2)
    idx = np.arange(len(data_true.meas))
    ax2.plot(idx, data_true.meas, "b-", lw=1.2, label="Ground Truth")
    ax2.plot(idx, data_est.meas, "r--", lw=1.2, label="Predicted")
    ax2.set_title("Boundary Voltage Sequence")
    ax2.set_xlabel("Measurement Index")
    ax2.set_ylabel("Voltage")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig_v.suptitle("Target/Predicted Boundary Voltage Comparison", fontsize=13, fontweight="bold")
    fig_v.tight_layout()
    fig_v.savefig(out_dir / "voltage_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig_v)

    print("Random anomaly:", anomaly)
    print("Metrics:", metrics)
    print(f"Results saved to {out_dir / 'simulation_outputs.npz'}")
    print(f"Conductivity comparison: {out_dir / 'conductivity_comparison.png'}")
    print(f"Voltage comparison: {out_dir / 'voltage_comparison.png'}")


if __name__ == "__main__":
    main()
