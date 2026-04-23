#!/usr/bin/env python3
"""Single-step difference Gauss-Newton (one linear solve) example using DOLFINx mesh.

Workflow:
1) Load cached 16-electrode circular domain mesh (eit_meshes/mesh_102070*).
2) Adjacent drive/measurement, contact impedance 1e-6.
3) Randomly place a circular anomaly, generate baseline/target measurements.
4) Compute Jacobian at baseline conductivity, directly solve one linear system
   delta_sigma = (J^T W J + lambda R) \\ (J^T W dv), no iteration.
5) Save conductivity comparison and measurement comparison plots.

Run:
    python scripts/demos/demo_dolfinx_single_step_diff.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
for candidate in (str(REPO_ROOT), str(SRC_PATH)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from dolfinx import fem

from pyeidors.data.structures import PatternConfig, EITImage
from pyeidors.data.synthetic_data import create_custom_phantom
from pyeidors.femx import function_get_array, function_set_array
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.forward.eit_forward_model import EITForwardModel
from pyeidors.inverse.jacobian.adjoint_jacobian import EidorsStyleAdjointJacobian
from pyeidors.inverse.regularization.smoothness import NOSERRegularization
from pyeidors.visualization import create_visualizer
from scripts.common.hdf5_outputs import DEMO_ARRAYS_SCHEMA, write_output_bundle
from scripts.demos._shared import (
    cell_to_node,
    make_random_anomaly,
    save_voltage_comparison_figure,
)


def main() -> None:
    rng = np.random.default_rng(20241116)
    # 1) Mesh and patterns
    mesh = load_or_create_mesh(
        mesh_dir="eit_meshes", mesh_name="mesh_102070", n_elec=16
    )
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
    fwd_model = EITForwardModel(
        n_elec=16,
        pattern_config=pattern_cfg,
        z=contact_impedance,
        mesh=mesh,
    )

    n_elem = int(fem.Function(fwd_model.V_sigma).x.array.size)

    # 2) Baseline and random anomaly
    sigma_bg = np.ones(n_elem, dtype=float)
    anomaly = make_random_anomaly(rng)
    sigma_true_fn = create_custom_phantom(
        fwd_model,
        background_conductivity=1.0,
        anomalies=[anomaly],
    )
    sigma_true = function_get_array(sigma_true_fn).copy()

    img_bg = EITImage(elem_data=sigma_bg, fwd_model=fwd_model)
    img_true = EITImage(elem_data=sigma_true, fwd_model=fwd_model)

    # 3) Forward: baseline / target
    data_bg, _ = fwd_model.fwd_solve(img_bg)
    data_true, _ = fwd_model.fwd_solve(img_true)
    dv = data_true.meas - data_bg.meas  # unnormalized difference

    # 4) Single-step difference: J, prior, one linear solve
    jac_calc = EidorsStyleAdjointJacobian(fwd_model, use_torch=False)
    sigma_fun_bg = fem.Function(fwd_model.V_sigma)
    function_set_array(sigma_fun_bg, sigma_bg)
    J = jac_calc.calculate(sigma_fun_bg, method="efficient")  # shape: n_meas x n_elem
    # EIDORS sign convention is built into EidorsStyleAdjointJacobian, no extra negation needed.

    # W = I (unweighted), prior uses NOSER, exponent=0.5 (EIDORS default)
    reg = NOSERRegularization(
        fwd_model, jac_calc, base_conductivity=1.0, alpha=1.0, exponent=0.5
    )
    R = reg.get_regularization_matrix()  # already numpy, shape (n_elem, n_elem)
    lam = 1e-2  # adjustable

    JTJ = J.T @ J
    RHS = J.T @ dv
    A = JTJ + lam * R

    delta_sigma = np.linalg.solve(A, RHS)
    sigma_est = sigma_bg + delta_sigma

    img_est = EITImage(elem_data=sigma_est, fwd_model=fwd_model)
    data_est, _ = fwd_model.fwd_solve(img_est)

    # 5) Metrics
    meas_rmse = float(np.sqrt(np.mean((data_est.meas - data_true.meas) ** 2)))
    sigma_rmse = float(np.sqrt(np.mean((sigma_est - sigma_true) ** 2)))
    sigma_mae = float(np.mean(np.abs(sigma_est - sigma_true)))
    metrics = {
        "meas_rmse": meas_rmse,
        "sigma_rmse": sigma_rmse,
        "sigma_mae": sigma_mae,
        "lambda": lam,
    }

    # 6) Visualization
    out_dir = Path("results/demo_single_step_diff")
    out_dir.mkdir(parents=True, exist_ok=True)

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
        title="Single-Step Difference: Ground Truth vs Reconstructed Conductivity",
    )
    fig_cmp.savefig(
        out_dir / "conductivity_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig_cmp)

    save_voltage_comparison_figure(
        output_path=out_dir / "voltage_comparison.png",
        measured=data_true.meas,
        predicted=data_est.meas,
        suptitle="Single-Step Difference: Target/Predicted Boundary Voltage Comparison",
    )

    # Save data
    outputs_path = write_output_bundle(
        out_dir / "single_step_outputs.h5",
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
            "dv": dv,
            "metrics": np.array(list(metrics.values())),
            "metric_names": np.array(list(metrics.keys())),
        },
        {"package_role": "demo_single_step_difference_outputs"},
        schema=DEMO_ARRAYS_SCHEMA,
    )

    print("Random anomaly:", anomaly)
    print("Metrics:", metrics)
    print(f"Results saved to {outputs_path}")
    print(f"Conductivity comparison: {out_dir / 'conductivity_comparison.png'}")
    print(f"Voltage comparison: {out_dir / 'voltage_comparison.png'}")


if __name__ == "__main__":
    main()
