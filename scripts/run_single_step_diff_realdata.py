#!/usr/bin/env python3
"""Gauss-Newton single-step reconstruction on real diff CSV data (EIDORS-style Jacobian).

Usage example:
    python scripts/run_single_step_diff_realdata.py \
        --csv data/measurements/tank/2025-11-14-22-18-02_1_10.00_50uA_3000Hz.csv \
        --lambda 0.1 \
        --output results/tank/single_step_from_cli
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.rcParams.update(
    {
        "axes.unicode_minus": False,
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
    }
)
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
SCRIPTS_PATH = REPO_ROOT / "scripts"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

from pyeidors.data.structures import PatternConfig, EITImage
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.forward.eit_forward_model import EITForwardModel
from pyeidors.inverse.jacobian.adjoint_jacobian import EidorsStyleAdjointJacobian
from pyeidors.inverse.regularization.smoothness import NOSERRegularization
from pyeidors.visualization import create_visualizer

# Use common modules
from common.io_utils import load_csv_measurements, align_frames_polarity
from common.mesh_utils import cell_to_node


def main(
    csv: Path,
    lam: float,
    use_part: str,
    output: Path,
    mesh_name: str = "mesh_16e_r0p025_ref10_cov0p5",
    pattern_amplitude: float | None = None,
    contact_impedance: float = 1e-6,
    measurement_gain: float = 10.0,
    step_size_calib: bool = False,
    step_size_min: float = 1e-3,
    step_size_max: float = 1e1,
    step_size_maxiter: int = 50,
    background_sigma: float = 1.0,
    colormap: str = "viridis",
    colorbar_scientific: bool = False,
    colorbar_format: str | None = None,
    transparent: bool = False,
):
    # Load data using common module
    vh, vi = load_csv_measurements(csv, use_part=use_part, measurement_gain=measurement_gain)

    # EIDORS-style diff imaging: use normalized stimulation current (1.0)
    # - Diff imaging is single-step reconstruction, result is relative conductivity change
    # - Focus on spatial distribution shape, not precise physical units
    # - For physically meaningful conductivity, use iterative absolute imaging
    #
    # If user explicitly specifies --pattern-amplitude, use that value
    stim_amplitude = pattern_amplitude if pattern_amplitude is not None else 1.0
    print(f"[INFO] Diff imaging: amplitude={stim_amplitude:.2e} (EIDORS-style, relative Δσ)")

    mesh = load_or_create_mesh(mesh_dir="eit_meshes", mesh_name=mesh_name, n_elec=16, radius=0.025)
    pattern_cfg = PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        amplitude=stim_amplitude,
        use_meas_current=False,
        rotate_meas=True,
    )
    z_contact = np.full(16, contact_impedance, dtype=float)
    fwd_model = EITForwardModel(n_elec=16, pattern_config=pattern_cfg, z=z_contact, mesh=mesh)

    n_elem = len(fwd_model.V_sigma.dofmap().dofs())
    sigma_bg = np.full(n_elem, background_sigma)
    img_bg = EITImage(elem_data=sigma_bg, fwd_model=fwd_model)
    print(f"Background conductivity: {background_sigma}")

    # Baseline forward for polarity detection (normal/inverted U-shape)
    base_forward, _ = fwd_model.fwd_solve(img_bg)
    vh_vi = np.vstack([vh, vi])
    vh_vi_aligned, flipped = align_frames_polarity(vh_vi, base_forward.meas)
    vh, vi = vh_vi_aligned
    if flipped:
        print(f"[INFO] Polarity correction: flipped frame indices {flipped}")

    jac_calc = EidorsStyleAdjointJacobian(fwd_model, use_torch=False)
    J = jac_calc.calculate_from_image(img_bg)

    dv = vi - vh
    if dv.shape[0] != J.shape[0]:
        raise RuntimeError(f"Data length {dv.shape[0]} does not match Jacobian rows {J.shape[0]}")

    # EIDORS-style NOSER regularization: exponent=0.5
    # NOSER: R = diag(J'J)^0.5, already matches J'J magnitude
    reg = NOSERRegularization(fwd_model, jac_calc, base_conductivity=background_sigma, alpha=1.0, exponent=0.5)
    R = reg.get_regularization_matrix()

    A = J.T @ J + lam * R
    b = J.T @ dv
    try:
        delta_sigma = np.linalg.solve(A, b)
    except Exception:
        delta_sigma = np.linalg.lstsq(A, b, rcond=None)[0]

    pred_vh, _ = fwd_model.fwd_solve(img_bg)

    alpha = 1.0
    if step_size_calib:
        def _objective(scale: float) -> float:
            sigma_try = sigma_bg + scale * delta_sigma
            img_try = EITImage(elem_data=sigma_try, fwd_model=fwd_model)
            pred_vi_try, _ = fwd_model.fwd_solve(img_try)
            pred_diff_try = pred_vi_try.meas - pred_vh.meas
            residual = pred_diff_try - dv
            return float(np.mean(residual ** 2))

        result = minimize_scalar(
            _objective,
            bounds=(step_size_min, step_size_max),
            method="bounded",
            options={"maxiter": int(max(1, step_size_maxiter))},
        )
        if result.success:
            alpha = float(result.x)
            print(f"Step-size calibration: alpha={alpha:.3g}, diff residual={result.fun:.3e}")
        else:
            print("Step-size calibration failed, reverting to alpha=1.0")

    sigma_est = sigma_bg + alpha * delta_sigma
    delta_sigma_scaled = alpha * delta_sigma
    img_est = EITImage(elem_data=sigma_est, fwd_model=fwd_model)
    pred_vi, _ = fwd_model.fwd_solve(img_est)
    pred_diff = pred_vi.meas - pred_vh.meas
    meas_diff = dv

    res = pred_vi.meas - vi
    rmse_abs = float(np.sqrt(np.mean(res**2)))

    output.mkdir(parents=True, exist_ok=True)

    # Reconstruction plot
    viz = create_visualizer()
    if len(delta_sigma_scaled) == mesh.num_cells():
        node_vals = cell_to_node(mesh, delta_sigma_scaled)
    else:
        node_vals = delta_sigma_scaled
    eidors_style = colormap.lower() in {"eidors_diff", "eidors-diff"}
    format_mode = colorbar_format or ("scientific" if colorbar_scientific else "plain")
    fig = viz.plot_conductivity(
        mesh,
        node_vals,
        title=f"Reconstruction Δσ (lam={lam})",
        colormap=colormap,
        minimal=not eidors_style,
        show_electrodes=True,
        scientific_notation=colorbar_scientific,
        colorbar_format=format_mode,
        transparent=transparent,
    )
    fig.savefig(
        output / "reconstruction.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.15,
        transparent=transparent,
    )
    plt.close(fig)

    # Diff comparison
    # Compute correlation coefficient
    corr_diff = np.corrcoef(meas_diff, pred_diff)[0, 1]
    
    fig = plt.figure(figsize=(12, 5))
    idx = np.arange(len(meas_diff))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(idx, meas_diff, "b-", lw=1.0, label="Measured diff (vi-vh)")
    ax.plot(idx, pred_diff, "r--", lw=1.0, label="Predicted diff")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlabel("Measurement index")
    ax.set_ylabel("Voltage")
    ax.set_title("Diff comparison")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(meas_diff, pred_diff, s=15, alpha=0.7, c='steelblue')
    vmin = min(meas_diff.min(), pred_diff.min())
    vmax = max(meas_diff.max(), pred_diff.max())
    ax2.plot([vmin, vmax], [vmin, vmax], "k--", lw=1.5)
    ax2.set_xlabel("Measured diff")
    ax2.set_ylabel("Predicted diff")
    ax2.grid(alpha=0.3)
    ax2.set_title(f"Scatter (r = {corr_diff:.4f})")
    ax2.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(output / "diff_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Absolute value comparison
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(vi, pred_vi.meas, s=10, alpha=0.7)
    vmin = min(vi.min(), pred_vi.meas.min())
    vmax = max(vi.max(), pred_vi.meas.max())
    ax1.plot([vmin, vmax], [vmin, vmax], "r--")
    ax1.set_title("Measured vs Predicted (abs, real)")
    ax1.grid(alpha=0.3)
    ax1.set_xlabel("Measured target")
    ax1.set_ylabel("Predicted")
    ax2 = fig.add_subplot(1, 2, 2)
    idx = np.arange(len(vi))
    ax2.plot(idx, vi, "b-", lw=1.0, label="Measured target")
    ax2.plot(idx, pred_vi.meas, "r--", lw=1.0, label="Predicted")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "voltage_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        output / "outputs.npz",
        sigma_est=sigma_est,
        delta_sigma=delta_sigma_scaled,
        sigma_bg=sigma_bg,
        dv=meas_diff,
        pred_diff=pred_diff,
        vi=vi,
        pred_vi=pred_vi.meas,
        lambda_=lam,
        rmse_abs=rmse_abs,
        step_size_alpha=alpha,
        pattern_amplitude=stim_amplitude,
        measurement_gain=measurement_gain,
    )
    print(f"RMSE(abs)={rmse_abs:.5f}")
    print("Saved to", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single-step diff reconstruction on tank CSV data")
    parser.add_argument("--csv", type=Path, required=True, help="4-column CSV: ref_re, ref_im, tgt_re, tgt_im")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.1, help="regularization lambda")
    parser.add_argument("--use-part", dest="use_part", choices=["real", "imag", "mag"], default="real", help="which part to reconstruct")
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    parser.add_argument("--pattern-amplitude", type=float, default=None, help="override stimulation amplitude (A)")
    parser.add_argument("--contact-impedance", type=float, default=1e-6, help="contact impedance (Ω·m²)")
    parser.add_argument("--measurement-gain", type=float, default=10.0, help="Divide measured voltages by this amplifier gain")
    parser.add_argument("--step-size-calibration", action="store_true", help="Enable 1-D step-size search on delta_sigma")
    parser.add_argument("--step-size-min", type=float, default=1e-3, help="Lower bound for step-size calibration")
    parser.add_argument("--step-size-max", type=float, default=1e1, help="Upper bound for step-size calibration")
    parser.add_argument("--step-size-maxiter", type=int, default=50, help="Max iterations for bounded optimizer")
    parser.add_argument("--background-sigma", type=float, default=1.0, help="Background conductivity (S/m)")
    parser.add_argument("--colormap", type=str, default="viridis", help="Colormap for reconstruction (e.g., viridis, eidors_diff)")
    parser.add_argument("--colorbar-scientific", action="store_true", help="Use scientific notation on colorbar")
    parser.add_argument("--colorbar-format", type=str, default=None,
                        choices=["plain", "scientific", "matlab_short"],
                        help="Colorbar format rule")
    parser.add_argument("--transparent", action="store_true", help="Save reconstruction with transparent background")
    args = parser.parse_args()
    main(
        csv=args.csv,
        lam=args.lam,
        use_part=args.use_part,
        output=args.output,
        pattern_amplitude=args.pattern_amplitude,
        contact_impedance=args.contact_impedance,
        measurement_gain=args.measurement_gain,
        step_size_calib=args.step_size_calibration,
        step_size_min=args.step_size_min,
        step_size_max=args.step_size_max,
        step_size_maxiter=args.step_size_maxiter,
        background_sigma=args.background_sigma,
        colormap=args.colormap,
        colorbar_scientific=args.colorbar_scientific,
        colorbar_format=args.colorbar_format,
        transparent=args.transparent,
    )
