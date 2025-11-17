#!/usr/bin/env python3
"""对实测差分 CSV 数据做高斯牛顿单步重构（EIDORS 风格雅可比）。

用法示例：
    python scripts/run_single_step_diff_realdata.py \
        --csv data/measurements/tank/2025-11-14-22-18-02_1_10.00_50uA_3000Hz.csv \
        --lambda 0.1 \
        --output results/tank/single_step_from_cli
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pyeidors.data.structures import PatternConfig, EITImage
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.forward.eit_forward_model import EITForwardModel
from pyeidors.inverse.jacobian.adjoint_jacobian import EidorsStyleAdjointJacobian
from pyeidors.inverse.regularization.smoothness import NOSERRegularization
from pyeidors.visualization import create_visualizer


def load_csv(csv_path: Path, use_part: str = "real") -> tuple[np.ndarray, np.ndarray]:
    """加载 4 列 CSV: ref_re, ref_im, tgt_re, tgt_im."""
    arr = np.loadtxt(csv_path, delimiter=",")
    if arr.shape[1] < 4:
        raise ValueError("期望 4 列: ref_re, ref_im, tgt_re, tgt_im")
    ref_re, ref_im, tgt_re, tgt_im = arr.T
    if use_part == "real":
        return ref_re, tgt_re
    elif use_part == "imag":
        return ref_im, tgt_im
    elif use_part == "mag":
        return np.abs(ref_re + 1j * ref_im), np.abs(tgt_re + 1j * tgt_im)
    else:
        raise ValueError(f"未知 use_part={use_part}")


def cell_to_node(mesh, cell_values: np.ndarray) -> np.ndarray:
    node_vals = np.zeros(mesh.num_vertices())
    counts = np.zeros(mesh.num_vertices())
    for ci, cell in enumerate(mesh.cells()):
        for v in cell:
            node_vals[v] += cell_values[ci]
            counts[v] += 1
    counts[counts == 0] = 1
    node_vals /= counts
    return node_vals


def main(
    csv: Path,
    lam: float,
    use_part: str,
    output: Path,
    mesh_name: str = "mesh_102070",
    contact_impedance: float = 1e-6,
):
    vh, vi = load_csv(csv, use_part=use_part)

    mesh = load_or_create_mesh(mesh_dir="eit_meshes", mesh_name=mesh_name, n_elec=16)
    pattern_cfg = PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        amplitude=1.0,
        use_meas_current=False,
        rotate_meas=True,
    )
    z_contact = np.full(16, contact_impedance, dtype=float)
    fwd_model = EITForwardModel(n_elec=16, pattern_config=pattern_cfg, z=z_contact, mesh=mesh)

    n_elem = len(fwd_model.V_sigma.dofmap().dofs())
    sigma_bg = np.ones(n_elem)
    img_bg = EITImage(elem_data=sigma_bg, fwd_model=fwd_model)

    jac_calc = EidorsStyleAdjointJacobian(fwd_model, use_torch=False)
    J = jac_calc.calculate_from_image(img_bg)

    dv = vi - vh
    if dv.shape[0] != J.shape[0]:
        raise RuntimeError(f"数据长度 {dv.shape[0]} 与雅可比行数 {J.shape[0]} 不一致")

    reg = NOSERRegularization(fwd_model, jac_calc, base_conductivity=1.0, alpha=1.0)
    R = reg.get_regularization_matrix()

    A = J.T @ J + lam * R
    b = J.T @ dv
    try:
        delta_sigma = np.linalg.solve(A, b)
    except Exception:
        delta_sigma = np.linalg.lstsq(A, b, rcond=None)[0]

    sigma_est = sigma_bg + delta_sigma
    img_est = EITImage(elem_data=sigma_est, fwd_model=fwd_model)

    pred_vh, _ = fwd_model.fwd_solve(img_bg)
    pred_vi, _ = fwd_model.fwd_solve(img_est)
    pred_diff = pred_vi.meas - pred_vh.meas
    meas_diff = dv

    res = pred_vi.meas - vi
    rmse_abs = float(np.sqrt(np.mean(res**2)))

    output.mkdir(parents=True, exist_ok=True)

    # 重构图
    viz = create_visualizer()
    if len(sigma_est) == mesh.num_cells():
        node_vals = cell_to_node(mesh, sigma_est)
    else:
        node_vals = sigma_est
    fig = viz.plot_conductivity(mesh, node_vals, title=f"Reconstruction (lam={lam})", minimal=True)
    fig.savefig(output / "reconstruction.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 差分对比
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
    ax2.scatter(meas_diff, pred_diff, s=10, alpha=0.7)
    vmin = min(meas_diff.min(), pred_diff.min())
    vmax = max(meas_diff.max(), pred_diff.max())
    ax2.plot([vmin, vmax], [vmin, vmax], "k--")
    ax2.set_xlabel("Measured diff")
    ax2.set_ylabel("Predicted diff")
    ax2.grid(alpha=0.3)
    ax2.set_title("Scatter")
    fig.tight_layout()
    fig.savefig(output / "diff_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 绝对值对比
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
        dv=meas_diff,
        pred_diff=pred_diff,
        vi=vi,
        pred_vi=pred_vi.meas,
        lambda_=lam,
        rmse_abs=rmse_abs,
    )
    print(f"RMSE(abs)={rmse_abs:.5f}")
    print("Saved to", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single-step diff reconstruction on tank CSV data")
    parser.add_argument("--csv", type=Path, required=True, help="4-column CSV: ref_re, ref_im, tgt_re, tgt_im")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.1, help="regularization lambda")
    parser.add_argument("--use-part", dest="use_part", choices=["real", "imag", "mag"], default="real", help="which part to reconstruct")
    parser.add_argument("--output", type=Path, default=Path("results/tank/single_step_cli"), help="output directory")
    args = parser.parse_args()
    main(csv=args.csv, lam=args.lam, use_part=args.use_part, output=args.output)
