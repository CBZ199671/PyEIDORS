#!/usr/bin/env python3
"""Render conductivity and voltage-fit plots for a same-geometry interop result."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pyeidors.interop import build_mesh_from_exchange_mat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-mat", type=Path, required=True)
    parser.add_argument("--details-mat", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--title-prefix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    mesh, _ = build_mesh_from_exchange_mat(args.geometry_mat)
    details = loadmat(args.details_mat, squeeze_me=True, struct_as_record=False)
    truth = np.asarray(details["truth_elem_data"], dtype=float).reshape(-1)
    recon = np.asarray(details["recon_elem_data"], dtype=float).reshape(-1)
    target_diff = np.asarray(details["target_diff"], dtype=float).reshape(-1)
    predicted_diff = np.asarray(details["predicted_diff"], dtype=float).reshape(-1)
    mesh_name = str(np.asarray(details.get("mesh_name", mesh.mesh_name)).reshape(-1)[0])
    voltage_rmse = float(np.asarray(details["voltage_rmse"]).reshape(-1)[0])
    conductivity_rmse = float(np.asarray(details["conductivity_rmse"]).reshape(-1)[0])

    title_prefix = args.title_prefix.replace("_", " ").strip()
    coordinates = mesh.coordinates()
    cells = mesh.cells()
    triangulation = tri.Triangulation(coordinates[:, 0], coordinates[:, 1], cells)

    cond_path = Path(f"{args.output_prefix}_conductivity.png")
    error = np.abs(recon - truth)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    if len(truth) == mesh.num_cells():
        vmin = min(float(np.min(truth)), float(np.min(recon)))
        vmax = max(float(np.max(truth)), float(np.max(recon)))
        im_true = axes[0].tripcolor(triangulation, facecolors=truth, cmap="viridis", shading="flat", vmin=vmin, vmax=vmax)
        im_recon = axes[1].tripcolor(triangulation, facecolors=recon, cmap="viridis", shading="flat", vmin=vmin, vmax=vmax)
        im_err = axes[2].tripcolor(triangulation, facecolors=error, cmap="hot", shading="flat")
    else:
        vmin = min(float(np.min(truth)), float(np.min(recon)))
        vmax = max(float(np.max(truth)), float(np.max(recon)))
        im_true = axes[0].tripcolor(triangulation, truth, cmap="viridis", shading="gouraud", vmin=vmin, vmax=vmax)
        im_recon = axes[1].tripcolor(triangulation, recon, cmap="viridis", shading="gouraud", vmin=vmin, vmax=vmax)
        im_err = axes[2].tripcolor(triangulation, error, cmap="hot", shading="gouraud")

    images = [im_true, im_recon, im_err]
    for ax, image in zip(axes, images):
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, shrink=0.82)

    fig.text(
        0.5,
        0.04,
        f"Conductivity RMSE = {conductivity_rmse:.4e}",
        ha="center",
        va="center",
        fontsize=13,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.6", "alpha": 0.92},
    )
    fig.tight_layout()
    fig.savefig(cond_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    voltage_path = Path(f"{args.output_prefix}_voltage.png")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(target_diff, color="black", linewidth=1.8, label="Measured ΔV")
    ax.plot(predicted_diff, color="#d62728", linewidth=1.8, label="Predicted ΔV")
    ax.set_xlabel("Measurement index")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.text(
        0.02,
        0.96,
        f"Voltage RMSE = {voltage_rmse:.4e}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.6", "alpha": 0.92},
    )
    fig.tight_layout()
    fig.savefig(voltage_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {cond_path}")
    print(f"Wrote {voltage_path}")


if __name__ == "__main__":
    main()
