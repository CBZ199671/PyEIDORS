#!/usr/bin/env python3
"""Plot 16-stimulation point-status schematic for all acquisition modes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from pyeidors.data.digit_plot import configure_times_new_roman
from pyeidors.data.holdout_point_audit import (
    drive_removed_frame_indices,
    far3_frame_indices,
)
from pyeidors.runtime_paths import pyeidors_output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw a 4x4 schematic showing local adjacent measurement points "
            "for each stimulation frame."
        ),
    )
    parser.add_argument("--n-elec", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=pyeidors_output_path("eit_bucket_all_modes_16stim_point_status.png"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args(argv)


def _electrode_points(n_elec: int) -> np.ndarray:
    angles = np.pi / 2.0 - 2.0 * np.pi * np.arange(n_elec, dtype=float) / n_elec
    points = np.empty((int(n_elec), 2), dtype=np.float64)
    np.cos(angles, out=points[:, 0])
    np.sin(angles, out=points[:, 1])
    return points


def plot_16stim_point_status(
    output_path: Path,
    *,
    n_elec: int = 16,
    dpi: int = 220,
) -> Path:
    """Plot the point status for every stimulation frame."""

    configure_times_new_roman()
    output = Path(output_path).with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    near_local = drive_removed_frame_indices(n_elec)
    far_local = far3_frame_indices(n_elec)
    xy = _electrode_points(n_elec)
    colors = {
        "near": "#d62728",
        "far": "#1f77b4",
        "train": "#2ca02c",
    }

    fig, axes = plt.subplots(4, 4, figsize=(16.8, 17.2))
    fig.subplots_adjust(
        left=0.035,
        right=0.985,
        top=0.89,
        bottom=0.075,
        wspace=0.16,
        hspace=0.24,
    )
    fig.suptitle(
        "16 stimulation frames: adjacent measurement point status",
        fontsize=22,
        y=0.982,
    )
    fig.text(
        0.5,
        0.952,
        "Local frame indices are relative to each stimulation pair. "
        "Near 3 = {15,0,1}; Far 3 = {7,8,9}.",
        ha="center",
        va="top",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.932,
        "Modes: full_256 keeps all 16; full_208 removes near 3; "
        "far3_drop_near3_keep_208 removes far 3; raw_160 removes near 3 + far 3; "
        "fitted_208 uses train 10 and predicts far 3.",
        ha="center",
        va="top",
        fontsize=11,
    )

    for stim, ax in enumerate(axes.ravel()):
        ax.add_patch(plt.Circle((0.0, 0.0), 1.0, fill=False, color="#222222", lw=1.2))
        ax.set_aspect("equal")
        ax.set_xlim(-1.22, 1.22)
        ax.set_ylim(-1.22, 1.22)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"stim {stim}: drive E{stim}-E{(stim + 1) % n_elec}",
            fontsize=12,
            pad=3,
        )

        for elec in range(n_elec):
            ax.text(
                xy[elec, 0] * 1.12,
                xy[elec, 1] * 1.12,
                str(elec),
                ha="center",
                va="center",
                fontsize=7.5,
                color="#777777",
            )

        drive_e1 = stim
        drive_e2 = (stim + 1) % n_elec
        ax.scatter(
            [xy[drive_e1, 0], xy[drive_e2, 0]],
            [xy[drive_e1, 1], xy[drive_e2, 1]],
            s=115,
            marker="*",
            color="#d62728",
            edgecolor="black",
            linewidth=0.6,
            zorder=5,
        )

        for local_idx in range(n_elec):
            meas_start = (stim + local_idx) % n_elec
            meas_end = (meas_start + 1) % n_elec
            pair_center = xy[[meas_start, meas_end]].mean(axis=0)
            pair_center = pair_center / np.linalg.norm(pair_center) * 0.82
            if local_idx in near_local:
                kind = "near"
                marker = "X"
                size = 95
            elif local_idx in far_local:
                kind = "far"
                marker = "D"
                size = 78
            else:
                kind = "train"
                marker = "o"
                size = 54
            ax.scatter(
                pair_center[0],
                pair_center[1],
                s=size,
                marker=marker,
                color=colors[kind],
                edgecolor="white",
                linewidth=0.75,
                zorder=4,
            )
            ax.text(
                pair_center[0],
                pair_center[1] - 0.105,
                str(local_idx),
                ha="center",
                va="center",
                fontsize=7.2,
                color="#111111",
                zorder=6,
            )

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#444444")

    legend_items = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#d62728",
            markeredgecolor="black",
            markersize=12,
            label="drive electrodes",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="none",
            markerfacecolor=colors["near"],
            markeredgecolor="white",
            markersize=10,
            label="near 3 local {15,0,1}",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=colors["far"],
            markeredgecolor="white",
            markersize=9,
            label="far 3 local {7,8,9}",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors["train"],
            markeredgecolor="white",
            markersize=8,
            label="kept/train 10",
        ),
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=4,
        frameon=True,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.012),
    )
    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = plot_16stim_point_status(
        args.output,
        n_elec=args.n_elec,
        dpi=args.dpi,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
