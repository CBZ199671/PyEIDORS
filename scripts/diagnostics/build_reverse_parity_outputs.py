#!/usr/bin/env python3
"""Build reverse-parity comparison assets (PyEIDORS on EIDORS data)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_forward_csv(path: Path, delimiter: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    tokens = [token.strip() for token in first_line.split(delimiter) if token.strip()]
    has_header = False
    for token in tokens:
        try:
            float(token)
        except ValueError:
            has_header = True
            break

    data = np.loadtxt(path, delimiter=delimiter, skiprows=1 if has_header else 0)
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {path}; got shape {data.shape}.")

    baseline = data[:, 0]
    phantom = data[:, 1]
    diff = data[:, 2] if data.shape[1] >= 3 else phantom - baseline
    return baseline, phantom, diff


def rotate_reconstruction_only(image_path: Path, output_path: Path) -> None:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        arr = np.array(img)

    h, w, _ = arr.shape
    near_white = np.all(arr > 250, axis=2)
    nonwhite_frac = 1.0 - near_white.mean(axis=0)

    # Identify non-white segments to locate the colorbar region.
    nonwhite_cols = np.where(nonwhite_frac > 0.2)[0]
    if nonwhite_cols.size == 0:
        img.save(output_path)
        return

    segments = []
    start = nonwhite_cols[0]
    prev = nonwhite_cols[0]
    for c in nonwhite_cols[1:]:
        if c == prev + 1:
            prev = c
            continue
        segments.append((start, prev))
        start = c
        prev = c
    segments.append((start, prev))

    colorbar_start = None
    for seg_start, seg_end in sorted(segments, key=lambda s: s[0], reverse=True):
        width = seg_end - seg_start + 1
        if width <= int(0.2 * w):
            colorbar_start = seg_start
            break

    if colorbar_start is None:
        colorbar_start = w

    panel = arr[:, :colorbar_start, :]
    panel_nonwhite = np.where(~np.all(panel > 250, axis=2))
    if panel_nonwhite[0].size == 0 or panel_nonwhite[1].size == 0:
        img.save(output_path)
        return

    y0, y1 = panel_nonwhite[0].min(), panel_nonwhite[0].max()
    x0, x1 = panel_nonwhite[1].min(), panel_nonwhite[1].max()

    # Rotate a square region to avoid distortion.
    box_w = x1 - x0 + 1
    box_h = y1 - y0 + 1
    side = min(box_w, box_h)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    sx0 = max(0, cx - side // 2)
    sy0 = max(0, cy - side // 2)
    sx1 = sx0 + side
    sy1 = sy0 + side
    if sx1 > panel.shape[1]:
        shift = sx1 - panel.shape[1]
        sx0 = max(0, sx0 - shift)
        sx1 = sx0 + side
    if sy1 > panel.shape[0]:
        shift = sy1 - panel.shape[0]
        sy0 = max(0, sy0 - shift)
        sy1 = sy0 + side

    roi = panel[sy0:sy1, sx0:sx1, :]
    roi_img = Image.fromarray(roi)
    rotated = roi_img.transpose(Image.ROTATE_270).transpose(Image.FLIP_TOP_BOTTOM)
    rotated_arr = np.array(rotated)

    panel_out = panel.copy()
    panel_out[sy0:sy1, sx0:sx1, :] = rotated_arr

    out = arr.copy()
    out[:, :colorbar_start, :] = panel_out

    Image.fromarray(out).save(output_path)


def build_delta_v_plot(measured: np.ndarray,
                        eidors_pred: np.ndarray,
                        pyeidors_pred: np.ndarray,
                        output_path: Path) -> None:
    idx = np.arange(1, measured.size + 1)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(idx, measured, label="Measured ΔV", color="black", linewidth=1.0)
    ax.plot(idx, eidors_pred, label="EIDORS ΔV_pred", color="tab:red", linewidth=1.2)
    ax.plot(idx, pyeidors_pred, label="PyEIDORS ΔV_pred", color="tab:green", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Measurement index")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Boundary voltage differences")
    ax.legend(loc="best")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eidors-forward", type=Path, required=True,
                        help="CSV with baseline + phantom voltages from EIDORS.")
    parser.add_argument("--eidors-pred-diff", type=Path, required=True,
                        help="CSV with EIDORS-predicted difference voltages.")
    parser.add_argument("--pyeidors-output", type=Path, required=True,
                        help="Output directory from run_synthetic_parity.py.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to store comparison assets.")
    parser.add_argument("--delimiter", type=str, default=",")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, _, measured_diff = load_forward_csv(args.eidors_forward, args.delimiter)
    eidors_pred = np.loadtxt(args.eidors_pred_diff, delimiter=args.delimiter).reshape(-1)

    pyeidors_pred_path = args.pyeidors_output / "difference" / "predicted_difference.csv"
    pyeidors_pred = np.loadtxt(pyeidors_pred_path, delimiter=",").reshape(-1)

    np.savetxt(args.output_dir / "eidors_measured_diff.csv", measured_diff, delimiter=",")
    np.savetxt(args.output_dir / "pyeidors_predicted_diff.csv", pyeidors_pred, delimiter=",")

    raw_py = args.pyeidors_output / "difference" / "reconstruction.png"
    raw_dst = args.output_dir / "pyeidors_reconstruction_raw.png"
    affine_dst = args.output_dir / "pyeidors_reconstruction_affine.png"

    if raw_py.exists():
        raw_dst.write_bytes(raw_py.read_bytes())
        rotate_reconstruction_only(raw_dst, affine_dst)

    delta_plot = args.output_dir / "deltaV_comparison.png"
    build_delta_v_plot(measured_diff, eidors_pred, pyeidors_pred, delta_plot)


if __name__ == "__main__":
    main()
