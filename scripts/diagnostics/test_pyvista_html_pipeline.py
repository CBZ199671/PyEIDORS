#!/usr/bin/env python3
"""Test PyVista HTML export on WSLg/Nix and render existing 3D inverse results.

This script is intentionally lightweight:
1. create a simple sphere smoke-test scene
2. load an existing HDF5 package produced by the 3D inverse demo
3. export interactive HTML scenes for manual inspection
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import vtk
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.hdf5_outputs import read_output_bundle

DEFAULT_INPUT = REPO_ROOT / "results" / "figures_3d_inverse_demo" / "inverse_3d_overview_data.h5"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "figures_3d_inverse_demo" / "pyvista_html_test"


def build_volume(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    resolution: tuple[int, int, int] = (44, 44, 28),
    smooth_sigma: float = 0.85,
) -> tuple[pv.ImageData, dict[str, float]]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    nx, ny, nz = resolution
    xs = np.linspace(mins[0], maxs[0], nx)
    ys = np.linspace(mins[1], maxs[1], ny)
    zs = np.linspace(mins[2], maxs[2], nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    interp = griddata(coords, values, (X, Y, Z), method="linear", fill_value=float(np.min(values)))
    interp = gaussian_filter(interp, sigma=smooth_sigma)

    spacing = (
        float(xs[1] - xs[0]) if nx > 1 else 1.0,
        float(ys[1] - ys[0]) if ny > 1 else 1.0,
        float(zs[1] - zs[0]) if nz > 1 else 1.0,
    )
    grid = pv.ImageData()
    grid.dimensions = np.array(interp.shape) + 1
    grid.origin = (float(xs[0]), float(ys[0]), float(zs[0]))
    grid.spacing = spacing
    grid.cell_data["sigma"] = interp.ravel(order="F")
    return grid, {
        "xmin": float(xs[0]),
        "xmax": float(xs[-1]),
        "ymin": float(ys[0]),
        "ymax": float(ys[-1]),
        "zmin": float(zs[0]),
        "zmax": float(zs[-1]),
        "sigma_min": float(np.min(interp)),
        "sigma_max": float(np.max(interp)),
    }


def add_threshold_surface(
    plotter: pv.Plotter,
    volume: pv.ImageData,
    *,
    threshold: float,
    cmap: str,
    scalar_bar_args: dict | None = None,
) -> float:
    values = np.asarray(volume.cell_data["sigma"], dtype=float)
    candidates = [
        float(threshold),
        float(np.percentile(values, 98.0)),
        float(np.percentile(values, 96.0)),
        float(np.percentile(values, 94.0)),
        float(np.percentile(values, 92.0)),
        float(np.percentile(values, 90.0)),
    ]
    surface = None
    used_threshold = float(threshold)
    for candidate in candidates:
        trial = volume.threshold(value=float(candidate), scalars="sigma")
        if trial.n_points > 0:
            surface = trial
            used_threshold = float(candidate)
            break
    if surface is None:
        surface = volume.outline()
    plotter.add_mesh(
        surface,
        scalars="sigma",
        cmap=cmap,
        opacity=0.92,
        scalar_bar_args=scalar_bar_args,
    )
    return used_threshold


def add_cylinder_outline(plotter: pv.Plotter, bounds: dict[str, float]) -> None:
    center = (
        0.5 * (bounds["xmin"] + bounds["xmax"]),
        0.5 * (bounds["ymin"] + bounds["ymax"]),
        0.5 * (bounds["zmin"] + bounds["zmax"]),
    )
    radius = 0.5 * max(bounds["xmax"] - bounds["xmin"], bounds["ymax"] - bounds["ymin"])
    height = bounds["zmax"] - bounds["zmin"]
    cyl = pv.Cylinder(center=center, direction=(0, 0, 1), radius=radius, height=height, resolution=80)
    plotter.add_mesh(cyl.extract_feature_edges(), color="black", line_width=1.0, opacity=0.18)


def build_scene_html(package_path: Path, output_dir: Path) -> dict[str, str]:
    payload = read_output_bundle(package_path)
    coords = np.asarray(payload["coords"], dtype=float)
    truth = np.asarray(payload["truth_sigma"], dtype=float)
    recon = np.asarray(payload["recon_sigma"], dtype=float)

    truth_grid, truth_bounds = build_volume(coords, truth)
    recon_grid, recon_bounds = build_volume(coords, recon)
    baseline = float(np.min(truth))
    truth_threshold = baseline + 0.55 * (float(np.max(truth)) - baseline)
    recon_threshold = max(
        baseline + 0.70 * (float(np.max(recon)) - baseline),
        float(np.percentile(recon, 97.5)),
    )

    plotter = pv.Plotter(shape=(1, 2), notebook=False, off_screen=True, window_size=(1600, 760))
    plotter.set_background("white")

    plotter.subplot(0, 0)
    truth_used_threshold = add_threshold_surface(
        plotter,
        truth_grid,
        threshold=truth_threshold,
        cmap="viridis",
        scalar_bar_args={"title": "Truth sigma", "vertical": True},
    )
    add_cylinder_outline(plotter, truth_bounds)
    plotter.view_isometric()

    plotter.subplot(0, 1)
    recon_used_threshold = add_threshold_surface(
        plotter,
        recon_grid,
        threshold=recon_threshold,
        cmap="inferno",
        scalar_bar_args={"title": "Recon sigma", "vertical": True},
    )
    add_cylinder_outline(plotter, recon_bounds)
    plotter.view_isometric()

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "inverse_3d_interactive.html"
    vtksz_path = output_dir / "inverse_3d_interactive.vtksz"
    png_path = output_dir / "inverse_3d_interactive.png"
    plotter.screenshot(png_path)
    plotter.export_html(html_path)
    plotter.export_vtksz(vtksz_path)

    return {
        "html": str(html_path),
        "vtksz": str(vtksz_path),
        "png": str(png_path),
        "truth_threshold": f"{truth_used_threshold:.8f}",
        "recon_threshold": f"{recon_used_threshold:.8f}",
    }


def build_smoke_html(output_dir: Path) -> dict[str, str]:
    plotter = pv.Plotter(notebook=False, off_screen=True, window_size=(900, 700))
    plotter.set_background("white")
    plotter.add_mesh(pv.Sphere(radius=0.45, theta_resolution=48, phi_resolution=48), color="tomato", smooth_shading=True)
    plotter.view_isometric()
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "smoke_test.html"
    vtksz_path = output_dir / "smoke_test.vtksz"
    png_path = output_dir / "smoke_test.png"
    plotter.screenshot(png_path)
    plotter.export_html(html_path)
    plotter.export_vtksz(vtksz_path)
    return {"html": str(html_path), "vtksz": str(vtksz_path), "png": str(png_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    pv.global_theme.window_size = [1200, 800]

    summary: dict[str, object] = {
        "display": os.environ.get("DISPLAY"),
        "wayland_display": os.environ.get("WAYLAND_DISPLAY"),
        "pyvista_version": pv.__version__,
        "vtk_version": vtk.vtkVersion.GetVTKVersion(),
        "input": str(args.input),
        "output_dir": str(args.output_dir),
    }

    smoke = build_smoke_html(args.output_dir)
    summary["smoke"] = smoke

    if args.input.exists():
        summary["inverse_scene"] = build_scene_html(args.input, args.output_dir)
    else:
        summary["inverse_scene"] = {"skipped": f"missing input: {args.input}"}

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
