#!/usr/bin/env python3
"""Forward p-refinement experiment for the CEM solver.

The experiment keeps the mesh and DG0 conductivity parameterization fixed while
raising the Lagrange order of the electric-potential finite-element space.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pyeidors.data.structures import EITImage, PatternConfig
from pyeidors.femx import build_eit_mesh, cell_midpoints
from pyeidors.forward.eit_forward_model import EITForwardModel
from pyeidors.forward.process_setup_cache import clear_process_forward_setup_cache
from pyeidors.geometry.simple_mesh_generator import create_simple_eit_mesh
from pyeidors.utils.numeric_ops import squared_distances_to_point

try:
    from pyeidors.geometry.mesh3d_generator import (
        GMSH_AVAILABLE,
        create_cylinder_3d_eit_mesh,
    )
except Exception:  # pragma: no cover - optional gmsh path
    GMSH_AVAILABLE = False
    create_cylinder_3d_eit_mesh = None  # type: ignore[assignment]


def _parse_orders(raw: str) -> list[int]:
    orders = [int(token.strip()) for token in str(raw).split(",") if token.strip()]
    if not orders:
        raise ValueError("--orders must contain at least one positive integer")
    if any(order < 1 for order in orders):
        raise ValueError("--orders values must be >= 1")
    seen: set[int] = set()
    deduped: list[int] = []
    for order in orders:
        if order in seen:
            continue
        seen.add(order)
        deduped.append(order)
    return deduped


def _make_tagged_unit_square(*, n_elec: int, refinement: int):
    mesh = dmesh.create_unit_square(MPI.COMM_WORLD, refinement, refinement)
    fdim = mesh.topology.dim - 1
    boundary_facets = dmesh.locate_entities_boundary(
        mesh,
        fdim,
        lambda x: np.full(x.shape[1], True, dtype=bool),
    ).astype(np.int32)
    mesh.topology.create_connectivity(fdim, 0)
    f2v = mesh.topology.connectivity(fdim, 0)
    coords = mesh.geometry.x[:, :2]

    centroids = np.zeros((boundary_facets.size, 2), dtype=np.float64)
    for idx, facet in enumerate(boundary_facets):
        centroids[idx, :] = coords[f2v.links(int(facet))].mean(axis=0)

    x = centroids[:, 0]
    y = centroids[:, 1]
    eps = 1.0e-10
    t = np.zeros_like(x)
    left = np.isclose(x, 0.0, atol=eps)
    top = (~left) & np.isclose(y, 1.0, atol=eps)
    right = (~left) & (~top) & np.isclose(x, 1.0, atol=eps)
    bottom = (~left) & (~top) & (~right) & np.isclose(y, 0.0, atol=eps)
    t[left] = y[left]
    t[top] = 1.0 + x[top]
    t[right] = 2.0 + (1.0 - y[right])
    t[bottom] = 3.0 + (1.0 - x[bottom])

    segment = 4.0 / float(n_elec)
    tags = (
        np.floor(np.clip(t, 0.0, 4.0 - 1.0e-12) / segment).astype(np.int32) + 2
    ).astype(np.int32)
    order = np.argsort(boundary_facets)
    facet_tags = dmesh.meshtags(mesh, fdim, boundary_facets[order], tags[order])
    association = {f"electrode_{idx + 1}": idx + 2 for idx in range(n_elec)}
    return build_eit_mesh(
        mesh,
        facet_tags=facet_tags,
        association_table=association,
        radius=1.0,
        mesh_family="triangle",
        geometry_version="unit_square",
        generator_revision="p-refinement-v1",
    )


def _make_mesh(args: argparse.Namespace):
    if int(args.mesh_dim) == 2:
        if str(args.mesh_shape).strip().lower() == "circle":
            return create_simple_eit_mesh(
                n_elec=int(args.n_elec),
                radius=float(args.radius),
                mesh_size=float(args.radius)
                / max(2.0 * float(args.mesh_refinement), 1.0),
                electrode_coverage=float(args.electrode_coverage),
                output_dir=str(Path(args.out_dir) / "mesh"),
            )
        return _make_tagged_unit_square(
            n_elec=int(args.n_elec),
            refinement=int(args.mesh_refinement),
        )
    if not GMSH_AVAILABLE or create_cylinder_3d_eit_mesh is None:
        raise RuntimeError("3D p-refinement experiment requires gmsh availability")
    return create_cylinder_3d_eit_mesh(
        n_elec=int(args.n_elec),
        radius=float(args.radius),
        height=float(args.height),
        refinement=int(args.mesh_refinement),
        electrode_coverage=float(args.electrode_coverage),
        electrode_height_ratio=float(args.electrode_height_ratio),
        output_dir=str(Path(args.out_dir) / "mesh"),
        mesh_name="p_refinement_forward_mesh",
    )


def _conductivity(mesh, centers: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    sigma = np.full(len(centers), float(args.background_conductivity), dtype=np.float64)
    if centers.size == 0 or float(args.anomaly_radius) <= 0.0:
        return sigma
    coords = np.asarray(centers, dtype=np.float64)
    if int(mesh.topology.dim) == 2:
        center = np.asarray([args.anomaly_x, args.anomaly_y], dtype=np.float64)
        distance2 = squared_distances_to_point(coords, center, ndim=2)
    else:
        center = np.asarray(
            [args.anomaly_x, args.anomaly_y, args.anomaly_z], dtype=np.float64
        )
        distance2 = squared_distances_to_point(coords, center, ndim=3)
    sigma[distance2 <= float(args.anomaly_radius) ** 2] = float(
        args.anomaly_conductivity
    )
    return sigma


def _dof_count(function_space) -> int:
    dofmap = function_space.dofmap
    return int(dofmap.index_map.size_local * dofmap.index_map_bs)


def _run_order(
    *,
    mesh,
    order: int,
    sigma_values: np.ndarray,
    pattern: PatternConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    init_start = time.perf_counter()
    fwd = EITForwardModel(
        n_elec=int(args.n_elec),
        pattern_config=pattern,
        z=np.full(int(args.n_elec), float(args.contact_impedance), dtype=np.float64),
        mesh=mesh,
        linear_backend=str(args.linear_backend),
        backend_config={
            "solver_preset": str(args.solver_preset),
            "petsc_device": str(args.petsc_device),
            "mat_solve_mode": str(args.forward_mat_solve),
        },
        potential_order=int(order),
    )
    init_seconds = time.perf_counter() - init_start

    image = EITImage(elem_data=sigma_values.copy(), fwd_model=fwd)
    solve_start = time.perf_counter()
    data, electrode_potentials = fwd.fwd_solve(image)
    solve_seconds = time.perf_counter() - solve_start
    diag = fwd.get_backend_diagnostics()
    measurements = np.asarray(data.meas, dtype=np.float64)
    return {
        "potential_order": int(order),
        "potential_dofs": _dof_count(fwd.V),
        "conductivity_dofs": _dof_count(fwd.V_sigma),
        "n_cells": int(mesh.num_cells()),
        "n_vertices": int(mesh.num_vertices()),
        "n_stim": int(fwd.pattern_manager.n_stim),
        "n_measurements": int(measurements.size),
        "init_seconds": float(init_seconds),
        "solve_seconds": float(solve_seconds),
        "max_abs_voltage": float(np.max(np.abs(measurements))),
        "l2_voltage": float(np.linalg.norm(measurements)),
        "measurements": measurements,
        "electrode_potentials": np.asarray(electrode_potentials, dtype=np.float64),
        "stim_matrix": np.asarray(fwd.pattern_manager.stim_matrix, dtype=np.float64),
        "backend": str(args.linear_backend),
        "solver_preset": str(diag.get("solver_preset", args.solver_preset)),
        "static_setup_cache_hit": bool(
            (diag.get("static_setup_lookup") or {}).get("hit", False)
            if isinstance(diag.get("static_setup_lookup"), dict)
            else False
        ),
    }


def _strip_arrays(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"measurements", "electrode_potentials", "stim_matrix"}
    }


def _configure_plot_fonts() -> None:
    for font_path in (
        "/mnt/c/Windows/Fonts/times.ttf",
        "/mnt/c/Windows/Fonts/timesbd.ttf",
        "/mnt/c/Windows/Fonts/timesi.ttf",
        "/mnt/c/Windows/Fonts/timesbi.ttf",
    ):
        path = Path(font_path)
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )


def _plot_voltage_overlay(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=160)
    for row in rows:
        measurements = np.asarray(row["measurements"], dtype=np.float64)
        ax.plot(
            np.arange(measurements.size),
            measurements,
            linewidth=1.6,
            label=f"P{row['potential_order']}",
        )
    ax.set_xlabel("Measurement-vector index")
    ax.set_ylabel("Measured voltage difference")
    ax.set_title("Concatenated measurement vector by polynomial order")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "measurement_vector_overlay.png")
    plt.close(fig)


def _plot_voltage_error(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    reference = np.asarray(rows[-1]["measurements"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=160)
    for row in rows[:-1]:
        measurements = np.asarray(row["measurements"], dtype=np.float64)
        ax.plot(
            np.arange(reference.size),
            np.abs(measurements - reference),
            linewidth=1.6,
            label=f"P{row['potential_order']} vs P{rows[-1]['potential_order']}",
        )
    ax.set_xlabel("Measurement-vector index")
    ax.set_ylabel("Absolute measured-voltage error")
    ax.set_title("Channelwise measurement error against highest-order reference")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "measurement_vector_error_vs_reference.png")
    plt.close(fig)


def _unwrapped_electrode_order(
    rows: list[dict[str, Any]], *, stim_index: int
) -> tuple[np.ndarray, dict[str, int]]:
    potentials = np.asarray(rows[0]["electrode_potentials"], dtype=np.float64)
    n_elec = int(potentials.shape[1])
    if n_elec <= 0:
        return np.empty(0, dtype=int), {}
    stim_matrix = np.asarray(rows[0].get("stim_matrix"), dtype=np.float64)
    if stim_matrix.ndim != 2 or stim_matrix.shape[0] <= int(stim_index):
        return np.arange(n_elec, dtype=int), {}

    stim = stim_matrix[int(stim_index), :n_elec]
    positive_candidates = np.flatnonzero(stim > 0.0)
    negative_candidates = np.flatnonzero(stim < 0.0)
    if positive_candidates.size == 0 or negative_candidates.size == 0:
        return np.arange(n_elec, dtype=int), {}

    positive = int(positive_candidates[np.argmax(stim[positive_candidates])])
    negative = int(negative_candidates[np.argmin(stim[negative_candidates])])
    distance_ccw = (negative - positive) % n_elec
    distance_cw = (positive - negative) % n_elec
    step = 1 if distance_ccw >= distance_cw else -1
    order = (positive + step * np.arange(n_elec, dtype=int)) % n_elec
    positions = {
        "positive": int(np.flatnonzero(order == positive)[0]),
        "negative": int(np.flatnonzero(order == negative)[0]),
    }
    return order.astype(int, copy=False), positions


def _annotate_drive_electrodes(
    ax, x: np.ndarray, y: np.ndarray, positions: dict[str, int]
) -> None:
    if "positive" in positions:
        pos_x = int(positions["positive"])
        ax.scatter(
            [x[pos_x]],
            [y[pos_x]],
            s=96,
            marker="^",
            color="#c1121f",
            edgecolor="white",
            zorder=5,
        )
        ax.annotate(
            "I+",
            (x[pos_x], y[pos_x]),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            color="#c1121f",
            fontsize=10,
            fontweight="bold",
        )
    if "negative" in positions:
        neg_x = int(positions["negative"])
        ax.scatter(
            [x[neg_x]],
            [y[neg_x]],
            s=96,
            marker="v",
            color="#003049",
            edgecolor="white",
            zorder=5,
        )
        ax.annotate(
            "I-",
            (x[neg_x], y[neg_x]),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            color="#003049",
            fontsize=10,
            fontweight="bold",
        )


def _plot_electrode_potential_profile(
    out_dir: Path, rows: list[dict[str, Any]], *, stim_index: int
) -> None:
    n_stim = int(np.asarray(rows[0]["electrode_potentials"]).shape[0])
    resolved_stim = min(max(int(stim_index), 0), max(n_stim - 1, 0))
    order, drive_positions = _unwrapped_electrode_order(rows, stim_index=resolved_stim)
    x = np.arange(order.size)
    labels = [str(int(index) + 1) for index in order]
    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=160)
    highest_profile = None
    for row in rows:
        potentials = np.asarray(row["electrode_potentials"], dtype=np.float64)
        profile = potentials[resolved_stim, order]
        if row is rows[-1]:
            highest_profile = profile
        ax.plot(
            x,
            profile,
            marker="o",
            linewidth=1.6,
            label=f"P{row['potential_order']}",
        )
    if highest_profile is not None:
        _annotate_drive_electrodes(ax, x, highest_profile, drive_positions)
    ax.set_xlabel("Unwrapped circular electrode order")
    ax.set_ylabel("Electrode potential")
    ax.set_title(f"Drive-cut electrode potential profile, stimulation {resolved_stim}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        out_dir / f"electrode_potential_profile_drive_cut_stim{resolved_stim}.png"
    )
    plt.close(fig)


def _plot_electrode_potential_error(
    out_dir: Path, rows: list[dict[str, Any]], *, stim_index: int
) -> None:
    reference_all = np.asarray(rows[-1]["electrode_potentials"], dtype=np.float64)
    resolved_stim = min(max(int(stim_index), 0), max(reference_all.shape[0] - 1, 0))
    order, drive_positions = _unwrapped_electrode_order(rows, stim_index=resolved_stim)
    x = np.arange(order.size)
    labels = [str(int(index) + 1) for index in order]
    reference = reference_all[resolved_stim, order]
    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=160)
    for row in rows[:-1]:
        potentials = np.asarray(row["electrode_potentials"], dtype=np.float64)
        profile = potentials[resolved_stim, order]
        ax.plot(
            x,
            np.abs(profile - reference),
            marker="o",
            linewidth=1.6,
            label=f"P{row['potential_order']} vs P{rows[-1]['potential_order']}",
        )
    _annotate_drive_electrodes(ax, x, np.zeros_like(reference), drive_positions)
    ax.set_xlabel("Unwrapped circular electrode order")
    ax.set_ylabel("Absolute electrode-potential error")
    ax.set_title(f"Drive-cut electrode-potential error, stimulation {resolved_stim}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        out_dir / f"electrode_potential_error_drive_cut_stim{resolved_stim}.png"
    )
    plt.close(fig)


def _plot_tradeoff(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    orders = np.asarray([row["potential_order"] for row in rows], dtype=int)
    dofs = np.asarray([row["potential_dofs"] for row in rows], dtype=float)
    errors = np.asarray(
        [row["relative_l2_delta_vs_reference"] for row in rows], dtype=float
    )
    solve_seconds = np.asarray([row["solve_seconds"] for row in rows], dtype=float)

    fig, ax_err = plt.subplots(figsize=(7.2, 4.5), dpi=160)
    ax_time = ax_err.twinx()
    ax_err.plot(
        orders,
        errors,
        marker="o",
        linewidth=1.8,
        color="#1f77b4",
        label="Relative L2 error",
    )
    ax_time.plot(
        orders,
        solve_seconds,
        marker="s",
        linewidth=1.8,
        color="#d62728",
        label="Solve seconds",
    )
    for order, dof in zip(orders, dofs):
        ax_err.annotate(
            f"{int(dof)} dofs",
            xy=(order, errors[order == orders][0]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax_err.set_xlabel("Polynomial order")
    ax_err.set_ylabel("Relative L2 error vs reference", color="#1f77b4")
    ax_time.set_ylabel("Solve seconds", color="#d62728")
    ax_err.tick_params(axis="y", labelcolor="#1f77b4")
    ax_time.tick_params(axis="y", labelcolor="#d62728")
    ax_err.set_xticks(orders)
    ax_err.grid(True, alpha=0.25)
    ax_err.set_title("Accuracy-cost tradeoff")
    lines = ax_err.get_lines() + ax_time.get_lines()
    ax_err.legend(lines, [line.get_label() for line in lines], frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "p_refinement_tradeoff.png")
    plt.close(fig)


def _write_plots(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    _configure_plot_fonts()
    _plot_voltage_overlay(out_dir, rows)
    if len(rows) > 1:
        _plot_voltage_error(out_dir, rows)
        _plot_electrode_potential_error(out_dir, rows, stim_index=0)
    _plot_electrode_potential_profile(out_dir, rows, stim_index=0)
    _plot_tradeoff(out_dir, rows)


def _write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [_strip_arrays(row) for row in payload["orders"]]
    csv_path = out_dir / "p_refinement_forward_summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    json_path = out_dir / "p_refinement_forward_summary.json"
    json_path.write_text(
        json.dumps({**payload, "orders": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# p-refinement forward experiment",
        "",
        f"- Created: `{payload['created_at']}`",
        f"- Mesh dim: `{payload['mesh_dim']}`",
        f"- Mesh refinement: `{payload['mesh_refinement']}`",
        f"- Orders: `{', '.join(str(row['potential_order']) for row in rows)}`",
        f"- Reference for error columns: highest order in this run, `P{rows[-1]['potential_order']}`; this is a convergence proxy, not an analytic truth.",
        "",
        "| P order | potential dofs | conductivity dofs | solve seconds | rel L2 delta vs ref |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {potential_order} | {potential_dofs} | {conductivity_dofs} | "
            "{solve_seconds:.6g} | {relative_l2_delta_vs_reference:.6g} |".format(**row)
        )
    (out_dir / "p_refinement_forward_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    _write_plots(out_dir, payload["orders"])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run same-mesh P1/P2/P3 forward p-refinement comparison.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--orders", default="1,2,3")
    parser.add_argument("--mesh-dim", type=int, default=2, choices=(2, 3))
    parser.add_argument("--mesh-shape", choices=("circle", "square"), default="circle")
    parser.add_argument("--mesh-refinement", type=int, default=6)
    parser.add_argument("--n-elec", type=int, default=8)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--height", type=float, default=1.0)
    parser.add_argument("--electrode-coverage", type=float, default=0.5)
    parser.add_argument("--electrode-height-ratio", type=float, default=0.2)
    parser.add_argument("--stim-pattern", default="{ad}")
    parser.add_argument("--meas-pattern", default="{ad}")
    parser.add_argument("--drive-mode", default="")
    parser.add_argument("--drive-value", type=float, default=1.0)
    parser.add_argument("--geometry-scale-to-m", type=float, default=1.0)
    parser.add_argument("--background-conductivity", type=float, default=1.0)
    parser.add_argument("--anomaly-conductivity", type=float, default=1.5)
    parser.add_argument("--anomaly-radius", type=float, default=0.18)
    parser.add_argument("--anomaly-x", type=float, default=0.55)
    parser.add_argument("--anomaly-y", type=float, default=0.5)
    parser.add_argument("--anomaly-z", type=float, default=0.0)
    parser.add_argument("--contact-impedance", type=float, default=1.0e-3)
    parser.add_argument("--linear-backend", choices=("scipy", "petsc"), default="scipy")
    parser.add_argument("--solver-preset", default="auto")
    parser.add_argument("--petsc-device", default="cpu")
    parser.add_argument("--forward-mat-solve", default="off")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    orders = _parse_orders(args.orders)
    out_dir = Path(args.out_dir)
    clear_process_forward_setup_cache()
    mesh = _make_mesh(args)
    centers = cell_midpoints(mesh.mesh)
    sigma_values = _conductivity(mesh.mesh, centers, args)
    drive_mode = str(args.drive_mode).strip()
    if not drive_mode:
        drive_mode = (
            "total_current" if int(args.mesh_dim) == 3 else "line_current_density"
        )
    pattern = PatternConfig(
        n_elec=int(args.n_elec),
        stim_pattern=str(args.stim_pattern),
        meas_pattern=str(args.meas_pattern),
        drive_mode=drive_mode,
        drive_value=float(args.drive_value),
        geometry_scale_to_m=float(args.geometry_scale_to_m),
    )

    rows = [
        _run_order(
            mesh=mesh,
            order=order,
            sigma_values=sigma_values,
            pattern=pattern,
            args=args,
        )
        for order in orders
    ]
    reference = np.asarray(rows[-1]["measurements"], dtype=np.float64)
    reference_norm = max(float(np.linalg.norm(reference)), 1.0e-15)
    for row in rows:
        delta = np.asarray(row["measurements"], dtype=np.float64) - reference
        row["l2_delta_vs_reference"] = float(np.linalg.norm(delta))
        row["relative_l2_delta_vs_reference"] = float(
            row["l2_delta_vs_reference"] / reference_norm
        )
        row["max_abs_delta_vs_reference"] = float(np.max(np.abs(delta)))

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "argv": list(sys.argv[1:] if argv is None else argv),
        "mesh_dim": int(args.mesh_dim),
        "mesh_refinement": int(args.mesh_refinement),
        "n_elec": int(args.n_elec),
        "orders": rows,
    }
    _write_outputs(out_dir, payload)
    print(json.dumps({"out_dir": str(out_dir), "orders": orders}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
