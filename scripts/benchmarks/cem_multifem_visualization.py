#!/usr/bin/env python3
"""Render common-mesh, electrode, target, and CEM forward-solution figures."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.cem_fair_common import canonical_mesh_fingerprint
from scripts.benchmarks.cem_multifem_common import solve_robin_from_blocks


VISUALIZATION_SCHEMA = "cem-multifem-forward-visualization-v1"
H0_NATIVE_ROOT = ROOT / "output/cem_multifem_accuracy/final/H0"
H0_EXISTING_ROOT = ROOT / "output/cem_continuum_accuracy/cases/C1_baseline/H0"
SIX_METHOD_REPORT = ROOT / "output/cem_multifem_accuracy/final/six_method_accuracy.json"
DEFAULT_OUTPUT_DIR = ROOT / "output/cem_multifem_accuracy/visualization"

METHOD_ORDER = (
    "EIDORS",
    "PyEIDORS-DOLFINx",
    "NGSolve",
    "MFEM",
    "FreeFEM",
    "GetFEM",
)

PALETTE = {
    "ink": "#20252B",
    "muted": "#66717E",
    "grid": "#D8DEE6",
    "mesh": "#B8C2CC",
    "blue": "#2563A6",
    "blue_dark": "#173F6C",
    "blue_light": "#DCE9F5",
    "gold": "#C48A24",
    "orange": "#D56A2C",
    "orange_light": "#F6DFD2",
    "pink": "#A64D79",
    "white": "#FFFFFF",
}


@dataclass(frozen=True)
class TargetConfig:
    center_x: float = 0.25
    center_y: float = 0.35
    radius: float = 0.23
    background_conductivity: float = 0.25
    target_conductivity: float = 1.0
    drive_index_zero_based: int = 0


@dataclass(frozen=True)
class ForwardFields:
    conductivity: np.ndarray
    target_mask: np.ndarray
    baseline_body: np.ndarray
    target_body: np.ndarray
    body_perturbation: np.ndarray
    baseline_voltage: np.ndarray
    target_voltage: np.ndarray
    voltage_perturbation: np.ndarray
    residuals: dict[str, float]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required visualization source is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_plotting() -> None:
    """Register and require Times New Roman for English text and numerals."""

    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    candidates = (
        Path("/mnt/c/Windows/Fonts/times.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbd.ttf"),
        Path("/mnt/c/Windows/Fonts/timesi.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbi.ttf"),
    )
    for font_path in candidates:
        if font_path.is_file():
            font_manager.fontManager.addfont(font_path)
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 240,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
        }
    )


def _existing_voltage(report: dict[str, Any], key: str) -> np.ndarray:
    voltage = np.asarray(report["raw_electrode_voltages"][key], dtype=np.float64)
    if voltage.ndim != 2 or voltage.shape[0] != voltage.shape[1]:
        raise ValueError(
            "registered electrode voltage must have shape (electrode, RHS)"
        )
    return voltage


def load_h0_sources() -> dict[str, Any]:
    """Load actual imported topology and all six registered voltage matrices."""

    native_paths = {
        solver: H0_NATIVE_ROOT / solver / f"{solver}_native_report.json"
        for solver in ("MFEM", "FreeFEM", "GetFEM")
    }
    native = {solver: _load_json(path) for solver, path in native_paths.items()}
    mesh_report = native["MFEM"]
    discretization = mesh_report["discretization"]
    nodes = np.asarray(discretization["imported_nodes"], dtype=np.float64)[:, :2]
    cells = np.asarray(discretization["imported_cells_zero_based"], dtype=np.int64)[
        :, :3
    ]
    tagged_edges = np.asarray(
        discretization["imported_tagged_boundary_edges_zero_based"], dtype=np.int64
    )[:, :3]
    mesh_fingerprint = canonical_mesh_fingerprint(nodes, cells, tagged_edges)
    if mesh_fingerprint != discretization["mesh_fingerprint"]:
        raise ValueError(
            "visualization mesh fingerprint does not match imported topology"
        )

    existing_paths = {
        "EIDORS": H0_EXISTING_ROOT / "eidors_report.json",
        "PyEIDORS-DOLFINx": H0_EXISTING_ROOT / "pyeidors_report.json",
        "NGSolve": H0_EXISTING_ROOT / "ngsolve_report.json",
    }
    existing = {solver: _load_json(path) for solver, path in existing_paths.items()}
    voltages = {
        "EIDORS": _existing_voltage(existing["EIDORS"], "classic"),
        "PyEIDORS-DOLFINx": _existing_voltage(
            existing["PyEIDORS-DOLFINx"], "robin_transconductance"
        ),
        "NGSolve": _existing_voltage(existing["NGSolve"], "robin_transconductance"),
    }
    for solver, report in native.items():
        voltages[solver] = np.asarray(
            report["solution"]["electrode_voltage"], dtype=np.float64
        )

    source_fingerprints: dict[str, str] = {}
    for solver, report in existing.items():
        source_fingerprints[solver] = report["discretization"]["mesh_fingerprint"]
    for solver, report in native.items():
        imported = report["discretization"]
        source_fingerprints[solver] = canonical_mesh_fingerprint(
            np.asarray(imported["imported_nodes"], dtype=np.float64)[:, :2],
            np.asarray(imported["imported_cells_zero_based"], dtype=np.int64)[:, :3],
            np.asarray(
                imported["imported_tagged_boundary_edges_zero_based"],
                dtype=np.int64,
            )[:, :3],
        )
    if set(source_fingerprints.values()) != {mesh_fingerprint}:
        raise ValueError("visualization sources do not share one actual imported mesh")
    if set(voltages) != set(METHOD_ORDER):
        raise ValueError("visualization requires the exact registered six methods")
    if len({value.shape for value in voltages.values()}) != 1:
        raise ValueError("six-method voltage shapes differ")

    all_pass = _load_json(SIX_METHOD_REPORT)
    if not bool(all_pass["all_pass"]):
        raise ValueError("six-method accuracy gate must pass before visualization")
    return {
        "nodes": nodes,
        "cells": cells,
        "tagged_edges": tagged_edges,
        "mesh_fingerprint": mesh_fingerprint,
        "mesh_fingerprints_by_method": source_fingerprints,
        "contact_impedance": np.asarray(
            mesh_report["physical_config"]["contact_impedance"], dtype=np.float64
        ),
        "currents": np.asarray(
            mesh_report["physical_config"]["currents"], dtype=np.float64
        ),
        "mfem_body_potential": np.asarray(
            mesh_report["solution"]["body_potential"], dtype=np.float64
        ),
        "voltages": voltages,
        "source_paths": {
            **{solver: str(path.resolve()) for solver, path in existing_paths.items()},
            **{solver: str(path.resolve()) for solver, path in native_paths.items()},
            "six_method_accuracy": str(SIX_METHOD_REPORT.resolve()),
        },
    }


def assemble_p1_cem_blocks(
    nodes: np.ndarray,
    cells: np.ndarray,
    tagged_edges: np.ndarray,
    conductivity: np.ndarray,
    contact_impedance: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assemble neutral straight-triangle P1 CEM blocks for visualization."""

    points = np.asarray(nodes, dtype=np.float64)
    triangles = np.asarray(cells, dtype=np.int64)
    edges = np.asarray(tagged_edges, dtype=np.int64)
    sigma = np.asarray(conductivity, dtype=np.float64)
    impedance = np.asarray(contact_impedance, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("nodes must have shape (N, 2)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("cells must have shape (M, 3)")
    if sigma.shape != (triangles.shape[0],) or np.any(sigma <= 0.0):
        raise ValueError("one positive conductivity value is required per cell")
    if impedance.ndim != 1 or np.any(impedance <= 0.0):
        raise ValueError("positive contact impedance vector required")

    node_count = points.shape[0]
    electrode_count = impedance.size
    stiffness = np.zeros((node_count, node_count), dtype=np.float64)
    boundary_mass = np.zeros_like(stiffness)
    coupling = np.zeros((node_count, electrode_count), dtype=np.float64)
    electrode_matrix = np.zeros((electrode_count, electrode_count), dtype=np.float64)

    for cell_index, triangle in enumerate(triangles):
        xy = points[triangle]
        twice_signed_area = float(
            (xy[1, 0] - xy[0, 0]) * (xy[2, 1] - xy[0, 1])
            - (xy[2, 0] - xy[0, 0]) * (xy[1, 1] - xy[0, 1])
        )
        area = 0.5 * abs(twice_signed_area)
        if area <= 0.0:
            raise ValueError("degenerate P1 triangle")
        grad_x = np.asarray(
            [xy[1, 1] - xy[2, 1], xy[2, 1] - xy[0, 1], xy[0, 1] - xy[1, 1]]
        ) / (2.0 * area)
        grad_y = np.asarray(
            [xy[2, 0] - xy[1, 0], xy[0, 0] - xy[2, 0], xy[1, 0] - xy[0, 0]]
        ) / (2.0 * area)
        local = (
            sigma[cell_index]
            * area
            * (np.outer(grad_x, grad_x) + np.outer(grad_y, grad_y))
        )
        stiffness[np.ix_(triangle, triangle)] += local

    for first, second, tag in edges:
        if tag <= 0 or tag > electrode_count:
            continue
        electrode = int(tag - 1)
        length = float(np.linalg.norm(points[int(first)] - points[int(second)]))
        inverse_contact = 1.0 / impedance[electrode]
        vertices = [int(first), int(second)]
        boundary_mass[np.ix_(vertices, vertices)] += (
            length * inverse_contact / 6.0 * np.asarray([[2.0, 1.0], [1.0, 2.0]])
        )
        coupling[vertices, electrode] += 0.5 * length * inverse_contact
        electrode_matrix[electrode, electrode] += length * inverse_contact

    return {
        "K": stiffness,
        "B": boundary_mass,
        "C_plus": coupling,
        "D": electrode_matrix,
        "A_R": stiffness + boundary_mass,
    }


def build_forward_fields(
    sources: dict[str, Any], config: TargetConfig
) -> ForwardFields:
    """Solve matched homogeneous and heterogeneous common-P1 CEM problems."""

    nodes = np.asarray(sources["nodes"], dtype=np.float64)
    cells = np.asarray(sources["cells"], dtype=np.int64)
    centroids = np.mean(nodes[cells], axis=1)
    target_mask = (centroids[:, 0] - config.center_x) ** 2 + (
        centroids[:, 1] - config.center_y
    ) ** 2 <= config.radius**2
    if not np.any(target_mask) or np.all(target_mask):
        raise ValueError("target must select a strict non-empty subset of cells")
    baseline_sigma = np.full(
        cells.shape[0], config.background_conductivity, dtype=np.float64
    )
    target_sigma = baseline_sigma.copy()
    target_sigma[target_mask] = config.target_conductivity
    baseline_blocks = assemble_p1_cem_blocks(
        nodes,
        cells,
        sources["tagged_edges"],
        baseline_sigma,
        sources["contact_impedance"],
    )
    target_blocks = assemble_p1_cem_blocks(
        nodes,
        cells,
        sources["tagged_edges"],
        target_sigma,
        sources["contact_impedance"],
    )
    currents = np.asarray(sources["currents"], dtype=np.float64)
    baseline = solve_robin_from_blocks(
        K=baseline_blocks["K"],
        B=baseline_blocks["B"],
        C_plus=baseline_blocks["C_plus"],
        D=baseline_blocks["D"],
        currents=currents,
    )
    target = solve_robin_from_blocks(
        K=target_blocks["K"],
        B=target_blocks["B"],
        C_plus=target_blocks["C_plus"],
        D=target_blocks["D"],
        currents=currents,
    )
    drive = config.drive_index_zero_based
    target_body = target.body_potential[:, drive]
    baseline_body = baseline.body_potential[:, drive]
    target_voltage = target.electrode_voltage[:, drive]
    baseline_voltage = baseline.electrode_voltage[:, drive]
    native_baseline_body = np.asarray(sources["mfem_body_potential"])[..., drive]
    native_baseline_voltage = np.asarray(sources["voltages"]["MFEM"])[..., drive]
    residuals = {
        "baseline_vs_mfem_body_relative_l2": float(
            np.linalg.norm(baseline_body - native_baseline_body)
            / np.linalg.norm(native_baseline_body)
        ),
        "baseline_vs_mfem_voltage_relative_l2": float(
            np.linalg.norm(baseline_voltage - native_baseline_voltage)
            / np.linalg.norm(native_baseline_voltage)
        ),
        "target_robin_scaled_inf": float(
            np.linalg.norm(
                target_blocks["A_R"] @ target.body_potential
                - target_blocks["C_plus"] @ target.electrode_voltage,
                ord=np.inf,
            )
            / max(
                np.linalg.norm(
                    target_blocks["C_plus"] @ target.electrode_voltage,
                    ord=np.inf,
                ),
                1.0,
            )
        ),
        "target_current_scaled_inf": float(
            np.linalg.norm(
                target_blocks["D"] @ target.electrode_voltage
                - target_blocks["C_plus"].T @ target.body_potential
                - currents,
                ord=np.inf,
            )
            / max(np.linalg.norm(currents, ord=np.inf), 1.0)
        ),
        "target_voltage_gauge_inf": float(
            np.max(np.abs(np.sum(target.electrode_voltage, axis=0)))
        ),
    }
    if max(residuals.values()) > 5.0e-11:
        raise ValueError(f"visualization forward solve failed: {residuals}")
    return ForwardFields(
        conductivity=target_sigma,
        target_mask=target_mask,
        baseline_body=baseline_body,
        target_body=target_body,
        body_perturbation=target_body - baseline_body,
        baseline_voltage=baseline_voltage,
        target_voltage=target_voltage,
        voltage_perturbation=target_voltage - baseline_voltage,
        residuals=residuals,
    )


def _draw_electrodes(
    axis: Any,
    nodes: np.ndarray,
    tagged_edges: np.ndarray,
    *,
    label: bool,
    drive: int,
) -> None:
    electrode_count = int(np.max(tagged_edges[:, 2]))
    active = {
        drive + 1: PALETTE["orange"],
        (drive + 1) % electrode_count + 1: PALETTE["pink"],
    }
    for tag in range(1, electrode_count + 1):
        selected = tagged_edges[tagged_edges[:, 2] == tag]
        if selected.size == 0:
            continue
        color = active.get(tag, PALETTE["blue_dark"])
        linewidth = 4.0 if tag in active else 2.8
        midpoints = []
        for first, second, _ in selected:
            segment = nodes[[int(first), int(second)]]
            axis.plot(
                segment[:, 0],
                segment[:, 1],
                color=color,
                linewidth=linewidth,
                solid_capstyle="round",
                zorder=8,
            )
            midpoints.append(np.mean(segment, axis=0))
        if label:
            midpoint = np.mean(midpoints, axis=0)
            norm = max(float(np.linalg.norm(midpoint)), np.finfo(float).eps)
            position = midpoint * (1.10 / norm)
            axis.text(
                position[0],
                position[1],
                f"E{tag}",
                ha="center",
                va="center",
                color=color,
                fontsize=7.2,
                fontweight="bold" if tag in active else "normal",
                zorder=10,
            )


def _draw_mesh(axis: Any, sources: dict[str, Any], config: TargetConfig) -> None:
    import matplotlib.tri as mtri

    nodes = sources["nodes"]
    cells = sources["cells"]
    triangulation = mtri.Triangulation(nodes[:, 0], nodes[:, 1], cells)
    axis.triplot(triangulation, color=PALETTE["mesh"], linewidth=0.45, zorder=1)
    _draw_electrodes(
        axis,
        nodes,
        sources["tagged_edges"],
        label=True,
        drive=config.drive_index_zero_based,
    )
    axis.scatter([], [], color=PALETTE["orange"], linewidth=4, label="Current +1")
    axis.scatter([], [], color=PALETTE["pink"], linewidth=4, label="Current -1")
    axis.set_title("Common H0 P1 mesh and electrodes")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.13))
    _equal_disk_axis(axis)


def _equal_disk_axis(axis: Any) -> None:
    axis.set_aspect("equal")
    axis.set_xlim(-1.16, 1.16)
    axis.set_ylim(-1.16, 1.16)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(False)


def _draw_conductivity(
    axis: Any, sources: dict[str, Any], fields: ForwardFields, config: TargetConfig
) -> Any:
    import matplotlib.tri as mtri
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Circle

    nodes = sources["nodes"]
    triangulation = mtri.Triangulation(nodes[:, 0], nodes[:, 1], sources["cells"])
    image = axis.tripcolor(
        triangulation,
        facecolors=fields.conductivity,
        shading="flat",
        cmap=ListedColormap([PALETTE["blue_light"], PALETTE["gold"]]),
        edgecolors=PALETTE["white"],
        linewidth=0.18,
        vmin=config.background_conductivity,
        vmax=config.target_conductivity,
    )
    axis.add_patch(
        Circle(
            (config.center_x, config.center_y),
            config.radius,
            fill=False,
            linestyle="--",
            linewidth=1.2,
            color=PALETTE["ink"],
            zorder=7,
        )
    )
    _draw_electrodes(
        axis,
        nodes,
        sources["tagged_edges"],
        label=False,
        drive=config.drive_index_zero_based,
    )
    axis.set_title("Conductivity and test object")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    _equal_disk_axis(axis)
    return image


def _blue_white_orange_colormap() -> Any:
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "blue_white_orange",
        [PALETTE["blue_dark"], PALETTE["white"], PALETTE["orange"]],
    )


def _draw_nodal_field(
    axis: Any,
    sources: dict[str, Any],
    values: np.ndarray,
    config: TargetConfig,
    *,
    title: str,
) -> Any:
    import matplotlib.tri as mtri

    nodes = sources["nodes"]
    triangulation = mtri.Triangulation(nodes[:, 0], nodes[:, 1], sources["cells"])
    limit = max(float(np.max(np.abs(values))), np.finfo(float).eps)
    image = axis.tripcolor(
        triangulation,
        values,
        shading="gouraud",
        cmap=_blue_white_orange_colormap(),
        vmin=-limit,
        vmax=limit,
    )
    _draw_electrodes(
        axis,
        nodes,
        sources["tagged_edges"],
        label=False,
        drive=config.drive_index_zero_based,
    )
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    _equal_disk_axis(axis)
    return image


def _voltage_styles() -> dict[str, dict[str, Any]]:
    return {
        "EIDORS": {"color": PALETTE["ink"], "linestyle": "-", "marker": "o"},
        "PyEIDORS-DOLFINx": {
            "color": PALETTE["blue"],
            "linestyle": "--",
            "marker": "s",
        },
        "NGSolve": {"color": PALETTE["gold"], "linestyle": "-.", "marker": "^"},
        "MFEM": {
            "color": PALETTE["blue_dark"],
            "linestyle": ":",
            "marker": "D",
        },
        "FreeFEM": {
            "color": PALETTE["orange"],
            "linestyle": (0, (5, 2)),
            "marker": "v",
        },
        "GetFEM": {
            "color": PALETTE["muted"],
            "linestyle": (0, (1, 1)),
            "marker": "x",
        },
    }


def _draw_voltage_overlay(
    axis: Any, sources: dict[str, Any], config: TargetConfig
) -> None:
    x = np.arange(1, len(sources["voltages"]["EIDORS"]) + 1)
    styles = _voltage_styles()
    for solver in METHOD_ORDER:
        voltage = sources["voltages"][solver][:, config.drive_index_zero_based]
        axis.plot(
            x,
            voltage,
            linewidth=1.25 if solver == "EIDORS" else 1.0,
            markersize=3.8,
            markevery=1,
            label=solver,
            **styles[solver],
        )
    axis.axhline(0.0, color=PALETTE["grid"], linewidth=0.8)
    axis.set_title("Six-method baseline voltages, drive E1 to E2")
    axis.set_xlabel("Electrode index")
    axis.set_ylabel("Electrode voltage")
    axis.set_xticks(x)
    axis.grid(axis="y", color=PALETTE["grid"], linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)


def _draw_target_voltage_effect(axis: Any, fields: ForwardFields) -> None:
    x = np.arange(1, fields.voltage_perturbation.size + 1)
    values = fields.voltage_perturbation
    colors = np.where(values >= 0.0, PALETTE["orange"], PALETTE["blue"])
    axis.bar(
        x,
        values,
        width=0.68,
        color=colors,
        edgecolor=PALETTE["ink"],
        linewidth=0.45,
    )
    axis.axhline(0.0, color=PALETTE["ink"], linewidth=0.8)
    axis.set_title("Target-induced electrode-voltage change")
    axis.set_xlabel("Electrode index")
    axis.set_ylabel(r"$U_{\mathrm{target}}-U_{\mathrm{baseline}}$")
    axis.set_xticks(x)
    axis.grid(axis="y", color=PALETTE["grid"], linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)


def _draw_voltage_residual(
    axis: Any, sources: dict[str, Any], config: TargetConfig
) -> None:
    reference = sources["voltages"]["EIDORS"][:, config.drive_index_zero_based]
    solvers = list(METHOD_ORDER[1:])
    values = [
        float(
            np.linalg.norm(
                sources["voltages"][solver][:, config.drive_index_zero_based]
                - reference
            )
            / np.linalg.norm(reference)
        )
        for solver in solvers
    ]
    y = np.arange(len(solvers))
    for index, (solver, value) in enumerate(zip(solvers, values, strict=True)):
        color = _voltage_styles()[solver]["color"]
        axis.hlines(index, 1.0e-16, value, color=color, linewidth=1.3)
        axis.plot(value, index, marker="o", color=color, markersize=5)
        axis.text(value * 1.12, index, f"{value:.2e}", va="center", fontsize=8)
    axis.set_xscale("log")
    axis.set_xlim(1.0e-16, 2.0e-14)
    axis.set_yticks(y, solvers)
    axis.invert_yaxis()
    axis.set_title("Relative difference from EIDORS")
    axis.set_xlabel(r"$\|U-U_{\mathrm{EIDORS}}\|_2 / \|U_{\mathrm{EIDORS}}\|_2$")
    axis.grid(axis="x", color=PALETTE["grid"], linewidth=0.6, which="both")
    axis.spines[["top", "right"]].set_visible(False)


def _save_figure(figure: Any, output_dir: Path, stem: str) -> list[Path]:
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.svg"]
    figure.savefig(paths[0], bbox_inches="tight", facecolor="white")
    figure.savefig(paths[1], bbox_inches="tight", facecolor="white")
    return paths


def render_visualizations(
    sources: dict[str, Any],
    fields: ForwardFields,
    config: TargetConfig,
    output_dir: Path,
) -> list[Path]:
    """Render standalone and combined publication-ready static figures."""

    import matplotlib.pyplot as plt

    configure_plotting()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    mesh_figure, mesh_axis = plt.subplots(figsize=(7.0, 7.0), constrained_layout=True)
    _draw_mesh(mesh_axis, sources, config)
    outputs.extend(_save_figure(mesh_figure, output_dir, "cem_h0_mesh_electrodes"))
    plt.close(mesh_figure)

    forward_figure, forward_axes = plt.subplots(
        1, 4, figsize=(18.0, 4.6), constrained_layout=True
    )
    conductivity_image = _draw_conductivity(forward_axes[0], sources, fields, config)
    target_image = _draw_nodal_field(
        forward_axes[1],
        sources,
        fields.target_body,
        config,
        title="Forward body potential with target",
    )
    perturbation_image = _draw_nodal_field(
        forward_axes[2],
        sources,
        fields.body_perturbation,
        config,
        title="Target-induced potential perturbation",
    )
    _draw_target_voltage_effect(forward_axes[3], fields)
    forward_figure.colorbar(
        conductivity_image,
        ax=forward_axes[0],
        shrink=0.78,
        label="Conductivity",
    )
    forward_figure.colorbar(
        target_image,
        ax=forward_axes[1],
        shrink=0.78,
        label="Potential",
    )
    forward_figure.colorbar(
        perturbation_image,
        ax=forward_axes[2],
        shrink=0.78,
        label="Potential difference",
    )
    outputs.extend(
        _save_figure(forward_figure, output_dir, "cem_heterogeneous_forward")
    )
    plt.close(forward_figure)

    voltage_figure, voltage_axes = plt.subplots(
        2,
        1,
        figsize=(10.4, 8.0),
        gridspec_kw={"height_ratios": [2.0, 1.15]},
    )
    voltage_figure.subplots_adjust(
        left=0.14, right=0.96, top=0.94, bottom=0.09, hspace=0.82
    )
    _draw_voltage_overlay(voltage_axes[0], sources, config)
    _draw_voltage_residual(voltage_axes[1], sources, config)
    outputs.extend(
        _save_figure(voltage_figure, output_dir, "cem_six_method_electrode_voltage")
    )
    plt.close(voltage_figure)

    summary_figure, summary_axes = plt.subplots(2, 3, figsize=(16.4, 9.6))
    summary_figure.subplots_adjust(
        left=0.055, right=0.95, top=0.94, bottom=0.08, wspace=0.42, hspace=0.52
    )
    _draw_mesh(summary_axes[0, 0], sources, config)
    conductivity_image = _draw_conductivity(summary_axes[0, 1], sources, fields, config)
    target_image = _draw_nodal_field(
        summary_axes[0, 2],
        sources,
        fields.target_body,
        config,
        title="Forward body potential with target",
    )
    perturbation_image = _draw_nodal_field(
        summary_axes[1, 0],
        sources,
        fields.body_perturbation,
        config,
        title="Target-induced potential perturbation",
    )
    _draw_voltage_overlay(summary_axes[1, 1], sources, config)
    _draw_target_voltage_effect(summary_axes[1, 2], fields)
    summary_figure.colorbar(
        conductivity_image,
        ax=summary_axes[0, 1],
        shrink=0.72,
        label="Conductivity",
    )
    summary_figure.colorbar(
        target_image,
        ax=summary_axes[0, 2],
        shrink=0.72,
        label="Potential",
    )
    summary_figure.colorbar(
        perturbation_image,
        ax=summary_axes[1, 0],
        shrink=0.72,
        label="Potential difference",
    )
    outputs.extend(
        _save_figure(summary_figure, output_dir, "cem_multifem_visual_summary")
    )
    plt.close(summary_figure)
    return outputs


def build_visualization_artifacts(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    config: TargetConfig | None = None,
) -> dict[str, Any]:
    """Build figures and a machine-readable chart/source manifest."""

    selected_config = config or TargetConfig()
    sources = load_h0_sources()
    fields = build_forward_fields(sources, selected_config)
    output_dir = output_dir.resolve()
    figures = render_visualizations(sources, fields, selected_config, output_dir)
    source_paths = {key: Path(value) for key, value in sources["source_paths"].items()}
    manifest = {
        "schema": VISUALIZATION_SCHEMA,
        "scope": "common-H0 P1 CEM geometry and forward visualization",
        "mesh": {
            "fingerprint": sources["mesh_fingerprint"],
            "fingerprints_by_method": sources["mesh_fingerprints_by_method"],
            "nodes": int(sources["nodes"].shape[0]),
            "cells": int(sources["cells"].shape[0]),
            "tagged_boundary_edges": int(sources["tagged_edges"].shape[0]),
            "potential_order": 1,
            "geometry_order": 1,
        },
        "physical_config": {
            **asdict(selected_config),
            "current_pattern": "electrode 1 injects +1; electrode 2 withdraws -1",
            "contact_impedance": sources["contact_impedance"].tolist(),
            "target_cells": int(np.count_nonzero(fields.target_mask)),
            "background_cells": int(np.count_nonzero(~fields.target_mask)),
        },
        "forward_solution": {
            "assembly": "neutral straight-triangle P1 Robin-transconductance CEM",
            "same_mesh_as_six_method_gate": True,
            "same_target_as_displayed_conductivity": True,
            "residuals": fields.residuals,
            "body_potential_range": [
                float(np.min(fields.target_body)),
                float(np.max(fields.target_body)),
            ],
            "body_perturbation_range": [
                float(np.min(fields.body_perturbation)),
                float(np.max(fields.body_perturbation)),
            ],
            "electrode_voltage_perturbation_relative_l2": float(
                np.linalg.norm(fields.voltage_perturbation)
                / np.linalg.norm(fields.baseline_voltage)
            ),
        },
        "chart_map": [
            {
                "figure": "cem_h0_mesh_electrodes",
                "question": "What exact P1 mesh, electrodes, and active drive are used?",
                "family": "geometry",
                "takeaway": "All six methods share one imported mesh and electrode tagging.",
                "panels": ["mesh", "electrode labels", "injection and sink"],
            },
            {
                "figure": "cem_heterogeneous_forward",
                "question": "How does the displayed target alter the CEM body potential?",
                "family": "field map",
                "takeaway": "The target and both potential fields come from one matched solve.",
                "panels": [
                    "conductivity",
                    "body potential",
                    "body perturbation",
                    "electrode-voltage perturbation",
                ],
            },
            {
                "figure": "cem_six_method_electrode_voltage",
                "question": "Do the six methods produce the same observable voltages?",
                "family": "comparison and benchmark",
                "takeaway": "Six traces overlap; differences from EIDORS are float64 scale.",
                "panels": ["baseline voltage overlay", "relative difference dots"],
            },
            {
                "figure": "cem_multifem_visual_summary",
                "question": "Can the complete geometry-to-observation forward chain be read at once?",
                "family": "multi-panel research summary",
                "takeaway": "Mesh, matched target/field perturbations, boundary-voltage effect, and baseline solver parity remain explicit.",
                "panels": [
                    "mesh and electrodes",
                    "conductivity",
                    "body potential",
                    "body perturbation",
                    "baseline voltage overlay",
                    "electrode-voltage perturbation",
                ],
            },
        ],
        "visual_contract": {
            "renderer": "Matplotlib static",
            "font": "Times New Roman",
            "palette_policy": "two-root blue/orange plus neutral ink; pink marks current sink",
            "non_color_distinction": "line style and marker vary by method",
            "formats": ["PNG", "SVG"],
        },
        "sources": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in source_paths.items()
        },
        "figures": [{"path": str(path), "sha256": _sha256(path)} for path in figures],
        "all_pass": True,
    }
    manifest_path = output_dir / "visualization_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-center-x", type=float, default=0.25)
    parser.add_argument("--target-center-y", type=float, default=0.35)
    parser.add_argument("--target-radius", type=float, default=0.23)
    parser.add_argument("--background-conductivity", type=float, default=0.25)
    parser.add_argument("--target-conductivity", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_visualization_artifacts(
        args.output_dir,
        config=TargetConfig(
            center_x=args.target_center_x,
            center_y=args.target_center_y,
            radius=args.target_radius,
            background_conductivity=args.background_conductivity,
            target_conductivity=args.target_conductivity,
        ),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
