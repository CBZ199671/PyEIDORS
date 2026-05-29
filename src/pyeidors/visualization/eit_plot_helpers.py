"""Low-level plotting helpers shared by EIT visualisation renderers."""

from __future__ import annotations

from typing import Any

import numpy as np
import ufl
from dolfinx import fem
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from mpi4py import MPI

from ..femx import (
    mesh_cell_vertices,
    mesh_coordinates,
    mesh_num_cells,
    mesh_num_vertices,
)


def raw_mesh(mesh):
    return mesh.mesh if hasattr(mesh, "mesh") else mesh


def coordinates(mesh) -> np.ndarray:
    if hasattr(mesh, "coordinates"):
        return mesh.coordinates()
    return mesh_coordinates(raw_mesh(mesh))


def cells(mesh) -> np.ndarray:
    if hasattr(mesh, "cells"):
        return mesh.cells()
    return mesh_cell_vertices(raw_mesh(mesh))


def num_cells(mesh) -> int:
    if hasattr(mesh, "num_cells"):
        return int(mesh.num_cells())
    return mesh_num_cells(raw_mesh(mesh))


def num_vertices(mesh) -> int:
    if hasattr(mesh, "num_vertices"):
        return int(mesh.num_vertices())
    return mesh_num_vertices(raw_mesh(mesh))


def interpolate_cell_to_node(mesh, cell_values):
    node_values = np.zeros(num_vertices(mesh))
    node_counts = np.zeros(num_vertices(mesh))
    for cell_idx, cell in enumerate(cells(mesh)):
        for vertex_idx in cell:
            node_values[vertex_idx] += cell_values[cell_idx]
            node_counts[vertex_idx] += 1
    node_counts[node_counts == 0] = 1
    node_values /= node_counts
    return node_values


def plot_electrodes(ax, electrode_vertices):
    for i, electrode in enumerate(electrode_vertices):
        electrode_array = np.asarray(electrode)
        if electrode_array.size == 0:
            continue
        ax.plot(
            electrode_array[:, 0],
            electrode_array[:, 1],
            "ro-",
            markersize=6,
            linewidth=2,
            label=f"Electrode {i + 1}" if i < 5 else "",
        )
    if len(electrode_vertices) <= 5:
        ax.legend()


def is_eidors_diff(colormap: str | Any) -> bool:
    return isinstance(colormap, str) and colormap.lower() in {
        "eidors_diff",
        "eidors-diff",
    }


def resolve_colormap(colormap: str | Any) -> Any:
    if isinstance(colormap, str) and colormap.lower() in {"eidors_diff", "eidors-diff"}:
        return LinearSegmentedColormap.from_list(
            "eidors_diff",
            ["#1f3a93", "#ffffff", "#b30000"],
        )
    return colormap


def resolve_eidors_diff_limits(
    values: np.ndarray,
    vmin: float | None,
    vmax: float | None,
):
    if vmin is None and vmax is None:
        max_abs = float(np.nanmax(np.abs(values)))
        if max_abs == 0.0:
            max_abs = 1e-12
        return -max_abs, max_abs
    if vmin is None and vmax is not None:
        return -abs(vmax), vmax
    if vmax is None and vmin is not None:
        return vmin, abs(vmin)
    return float(vmin), float(vmax)


def eidors_tick_vals(
    max_scale: float,
    ref_lev: float,
    tick_div_in: int | None = None,
) -> np.ndarray:
    if max_scale <= 0:
        return np.array([ref_lev], dtype=float)
    F = 2.0
    ord_of_mag = 10 ** np.floor(np.log10(max_scale * F)) / F
    scale1 = np.floor(max_scale / ord_of_mag + 2 * np.finfo(float).eps)
    if scale1 / F >= 8:
        fms, tick_div = F * 8, 2
    elif scale1 / F >= 6:
        fms, tick_div = F * 6, 2
    elif scale1 / F >= 4:
        fms, tick_div = F * 4, 2
    elif scale1 / F >= 3:
        fms, tick_div = F * 3, 3
    elif scale1 / F >= 2:
        fms, tick_div = F * 2, 2
    elif scale1 / F >= 1.5:
        fms, tick_div = F * 1.5, 3
    elif scale1 / F >= 1:
        fms, tick_div = F * 1, 2
    else:
        fms, tick_div = F * 0.5, 2
    if tick_div_in is not None:
        tick_div = tick_div_in

    scale_r = ord_of_mag * fms
    ord_of_mag = 10 ** np.floor(np.log10(max_scale))
    ref_r = ord_of_mag * np.round(ref_lev / ord_of_mag)
    return np.linspace(-2, 2, tick_div * 4 + 1) * scale_r + ref_r


def apply_eidors_ticks(
    cbar: Any,
    vmin: float | None,
    vmax: float | None,
    ref_lev: float = 0.0,
    tick_div: int | None = None,
) -> None:
    if vmin is None or vmax is None:
        return
    max_scale = max(abs(vmax - ref_lev), abs(ref_lev - vmin))
    tick_vals = eidors_tick_vals(max_scale, ref_lev, tick_div)
    if tick_vals.size == 0:
        return
    eps = max(max_scale * 1e-12, 1e-12)
    tick_vals = tick_vals[(tick_vals >= vmin - eps) & (tick_vals <= vmax + eps)]
    if tick_vals.size > 0:
        cbar.set_ticks(tick_vals)


def format_colorbar(cbar: Any, format_mode: str) -> None:
    mode = (format_mode or "plain").lower()

    def _fmt_sci_adaptive(x: float, _: float) -> str:
        if x == 0:
            return "0"
        ax = abs(x)
        exp = int(np.floor(np.log10(ax)))
        mantissa = x / (10**exp)
        mantissa_rounded = round(mantissa, 1)
        if abs(mantissa_rounded) >= 10:
            mantissa_rounded /= 10
            exp += 1
        if abs(mantissa_rounded - round(mantissa_rounded)) < 1e-8:
            mantissa_str = f"{mantissa_rounded:.0f}"
        else:
            mantissa_str = f"{mantissa_rounded:.1f}"
        return f"{mantissa_str}e{exp:+03d}"

    if mode == "scientific":
        formatter = FuncFormatter(_fmt_sci_adaptive)
    elif mode == "matlab_short":

        def _fmt_matlab_short(x: float, _: float) -> str:
            if x == 0:
                return "0.0000"
            ax = abs(x)
            if 1e-3 <= ax < 1e4:
                return f"{x:.4f}"
            return _fmt_sci_adaptive(x, _)

        formatter = FuncFormatter(_fmt_matlab_short)
    else:
        formatter = ScalarFormatter(useMathText=False)
        formatter.set_useOffset(False)
        formatter.set_scientific(False)

    cbar.formatter = formatter
    cbar.ax.yaxis.get_offset_text().set_visible(False)
    cbar.update_ticks()


def extract_electrode_tags(mesh) -> list[int]:
    assoc = getattr(mesh, "association_table", {}) or {}
    tags: list[int] = []
    for key, value in assoc.items():
        try:
            tag_val = int(value)
        except Exception:
            continue
        if isinstance(key, str) and key.lower().startswith("electrode"):
            tags.append(tag_val)
        elif isinstance(key, (int, np.integer)) and key >= 2:
            tags.append(tag_val)
    return sorted(set(tags))


def overlay_electrode_labels(ax, mesh, label_outset: float = 0.08):
    coords = coordinates(mesh)
    center = coords.mean(axis=0)
    radius = np.max(np.linalg.norm(coords - center, axis=1)) + 1e-12

    tags = extract_electrode_tags(mesh)
    if not tags:
        raise RuntimeError("No electrode tags parsed from association_table")

    facet_tags = getattr(mesh, "facet_tags", None)
    if facet_tags is None:
        raise RuntimeError("Mesh has no facet tags")

    tag_points: dict[int, list[np.ndarray]] = {tag: [] for tag in tags}
    mesh_obj = raw_mesh(mesh)
    tdim = mesh_obj.topology.dim
    fdim = tdim - 1
    mesh_obj.topology.create_connectivity(fdim, 0)
    f2v = mesh_obj.topology.connectivity(fdim, 0)
    if f2v is None:
        raise RuntimeError("Cannot read facet->vertex connectivity")

    for facet_idx, tag in zip(facet_tags.indices, facet_tags.values):
        tag_int = int(tag)
        if tag_int in tag_points:
            vertices = f2v.links(int(facet_idx))
            tag_points[tag_int].append(coords[vertices][:, :2])

    ds = ufl.Measure("ds", domain=mesh_obj, subdomain_data=facet_tags)
    one = fem.Constant(mesh_obj, 1.0)
    _lengths = {
        tag: float(
            mesh_obj.comm.allreduce(
                fem.assemble_scalar(fem.form(one * ds(tag))), op=MPI.SUM
            )
        )
        for tag in tags
    }

    for idx, tag in enumerate(tags, start=1):
        if not tag_points[tag]:
            continue
        point_sum = np.zeros(2, dtype=np.float64)
        point_count = 0
        for seg in tag_points[tag]:
            point_sum += np.sum(seg, axis=0)
            point_count += int(seg.shape[0])
        centroid = point_sum / max(point_count, 1)
        direction = centroid - center[:2]
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            direction = np.array([1.0, 0.0])
            norm = 1.0
        direction /= norm
        label_pos = centroid + direction * (label_outset * radius)

        for seg in tag_points[tag]:
            ax.plot(seg[:, 0], seg[:, 1], color="tab:red", lw=3)

        ax.text(
            label_pos[0],
            label_pos[1],
            f"{idx}",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="circle,pad=0.45", fc="white", ec="gray", alpha=0.9),
        )
