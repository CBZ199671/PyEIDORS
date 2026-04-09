"""Renderer functions for EIT visualizations."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from dolfinx import fem

from .eit_plot_helpers import (
    apply_eidors_ticks,
    cells,
    coordinates,
    format_colorbar,
    interpolate_cell_to_node,
    is_eidors_diff,
    num_cells,
    resolve_colormap,
    resolve_eidors_diff_limits,
)

Function = fem.Function


def render_mesh(viz, mesh, title: str, show_electrodes: bool, save_path: str | None):
    fig, ax = plt.subplots(figsize=viz.figsize)
    coords = coordinates(mesh)
    mesh_cells = cells(mesh)
    triangulation = tri.Triangulation(coords[:, 0], coords[:, 1], mesh_cells)

    ax.triplot(triangulation, "k-", alpha=0.3, linewidth=0.5)
    ax.scatter(coords[:, 0], coords[:, 1], s=1, c="blue", alpha=0.6)

    if show_electrodes:
        electrode_vertices = getattr(mesh, "electrode_vertices", None)
        if electrode_vertices:
            viz._plot_electrodes(ax, electrode_vertices)

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(viz._text("axis_x"))
    ax.set_ylabel(viz._text("axis_y"))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def render_conductivity(
    viz,
    mesh,
    conductivity: Function | np.ndarray,
    title: str | None,
    colormap,
    save_path: str | None,
    vmin: float | None,
    vmax: float | None,
    minimal: bool,
    show_electrodes: bool,
    scientific_notation: bool,
    colorbar_format: str | None,
    transparent: bool,
):
    fig, ax = plt.subplots(figsize=viz.figsize)
    fig.patch.set_facecolor("white")
    if transparent:
        fig.patch.set_alpha(0.0)

    conductivity_values = conductivity.x.array if isinstance(conductivity, Function) else np.asarray(conductivity)
    coords = coordinates(mesh)
    mesh_cells = cells(mesh)
    triangulation = tri.Triangulation(coords[:, 0], coords[:, 1], mesh_cells)

    node_values = (
        interpolate_cell_to_node(mesh, conductivity_values)
        if len(conductivity_values) == num_cells(mesh)
        else conductivity_values
    )

    eidors_style = is_eidors_diff(colormap)
    cmap = resolve_colormap(colormap)
    if eidors_style:
        vmin, vmax = resolve_eidors_diff_limits(node_values, vmin, vmax)

    im = ax.tripcolor(
        triangulation,
        node_values,
        cmap=cmap,
        shading="flat" if eidors_style else "gouraud",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    if eidors_style:
        apply_eidors_ticks(cbar, vmin, vmax, ref_lev=0.0)
    format_mode = colorbar_format or ("scientific" if scientific_notation else "plain")
    format_colorbar(cbar, format_mode)
    cbar.ax.tick_params(labelsize=16, width=1.5)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
    if minimal:
        cbar.set_label("")
    else:
        cbar.set_label(viz._text("conductivity_label"), fontsize=18, fontweight="bold")

    if eidors_style:
        ax.triplot(triangulation, "k-", alpha=0.6, linewidth=0.4)
    elif not minimal:
        ax.triplot(triangulation, "k-", alpha=0.2, linewidth=0.3)

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    center_x, center_y = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    half_span = 0.5 * max(x_max - x_min, y_max - y_min)
    pad = 0.05 * half_span if half_span > 0 else 0.0
    limit = half_span + pad
    ax.set_xlim(center_x - limit, center_x + limit)
    ax.set_ylim(center_y - limit, center_y + limit)
    ax.set_aspect("equal")

    if minimal:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.set_facecolor("white")
    elif transparent:
        ax.set_facecolor("none")
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        if title:
            ax.set_title(title, fontsize=22, fontweight="bold")
        ax.set_xlabel(viz._text("axis_x"), fontsize=18, fontweight="bold")
        ax.set_ylabel(viz._text("axis_y"), fontsize=18, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=16, width=1.5)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

    if show_electrodes and getattr(mesh, "facet_tags", None) is not None:
        try:
            viz._overlay_electrode_labels(ax, mesh)
        except Exception as exc:
            viz._logger.warning("Electrode visualization failed: %s", exc)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=transparent)
    return fig


def render_measurements(viz, data, title: str, save_path: str | None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=viz.figsize)
    measurements = data.meas if hasattr(data, "meas") else np.asarray(data)

    ax1.plot(measurements, "b-", linewidth=1.5, alpha=0.8)
    ax1.set_title(viz._text("measurement_sequence"), fontweight="bold")
    ax1.set_xlabel(viz._text("measurement_index"))
    ax1.set_ylabel(viz._text("voltage"))
    ax1.grid(True, alpha=0.3)

    ax2.hist(measurements, bins=50, density=True, alpha=0.7, color="skyblue", edgecolor="black")
    ax2.set_title(viz._text("measurement_distribution"), fontweight="bold")
    ax2.set_xlabel(viz._text("voltage"))
    ax2.set_ylabel(viz._text("probability_density"))
    ax2.grid(True, alpha=0.3)

    mean_val = np.mean(measurements)
    std_val = np.std(measurements)
    ax2.axvline(mean_val, color="red", linestyle="--", label=viz._text("mean", value=mean_val))
    ax2.axvline(
        mean_val + std_val,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=viz._text("std", value=std_val),
    )
    ax2.axvline(mean_val - std_val, color="orange", linestyle="--", alpha=0.7)
    ax2.legend()

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def render_reconstruction_comparison(
    viz,
    mesh,
    true_conductivity,
    reconstructed_conductivity,
    title: str,
    save_path: str | None,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    true_values = true_conductivity.x.array if isinstance(true_conductivity, Function) else np.asarray(true_conductivity)
    recon_values = (
        reconstructed_conductivity.x.array
        if isinstance(reconstructed_conductivity, Function)
        else np.asarray(reconstructed_conductivity)
    )

    coords = coordinates(mesh)
    mesh_cells = cells(mesh)
    triangulation = tri.Triangulation(coords[:, 0], coords[:, 1], mesh_cells)

    true_plot_values = interpolate_cell_to_node(mesh, true_values) if len(true_values) == num_cells(mesh) else true_values
    recon_plot_values = interpolate_cell_to_node(mesh, recon_values) if len(recon_values) == num_cells(mesh) else recon_values

    vmin, vmax = min(np.min(true_plot_values), np.min(recon_plot_values)), max(np.max(true_plot_values), np.max(recon_plot_values))

    im1 = axes[0].tripcolor(triangulation, true_plot_values, cmap="viridis", vmin=vmin, vmax=vmax, shading="gouraud")
    axes[0].set_title(viz._text("true_distribution"), fontweight="bold")
    axes[0].set_aspect("equal")
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    im2 = axes[1].tripcolor(triangulation, recon_plot_values, cmap="viridis", vmin=vmin, vmax=vmax, shading="gouraud")
    axes[1].set_title(viz._text("reconstructed_distribution"), fontweight="bold")
    axes[1].set_aspect("equal")
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    error = np.abs(true_plot_values - recon_plot_values)
    im3 = axes[2].tripcolor(triangulation, error, cmap="hot", shading="gouraud")
    axes[2].set_title(viz._text("absolute_error"), fontweight="bold")
    axes[2].set_aspect("equal")
    plt.colorbar(im3, ax=axes[2], shrink=0.8)

    relative_error = np.linalg.norm(error) / np.linalg.norm(true_plot_values)
    fig.suptitle(
        f"{title} ({viz._text('relative_error', value=relative_error)})",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def render_convergence(viz, iterations, errors, title: str, save_path: str | None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(iterations, errors, "b-o", linewidth=2, markersize=6)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(viz._text("iteration"))
    ax.set_ylabel(viz._text("error_log"))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
