"""EIT visualization entrypoints."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .eit_plot_helpers import (
    apply_eidors_ticks,
    cells,
    coordinates,
    eidors_tick_vals,
    extract_electrode_tags,
    format_colorbar,
    interpolate_cell_to_node,
    is_eidors_diff,
    num_cells,
    num_vertices,
    overlay_electrode_labels,
    plot_electrodes,
    raw_mesh,
    resolve_colormap,
    resolve_eidors_diff_limits,
)
from ..utils.plot_font_i18n import configure_plot_fonts, get_plot_texts, resolve_plot_language
from .eit_plot_renderers import (
    render_conductivity,
    render_convergence,
    render_measurements,
    render_mesh,
    render_reconstruction_comparison,
)

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    from dolfinx import fem
    Function = fem.Function
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class EITVisualizer:
    """Visualize mesh, conductivity and measurement diagnostics."""

    def __init__(
        self,
        style: str = "seaborn",
        figsize: tuple[int, int] = (12, 8),
        language: str | None = None,
    ):
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib not available, cannot perform visualization")
        self.figsize = figsize
        self._logger = logger
        self.requested_language = resolve_plot_language(language)
        try:
            plt.style.use(style)
        except Exception:
            logger.warning("Style %s not available, using default style", style)
        font_result = configure_plot_fonts(self.requested_language)
        self.language = font_result.effective_language
        self.selected_fonts = font_result.selected_fonts
        self._texts = get_plot_texts(self.language)

    def plot_mesh(
        self,
        mesh,
        title: str | None = None,
        show_electrodes: bool = True,
        save_path: str | None = None,
    ) -> plt.Figure:
        resolved_title = self._text("mesh_title") if title is None else title
        return render_mesh(self, mesh, title=resolved_title, show_electrodes=show_electrodes, save_path=save_path)

    def plot_conductivity(
        self,
        mesh,
        conductivity: np.ndarray | Any,
        title: str | None = None,
        colormap: str = "viridis",
        save_path: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        minimal: bool = False,
        show_electrodes: bool = False,
        scientific_notation: bool = False,
        colorbar_format: str | None = None,
        transparent: bool = False,
    ) -> plt.Figure:
        resolved_title = self._text("conductivity_title") if title is None else title
        return render_conductivity(
            self,
            mesh,
            conductivity,
            resolved_title,
            colormap,
            save_path,
            vmin,
            vmax,
            minimal,
            show_electrodes,
            scientific_notation,
            colorbar_format,
            transparent,
        )

    def plot_measurements(
        self,
        data,
        title: str | None = None,
        save_path: str | None = None,
    ) -> plt.Figure:
        resolved_title = self._text("measurement_title") if title is None else title
        return render_measurements(self, data, title=resolved_title, save_path=save_path)

    def plot_reconstruction_comparison(
        self,
        mesh,
        true_conductivity,
        reconstructed_conductivity,
        title: str | None = None,
        save_path: str | None = None,
    ) -> plt.Figure:
        resolved_title = self._text("recon_comparison") if title is None else title
        return render_reconstruction_comparison(
            self,
            mesh,
            true_conductivity,
            reconstructed_conductivity,
            resolved_title,
            save_path,
        )

    def plot_convergence(
        self,
        iterations,
        errors,
        title: str | None = None,
        save_path: str | None = None,
    ) -> plt.Figure:
        resolved_title = self._text("convergence") if title is None else title
        return render_convergence(self, iterations, errors, title=resolved_title, save_path=save_path)

    def _text(self, key: str, **kwargs) -> str:
        template = self._texts.get(key, key)
        if kwargs:
            return template.format(**kwargs)
        return template

    # Thin delegates kept as part of the renderer/test-facing helper surface.
    def _plot_electrodes(self, ax, electrode_vertices):
        return plot_electrodes(ax, electrode_vertices)

    def _interpolate_cell_to_node(self, mesh, cell_values):
        return interpolate_cell_to_node(mesh, cell_values)

    @staticmethod
    def _is_eidors_diff(colormap: str | Any) -> bool:
        return is_eidors_diff(colormap)

    @staticmethod
    def _resolve_colormap(colormap: str | Any) -> Any:
        return resolve_colormap(colormap)

    @staticmethod
    def _resolve_eidors_diff_limits(values: np.ndarray, vmin: float | None, vmax: float | None):
        return resolve_eidors_diff_limits(values, vmin, vmax)

    @staticmethod
    def _apply_eidors_ticks(cbar: Any, vmin: float | None, vmax: float | None, ref_lev: float = 0.0, tick_div: int | None = None) -> None:
        apply_eidors_ticks(cbar, vmin, vmax, ref_lev, tick_div)

    @staticmethod
    def _eidors_tick_vals(max_scale: float, ref_lev: float, tick_div_in: int | None = None):
        return eidors_tick_vals(max_scale, ref_lev, tick_div_in)

    @staticmethod
    def _format_colorbar(cbar: Any, format_mode: str) -> None:
        format_colorbar(cbar, format_mode)

    def _overlay_electrode_labels(self, ax, mesh, label_outset: float = 0.08):
        return overlay_electrode_labels(ax, mesh, label_outset)

    def _extract_electrode_tags(self, mesh):
        return extract_electrode_tags(mesh)

    @staticmethod
    def _raw_mesh(mesh):
        return raw_mesh(mesh)

    def _coordinates(self, mesh) -> np.ndarray:
        return coordinates(mesh)

    def _cells(self, mesh) -> np.ndarray:
        return cells(mesh)

    def _num_cells(self, mesh) -> int:
        return num_cells(mesh)

    def _num_vertices(self, mesh) -> int:
        return num_vertices(mesh)


def create_visualizer(style: str = "seaborn", language: str | None = None) -> EITVisualizer:
    return EITVisualizer(style=style, language=language)
