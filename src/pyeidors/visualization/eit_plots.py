"""EIT visualization entrypoints."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from dolfinx import fem

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
from .eit_plot_renderers import (
    render_conductivity,
    render_convergence,
    render_measurements,
    render_mesh,
    render_reconstruction_comparison,
)

logger = logging.getLogger(__name__)

try:
    Function = fem.Function
    FENICS_AVAILABLE = True
except Exception:
    FENICS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from pyeidors.utils.chinese_font_config import configure_chinese_font
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from pyeidors.utils.chinese_font_config import configure_chinese_font


class EITVisualizer:
    """Visualize mesh, conductivity and measurement diagnostics."""

    def __init__(self, style: str = "seaborn", figsize: Tuple[int, int] = (12, 8)):
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib not available, cannot perform visualization")
        self.figsize = figsize
        self._logger = logger
        try:
            plt.style.use(style)
        except Exception:
            logger.warning("Style %s not available, using default style", style)
        configure_chinese_font()

    def plot_mesh(
        self,
        mesh,
        title: str = "Mesh Structure",
        show_electrodes: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        return render_mesh(self, mesh, title=title, show_electrodes=show_electrodes, save_path=save_path)

    def plot_conductivity(
        self,
        mesh,
        conductivity: Union[Function, np.ndarray],
        title: Optional[str] = "Conductivity Distribution",
        colormap: str = "viridis",
        save_path: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        minimal: bool = False,
        show_electrodes: bool = False,
        scientific_notation: bool = False,
        colorbar_format: Optional[str] = None,
        transparent: bool = False,
    ) -> plt.Figure:
        return render_conductivity(
            self,
            mesh,
            conductivity,
            title,
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
        title: str = "Measurement Data",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        return render_measurements(self, data, title=title, save_path=save_path)

    def plot_reconstruction_comparison(
        self,
        mesh,
        true_conductivity,
        reconstructed_conductivity,
        title: str = "Reconstruction Comparison",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        return render_reconstruction_comparison(
            self,
            mesh,
            true_conductivity,
            reconstructed_conductivity,
            title,
            save_path,
        )

    def plot_convergence(
        self,
        iterations,
        errors,
        title: str = "Convergence Curve",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        return render_convergence(iterations, errors, title=title, save_path=save_path)

    # Compatibility wrappers used by tests/diagnostics.
    def _plot_electrodes(self, ax, electrode_vertices):
        return plot_electrodes(ax, electrode_vertices)

    def _interpolate_cell_to_node(self, mesh, cell_values):
        return interpolate_cell_to_node(mesh, cell_values)

    @staticmethod
    def _is_eidors_diff(colormap: Union[str, Any]) -> bool:
        return is_eidors_diff(colormap)

    @staticmethod
    def _resolve_colormap(colormap: Union[str, Any]) -> Any:
        return resolve_colormap(colormap)

    @staticmethod
    def _resolve_eidors_diff_limits(values: np.ndarray, vmin: Optional[float], vmax: Optional[float]):
        return resolve_eidors_diff_limits(values, vmin, vmax)

    @staticmethod
    def _apply_eidors_ticks(cbar: Any, vmin: Optional[float], vmax: Optional[float], ref_lev: float = 0.0, tick_div: Optional[int] = None) -> None:
        apply_eidors_ticks(cbar, vmin, vmax, ref_lev, tick_div)

    @staticmethod
    def _eidors_tick_vals(max_scale: float, ref_lev: float, tick_div_in: Optional[int] = None):
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


def create_visualizer(style: str = "seaborn") -> EITVisualizer:
    return EITVisualizer(style=style)
