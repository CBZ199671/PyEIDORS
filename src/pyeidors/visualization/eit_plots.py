"""EIT Visualization Module - Provides visualization for mesh, conductivity distribution, and measurement data."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import matplotlib.tri as tri
from typing import Optional, Tuple, Any, Union, Dict, List
import logging
import os
import sys

logger = logging.getLogger(__name__)

# Check optional dependencies
try:
    from fenics import Function, plot as fenics_plot, Measure, assemble
    FENICS_AVAILABLE = True
except ImportError:
    FENICS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from pyeidors.utils.chinese_font_config import configure_chinese_font
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from pyeidors.utils.chinese_font_config import configure_chinese_font


class EITVisualizer:
    """EIT Visualizer - Provides various EIT-related visualization functions."""

    def __init__(self, style: str = 'seaborn', figsize: Tuple[int, int] = (12, 8)):
        """Initialize visualizer.

        Args:
            style: matplotlib style.
            figsize: Default figure size.
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib not available, cannot perform visualization")

        self.figsize = figsize
        try:
            plt.style.use(style)
        except:
            logger.warning(f"Style {style} not available, using default style")
        configure_chinese_font()

    def plot_mesh(self, mesh, title: str = "Mesh Structure",
                  show_electrodes: bool = True, save_path: Optional[str] = None) -> plt.Figure:
        """Plot mesh structure.

        Args:
            mesh: FEniCS mesh object.
            title: Figure title.
            show_electrodes: Whether to show electrode positions.
            save_path: Save path (optional).

        Returns:
            matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Get mesh coordinates and connectivity
        coordinates = mesh.coordinates()
        cells = mesh.cells()

        # Create triangulation
        triangulation = tri.Triangulation(coordinates[:, 0], coordinates[:, 1], cells)

        # Plot mesh
        ax.triplot(triangulation, 'k-', alpha=0.3, linewidth=0.5)
        ax.scatter(coordinates[:, 0], coordinates[:, 1], s=1, c='blue', alpha=0.6)

        # If electrode info available, plot electrode positions
        if show_electrodes and hasattr(mesh, 'vertex_elec') and mesh.vertex_elec:
            self._plot_electrodes(ax, mesh.vertex_elec)

        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            bbox = None if minimal else 'tight'
            plt.savefig(save_path, dpi=300, bbox_inches=bbox)

        return fig

    def plot_conductivity(self, mesh, conductivity: Union[Function, np.ndarray],
                         title: Optional[str] = "Conductivity Distribution", colormap: str = 'viridis',
                         save_path: Optional[str] = None,
                         vmin: Optional[float] = None,
                         vmax: Optional[float] = None,
                         minimal: bool = False,
                         show_electrodes: bool = False,
                         scientific_notation: bool = False,
                         colorbar_format: Optional[str] = None,
                         transparent: bool = False) -> plt.Figure:
        """Plot conductivity distribution.

        Args:
            mesh: FEniCS mesh object.
            conductivity: Conductivity distribution (Function object or numpy array).
            title: Figure title.
            colormap: Color map (e.g., 'viridis' or 'eidors_diff').
            save_path: Save path (optional).
            colorbar_format: Colorbar format (plain / scientific / matlab_short).
            transparent: Whether to use transparent background.

        Returns:
            matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        fig.patch.set_facecolor('white')
        if transparent:
            fig.patch.set_alpha(0.0)

        # Handle different conductivity input types
        if isinstance(conductivity, Function):
            conductivity_values = conductivity.vector()[:]
        else:
            conductivity_values = np.array(conductivity)

        # Get mesh info
        coordinates = mesh.coordinates()
        cells = mesh.cells()

        # Create triangulation
        triangulation = tri.Triangulation(coordinates[:, 0], coordinates[:, 1], cells)

        # If conductivity is cell values, interpolate to nodes
        if len(conductivity_values) == mesh.num_cells():
            # Interpolate cell center values to nodes
            node_values = self._interpolate_cell_to_node(mesh, conductivity_values)
        else:
            node_values = conductivity_values

        eidors_style = self._is_eidors_diff(colormap)
        cmap = self._resolve_colormap(colormap)
        if eidors_style:
            vmin, vmax = self._resolve_eidors_diff_limits(node_values, vmin, vmax)

        # Plot conductivity distribution
        im = ax.tripcolor(
            triangulation,
            node_values,
            cmap=cmap,
            shading='flat' if eidors_style else 'gouraud',
            vmin=vmin,
            vmax=vmax,
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        if eidors_style:
            self._apply_eidors_ticks(cbar, vmin, vmax, ref_lev=0.0)
        format_mode = colorbar_format or ("scientific" if scientific_notation else "plain")
        self._format_colorbar(cbar, format_mode)
        cbar.ax.tick_params(labelsize=16, width=1.5)
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight("bold")
        if minimal:
            cbar.set_label('')
        else:
            cbar.set_label('Conductivity (S/m)', fontsize=18, fontweight='bold')

        # Plot mesh outline
        if eidors_style:
            ax.triplot(triangulation, 'k-', alpha=0.6, linewidth=0.4)
        elif not minimal:
            ax.triplot(triangulation, 'k-', alpha=0.2, linewidth=0.3)
        
        x_min, x_max = coordinates[:, 0].min(), coordinates[:, 0].max()
        y_min, y_max = coordinates[:, 1].min(), coordinates[:, 1].max()
        center_x = 0.5 * (x_min + x_max)
        center_y = 0.5 * (y_min + y_max)
        half_span = 0.5 * max(x_max - x_min, y_max - y_min)
        pad = 0.05 * half_span if half_span > 0 else 0.0
        limit = half_span + pad
        ax.set_xlim(center_x - limit, center_x + limit)
        ax.set_ylim(center_y - limit, center_y + limit)
        ax.set_aspect('equal')
        
        if minimal:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            if title:
                ax.set_title('')
            ax.set_facecolor('white')
        elif transparent:
            ax.set_facecolor('none')
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            if title:
                ax.set_title(title, fontsize=22, fontweight='bold')
            ax.set_xlabel('X', fontsize=18, fontweight='bold')
            ax.set_ylabel('Y', fontsize=18, fontweight='bold')
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=16, width=1.5)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight("bold")

        if show_electrodes and hasattr(mesh, "boundaries_mf") and mesh.boundaries_mf is not None:
            try:
                self._overlay_electrode_labels(ax, mesh)
            except Exception as exc:  # Don't interrupt main flow due to visualization failure
                logger.warning(f"Electrode visualization failed: {exc}")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=transparent)
        
        return fig
    
    def plot_measurements(self, data, title: str = "Measurement Data",
                         save_path: Optional[str] = None) -> plt.Figure:
        """Plot measurement data.

        Args:
            data: EITData object or measurement array.
            title: Figure title.
            save_path: Save path (optional).

        Returns:
            matplotlib figure object.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # Extract measurement data
        if hasattr(data, 'meas'):
            measurements = data.meas
        else:
            measurements = np.array(data)

        # Plot measurement time series
        ax1.plot(measurements, 'b-', linewidth=1.5, alpha=0.8)
        ax1.set_title('Measurement Sequence', fontweight='bold')
        ax1.set_xlabel('Measurement Index')
        ax1.set_ylabel('Voltage (V)')
        ax1.grid(True, alpha=0.3)

        # Plot measurement histogram
        ax2.hist(measurements, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('Measurement Distribution', fontweight='bold')
        ax2.set_xlabel('Voltage (V)')
        ax2.set_ylabel('Probability Density')
        ax2.grid(True, alpha=0.3)

        # Add statistics
        mean_val = np.mean(measurements)
        std_val = np.std(measurements)
        ax2.axvline(mean_val, color='red', linestyle='--',
                   label=f'Mean: {mean_val:.4f}')
        ax2.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7,
                   label=f'±Std: {std_val:.4f}')
        ax2.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7)
        ax2.legend()

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_reconstruction_comparison(self, mesh, true_conductivity, reconstructed_conductivity,
                                     title: str = "Reconstruction Comparison", save_path: Optional[str] = None) -> plt.Figure:
        """Plot comparison between true and reconstructed distributions.

        Args:
            mesh: FEniCS mesh object.
            true_conductivity: True conductivity distribution.
            reconstructed_conductivity: Reconstructed conductivity distribution.
            title: Figure title.
            save_path: Save path (optional).

        Returns:
            matplotlib figure object.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Process input data
        if isinstance(true_conductivity, Function):
            true_values = true_conductivity.vector()[:]
        else:
            true_values = np.array(true_conductivity)

        if isinstance(reconstructed_conductivity, Function):
            recon_values = reconstructed_conductivity.vector()[:]
        else:
            recon_values = np.array(reconstructed_conductivity)

        # Get mesh info
        coordinates = mesh.coordinates()
        cells = mesh.cells()
        triangulation = tri.Triangulation(coordinates[:, 0], coordinates[:, 1], cells)

        # Determine color range
        vmin = min(np.min(true_values), np.min(recon_values))
        vmax = max(np.max(true_values), np.max(recon_values))

        # Plot true distribution
        im1 = axes[0].tripcolor(triangulation, true_values, cmap='viridis',
                               vmin=vmin, vmax=vmax, shading='gouraud')
        axes[0].set_title('True Distribution', fontweight='bold')
        axes[0].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)

        # Plot reconstructed distribution
        im2 = axes[1].tripcolor(triangulation, recon_values, cmap='viridis',
                               vmin=vmin, vmax=vmax, shading='gouraud')
        axes[1].set_title('Reconstructed Distribution', fontweight='bold')
        axes[1].set_aspect('equal')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)

        # Plot error distribution
        error = np.abs(true_values - recon_values)
        im3 = axes[2].tripcolor(triangulation, error, cmap='hot', shading='gouraud')
        axes[2].set_title('Absolute Error', fontweight='bold')
        axes[2].set_aspect('equal')
        plt.colorbar(im3, ax=axes[2], shrink=0.8)

        # Calculate reconstruction error metric
        relative_error = np.linalg.norm(error) / np.linalg.norm(true_values)

        fig.suptitle(f'{title} (Relative Error: {relative_error:.4f})', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def _plot_electrodes(self, ax, electrode_vertices):
        """Plot electrode positions."""
        for i, electrode in enumerate(electrode_vertices):
            if electrode:  # Check if electrode has vertices
                electrode_array = np.array(electrode)
                ax.plot(electrode_array[:, 0], electrode_array[:, 1], 'ro-',
                       markersize=6, linewidth=2, label=f'Electrode {i+1}' if i < 5 else "")

        if len(electrode_vertices) <= 5:
            ax.legend()

    def _interpolate_cell_to_node(self, mesh, cell_values):
        """Interpolate cell center values to nodes."""
        node_values = np.zeros(mesh.num_vertices())
        node_counts = np.zeros(mesh.num_vertices())

        for cell_idx, cell in enumerate(mesh.cells()):
            for vertex_idx in cell:
                node_values[vertex_idx] += cell_values[cell_idx]
                node_counts[vertex_idx] += 1

        # Avoid division by zero
        node_counts[node_counts == 0] = 1
        node_values /= node_counts

        return node_values

    @staticmethod
    def _is_eidors_diff(colormap: Union[str, Any]) -> bool:
        return isinstance(colormap, str) and colormap.lower() in {"eidors_diff", "eidors-diff"}

    @staticmethod
    def _resolve_colormap(colormap: Union[str, Any]) -> Any:
        if isinstance(colormap, str) and colormap.lower() in {"eidors_diff", "eidors-diff"}:
            return LinearSegmentedColormap.from_list(
                "eidors_diff",
                ["#1f3a93", "#ffffff", "#b30000"],
            )
        return colormap

    @staticmethod
    def _resolve_eidors_diff_limits(values: np.ndarray,
                                    vmin: Optional[float],
                                    vmax: Optional[float]) -> Tuple[float, float]:
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

    @staticmethod
    def _eidors_tick_vals(max_scale: float,
                          ref_lev: float,
                          tick_div_in: Optional[int] = None) -> np.ndarray:
        if max_scale <= 0:
            return np.array([ref_lev], dtype=float)
        F = 2.0
        ord_of_mag = 10 ** np.floor(np.log10(max_scale * F)) / F
        scale1 = np.floor(max_scale / ord_of_mag + 2 * np.finfo(float).eps)
        if scale1 / F >= 8:
            fms = F * 8
            tick_div = 2
        elif scale1 / F >= 6:
            fms = F * 6
            tick_div = 2
        elif scale1 / F >= 4:
            fms = F * 4
            tick_div = 2
        elif scale1 / F >= 3:
            fms = F * 3
            tick_div = 3
        elif scale1 / F >= 2:
            fms = F * 2
            tick_div = 2
        elif scale1 / F >= 1.5:
            fms = F * 1.5
            tick_div = 3
        elif scale1 / F >= 1:
            fms = F * 1
            tick_div = 2
        else:
            fms = F * 0.5
            tick_div = 2

        if tick_div_in is not None:
            tick_div = tick_div_in

        scale_r = ord_of_mag * fms
        ord_of_mag = 10 ** np.floor(np.log10(max_scale))
        ref_r = ord_of_mag * np.round(ref_lev / ord_of_mag)
        return np.linspace(-2, 2, tick_div * 4 + 1) * scale_r + ref_r

    @staticmethod
    def _apply_eidors_ticks(cbar: Any,
                            vmin: Optional[float],
                            vmax: Optional[float],
                            ref_lev: float = 0.0,
                            tick_div: Optional[int] = None) -> None:
        if vmin is None or vmax is None:
            return
        max_scale = max(abs(vmax - ref_lev), abs(ref_lev - vmin))
        tick_vals = EITVisualizer._eidors_tick_vals(max_scale, ref_lev, tick_div)
        if tick_vals.size == 0:
            return
        eps = max(max_scale * 1e-12, 1e-12)
        tick_vals = tick_vals[(tick_vals >= vmin - eps) & (tick_vals <= vmax + eps)]
        if tick_vals.size > 0:
            cbar.set_ticks(tick_vals)

    @staticmethod
    def _format_colorbar(cbar: Any, format_mode: str) -> None:
        mode = (format_mode or "plain").lower()
        def _fmt_sci_adaptive(x: float, _: float) -> str:
            if x == 0:
                return "0"
            ax = abs(x)
            exp = int(np.floor(np.log10(ax)))
            mantissa = x / (10 ** exp)
            mantissa_rounded = round(mantissa, 1)
            if abs(mantissa_rounded) >= 10:
                mantissa_rounded /= 10
                exp += 1
            if abs(mantissa_rounded - round(mantissa_rounded)) < 1e-8:
                mantissa_str = f"{mantissa_rounded:.0f}"
            else:
                mantissa_str = f"{mantissa_rounded:.1f}"
            exp_str = f"{exp:+03d}"
            return f"{mantissa_str}e{exp_str}"

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
            # Avoid mathtext to prevent missing glyph warnings with CJK fonts.
            formatter = ScalarFormatter(useMathText=False)
            formatter.set_useOffset(False)
            formatter.set_scientific(False)
        cbar.formatter = formatter
        cbar.ax.yaxis.get_offset_text().set_visible(False)
        cbar.update_ticks()

    def _overlay_electrode_labels(self, ax, mesh, label_outset: float = 0.08):
        """Overlay electrode numbers and lengths on conductivity plot based on boundaries_mf."""
        try:
            from dolfin import facets  # Delayed import to avoid errors in non-FEniCS environments
        except Exception as exc:
            raise RuntimeError(f"Cannot import dolfin.facets: {exc}")

        coords = mesh.coordinates()
        center = coords.mean(axis=0)
        radius = np.max(np.linalg.norm(coords - center, axis=1)) + 1e-12

        # Parse electrode tags (numbered 1..N in sorted tag order)
        tags = self._extract_electrode_tags(mesh)
        if not tags:
            raise RuntimeError("No electrode tags parsed from association_table")

        # Collect all vertex coordinates for each tag and compute centroid and length
        tag_points: Dict[int, List[np.ndarray]] = {t: [] for t in tags}
        for facet in facets(mesh):
            tag = mesh.boundaries_mf[facet.index()]
            if tag in tag_points:
                vs = facet.entities(0)
                tag_points[tag].append(coords[vs][:, :2])

        # Compute measures
        ds = Measure("ds", domain=mesh, subdomain_data=mesh.boundaries_mf)
        lengths = {tag: float(assemble(1 * ds(tag))) for tag in tags}

        for idx, tag in enumerate(tags, start=1):
            if not tag_points[tag]:
                continue
            pts = np.vstack(tag_points[tag])
            centroid = pts.mean(axis=0)
            direction = centroid - center[:2]
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                direction = np.array([1.0, 0.0])
                norm = 1.0
            direction /= norm
            label_pos = centroid + direction * (label_outset * radius)

            # Plot electrode segments
            for seg in tag_points[tag]:
                ax.plot(seg[:, 0], seg[:, 1], color="tab:red", lw=3)

            # Label with electrode number
            text = f"{idx}"
            ax.text(
                label_pos[0],
                label_pos[1],
                text,
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color="black",
                bbox=dict(boxstyle="circle,pad=0.45", fc="white", ec="gray", alpha=0.9),
            )

    def _extract_electrode_tags(self, mesh) -> List[int]:
        assoc = getattr(mesh, "association_table", {}) or {}
        tags: List[int] = []
        for k, v in assoc.items():
            try:
                tag_val = int(v)
            except Exception:
                continue
            if isinstance(k, str) and k.lower().startswith("electrode"):
                tags.append(tag_val)
            elif isinstance(k, (int, np.integer)) and k >= 2:
                tags.append(tag_val)
        tags = sorted(set(tags))
        return tags
    
    def plot_convergence(self, iterations, errors, title: str = "Convergence Curve",
                        save_path: Optional[str] = None) -> plt.Figure:
        """Plot algorithm convergence curve.

        Args:
            iterations: Iteration count array.
            errors: Corresponding error values.
            title: Figure title.
            save_path: Save path (optional).

        Returns:
            matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.semilogy(iterations, errors, 'b-o', linewidth=2, markersize=6)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Error (log scale)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


def create_visualizer(style: str = 'seaborn') -> EITVisualizer:
    """Create EIT visualizer instance."""
    return EITVisualizer(style=style)
