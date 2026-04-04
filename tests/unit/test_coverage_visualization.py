"""Tests for visualization module edge cases."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest


class TestEITPlots:
    """Cover lines in eit_plots.py."""

    def test_style_not_available_warning(self):
        """Lines 69-70: style unavailable fallback."""
        import matplotlib.pyplot as plt
        from pyeidors.visualization.eit_plots import EITVisualizer
        with mock.patch.object(plt, "style") as mock_style:
            mock_style.use.side_effect = Exception("bad style")
            viz = EITVisualizer(style="nonexistent_style")
        assert viz is not None


class TestEITPlotHelpers:
    """Cover lines in eit_plot_helpers.py."""

    def test_resolve_eidors_diff_limits_both_none(self):
        from pyeidors.visualization.eit_plot_helpers import resolve_eidors_diff_limits
        vmin, vmax = resolve_eidors_diff_limits(np.array([1.0, -2.0, 3.0]), None, None)
        assert vmin == -3.0
        assert vmax == 3.0

    def test_resolve_eidors_diff_limits_zero_values(self):
        from pyeidors.visualization.eit_plot_helpers import resolve_eidors_diff_limits
        vmin, vmax = resolve_eidors_diff_limits(np.array([0.0, 0.0]), None, None)
        assert vmin < 0
        assert vmax > 0

    def test_resolve_eidors_diff_limits_vmin_only(self):
        from pyeidors.visualization.eit_plot_helpers import resolve_eidors_diff_limits
        vmin, vmax = resolve_eidors_diff_limits(np.array([1.0]), vmin=-5.0, vmax=None)
        assert vmin == -5.0
        assert vmax == 5.0

    def test_resolve_eidors_diff_limits_vmax_only(self):
        from pyeidors.visualization.eit_plot_helpers import resolve_eidors_diff_limits
        vmin, vmax = resolve_eidors_diff_limits(np.array([1.0]), vmin=None, vmax=5.0)
        assert vmin == -5.0
        assert vmax == 5.0

    def test_eidors_tick_vals_zero_scale(self):
        from pyeidors.visualization.eit_plot_helpers import eidors_tick_vals
        ticks = eidors_tick_vals(0.0, 0.0)
        assert len(ticks) == 1
        assert ticks[0] == 0.0

    def test_eidors_tick_vals_normal(self):
        from pyeidors.visualization.eit_plot_helpers import eidors_tick_vals
        ticks = eidors_tick_vals(1.0, 0.0)
        assert len(ticks) > 0

    def test_eidors_tick_vals_with_tick_div(self):
        from pyeidors.visualization.eit_plot_helpers import eidors_tick_vals
        ticks = eidors_tick_vals(2.0, 0.0, tick_div_in=4)
        assert len(ticks) > 0

    def test_apply_eidors_ticks_none_limits(self):
        from pyeidors.visualization.eit_plot_helpers import apply_eidors_ticks
        cbar = mock.MagicMock()
        apply_eidors_ticks(cbar, vmin=None, vmax=None)
        cbar.set_ticks.assert_not_called()

    def test_apply_eidors_ticks_normal(self):
        from pyeidors.visualization.eit_plot_helpers import apply_eidors_ticks
        cbar = mock.MagicMock()
        apply_eidors_ticks(cbar, vmin=-1.0, vmax=1.0)

    def test_format_colorbar_scientific(self):
        from pyeidors.visualization.eit_plot_helpers import format_colorbar
        cbar = mock.MagicMock()
        cbar.ax.yaxis.get_offset_text.return_value = mock.MagicMock()
        format_colorbar(cbar, "scientific")

    def test_format_colorbar_matlab_short(self):
        from pyeidors.visualization.eit_plot_helpers import format_colorbar
        cbar = mock.MagicMock()
        cbar.ax.yaxis.get_offset_text.return_value = mock.MagicMock()
        format_colorbar(cbar, "matlab_short")

    def test_format_colorbar_plain(self):
        from pyeidors.visualization.eit_plot_helpers import format_colorbar
        cbar = mock.MagicMock()
        cbar.ax.yaxis.get_offset_text.return_value = mock.MagicMock()
        format_colorbar(cbar, "plain")

    def test_extract_electrode_tags_with_non_integer(self):
        from pyeidors.visualization.eit_plot_helpers import extract_electrode_tags
        mesh = mock.MagicMock()
        mesh.association_table = {"electrode_1": 2, "non_electrode": "abc", "electrode_2": 3}
        tags = extract_electrode_tags(mesh)
        assert 2 in tags
        assert 3 in tags


class TestEITPlotRenderers:
    """Cover lines 244, 257 in eit_plot_renderers.py."""

    def test_render_convergence_with_save_path(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from pyeidors.visualization.eit_plot_renderers import render_convergence
        viz = mock.MagicMock()
        viz._text.return_value = "test"
        save = str(tmp_path / "conv.png")
        fig = render_convergence(
            viz,
            iterations=[1, 2, 3],
            errors=[0.5, 0.3, 0.1],
            title="Test",
            save_path=save,
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
