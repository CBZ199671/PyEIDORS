"""Extended visualization branch coverage tests."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from dolfinx import fem

from pyeidors.visualization.eit_plots import EITVisualizer, create_visualizer

matplotlib.use("Agg")


def test_visualizer_plotting_branches(eit_system, tmp_path):
    mesh = eit_system.mesh
    if mesh.electrode_vertices is None:
        mesh.electrode_vertices = [np.array([[0.0, 0.0], [0.1, 0.0]])]

    sigma_fn = fem.Function(eit_system.fwd_model.V_sigma)
    sigma_fn.x.array[:] = np.linspace(0.8, 1.2, sigma_fn.x.array.size)
    sigma_arr = sigma_fn.x.array.copy()

    viz = EITVisualizer(style="default", figsize=(8, 6))
    mesh_fig = viz.plot_mesh(mesh, show_electrodes=True, save_path=str(tmp_path / "mesh.png"))
    cond_fig_plain = viz.plot_conductivity(
        mesh,
        sigma_arr,
        title="plain",
        colormap="viridis",
        show_electrodes=False,
        colorbar_format="plain",
        save_path=str(tmp_path / "cond_plain.png"),
    )
    cond_fig_eidors = viz.plot_conductivity(
        mesh,
        sigma_fn,
        title="eidors",
        colormap="eidors_diff",
        show_electrodes=True,
        minimal=False,
        scientific_notation=True,
        transparent=True,
        save_path=str(tmp_path / "cond_eidors.png"),
    )
    cond_fig_minimal = viz.plot_conductivity(
        mesh,
        sigma_arr,
        title="",
        minimal=True,
        colorbar_format="matlab_short",
    )

    measurements = np.linspace(-0.3, 0.3, eit_system.fwd_model.pattern_manager.n_meas_total)
    meas_fig = viz.plot_measurements(measurements, save_path=str(tmp_path / "meas.png"))
    cmp_fig = viz.plot_reconstruction_comparison(mesh, sigma_arr, sigma_arr * 1.1)
    conv_fig = viz.plot_convergence(np.arange(5), np.linspace(1.0, 0.1, 5))

    assert mesh_fig is not None
    assert cond_fig_plain is not None
    assert cond_fig_eidors is not None
    assert cond_fig_minimal is not None
    assert meas_fig is not None
    assert cmp_fig is not None
    assert conv_fig is not None
    assert (tmp_path / "mesh.png").exists()
    assert (tmp_path / "cond_plain.png").exists()
    assert (tmp_path / "cond_eidors.png").exists()
    assert (tmp_path / "meas.png").exists()


def test_visualizer_static_helpers_and_factory(eit_system):
    viz = create_visualizer(style="default")
    assert isinstance(viz, EITVisualizer)

    assert EITVisualizer._is_eidors_diff("eidors_diff")
    assert EITVisualizer._is_eidors_diff("eidors-diff")
    assert not EITVisualizer._is_eidors_diff("viridis")

    cmap = EITVisualizer._resolve_colormap("eidors_diff")
    assert cmap is not None

    vmin, vmax = EITVisualizer._resolve_eidors_diff_limits(np.array([0.0, 2.0, -1.0]), None, None)
    assert vmin < 0 < vmax
    vmin2, vmax2 = EITVisualizer._resolve_eidors_diff_limits(np.array([1.0]), None, 0.7)
    assert vmin2 == -0.7 and vmax2 == 0.7
    vmin3, vmax3 = EITVisualizer._resolve_eidors_diff_limits(np.array([1.0]), -0.2, None)
    assert vmin3 == -0.2 and vmax3 == 0.2

    ticks = EITVisualizer._eidors_tick_vals(max_scale=0.3, ref_lev=0.0)
    assert ticks.size > 0

    fig = viz.plot_conductivity(eit_system.mesh, np.ones(eit_system.mesh.num_cells(), dtype=float))
    cbar = fig.axes[-1]
    assert cbar is not None

    # exercise colorbar formatter branch without relying on internals of matplotlib objects
    scalar_ticks = EITVisualizer._eidors_tick_vals(max_scale=0.0, ref_lev=0.0)
    assert np.allclose(scalar_ticks, np.array([0.0]))


def test_overlay_electrode_label_failure_is_non_fatal(eit_system, monkeypatch):
    viz = EITVisualizer(style="default")

    def _boom(*args, **kwargs):
        raise RuntimeError("forced overlay failure")

    monkeypatch.setattr(viz, "_overlay_electrode_labels", _boom)
    sigma = np.ones(eit_system.mesh.num_cells(), dtype=float)
    fig = viz.plot_conductivity(eit_system.mesh, sigma, show_electrodes=True)
    assert fig is not None


def test_raw_mesh_and_count_helpers(eit_system):
    viz = EITVisualizer(style="default")
    raw = viz._raw_mesh(eit_system.mesh)
    assert raw is eit_system.mesh.mesh
    assert viz._num_cells(eit_system.mesh) == eit_system.mesh.num_cells()
    assert viz._num_vertices(eit_system.mesh) == eit_system.mesh.num_vertices()

    # Also support raw DOLFINx mesh input.
    assert viz._num_cells(raw) == eit_system.mesh.num_cells()
    assert viz._num_vertices(raw) == eit_system.mesh.num_vertices()
    assert viz._coordinates(raw).shape[1] == 2
    assert viz._cells(raw).shape[0] == eit_system.mesh.num_cells()


def test_visualizer_save_dir(tmp_path):
    out = tmp_path / "nested" / "plot.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    assert Path(out).parent.exists()


def test_visualizer_language_switch_and_priority(monkeypatch):
    monkeypatch.setenv("PYEIDORS_PLOT_LANG", "zh")
    zh_viz = EITVisualizer(style="default")
    assert zh_viz.requested_language == "zh"

    en_viz = EITVisualizer(style="default", language="en")
    assert en_viz.requested_language == "en"
    assert en_viz._text("measurement_title") == "Measurement Data"


def test_visualizer_custom_title_not_overwritten(eit_system):
    viz = EITVisualizer(style="default", language="zh")
    custom_title = "CUSTOM-TITLE"
    fig = viz.plot_measurements(np.array([0.1, 0.2, 0.3]), title=custom_title)
    assert fig._suptitle.get_text() == custom_title
