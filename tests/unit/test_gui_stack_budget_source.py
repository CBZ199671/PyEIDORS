from __future__ import annotations

import inspect

import numpy as np

import eit_app.ui.complex_channels as complex_channels
import eit_app.ui.boundary_voltage_plot_widget as boundary_voltage_module
import eit_app.ui.conductivity_3d_widget as widget3d
import eit_app.ui.conductivity_image_widget as image_widget
import eit_app.ui.main_window as main_window_module
from eit_app.ui.boundary_voltage_plot_widget import BoundaryVoltagePlotWidget
from eit_app.ui.conductivity_3d_widget import (
    ANOMALY_MODE_POSITIVE,
    Conductivity3DWidget,
    DISPLAY_MODE_VOLUME,
    _cell_centers,
    _extract_cells_from_mask,
    _face_cell_values,
    _finite_mask_or_none,
    _highlight_face_vertices_and_values,
    _point_cloud_highlight_arrays,
    _sample_background_indices,
    _point_cloud_sample_indices,
    _score_count_peak_above_floor,
)
from eit_app.ui.complex_channels import (
    COMPOSITE_CHANNEL,
    IMAG_CHANNEL,
    MAGNITUDE_CHANNEL,
    PHASE_CHANNEL,
    REAL_CHANNEL,
    channel_values,
    has_complex_component,
)
from eit_app.ui.hardware.equipotential_plot_widget import (
    EquipotentialPlotWidget,
    _display_float_array as _equipotential_display_float_array,
    _finite_min_max,
)
from eit_app.ui.hardware.reconstruction_widget import (
    ReconstructionWidget,
    _display_float_array as _reconstruction_display_float_array,
)
from eit_app.ui.main_window import EITWorkstation


def test_v285_mpl3d_electrode_patch_generation_avoids_column_stack() -> None:
    source = inspect.getsource(Conductivity3DWidget._build_mpl3d_electrode_collection)

    assert "np.column_stack" not in source


def test_v370_mpl3d_electrode_patch_polygons_use_float32_direct_fill() -> None:
    source = inspect.getsource(Conductivity3DWidget._build_mpl3d_electrode_collection)

    assert "dtype=np.float64" not in source
    assert "dtype=float" not in source
    assert "np.array(" not in source
    assert "quad = np.empty((4, 3), dtype=np.float32)" in source
    assert "polys.append(quad)" in source


def test_v308_point_cloud_sampling_direct_fills_selected_indices() -> None:
    source = inspect.getsource(_point_cloud_sample_indices)

    assert "np.concatenate" not in source
    assert "return np.sort(sampled)" not in source
    assert "sampled.sort()" in source

    centers = np.zeros((30, 3), dtype=np.float64)
    centers[:, 0] = np.linspace(0.0, 1.0, 30)
    sigma = np.ones(30, dtype=np.float64)
    sigma[[4, 20]] = 2.0

    sample_idx = _point_cloud_sample_indices(
        sigma,
        centers,
        anomaly_mode=ANOMALY_MODE_POSITIVE,
        max_points=6,
    )

    assert sample_idx.size <= 6
    assert {4, 20}.issubset(set(sample_idx.tolist()))


def test_v450_point_cloud_background_sampling_avoids_recount_pass() -> None:
    source = inspect.getsource(_sample_background_indices)

    assert "np.count_nonzero(mask_arr)" not in source
    assert "actual_background_count = background_count" in source
    assert "candidates = ranks.copy()" not in source
    assert "candidates = ranks" in source

    mask = np.array([True, False, True, False, False], dtype=bool)
    background = _sample_background_indices(
        mask,
        np.array([0, 2], dtype=np.int64),
        max_count=3,
    )

    np.testing.assert_array_equal(background, [1, 3, 4])


def test_v471_spatial_candidate_center_finite_scan_uses_work_buffer() -> None:
    source = inspect.getsource(widget3d._spatially_coherent_anomaly_mask)
    helper_source = inspect.getsource(widget3d._all_finite_values)

    assert "np.isfinite(candidate_centers).all()" not in source
    assert "_all_finite_values(candidate_centers)" in source
    assert "np.isfinite(chunk, out=chunk_mask)" in helper_source
    assert widget3d._all_finite_values(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    assert not widget3d._all_finite_values(
        np.array([[1.0, np.nan, 3.0]], dtype=np.float32)
    )


def test_v600_spatial_nearest_radius_sanitizes_in_bounded_chunks(
    monkeypatch,
) -> None:
    helper_source = inspect.getsource(widget3d._nan_invalid_nearest_distances)
    source = inspect.getsource(widget3d._spatially_coherent_anomaly_mask)

    assert "nearest_valid" not in source
    assert "_nan_invalid_nearest_distances(nearest)" in source
    assert "np.isfinite(chunk, out=mask)" in helper_source
    assert "np.greater(chunk, 1.0e-12, out=mask)" in helper_source
    assert "np.copyto(chunk, np.nan, where=mask)" in helper_source

    monkeypatch.setattr(widget3d, "_FINITE_SCAN_CHUNK_ITEMS", 2)
    nearest = np.array([0.4, 0.0, np.inf, 0.8, np.nan, 1.0e-13], dtype=np.float64)

    assert widget3d._nan_invalid_nearest_distances(nearest)
    np.testing.assert_allclose(nearest[[0, 3]], np.array([0.4, 0.8]))
    assert np.isnan(nearest[[1, 2, 4, 5]]).all()

    invalid = np.array([0.0, np.inf, np.nan, 1.0e-13], dtype=np.float64)
    assert not widget3d._nan_invalid_nearest_distances(invalid)
    assert np.isnan(invalid).all()


def test_v547_spatial_component_masses_use_bounded_any_finite() -> None:
    source = inspect.getsource(widget3d._spatially_coherent_anomaly_mask)
    helper_source = inspect.getsource(widget3d._any_finite_values)

    assert "np.isfinite(masses).any()" not in source
    assert "_any_finite_values(masses)" in source
    assert "np.isfinite(chunk, out=chunk_mask)" in helper_source
    assert widget3d._any_finite_values(np.array([np.nan, 2.0], dtype=np.float32))
    assert not widget3d._any_finite_values(
        np.array([np.nan, np.inf, -np.inf], dtype=np.float32)
    )


def test_v474_simulation_voltage_fit_finite_check_uses_bounded_scan_source() -> None:
    source = inspect.getsource(EITWorkstation._simulation_reconstructed_voltage_fit)
    helper_source = inspect.getsource(main_window_module._all_finite_values)

    assert "np.isfinite(reconstructed).all()" not in source
    assert "_all_finite_values(reconstructed)" in source
    assert "np.isfinite(chunk, out=chunk_mask)" in helper_source
    assert main_window_module._all_finite_values(np.array([1.0, 2.0]))
    assert not main_window_module._all_finite_values(np.array([1.0, np.nan]))


def test_v501_main_window_complex_measurement_scan_uses_work_buffers() -> None:
    source = inspect.getsource(EITWorkstation._on_run_sim_inverse)
    helper_source = inspect.getsource(main_window_module._has_abs_value_above)

    assert "np.any(np.abs(homog_imag) > 1.0e-12)" not in source
    assert "np.any(np.abs(target_imag) > 1.0e-12)" not in source
    assert "_has_abs_value_above(homog_imag" in source
    assert "_has_abs_value_above(target_imag" in source
    assert "np.abs(chunk, out=abs_chunk)" in helper_source
    assert "np.greater(abs_chunk, resolved_threshold, out=mask_chunk)" in helper_source
    assert "abs_work = np.empty(block_size, dtype=np.float64)" not in helper_source
    assert main_window_module._has_abs_value_above(
        np.array([0.0, 2.0e-12]), threshold=1.0e-12, chunk_size=1
    )
    assert not main_window_module._has_abs_value_above(
        np.array([0.0, np.nan]), threshold=1.0e-12, chunk_size=1
    )
    assert main_window_module._has_abs_value_above(
        np.array([0.0, 2.0e-12], dtype=np.float32),
        threshold=1.0e-12,
        chunk_size=1,
    )


def test_v310_boundary_voltage_y_range_streams_min_max_without_concatenate() -> None:
    source = inspect.getsource(BoundaryVoltagePlotWidget._apply_y_range)
    helper_source = inspect.getsource(boundary_voltage_module._finite_min_max)

    assert "np.concatenate" not in source
    assert "[np.isfinite" not in source
    assert "dtype=np.float64" not in source
    assert "_finite_min_max(values)" in source
    assert "np.isfinite(chunk, out=finite)" in helper_source
    assert "np.min(chunk, where=finite" in helper_source
    assert "np.max(chunk, where=finite" in helper_source
    assert "y_min" in source
    assert "y_max" in source

    values = np.array([np.nan, 2.0, -1.5, np.inf], dtype=np.float32)
    assert boundary_voltage_module._finite_min_max(values) == (-1.5, 2.0)
    assert boundary_voltage_module._finite_min_max(np.array([np.nan, np.inf])) is None


def test_v354_boundary_voltage_overlay_preserves_projected_dtype() -> None:
    source = inspect.getsource(BoundaryVoltagePlotWidget._set_reconstructed_overlay)

    assert "np.asarray(reconstructed, dtype=np.float64)" not in source
    assert "np.asarray(reconstructed)" in source
    assert "dtype=np.float32" in source


def test_v353_conductivity_image_display_preserves_float32_dtype() -> None:
    update_source = inspect.getsource(image_widget.ConductivityImageWidget.update_image)
    helper_source = inspect.getsource(image_widget._display_conductivity_values)
    projection_source = inspect.getsource(image_widget._project_cells_to_triangles)

    assert "np.asarray(display_conductivity, dtype=float)" not in update_source
    assert "np.asarray(z, dtype=float)" not in projection_source
    assert "dtype=np.float64" not in helper_source

    values = np.array([1.0, 2.0], dtype=np.float32)
    display_values = image_widget._display_conductivity_values(values)

    assert display_values.dtype == np.dtype(np.float32)
    assert np.shares_memory(display_values, values)

    complex_values = np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64)
    complex_display = image_widget._display_conductivity_values(complex_values)

    assert complex_display.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(
        complex_display,
        np.array([1.0, 3.0], dtype=np.float32),
    )

    integer_values = np.array([1, 2], dtype=np.int32)
    integer_display = image_widget._display_conductivity_values(integer_values)

    assert integer_display.dtype == np.dtype(np.float32)


def test_v378_conductivity_image_square_limits_stream_finite_bounds(
    monkeypatch,
) -> None:
    source = inspect.getsource(
        image_widget.ConductivityImageWidget._apply_square_data_limits
    )
    helper_source = inspect.getsource(image_widget._finite_xy_bounds)

    assert "x[finite]" not in source
    assert "y[finite]" not in source
    assert "np.isfinite(x_chunk) & np.isfinite(y_chunk)" not in helper_source
    assert "np.isfinite(x_chunk, out=finite)" in helper_source
    assert "np.isfinite(y_chunk, out=other_finite)" in helper_source
    assert "np.logical_and(finite, other_finite, out=finite)" in helper_source
    assert "np.min(x_chunk, where=finite" in helper_source
    assert "np.max(y_chunk, where=finite" in helper_source

    monkeypatch.setattr(image_widget, "_IMAGE_SCAN_CHUNK_ITEMS", 2)
    x = np.array([np.nan, -1.0, 3.0, np.inf, 5.0], dtype=np.float32)
    y = np.array([0.0, 4.0, -2.0, 8.0, np.inf], dtype=np.float32)

    assert image_widget._finite_xy_bounds(x, y) == (-1.0, 3.0, -2.0, 4.0)
    assert (
        image_widget._finite_xy_bounds(
            np.array([np.nan, np.inf], dtype=np.float32),
            np.array([1.0, 2.0], dtype=np.float32),
        )
        is None
    )


def test_v314_conductivity_3d_cell_centers_fallback_streams_vertices() -> None:
    source = inspect.getsource(_cell_centers)

    assert "coords, dtype=float)[cells" not in source
    assert "dtype=float" not in source
    assert "[cells, :3]" not in source
    assert "_display_coords_array(coords)[:, :3]" in source
    assert "_compute_cell_centers" in source


def test_v365_conductivity_3d_cell_centers_fallback_preserves_float32(
    monkeypatch,
) -> None:
    monkeypatch.setattr(widget3d, "cached_cell_centers", lambda *_args, **_kwargs: None)
    coords = np.arange(15, dtype=np.float32).reshape(5, 3)
    cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)

    centers = _cell_centers(coords, cells)

    assert centers.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(centers, coords[cells].mean(axis=1))


def test_v567_conductivity_3d_display_payload_downcasts_to_float32() -> None:
    coords_source = inspect.getsource(widget3d._display_coords_array)
    sigma_source = inspect.getsource(widget3d._display_sigma_array)

    assert "coords.dtype == np.dtype(np.float32)" in coords_source
    assert "np.asarray(coords, dtype=np.float32)" in coords_source
    assert "np.asarray(sigma).dtype == np.dtype(np.float32)" in sigma_source
    assert "np.asarray(sigma, dtype=np.float32)" in sigma_source


def test_v332_conductivity_3d_entrypoint_has_no_duplicate_array_work() -> None:
    update_source = inspect.getsource(Conductivity3DWidget.update_image)
    point_source = inspect.getsource(
        Conductivity3DWidget._add_pyvista_point_cloud_actors
    )

    assert update_source.count("_display_coords_array(node_coords)") == 1
    assert point_source.count("_point_cloud_highlight_arrays(") == 1


def test_v389_point_cloud_highlight_direct_fills_from_mask() -> None:
    helper_source = inspect.getsource(_point_cloud_highlight_arrays)
    pyvista_source = inspect.getsource(
        Conductivity3DWidget._add_pyvista_point_cloud_actors
    )
    mpl_source = inspect.getsource(
        Conductivity3DWidget._render_matplotlib_point_cloud_scene
    )

    assert "display_centers[inhom_mask" not in pyvista_source
    assert "display_centers[inhom_mask" not in mpl_source
    assert "display_sigma[inhom_mask" not in pyvista_source
    assert "display_sigma[inhom_mask" not in mpl_source
    assert "np.flatnonzero" not in helper_source
    assert "np.take(" not in helper_source
    assert "active_count = int(np.count_nonzero(active_mask))" in helper_source
    assert "highlight_centers = np.empty(" in helper_source
    assert pyvista_source.count("_point_cloud_highlight_arrays(") == 1
    assert mpl_source.count("_point_cloud_highlight_arrays(") == 1

    centers = np.arange(18, dtype=np.float32).reshape(6, 3)
    sigma = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    mask = np.array([False, True, False, True, False, False])

    highlight_centers, highlight_sigma = _point_cloud_highlight_arrays(
        centers,
        sigma,
        mask,
    )

    np.testing.assert_array_equal(highlight_centers, centers[[1, 3]])
    np.testing.assert_array_equal(highlight_sigma, sigma[[1, 3]])
    assert highlight_centers.dtype == np.dtype(np.float32)
    assert highlight_sigma.dtype == np.dtype(np.float32)


def test_v355_matplotlib_surface_uses_dtype_preserving_face_values() -> None:
    source = inspect.getsource(Conductivity3DWidget._render_matplotlib_scene)
    helper_source = inspect.getsource(_face_cell_values)

    assert "sigma.astype(float" not in source
    assert "dtype=float" not in source
    assert "face_values = np.take(cell_sigma, source_indices)" not in source
    assert "_face_cell_values(cell_sigma, source_indices)" in source
    assert "np.take(values, source_indices, out=out)" in helper_source
    highlight_source = inspect.getsource(_highlight_face_vertices_and_values)
    assert "dtype=sigma_arr.dtype" in highlight_source

    sigma = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    face_values = _face_cell_values(sigma, np.array([2, 0], dtype=np.int64))

    np.testing.assert_allclose(face_values, np.array([3.0, 1.0], dtype=np.float32))
    assert face_values.dtype == np.dtype(np.float32)


def test_v371_matplotlib_point_face_values_avoid_indexed_nanmean() -> None:
    source = inspect.getsource(Conductivity3DWidget._render_matplotlib_scene)

    assert "np.nanmean(point_sigma[np.asarray(face" not in source
    assert "_face_nanmean_value(point_sigma, face)" in source


def test_v372_matplotlib_surface_face_vertices_avoid_index_arrays() -> None:
    source = inspect.getsource(Conductivity3DWidget._render_matplotlib_scene)

    assert "coords[np.asarray(face" not in source
    assert "_face_vertices_array(coords, valid_faces)" in source


def test_v369_matplotlib_surface_reuses_facecolor_buffers() -> None:
    source = inspect.getsource(Conductivity3DWidget._render_matplotlib_scene)

    assert "self._mpl3d_mesh_facecolors = colors.copy()" not in source
    assert "self._mpl3d_highlight_facecolors = highlight_colors.copy()" not in source
    assert "self._mpl3d_mesh_facecolors = colors" in source
    assert "self._mpl3d_highlight_facecolors = highlight_colors" in source


def test_v356_equipotential_update_preserves_float32_inputs() -> None:
    source = inspect.getsource(EquipotentialPlotWidget.update_reconstruction)

    assert "np.asarray(result.node_coords, dtype=np.float64)" not in source
    assert "np.asarray(result.conductivity, dtype=np.float64)" not in source
    assert "_display_float_array(result.node_coords)" in source
    assert "_display_float_array(result.conductivity)" in source

    values = np.array([1.0, 2.0], dtype=np.float32)
    display_values = _equipotential_display_float_array(values)

    assert display_values.dtype == np.dtype(np.float32)
    assert np.shares_memory(display_values, values)


def test_v364_equipotential_camera_preserves_float32_coordinate_axes() -> None:
    source = inspect.getsource(EquipotentialPlotWidget._apply_recon_aligned_camera)
    reset_source = inspect.getsource(
        EquipotentialPlotWidget._apply_recon_aligned_camera_from_bounds
    )

    assert "dtype=float" not in source
    assert "dtype=float" not in reset_source
    assert "_display_float_array(coords[:, 0])" in source
    assert "_display_float_array(coords[:, 1])" in source
    assert "_display_float_array(coords[:, 0])" in reset_source
    assert "_display_float_array(coords[:, 1])" in reset_source

    coords = np.arange(12, dtype=np.float32).reshape(4, 3)
    x_axis = _equipotential_display_float_array(coords[:, 0])
    y_axis = _equipotential_display_float_array(coords[:, 1])

    assert x_axis.dtype == np.dtype(np.float32)
    assert y_axis.dtype == np.dtype(np.float32)
    assert np.shares_memory(x_axis, coords)
    assert np.shares_memory(y_axis, coords)


def test_v357_hardware_reconstruction_preserves_float32_inputs() -> None:
    update_source = inspect.getsource(ReconstructionWidget.update_reconstruction)
    node_source = inspect.getsource(ReconstructionWidget._to_node_values)

    assert "np.asarray(result.node_coords, dtype=np.float64)" not in update_source
    assert "np.asarray(result.conductivity, dtype=np.float64)" not in update_source
    assert "_display_float_array(result.node_coords)" in update_source
    assert "_display_float_array(result.conductivity)" in update_source
    assert "np.zeros(n_nodes, dtype=np.float64)" not in node_source
    assert "np.result_type(sigma.dtype, np.float32)" in node_source

    values = np.array([1.0, 2.0], dtype=np.float32)
    display_values = _reconstruction_display_float_array(values)

    assert display_values.dtype == np.dtype(np.float32)
    assert np.shares_memory(display_values, values)


def test_v343_finite_scan_avoids_full_mask_on_common_3d_path(monkeypatch) -> None:
    anomaly_source = inspect.getsource(widget3d._cell_anomaly_mask)
    color_source = inspect.getsource(widget3d._conductivity_color_limits)
    helper_source = inspect.getsource(_finite_mask_or_none)

    assert "_finite_mask_or_none(values)" in anomaly_source
    assert "_finite_mask_or_none(values)" in color_source
    assert "np.isfinite(values)" not in anomaly_source
    assert "np.isfinite(values)" not in color_source
    assert "np.isfinite(chunk).all()" not in helper_source
    assert "np.isfinite(chunk, out=chunk_mask)" in helper_source
    assert "np.isfinite(tail, out=tail_mask)" in helper_source

    monkeypatch.setattr(widget3d, "_FINITE_SCAN_CHUNK_ITEMS", 4)
    values = np.linspace(0.0, 1.0, 10, dtype=np.float32)

    assert _finite_mask_or_none(values) is None

    values_with_nan = values.copy()
    values_with_nan[5] = np.nan
    finite_mask = _finite_mask_or_none(values_with_nan)

    assert finite_mask is not None
    assert finite_mask.dtype == np.dtype("bool")
    assert finite_mask.tolist() == [
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
    ]


def test_v447_color_limits_nonfinite_median_avoids_finite_subset_copy() -> None:
    color_source = inspect.getsource(widget3d._conductivity_color_limits)
    helper_source = inspect.getsource(widget3d._nanmedian_with_finite_mask)

    assert "values[finite_mask]" not in color_source
    assert "_nanmedian_with_finite_mask(values, finite_mask)" in color_source
    assert "np.logical_not(finite_mask, out=finite_mask)" in helper_source
    assert "np.copyto(work, np.nan, where=finite_mask)" in helper_source

    values = np.array([1.0, np.inf, 3.0, np.nan, 5.0], dtype=np.float32)
    finite_mask = np.array([True, False, True, False, True], dtype=bool)

    median = widget3d._nanmedian_with_finite_mask(values, finite_mask)

    assert median == 3.0
    assert finite_mask.tolist() == [False, True, False, True, False]
    assert widget3d._conductivity_color_limits(values) == (1.0, 5.0)


def test_v344_anomaly_mask_reuses_candidate_mask_and_finite_scan() -> None:
    anomaly_source = inspect.getsource(widget3d._cell_anomaly_mask)
    helper_source = inspect.getsource(_score_count_peak_above_floor)

    assert (
        "candidate_count, peak, mask = _score_count_peak_above_floor" in anomaly_source
    )
    assert "np.greater_equal(score, threshold, out=mask)" in anomaly_source
    assert "mask = np.greater_equal(score, threshold)" not in anomaly_source
    assert "np.isfinite(score)" not in helper_source
    assert "finite_mask" in helper_source
    assert "return_mask=True" in anomaly_source

    score = np.array([0.0, np.nan, 2.0, -np.inf], dtype=np.float32)
    finite_mask = np.array([True, False, True, False])

    count, peak, mask = _score_count_peak_above_floor(
        score,
        0.5,
        all_finite=False,
        finite_mask=finite_mask,
        return_mask=True,
    )

    assert count == 1
    assert peak == 2.0
    assert mask.tolist() == [False, False, True, False]


def test_v345_pyvista_geometry_buffers_use_views_not_flatten_copies() -> None:
    offscreen_source = inspect.getsource(
        Conductivity3DWidget._render_pyvista_offscreen_scene
    )
    embedded_source = inspect.getsource(Conductivity3DWidget._build_scene)
    electrode_source = inspect.getsource(Conductivity3DWidget._build_electrode_polydata)

    assert ".flatten()" not in offscreen_source
    assert ".flatten()" not in embedded_source
    assert ".flatten()" not in electrode_source
    assert "cell_array.ravel()" in offscreen_source
    assert "cell_array.ravel()" in embedded_source
    assert "face_buffer.ravel()" in electrode_source


def test_v509_pyvista_volume_highlight_prefers_existing_bool_mask() -> None:
    offscreen_source = inspect.getsource(
        Conductivity3DWidget._render_pyvista_offscreen_scene
    )
    embedded_source = inspect.getsource(Conductivity3DWidget._build_scene)
    helper_source = inspect.getsource(_extract_cells_from_mask)

    assert "np.where(inhom_mask)" not in offscreen_source
    assert "np.where(inhom_mask)" not in embedded_source
    assert "np.flatnonzero(inhom_mask)" not in offscreen_source
    assert "np.flatnonzero(inhom_mask)" not in embedded_source
    assert "_extract_cells_from_mask(grid, inhom_mask)" in offscreen_source
    assert "_extract_cells_from_mask(grid, inhom_mask)" in embedded_source
    assert "grid.extract_cells(mask_arr)" in helper_source
    assert "np.flatnonzero(mask_arr)" in helper_source

    class _MaskGrid:
        def __init__(self):
            self.calls: list[np.ndarray] = []

        def extract_cells(self, selector):
            self.calls.append(np.asarray(selector).copy())
            return "selected"

    grid = _MaskGrid()
    selected = _extract_cells_from_mask(grid, np.array([False, True, False]))

    assert selected == "selected"
    assert grid.calls[0].dtype == np.dtype(bool)


def test_v473_pyvista_volume_highlight_scans_mask_once() -> None:
    offscreen_source = inspect.getsource(
        Conductivity3DWidget._render_pyvista_offscreen_scene
    )
    embedded_source = inspect.getsource(Conductivity3DWidget._build_scene)

    assert "np.any(inhom_mask)" not in offscreen_source
    assert "np.any(inhom_mask)" not in embedded_source
    assert "if inhom_indices.size:" not in offscreen_source
    assert "if inhom_indices.size:" not in embedded_source
    assert "inhom_grid is not None and inhom_grid.n_cells > 0" in offscreen_source
    assert "inhom_grid is not None and inhom_grid.n_cells > 0" in embedded_source


def test_v347_equipotential_surface_avoids_flatten_and_finite_subset() -> None:
    render_source = inspect.getsource(EquipotentialPlotWidget._render_pyvista)
    warp_source = inspect.getsource(EquipotentialPlotWidget._compute_warp_factor)
    helper_source = inspect.getsource(_finite_min_max)

    assert ".flatten()" not in render_source
    assert "faces.ravel()" in render_source
    assert "node_values[np.isfinite" not in warp_source
    assert "_finite_min_max(node_values)" in warp_source
    assert "[np.isfinite" not in helper_source
    assert "np.isfinite(chunk, out=finite)" in helper_source
    assert "np.min(chunk, where=finite" in helper_source
    assert "np.max(chunk, where=finite" in helper_source

    values = np.array([np.nan, 1.0, np.inf, -2.0], dtype=np.float32)

    assert _finite_min_max(values) == (-2.0, 1.0)
    assert _finite_min_max(np.array([np.nan, np.inf])) is None


def test_v559_equipotential_pyvista_points_preserve_coord_dtype() -> None:
    render_source = inspect.getsource(EquipotentialPlotWidget._render_pyvista)

    assert "points = np.zeros((n_pts, 3), dtype=np.float64)" not in render_source
    assert "points = np.zeros((n_pts, 3), dtype=np.asarray(coords).dtype)" in (
        render_source
    )


def test_v332_offscreen_failure_cache_bypasses_retry_without_qt(monkeypatch) -> None:
    calls: list[str] = []

    class _DummyWidget:
        _display_mode = DISPLAY_MODE_VOLUME
        _last_vtk_disabled_reason = None
        _pending_render = None

        def _discard_actors(self) -> None:
            calls.append("discard")

        def _render_pyvista_offscreen_scene(self, _sigma, _coords, _cells) -> bool:
            calls.append("offscreen")
            return True

        def _render_matplotlib_scene(self, _sigma, _coords, _cells) -> None:
            calls.append("mpl3d")

    monkeypatch.delenv("EIT_APP_3D_PYVISTA_OFFSCREEN_NEGATIVE_CACHE", raising=False)
    widget3d._clear_pyvista_offscreen_failure_cache()
    try:
        widget3d._mark_pyvista_offscreen_failure("unit failure")
        Conductivity3DWidget._render_without_embedded_vtk(
            _DummyWidget(),
            np.ones(1, dtype=np.float32),
            np.zeros((4, 3), dtype=np.float32),
            np.array([[0, 1, 2, 3]], dtype=np.int32),
            reason="embedded disabled",
        )

        assert calls == ["discard", "mpl3d"]
    finally:
        widget3d._clear_pyvista_offscreen_failure_cache()


def test_v568_wslg_offscreen_skip_uses_reason_gate() -> None:
    render_source = inspect.getsource(Conductivity3DWidget._render_without_embedded_vtk)
    helper_source = inspect.getsource(
        widget3d._should_skip_pyvista_offscreen_for_reason
    )

    assert "_should_skip_pyvista_offscreen_for_reason" in render_source
    assert "WSLg embedded VTK requires" in helper_source
    assert "_PYVISTA_OFFSCREEN_WSLG_ENV" in inspect.getsource(
        widget3d._wslg_pyvista_offscreen_enabled
    )


def test_v337_channel_values_preserves_single_precision_display_dtype() -> None:
    real_values = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    complex_values = (real_values + 1j * real_values[::-1]).astype(np.complex64)

    real_channel = channel_values(real_values, REAL_CHANNEL)
    assert real_channel.dtype == np.dtype("float32")
    assert np.shares_memory(real_channel, real_values)

    for channel in (
        REAL_CHANNEL,
        IMAG_CHANNEL,
        MAGNITUDE_CHANNEL,
        PHASE_CHANNEL,
        COMPOSITE_CHANNEL,
    ):
        assert channel_values(complex_values, channel).dtype == np.dtype("float32")


def test_v338_complex_component_scan_avoids_full_finite_subset(monkeypatch) -> None:
    source = inspect.getsource(complex_channels.has_complex_component)
    helper_source = inspect.getsource(complex_channels._has_significant_imaginary)

    assert "imag[np.isfinite" not in source
    assert "imag[np.isfinite" not in helper_source
    assert "_COMPLEX_SCAN_CHUNK_ITEMS" in helper_source
    assert "finite = np.empty(chunk_items, dtype=bool)" in helper_source
    assert "abs_work = np.empty(chunk_items, dtype=imag.dtype)" in helper_source
    assert "np.isfinite(chunk, out=finite_chunk)" in helper_source
    assert "np.abs(chunk, out=abs_chunk)" in helper_source
    assert "np.nanmax" not in helper_source

    monkeypatch.setattr(complex_channels, "_COMPLEX_SCAN_CHUNK_ITEMS", 3)
    values = np.zeros(10, dtype=np.complex64)
    values[7] = 1.0 + 0.25j

    assert has_complex_component(values, tol=1.0e-3) is True
    assert has_complex_component(values.real.astype(np.float32), tol=1.0e-3) is False


def test_v339_composite_channel_values_are_chunked(monkeypatch) -> None:
    helper_source = inspect.getsource(complex_channels._composite_channel_values)

    assert "np.angle" not in helper_source
    assert "np.abs" not in helper_source
    assert "_COMPLEX_SCAN_CHUNK_ITEMS" in helper_source

    monkeypatch.setattr(complex_channels, "_COMPLEX_SCAN_CHUNK_ITEMS", 4)
    real = np.linspace(0.25, 1.25, 11, dtype=np.float32)
    imag = np.linspace(-0.5, 0.5, 11, dtype=np.float32)
    values = (real + 1j * imag).astype(np.complex64).reshape(11, 1)

    actual = channel_values(values, COMPOSITE_CHANNEL)
    expected = (np.abs(values) * (np.angle(values) / np.pi)).astype(np.float32)

    assert actual.dtype == np.dtype("float32")
    np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)
