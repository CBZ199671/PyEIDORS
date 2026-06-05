from __future__ import annotations

import inspect
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

import eit_app.controllers.dataset_generator_controller as dataset_controller  # noqa: E402
import eit_app.controllers.forward_solver_controller as fwd_controller  # noqa: E402
from eit_app.controllers.forward_solver_controller import (  # noqa: E402
    ForwardSolverRequest,
    _paint_shape,
    _forward_request_requires_complex_admittivity,
    _resolve_forward_runtime,
)
from eit_app.controllers import reconstruction_controller as rc  # noqa: E402
from eit_app.controllers.reconstruction_controller import (  # noqa: E402
    _resolve_reconstruction_runtime,
)
from eit_app.i18n import current_language, set_language  # noqa: E402
from eit_app.models.frame_model import FrameData  # noqa: E402
from eit_app.models.forward_model_config import (  # noqa: E402
    ForwardModelConfig,
    electrode_level_fractions_for_rings,
    max_electrode_height_ratio_for_rings,
)
from eit_app.models.simulation_state import InhomogeneitySpec  # noqa: E402
from pyeidors.electrodes.layout import (  # noqa: E402
    effective_pattern_layout_for_3d_mesh,
    effective_pattern_layout_for_zigzag_3d_mesh,
)
import eit_app.ui.conductivity_3d_widget as widget3d  # noqa: E402
from eit_app.ui.conductivity_3d_widget import (  # noqa: E402
    ANOMALY_MODE_ABSOLUTE,
    ANOMALY_MODE_NEGATIVE,
    ANOMALY_MODE_POSITIVE,
    Conductivity3DWidget,
    DISPLAY_MODE_POINTS,
    DISPLAY_MODE_VOLUME,
    SUPPORTED_3D_CELL_VERTEX_COUNTS,
    _cell_anomaly_mask,
    _cell_inhomogeneity_mask,
    _cell_mean_values,
    _apply_candidate_keep_mask,
    _candidate_indices_and_centers,
    _component_score_masses,
    _conductivity_color_limits,
    _boundary_faces,
    _valid_boundary_faces_and_sources,
    _display_cells_array,
    _display_coords_array,
    _display_sigma_array,
    _face_nanmean_value,
    _face_vertices_array,
    _face_vertices,
    _highlight_face_vertices_and_values,
    embedded_vtk_enabled,
    embedded_vtk_status,
    _point_cloud_sample_indices,
    _pyvista_feature_outline,
    _sample_background_indices,
    _sample_true_indices,
    _should_skip_pyvista_offscreen,
    _should_skip_pyvista_offscreen_for_reason,
    _spatially_coherent_anomaly_mask,
    _true_indices_from_mask,
)
from eit_app.ui.simulation.simulation_results_widget import (  # noqa: E402
    _ConductivityViewSlot,
)
from eit_app.ui.simulation.mesh_setup_panel import MeshSetupPanel  # noqa: E402
from pyeidors.geometry.mesh3d_generator import Cylinder3DMeshConfig  # noqa: E402


def _get_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_v104_3d_highlight_ignores_near_constant_absolute_sigma_noise() -> None:
    near_constant = np.array([1.0, 1.003, 0.997, 1.004, 0.998], dtype=np.float64)
    visible_anomaly = np.array([1.0, 1.0, 1.0, 1.12, 1.0], dtype=np.float64)

    assert not np.any(_cell_inhomogeneity_mask(near_constant))
    assert np.flatnonzero(_cell_inhomogeneity_mask(visible_anomaly)).tolist() == [3]


def test_v203_anomaly_mask_preserves_float32_score(monkeypatch) -> None:
    captured: list[np.dtype] = []

    def _fake_spatial(mask, score, cell_centers):
        assert cell_centers is None
        captured.append(np.asarray(score).dtype)
        return mask

    monkeypatch.setattr(widget3d, "_spatially_coherent_anomaly_mask", _fake_spatial)

    mask = _cell_anomaly_mask(
        np.array([1.0, 1.0, 1.0, 1.5, 1.0], dtype=np.float32),
        ANOMALY_MODE_POSITIVE,
    )

    assert captured == [np.dtype(np.float32)]
    assert np.flatnonzero(mask).tolist() == [3]


def test_v203_point_cloud_sampling_passes_float32_to_anomaly_mask(monkeypatch) -> None:
    seen: list[np.dtype] = []

    def _fake_anomaly_mask(
        cell_sigma,
        mode,
        *,
        cell_centers=None,
        prefer_central_region=False,
    ):
        del mode, cell_centers, prefer_central_region
        arr = np.asarray(cell_sigma)
        seen.append(arr.dtype)
        mask = np.zeros(arr.shape, dtype=bool)
        mask[2] = True
        return mask

    monkeypatch.setattr(widget3d, "_cell_anomaly_mask", _fake_anomaly_mask)
    centers = np.column_stack(
        [
            np.linspace(0.0, 1.0, 20, dtype=np.float32),
            np.zeros(20, dtype=np.float32),
            np.zeros(20, dtype=np.float32),
        ]
    )

    sample_idx = _point_cloud_sample_indices(
        np.ones(20, dtype=np.float32),
        centers,
        anomaly_mode=ANOMALY_MODE_POSITIVE,
        max_points=5,
    )

    assert seen == [np.dtype(np.float32)]
    assert 2 in sample_idx


def test_v207_point_cloud_display_arrays_sample_then_preserve_float32() -> None:
    source = inspect.getsource(widget3d._point_cloud_display_arrays)

    assert "center_values[sample]" not in source
    assert "sigma_values[sample]" not in source
    assert "np.take(center_values, sample, axis=0, out=display_centers)" in source
    assert "np.take(sigma_values, sample, out=display_sigma)" in source

    centers = np.column_stack(
        [
            np.linspace(0.0, 1.0, 20, dtype=np.float32),
            np.ones(20, dtype=np.float32),
            np.zeros(20, dtype=np.float32),
        ]
    )
    sigma = np.linspace(1.0, 2.0, 20, dtype=np.float32)
    sample_idx = np.array([0, 4, 19], dtype=np.int64)

    display_centers, display_sigma = widget3d._point_cloud_display_arrays(
        centers,
        sigma,
        sample_idx,
    )

    assert display_centers.dtype == np.dtype(np.float32)
    assert display_sigma.dtype == np.dtype(np.float32)
    assert display_centers.flags.c_contiguous
    assert display_sigma.flags.c_contiguous
    np.testing.assert_allclose(display_centers, centers[sample_idx])
    np.testing.assert_allclose(display_sigma, sigma[sample_idx])

    int_centers = np.arange(18, dtype=np.int32).reshape(6, 3)
    display_centers, _ = widget3d._point_cloud_display_arrays(
        int_centers,
        np.ones(6, dtype=np.float32),
        np.array([4, 1], dtype=np.int64),
    )
    assert display_centers.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(display_centers, int_centers[[4, 1]])


def test_v207_pyvista_point_cloud_actor_samples_before_casting() -> None:
    source = inspect.getsource(
        widget3d.Conductivity3DWidget._add_pyvista_point_cloud_actors
    )

    assert "np.asarray(centers, dtype=float)[sample_idx]" not in source
    assert "np.asarray(cell_sigma, dtype=float)[sample_idx]" not in source
    assert "_point_cloud_display_arrays" in source


def test_v204_spatial_anomaly_without_centers_avoids_flatnonzero(monkeypatch) -> None:
    mask = np.zeros(100, dtype=bool)
    mask[10:40] = True
    score = np.ones(100, dtype=np.float32)

    def _fail_flatnonzero(*_args, **_kwargs):
        raise AssertionError("cell_centers=None should not allocate candidate indices")

    monkeypatch.setattr(widget3d.np, "flatnonzero", _fail_flatnonzero)

    result = _spatially_coherent_anomaly_mask(mask, score, None)

    assert result is mask


def test_v209_spatial_anomaly_prepares_only_candidate_centers() -> None:
    source = inspect.getsource(widget3d._spatially_coherent_anomaly_mask)

    assert "np.asarray(cell_centers, dtype=np.float64)" not in source
    assert "centers[candidate_idx, :3]" not in source
    assert "_candidate_indices_and_centers(" in source


def test_v351_spatial_anomaly_reuses_nearest_distance_buffer() -> None:
    source = inspect.getsource(widget3d._spatially_coherent_anomaly_mask)
    helper_source = inspect.getsource(widget3d._nan_invalid_nearest_distances)

    assert "nearest[np.isfinite" not in source
    assert "nearest_valid" not in source
    assert "_nan_invalid_nearest_distances(nearest)" in source
    assert "np.isfinite(chunk, out=mask)" in helper_source
    assert "np.greater(chunk, 1.0e-12, out=mask)" in helper_source
    assert "np.logical_not(mask, out=mask)" in helper_source
    assert "nearest[nearest_valid] = np.nan" not in source
    assert "np.copyto(chunk, np.nan, where=mask)" in helper_source


def test_v366_spatial_anomaly_candidate_scores_preserve_float32(
    monkeypatch,
) -> None:
    source = inspect.getsource(widget3d._spatially_coherent_anomaly_mask)

    assert "np.asarray(score[candidate_idx], dtype=np.float64)" not in source
    assert "_component_score_masses(score_values, candidate_idx, components)" in source

    cluster = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.04, 0.00, 0.00],
            [0.00, 0.04, 0.00],
            [0.04, 0.04, 0.00],
            [0.02, 0.02, 0.04],
            [0.02, 0.02, -0.04],
        ],
        dtype=np.float32,
    )
    isolated = np.array(
        [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=np.float32,
    )
    centers = np.vstack([cluster, isolated])
    mask = np.ones(centers.shape[0], dtype=bool)
    score = np.ones(centers.shape[0], dtype=np.float32)
    score[cluster.shape[0] :] = 1.5
    seen_dtypes: list[np.dtype] = []
    original_component_score_masses = widget3d._component_score_masses

    def _capture_component_score_masses(score_values, candidate_idx, components):
        seen_dtypes.append(np.asarray(score_values).dtype)
        return original_component_score_masses(score_values, candidate_idx, components)

    monkeypatch.setattr(
        widget3d, "_component_score_masses", _capture_component_score_masses
    )

    result = _spatially_coherent_anomaly_mask(mask, score, centers)

    assert result.dtype == np.dtype(bool)
    assert seen_dtypes
    assert set(seen_dtypes) == {np.dtype(np.float32)}


def test_v379_spatial_anomaly_candidate_centers_preserve_float32(
    monkeypatch,
) -> None:
    source = inspect.getsource(widget3d._spatially_coherent_anomaly_mask)
    helper_source = inspect.getsource(widget3d._candidate_indices_and_centers)

    assert "np.ascontiguousarray(candidate_centers, dtype=np.float64)" not in source
    assert "np.asarray(candidate_centers, dtype=np.float64)" not in source
    assert "candidate_centers = np.empty((int(candidate_count), 3)" in helper_source
    assert "else np.dtype(np.float32)" in helper_source
    assert "nearest = np.asarray(distances[:, 1], dtype=np.float64)" not in source

    cluster = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.04, 0.00, 0.00],
            [0.00, 0.04, 0.00],
            [0.04, 0.04, 0.00],
            [0.02, 0.02, 0.04],
            [0.02, 0.02, -0.04],
        ],
        dtype=np.float32,
    )
    isolated = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=np.float32)
    centers = np.vstack([cluster, isolated])
    mask = np.ones(centers.shape[0], dtype=bool)
    score = np.ones(centers.shape[0], dtype=np.float32)

    class _FakeTree:
        seen_dtypes: list[np.dtype] = []

        def __init__(self, points):
            arr = np.asarray(points)
            self.points = arr
            self.seen_dtypes.append(arr.dtype)

        def query(self, points, k):
            del k
            n_points = np.asarray(points).shape[0]
            distances = np.empty((n_points, 2), dtype=np.float32)
            distances[:, 0] = 0.0
            distances[:, 1] = 0.04
            indices = np.zeros((n_points, 2), dtype=np.int64)
            return distances, indices

        def query_ball_point(self, points, radius):
            del radius
            n_points = np.asarray(points).shape[0]
            return [list(range(n_points)) for _ in range(n_points)]

    monkeypatch.setattr(widget3d, "cKDTree", _FakeTree, raising=False)
    monkeypatch.setitem(
        sys.modules, "scipy.spatial", SimpleNamespace(cKDTree=_FakeTree)
    )

    result = _spatially_coherent_anomaly_mask(mask, score, centers)

    assert result.dtype == np.dtype(bool)
    assert _FakeTree.seen_dtypes == [np.dtype(np.float32)]


def test_v390_spatial_anomaly_candidate_centers_direct_fill_without_flatnonzero() -> (
    None
):
    source = inspect.getsource(widget3d._spatially_coherent_anomaly_mask)
    helper_source = inspect.getsource(widget3d._candidate_indices_and_centers)

    assert "candidate_idx = np.flatnonzero(mask)" not in source
    assert "centers[candidate_idx, :3]" not in source
    assert (
        "candidate_idx = np.empty(int(candidate_count), dtype=np.int64)"
        in helper_source
    )
    assert "candidate_centers[out_idx] = centers_arr[center_idx, :3]" in helper_source

    centers = np.arange(18, dtype=np.float32).reshape(6, 3)
    mask = np.array([False, True, False, True, True, False], dtype=bool)

    candidate_idx, candidate_centers = _candidate_indices_and_centers(
        mask, centers, int(np.count_nonzero(mask))
    )

    np.testing.assert_array_equal(candidate_idx, [1, 3, 4])
    assert candidate_centers.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(candidate_centers, centers[[1, 3, 4], :3])


def test_v471_spatial_anomaly_candidate_center_finite_scan_uses_work_buffer() -> None:
    source = inspect.getsource(widget3d._spatially_coherent_anomaly_mask)
    helper_source = inspect.getsource(widget3d._all_finite_values)

    assert "np.isfinite(candidate_centers).all()" not in source
    assert "_all_finite_values(candidate_centers)" in source
    assert "np.isfinite(chunk, out=chunk_mask)" in helper_source

    assert widget3d._all_finite_values(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    assert not widget3d._all_finite_values(
        np.array([[1.0, np.nan, 3.0]], dtype=np.float32)
    )

    mask = np.ones(8, dtype=bool)
    score = np.arange(8, dtype=np.float32)
    centers = np.zeros((8, 3), dtype=np.float32)
    centers[3, 1] = np.nan

    result = _spatially_coherent_anomaly_mask(mask, score, centers)

    assert result is mask


def test_v391_spatial_anomaly_applies_keep_mask_without_index_subset() -> None:
    source = inspect.getsource(widget3d._spatially_coherent_anomaly_mask)
    helper_source = inspect.getsource(widget3d._apply_candidate_keep_mask)

    assert "coherent[candidate_idx[keep]]" not in source
    assert "_apply_candidate_keep_mask(coherent, candidate_idx, keep)" in source
    assert "candidate_idx[keep]" not in helper_source

    coherent = np.zeros(7, dtype=bool)
    candidate_idx = np.array([1, 3, 4, 6], dtype=np.int64)
    keep = np.array([False, True, True, False], dtype=bool)

    _apply_candidate_keep_mask(coherent, candidate_idx, keep)

    np.testing.assert_array_equal(
        coherent,
        np.array([False, False, False, True, True, False, False]),
    )


def test_v373_spatial_anomaly_component_masses_avoid_score_subsets() -> None:
    source = inspect.getsource(widget3d._spatially_coherent_anomaly_mask)
    helper_source = inspect.getsource(widget3d._component_score_masses)

    assert "candidate_scores = score_values[candidate_idx]" not in source
    assert "candidate_scores[component]" not in source
    assert "np.nansum" not in source
    assert "np.nansum" not in helper_source

    score_values = np.array([1.0, np.nan, 3.0, np.inf], dtype=np.float32)
    candidate_idx = np.array([3, 0, 1, 2], dtype=np.int64)
    components = [
        np.array([0, 2], dtype=np.int64),
        np.array([1, 3], dtype=np.int64),
    ]

    masses = _component_score_masses(score_values, candidate_idx, components)

    assert masses.dtype == np.dtype(np.float64)
    assert np.isinf(masses[0])
    assert masses[1] == pytest.approx(4.0)


def test_v209_spatial_anomaly_accepts_float32_centers() -> None:
    mask = np.zeros(16, dtype=bool)
    mask[:8] = True
    score = np.linspace(0.1, 1.0, 16, dtype=np.float32)
    centers = np.column_stack(
        [
            np.linspace(0.0, 0.07, 16, dtype=np.float32),
            np.zeros(16, dtype=np.float32),
            np.zeros(16, dtype=np.float32),
        ]
    )

    result = _spatially_coherent_anomaly_mask(mask, score, centers)

    assert result.dtype == np.dtype(bool)
    assert result.shape == mask.shape


def test_v205_score_peak_stats_excludes_nonfinite_scores() -> None:
    score = np.array([np.nan, -1.0, 0.2, np.inf, 0.5], dtype=np.float32)

    count, peak = widget3d._score_count_peak_above_floor(
        score,
        0.1,
        all_finite=False,
    )

    assert count == 2
    assert peak == pytest.approx(0.5)


def test_v205_anomaly_threshold_sparse_path_uses_where_peak(monkeypatch) -> None:
    def _fail_nanmax(*_args, **_kwargs):
        raise AssertionError("sparse anomaly path should use mask+where peak stats")

    monkeypatch.setattr(widget3d.np, "nanmax", _fail_nanmax)
    values = np.ones(256, dtype=np.float32)
    values[17] = 1.7

    mask = _cell_anomaly_mask(values, ANOMALY_MODE_POSITIVE)

    assert np.flatnonzero(mask).tolist() == [17]


def test_v205_cell_anomaly_mask_has_no_eager_finite_score_subset() -> None:
    source = inspect.getsource(widget3d._cell_anomaly_mask)

    assert "score[np.isfinite(score)]" not in source
    assert "finite_scores =" not in source
    assert "np.nanmax(finite_scores)" not in source


def test_v433_cell_anomaly_crowded_percentile_reuses_finite_mask(
    monkeypatch,
) -> None:
    anomaly_source = inspect.getsource(widget3d._cell_anomaly_mask)
    helper_source = inspect.getsource(widget3d._nanpercentile_with_finite_mask)

    assert "score[finite_values]" not in anomaly_source
    assert "_nanpercentile_with_finite_mask(" in anomaly_source
    assert "np.logical_not(finite_mask, out=invalid_mask)" in helper_source
    assert "np.copyto(score, np.nan, where=invalid_mask)" in helper_source

    original_nanpercentile = widget3d.np.nanpercentile
    percentile_inputs: list[tuple[tuple[int, ...], int]] = []

    def _record_nanpercentile(values, percentile, *args, **kwargs):
        arr = np.asarray(values)
        percentile_inputs.append((arr.shape, int(np.count_nonzero(np.isnan(arr)))))
        return original_nanpercentile(values, percentile, *args, **kwargs)

    monkeypatch.setattr(widget3d.np, "nanpercentile", _record_nanpercentile)
    values = np.ones(100, dtype=np.float32)
    values[:20] = 2.0
    values[-1] = np.nan

    mask = _cell_anomaly_mask(values, ANOMALY_MODE_POSITIVE)

    assert percentile_inputs[-1] == ((100,), 1)
    assert np.flatnonzero(mask).tolist() == list(range(20))


def test_v340_cell_anomaly_mask_reuses_score_buffer_for_absolute_mode() -> None:
    source = inspect.getsource(widget3d._cell_anomaly_mask)

    assert "score = np.abs(" not in source
    assert "score = -residual" not in source
    assert "np.abs(score, out=score)" in source
    assert "np.negative(score, out=score)" in source

    values = np.array([1.0, 1.18, 0.82, 1.01, 0.99], dtype=np.float32)
    absolute = _cell_anomaly_mask(values, ANOMALY_MODE_ABSOLUTE)

    assert np.flatnonzero(absolute).tolist() == [1, 2]


def test_v367_cell_anomaly_signed_mad_reuses_score_buffer() -> None:
    source = inspect.getsource(widget3d._cell_anomaly_mask)

    assert "np.nanmedian(np.abs(score))" not in source
    assert "np.subtract(values, median, out=score)" in source
    assert "np.negative(score, out=score)" in source

    values = np.array([1.0, 1.18, 0.82, 1.01, 0.99], dtype=np.float32)
    positive = _cell_anomaly_mask(values, ANOMALY_MODE_POSITIVE)
    negative = _cell_anomaly_mask(values, ANOMALY_MODE_NEGATIVE)
    absolute = _cell_anomaly_mask(values, ANOMALY_MODE_ABSOLUTE)

    assert np.flatnonzero(positive).tolist() == [1]
    assert np.flatnonzero(negative).tolist() == [2]
    assert np.flatnonzero(absolute).tolist() == [1, 2]


def test_v201_cell_mean_values_streams_vertex_slices(monkeypatch) -> None:
    values = np.array([1.0, 3.0, 5.0, 7.0, 9.0], dtype=np.float32)
    cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    original_take = np.take
    calls: list[tuple[tuple[int, ...], bool]] = []

    def _count_take(a, indices, axis=None, out=None, **kwargs):
        calls.append((np.asarray(indices).shape, out is not None))
        return original_take(a, indices, axis=axis, out=out, **kwargs)

    monkeypatch.setattr(widget3d.np, "take", _count_take)

    means = _cell_mean_values(values, cells)

    assert means.dtype == np.float32
    assert calls == [((2,), True), ((2,), True), ((2,), True), ((2,), True)]
    np.testing.assert_allclose(means, np.array([4.0, 6.0], dtype=np.float32))


def test_v363_3d_display_integer_fallbacks_use_float32() -> None:
    assert "dtype=np.float64" not in inspect.getsource(widget3d._display_float_values)
    assert "dtype=np.float64" not in inspect.getsource(widget3d._cell_center_sigma)
    assert "dtype=np.float64" not in inspect.getsource(widget3d._cell_mean_values)

    display = widget3d._display_float_values(np.array([1, 2], dtype=np.int32))
    assert display.dtype == np.dtype(np.float32)

    cell_sigma, scalar_mode = widget3d._cell_center_sigma(
        np.array([1, 3], dtype=np.int32),
        np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int32),
    )
    assert scalar_mode == "cell"
    assert cell_sigma.dtype == np.dtype(np.float32)

    means = widget3d._cell_mean_values(
        np.array([1, 3, 5], dtype=np.int32),
        np.array([[0, 1], [1, 2]], dtype=np.int32),
    )
    assert means.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(means, np.array([2.0, 4.0], dtype=np.float32))


def test_v208_cell_center_sigma_point_values_use_streamed_means(monkeypatch) -> None:
    calls: list[tuple[np.dtype, np.dtype]] = []

    def _fake_cell_mean_values(point_values, cells):
        calls.append((np.asarray(point_values).dtype, np.asarray(cells).dtype))
        return np.array([4.0, 6.0], dtype=np.float32)

    monkeypatch.setattr(widget3d, "_cell_mean_values", _fake_cell_mean_values)
    values = np.array([1.0, 3.0, 5.0, 7.0, 9.0], dtype=np.float32)
    cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)

    cell_sigma, scalar_mode = widget3d._cell_center_sigma(values, cells)

    assert scalar_mode == "point"
    assert calls == [(np.dtype(np.float32), np.dtype(np.int32))]
    assert cell_sigma.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(cell_sigma, np.array([4.0, 6.0], dtype=np.float32))


def test_v208_cell_center_sigma_has_no_expanded_point_value_indexing() -> None:
    source = inspect.getsource(widget3d._cell_center_sigma)

    assert "values[cells].mean(axis=1)" not in source
    assert "_cell_mean_values" in source


def test_v371_face_nanmean_value_avoids_indexed_subset() -> None:
    source = inspect.getsource(widget3d._face_nanmean_value)

    assert "point_values[" in source
    assert "np.asarray(face" not in source
    assert "np.nanmean" not in source

    values = np.array([1.0, np.nan, 3.0, np.inf], dtype=np.float32)

    assert _face_nanmean_value(values, (0, 1, 2)) == pytest.approx(2.0)
    assert np.isinf(_face_nanmean_value(values, (0, 3)))
    assert np.isnan(_face_nanmean_value(values, (1,)))


def test_v372_face_vertices_direct_fills_without_index_array() -> None:
    source = inspect.getsource(widget3d._face_vertices)

    assert "np.asarray(face" not in source
    assert "coords_arr[" in source

    coords = np.arange(15, dtype=np.float32).reshape(5, 3)
    vertices = _face_vertices(coords, (0, 2, 4))

    assert vertices.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(vertices, coords[[0, 2, 4]])


def test_v386_face_vertices_array_direct_fills_batch_without_per_face_arrays() -> None:
    helper_source = inspect.getsource(widget3d._face_vertices_array)

    assert "vertices = np.empty((len(faces), vertices_per_face, 3)" in helper_source

    coords = np.arange(18, dtype=np.float32).reshape(6, 3)
    faces = [(0, 2, 4), (1, 3, 5)]

    vertices = _face_vertices_array(coords, faces)

    assert vertices.dtype == np.dtype(np.float32)
    assert vertices.shape == (2, 3, 3)
    np.testing.assert_array_equal(vertices[0], coords[[0, 2, 4]])
    np.testing.assert_array_equal(vertices[1], coords[[1, 3, 5]])


def test_v387_highlight_face_vertices_direct_fills_values_without_lists() -> None:
    helper_source = inspect.getsource(widget3d._highlight_face_vertices_and_values)

    assert "active_count = int(np.count_nonzero(active_mask))" in helper_source
    assert "vertices = np.empty((max_faces, vertices_per_face, 3)" in helper_source
    assert "values = np.empty(max_faces, dtype=sigma_arr.dtype)" in helper_source

    coords = np.arange(15, dtype=np.float32).reshape(5, 3)
    cells = np.array(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ],
        dtype=np.int32,
    )
    sigma = np.array([1.25, 2.5], dtype=np.float32)

    vertices, values = _highlight_face_vertices_and_values(
        coords, cells, sigma, np.array([True, True], dtype=bool)
    )

    assert vertices.dtype == np.dtype(np.float32)
    assert values.dtype == np.dtype(np.float32)
    assert vertices.shape == (8, 3, 3)
    np.testing.assert_array_equal(values, np.repeat(sigma, 4))


def test_v387_highlight_face_vertices_skips_invalid_faces() -> None:
    coords = np.arange(12, dtype=np.float32).reshape(4, 3)
    cells = np.array([[0, 1, 2, 99]], dtype=np.int32)
    sigma = np.array([3.0], dtype=np.float32)

    vertices, values = _highlight_face_vertices_and_values(
        coords, cells, sigma, np.array([True], dtype=bool)
    )

    assert vertices.shape == (1, 3, 3)
    np.testing.assert_array_equal(vertices[0], coords[[0, 1, 2]])
    np.testing.assert_array_equal(values, sigma)


def test_v384_boundary_faces_direct_fills_sources_without_kept_list() -> None:
    source = inspect.getsource(widget3d._boundary_faces)

    assert "kept = [" not in source
    assert "[face for face" not in source
    assert "[idx for _face" not in source
    assert "kept_count = sum(" in source
    assert "source_cells = np.empty(kept_count, dtype=np.int64)" in source

    cells = np.array(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ],
        dtype=np.int32,
    )

    faces, sources = _boundary_faces(cells)

    assert len(faces) == 6
    assert sources.dtype == np.dtype(np.int64)
    internal_face = (1, 2, 3)
    assert internal_face not in {tuple(sorted(face)) for face in faces}
    np.testing.assert_array_equal(np.bincount(sources, minlength=2), [3, 3])


def test_v385_valid_boundary_faces_reuses_all_valid_source_array() -> None:
    helper_source = inspect.getsource(widget3d._valid_boundary_faces_and_sources)

    assert "valid_face_payload" not in helper_source
    assert "return faces, np.asarray(source_cells, dtype=np.intp)" in helper_source

    faces = [(0, 1, 2), (0, 2, 3)]
    source_cells = np.array([0, 1], dtype=np.intp)

    valid_faces, valid_sources = _valid_boundary_faces_and_sources(
        faces, source_cells, n_coords=4
    )

    assert valid_faces is faces
    assert valid_sources is source_cells


def test_v385_valid_boundary_faces_direct_fills_invalid_fallback() -> None:
    faces = [(0, 1, 2), (0, 2, 99), (1, 2, 3)]
    source_cells = np.array([0, 1, 2], dtype=np.int64)

    valid_faces, valid_sources = _valid_boundary_faces_and_sources(
        faces, source_cells, n_coords=4
    )

    assert valid_faces == [(0, 1, 2), (1, 2, 3)]
    assert valid_sources.dtype == np.dtype(np.intp)
    np.testing.assert_array_equal(valid_sources, [0, 2])


def test_v119_3d_anomaly_mask_separates_positive_negative_absolute_modes() -> None:
    values = np.array([1.0, 1.18, 0.82, 1.01, 0.99], dtype=np.float64)

    positive = _cell_anomaly_mask(values, ANOMALY_MODE_POSITIVE)
    negative = _cell_anomaly_mask(values, ANOMALY_MODE_NEGATIVE)
    absolute = _cell_anomaly_mask(values, ANOMALY_MODE_ABSOLUTE)

    assert np.flatnonzero(positive).tolist() == [1]
    assert np.flatnonzero(negative).tolist() == [2]
    assert np.flatnonzero(absolute).tolist() == [1, 2]
    assert np.array_equal(_cell_inhomogeneity_mask(values), absolute)


def test_v119_3d_anomaly_mask_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown anomaly mode"):
        _cell_anomaly_mask(np.array([1.0, 2.0]), "signed")


def test_v120_3d_anomaly_mask_suppresses_diffuse_low_amplitude_artifacts() -> None:
    values = np.ones(100, dtype=np.float64)
    values[10:15] = 1.60
    values[50:80] = 1.30

    mask = _cell_anomaly_mask(values, ANOMALY_MODE_POSITIVE)

    assert np.flatnonzero(mask).tolist() == list(range(10, 15))


def test_v120_3d_anomaly_mask_keeps_strongest_spatially_coherent_blob() -> None:
    cluster = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.04, 0.00, 0.00],
            [0.00, 0.04, 0.00],
            [0.04, 0.04, 0.00],
            [0.02, 0.02, 0.04],
            [0.02, 0.02, -0.04],
        ],
        dtype=np.float64,
    )
    isolated = np.array(
        [
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=np.float64,
    )
    background = np.array(
        [[1.0 + idx, 1.0, 1.0] for idx in range(21)],
        dtype=np.float64,
    )
    centers = np.vstack([cluster, isolated, background])
    values = np.ones(centers.shape[0], dtype=np.float64)
    values[: cluster.shape[0]] = 1.70
    values[cluster.shape[0] : cluster.shape[0] + isolated.shape[0]] = 1.90

    mask = _cell_anomaly_mask(
        values,
        ANOMALY_MODE_POSITIVE,
        cell_centers=centers,
    )

    assert np.flatnonzero(mask).tolist() == list(range(cluster.shape[0]))


def test_v619_phase_region_focus_prefers_central_blob_over_boundary_spikes() -> None:
    central = np.array(
        [
            [x, y, z]
            for x in (-0.03, 0.0, 0.03)
            for y in (-0.03, 0.0, 0.03)
            for z in (-0.03, 0.0, 0.03)
        ],
        dtype=np.float64,
    )
    edge_clusters = np.array(
        [
            [sign, offset, z]
            for sign in (-1.0, 1.0)
            for offset in (-0.04, 0.04)
            for z in (-0.04, 0.04)
        ]
        + [
            [offset, sign, z]
            for sign in (-1.0, 1.0)
            for offset in (-0.04, 0.04)
            for z in (-0.04, 0.04)
        ],
        dtype=np.float64,
    )
    background = np.array(
        [
            [x, y, z]
            for x in np.linspace(-0.72, 0.72, 5)
            for y in np.linspace(-0.72, 0.72, 5)
            for z in (-0.08, 0.08)
            if abs(x) > 0.18 or abs(y) > 0.18
        ],
        dtype=np.float64,
    )
    centers = np.vstack([central, edge_clusters, background])
    values = np.full(centers.shape[0], 63.0, dtype=np.float64)
    values[: central.shape[0]] = 60.7
    edge_start = central.shape[0]
    edge_stop = edge_start + edge_clusters.shape[0]
    values[edge_start:edge_stop:2] = 56.0
    values[edge_start + 1 : edge_stop : 2] = 70.0

    expected = list(range(central.shape[0]))
    for mode in (ANOMALY_MODE_POSITIVE, ANOMALY_MODE_NEGATIVE, ANOMALY_MODE_ABSOLUTE):
        focused = _cell_anomaly_mask(
            values,
            mode,
            cell_centers=centers,
            prefer_central_region=True,
        )
        assert np.flatnonzero(focused).tolist() == expected


def test_v107_3d_color_limits_do_not_amplify_near_constant_sigma_noise() -> None:
    near_constant = np.array([1.0, 1.003, 0.997, 1.004, 0.998], dtype=np.float64)
    visible_anomaly = np.array([1.0, 1.0, 1.0, 1.12, 1.0], dtype=np.float64)

    sigma_min, sigma_max = _conductivity_color_limits(near_constant)
    assert sigma_min == pytest.approx(0.98)
    assert sigma_max == pytest.approx(1.02)
    assert _conductivity_color_limits(visible_anomaly) == pytest.approx((1.0, 1.12))


def test_v206_color_limits_all_finite_path_uses_plain_reductions(monkeypatch) -> None:
    def _fail_nan_reduction(*_args, **_kwargs):
        raise AssertionError("all-finite color limits should avoid nan reductions")

    monkeypatch.setattr(widget3d.np, "nanmin", _fail_nan_reduction)
    monkeypatch.setattr(widget3d.np, "nanmax", _fail_nan_reduction)
    values = np.array([1.0, 1.1, 1.4], dtype=np.float32)

    assert _conductivity_color_limits(values) == pytest.approx((1.0, 1.4))


def test_v206_color_limits_preserve_nonfinite_exclusion_semantics() -> None:
    values = np.array([np.nan, np.inf, 0.9, 1.1], dtype=np.float32)

    assert _conductivity_color_limits(values) == pytest.approx((0.9, 1.1))


def test_v206_color_limits_has_no_eager_float64_finite_subset() -> None:
    source = inspect.getsource(widget3d._conductivity_color_limits)

    assert "dtype=np.float64" not in source
    assert "np.nanmin" not in source
    assert "np.nanmax" not in source
    assert "values[np.isfinite(values)]" not in source


def _tetra_payload() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    cells = np.array([[0, 1, 2, 3]], dtype=np.int64)
    sigma = np.array([1.25], dtype=float)
    return sigma, coords, cells


def _hex_payload() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    sigma = np.array([1.75], dtype=float)
    return sigma, coords, cells


def _inhomogeneous_tetra_payload() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    sigma = np.array([1.0, 2.0], dtype=float)
    return sigma, coords, cells


def _frame(index: int) -> FrameData:
    return FrameData(
        real=np.array([1.0, 2.0, 3.0], dtype=float),
        imag=np.zeros(3, dtype=float),
        timestamp=0.0,
        frame_index=index,
    )


def test_supported_3d_cell_types_cover_tetra_and_hex():
    assert {4, 8}.issubset(SUPPORTED_3D_CELL_VERTEX_COUNTS)


def test_v191_display_payload_arrays_preserve_float32_and_int32() -> None:
    coords = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    cells = np.asarray([[0, 1]], dtype=np.int32)
    sigma = np.asarray([1.0], dtype=np.float32)

    assert _display_coords_array(coords) is coords
    assert _display_cells_array(cells) is cells
    assert _display_sigma_array(sigma) is sigma


def test_v567_display_payload_arrays_downcast_float64_for_3d_view() -> None:
    coords = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.5, 0.25]],
        dtype=np.float64,
    )
    sigma = np.asarray([1.0, 1.25], dtype=np.float64)
    complex_sigma = np.asarray([1.0 + 0.5j, 1.25 - 0.25j], dtype=np.complex128)

    display_coords = _display_coords_array(coords)
    display_sigma = _display_sigma_array(sigma)
    display_complex_sigma = _display_sigma_array(complex_sigma)

    assert display_coords.dtype == np.dtype(np.float32)
    assert display_sigma.dtype == np.dtype(np.float32)
    assert display_complex_sigma.dtype == np.dtype(np.float32)
    assert display_coords is not coords
    assert display_sigma is not sigma
    np.testing.assert_allclose(display_coords, coords.astype(np.float32))
    np.testing.assert_allclose(display_sigma, sigma.astype(np.float32))
    np.testing.assert_allclose(
        display_complex_sigma, complex_sigma.real.astype(np.float32)
    )


def test_v191_update_image_passes_float32_int32_without_entry_copy(monkeypatch):
    _get_app()
    monkeypatch.setattr(widget3d, "embedded_vtk_status", lambda: (False, "unit"))
    widget = Conductivity3DWidget("Conductivity")
    captured: dict[str, np.ndarray] = {}

    def fake_render(sigma, coords, cells, *, reason):
        captured["sigma"] = sigma
        captured["coords"] = coords
        captured["cells"] = cells
        captured["reason"] = reason

    monkeypatch.setattr(widget, "_render_without_embedded_vtk", fake_render)
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    cells = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
    sigma = np.asarray([1.0], dtype=np.float32)

    widget.update_image(sigma, coords, cells, title="Truth")

    assert captured["sigma"] is sigma
    assert captured["coords"] is coords
    assert captured["cells"] is cells
    assert captured["reason"] == "unit"
    widget.close()


def test_zigzag_3d_mesh_uses_total_electrode_pattern_layout():
    assert effective_pattern_layout_for_3d_mesh(
        mesh_tdim=3,
        n_elec=8,
        n_rings=2,
        electrode_layout="ring_major",
    ) == (8, 2)
    assert effective_pattern_layout_for_zigzag_3d_mesh(
        mesh_tdim=3,
        n_elec=8,
        n_rings=2,
    ) == (16, 1)
    assert effective_pattern_layout_for_zigzag_3d_mesh(
        mesh_tdim=2,
        n_elec=8,
        n_rings=2,
    ) == (8, 2)


def test_paint_circle_is_area_in_2d_even_with_2d_centers():
    centers = np.array([[0.0, 0.0], [0.24, 0.0], [0.3, 0.0]], dtype=float)
    values = np.ones(centers.shape[0], dtype=float)
    spec = InhomogeneitySpec(shape="circle", size_x=0.25, conductivity=2.0)

    _paint_shape(values, centers, spec, mesh_dimension=2)

    assert values.tolist() == [2.0, 2.0, 1.0]


def test_paint_circle_is_sphere_in_3d_not_vertical_cylinder():
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.3],
            [0.1, 0.0, 0.1],
        ],
        dtype=float,
    )
    values = np.ones(centers.shape[0], dtype=float)
    spec = InhomogeneitySpec(shape="circle", size_x=0.2, conductivity=2.0)

    _paint_shape(values, centers, spec, mesh_dimension=3)

    assert values.tolist() == [2.0, 1.0, 2.0]


def test_v109_paint_3d_sphere_respects_z_radius_when_sizes_disagree():
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.07],
            [0.04, 0.0, 0.0],
        ],
        dtype=float,
    )
    values = np.ones(centers.shape[0], dtype=float)
    spec = InhomogeneitySpec(
        shape="circle",
        size_x=0.2,
        size_y=0.2,
        size_z=0.05,
        conductivity=2.0,
    )

    _paint_shape(values, centers, spec, mesh_dimension=3)

    assert values.tolist() == [2.0, 1.0, 2.0]


def test_v101_paint_3d_sphere_uses_volume_fraction_for_coarse_hex_layers():
    x0, x1 = -0.04, 0.04
    y0, y1 = -0.04, 0.04
    z_levels = np.array([-0.08, -0.056, -0.024, 0.0, 0.024, 0.056, 0.08])
    cell_vertices = []
    for z0, z1 in zip(z_levels[:-1], z_levels[1:], strict=True):
        cell_vertices.append(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ]
        )
    cell_vertices_arr = np.asarray(cell_vertices, dtype=float)
    centers = cell_vertices_arr.mean(axis=1)
    values = np.ones(centers.shape[0], dtype=float)
    spec = InhomogeneitySpec(shape="circle", size_x=0.063, conductivity=2.0)

    _paint_shape(
        values,
        centers,
        spec,
        mesh_dimension=3,
        cell_vertices=cell_vertices_arr,
    )

    assert np.all(values > 1.0)
    assert values[0] == pytest.approx(values[-1])
    assert values[1] == pytest.approx(values[-2])
    assert values[2] == pytest.approx(values[-3])
    assert values[0] < values[1]
    assert values[0] < values[2]
    assert values[2] > 1.5


def test_v285_hex_volume_sample_weights_are_precomputed() -> None:
    source = inspect.getsource(fwd_controller._cell_volume_sample_points)
    helper_source = inspect.getsource(fwd_controller._build_hex_sample_weights)
    electrode_polydata_source = inspect.getsource(
        Conductivity3DWidget._build_electrode_polydata
    )

    assert "np.column_stack" not in source
    assert "_HEX_SAMPLE_WEIGHTS" in source
    assert "np.column_stack" not in helper_source
    assert "np.column_stack" not in electrode_polydata_source
    assert fwd_controller._HEX_SAMPLE_WEIGHTS.shape == (64, 8)
    np.testing.assert_allclose(
        fwd_controller._HEX_SAMPLE_WEIGHTS.sum(axis=1),
        np.ones(64, dtype=float),
    )

    vertices = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        ],
        dtype=float,
    )
    points = fwd_controller._cell_volume_sample_points(vertices)
    assert points is not None
    np.testing.assert_allclose(points[0], fwd_controller._HEX_SAMPLE_GRID)


def test_paint_3d_volume_fraction_streams_vertices_from_connectivity():
    x0, x1 = -0.04, 0.04
    y0, y1 = -0.04, 0.04
    z_levels = np.array([-0.08, -0.056, -0.024, 0.0, 0.024, 0.056, 0.08])
    cell_vertices = []
    for z0, z1 in zip(z_levels[:-1], z_levels[1:], strict=True):
        cell_vertices.append(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ]
        )
    cell_vertices_arr = np.asarray(cell_vertices, dtype=float)
    centers = cell_vertices_arr.mean(axis=1)
    full_values = np.ones(centers.shape[0], dtype=float)
    streamed_values = np.ones_like(full_values)
    spec = InhomogeneitySpec(shape="circle", size_x=0.063, conductivity=2.0)

    _paint_shape(
        full_values,
        centers,
        spec,
        mesh_dimension=3,
        cell_vertices=cell_vertices_arr,
    )
    node_coords = cell_vertices_arr.reshape(-1, 3)
    cell_connectivity = np.arange(node_coords.shape[0], dtype=np.int32).reshape(-1, 8)
    _paint_shape(
        streamed_values,
        centers,
        spec,
        mesh_dimension=3,
        node_coords=node_coords,
        cell_connectivity=cell_connectivity,
    )

    np.testing.assert_allclose(streamed_values, full_values)
    streaming_source = inspect.getsource(
        fwd_controller._apply_volume_fraction_streaming
    )
    assert "np.take" in streaming_source
    assert "_cell_volume_sample_points" not in streaming_source
    assert "vertices = np.take" not in streaming_source
    assert "coords[cells" not in streaming_source
    assert "node_coords[cell_connectivity]" not in inspect.getsource(
        fwd_controller.execute_forward_request
    )
    assert "node_coords[cell_connectivity]" not in inspect.getsource(
        dataset_controller._DatasetGeneratorWorker.run
    )


def test_paint_ellipsoid_and_box_use_z_extent_in_3d():
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.16],
            [0.19, 0.0, 0.0],
        ],
        dtype=float,
    )

    ellipsoid_values = np.ones(centers.shape[0], dtype=float)
    ellipsoid = InhomogeneitySpec(
        shape="ellipse",
        size_x=0.2,
        size_y=0.2,
        size_z=0.1,
        conductivity=2.0,
    )
    _paint_shape(ellipsoid_values, centers, ellipsoid, mesh_dimension=3)
    assert ellipsoid_values.tolist() == [2.0, 1.0, 2.0]

    box_values = np.ones(centers.shape[0], dtype=float)
    box = InhomogeneitySpec(
        shape="rectangle",
        size_x=0.2,
        size_y=0.2,
        size_z=0.1,
        conductivity=3.0,
    )
    _paint_shape(box_values, centers, box, mesh_dimension=3)
    assert box_values.tolist() == [3.0, 1.0, 3.0]


def test_single_step_cached_promotes_3d_line_current_density_to_total_current():
    request = rc.ReconstructionRequest(
        reference_frame=_frame(0),
        target_frame=_frame(1),
        mesh_dimension=3,
        mesh_refinement=0.1,
        metadata={
            "mesh_dimension": 3,
            "n_elec": 8,
            "n_rings": 2,
            "drive_mode": "line_current_density",
            "drive_value": 1.0,
            "mesh_size": 0.1,
            "radius": 0.18,
            "height": 0.16,
            "mesh_family": "hex",
        },
    )

    runtime = rc._prepare_single_step_cached_runtime(request)

    assert runtime.meta["drive_mode"] == "total_current"
    assert "line_current_density" not in runtime.cache_key


def test_single_step_cached_uses_measurement_space_when_operator_shape_matches(
    monkeypatch: pytest.MonkeyPatch,
):
    reference = FrameData(
        real=np.array([1.0, 2.0, 3.0], dtype=float),
        imag=np.zeros(3, dtype=float),
        timestamp=0.0,
        frame_index=0,
    )
    target = FrameData(
        real=np.array([1.5, 2.5, 4.0], dtype=float),
        imag=np.zeros(3, dtype=float),
        timestamp=0.0,
        frame_index=1,
    )
    expected_dv = target.real - reference.real
    delta_sigma = np.array([0.25, -0.5], dtype=float)
    base_meas = np.array([10.0, 20.0, 30.0], dtype=float)
    pred_diff = np.array([0.1, 0.2, 0.3], dtype=float)
    calls = {"measurement": 0, "parameter": 0}

    class _StubForwardModel:
        def fwd_solve(self, image):
            assert image.elem_data is not None
            return SimpleNamespace(meas=base_meas + pred_diff), None

    ctx = {
        "mesh": object(),
        "display_node_coords": np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=float,
        ),
        "display_cell_connectivity": np.array([[0, 1, 2]], dtype=int),
        "operator_bundle": {
            "mode": "strict",
            "strict_solver_backend_effective": "dense-param",
            "A": np.eye(expected_dv.size, dtype=float),
            "Jt": np.ones((delta_sigma.size, expected_dv.size), dtype=float),
            "inv_reg_diag": np.ones(delta_sigma.size, dtype=float),
        },
        "sigma_bg": np.ones_like(delta_sigma),
        "fwd_model": _StubForwardModel(),
        "base_meas": base_meas,
        "cache_build_seconds": {},
        "cache_miss_reasons": {},
        "cache_manager": None,
    }

    def _measurement_space_delta(*, operator_bundle, rhs):
        calls["measurement"] += 1
        assert operator_bundle is ctx["operator_bundle"]
        assert np.allclose(rhs, expected_dv)
        return delta_sigma

    def _solve_linear_from_bundle(_operator_bundle, _rhs):
        calls["parameter"] += 1
        raise AssertionError("parameter-space solve should not be used")

    fake_diff_runner = SimpleNamespace(
        STRICT_SOLVER_BACKEND_MEASUREMENT="measurement-exact",
        _calibrate_step_size=lambda **_kwargs: 1.0,
        _measurement_space_delta=_measurement_space_delta,
        _solve_linear_from_bundle=_solve_linear_from_bundle,
        build_shared_context=lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        rc, "_load_gn_difference_runner_module", lambda: fake_diff_runner
    )
    monkeypatch.setattr(
        rc,
        "_ensure_single_step_cached_context",
        lambda _runtime, *, emit, build_shared_context: ctx,
    )

    result = rc._run_single_step_cached_request(
        rc.ReconstructionRequest(
            reference_frame=reference,
            target_frame=target,
            mesh_dimension=3,
            metadata={
                "reconstruction_runtime": "single_step_cached",
                "step_size_calib": False,
                "n_elec": 8,
                "n_rings": 2,
            },
        )
    )

    assert calls == {"measurement": 1, "parameter": 0}
    assert np.allclose(result.conductivity, np.ones_like(delta_sigma) + delta_sigma)
    assert np.allclose(result.measured, expected_dv)
    assert np.allclose(result.simulated, pred_diff)
    assert result.metadata["single_step_operator_space"] == "measurement"


def test_single_step_cached_limits_alpha_before_forward_validation(
    monkeypatch: pytest.MonkeyPatch,
):
    reference = FrameData(
        real=np.array([1.0, 2.0, 3.0], dtype=float),
        imag=np.zeros(3, dtype=float),
        timestamp=0.0,
        frame_index=0,
    )
    target = FrameData(
        real=np.array([2.0, 4.0, 6.0], dtype=float),
        imag=np.zeros(3, dtype=float),
        timestamp=0.0,
        frame_index=1,
    )
    delta_sigma = np.array([-2.0, 0.1], dtype=float)
    sigma_bg = np.ones_like(delta_sigma)
    sigma_floor = 0.2
    base_meas = np.array([10.0, 20.0, 30.0], dtype=float)
    pred_diff = np.array([0.05, 0.1, 0.15], dtype=float)
    captured_sigma: list[np.ndarray] = []

    class _StubForwardModel:
        def fwd_solve(self, image):
            sigma = np.asarray(image.elem_data, dtype=float)
            captured_sigma.append(sigma.copy())
            assert np.all(np.isfinite(sigma))
            assert float(np.min(sigma)) > sigma_floor
            return SimpleNamespace(meas=base_meas + pred_diff), None

    ctx = {
        "mesh": object(),
        "display_node_coords": np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=float,
        ),
        "display_cell_connectivity": np.array([[0, 1, 2]], dtype=int),
        "operator_bundle": {
            "strict_solver_backend_effective": "measurement-exact",
        },
        "sigma_bg": sigma_bg,
        "fwd_model": _StubForwardModel(),
        "base_meas": base_meas,
        "cache_build_seconds": {},
        "cache_miss_reasons": {},
        "cache_manager": None,
    }

    def _calibration_failed(**_kwargs):
        raise RuntimeError("candidate sigma was infeasible")

    fake_diff_runner = SimpleNamespace(
        STRICT_SOLVER_BACKEND_MEASUREMENT="measurement-exact",
        _calibrate_step_size=_calibration_failed,
        _measurement_space_delta=lambda *, operator_bundle, rhs: delta_sigma,
        _solve_linear_from_bundle=lambda *_args, **_kwargs: delta_sigma,
        build_shared_context=lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        rc, "_load_gn_difference_runner_module", lambda: fake_diff_runner
    )
    monkeypatch.setattr(
        rc,
        "_ensure_single_step_cached_context",
        lambda _runtime, *, emit, build_shared_context: ctx,
    )

    result = rc._run_single_step_cached_request(
        rc.ReconstructionRequest(
            reference_frame=reference,
            target_frame=target,
            mesh_dimension=3,
            metadata={
                "reconstruction_runtime": "single_step_cached",
                "step_size_calib": True,
                "sigma_floor": sigma_floor,
                "n_elec": 8,
                "n_rings": 2,
            },
        )
    )

    assert captured_sigma
    assert float(np.min(captured_sigma[-1])) > sigma_floor
    assert 0.0 < result.metadata["step_size_alpha"] < 0.4
    assert result.metadata["step_size_alpha_requested"] == pytest.approx(1.0)
    assert result.metadata["step_size_alpha_limited"] is True
    np.testing.assert_allclose(result.conductivity, captured_sigma[-1])


def test_single_step_cached_uses_linearized_operator_solver(
    monkeypatch: pytest.MonkeyPatch,
):
    reference = FrameData(
        real=np.array([1.0, 2.0, 3.0], dtype=float),
        imag=np.zeros(3, dtype=float),
        timestamp=0.0,
        frame_index=0,
    )
    target = FrameData(
        real=np.array([1.5, 2.25, 2.75], dtype=float),
        imag=np.zeros(3, dtype=float),
        timestamp=0.0,
        frame_index=1,
    )
    expected_dv = target.real - reference.real
    delta_sigma = np.array([0.2, -0.1], dtype=float)
    base_meas = np.array([10.0, 20.0, 30.0], dtype=float)
    pred_diff = np.array([0.05, -0.1, 0.15], dtype=float)
    calls = {"linearized": 0, "measurement": 0, "parameter": 0}

    class _StubForwardModel:
        def fwd_solve(self, image):
            assert image.elem_data is not None
            return SimpleNamespace(meas=base_meas + pred_diff), None

    ctx = {
        "mesh": object(),
        "display_node_coords": np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=float,
        ),
        "display_cell_connectivity": np.array([[0, 1, 2]], dtype=int),
        "operator_bundle": {
            "jacobian_representation": "linearized",
            "strict_solver_backend_effective": "dense-param",
        },
        "jacobian_representation": "linearized",
        "sigma_bg": np.ones_like(delta_sigma),
        "fwd_model": _StubForwardModel(),
        "base_meas": base_meas,
        "cache_build_seconds": {},
        "cache_miss_reasons": {},
        "cache_manager": None,
    }

    def _solve_linearized_delta(*, operator_bundle, rhs):
        calls["linearized"] += 1
        assert operator_bundle is ctx["operator_bundle"]
        assert np.allclose(rhs, expected_dv)
        return delta_sigma

    fake_diff_runner = SimpleNamespace(
        STRICT_SOLVER_BACKEND_MEASUREMENT="measurement-exact",
        _calibrate_step_size=lambda **_kwargs: 1.0,
        _measurement_space_delta=lambda **_kwargs: calls.__setitem__("measurement", 1),
        _solve_linear_from_bundle=lambda *_args, **_kwargs: calls.__setitem__(
            "parameter", 1
        ),
        _solve_linearized_delta=_solve_linearized_delta,
        build_shared_context=lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        rc, "_load_gn_difference_runner_module", lambda: fake_diff_runner
    )
    monkeypatch.setattr(
        rc,
        "_ensure_single_step_cached_context",
        lambda _runtime, *, emit, build_shared_context: ctx,
    )

    result = rc._run_single_step_cached_request(
        rc.ReconstructionRequest(
            reference_frame=reference,
            target_frame=target,
            mesh_dimension=3,
            metadata={
                "reconstruction_runtime": "single_step_cached",
                "step_size_calib": False,
                "n_elec": 8,
                "n_rings": 2,
            },
        )
    )

    assert calls == {"linearized": 1, "measurement": 0, "parameter": 0}
    assert np.allclose(result.conductivity, np.ones_like(delta_sigma) + delta_sigma)
    assert result.metadata["single_step_operator_space"] == "linearized"


def test_mesh_setup_panel_exposes_tetra_and_hex_3d_families():
    _get_app()
    panel = MeshSetupPanel()
    try:
        panel.set_config({"mesh_dimension": 3, "mesh_family": "tetra"})
        assert panel.get_config()["mesh_family"] == "tetra"

        panel.set_config({"mesh_dimension": 3, "mesh_family": "hex"})
        assert panel.get_config()["mesh_family"] == "hex"

        panel.set_config({"mesh_dimension": 2, "mesh_family": "hex"})
        assert panel.get_config()["mesh_family"] == "tetra"
    finally:
        panel.close()


def test_mesh_setup_panel_exposes_2d_length_and_3d_area_geometry():
    _get_app()
    panel = MeshSetupPanel()
    try:
        panel.set_config({"mesh_dimension": 2, "radius": 1.0, "n_electrodes": 16})
        cfg_default_2d = panel.get_config()
        assert cfg_default_2d["electrode_length_m_override"] is None
        assert cfg_default_2d["electrode_coverage"] == pytest.approx(0.5)
        assert panel._electrode_length_spin.value() == pytest.approx(
            2.0 * np.pi * 0.5 / 16.0,
            abs=1.0e-6,
        )

        panel.set_config(
            {
                "mesh_dimension": 2,
                "radius": 1.0,
                "n_electrodes": 16,
                "electrode_length_m_override": 0.125,
            }
        )
        cfg_2d = panel.get_config()
        assert cfg_2d["electrode_length_m_override"] == pytest.approx(0.125)
        assert cfg_2d["electrode_coverage"] == pytest.approx(
            0.125 / (2.0 * np.pi / 16.0)
        )
        assert cfg_2d["electrode_area_m2_override"] is None
        assert panel._electrode_length_spin.isEnabled()
        assert not panel._electrode_area_spin.isEnabled()

        panel.set_config(
            {
                "mesh_dimension": 3,
                "radius": 0.18,
                "height": 0.16,
                "n_electrodes": 8,
                "n_rings": 2,
                "electrode_area_m2_override": 0.003,
            }
        )
        cfg_3d = panel.get_config()
        expected_length = 2.0 * np.pi * 0.18 * 0.5 / 8.0
        assert cfg_3d["electrode_length_m_override"] == pytest.approx(expected_length)
        assert cfg_3d["electrode_area_m2_override"] == pytest.approx(0.003)
        assert cfg_3d["electrode_height_ratio"] == pytest.approx(
            0.003 / (expected_length * 0.16)
        )
        assert not panel._electrode_length_spin.isEnabled()
        assert panel._electrode_area_spin.isEnabled()
    finally:
        panel.close()


def test_mesh_setup_panel_clamps_3d_area_to_non_overlapping_ring_windows():
    _get_app()
    panel = MeshSetupPanel()
    try:
        panel.set_config(
            {
                "mesh_dimension": 3,
                "radius": 0.18,
                "height": 0.16,
                "n_electrodes": 8,
                "n_rings": 8,
                "electrode_layout": "ring_major",
                "electrode_area_m2_override": 0.003,
            }
        )

        cfg = panel.get_config()

        levels = electrode_level_fractions_for_rings(8)
        max_ratio = max_electrode_height_ratio_for_rings(8)
        expected_length = 2.0 * np.pi * 0.18 * 0.5 / 8.0
        assert cfg["electrode_height_ratio"] <= max_ratio
        assert cfg["electrode_height_ratio"] == pytest.approx(max_ratio, rel=1.0e-5)
        assert cfg["electrode_area_m2_override"] == pytest.approx(
            expected_length * 0.16 * cfg["electrode_height_ratio"]
        )
        assert panel._electrode_area_spin.value() == pytest.approx(
            cfg["electrode_area_m2_override"]
        )
        Cylinder3DMeshConfig(
            radius=cfg["radius"],
            height=cfg["height"],
            electrode_height_ratio=cfg["electrode_height_ratio"],
            electrode_level_fractions=levels,
        )
    finally:
        panel.close()


def test_gpu_forward_runtime_keeps_tetra_and_hex_distinct(monkeypatch):
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "gpu")
    monkeypatch.setattr(
        "eit_app.controllers.forward_solver_controller.probe_petsc_cuda_runtime",
        lambda: {
            "petsc_cuda": True,
            "petsc_hypre": True,
            "petsc_amgx": False,
            "probe_cache": {"hit": True, "layer": "disk"},
        },
    )

    tetra = _resolve_forward_runtime(
        ForwardModelConfig(mesh_dimension=3, mesh_family="tetra")
    )
    assert tetra["mesh_family"] == "tetra"
    assert tetra["forward_backend"] == "dolfinx"
    assert tetra["petsc_device"] == "cuda"
    assert tetra["device"] == "cuda"
    assert tetra["acceleration_profile"] == "gpu3d"
    assert tetra["forward_solver_preset"] == "spd_gamg"
    assert tetra["petsc_amgx_available"] is False
    assert tetra["petsc_cuda_probe_cache_hit"] is True
    assert tetra["petsc_cuda_probe_cache"]["layer"] == "disk"
    assert tetra["forward_mat_solve"] == "off"
    assert (
        tetra["forward_mat_solve_policy_reason"] == "cuda_spd_gamg_matsolve_disabled_b6"
    )

    hex_cfg = _resolve_forward_runtime(
        ForwardModelConfig(mesh_dimension=3, mesh_family="hex")
    )
    assert hex_cfg["mesh_family"] == "hex"
    assert hex_cfg["forward_backend"] == "cuda_structured"
    assert hex_cfg["petsc_device"] == "cuda"


def test_v624_complex_gpu_forward_runtime_keeps_hex_on_dolfinx_petsc_cuda(
    monkeypatch,
):
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "gpu")
    monkeypatch.setattr(
        "eit_app.controllers.forward_solver_controller.probe_petsc_cuda_runtime",
        lambda: {
            "petsc_cuda": True,
            "petsc_hypre": True,
            "petsc_amgx": False,
            "probe_cache": {"hit": True, "layer": "disk"},
        },
    )
    cfg = ForwardModelConfig(
        mesh_dimension=3,
        mesh_family="hex",
        background_conductivity=1.0 + 0.2j,
    )

    runtime = _resolve_forward_runtime(cfg)

    assert runtime["mesh_family"] == "hex"
    assert runtime["forward_backend"] == "dolfinx"
    assert runtime["petsc_device"] == "cuda"
    assert runtime["device"] == "cuda"
    assert runtime["complex_admittivity_requested"] is True


def test_v624_complex_inhomogeneity_marks_forward_request_complex() -> None:
    req = ForwardSolverRequest(
        mesh_dimension=3,
        background_conductivity=1.0,
        inhomogeneities=[InhomogeneitySpec(shape="sphere", conductivity="2+0.25j")],
        forward_model_config={
            "mesh_dimension": 3,
            "mesh_family": "hex",
            "background_conductivity": 1.0,
        },
    )
    cfg = ForwardModelConfig.from_mapping(req.forward_model_config)

    assert _forward_request_requires_complex_admittivity(req, cfg) is True


def test_v83_gpu_forward_runtime_keeps_2d_auto_petsc_on_cpu(monkeypatch):
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "gpu")

    runtime = _resolve_forward_runtime(
        ForwardModelConfig(mesh_dimension=2, petsc_device="auto")
    )

    assert runtime["acceleration_profile"] == "default"
    assert runtime["forward_backend"] == "dolfinx"
    assert runtime["petsc_device"] == "cpu"
    assert runtime["device"] == "auto"
    assert runtime["forward_mat_solve"] == "off"


def test_gpu_reconstruction_runtime_keeps_tetra_and_hex_distinct(monkeypatch):
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "gpu")
    monkeypatch.setattr(
        "eit_app.controllers.reconstruction_controller.probe_petsc_cuda_runtime",
        lambda: {
            "petsc_cuda": True,
            "petsc_hypre": True,
            "petsc_amgx": False,
        },
    )

    tetra = _resolve_reconstruction_runtime(
        {"mesh_family": "tetra", "forward_backend": "cuda_structured"},
        mesh_dim=3,
    )
    assert tetra["mesh_family"] == "tetra"
    assert tetra["forward_backend"] == "dolfinx"
    assert tetra["petsc_device"] == "cuda"
    assert tetra["device"] == "cuda"
    assert tetra["acceleration_profile"] == "gpu3d"
    assert tetra["forward_solver_preset"] == "spd_gamg"
    assert (
        tetra["forward_solver_policy_reason"]
        == "amgx_unavailable_downgraded_to_spd_gamg"
    )
    assert tetra["petsc_amgx_available"] is False
    assert tetra["forward_mat_solve"] == "off"
    assert (
        tetra["forward_mat_solve_policy_reason"] == "cuda_spd_gamg_matsolve_disabled_b6"
    )

    requested_amgx = _resolve_reconstruction_runtime(
        {"mesh_family": "tetra", "forward_solver_preset": "cuda_amgx"},
        mesh_dim=3,
    )
    assert requested_amgx["forward_solver_preset_requested"] == "cuda_amgx"
    assert requested_amgx["forward_solver_preset"] == "spd_gamg"
    assert requested_amgx["forward_mat_solve"] == "off"

    explicit_matsolve = _resolve_reconstruction_runtime(
        {
            "mesh_family": "tetra",
            "forward_solver_preset": "spd_gamg",
            "forward_mat_solve": "on",
        },
        mesh_dim=3,
    )
    assert explicit_matsolve["forward_mat_solve_requested"] == "on"
    assert explicit_matsolve["forward_mat_solve"] == "on"
    assert explicit_matsolve["forward_mat_solve_policy_reason"] == ""

    hex_cfg = _resolve_reconstruction_runtime({"mesh_family": "hex"}, mesh_dim=3)
    assert hex_cfg["mesh_family"] == "hex"
    assert hex_cfg["forward_backend"] == "cuda_structured"


def test_v83_gpu_reconstruction_runtime_keeps_2d_auto_petsc_on_cpu(monkeypatch):
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "gpu")

    runtime = _resolve_reconstruction_runtime({"petsc_device": "auto"}, mesh_dim=2)

    assert runtime["acceleration_profile"] == "default"
    assert runtime["forward_backend"] == "dolfinx"
    assert runtime["petsc_device"] == "cpu"
    assert runtime["device"] == "auto"
    assert runtime["forward_mat_solve"] == "off"


def test_single_step_solver_diagnostics_exposes_runtime_summary():
    diagnostics = rc._single_step_cached_solver_diagnostics(
        {
            "mesh_family": "tetra",
            "forward_backend": "dolfinx",
            "petsc_device": "cuda",
            "petsc_backend_info": {
                "forward_backend_effective": "dolfinx",
                "solver_preset": "spd_gamg",
                "petsc_amgx_available": False,
                "petsc_device_requested": "cuda",
                "petsc_device_effective": "cuda",
            },
            "device_requested": "cuda",
            "device_effective": "cuda",
            "torch_device": "cuda",
            "jacobian_representation": "linearized",
            "mesh_cache_hit": True,
            "mesh_cache_layer": "disk",
            "mesh_cache_name": "mesh3d_demo",
            "cache_lookups": {
                "base_meas": {"hit": True, "layer": "disk"},
                "operator_A": {"hit": False, "layer": "process"},
                "operator_rom_reduced_rm": {"hit": False, "layer": "disabled"},
            },
            "cache_build_seconds": {},
            "cache_miss_reasons": {},
            "cache_manager": None,
        },
        strict_backend="measurement-exact",
    )

    runtime = diagnostics["runtime"]
    assert runtime["mesh_family"] == "tetra"
    assert runtime["forward_backend_effective"] == "dolfinx"
    assert runtime["forward_solver_preset"] == "spd_gamg"
    assert runtime["petsc_amgx_available"] is False
    assert runtime["petsc_device_effective"] == "cuda"
    assert runtime["torch_device"] == "cuda"
    assert runtime["jacobian_representation"] == "linearized"
    assert runtime["mesh_cache_hit"] is True
    assert runtime["cache_hit"] is False
    assert runtime["cache_hits"] == {"base_meas": True, "operator_A": False}


def test_embedded_vtk_disabled_for_offscreen_qt(monkeypatch):
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.delenv("EIT_APP_DISABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    enabled, reason = embedded_vtk_status()

    assert enabled is False
    assert embedded_vtk_enabled() is False
    assert "offscreen" in reason


def test_embedded_vtk_can_be_forced(monkeypatch):
    monkeypatch.delenv("EIT_APP_DISABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("EIT_APP_ENABLE_EMBEDDED_VTK", "1")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    enabled, reason = embedded_vtk_status()

    assert enabled is True
    assert embedded_vtk_enabled() is True
    assert "forced" in reason


def test_embedded_vtk_enabled_on_wsl_when_qt_uses_xcb(monkeypatch):
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.delenv("EIT_APP_DISABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "xcb")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-22.04")
    monkeypatch.setenv("DISPLAY", ":0")

    enabled, reason = embedded_vtk_status()

    assert enabled is True
    assert embedded_vtk_enabled() is True
    assert "XCB" in reason or "compatible" in reason


def test_embedded_vtk_disabled_on_wsl_without_xcb(monkeypatch):
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.delenv("EIT_APP_DISABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-22.04")
    monkeypatch.setenv("DISPLAY", ":0")

    enabled, reason = embedded_vtk_status()

    assert enabled is False
    assert embedded_vtk_enabled() is False
    assert "xcb" in reason


def test_3d_payload_stays_in_3d_widget_when_vtk_disabled(monkeypatch):
    _get_app()
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    slot = _ConductivityViewSlot("Conductivity")
    calls: list[tuple[str, str | None]] = []

    def unexpected_mpl_update(_sigma, _coords, _cells, title=None, **_kwargs):
        raise AssertionError("3D volume data must not fall back to the 2D plot")

    def fake_3d_update(_sigma, _coords, _cells, title=None, **_kwargs):
        calls.append(("3d", title))

    monkeypatch.setattr(slot._mpl, "update_image", unexpected_mpl_update)
    monkeypatch.setattr(slot._three_d, "update_image", fake_3d_update)

    sigma, coords, cells = _tetra_payload()
    slot.update_image(sigma, coords, cells, title="Truth")

    assert calls == [("3d", "Truth")]
    assert slot._stack.currentWidget() is slot._three_d
    slot.close()


def test_pyvista_offscreen_backend_renders_small_tetra(monkeypatch):
    _get_app()
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    slot = _ConductivityViewSlot("Conductivity")

    sigma, coords, cells = _tetra_payload()
    slot.update_image(sigma, coords, cells, title="Truth")

    assert slot._stack.currentWidget() is slot._three_d
    assert slot._three_d._stack.currentWidget() is slot._three_d._offscreen_host
    assert slot._three_d._last_image is not None
    assert slot._three_d._render_backend == "pyvista_offscreen"
    assert slot._three_d._offscreen_label.pixmap() is not None
    slot.close()


def test_pyvista_offscreen_backend_renders_hex_when_vtk_disabled(monkeypatch):
    _get_app()
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    slot = _ConductivityViewSlot("Conductivity")

    sigma, coords, cells = _hex_payload()
    slot.update_image(sigma, coords, cells, title="Hex Truth")

    assert slot._stack.currentWidget() is slot._three_d
    assert slot._three_d._stack.currentWidget() is slot._three_d._offscreen_host
    assert slot._three_d._last_image is not None
    assert slot._three_d._last_image[3] == "Hex Truth"
    assert slot._three_d._render_backend == "pyvista_offscreen"
    assert slot._three_d._offscreen_label.pixmap() is not None
    slot.close()


def test_pyvista_offscreen_controls_keep_rendered_canvas(monkeypatch):
    _get_app()
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widget = Conductivity3DWidget("Conductivity")

    sigma, coords, cells = _inhomogeneous_tetra_payload()
    widget.update_image(sigma, coords, cells, title="Truth")
    initial_pixmap = widget._offscreen_label.pixmap()
    assert widget._render_backend == "pyvista_offscreen"
    assert initial_pixmap is not None

    widget._opacity_slider.setValue(30)
    QApplication.processEvents()
    assert widget._offscreen_label.pixmap() is not None
    assert widget._offscreen_mesh_actor is not None
    assert widget._offscreen_mesh_actor.GetProperty().GetOpacity() == pytest.approx(
        0.30
    )

    assert widget._offscreen_highlight_actor is not None
    widget._highlight_check.setChecked(False)
    QApplication.processEvents()
    assert widget._offscreen_label.pixmap() is not None
    assert widget._offscreen_highlight_actor.GetVisibility() == 0

    widget._wire_check.setChecked(False)
    QApplication.processEvents()
    assert widget._offscreen_label.pixmap() is not None
    assert widget._offscreen_wire_actor is not None
    assert widget._offscreen_wire_actor.GetVisibility() == 0

    widget._reset_btn.click()
    QApplication.processEvents()
    assert widget._offscreen_label.pixmap() is not None
    assert widget._stack.currentWidget() is widget._offscreen_host
    widget.close()


def test_pyvista_offscreen_backend_renders_point_cloud_mode(monkeypatch):
    _get_app()
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widget = Conductivity3DWidget("Conductivity")
    widget.set_display_mode(DISPLAY_MODE_POINTS)

    sigma, coords, cells = _inhomogeneous_tetra_payload()
    widget.update_image(sigma, coords, cells, title="Truth")

    assert widget._render_backend == "pyvista_offscreen"
    assert widget._stack.currentWidget() is widget._offscreen_host
    assert widget._offscreen_label.pixmap() is not None
    assert widget._offscreen_mesh_actor is not None
    assert widget._offscreen_highlight_actor is not None
    widget.close()


def test_v623_pyvista_surface_helper_supports_legacy_extract_surface_signature():
    class _Surface:
        def __init__(self) -> None:
            self.feature_kwargs = None

        def extract_feature_edges(self, **kwargs):
            self.feature_kwargs = kwargs
            return SimpleNamespace(n_points=1, source=self)

    class _Grid:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.surface = _Surface()

        def extract_surface(self, **kwargs):
            self.calls.append(dict(kwargs))
            if "algorithm" in kwargs:
                raise TypeError(
                    "DataSetFilters.extract_surface() got an unexpected keyword "
                    "argument 'algorithm'"
                )
            return self.surface

    grid = _Grid()
    outline = _pyvista_feature_outline(grid, feature_angle=30.0)

    assert outline.n_points == 1
    assert grid.calls == [{"algorithm": "dataset_surface"}, {}]
    assert grid.surface.feature_kwargs == {
        "boundary_edges": True,
        "feature_edges": True,
        "feature_angle": 30.0,
        "non_manifold_edges": False,
        "manifold_edges": False,
    }


def test_v623_pyvista_offscreen_exception_reports_without_gui_crash(monkeypatch):
    _get_app()
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.delenv("EIT_APP_3D_PYVISTA_OFFSCREEN_NEGATIVE_CACHE", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widget3d._clear_pyvista_offscreen_failure_cache()
    widget = Conductivity3DWidget("Conductivity")
    calls: list[str] = []

    def fail_offscreen(_sigma, _coords, _cells):
        calls.append("offscreen")
        raise TypeError("legacy pyvista extract_surface signature")

    monkeypatch.setattr(widget, "_render_pyvista_offscreen_scene", fail_offscreen)

    try:
        sigma, coords, cells = _tetra_payload()
        widget.update_image(sigma, coords, cells, title="Truth")

        assert calls == ["offscreen"]
        assert widget._render_backend == "caption"
        assert widget._stack.currentWidget() is widget._caption_label
        assert (
            "legacy pyvista extract_surface signature" in widget._caption_label.text()
        )
        assert "legacy pyvista extract_surface" in (
            widget3d._pyvista_offscreen_failure_reason() or ""
        )
    finally:
        widget3d._clear_pyvista_offscreen_failure_cache()
        widget.close()


def test_v332_pyvista_offscreen_failure_cache_skips_retry(monkeypatch):
    _get_app()
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.delenv("EIT_APP_3D_PYVISTA_OFFSCREEN_NEGATIVE_CACHE", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widget3d._clear_pyvista_offscreen_failure_cache()
    widget = Conductivity3DWidget("Conductivity")
    calls: list[str] = []

    def fail_offscreen(_sigma, _coords, _cells):
        calls.append("offscreen")
        widget3d._mark_pyvista_offscreen_failure("unit offscreen failure")
        return False

    monkeypatch.setattr(widget, "_render_pyvista_offscreen_scene", fail_offscreen)

    try:
        sigma, coords, cells = _tetra_payload()
        widget.update_image(sigma, coords, cells, title="Truth")
        widget.update_image(sigma, coords, cells, title="Truth")

        assert calls == ["offscreen"]
        assert widget._render_backend == "caption"
        assert "unit offscreen failure" in widget._caption_label.text()
    finally:
        widget3d._clear_pyvista_offscreen_failure_cache()
        widget.close()


def test_pyvista_offscreen_drag_defaults_to_full_resolution_60fps(monkeypatch):
    _get_app()
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.delenv("EIT_APP_3D_DRAG_FPS", raising=False)
    monkeypatch.delenv("EIT_APP_3D_DRAG_RENDER_SCALE", raising=False)

    class FakePlotter:
        def __init__(self) -> None:
            self._window_size = (0, 0)
            self.window_size_sets = 0
            self.screenshot_sizes: list[tuple[int, int]] = []

        @property
        def window_size(self) -> tuple[int, int]:
            return self._window_size

        @window_size.setter
        def window_size(self, value: tuple[int, int]) -> None:
            self.window_size_sets += 1
            self._window_size = value

        def render(self) -> None:
            pass

        def screenshot(self, *, return_img: bool):  # noqa: ANN001
            assert return_img is True
            width, height = self.window_size
            self.screenshot_sizes.append((width, height))
            return np.zeros((height, width, 3), dtype=np.uint8)

    widget = Conductivity3DWidget("Conductivity")
    widget._offscreen_label.resize(800, 600)
    plotter = FakePlotter()
    widget._offscreen_plotter = plotter
    widget._render_backend = "pyvista_offscreen"

    widget._is_dragging_offscreen = False
    widget._refresh_offscreen_pixmap()
    idle_pixmap = widget._offscreen_label.pixmap()
    assert idle_pixmap is not None
    idle_logical = (
        idle_pixmap.width() / idle_pixmap.devicePixelRatioF(),
        idle_pixmap.height() / idle_pixmap.devicePixelRatioF(),
    )

    widget._is_dragging_offscreen = True
    widget._refresh_offscreen_pixmap()
    drag_pixmap = widget._offscreen_label.pixmap()
    assert drag_pixmap is not None
    drag_logical = (
        drag_pixmap.width() / drag_pixmap.devicePixelRatioF(),
        drag_pixmap.height() / drag_pixmap.devicePixelRatioF(),
    )

    assert widget._offscreen_render_timer.interval() == 17
    assert widget._offscreen_drag_render_scale == pytest.approx(1.0)
    assert plotter.screenshot_sizes[1] == plotter.screenshot_sizes[0]
    assert plotter.window_size_sets == 1
    assert drag_pixmap.width() == idle_pixmap.width()
    assert drag_pixmap.height() == idle_pixmap.height()
    assert drag_logical == pytest.approx(idle_logical)
    assert drag_logical == pytest.approx((800.0, 600.0))

    widget.close()


def test_pyvista_offscreen_drag_scale_can_be_reduced_without_size_jitter(monkeypatch):
    _get_app()
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("EIT_APP_3D_DRAG_FPS", "30")
    monkeypatch.setenv("EIT_APP_3D_DRAG_RENDER_SCALE", "0.5")

    class FakePlotter:
        def __init__(self) -> None:
            self._window_size = (0, 0)
            self.window_size_sets = 0
            self.screenshot_sizes: list[tuple[int, int]] = []

        @property
        def window_size(self) -> tuple[int, int]:
            return self._window_size

        @window_size.setter
        def window_size(self, value: tuple[int, int]) -> None:
            self.window_size_sets += 1
            self._window_size = value

        def render(self) -> None:
            pass

        def screenshot(self, *, return_img: bool):  # noqa: ANN001
            assert return_img is True
            width, height = self.window_size
            self.screenshot_sizes.append((width, height))
            return np.zeros((height, width, 3), dtype=np.uint8)

    widget = Conductivity3DWidget("Conductivity")
    widget._offscreen_label.resize(800, 600)
    plotter = FakePlotter()
    widget._offscreen_plotter = plotter
    widget._render_backend = "pyvista_offscreen"

    widget._is_dragging_offscreen = False
    widget._refresh_offscreen_pixmap()
    idle_pixmap = widget._offscreen_label.pixmap()
    assert idle_pixmap is not None
    idle_logical = (
        idle_pixmap.width() / idle_pixmap.devicePixelRatioF(),
        idle_pixmap.height() / idle_pixmap.devicePixelRatioF(),
    )

    widget._is_dragging_offscreen = True
    widget._refresh_offscreen_pixmap()
    drag_pixmap = widget._offscreen_label.pixmap()
    assert drag_pixmap is not None
    drag_logical = (
        drag_pixmap.width() / drag_pixmap.devicePixelRatioF(),
        drag_pixmap.height() / drag_pixmap.devicePixelRatioF(),
    )

    assert widget._offscreen_render_timer.interval() == 33
    assert widget._offscreen_drag_render_scale == pytest.approx(0.5)
    assert plotter.screenshot_sizes[1][0] < plotter.screenshot_sizes[0][0]
    assert plotter.window_size_sets == 2
    assert drag_pixmap.width() == idle_pixmap.width()
    assert drag_pixmap.height() == idle_pixmap.height()
    assert drag_logical == pytest.approx(idle_logical)

    widget.close()


def test_3d_payload_uses_vtk_widget_when_forced(monkeypatch):
    _get_app()
    monkeypatch.setenv("EIT_APP_ENABLE_EMBEDDED_VTK", "1")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    slot = _ConductivityViewSlot("Conductivity")
    calls: list[tuple[str, str | None]] = []

    def unexpected_mpl_update(_sigma, _coords, _cells, title=None, **_kwargs):
        raise AssertionError("Matplotlib fallback should not run when VTK is forced")

    def fake_vtk_update(_sigma, _coords, _cells, title=None, **_kwargs):
        calls.append(("vtk", title))

    monkeypatch.setattr(slot._mpl, "update_image", unexpected_mpl_update)
    monkeypatch.setattr(slot._three_d, "update_image", fake_vtk_update)

    sigma, coords, cells = _tetra_payload()
    slot.update_image(sigma, coords, cells, title="Truth")

    assert calls == [("vtk", "Truth")]
    assert slot._stack.currentWidget() is slot._three_d
    slot.close()


def test_hex_3d_payload_uses_vtk_widget_when_forced(monkeypatch):
    _get_app()
    monkeypatch.setenv("EIT_APP_ENABLE_EMBEDDED_VTK", "1")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    slot = _ConductivityViewSlot("Conductivity")
    calls: list[tuple[str, tuple[int, int], str | None]] = []

    def unexpected_mpl_update(_sigma, _coords, _cells, title=None, **_kwargs):
        raise AssertionError("Hex volume data must use the 3D VTK widget")

    def fake_vtk_update(_sigma, _coords, cells, title=None, **_kwargs):
        calls.append(("vtk", tuple(cells.shape), title))

    monkeypatch.setattr(slot._mpl, "update_image", unexpected_mpl_update)
    monkeypatch.setattr(slot._three_d, "update_image", fake_vtk_update)

    sigma, coords, cells = _hex_payload()
    slot.update_image(sigma, coords, cells, title="Hex Truth")

    assert calls == [("vtk", (1, 8), "Hex Truth")]
    assert slot._stack.currentWidget() is slot._three_d
    slot.close()


def test_3d_widget_builds_pyvista_hex_grid():
    pv = pytest.importorskip("pyvista")
    _get_app()

    class _FakeActor:
        def __init__(self) -> None:
            self.visible = True

        def SetVisibility(self, visible):  # noqa: N802 (VTK API)
            self.visible = bool(visible)

        def GetProperty(self):  # noqa: N802 (VTK API)
            return self

        def SetOpacity(self, _opacity):  # noqa: N802 (VTK API)
            pass

    class _FakePlotter:
        def __init__(self) -> None:
            self.meshes = []
            self.render_count = 0

        def add_mesh(self, mesh, *args, **kwargs):
            self.meshes.append((mesh, kwargs))
            return _FakeActor()

        def remove_actor(self, _actor, render=False):
            pass

        def reset_camera(self):
            pass

        def render(self):
            self.render_count += 1

    widget = Conductivity3DWidget("Hex")
    fake_plotter = _FakePlotter()
    widget._plotter = fake_plotter

    sigma, coords, cells = _hex_payload()
    widget._build_scene(sigma, coords, cells)

    grid, kwargs = fake_plotter.meshes[0]
    assert grid.n_cells == 1
    assert int(grid.celltypes[0]) == int(pv.CellType.HEXAHEDRON)
    assert kwargs["preference"] == "cell"
    assert fake_plotter.render_count == 1
    widget.close()


def test_3d_widget_point_cloud_mode_builds_cell_center_polydata():
    pv = pytest.importorskip("pyvista")
    _get_app()

    class _FakeActor:
        def __init__(self) -> None:
            self.visible = True

        def SetVisibility(self, visible):  # noqa: N802 (VTK API)
            self.visible = bool(visible)

        def GetProperty(self):  # noqa: N802 (VTK API)
            return self

        def SetOpacity(self, _opacity):  # noqa: N802 (VTK API)
            pass

    class _FakePlotter:
        def __init__(self) -> None:
            self.meshes = []
            self.render_count = 0

        def add_mesh(self, mesh, *args, **kwargs):
            self.meshes.append((mesh, kwargs))
            return _FakeActor()

        def remove_actor(self, _actor, render=False):
            pass

        def reset_camera(self):
            pass

        def render(self):
            self.render_count += 1

    widget = Conductivity3DWidget("Hex")
    widget.set_display_mode(DISPLAY_MODE_POINTS)
    fake_plotter = _FakePlotter()
    widget._plotter = fake_plotter

    sigma, coords, cells = _hex_payload()
    widget._build_scene(sigma, coords, cells)

    cloud, kwargs = fake_plotter.meshes[0]
    assert isinstance(cloud, pv.PolyData)
    assert cloud.n_points == cells.shape[0]
    np.testing.assert_allclose(cloud.points[0], coords[cells[0], :3].mean(axis=0))
    assert kwargs["scalars"] == "sigma"
    assert kwargs["render_points_as_spheres"] is True
    assert kwargs["point_size"] >= 4.0
    assert fake_plotter.render_count == 1
    widget.close()


def test_3d_display_mode_switch_rerenders_cached_payload(monkeypatch):
    _get_app()
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widget = Conductivity3DWidget("Conductivity")
    calls: list[str] = []

    def fake_offscreen_render(_sigma, _coords, _cells):
        calls.append(widget.display_mode())
        widget._render_backend = "pyvista_offscreen"
        return True

    monkeypatch.setattr(
        widget,
        "_render_pyvista_offscreen_scene",
        fake_offscreen_render,
    )

    sigma, coords, cells = _tetra_payload()
    widget.update_image(sigma, coords, cells, title="Truth")
    widget.set_display_mode(DISPLAY_MODE_POINTS)

    assert calls == [DISPLAY_MODE_VOLUME, DISPLAY_MODE_POINTS]
    assert widget._points_mode_btn.isChecked()
    assert not widget._volume_mode_btn.isChecked()
    widget.close()


def test_v148_large_3d_payload_auto_switches_to_point_cloud(monkeypatch):
    _get_app()
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("EIT_APP_3D_AUTO_POINTS_CELLS", "1")
    widget = Conductivity3DWidget("Conductivity")
    calls: list[str] = []

    def fake_offscreen_render(_sigma, _coords, _cells):
        calls.append(widget.display_mode())
        widget._render_backend = "pyvista_offscreen"
        return True

    monkeypatch.setattr(
        widget,
        "_render_pyvista_offscreen_scene",
        fake_offscreen_render,
    )

    sigma, coords, cells = _tetra_payload()
    widget.update_image(sigma, coords, cells, title="Truth")

    assert calls == [DISPLAY_MODE_POINTS]
    assert widget._points_mode_btn.isChecked()
    assert not widget._volume_mode_btn.isChecked()
    widget.close()


def test_v629_manual_volume_mode_overrides_large_payload_auto_points(monkeypatch):
    _get_app()
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("EIT_APP_3D_AUTO_POINTS_CELLS", "1")
    widget = Conductivity3DWidget("Conductivity")
    calls: list[str] = []

    def fake_offscreen_render(_sigma, _coords, _cells):
        calls.append(widget.display_mode())
        widget._render_backend = "pyvista_offscreen"
        return True

    monkeypatch.setattr(
        widget,
        "_render_pyvista_offscreen_scene",
        fake_offscreen_render,
    )

    sigma, coords, cells = _tetra_payload()
    widget.update_image(sigma, coords, cells, title="Truth")
    widget.set_display_mode(DISPLAY_MODE_VOLUME)

    assert calls == [DISPLAY_MODE_POINTS, DISPLAY_MODE_VOLUME]
    assert widget._volume_mode_btn.isChecked()
    assert not widget._points_mode_btn.isChecked()
    widget.close()


def test_progressive_volume_upgrade_can_follow_auto_point_cloud(monkeypatch):
    _get_app()
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("EIT_APP_3D_AUTO_POINTS_CELLS", "1")
    monkeypatch.setenv("EIT_APP_3D_PROGRESSIVE_VOLUME_UPGRADE", "1")
    monkeypatch.setenv("EIT_APP_3D_PROGRESSIVE_VOLUME_DELAY_MS", "0")
    widget = Conductivity3DWidget("Conductivity")
    calls: list[str] = []

    def fake_offscreen_render(_sigma, _coords, _cells):
        calls.append(widget.display_mode())
        widget._render_backend = "pyvista_offscreen"
        return True

    monkeypatch.setattr(
        widget,
        "_render_pyvista_offscreen_scene",
        fake_offscreen_render,
    )

    sigma, coords, cells = _tetra_payload()
    widget.update_image(sigma, coords, cells, title="Truth")
    widget._run_progressive_volume_upgrade()

    assert calls == [DISPLAY_MODE_POINTS, DISPLAY_MODE_VOLUME]
    assert widget._volume_mode_btn.isChecked()
    assert not widget._points_mode_btn.isChecked()
    widget.close()


def test_v189_large_point_cloud_still_uses_pyvista_offscreen(monkeypatch):
    _get_app()
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("EIT_APP_3D_AUTO_POINTS_CELLS", "1")
    monkeypatch.setenv("EIT_APP_3D_PYVISTA_OFFSCREEN_MAX_CELLS", "1")
    widget = Conductivity3DWidget("Conductivity")
    calls: list[str] = []

    def fake_offscreen(_sigma, _coords, _cells):
        calls.append(widget.display_mode())
        widget._render_backend = "pyvista_offscreen"
        return True

    monkeypatch.setattr(widget, "_render_pyvista_offscreen_scene", fake_offscreen)

    sigma, coords, cells = _tetra_payload()
    widget.update_image(sigma, coords, cells, title="Truth")

    assert calls == [DISPLAY_MODE_POINTS]
    assert widget._render_backend == "pyvista_offscreen"
    widget.close()


def test_v189_pyvista_offscreen_skip_can_be_disabled(monkeypatch):
    monkeypatch.setenv("EIT_APP_3D_PYVISTA_OFFSCREEN_MAX_CELLS", "off")

    assert not _should_skip_pyvista_offscreen(1_000_000, DISPLAY_MODE_POINTS)


def test_v252_default_large_point_cloud_does_not_bypass_pyvista_offscreen(monkeypatch):
    monkeypatch.delenv("EIT_APP_3D_PYVISTA_OFFSCREEN_MAX_CELLS", raising=False)

    assert not _should_skip_pyvista_offscreen(11_999, DISPLAY_MODE_POINTS)
    assert not _should_skip_pyvista_offscreen(12_000, DISPLAY_MODE_POINTS)
    assert not _should_skip_pyvista_offscreen(12_000, DISPLAY_MODE_VOLUME)


def test_v568_wslg_wayland_does_not_skip_pyvista_offscreen(monkeypatch):
    reason = "WSLg embedded VTK requires QT_QPA_PLATFORM=xcb"

    monkeypatch.setattr(widget3d, "_running_under_wsl", lambda: True)
    monkeypatch.delenv("EIT_APP_3D_WSLG_PYVISTA_OFFSCREEN", raising=False)

    assert not _should_skip_pyvista_offscreen_for_reason(
        1,
        DISPLAY_MODE_VOLUME,
        reason,
    )

    monkeypatch.setenv("EIT_APP_3D_WSLG_PYVISTA_OFFSCREEN", "1")

    assert not _should_skip_pyvista_offscreen_for_reason(
        1,
        DISPLAY_MODE_VOLUME,
        reason,
    )


def test_v188_point_cloud_sampling_keeps_anomalies_and_caps_points(monkeypatch):
    monkeypatch.setenv("EIT_APP_3D_POINT_CLOUD_MAX_POINTS", "6")
    centers = np.column_stack(
        [
            np.linspace(0.0, 1.0, 30),
            np.zeros(30),
            np.zeros(30),
        ]
    )
    sigma = np.ones(30, dtype=float)
    sigma[[4, 20]] = 2.0

    sample_idx = _point_cloud_sample_indices(
        sigma,
        centers,
        anomaly_mode=ANOMALY_MODE_POSITIVE,
    )

    assert sample_idx.size <= 6
    assert {4, 20}.issubset(set(sample_idx.tolist()))


def test_v190_point_cloud_sampling_avoids_full_spatial_filter(monkeypatch):
    monkeypatch.setenv("EIT_APP_3D_POINT_CLOUD_MAX_POINTS", "8")
    centers = np.column_stack(
        [
            np.linspace(0.0, 1.0, 200),
            np.zeros(200),
            np.zeros(200),
        ]
    )
    sigma = np.ones(200, dtype=float)
    sigma[20:80] = 2.0
    seen: list[object] = []

    def _fake_spatial(mask, score, cell_centers):
        seen.append(cell_centers)
        if cell_centers is not None:
            raise AssertionError("full-data point-cloud sampling must stay O(n)")
        return mask

    monkeypatch.setattr(widget3d, "_spatially_coherent_anomaly_mask", _fake_spatial)

    sample_idx = _point_cloud_sample_indices(
        sigma,
        centers,
        anomaly_mode=ANOMALY_MODE_POSITIVE,
    )

    assert sample_idx.size <= 8
    assert seen == [None]


def test_v202_point_cloud_sampling_invalid_centers_avoids_full_arange(monkeypatch):
    n_points = 1_000_000
    sigma = np.ones(n_points, dtype=np.float32)
    centers = np.empty((0, 3), dtype=np.float32)
    original_arange = np.arange

    def _guard_arange(*args, **kwargs):
        if args and int(args[0]) == n_points:
            raise AssertionError("sampling must not allocate full range before capping")
        return original_arange(*args, **kwargs)

    monkeypatch.setattr(widget3d.np, "arange", _guard_arange)

    sample_idx = _point_cloud_sample_indices(
        sigma,
        centers,
        anomaly_mode=ANOMALY_MODE_POSITIVE,
        max_points=7,
    )

    assert sample_idx.size == 7
    assert sample_idx[0] == 0
    assert sample_idx[-1] == n_points - 1


def test_v202_point_cloud_sampling_avoids_full_background_flatnonzero(monkeypatch):
    centers = np.column_stack(
        [
            np.linspace(0.0, 1.0, 200, dtype=np.float32),
            np.zeros(200, dtype=np.float32),
            np.zeros(200, dtype=np.float32),
        ]
    )
    sigma = np.ones(200, dtype=np.float32)
    sigma[[12, 155]] = 2.0
    original_flatnonzero = np.flatnonzero

    def _guard_flatnonzero(mask):
        arr = np.asarray(mask)
        if arr.dtype == np.bool_ and arr.size == sigma.size:
            if int(np.count_nonzero(arr)) > 20:
                raise AssertionError("sampling must not materialize full background")
        return original_flatnonzero(mask)

    monkeypatch.setattr(widget3d.np, "flatnonzero", _guard_flatnonzero)

    sample_idx = _point_cloud_sample_indices(
        sigma,
        centers,
        anomaly_mode=ANOMALY_MODE_POSITIVE,
        max_points=8,
    )

    assert sample_idx.size <= 8
    assert {12, 155}.issubset(set(sample_idx.tolist()))


def test_v392_sample_background_indices_direct_fills_when_background_fits(
    monkeypatch,
) -> None:
    source = inspect.getsource(widget3d._sample_background_indices)
    helper_source = inspect.getsource(widget3d._background_indices_from_mask)

    assert "np.flatnonzero(~anomaly_mask)" not in source
    assert "_background_indices_from_mask(mask_arr, actual_background_count)" in source
    assert "out = np.empty(int(background_count), dtype=np.int64)" in helper_source

    def _fail_flatnonzero(*_args, **_kwargs):
        raise AssertionError("background sampling must direct-fill false indices")

    monkeypatch.setattr(widget3d.np, "flatnonzero", _fail_flatnonzero)
    mask = np.array([True, False, True, False, False], dtype=bool)

    background = _sample_background_indices(
        mask,
        np.array([0, 2], dtype=np.int64),
        max_count=3,
    )

    np.testing.assert_array_equal(background, [1, 3, 4])


def test_v450_sample_background_indices_reuses_anomaly_index_count(
    monkeypatch,
) -> None:
    source = inspect.getsource(widget3d._sample_background_indices)

    assert "np.count_nonzero(mask_arr)" not in source
    assert "actual_background_count = background_count" in source

    def _fail_count_nonzero(*_args, **_kwargs):
        raise AssertionError("background sampling should not rescan anomaly mask")

    monkeypatch.setattr(widget3d.np, "count_nonzero", _fail_count_nonzero)
    mask = np.array([True, False, True, False, False], dtype=bool)

    background = _sample_background_indices(
        mask,
        np.array([0, 2], dtype=np.int64),
        max_count=3,
    )

    np.testing.assert_array_equal(background, [1, 3, 4])


def test_v393_sample_true_indices_direct_fills_when_true_count_fits(
    monkeypatch,
) -> None:
    source = inspect.getsource(widget3d._sample_true_indices)
    point_source = inspect.getsource(widget3d._point_cloud_sample_indices)
    helper_source = inspect.getsource(widget3d._true_indices_from_mask)

    assert "return np.flatnonzero(mask)" not in source
    assert "np.flatnonzero(anomaly_mask)" not in point_source
    assert "_true_indices_from_mask(mask, true_count)" in source
    assert "_true_indices_from_mask(anomaly_mask, anomaly_count)" in point_source
    assert "out = np.empty(int(true_count), dtype=np.int64)" in helper_source

    def _fail_flatnonzero(*_args, **_kwargs):
        raise AssertionError("all-retained true indices must be direct-filled")

    monkeypatch.setattr(widget3d.np, "flatnonzero", _fail_flatnonzero)
    mask = np.array([True, False, True, False, True], dtype=bool)

    true_idx = _sample_true_indices(mask, true_count=3, max_count=5)
    direct_idx = _true_indices_from_mask(mask, true_count=3)

    np.testing.assert_array_equal(true_idx, [0, 2, 4])
    np.testing.assert_array_equal(direct_idx, true_idx)

    centers = np.column_stack(
        [
            np.linspace(0.0, 1.0, 12, dtype=np.float32),
            np.zeros(12, dtype=np.float32),
            np.zeros(12, dtype=np.float32),
        ]
    )
    sigma = np.ones(12, dtype=np.float32)
    sigma[[1, 9]] = 2.0

    sample_idx = _point_cloud_sample_indices(
        sigma,
        centers,
        anomaly_mode=ANOMALY_MODE_POSITIVE,
        max_points=8,
    )

    assert sample_idx.size <= 8
    assert {1, 9}.issubset(set(sample_idx.tolist()))


def test_v416_sample_true_indices_downsamples_without_chunk_flatnonzero(
    monkeypatch,
) -> None:
    source = inspect.getsource(widget3d._sample_true_indices)
    assert "local_true = np.flatnonzero(chunk)" not in source
    assert "true_seen == int(ranks[rank_pos])" in source

    def _fail_flatnonzero(*_args, **_kwargs):
        raise AssertionError("downsampled true indices must scan ranks directly")

    monkeypatch.setattr(widget3d.np, "flatnonzero", _fail_flatnonzero)
    monkeypatch.setattr(widget3d, "_POINT_CLOUD_SAMPLE_CHUNK_ITEMS", 3)
    mask = np.array(
        [True, False, True, False, True, False, False, True, True, False, False, True],
        dtype=bool,
    )

    sampled = _sample_true_indices(mask, true_count=6, max_count=3)

    np.testing.assert_array_equal(sampled, np.array([0, 4, 11], dtype=np.int64))


def test_v341_point_cloud_sampling_avoids_full_anomaly_flatnonzero(monkeypatch):
    centers = np.column_stack(
        [
            np.linspace(0.0, 1.0, 240, dtype=np.float32),
            np.zeros(240, dtype=np.float32),
            np.zeros(240, dtype=np.float32),
        ]
    )
    sigma = np.ones(240, dtype=np.float32)
    sigma[20:80] = 2.0
    monkeypatch.setattr(widget3d, "_POINT_CLOUD_SAMPLE_CHUNK_ITEMS", 32)
    original_flatnonzero = np.flatnonzero

    def _guard_flatnonzero(mask):
        arr = np.asarray(mask)
        if arr.dtype == np.bool_ and arr.size == sigma.size:
            raise AssertionError("sampling must not materialize full anomaly indices")
        return original_flatnonzero(mask)

    monkeypatch.setattr(widget3d.np, "flatnonzero", _guard_flatnonzero)

    sample_idx = _point_cloud_sample_indices(
        sigma,
        centers,
        anomaly_mode=ANOMALY_MODE_POSITIVE,
        max_points=9,
    )

    assert sample_idx.size == 9
    assert sample_idx[0] >= 20
    assert sample_idx[-1] < 80


def test_v188_pyvista_point_cloud_actor_uses_sampled_points(monkeypatch):
    _get_app()
    monkeypatch.setenv("EIT_APP_3D_POINT_CLOUD_MAX_POINTS", "5")
    widget = Conductivity3DWidget("Conductivity")
    centers = np.column_stack(
        [
            np.linspace(0.0, 1.0, 20),
            np.zeros(20),
            np.zeros(20),
        ]
    )
    sigma = np.ones(20, dtype=float)
    sigma[[3, 15]] = 2.0
    clouds: list[object] = []

    class _FakeCloud:
        def __init__(self, points) -> None:
            self.points = np.asarray(points)
            self.data = {}

        def __setitem__(self, key, value) -> None:
            self.data[key] = np.asarray(value)

    class _FakeActor:
        def SetVisibility(self, _visible):  # noqa: N802 (VTK API)
            return None

    class _FakePlotter:
        def add_mesh(self, cloud, **_kwargs):
            clouds.append(cloud)
            return _FakeActor()

    widget._add_pyvista_point_cloud_actors(
        pv=SimpleNamespace(PolyData=_FakeCloud),
        plotter=_FakePlotter(),
        centers=centers,
        cell_sigma=sigma,
        sigma_min=1.0,
        sigma_max=2.0,
        opacity=0.45,
        colorbar_label="S/m",
        colormap="viridis",
        text_color=(0.0, 0.0, 0.0),
        offscreen=True,
    )

    assert widget._point_cloud_original_count == 20
    assert widget._point_cloud_display_count <= 5
    assert clouds[0].points.shape[0] <= 5
    assert set(clouds[0].data["sigma"].tolist()).issuperset({2.0})
    widget.close()


@pytest.mark.parametrize("language", ["zh", "en"])
def test_v114_3d_controls_use_compact_two_row_layout(language: str) -> None:
    app = _get_app()
    previous_language = current_language()
    set_language(language, persist=False)
    widget = Conductivity3DWidget("Conductivity")
    try:
        widget._controls.show()
        app.processEvents()

        assert widget._controls_layout.rowCount() == 2
        assert widget._controls.minimumSizeHint().width() <= 440
        assert widget._controls.minimumSizeHint().height() <= 64
        assert (
            widget._controls_layout.getItemPosition(
                widget._controls_layout.indexOf(widget._reset_btn)
            )[0]
            == 0
        )
        for control in (
            widget._opacity_slider,
            widget._highlight_check,
            widget._wire_check,
            widget._electrode_check,
        ):
            row, _column, _row_span, _col_span = (
                widget._controls_layout.getItemPosition(
                    widget._controls_layout.indexOf(control)
                )
            )
            assert row == 1
        assert widget._highlight_check.toolTip()
        assert widget._wire_check.toolTip()
        assert widget._electrode_check.toolTip()
        assert widget._reset_btn.toolTip()
    finally:
        widget.close()
        set_language(previous_language, persist=False)


def test_v628_3d_widget_does_not_create_matplotlib_canvas():
    _get_app()
    widget = Conductivity3DWidget("Conductivity")
    try:
        assert widget._mpl3d_host is None
        assert widget._mpl3d_canvas is None
        for index in range(widget._stack.count()):
            assert widget._stack.widget(index) is not widget._mpl3d_host
    finally:
        widget.close()


def test_v628_matplotlib_3d_scene_entry_is_disabled():
    _get_app()
    widget = Conductivity3DWidget("Conductivity")
    widget.set_display_mode(DISPLAY_MODE_POINTS)
    try:
        sigma, coords, cells = _inhomogeneous_tetra_payload()
        widget._render_matplotlib_scene(sigma, coords, cells)

        assert widget._render_backend == "caption"
        assert widget._stack.currentWidget() is widget._caption_label
        assert "Matplotlib 3D rendering is disabled" in widget._caption_label.text()
    finally:
        widget.close()
