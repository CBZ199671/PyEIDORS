from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from eit_app.controllers import reconstruction_controller as rc
from eit_app.models.frame_model import FrameData
from eit_app.ui.boundary_voltage_plot_widget import BoundaryVoltagePlotWidget
from eit_app.ui.hardware.equipotential_plot_widget import EquipotentialPlotWidget
from eit_app.ui.hardware.live_plot_widget import LivePlotWidget
from eit_app.ui.hardware.reconstruction_widget import ReconstructionWidget
import eit_app.ui.simulation.metrics_panel as metrics_module
from eit_app.ui.simulation.metrics_panel import MetricsPanel

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _get_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_v255_array_pair_hash_streams_payloads_without_local_tobytes() -> None:
    coords = np.arange(24, dtype=np.float32).reshape(8, 3)[::2]
    cells = np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int32)
    expected = hashlib.sha256()
    for array in (coords, cells):
        arr = np.ascontiguousarray(np.asarray(array))
        expected.update(str(arr.dtype).encode("utf-8"))
        expected.update(str(arr.shape).encode("utf-8"))
        expected.update(arr.tobytes())

    assert rc._array_pair_hash(coords, cells) == expected.hexdigest()
    source = inspect.getsource(rc._array_pair_hash)
    assert ".tobytes(" not in source
    assert "ascontiguousarray" not in source
    assert "update_digest_with_array_payload" in source


def test_reconstruction_widget_can_replace_colorbar_repeatedly() -> None:
    _get_app()
    widget = ReconstructionWidget()
    result = SimpleNamespace(
        error_msg=None,
        conductivity=np.array([1.0], dtype=float),
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float),
        cell_connectivity=np.array([[0, 1, 2]], dtype=int),
        measured=None,
        simulated=None,
    )

    widget.update_reconstruction(result)
    widget.update_reconstruction(result)
    widget.clear()
    widget.clear()


def test_v356_equipotential_widget_passes_float32_to_render(monkeypatch) -> None:
    _get_app()
    widget = EquipotentialPlotWidget()
    captured: dict[str, np.dtype] = {}

    def _fake_render_pyvista(node_values, coords, cells):
        captured["node_values"] = np.asarray(node_values).dtype
        captured["coords"] = np.asarray(coords).dtype
        captured["cells"] = np.asarray(cells).dtype
        return True

    monkeypatch.setattr(widget, "_render_pyvista", _fake_render_pyvista)
    result = SimpleNamespace(
        error_msg=None,
        conductivity=np.array([1.0], dtype=np.float32),
        node_coords=np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        ),
        cell_connectivity=np.array([[0, 1, 2]], dtype=np.int32),
    )

    try:
        widget.update_reconstruction(result)

        assert captured == {
            "node_values": np.dtype(np.float32),
            "coords": np.dtype(np.float32),
            "cells": np.dtype(np.int32),
        }
    finally:
        widget.close()


def test_v357_reconstruction_widget_preserves_float32_payload(monkeypatch) -> None:
    _get_app()
    widget = ReconstructionWidget()
    captured: dict[str, np.dtype] = {}

    def _fake_prepare_static(coords, cells, n_elec, electrode_coverage):
        del n_elec, electrode_coverage
        captured["coords"] = np.asarray(coords).dtype
        captured["cells"] = np.asarray(cells).dtype

    def _fake_prepare_grid(coords):
        captured["grid_coords"] = np.asarray(coords).dtype

    def _fake_interpolate(node_values):
        captured["node_values"] = np.asarray(node_values).dtype
        return np.zeros((2, 2, 4), dtype=np.ubyte)

    monkeypatch.setattr(widget, "_prepare_static_scene", _fake_prepare_static)
    monkeypatch.setattr(widget, "_prepare_grid_cache", _fake_prepare_grid)
    monkeypatch.setattr(widget, "_interpolate_to_rgba", _fake_interpolate)
    result = SimpleNamespace(
        error_msg=None,
        conductivity=np.array([1.0, 2.0], dtype=np.float32),
        node_coords=np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        ),
        cell_connectivity=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        metadata={},
    )

    try:
        widget.update_reconstruction(result)

        assert captured == {
            "coords": np.dtype(np.float32),
            "cells": np.dtype(np.int32),
            "grid_coords": np.dtype(np.float32),
            "node_values": np.dtype(np.float32),
        }
    finally:
        widget.close()


def test_v268_reconstruction_widget_grid_cache_direct_fills_sample_points() -> None:
    source = inspect.getsource(ReconstructionWidget._prepare_grid_cache)

    assert "np.meshgrid" not in source
    assert "np.tile" not in source
    assert "np.repeat" not in source
    assert "np.column_stack" not in source
    assert "np.copyto" in source

    _get_app()
    widget = ReconstructionWidget()
    widget._GRID_SIZE = 4
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    widget._prepare_grid_cache(coords)

    assert widget._grid_shape == (4, 4)
    assert widget._grid_vertices.shape == (16, 3)
    assert widget._grid_weights.shape == (16, 3)
    assert widget._grid_valid_mask.shape == (16,)


def test_v442_reconstruction_widget_grid_cache_direct_fills_barycentric_arrays() -> (
    None
):
    source = inspect.getsource(ReconstructionWidget._prepare_grid_cache)

    assert "simplex[valid_mask]" not in source
    assert "sample_points[valid_mask]" not in source
    assert "vertices[valid_mask]" not in source
    assert "weights[valid_mask" not in source
    assert "safe_simplex" in source
    assert "np.take(delaunay.simplices, safe_simplex, axis=0, out=vertices)" in source
    assert "np.copyto(weights, 0.0, where=invalid_mask[:, None])" in source

    _get_app()
    widget = ReconstructionWidget()
    widget._GRID_SIZE = 4
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    widget._prepare_grid_cache(coords)

    assert widget._grid_vertices is not None
    assert widget._grid_weights is not None
    assert widget._grid_valid_mask is not None
    assert widget._grid_invalid_mask is not None
    assert widget._grid_vertices.shape == (16, 3)
    assert widget._grid_weights.shape == (16, 3)
    assert np.all(widget._grid_vertices[widget._grid_invalid_mask] == 0)
    np.testing.assert_allclose(
        widget._grid_weights[widget._grid_valid_mask].sum(axis=1),
        np.ones(int(np.count_nonzero(widget._grid_valid_mask))),
    )
    np.testing.assert_allclose(widget._grid_weights[widget._grid_invalid_mask], 0.0)


def test_v441_reconstruction_widget_interpolation_reuses_grid_buffers() -> None:
    source = inspect.getsource(ReconstructionWidget._interpolate_to_rgba)

    assert "self._grid_vertices[self._grid_valid_mask]" not in source
    assert "self._grid_weights[self._grid_valid_mask]" not in source
    assert "interpolated[self._grid_valid_mask]" not in source
    assert "rgba[~self._grid_valid_mask" not in source
    assert "np.take(node_values, self._grid_vertices, out=sample_values)" in source
    assert 'casting="same_kind"' in source
    assert "np.copyto(rgba[..., 3].reshape(-1), 0, where=invalid_mask)" in source
    assert "dtype=np.float64" not in source

    _get_app()
    widget = ReconstructionWidget()
    widget._grid_vertices = np.array([[0, 1, 2], [0, 0, 0]], dtype=np.int32)
    widget._grid_weights = np.array(
        [[0.2, 0.3, 0.5], [0.0, 0.0, 0.0]], dtype=np.float64
    )
    widget._grid_valid_mask = np.array([True, False], dtype=bool)
    widget._grid_invalid_mask = np.array([False, True], dtype=bool)
    widget._grid_shape = (1, 2)

    rgba = widget._interpolate_to_rgba(np.array([1.0, 3.0, 5.0], dtype=np.float32))

    assert rgba is not None
    assert rgba.shape == (1, 2, 4)
    assert rgba[0, 0, 3] == 255
    assert rgba[0, 1, 3] == 0
    assert widget._grid_sample_values is not None
    assert widget._grid_sample_values.dtype == np.dtype(np.float32)
    assert widget._grid_interpolated is not None
    assert widget._grid_interpolated.dtype == np.dtype(np.float32)
    assert widget._grid_abs_values is not None
    assert widget._grid_abs_values.dtype == np.dtype(np.float32)
    assert widget._grid_normalized is not None
    assert widget._grid_normalized.dtype == np.dtype(np.float32)
    first_sample_buffer = widget._grid_sample_values
    first_interpolated_buffer = widget._grid_interpolated

    rgba_second = widget._interpolate_to_rgba(
        np.array([2.0, 4.0, 6.0], dtype=np.float32)
    )
    assert rgba_second is not None
    assert widget._grid_sample_values is first_sample_buffer
    assert widget._grid_interpolated is first_interpolated_buffer


def test_v566_reconstruction_widget_defers_grid_work_buffers_until_dtype_known() -> (
    None
):
    source = inspect.getsource(ReconstructionWidget._prepare_grid_cache)
    interp_source = inspect.getsource(ReconstructionWidget._interpolate_to_rgba)

    assert (
        "self._grid_interpolated = np.empty(vertices.shape[0], dtype=np.float64)"
        not in source
    )
    assert "self._grid_interpolated = None" in source
    assert "dtype=node_values.dtype" in interp_source

    _get_app()
    widget = ReconstructionWidget()
    widget._GRID_SIZE = 4
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    widget._prepare_grid_cache(coords)

    assert widget._grid_interpolated is None
    assert widget._grid_abs_values is None
    assert widget._grid_normalized is None

    rgba = widget._interpolate_to_rgba(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    assert rgba is not None
    assert widget._grid_interpolated is not None
    assert widget._grid_interpolated.dtype == np.dtype(np.float32)
    assert widget._grid_abs_values is not None
    assert widget._grid_abs_values.dtype == np.dtype(np.float32)
    assert widget._grid_normalized is not None
    assert widget._grid_normalized.dtype == np.dtype(np.float32)


def test_v210_rm_artifact_geometry_preserves_float32_coords() -> None:
    node_coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    cells = np.array([[0, 1, 2]], dtype=np.int32)

    coords_out, cells_out = rc._rm_artifact_geometry(
        {
            "node_coords": node_coords,
            "cell_connectivity": cells,
        },
        {},
    )

    assert coords_out.dtype == np.dtype(np.float32)
    assert cells_out.dtype == np.dtype(np.int32)
    np.testing.assert_allclose(coords_out, node_coords)
    np.testing.assert_array_equal(cells_out, cells)


def test_metrics_panel_compares_values_by_geometry_not_cell_order() -> None:
    _get_app()
    _get_app()
    panel = MetricsPanel()
    node_coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    truth_cells = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    recon_cells = np.array([[1, 3, 2], [0, 1, 2]], dtype=int)

    panel.update_metrics(
        np.array([1.0, 2.0], dtype=float),
        np.array([2.0, 1.0], dtype=float),
        ground_truth_node_coords=node_coords,
        ground_truth_cell_connectivity=truth_cells,
        reconstructed_node_coords=node_coords,
        reconstructed_cell_connectivity=recon_cells,
    )

    assert panel._l2_label.text() == "0.0000"
    assert panel._corr_label.text() == "1.0000"
    assert panel._rmse_label.text() == "0.000000"


def test_v197_metrics_panel_same_geometry_skips_nearest_resample(
    monkeypatch,
) -> None:
    _get_app()
    panel = MetricsPanel()
    node_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    cells = np.array([[0, 1, 2, 3]], dtype=np.int32)

    def _fail_resample(*_args, **_kwargs):
        raise AssertionError("same mesh metrics must not build nearest-neighbor index")

    monkeypatch.setattr(metrics_module, "_nearest_resample", _fail_resample)

    panel.update_metrics(
        np.array([1.0], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
        ground_truth_node_coords=node_coords,
        ground_truth_cell_connectivity=cells,
        reconstructed_node_coords=node_coords.copy(),
        reconstructed_cell_connectivity=cells.copy(),
    )

    assert panel._l2_label.text() == "0.0000"
    assert panel._corr_label.text() == "0.0000"
    assert panel._rmse_label.text() == "0.000000"


def test_v197_metrics_samples_preserve_float32_int32_geometry_dtype() -> None:
    node_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    cells = np.array([[0, 1, 2, 3]], dtype=np.int32)

    positions, values = metrics_module._metric_samples(
        np.array([1.0], dtype=np.float32),
        node_coords=node_coords,
        cell_connectivity=cells,
    )

    assert positions.dtype == np.float32
    assert values.dtype == np.float32


def test_v227_metrics_nearest_fallback_streams_targets() -> None:
    resample_source = inspect.getsource(metrics_module._nearest_resample)
    fallback_source = inspect.getsource(metrics_module._nearest_indices_bruteforce)

    assert "_nearest_indices_bruteforce" in resample_source
    assert "target_pos[target_finite, None" not in resample_source
    assert "valid_source_pos[None" not in resample_source
    assert "[:, None" not in fallback_source
    assert "[None, :" not in fallback_source

    source = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
        ],
        dtype=np.float32,
    )
    target = np.array(
        [
            [9.0, 1.0, 0.0],
            [1.0, 9.0, 0.0],
        ],
        dtype=np.float32,
    )
    idx = metrics_module._nearest_indices_bruteforce(source, target)
    np.testing.assert_array_equal(idx, [1, 2])


def test_v377_metrics_bruteforce_nearest_preserves_float32_work_buffers(
    monkeypatch,
) -> None:
    fallback_source = inspect.getsource(metrics_module._nearest_indices_bruteforce)

    assert "dtype=np.float64" not in fallback_source
    assert "work_dtype = np.result_type" in fallback_source

    source = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
        ],
        dtype=np.float32,
    )
    target = np.array(
        [
            [9.0, 1.0],
            [1.0, 9.0],
        ],
        dtype=np.float32,
    )
    original_empty = metrics_module.np.empty
    empty_calls: list[tuple[int | tuple[int, ...], np.dtype]] = []

    def _capture_empty(shape, *args, dtype=None, **kwargs):
        empty_calls.append((shape, np.dtype(dtype)))
        return original_empty(shape, *args, dtype=dtype, **kwargs)

    monkeypatch.setattr(metrics_module.np, "empty", _capture_empty)

    idx = metrics_module._nearest_indices_bruteforce(source, target)

    np.testing.assert_array_equal(idx, [1, 2])
    work_dtypes = [dtype for shape, dtype in empty_calls if shape == source.shape[0]]
    assert work_dtypes == [np.dtype(np.float32), np.dtype(np.float32)]

    empty_calls.clear()
    idx = metrics_module._nearest_indices_bruteforce(source, target.astype(np.float64))

    np.testing.assert_array_equal(idx, [1, 2])
    work_dtypes = [dtype for shape, dtype in empty_calls if shape == source.shape[0]]
    assert work_dtypes == [np.dtype(np.float64), np.dtype(np.float64)]


def test_v358_metrics_finite_pair_stats_stream_without_subset_copies(
    monkeypatch,
) -> None:
    update_source = inspect.getsource(MetricsPanel.update_metrics)
    helper_source = inspect.getsource(metrics_module._finite_pair_stats)

    assert "gt[finite]" not in update_source
    assert "rc[finite]" not in update_source
    assert "np.corrcoef" not in update_source
    assert "_finite_pair_stats(gt, rc)" in update_source
    assert "ground_truth[finite]" not in helper_source
    assert "reconstructed[finite]" not in helper_source

    monkeypatch.setattr(metrics_module, "_METRIC_SCAN_CHUNK_ITEMS", 2)
    truth = np.array([1.0, np.nan, 2.0, 4.0, np.inf], dtype=np.float32)
    recon = np.array([1.5, 3.0, np.inf, 1.0, 2.0], dtype=np.float32)
    finite = np.isfinite(truth) & np.isfinite(recon)
    gt = truth[finite]
    rc = recon[finite]

    (
        count,
        diff_sq_sum,
        gt_sq_sum,
        gt_sum,
        rc_sum,
        rc_sq_sum,
        cross_sum,
    ) = metrics_module._finite_pair_stats(truth, recon)

    assert count == int(np.count_nonzero(finite))
    assert diff_sq_sum == pytest.approx(float(np.sum((gt - rc) ** 2)))
    assert gt_sq_sum == pytest.approx(float(np.sum(gt**2)))
    assert gt_sum == pytest.approx(float(np.sum(gt)))
    assert rc_sum == pytest.approx(float(np.sum(rc)))
    assert rc_sq_sum == pytest.approx(float(np.sum(rc**2)))
    assert cross_sum == pytest.approx(float(np.sum(gt * rc)))

    _get_app()
    panel = MetricsPanel()
    try:
        panel.update_metrics(truth, recon)

        assert panel._l2_label.text() == "0.7376"
        assert panel._corr_label.text() == "-1.0000"
        assert panel._rmse_label.text() == "2.150581"
    finally:
        panel.close()


def test_v436_metrics_finite_pair_stats_reuses_chunk_bool_buffers(
    monkeypatch,
) -> None:
    helper_source = inspect.getsource(metrics_module._finite_pair_stats)

    assert "np.isfinite(gt_chunk) & np.isfinite(rc_chunk)" not in helper_source
    assert "finite = np.empty(work_size, dtype=bool)" in helper_source
    assert "finite_work = np.empty(work_size, dtype=bool)" in helper_source
    assert "np.isfinite(gt_chunk, out=finite_chunk)" in helper_source
    assert "np.isfinite(rc_chunk, out=finite_work_chunk)" in helper_source
    assert "np.logical_and(finite_chunk, finite_work_chunk, out=finite_chunk)" in (
        helper_source
    )

    monkeypatch.setattr(metrics_module, "_METRIC_SCAN_CHUNK_ITEMS", 2)
    original_isfinite = metrics_module.np.isfinite
    out_base_ids: list[int] = []

    def _record_isfinite(values, *args, out=None, **kwargs):
        if out is not None:
            base = out.base if out.base is not None else out
            out_base_ids.append(id(base))
        return original_isfinite(values, *args, out=out, **kwargs)

    monkeypatch.setattr(metrics_module.np, "isfinite", _record_isfinite)
    truth = np.array([1.0, np.nan, 2.0, 4.0, np.inf], dtype=np.float32)
    recon = np.array([1.5, 3.0, np.inf, 1.0, 2.0], dtype=np.float32)

    stats = metrics_module._finite_pair_stats(truth, recon)

    assert stats[0] == 2
    assert len(set(out_base_ids)) == 2


def test_v359_metrics_nearest_resample_uses_all_finite_fast_path(
    monkeypatch,
) -> None:
    resample_source = inspect.getsource(metrics_module._nearest_resample)
    mask_source = inspect.getsource(metrics_module._finite_row_mask_or_none)

    assert "source_finite =" not in resample_source
    assert "target_finite =" not in resample_source
    assert "target_pos[target_finite]" not in resample_source
    assert "_finite_row_mask_or_none(source_pos, source_values)" in resample_source
    assert "_finite_row_mask_or_none(target_pos)" in resample_source
    assert "return None" in mask_source

    monkeypatch.setattr(metrics_module, "_METRIC_SCAN_CHUNK_ITEMS", 2)
    source = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
        ],
        dtype=np.float32,
    )
    target = np.array(
        [
            [9.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    values = np.array([1.0, 2.0], dtype=np.float32)

    assert metrics_module._finite_row_mask_or_none(source, values) is None
    mapped = metrics_module._nearest_resample(source, values, target)

    assert mapped is not None
    assert mapped.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(mapped, np.array([2.0, 1.0], dtype=np.float32))

    dirty_source = source.copy()
    dirty_source[0, 0] = np.nan
    source_mask = metrics_module._finite_row_mask_or_none(dirty_source, values)

    assert source_mask is not None
    assert source_mask.tolist() == [False, True]


def test_v435_metrics_finite_row_scan_reuses_1d_chunk_buffers(monkeypatch) -> None:
    mask_source = inspect.getsource(metrics_module._finite_row_mask_or_none)
    chunk_source = inspect.getsource(metrics_module._finite_row_chunk_mask)

    assert "np.isfinite(pos[start:stop]).all(axis=1)" not in mask_source
    assert "np.isfinite(pos[tail_start:tail_stop]).all(axis=1)" not in mask_source
    assert "row_work = np.empty" in mask_source
    assert "axis_work = np.empty" in mask_source
    assert "np.isfinite(pos[start:stop, axis], out=work)" in chunk_source
    assert "np.logical_and(out, work, out=out)" in chunk_source

    monkeypatch.setattr(metrics_module, "_METRIC_SCAN_CHUNK_ITEMS", 2)
    original_chunk_mask = metrics_module._finite_row_chunk_mask
    out_ids: list[int] = []
    work_ids: list[int] = []

    def _record_chunk_mask(*args, out, work, **kwargs):
        out_ids.append(id(out.base if out.base is not None else out))
        work_ids.append(id(work.base if work.base is not None else work))
        return original_chunk_mask(*args, out=out, work=work, **kwargs)

    monkeypatch.setattr(metrics_module, "_finite_row_chunk_mask", _record_chunk_mask)
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [np.nan, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, np.inf, 4.0],
        ],
        dtype=np.float32,
    )
    values = np.array([1.0, 2.0, 3.0, np.nan, 5.0], dtype=np.float32)

    mask = metrics_module._finite_row_mask_or_none(positions, values)

    assert mask is not None
    assert mask.tolist() == [True, True, False, False, False]
    assert len(set(out_ids)) == 1
    assert len(set(work_ids)) == 1


def test_v376_metrics_nearest_resample_direct_fills_all_finite_output(
    monkeypatch,
) -> None:
    resample_source = inspect.getsource(metrics_module._nearest_resample)

    assert "np.take(valid_source_values, idx_arr, out=mapped)" in resample_source
    assert "mapped[:] = mapped_values" not in resample_source
    assert "valid_source_values[np.asarray(idx" not in resample_source

    source = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    target = np.array([[1.9], [0.1]], dtype=np.float32)
    values = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    original_take = metrics_module.np.take
    take_calls: list[tuple[bool, np.dtype]] = []

    def _capture_take(a, indices, *args, out=None, **kwargs):
        if np.asarray(a).shape == values.shape:
            take_calls.append((out is not None, np.asarray(a).dtype))
        return original_take(a, indices, *args, out=out, **kwargs)

    monkeypatch.setattr(metrics_module.np, "take", _capture_take)

    mapped = metrics_module._nearest_resample(source, values, target)

    assert mapped is not None
    assert mapped.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(mapped, np.array([30.0, 10.0], dtype=np.float32))
    assert take_calls == [(True, np.dtype(np.float32))]


def test_v404_metrics_nearest_resample_direct_fills_masked_targets(
    monkeypatch,
) -> None:
    resample_source = inspect.getsource(metrics_module._nearest_resample)
    helper_source = inspect.getsource(metrics_module._fill_masked_resample_values)

    assert "mapped_values =" not in resample_source
    assert "mapped[target_mask]" not in resample_source
    assert "_fill_masked_resample_values(" in resample_source
    assert "for target_idx, is_valid in enumerate" in helper_source

    source = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    target = np.array([[1.9], [np.nan], [0.1]], dtype=np.float32)
    values = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    original_take = metrics_module.np.take

    def _fail_take(a, *_args, **_kwargs):
        if np.asarray(a).shape == values.shape:
            raise AssertionError("masked target resample must direct-fill output")
        return original_take(a, *_args, **_kwargs)

    monkeypatch.setattr(metrics_module.np, "take", _fail_take)

    mapped = metrics_module._nearest_resample(source, values, target)

    assert mapped is not None
    assert mapped.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(
        mapped,
        np.array([30.0, np.nan, 10.0], dtype=np.float32),
    )


def test_v434_metrics_nearest_resample_compacts_masked_rows_by_direct_fill(
    monkeypatch,
) -> None:
    resample_source = inspect.getsource(metrics_module._nearest_resample)
    source_helper = inspect.getsource(metrics_module._compact_source_samples)
    row_helper = inspect.getsource(metrics_module._compact_rows_by_mask)

    assert "source_pos[source_mask]" not in resample_source
    assert "source_values[source_mask]" not in resample_source
    assert "target_pos[target_mask]" not in resample_source
    assert "_compact_source_samples(" in resample_source
    assert "_compact_rows_by_mask(target_pos, target_mask)" in resample_source
    assert "[mask_arr]" not in source_helper
    assert "[mask_arr]" not in row_helper

    original_source_compact = metrics_module._compact_source_samples
    original_row_compact = metrics_module._compact_rows_by_mask
    source_compacts: list[tuple[tuple[int, ...], np.dtype]] = []
    row_compacts: list[tuple[tuple[int, ...], np.dtype]] = []

    def _record_source_compact(positions, values, mask, *, value_dtype):
        out_positions, out_values = original_source_compact(
            positions,
            values,
            mask,
            value_dtype=value_dtype,
        )
        source_compacts.append((out_positions.shape, out_values.dtype))
        return out_positions, out_values

    def _record_row_compact(rows, mask):
        out = original_row_compact(rows, mask)
        row_compacts.append((out.shape, out.dtype))
        return out

    monkeypatch.setattr(
        metrics_module,
        "_compact_source_samples",
        _record_source_compact,
    )
    monkeypatch.setattr(metrics_module, "_compact_rows_by_mask", _record_row_compact)

    source = np.array([[0.0], [1.0], [np.nan], [2.0]], dtype=np.float32)
    target = np.array([[1.9], [np.nan], [0.1]], dtype=np.float32)
    values = np.array([10.0, 20.0, 40.0, 30.0], dtype=np.float32)

    mapped = metrics_module._nearest_resample(source, values, target)

    assert source_compacts == [((3, 1), np.dtype(np.float32))]
    assert row_compacts == [((2, 1), np.dtype(np.float32))]
    assert mapped is not None
    assert mapped.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(
        mapped,
        np.array([30.0, np.nan, 10.0], dtype=np.float32),
    )


def test_mesh_loader_default_mesh_skips_incompatible_3d_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyeidors.geometry.mesh_loader import MeshLoader

    (tmp_path / "mesh3d_first.msh").write_text("3d", encoding="utf-8")
    (tmp_path / "mesh_2d_second.msh").write_text("2d", encoding="utf-8")

    loader = MeshLoader(mesh_dir=str(tmp_path), gdim=2)
    sentinel = object()

    def _fake_load_mesh(name: str):
        if name == "mesh3d_first":
            raise ValueError(
                "Topological dimension cannot be larger than geometric dimension."
            )
        if name == "mesh_2d_second":
            return sentinel
        raise AssertionError(f"Unexpected mesh candidate: {name}")

    monkeypatch.setattr(loader, "load_mesh", _fake_load_mesh)

    assert loader.get_default_mesh() is sentinel


def _make_frame(index: int) -> FrameData:
    return FrameData(
        real=np.array([1.0, 2.0, 3.0], dtype=float),
        imag=np.array([0.1, 0.2, 0.3], dtype=float),
        timestamp=0.0,
        frame_index=index,
    )


def test_effective_refinement_accepts_simulation_mesh_size_without_inflation() -> None:
    assert rc._compute_effective_refinement(1.0, 0.1) == 5
    assert rc._compute_effective_refinement(1.0, 10.0) == 20
    assert rc._compute_effective_refinement(1.0, 10.0, mesh_size=0.1) == 5


def test_v106_default_3d_rm_inverse_mesh_size_stays_coarse() -> None:
    size = rc.default_rm_inverse_mesh_size(0.1, 0.18, mesh_dimension=3)

    assert size == pytest.approx(0.1)
    assert rc._compute_effective_refinement(0.18, 0.1, mesh_size=size) == 2
    assert rc.default_rm_inverse_mesh_size(
        0.02, 0.18, mesh_dimension=3
    ) == pytest.approx(0.06)


def test_single_step_cached_runtime_uses_3d_multiring_fast_defaults() -> None:
    request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        mesh_dimension=3,
        mesh_refinement=0.1,
        metadata={
            "mesh_dimension": 3,
            "mesh_size": 0.1,
            "n_elec": 8,
            "n_rings": 2,
            "drive_mode": "line_current_density",
        },
    )

    runtime = rc._prepare_single_step_cached_runtime(request)

    assert rc._total_electrodes_from_meta(runtime.meta) == 16
    assert runtime.meta["drive_mode"] == "total_current"
    assert runtime.meta["solver_mode"] == "fast"
    assert runtime.meta["forward_mat_solve"] == "auto"
    assert runtime.meta["mesh_family"] == "tetra"
    assert runtime.meta["jacobian_representation"] == "linearized"
    assert runtime.refinement == 5


def test_single_step_cached_runtime_keeps_large_3d_auto_on_dense_jacobian() -> None:
    large_ref = FrameData(
        real=np.ones(5936, dtype=float),
        imag=np.zeros(5936, dtype=float),
        timestamp=0.0,
        frame_index=0,
    )
    large_tgt = FrameData(
        real=np.ones(5936, dtype=float) * 1.001,
        imag=np.zeros(5936, dtype=float),
        timestamp=0.0,
        frame_index=1,
    )
    request = rc.ReconstructionRequest(
        reference_frame=large_ref,
        target_frame=large_tgt,
        mesh_dimension=3,
        mesh_refinement=0.1,
        metadata={
            "mesh_dimension": 3,
            "mesh_size": 0.1,
            "n_elec": 16,
            "n_rings": 3,
            "jacobian_representation": "auto",
        },
    )

    runtime = rc._prepare_single_step_cached_runtime(request)

    assert runtime.meta["solver_mode"] == "fast"
    assert runtime.meta["jacobian_representation"] == "dense"
    assert runtime.meta["jacobian_representation_reason"] == "auto_dense_large_or_non3d"


def test_single_step_cached_runtime_uses_request_alpha_when_lambda_is_absent() -> None:
    request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        regularization_alpha=0.75,
        metadata={"reconstruction_runtime": "single_step_cached"},
    )

    runtime = rc._prepare_single_step_cached_runtime(request)

    assert runtime.lam == pytest.approx(0.75)
    assert runtime.meta["difference_lambda"] == pytest.approx(0.75)
    assert runtime.meta["jacobian_representation"] == "dense"


def test_single_step_cached_runtime_prefers_explicit_difference_lambda() -> None:
    request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        regularization_alpha=0.75,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "difference_lambda": 0.02,
        },
    )

    runtime = rc._prepare_single_step_cached_runtime(request)

    assert runtime.lam == pytest.approx(0.02)


def test_single_step_cached_runtime_keys_semantics_with_version_fallback() -> None:
    default_request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={"reconstruction_runtime": "single_step_cached"},
    )
    semantic_request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "single_step_projection_math_convention": "test-projection-v2",
        },
    )
    version_request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "single_step_algorithm_version": "test-local-version",
        },
    )

    default_runtime = rc._prepare_single_step_cached_runtime(default_request)
    semantic_runtime = rc._prepare_single_step_cached_runtime(semantic_request)
    version_runtime = rc._prepare_single_step_cached_runtime(version_request)
    semantic_signature = default_runtime.cache_key[0]

    assert isinstance(semantic_signature, tuple)
    assert default_runtime.meta["single_step_jacobian_calculator"] in semantic_signature
    assert (
        default_runtime.meta["single_step_jacobian_math_convention"]
        in semantic_signature
    )
    assert (
        default_runtime.meta["single_step_projection_math_convention"]
        in semantic_signature
    )
    assert (
        default_runtime.meta["single_step_operator_math_convention"]
        in semantic_signature
    )
    assert (
        default_runtime.meta["single_step_algorithm_version"]
        in default_runtime.cache_key
    )
    assert default_runtime.cache_key != semantic_runtime.cache_key
    assert default_runtime.cache_key != version_runtime.cache_key


def test_one_step_rm_signature_rejects_stale_normalized_jacobian_semantics() -> None:
    from scripts.common import gn_difference_runner

    assert (
        gn_difference_runner.SINGLE_STEP_JACOBIAN_MATH_CONVENTION
        == rc._SINGLE_STEP_JACOBIAN_MATH_CONVENTION
    )
    assert (
        gn_difference_runner.SINGLE_STEP_PROJECTION_MATH_CONVENTION
        == rc._SINGLE_STEP_PROJECTION_MATH_CONVENTION
    )
    assert (
        gn_difference_runner.SINGLE_STEP_ALGORITHM_VERSION
        == rc._SINGLE_STEP_CACHED_ALGORITHM_VERSION
    )

    base_meta = {
        "reconstruction_runtime": "single_step_cached",
        "simulation_inverse_route": "noser_rm",
        "rm_auto_build": True,
        "mesh_size": 0.1,
        "rm_inverse_mesh_size": 0.1,
        "difference_mode": "normalized",
        "difference_orientation": "target_minus_reference",
        "n_elec": 16,
        "radius": 1.0,
    }
    current_request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata=base_meta,
    )
    stale_request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={
            **base_meta,
            "single_step_jacobian_math_convention": (
                "eidors_adapter_difference_dv_dsigma_v3"
            ),
            "single_step_projection_math_convention": (
                "difference_projection_weights_v2"
            ),
            "single_step_algorithm_version": "eidors_noser_single_step_v4",
            "one_step_rm_jacobian_build_convention": (
                "dense_eidors_adapter_jacobian_v2"
            ),
            "one_step_rm_algorithm_version": (
                "one_step_rm_auto_build_dense_jacobian_v6"
            ),
            "one_step_rm_content_contract": (
                "one_step_rm_hdf5_dense_fit_jacobian_contract_v0"
            ),
            "single_step_context_cache_scope": "both",
        },
    )

    current_runtime = rc._prepare_single_step_cached_runtime(current_request)
    stale_runtime = rc._prepare_single_step_cached_runtime(stale_request)
    current_signature, current_payload = rc._planned_one_step_rm_signature(
        current_request,
        current_runtime,
    )
    stale_signature, stale_payload = rc._planned_one_step_rm_signature(
        stale_request,
        stale_runtime,
    )

    assert current_runtime.cache_key != stale_runtime.cache_key
    assert current_signature != stale_signature
    assert current_runtime.meta["single_step_context_cache_scope"] == "process"
    assert (
        current_payload["hyperparameters"]["rm_jacobian_math_convention"]
        == rc._SINGLE_STEP_JACOBIAN_MATH_CONVENTION
    )
    assert (
        current_payload["hyperparameters"]["rm_projection_math_convention"]
        == rc._SINGLE_STEP_PROJECTION_MATH_CONVENTION
    )
    assert (
        current_payload["hyperparameters"]["rm_jacobian_build_convention"]
        == rc._ONE_STEP_RM_JACOBIAN_BUILD_CONVENTION
    )
    assert (
        current_payload["hyperparameters"]["rm_algorithm_version"]
        == rc._ONE_STEP_RM_ALGORITHM_VERSION
    )
    assert (
        current_payload["hyperparameters"]["rm_content_contract"]
        == rc._ONE_STEP_RM_CONTENT_CONTRACT
    )
    assert (
        current_payload["hyperparameters"]["rm_jacobian_source_cache_scope"]
        == "process"
    )
    assert (
        stale_payload["hyperparameters"]["rm_content_contract"]
        != current_payload["hyperparameters"]["rm_content_contract"]
    )
    assert stale_payload["hyperparameters"]["rm_jacobian_source_cache_scope"] == "both"
    assert stale_payload["difference_mode"] == "normalized"


def test_noser_rm_signature_ignores_device_backend_storage_axes() -> None:
    base_meta = {
        "reconstruction_runtime": "single_step_cached",
        "simulation_inverse_route": "noser_rm",
        "rm_auto_build": True,
        "mesh_size": 0.25,
        "rm_inverse_mesh_size": 0.25,
        "difference_mode": "raw",
        "difference_orientation": "target_minus_reference",
        "n_elec": 16,
        "radius": 1.0,
    }
    cpu_request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={
            **base_meta,
            "device": "cpu",
            "petsc_device": "cpu",
            "forward_backend": "dolfinx",
        },
    )
    cuda_request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={
            **base_meta,
            "device": "cuda",
            "petsc_device": "cuda",
            "forward_backend": "cuda_structured",
        },
    )

    cpu_runtime = rc._prepare_single_step_cached_runtime(cpu_request)
    cuda_runtime = rc._prepare_single_step_cached_runtime(cuda_request)
    assert cpu_runtime.cache_key != cuda_runtime.cache_key

    cpu_signature, _ = rc._planned_noser_rm_signature(cpu_request, cpu_runtime)
    cuda_signature, _ = rc._planned_noser_rm_signature(cuda_request, cuda_runtime)
    assert cpu_signature == cuda_signature


def test_complex_rm_runtime_forces_dolfinx_over_real_only_cuda_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rc, "probe_petsc_cuda_runtime", lambda: {})
    request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        use_part="complex",
        mesh_dimension=3,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": "noser_rm",
            "rm_auto_build": True,
            "mesh_family": "hex",
            "acceleration_profile": "gpu3d",
            "forward_backend": "cuda_structured",
            "background_conductivity": "1+2j",
            "contact_impedance": "0.01+0.05j",
            "rm_dtype": "complex64",
        },
    )

    runtime = rc._prepare_single_step_cached_runtime(request)

    assert runtime.meta["forward_backend"] == "dolfinx"
    assert runtime.meta["petsc_device"] == "cuda"


def test_one_step_rm_signature_tracks_effective_measurement_count() -> None:
    base_meta = {
        "reconstruction_runtime": "single_step_cached",
        "simulation_inverse_route": "laplace_rm",
        "rm_auto_build": True,
        "mesh_size": 0.25,
        "rm_inverse_mesh_size": 0.25,
        "difference_mode": "raw",
        "difference_orientation": "target_minus_reference",
        "n_elec": 16,
        "n_rings": 3,
        "radius": 0.18,
        "height": 0.16,
        "rm_regularization": "laplace",
    }
    ref_2160 = FrameData(
        real=np.ones(2160, dtype=float),
        imag=np.zeros(2160, dtype=float),
        timestamp=0.0,
        frame_index=0,
    )
    tgt_2160 = FrameData(
        real=np.ones(2160, dtype=float) * 1.01,
        imag=np.zeros(2160, dtype=float),
        timestamp=0.0,
        frame_index=1,
    )
    ref_5936 = FrameData(
        real=np.ones(5936, dtype=float),
        imag=np.zeros(5936, dtype=float),
        timestamp=0.0,
        frame_index=0,
    )
    tgt_5936 = FrameData(
        real=np.ones(5936, dtype=float) * 1.01,
        imag=np.zeros(5936, dtype=float),
        timestamp=0.0,
        frame_index=1,
    )
    request_2160 = rc.ReconstructionRequest(
        reference_frame=ref_2160,
        target_frame=tgt_2160,
        mesh_dimension=3,
        metadata=base_meta,
    )
    request_5936 = rc.ReconstructionRequest(
        reference_frame=ref_5936,
        target_frame=tgt_5936,
        mesh_dimension=3,
        metadata=base_meta,
    )

    sig_2160, payload_2160 = rc._planned_one_step_rm_signature(
        request_2160,
        rc._prepare_single_step_cached_runtime(request_2160),
    )
    sig_5936, payload_5936 = rc._planned_one_step_rm_signature(
        request_5936,
        rc._prepare_single_step_cached_runtime(request_5936),
    )

    assert sig_2160 != sig_5936
    assert payload_2160["stim_meas_protocol"]["n_measurements"] == 2160
    assert payload_5936["stim_meas_protocol"]["n_measurements"] == 5936


def test_smooth_rm_signature_tracks_graph_prior_semantics_not_storage_axes() -> None:
    base_meta = {
        "reconstruction_runtime": "single_step_cached",
        "rm_auto_build": True,
        "mesh_size": 0.25,
        "rm_inverse_mesh_size": 0.25,
        "difference_mode": "raw",
        "difference_orientation": "target_minus_reference",
        "n_elec": 16,
        "radius": 1.0,
        "rm_graph_weight": "unit",
    }
    laplace_cpu = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={
            **base_meta,
            "simulation_inverse_route": "laplace_rm",
            "rm_regularization": "laplace",
            "device": "cpu",
            "petsc_device": "cpu",
        },
    )
    laplace_cuda = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={
            **base_meta,
            "simulation_inverse_route": "laplace_rm",
            "rm_regularization": "laplace",
            "device": "cuda",
            "petsc_device": "cuda",
        },
    )
    curvature = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={
            **base_meta,
            "simulation_inverse_route": "curvature_rm",
            "rm_regularization": "curvature",
            "device": "cpu",
            "petsc_device": "cpu",
        },
    )

    laplace_cpu_runtime = rc._prepare_single_step_cached_runtime(laplace_cpu)
    laplace_cuda_runtime = rc._prepare_single_step_cached_runtime(laplace_cuda)
    curvature_runtime = rc._prepare_single_step_cached_runtime(curvature)
    laplace_cpu_signature, laplace_payload = rc._planned_one_step_rm_signature(
        laplace_cpu,
        laplace_cpu_runtime,
    )
    laplace_cuda_signature, _ = rc._planned_one_step_rm_signature(
        laplace_cuda,
        laplace_cuda_runtime,
    )
    curvature_signature, curvature_payload = rc._planned_one_step_rm_signature(
        curvature,
        curvature_runtime,
    )

    assert laplace_cpu_runtime.cache_key != laplace_cuda_runtime.cache_key
    assert laplace_cpu_signature == laplace_cuda_signature
    assert laplace_cpu_signature != curvature_signature
    assert laplace_payload["regularization_type"] == "laplace"
    assert (
        laplace_payload["hyperparameters"]["prior_operator"]
        == "eidors_prior_laplace_graph_x2"
    )
    assert (
        laplace_payload["hyperparameters"]["rm_jacobian_build_representation"]
        == "dense"
    )
    assert laplace_payload["hyperparameters"]["form"] == "param"
    assert (
        laplace_payload["hyperparameters"]["singular_prior_form_policy"]
        == "param_for_graph_laplace_curvature_v1"
    )
    assert (
        laplace_payload["hyperparameters"]["rm_algorithm_version"]
        == rc._ONE_STEP_RM_ALGORITHM_VERSION
    )
    assert curvature_payload["regularization_type"] == "curvature"
    assert curvature_payload["hyperparameters"]["form"] == "param"
    assert (
        curvature_payload["hyperparameters"]["prior_operator"]
        == "eidors_prior_laplace_squared"
    )


def test_greit_center_cloud_geometry_uses_axis_spacing_not_cloud_median() -> None:
    centers = np.asarray(
        [
            [-0.75, -0.75, 0.0],
            [-0.25, -0.75, 0.0],
            [0.25, -0.75, 0.0],
            [0.75, -0.75, 0.0],
            [-0.25, -0.25, 0.0],
            [0.25, -0.25, 0.0],
            [-0.25, 0.25, 0.0],
            [0.25, 0.25, 0.0],
        ],
        dtype=float,
    )

    coords, cells = rc._center_cloud_hexa_geometry(centers, {"radius": 1.0})

    assert coords.shape == (centers.shape[0] * 8, 3)
    assert cells.shape == (centers.shape[0], 8)
    first_cell = coords[cells[0]]
    assert np.ptp(first_cell[:, 0]) == pytest.approx(0.45)
    assert np.ptp(first_cell[:, 1]) == pytest.approx(0.45)
    assert np.ptp(first_cell[:, 2]) == pytest.approx(0.45)


def test_v125_greit_2d_rec_model_geometry_uses_planar_quads() -> None:
    centers = np.asarray(
        [
            [-0.75, -0.75, 0.0],
            [-0.25, -0.75, 0.0],
            [-0.75, -0.25, 0.0],
            [-0.25, -0.25, 0.0],
        ],
        dtype=float,
    )

    coords, cells = rc._greit_rec_model_geometry(
        centers,
        n_parameters=centers.shape[0],
        meta={"mesh_dimension": 2, "radius": 1.0},
    )

    assert coords.shape == (centers.shape[0] * 4, 2)
    assert cells.shape == (centers.shape[0], 4)
    assert np.ptp(coords[cells[0]][:, 0]) == pytest.approx(0.45)
    assert np.ptp(coords[cells[0]][:, 1]) == pytest.approx(0.45)


def test_v223_center_cloud_geometry_preserves_float32_and_vector_cells() -> None:
    hexa_source = inspect.getsource(rc._center_cloud_hexa_geometry)
    quad_source = inspect.getsource(rc._center_cloud_quad_geometry)
    rec_model_source = inspect.getsource(rc._greit_rec_model_geometry)

    assert "centers[:, None" not in hexa_source
    assert "centers_xy[:, None" not in quad_source
    assert "for idx in range" not in hexa_source
    assert "for idx in range" not in quad_source
    assert "np.column_stack" not in rec_model_source
    assert (
        "padded = np.zeros((centers.shape[0], 3), dtype=np.float64)"
        not in rec_model_source
    )
    assert "np.arange" in hexa_source
    assert "np.arange" in quad_source

    centers3d = np.asarray(
        [
            [-0.75, -0.75, 0.0],
            [-0.25, -0.75, 0.0],
        ],
        dtype=np.float32,
    )
    coords3d, cells3d = rc._center_cloud_hexa_geometry(centers3d, {"radius": 1.0})
    assert coords3d.dtype == np.dtype(np.float32)
    assert cells3d.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(
        cells3d,
        np.arange(centers3d.shape[0] * 8, dtype=np.int32).reshape(-1, 8),
    )

    coords2d, cells2d = rc._greit_rec_model_geometry(
        centers3d,
        n_parameters=centers3d.shape[0],
        meta={"mesh_dimension": 2, "radius": 1.0},
    )
    assert coords2d.dtype == np.dtype(np.float32)
    assert cells2d.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(
        cells2d,
        np.arange(centers3d.shape[0] * 4, dtype=np.int32).reshape(-1, 4),
    )
    coords_from_xy, cells_from_xy = rc._greit_rec_model_geometry(
        centers3d[:, :2],
        n_parameters=centers3d.shape[0],
        meta={"mesh_dimension": 3, "radius": 1.0},
    )
    assert coords_from_xy.dtype == np.dtype(np.float32)
    assert cells_from_xy.dtype == np.dtype(np.int32)
    assert coords_from_xy.shape[1] == 3


def test_v477_reconstruction_controller_finite_and_complex_guards_are_bounded() -> None:
    fit_source = inspect.getsource(rc._fit_jacobian_array)
    contact_source = inspect.getsource(rc._contact_impedance_array)
    hexa_source = inspect.getsource(rc._center_cloud_hexa_geometry)
    quad_source = inspect.getsource(rc._center_cloud_quad_geometry)
    training_fit_source = inspect.getsource(rc._greit_training_space_fit)
    streaming_source = inspect.getsource(rc._stream_hdf5_rm_matmul)
    cached_rm_source = inspect.getsource(rc._try_run_cached_rm_request)

    assert "all_finite_values(arr)" in fit_source
    assert "has_nonzero_imaginary(" in contact_source
    assert "all_finite_values(centers)" in hexa_source
    assert "all_finite_values(centers_xy)" in quad_source
    assert "all_finite_values(fitted)" in training_fit_source
    assert "all_finite_values(values)" in streaming_source
    assert "all_finite_values(simulated_dv)" in cached_rm_source

    for source in (
        fit_source,
        contact_source,
        hexa_source,
        quad_source,
        training_fit_source,
        streaming_source,
        cached_rm_source,
    ):
        assert "np.any(np.abs(np.imag" not in source
        assert "np.isfinite(arr).all()" not in source
        assert "np.isfinite(centers).all()" not in source
        assert "np.isfinite(centers_xy).all()" not in source
        assert "np.isfinite(fitted).all()" not in source
        assert "np.isfinite(values).all()" not in source
        assert "np.isfinite(simulated_dv).all()" not in source


def test_v493_single_step_sigma_update_uses_bounded_finite_scans() -> None:
    limit_source = inspect.getsource(rc._limit_single_step_alpha_for_sigma_floor)
    constrain_source = inspect.getsource(rc._constrain_single_step_sigma_update)
    bounds_source = inspect.getsource(rc._voxel_bounds_from_meta)

    assert "all_finite_values(sigma)" in limit_source
    assert "all_finite_values(delta)" in limit_source
    assert "min_alpha_for_value_floor(sigma, delta" in limit_source
    assert "np.all(np.isfinite(sigma))" not in limit_source
    assert "np.all(np.isfinite(delta))" not in limit_source
    assert "sigma[negative_update]" not in limit_source
    assert "delta[negative_update]" not in limit_source
    assert "all_finite_values(sigma)" in constrain_source
    assert "all_finite_values(delta)" in constrain_source
    assert "all_finite_values(raw_sigma_est)" in constrain_source
    assert "any_not_equal_values(sigma_est, raw_sigma_est)" in constrain_source
    assert "np.all(np.isfinite(sigma))" not in constrain_source
    assert "np.all(np.isfinite(delta))" not in constrain_source
    assert "np.all(np.isfinite(raw_sigma_est))" not in constrain_source
    assert "np.any(sigma_est != raw_sigma_est)" not in constrain_source
    assert "all_finite_values(bounds)" in bounds_source
    assert "np.all(np.isfinite(bounds))" not in bounds_source


def test_v556_single_step_sigma_update_preserves_float32_dtype() -> None:
    limit_source = inspect.getsource(rc._limit_single_step_alpha_for_sigma_floor)
    constrain_source = inspect.getsource(rc._constrain_single_step_sigma_update)

    assert "np.asarray(sigma_bg, dtype=np.float64)" not in limit_source
    assert "np.asarray(delta_sigma, dtype=np.float64)" not in limit_source
    assert "np.asarray(sigma_bg, dtype=np.float64)" not in constrain_source
    assert "np.asarray(delta_sigma, dtype=np.float64)" not in constrain_source
    assert "_real_sigma_update_array" in limit_source
    assert "_real_sigma_update_array" in constrain_source

    sigma = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    delta = np.array([-0.5, 0.25, -0.9], dtype=np.float32)

    _alpha, display_delta, sigma_est, floor_applied = (
        rc._constrain_single_step_sigma_update(
            sigma,
            delta,
            1.0,
            sigma_floor=0.25,
        )
    )

    assert display_delta.dtype == np.dtype(np.float32)
    assert sigma_est.dtype == np.dtype(np.float32)
    assert floor_applied is False


def test_v350_center_cloud_axis_spacing_reuses_sorted_unique_diffs() -> None:
    hexa_source = inspect.getsource(rc._center_cloud_hexa_geometry)
    quad_source = inspect.getsource(rc._center_cloud_quad_geometry)

    assert "np.diff(np.sort(unique))" not in hexa_source
    assert "np.diff(np.sort(unique))" not in quad_source
    assert "diffs[np.isfinite" not in hexa_source
    assert "diffs[np.isfinite" not in quad_source
    assert "np.diff(unique)" in hexa_source
    assert "np.diff(unique)" in quad_source

    centers = np.asarray(
        [
            [-0.75, -0.75, 0.0],
            [-0.25, -0.75, 0.0],
            [0.25, -0.25, 0.0],
            [0.75, -0.25, 0.0],
        ],
        dtype=np.float32,
    )

    coords3d, cells3d = rc._center_cloud_hexa_geometry(centers, {"radius": 1.0})
    coords2d, cells2d = rc._center_cloud_quad_geometry(centers, {"radius": 1.0})

    assert coords3d.dtype == np.dtype(np.float32)
    assert coords2d.dtype == np.dtype(np.float32)
    assert cells3d.shape == (centers.shape[0], 8)
    assert cells2d.shape == (centers.shape[0], 4)


def test_run_reconstruction_request_dispatches_to_single_step_cached_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={"reconstruction_runtime": "single_step_cached"},
    )
    sentinel = rc.ReconstructionResult(
        conductivity=np.array([1.0], dtype=float),
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float),
        cell_connectivity=np.array([[0, 1, 2]], dtype=int),
        metadata={"reconstruction_runtime": "single_step_cached"},
    )

    monkeypatch.setattr(
        rc,
        "_run_single_step_cached_request",
        lambda req, progress_cb=None: sentinel,
    )

    def _unexpected_full(*_args, **_kwargs):
        raise AssertionError(
            "full GN path should not be used for realtime single-step requests"
        )

    monkeypatch.setattr(rc, "_run_full_gn_request", _unexpected_full)

    result = rc.run_reconstruction_request(request)

    assert result is sentinel


def test_run_reconstruction_request_routes_complex_to_native_full_gn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=np.array([1.0, 2.0], dtype=np.float32),
            imag=np.array([0.10, 0.20], dtype=np.float32),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=np.array([1.5, 2.5], dtype=np.float32),
            imag=np.array([0.30, 0.50], dtype=np.float32),
            timestamp=0.0,
            frame_index=1,
        ),
        use_part="complex",
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "background_conductivity": "2+0.5j",
            "contact_impedance": "0.01+0.002j",
            "compute_dtype": "complex64",
        },
    )
    captured: list[rc.ReconstructionRequest] = []
    sentinel = rc.ReconstructionResult(
        conductivity=np.array([2.0 + 0.5j], dtype=np.complex128),
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float),
        cell_connectivity=np.array([[0, 1, 2]], dtype=int),
        metadata={"complex_reconstruction_mode": "native_complex_linearized_gn"},
    )

    monkeypatch.setattr(
        rc,
        "_run_single_step_cached_request",
        lambda *_args, **_kwargs: pytest.fail(
            "complex request must not split/fast-path"
        ),
    )
    monkeypatch.setattr(
        rc,
        "_run_full_gn_request",
        lambda req, progress_cb=None: captured.append(req) or sentinel,
    )

    result = rc.run_reconstruction_request(request)

    assert result is sentinel
    assert captured == [request]
    assert captured[0].use_part == "complex"


def test_run_reconstruction_request_dispatches_complex_rm_to_single_step_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=np.array([1.0, 2.0], dtype=np.float32),
            imag=np.array([0.10, 0.20], dtype=np.float32),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=np.array([1.5, 2.5], dtype=np.float32),
            imag=np.array([0.30, 0.50], dtype=np.float32),
            timestamp=0.0,
            frame_index=1,
        ),
        use_part="complex",
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": "noser_rm",
            "rm_route_requires_artifact": True,
            "rm_auto_build": True,
            "background_conductivity": "1+2j",
            "contact_impedance": "0.01+0.05j",
            "rm_dtype": "complex64",
        },
    )
    sentinel = rc.ReconstructionResult(
        conductivity=np.array([1.0 + 2.0j], dtype=np.complex64),
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float),
        cell_connectivity=np.array([[0, 1, 2]], dtype=int),
        metadata={"reconstruction_runtime": "single_step_cached"},
    )

    monkeypatch.setattr(
        rc,
        "_run_single_step_cached_request",
        lambda req, progress_cb=None: sentinel,
    )

    def _unexpected_full(*_args, **_kwargs):
        raise AssertionError("complex RM requests must use the one-step RM path")

    monkeypatch.setattr(rc, "_run_full_gn_request", _unexpected_full)

    result = rc.run_reconstruction_request(request)

    assert result is sentinel


def test_v134_complex_rm_route_uses_one_step_rm_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    sentinel = rc.ReconstructionResult(
        conductivity=np.array([1.0 + 0.1j], dtype=np.complex64),
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float),
        cell_connectivity=np.array([[0, 1, 2]], dtype=int),
        metadata={"reconstruction_runtime": "single_step_cached"},
    )

    def _fake_single_step(req, progress_cb=None):
        captured["request"] = req
        return sentinel

    monkeypatch.setattr(
        rc,
        "_run_single_step_cached_request",
        _fake_single_step,
    )

    def _unexpected_full(*_args, **_kwargs):
        raise AssertionError("complex RM requests must use the one-step RM path")

    monkeypatch.setattr(rc, "_run_full_gn_request", _unexpected_full)
    n_meas = 208
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=np.linspace(1.0, 2.0, n_meas, dtype=np.float32),
            imag=np.linspace(0.10, 0.20, n_meas, dtype=np.float32),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=np.linspace(1.5, 2.5, n_meas, dtype=np.float32),
            imag=np.linspace(0.30, 0.50, n_meas, dtype=np.float32),
            timestamp=0.0,
            frame_index=1,
        ),
        use_part="complex",
        method="gn-difference",
        mesh_dimension=2,
        mesh_refinement=0.1,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": "noser_rm",
            "difference_preset": "noser_rm",
            "background_conductivity": "1+0.1j",
            "contact_impedance": "0.01+0.002j",
            "compute_dtype": "complex64",
        },
    )

    result = rc.run_reconstruction_request(request)

    assert result is sentinel
    assert captured["request"] is request
    assert captured["request"].use_part == "complex"
    assert captured["request"].metadata["simulation_inverse_route"] == "noser_rm"


def test_v512_native_complex_controller_identity_regularization_stays_lazy() -> None:
    source = inspect.getsource(rc._regularization_for_native_complex)

    assert "np.eye(int(n_param)" not in source
    assert rc._regularization_for_native_complex(SimpleNamespace(), 4) is None

    diag = np.array([1.0, 2.0, 3.0], dtype=float)
    resolved_diag = rc._regularization_for_native_complex(
        SimpleNamespace(R_diag=diag), 3
    )
    np.testing.assert_allclose(resolved_diag, diag)
    assert np.shares_memory(resolved_diag, diag)

    matrix = np.diag([1.0, 2.0, 3.0])
    assert (
        rc._regularization_for_native_complex(SimpleNamespace(R_matrix=matrix), 3)
        is matrix
    )


def test_native_complex_normal_step_uses_hermitian_coupled_system() -> None:
    from pyeidors.inverse.solvers.gauss_newton_linear_system import (
        solve_native_complex_normal_step,
    )

    jacobian = np.array(
        [
            [1.0 + 2.0j, 0.5 - 0.25j],
            [0.25 + 0.5j, -1.0 + 0.75j],
            [1.5 - 0.5j, 0.25 + 1.0j],
        ],
        dtype=np.complex128,
    )
    true_delta = np.array([0.2 + 0.3j, -0.1 + 0.4j], dtype=np.complex128)
    measured_diff = jacobian @ true_delta
    residual = -measured_diff

    delta, meta = solve_native_complex_normal_step(
        jacobian=jacobian,
        residual=residual,
        lambda_eff=0.0,
        regularization=np.eye(2),
    )

    assert np.allclose(delta, true_delta)
    assert meta["native_complex_linear_algebra"] is True
    assert meta["transpose"] == "hermitian_conjugate"


def test_direct_jacobian_numpy_assembly_preserves_complex_sensitivity() -> None:
    from pyeidors.inverse.jacobian._core import assemble_jacobian_efficient_numpy

    grad_u = np.array(
        [
            [1.0 + 0.5j, 2.0 - 0.25j],
            [0.5 - 1.0j, -1.0 + 0.75j],
        ],
        dtype=np.complex128,
    )
    adjoint = np.array(
        [
            [[0.25 + 1.0j, 1.5 - 0.5j], [1.0 - 0.25j, 0.5 + 0.5j]],
            [[-1.0 + 0.25j, 0.75 + 0.5j], [0.5 + 1.5j, -0.25 + 0.75j]],
        ],
        dtype=np.complex128,
    )
    areas = np.array([0.2, 0.4], dtype=np.float64)

    jacobian, _elapsed = assemble_jacobian_efficient_numpy(
        grad_u_all=[grad_u],
        adjoint_gradients=[adjoint[0], adjoint[1]],
        cell_areas=areas,
        n_meas_per_stim=[2],
        block_size=1,
    )
    expected = np.einsum("eg,meg->me", grad_u, adjoint, optimize=True) * areas

    assert np.iscomplexobj(jacobian)
    assert np.allclose(jacobian, expected)


def test_single_step_cached_request_returns_absolute_sigma_for_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.common.gn_difference_runner as diff_runner

    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=np.array([1.0, 2.0, 3.0], dtype=float),
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=np.array([4.0, 6.0, 9.0], dtype=float),
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "step_size_calib": False,
        },
    )

    delta_sigma = np.array([0.25, -0.5], dtype=float)
    base_meas = np.array([10.0, 20.0, 30.0], dtype=float)
    pred_diff = np.array([0.5, 1.0, 1.5], dtype=float)

    class _StubForwardModel:
        def fwd_solve(self, image):
            assert image.elem_data is not None
            return SimpleNamespace(meas=base_meas + pred_diff), None

    ctx = {
        "mesh": object(),
        "display_node_coords": np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=float,
        ),
        "display_cell_connectivity": np.array([[0, 1, 2]], dtype=int),
        "operator_bundle": {
            "strict_solver_backend_effective": diff_runner.STRICT_SOLVER_BACKEND_MEASUREMENT,
        },
        "sigma_bg": np.ones_like(delta_sigma),
        "fwd_model": _StubForwardModel(),
        "base_meas": base_meas,
        "cache_build_seconds": {},
        "cache_miss_reasons": {},
        "cache_manager": None,
    }

    monkeypatch.setattr(rc, "_get_cached_fast_context", lambda _cache_key: ctx)
    monkeypatch.setattr(rc, "_put_cached_fast_context", lambda _cache_key, _ctx: None)
    monkeypatch.setattr(
        diff_runner,
        "_measurement_space_delta",
        lambda operator_bundle, rhs: delta_sigma,
    )

    result = rc._run_single_step_cached_request(request)

    assert np.allclose(result.conductivity, np.ones_like(delta_sigma) + delta_sigma)
    assert result.metadata["conductivity_display_mode"] == "absolute_sigma"
    assert np.allclose(
        result.measured,
        request.target_frame.real - request.reference_frame.real,
    )
    assert np.allclose(result.simulated, pred_diff)


def test_single_step_cached_request_uses_normalized_difference_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = np.array([2.0, 4.0, -8.0], dtype=float)
    target = np.array([3.0, 2.0, -4.0], dtype=float)
    base_meas = reference.copy()
    pred_target = np.array([2.5, 5.0, -12.0], dtype=float)
    delta_sigma = np.array([0.2, 0.3], dtype=float)
    captured_rhs: list[np.ndarray] = []
    measurement_backend = "measurement-exact"

    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "difference_mode": "normalized",
            "difference_orientation": "target_minus_reference",
            "step_size_calib": False,
        },
    )

    class _StubForwardModel:
        def fwd_solve(self, image):
            assert image.elem_data is not None
            return SimpleNamespace(meas=pred_target), None

    ctx = {
        "mesh": object(),
        "display_node_coords": np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=float,
        ),
        "display_cell_connectivity": np.array([[0, 1, 2]], dtype=int),
        "operator_bundle": {
            "strict_solver_backend_effective": measurement_backend,
        },
        "sigma_bg": np.ones_like(delta_sigma),
        "fwd_model": _StubForwardModel(),
        "base_meas": base_meas,
        "cache_build_seconds": {},
        "cache_miss_reasons": {},
        "cache_manager": None,
    }

    def _fake_delta(*, operator_bundle, rhs):
        captured_rhs.append(np.asarray(rhs, dtype=float))
        return delta_sigma

    monkeypatch.setattr(rc, "_get_cached_fast_context", lambda _cache_key: ctx)
    monkeypatch.setattr(rc, "_put_cached_fast_context", lambda _cache_key, _ctx: None)
    monkeypatch.setattr(
        rc,
        "_load_gn_difference_runner_module",
        lambda: SimpleNamespace(
            STRICT_SOLVER_BACKEND_MEASUREMENT=measurement_backend,
            _measurement_space_delta=_fake_delta,
            _solve_linear_from_bundle=lambda operator_bundle, rhs: delta_sigma,
            _calibrate_step_size=lambda **kwargs: 1.0,
            build_shared_context=lambda **kwargs: ctx,
        ),
    )

    result = rc._run_single_step_cached_request(request)

    expected_measured = (target - reference) / reference
    expected_simulated = (pred_target - base_meas) / base_meas
    assert np.allclose(captured_rhs[0], expected_measured)
    assert np.allclose(result.measured, expected_measured)
    assert np.allclose(result.simulated, expected_simulated)


def test_single_step_cached_request_uses_rm_artifact_hot_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = np.array([2.0, 4.0, -8.0], dtype=float)
    target = np.array([3.0, 8.0, -4.0], dtype=float)
    rm = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    from pyeidors.inverse import write_rm_artifact

    artifact = tmp_path / "one_step_rm.h5"
    write_rm_artifact(
        artifact,
        rm=rm,
        voxel_shape=np.asarray([2, 1, 1], dtype=np.int64),
        metadata={"algorithm": "one-step-noser"},
    )
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=3,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "difference_mode": "normalized",
            "difference_orientation": "target_minus_reference",
            "dual_model_rm_path": str(artifact),
            "device": "cpu",
            "n_elec": 8,
            "n_rings": 2,
            "radius": 0.18,
            "height": 0.16,
        },
    )

    def _unexpected_context(*_args, **_kwargs):
        raise AssertionError("RM hot path must not build GN context/Jacobian.")

    def _unexpected_runner():
        raise AssertionError("RM hot path must not import the GN runner.")

    monkeypatch.setattr(rc, "_ensure_single_step_cached_context", _unexpected_context)
    monkeypatch.setattr(rc, "_load_gn_difference_runner_module", _unexpected_runner)

    result = rc._run_single_step_cached_request(request)

    expected_dv = (target - reference) / reference
    expected_sigma = rm @ expected_dv
    assert np.allclose(result.conductivity, expected_sigma)
    assert np.allclose(result.measured, expected_dv)
    assert result.simulated is None
    assert result.node_coords.shape[1] == 3
    assert result.cell_connectivity.shape == (2, 8)
    assert result.metadata["single_step_operator_space"] == "rm"
    assert result.metadata["online_hot_path"] == "rm_matmul"
    diagnostics = result.metadata["solver_diagnostics"]
    assert diagnostics["path"] == "single_step_cached_rm"
    assert diagnostics["runtime"]["forward_solve_count"] == 0
    assert diagnostics["runtime"]["adjoint_solve_count"] == 0
    assert diagnostics["runtime"]["jacobian_rebuild_count"] == 0
    assert diagnostics["runtime"]["ksp_solve_count"] == 0
    assert diagnostics["runtime"]["rm_persistent"] is True
    assert diagnostics["runtime"]["rm_prepare_mode"] == "reused_handle"
    assert diagnostics["runtime"]["rm_dtype"] == "float64"
    assert diagnostics["runtime"]["rm_artifact_cache_hit"] is False
    assert diagnostics["cache_lookups"]["rm_artifact"]["layer"] == "artifact"
    assert diagnostics["rm_metadata"]["algorithm"] == "one-step-noser"

    result_warm = rc._run_single_step_cached_request(request)
    warm_diagnostics = result_warm.metadata["solver_diagnostics"]
    assert np.allclose(result_warm.conductivity, expected_sigma)
    assert warm_diagnostics["runtime"]["rm_artifact_cache_hit"] is True
    assert warm_diagnostics["cache_lookups"]["rm_artifact"]["layer"] == "process"
    assert warm_diagnostics["rm_matmul"]["rm_prepare_mode"] == "reused_handle"


def test_single_step_cached_complex_runtime_preserves_complex_scalars() -> None:
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=np.array([1.0, 2.0], dtype=np.float32),
            imag=np.array([0.1, 0.2], dtype=np.float32),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=np.array([1.5, 2.5], dtype=np.float32),
            imag=np.array([0.3, 0.5], dtype=np.float32),
            timestamp=0.0,
            frame_index=1,
        ),
        use_part="complex",
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": "noser_rm",
            "rm_route_requires_artifact": True,
            "rm_auto_build": True,
            "background_conductivity": "1+2j",
            "contact_impedance": "0.01+0.05j",
            "rm_dtype": "complex64",
        },
    )

    runtime = rc._prepare_single_step_cached_runtime(request)
    _signature, payload = rc._planned_one_step_rm_signature(request, runtime)

    assert runtime.background_sigma == 1.0 + 2.0j
    assert runtime.contact_impedance == 0.01 + 0.05j
    assert runtime.meta["rm_dtype"] == "complex64"
    assert payload["background"]["sigma0"] == {"real": 1.0, "imag": 2.0}
    assert payload["background"]["z0"] == {"real": 0.01, "imag": 0.05}


def test_single_step_cached_complex_rm_hot_path_adds_complex_background(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = np.array([1.0 + 1.0j, 2.0 + 0.0j], dtype=np.complex128)
    target = np.array([1.5 + 1.0j, 2.0 + 0.25j], dtype=np.complex128)
    rm = np.array(
        [
            [1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 1.0j],
        ],
        dtype=np.complex128,
    )
    jacobian = np.array(
        [
            [1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 1.0j],
        ],
        dtype=np.complex128,
    )
    node_coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=float,
    )
    cells = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)

    from pyeidors.inverse import write_rm_artifact

    artifact = tmp_path / "complex_one_step_rm.h5"
    write_rm_artifact(
        artifact,
        rm=rm,
        metadata={
            "algorithm": "one-step-noser",
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
        },
        node_coords=node_coords,
        cell_connectivity=cells,
        jacobian=jacobian,
    )
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=np.real(reference),
            imag=np.imag(reference),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=np.real(target),
            imag=np.imag(target),
            timestamp=0.0,
            frame_index=1,
        ),
        use_part="complex",
        mesh_dimension=2,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": "noser_rm",
            "rm_route_requires_artifact": True,
            "rm_output_display_mode": "absolute_sigma",
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
            "dual_model_rm_path": str(artifact),
            "device": "cpu",
            "background_conductivity": "1+2j",
            "contact_impedance": "0.01+0.05j",
            "rm_dtype": "complex128",
        },
    )

    def _unexpected_context(*_args, **_kwargs):
        raise AssertionError("complex RM hot path must not build GN context/Jacobian.")

    monkeypatch.setattr(rc, "_ensure_single_step_cached_context", _unexpected_context)

    result = rc._run_single_step_cached_request(request)

    expected_dv = target - reference
    expected_delta = rm @ expected_dv
    np.testing.assert_allclose(result.conductivity, (1.0 + 2.0j) + expected_delta)
    np.testing.assert_allclose(result.measured, expected_dv)
    np.testing.assert_allclose(result.simulated, jacobian @ expected_delta)
    assert np.iscomplexobj(result.conductivity)
    assert result.metadata["rm_dtype"] == "complex128"
    assert result.metadata["single_step_operator_space"] == "rm"


def test_single_step_cached_rm_artifact_rejects_measurement_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyeidors.inverse import write_rm_artifact

    artifact = tmp_path / "wrong_measurement_count_rm.h5"
    write_rm_artifact(
        artifact,
        rm=np.ones((2, 5), dtype=np.float64),
        voxel_shape=np.asarray([2, 1, 1], dtype=np.int64),
        metadata={"algorithm": "one-step-noser", "n_measurements": 5},
    )
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=np.ones(3, dtype=float),
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=np.ones(3, dtype=float) * 1.01,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=3,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
            "dual_model_rm_path": str(artifact),
            "device": "cpu",
            "n_elec": 8,
            "n_rings": 2,
            "radius": 0.18,
            "height": 0.16,
        },
    )

    def _unexpected_runner():
        raise AssertionError("mismatched RM artifact must fail before GN fallback.")

    monkeypatch.setattr(rc, "_load_gn_difference_runner_module", _unexpected_runner)

    with pytest.raises(
        ValueError,
        match="RM artifact measurement dimension 5 does not match request measurement dimension 3",
    ):
        rc._run_single_step_cached_request(request)


def test_single_step_cached_noser_rm_route_auto_builds_hdf5_hot_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyeidors.inverse import build_one_step_rm

    reference = np.array([2.0, 4.0, 8.0], dtype=float)
    target = np.array([3.0, 5.0, 10.0], dtype=float)
    jacobian = np.array(
        [
            [1.0, 0.2],
            [0.1, 0.8],
            [0.4, 0.3],
        ],
        dtype=float,
    )
    node_coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=float,
    )
    cells = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    fake_ctx = {
        "J": jacobian,
        "display_node_coords": node_coords,
        "display_cell_connectivity": cells,
        "sigma_bg": np.ones(2, dtype=float),
        "mesh": SimpleNamespace(coordinates=lambda: node_coords, cells=lambda: cells),
    }
    context_calls = {"count": 0}

    def _fake_context(*_args, **_kwargs):
        context_calls["count"] += 1
        return fake_ctx

    monkeypatch.setattr(rc, "_ensure_single_step_cached_context", _fake_context)
    monkeypatch.setattr(
        rc,
        "_load_gn_difference_runner_module",
        lambda: SimpleNamespace(build_shared_context=lambda **_kwargs: fake_ctx),
    )
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=2,
        regularization_alpha=0.1,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": "noser_rm",
            "rm_route_requires_artifact": True,
            "rm_auto_build": True,
            "rm_artifact_dir": str(tmp_path),
            "rm_output_display_mode": "absolute_sigma",
            "difference_lambda": 0.04,
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
            "device": "cpu",
        },
    )

    result = rc._run_single_step_cached_request(request)

    expected_dv = target - reference
    expected_rm = build_one_step_rm(
        jacobian,
        lambda_=0.2,
        mode="noser",
        form="measurement",
    )
    expected_sigma = 1.0 + expected_rm @ expected_dv
    np.testing.assert_allclose(result.conductivity, expected_sigma)
    np.testing.assert_allclose(result.simulated, jacobian @ (expected_sigma - 1.0))
    assert result.error_msg is None
    assert context_calls["count"] == 1
    assert result.metadata["single_step_operator_space"] == "rm"
    assert result.metadata["online_hot_path"] == "rm_matmul"
    assert result.metadata["rm_output_display_mode"] == "absolute_sigma"
    artifact_path = Path(result.metadata["rm_artifact_path"])
    assert artifact_path.suffix == ".h5"
    assert artifact_path.exists()
    diagnostics = result.metadata["solver_diagnostics"]
    assert diagnostics["path"] == "single_step_cached_rm"
    assert diagnostics["runtime"]["forward_solve_count"] == 0
    assert diagnostics["runtime"]["jacobian_rebuild_count"] == 0
    assert diagnostics["rm_metadata"]["rm_build_route"] == "noser_rm"
    assert diagnostics["rm_metadata"]["rm_signature"]
    assert diagnostics["rm_metadata"]["form"] == "measurement"

    warm = rc._run_single_step_cached_request(request)
    np.testing.assert_allclose(warm.conductivity, expected_sigma)
    np.testing.assert_allclose(warm.simulated, jacobian @ (expected_sigma - 1.0))
    assert context_calls["count"] == 1
    assert (
        warm.metadata["solver_diagnostics"]["cache_lookups"]["rm_artifact"]["layer"]
        == "process"
    )


def test_single_step_cached_auto_built_rm_rebuilds_stale_fitless_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyeidors.inverse import build_one_step_rm, write_rm_artifact

    with rc._RM_FIT_JACOBIAN_CACHE_LOCK:
        rc._RM_FIT_JACOBIAN_CACHE.clear()
    with rc._RM_ARTIFACT_CACHE_LOCK:
        rc._RM_ARTIFACT_CACHE.clear()

    reference = np.array([2.0, 4.0, 8.0], dtype=float)
    target = np.array([3.0, 5.0, 10.0], dtype=float)
    jacobian = np.array(
        [
            [1.0, 0.2],
            [0.1, 0.8],
            [0.4, 0.3],
        ],
        dtype=float,
    )
    node_coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=float,
    )
    cells = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    fake_ctx = {
        "J": jacobian,
        "display_node_coords": node_coords,
        "display_cell_connectivity": cells,
        "sigma_bg": np.ones(2, dtype=float),
        "mesh": SimpleNamespace(coordinates=lambda: node_coords, cells=lambda: cells),
    }
    context_calls = {"count": 0}

    def _fake_context(*_args, **_kwargs):
        context_calls["count"] += 1
        return fake_ctx

    monkeypatch.setattr(rc, "_ensure_single_step_cached_context", _fake_context)
    monkeypatch.setattr(
        rc,
        "_load_gn_difference_runner_module",
        lambda: SimpleNamespace(build_shared_context=lambda **_kwargs: fake_ctx),
    )
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=2,
        regularization_alpha=0.1,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": "noser_rm",
            "rm_route_requires_artifact": True,
            "rm_auto_build": True,
            "rm_artifact_dir": str(tmp_path),
            "rm_output_display_mode": "absolute_sigma",
            "difference_lambda": 0.04,
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
            "device": "cpu",
        },
    )
    runtime = rc._prepare_single_step_cached_runtime(request)
    stale_path, _signature, _payload = rc._planned_one_step_rm_artifact_path(
        request, runtime
    )
    stale_rm = build_one_step_rm(
        jacobian,
        lambda_=0.2,
        mode="noser",
        form="measurement",
    )
    write_rm_artifact(
        stale_path,
        stale_rm,
        metadata={"algorithm": "one-step-noser", "rm_build_route": "noser_rm"},
        node_coords=node_coords,
        cell_connectivity=cells,
    )

    result = rc._run_single_step_cached_request(request)

    expected_dv = target - reference
    expected_sigma = 1.0 + stale_rm @ expected_dv
    np.testing.assert_allclose(result.conductivity, expected_sigma)
    np.testing.assert_allclose(result.simulated, jacobian @ (expected_sigma - 1.0))
    assert result.metadata["rm_artifact_cache_status"] == "built"
    assert result.metadata["rm_fit_jacobian_cache_status"].startswith("built_")
    assert context_calls["count"] == 1


def test_single_step_cached_auto_built_rm_skips_oversize_fit_jacobian_without_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyeidors.inverse import build_one_step_rm, write_rm_artifact

    with rc._RM_FIT_JACOBIAN_CACHE_LOCK:
        rc._RM_FIT_JACOBIAN_CACHE.clear()
    with rc._RM_ARTIFACT_CACHE_LOCK:
        rc._RM_ARTIFACT_CACHE.clear()

    reference = np.array([2.0, 4.0, 8.0], dtype=float)
    target = np.array([3.0, 5.0, 10.0], dtype=float)
    jacobian = np.array(
        [
            [1.0, 0.2],
            [0.1, 0.8],
            [0.4, 0.3],
        ],
        dtype=float,
    )
    node_coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=float,
    )
    cells = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)

    def _unexpected_context(*_args, **_kwargs):
        raise AssertionError("oversize persisted fit Jacobian should not rebuild RM")

    monkeypatch.setattr(rc, "_ensure_single_step_cached_context", _unexpected_context)
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=2,
        regularization_alpha=0.1,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": "noser_rm",
            "rm_route_requires_artifact": True,
            "rm_auto_build": True,
            "rm_artifact_dir": str(tmp_path),
            "rm_fit_jacobian_max_bytes": 1,
            "rm_output_display_mode": "absolute_sigma",
            "difference_lambda": 0.04,
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
            "device": "cpu",
        },
    )
    runtime = rc._prepare_single_step_cached_runtime(request)
    artifact_path, _signature, _payload = rc._planned_one_step_rm_artifact_path(
        request, runtime
    )
    rm = build_one_step_rm(
        jacobian,
        lambda_=0.2,
        mode="noser",
        form="measurement",
    )
    write_rm_artifact(
        artifact_path,
        rm,
        metadata={
            "algorithm": "one-step-noser",
            "fit_jacobian_persisted": True,
            "rm_build_route": "noser_rm",
        },
        node_coords=node_coords,
        cell_connectivity=cells,
        jacobian=jacobian,
    )

    result = rc._run_single_step_cached_request(request)

    expected_sigma = 1.0 + rm @ (target - reference)
    np.testing.assert_allclose(result.conductivity, expected_sigma)
    assert result.simulated is None
    assert result.metadata["rm_artifact_cache_status"] == "disk_hit"
    assert result.metadata["rm_fit_jacobian_cache_status"] == "artifact_too_large"
    assert result.metadata["rm_fit_jacobian_available_but_skipped"] is True
    assert result.metadata["rm_fit_jacobian_bytes"] == jacobian.nbytes
    assert result.metadata["rm_fit_jacobian_max_bytes"] == 1


def test_auto_build_skips_persisting_oversize_fit_jacobian_and_warm_hits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import h5py

    with rc._RM_FIT_JACOBIAN_CACHE_LOCK:
        rc._RM_FIT_JACOBIAN_CACHE.clear()
    with rc._RM_ARTIFACT_CACHE_LOCK:
        rc._RM_ARTIFACT_CACHE.clear()

    reference = np.array([2.0, 4.0, 8.0], dtype=float)
    target = np.array([3.0, 5.0, 10.0], dtype=float)
    jacobian = np.array(
        [
            [1.0, 0.2],
            [0.1, 0.8],
            [0.4, 0.3],
        ],
        dtype=float,
    )
    node_coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=float,
    )
    cells = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    fake_ctx = {
        "J": jacobian,
        "display_node_coords": node_coords,
        "display_cell_connectivity": cells,
        "sigma_bg": np.ones(2, dtype=float),
        "mesh": SimpleNamespace(coordinates=lambda: node_coords, cells=lambda: cells),
    }
    context_calls = {"count": 0}

    def _fake_context(*_args, **_kwargs):
        context_calls["count"] += 1
        return fake_ctx

    monkeypatch.setattr(rc, "_ensure_single_step_cached_context", _fake_context)
    monkeypatch.setattr(
        rc,
        "_load_gn_difference_runner_module",
        lambda: SimpleNamespace(build_shared_context=lambda **_kwargs: fake_ctx),
    )
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=2,
        regularization_alpha=0.1,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": "noser_rm",
            "rm_route_requires_artifact": True,
            "rm_auto_build": True,
            "rm_artifact_dir": str(tmp_path),
            "rm_fit_jacobian_max_bytes": 1,
            "rm_output_display_mode": "absolute_sigma",
            "difference_lambda": 0.04,
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
            "device": "cpu",
        },
    )

    first = rc._run_single_step_cached_request(request)
    artifact_path = Path(first.metadata["rm_artifact_path"])

    assert first.simulated is None
    assert first.metadata["rm_fit_jacobian_available_but_skipped"] is True
    assert first.metadata["rm_fit_jacobian_bytes"] == jacobian.nbytes
    with h5py.File(artifact_path, "r") as handle:
        assert "jacobian" not in handle["arrays"]
        metadata = handle.attrs["metadata_json"]
        assert '"fit_jacobian_persisted": false' in str(metadata)
        assert '"fit_jacobian_persist_skip_reason": "too_large"' in str(metadata)

    monkeypatch.setattr(
        rc,
        "_ensure_single_step_cached_context",
        lambda *_args, **_kwargs: pytest.fail("oversize fitless RM should warm-hit"),
    )
    warm = rc._run_single_step_cached_request(request)

    assert warm.simulated is None
    assert warm.metadata["rm_artifact_cache_status"] == "disk_hit"
    assert warm.metadata["rm_fit_jacobian_cache_status"] == "artifact_too_large"
    assert context_calls["count"] == 1


def test_rm_artifact_process_cache_respects_byte_budget(tmp_path: Path) -> None:
    from pyeidors.inverse import write_rm_artifact

    with rc._RM_ARTIFACT_CACHE_LOCK:
        rc._RM_ARTIFACT_CACHE.clear()

    rm = np.eye(3, dtype=np.float64)
    artifact_path = write_rm_artifact(
        tmp_path / "small_rm.h5",
        rm,
        metadata={"algorithm": "one-step-noser"},
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float),
        cell_connectivity=np.array([[0, 1, 2]], dtype=np.int32),
    )
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=np.array([1.0, 2.0, 3.0], dtype=float),
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=np.array([2.0, 3.0, 4.0], dtype=float),
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=2,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": "noser_rm",
            "rm_artifact_path": str(artifact_path),
            "rm_artifact_process_cache_max_bytes": 1,
            "rm_streaming_matmul": "off",
            "device": "cpu",
        },
    )

    first = rc._run_single_step_cached_request(request)
    second = rc._run_single_step_cached_request(request)

    np.testing.assert_allclose(first.conductivity, np.ones(3, dtype=float))
    np.testing.assert_allclose(second.conductivity, np.ones(3, dtype=float))
    assert first.metadata["rm_artifact_cache_hit"] is False
    assert second.metadata["rm_artifact_cache_hit"] is False
    assert first.metadata["rm_artifact_process_cache_stored"] is False
    assert first.metadata["rm_artifact_process_cache_skip_reason"] == "entry_too_large"
    assert first.metadata["rm_artifact_process_cache_bytes"] > 1
    assert first.metadata["rm_artifact_process_cache_max_bytes"] == 1
    with rc._RM_ARTIFACT_CACHE_LOCK:
        assert not rc._RM_ARTIFACT_CACHE


def test_hdf5_rm_streaming_matmul_avoids_full_rm_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyeidors.inverse import write_rm_artifact
    from pyeidors.io import hdf5_artifacts

    with rc._RM_ARTIFACT_CACHE_LOCK:
        rc._RM_ARTIFACT_CACHE.clear()

    rm = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 2.0, 0.0],
            [0.25, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    artifact_path = write_rm_artifact(
        tmp_path / "streaming_rm.h5",
        rm,
        metadata={
            "algorithm": "one-step-noser",
            "rm_hdf5_streaming_chunk_bytes": 24,
        },
        node_coords=np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=float,
        ),
        cell_connectivity=np.array([[0, 1, 2]], dtype=np.int32),
    )
    monkeypatch.setattr(
        rc,
        "_load_rm_artifact",
        lambda *_args, **_kwargs: pytest.fail("streaming path loaded full RM"),
    )
    monkeypatch.setattr(
        hdf5_artifacts.HDF5LazyDataset,
        "__getitem__",
        lambda *_args, **_kwargs: pytest.fail(
            "streaming path should keep the HDF5 file open instead of __getitem__"
        ),
    )
    reference = np.array([1.0, 2.0, 3.0], dtype=float)
    target = np.array([2.0, 4.0, 6.0], dtype=float)
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=2,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": "noser_rm",
            "rm_artifact_path": str(artifact_path),
            "rm_artifact_process_cache_max_bytes": 1,
            "rm_streaming_chunk_bytes": 48,
            "device": "cpu",
        },
    )

    result = rc._run_single_step_cached_request(request)

    expected = rm @ (target - reference)
    np.testing.assert_allclose(result.conductivity, expected)
    assert result.metadata["rm_streaming"] is True
    assert result.metadata["online_hot_path"] == "rm_hdf5_streaming_matmul"
    assert result.metadata["rm_artifact_process_cache_stored"] is False
    assert result.metadata["rm_artifact_process_cache_skip_reason"] == "streaming_hdf5"
    diagnostics = result.metadata["solver_diagnostics"]
    assert diagnostics["runtime"]["rm_streaming"] is True
    assert diagnostics["rm_matmul"]["backend"] == "hdf5_chunked"
    assert diagnostics["rm_matmul"]["rm_hdf5_dataset_chunks"] == (1, 3)
    assert diagnostics["rm_matmul"]["rm_streaming_rows_per_chunk"] == 2
    assert diagnostics["rm_matmul"]["rm_hdf5_file_open_mode"] == "single_open"
    assert diagnostics["rm_matmul"]["rm_streaming_chunks"] == 2
    with rc._RM_ARTIFACT_CACHE_LOCK:
        assert not rc._RM_ARTIFACT_CACHE


def test_hdf5_rm_cuda_request_streams_when_resident_budget_exceeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyeidors.inverse import write_rm_artifact

    with rc._RM_ARTIFACT_CACHE_LOCK:
        rc._RM_ARTIFACT_CACHE.clear()

    rm = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 2.0, 0.0],
            [0.25, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    artifact_path = write_rm_artifact(
        tmp_path / "cuda_budget_rm.h5",
        rm,
        metadata={"algorithm": "one-step-noser"},
        node_coords=np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=float,
        ),
        cell_connectivity=np.array([[0, 1, 2]], dtype=np.int32),
    )
    monkeypatch.setattr(
        rc,
        "_load_rm_artifact",
        lambda *_args, **_kwargs: pytest.fail(
            "CUDA budget fallback should not load the full RM"
        ),
    )
    reference = np.array([1.0, 2.0, 3.0], dtype=float)
    target = np.array([2.0, 4.0, 6.0], dtype=float)
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=2,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": "noser_rm",
            "rm_artifact_path": str(artifact_path),
            "rm_device_resident_max_bytes": 1,
            "rm_streaming_chunk_bytes": 48,
            "device": "cuda",
        },
    )

    result = rc._run_single_step_cached_request(request)

    np.testing.assert_allclose(result.conductivity, rm @ (target - reference))
    assert result.metadata["rm_streaming"] is True
    assert result.metadata["rm_streaming_decision"] == "cuda_resident_budget_exceeded"
    assert result.metadata["rm_device_resident_max_bytes"] == 1
    diagnostics = result.metadata["solver_diagnostics"]
    assert diagnostics["runtime"]["device_requested"] == "cuda"
    assert diagnostics["runtime"]["device_effective"] == "cpu"
    assert diagnostics["runtime"]["rm_streaming_decision"] == (
        "cuda_resident_budget_exceeded"
    )


def test_hdf5_rm_streaming_keeps_greit_training_matrices_lazy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyeidors.io import hdf5_artifacts
    from pyeidors.io.hdf5_artifacts import write_hdf5_artifact

    with rc._RM_ARTIFACT_CACHE_LOCK:
        rc._RM_ARTIFACT_CACHE.clear()

    rm = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 2.0, 0.0],
            [0.25, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    artifact_path = write_hdf5_artifact(
        tmp_path / "streaming_greit_aux.h5",
        {
            "rm": rm,
            "Y": np.arange(12, dtype=np.float64).reshape(3, 4),
            "D": np.arange(12, dtype=np.float64).reshape(3, 4) / 10.0,
            "node_coords": np.array(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=np.float64,
            ),
            "cell_connectivity": np.array([[0, 1, 2]], dtype=np.int32),
            "rec_model": np.arange(9, dtype=np.float64).reshape(3, 3),
        },
        {
            "artifact_schema": "pyeidors-rm-hdf5-v1",
            "algorithm": "one-step-noser",
            "rm_shape": [3, 3],
        },
        schema="pyeidors-rm-hdf5-v1",
    )
    original_array = hdf5_artifacts.HDF5LazyDataset.__array__
    loaded_aux: list[str] = []

    def _guard_aux_array(self, dtype=None):
        if self.info.name in {"Y", "D", "rec_model"}:
            loaded_aux.append(self.info.name)
            raise AssertionError(f"{self.info.name} should stay lazy")
        return original_array(self, dtype=dtype)

    monkeypatch.setattr(hdf5_artifacts.HDF5LazyDataset, "__array__", _guard_aux_array)
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=np.array([1.0, 2.0, 3.0], dtype=float),
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=np.array([2.0, 4.0, 6.0], dtype=float),
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=2,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": "noser_rm",
            "rm_artifact_path": str(artifact_path),
            "rm_artifact_process_cache_max_bytes": 1,
            "device": "cpu",
        },
    )

    result = rc._run_single_step_cached_request(request)

    np.testing.assert_allclose(result.conductivity, rm @ np.array([1.0, 2.0, 3.0]))
    assert result.metadata["rm_streaming"] is True
    assert loaded_aux == []


@pytest.mark.parametrize(
    ("route", "mode", "prior_builder", "expected_source"),
    [
        ("laplace_rm", "laplace", "graph_laplacian", "provided_laplace"),
        ("curvature_rm", "curvature", "graph_ltl_prior", "provided_graph_ltl"),
    ],
)
def test_single_step_cached_smooth_rm_routes_auto_build_graph_prior_hdf5_hot_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    mode: str,
    prior_builder: str,
    expected_source: str,
) -> None:
    from pyeidors.inverse import CellMesh
    from pyeidors.inverse.prior import graph_laplacian, graph_ltl_prior
    from pyeidors.inverse.reconstruction_matrix import build_one_step_rm

    reference = np.ones(4, dtype=float)
    target = reference + np.array([0.0, 1.0, 1.0, 0.0], dtype=float)
    jacobian = np.eye(4, dtype=float)
    node_coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
        dtype=float,
    )
    cells = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int32)
    fake_ctx = {
        "J": jacobian,
        "display_node_coords": node_coords,
        "display_cell_connectivity": cells,
        "sigma_bg": np.ones(4, dtype=float),
        "mesh": SimpleNamespace(coordinates=lambda: node_coords, cells=lambda: cells),
    }
    context_calls = {"count": 0}

    def _fake_context(*_args, **_kwargs):
        context_calls["count"] += 1
        return fake_ctx

    monkeypatch.setattr(rc, "_ensure_single_step_cached_context", _fake_context)
    monkeypatch.setattr(
        rc,
        "_load_gn_difference_runner_module",
        lambda: SimpleNamespace(build_shared_context=lambda **_kwargs: fake_ctx),
    )
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros(4, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros(4, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=2,
        regularization_alpha=0.1,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": route,
            "rm_regularization": mode,
            "rm_route_requires_artifact": True,
            "rm_auto_build": True,
            "rm_artifact_dir": str(tmp_path),
            "rm_output_display_mode": "absolute_sigma",
            "difference_lambda": 0.25,
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
            "device": "cpu",
        },
    )

    result = rc._run_single_step_cached_request(request)

    inverse_mesh = CellMesh(node_coords, cells, name=f"{route}-expected")
    regularization = (
        graph_laplacian(inverse_mesh)
        if prior_builder == "graph_laplacian"
        else graph_ltl_prior(inverse_mesh)
    )
    expected_rm = build_one_step_rm(
        jacobian,
        regularization=regularization,
        lambda_=0.5,
        mode=mode,
        form="param",
    )
    expected_delta = expected_rm @ (target - reference)
    np.testing.assert_allclose(result.conductivity, 1.0 + expected_delta)
    assert result.error_msg is None
    assert context_calls["count"] == 1
    assert Path(result.metadata["rm_artifact_path"]).suffix == ".h5"
    assert result.metadata["single_step_operator_space"] == "rm"
    assert result.metadata["online_hot_path"] == "rm_matmul"
    delta = result.conductivity - 1.0
    assert delta[1:3].mean() > delta[[0, 3]].mean()
    if route == "laplace_rm":
        assert abs(delta[1] - delta[2]) <= 1.0e-12
        assert abs(delta[0] - delta[3]) <= 1.0e-12
    else:
        assert np.isfinite(delta).all()

    diagnostics = result.metadata["solver_diagnostics"]
    assert diagnostics["path"] == "single_step_cached_rm"
    assert diagnostics["runtime"]["forward_solve_count"] == 0
    assert diagnostics["runtime"]["jacobian_rebuild_count"] == 0
    assert diagnostics["rm_metadata"]["rm_build_route"] == route
    assert diagnostics["rm_metadata"]["regularization_type"] == mode
    assert diagnostics["rm_metadata"]["regularization_source"] == expected_source
    assert diagnostics["rm_metadata"]["RtR_signature_hash"]
    assert diagnostics["rm_metadata"]["form"] == "param"

    warm = rc._run_single_step_cached_request(request)
    np.testing.assert_allclose(warm.conductivity, result.conductivity)
    assert context_calls["count"] == 1
    assert (
        warm.metadata["solver_diagnostics"]["cache_lookups"]["rm_artifact"]["layer"]
        == "process"
    )


def test_single_step_cached_auto_built_rm_honors_float32_precision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyeidors.inverse.reconstruction_matrix import load_rm_artifact

    with rc._RM_ARTIFACT_CACHE_LOCK:
        rc._RM_ARTIFACT_CACHE.clear()
    with rc._RM_FIT_JACOBIAN_CACHE_LOCK:
        rc._RM_FIT_JACOBIAN_CACHE.clear()

    reference = np.ones(4, dtype=np.float32)
    target = reference + np.array([0.0, 0.5, 1.0, 0.0], dtype=np.float32)
    jacobian = np.eye(4, dtype=np.float64)
    node_coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    cells = np.array([[0, 1, 2, 3]], dtype=np.int32)
    fake_ctx = {
        "J": jacobian,
        "display_node_coords": node_coords,
        "display_cell_connectivity": cells,
        "sigma_bg": np.ones(4, dtype=np.float64),
        "mesh": SimpleNamespace(coordinates=lambda: node_coords, cells=lambda: cells),
    }

    monkeypatch.setattr(
        rc,
        "_ensure_single_step_cached_context",
        lambda *_args, **_kwargs: fake_ctx,
    )
    monkeypatch.setattr(
        rc,
        "_load_gn_difference_runner_module",
        lambda: SimpleNamespace(build_shared_context=lambda **_kwargs: fake_ctx),
    )
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros(4, dtype=np.float32),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros(4, dtype=np.float32),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=2,
        regularization_alpha=0.1,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": "noser_rm",
            "rm_regularization": "noser",
            "rm_form": "measurement",
            "rm_route_requires_artifact": True,
            "rm_auto_build": True,
            "rm_artifact_dir": str(tmp_path),
            "rm_output_display_mode": "delta_sigma",
            "difference_lambda": 0.25,
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
            "compute_precision": "float32",
            "compute_dtype": "float32",
            "rm_dtype": "float32",
            "rm_matmul_dtype": "float32",
            "device": "cpu",
        },
    )

    result = rc._run_single_step_cached_request(request)

    artifact_path = Path(result.metadata["rm_artifact_path"])
    artifact = load_rm_artifact(artifact_path)
    diagnostics = result.metadata["solver_diagnostics"]

    assert result.error_msg is None
    assert artifact.rm.dtype == np.float32
    assert artifact.metadata["rm_dtype"] == "float32"
    assert artifact.metadata["build_dtype"] == "float32"
    assert artifact.metadata["prior_inverse_solver"] == "diagonal"
    assert diagnostics["runtime"]["rm_dtype"] == "float32"
    assert diagnostics["rm_metadata"]["rm_dtype"] == "float32"
    assert diagnostics["rm_matmul"]["rm_dtype"] == "float32"
    assert result.metadata["rm_signature_payload"]["hyperparameters"]["rm_dtype"] == (
        "float32"
    )


def test_single_step_cached_3d_rm_auto_build_forces_dense_jacobian_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with rc._RM_FIT_JACOBIAN_CACHE_LOCK:
        rc._RM_FIT_JACOBIAN_CACHE.clear()

    from pyeidors.inverse import CellMesh
    from pyeidors.inverse.prior import graph_laplacian
    from pyeidors.inverse.reconstruction_matrix import build_one_step_rm

    reference = np.ones(4, dtype=float)
    target = reference + np.array([0.0, 1.0, 0.5, -0.25], dtype=float)
    jacobian = np.array(
        [
            [1.0, 0.2],
            [0.1, 0.8],
            [0.4, 0.3],
            [0.0, 0.5],
        ],
        dtype=float,
    )
    node_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    fake_ctx = {
        "J": jacobian,
        "display_node_coords": node_coords,
        "display_cell_connectivity": cells,
        "sigma_bg": np.ones(2, dtype=float),
        "mesh": SimpleNamespace(coordinates=lambda: node_coords, cells=lambda: cells),
    }
    build_calls: list[dict[str, object]] = []

    def _fake_build_shared_context(**kwargs):
        build_calls.append(dict(kwargs))
        return fake_ctx

    monkeypatch.setattr(rc, "_get_cached_fast_context", lambda _cache_key: None)
    monkeypatch.setattr(rc, "_put_cached_fast_context", lambda _cache_key, _ctx: None)
    monkeypatch.setattr(
        rc,
        "_load_gn_difference_runner_module",
        lambda: SimpleNamespace(build_shared_context=_fake_build_shared_context),
    )
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros(4, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros(4, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=3,
        regularization_alpha=0.1,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "mesh_dimension": 3,
            "simulation_inverse_route": "laplace_rm",
            "rm_regularization": "laplace",
            "rm_route_requires_artifact": True,
            "rm_auto_build": True,
            "rm_artifact_dir": str(tmp_path),
            "rm_output_display_mode": "absolute_sigma",
            "difference_lambda": 0.04,
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
            "solver_mode": "fast",
            "jacobian_representation": "auto",
            "device": "cpu",
        },
    )

    runtime = rc._prepare_single_step_cached_runtime(request)
    assert runtime.meta["jacobian_representation"] == "dense"
    assert runtime.meta["rm_build_jacobian_representation_requested"] == "linearized"
    assert runtime.meta["single_step_context_cache_scope"] == "process"

    result = rc._run_single_step_cached_request(request)

    inverse_mesh = CellMesh(node_coords, cells, name="expected-3d-laplace-rm")
    expected_rm = build_one_step_rm(
        jacobian,
        regularization=graph_laplacian(inverse_mesh),
        lambda_=0.2,
        mode="laplace",
        form="param",
    )
    expected_delta = expected_rm @ (target - reference)
    np.testing.assert_allclose(result.conductivity, 1.0 + expected_delta)
    np.testing.assert_allclose(result.simulated, jacobian @ expected_delta)
    assert result.error_msg is None
    assert len(build_calls) == 1
    assert build_calls[0]["jacobian_representation"] == "dense"
    assert build_calls[0]["cache_scope"] == "process"
    assert "_inmem_jacobian" not in result.metadata
    assert result.metadata["rm_build_jacobian_representation"] == "dense"
    assert result.metadata["rm_build_jacobian_representation_requested"] == "linearized"
    assert (
        result.metadata["solver_diagnostics"]["rm_metadata"][
            "rm_jacobian_source_cache_scope"
        ]
        == "process"
    )
    assert (
        result.metadata["solver_diagnostics"]["rm_metadata"]["rm_build_route"]
        == "laplace_rm"
    )

    warm_result = rc._run_single_step_cached_request(request)

    np.testing.assert_allclose(warm_result.conductivity, 1.0 + expected_delta)
    np.testing.assert_allclose(warm_result.simulated, jacobian @ expected_delta)
    assert len(build_calls) == 1
    assert "_inmem_jacobian" not in warm_result.metadata
    assert warm_result.metadata["rm_artifact_cache_status"] == "disk_hit"
    assert warm_result.metadata["rm_fit_jacobian_cache_status"] == "process_hit"

    with rc._RM_FIT_JACOBIAN_CACHE_LOCK:
        rc._RM_FIT_JACOBIAN_CACHE.clear()

    artifact_fit_result = rc._run_single_step_cached_request(request)

    np.testing.assert_allclose(artifact_fit_result.conductivity, 1.0 + expected_delta)
    np.testing.assert_allclose(artifact_fit_result.simulated, jacobian @ expected_delta)
    assert len(build_calls) == 1
    assert "_inmem_jacobian" not in artifact_fit_result.metadata
    assert artifact_fit_result.metadata["rm_artifact_cache_status"] == "disk_hit"
    assert artifact_fit_result.metadata["rm_fit_jacobian_cache_status"].startswith(
        "artifact_hit_"
    )


def test_single_step_cached_2d_cuda_rm_auto_build_uses_cuda_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with rc._RM_FIT_JACOBIAN_CACHE_LOCK:
        rc._RM_FIT_JACOBIAN_CACHE.clear()
    with rc._RM_ARTIFACT_CACHE_LOCK:
        rc._RM_ARTIFACT_CACHE.clear()

    reference = np.ones(4, dtype=float)
    target = reference + np.array([0.0, 0.20, -0.10, 0.05], dtype=float)
    jacobian = np.array(
        [
            [1.0, 0.2],
            [0.1, 0.8],
            [0.4, 0.3],
            [0.2, 0.5],
        ],
        dtype=float,
    )
    node_coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=float,
    )
    cells = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    fake_ctx = {
        "J": jacobian,
        "display_node_coords": node_coords,
        "display_cell_connectivity": cells,
        "sigma_bg": np.ones(2, dtype=float),
        "mesh": SimpleNamespace(coordinates=lambda: node_coords, cells=lambda: cells),
    }
    build_calls: list[dict[str, object]] = []

    def _fake_build_shared_context(**kwargs):
        build_calls.append(dict(kwargs))
        return fake_ctx

    monkeypatch.setattr(rc, "_get_cached_fast_context", lambda _cache_key: None)
    monkeypatch.setattr(rc, "_put_cached_fast_context", lambda _cache_key, _ctx: None)
    monkeypatch.setattr(
        rc,
        "_load_gn_difference_runner_module",
        lambda: SimpleNamespace(build_shared_context=_fake_build_shared_context),
    )
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros(4, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros(4, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=2,
        regularization_alpha=0.1,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "mesh_dimension": 2,
            "mesh_size": 0.1,
            "simulation_inverse_route": "noser_rm",
            "rm_regularization": "noser",
            "rm_route_requires_artifact": True,
            "rm_auto_build": True,
            "rm_artifact_dir": str(tmp_path),
            "rm_device": "cpu",
            "rm_output_display_mode": "absolute_sigma",
            "difference_lambda": 1.0e-2,
            "difference_mode": "normalized",
            "difference_orientation": "target_minus_reference",
            "solver_mode": "strict",
            "jacobian_representation": "auto",
            "device": "cuda",
            "petsc_device": "cuda",
            "forward_backend": "dolfinx",
            "mesh_family": "tetra",
            "potential_order": 1,
        },
    )

    result = rc._run_single_step_cached_request(request)

    assert result.error_msg is None
    assert np.isfinite(result.conductivity).all()
    assert len(build_calls) == 1
    call = build_calls[0]
    assert call["mesh_dim"] == 2
    assert call["petsc_device"] == "cuda"
    assert call["device"] == "cuda"
    assert call["forward_backend"] == "dolfinx"
    assert call["jacobian_representation"] == "dense"
    assert call["cache_scope"] == "process"
    assert call["difference_mode"] == "normalized"
    assert (
        result.metadata["solver_diagnostics"]["rm_metadata"][
            "rm_jacobian_source_cache_scope"
        ]
        == "process"
    )


def test_single_step_cached_non_noser_rm_route_requires_artifact_before_dense_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = np.array([2.0, 4.0, -8.0], dtype=float)
    target = np.array([3.0, 8.0, -4.0], dtype=float)
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=2,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": "laplace_rm",
            "rm_route_requires_artifact": True,
            "rm_route_pending_task": "T101",
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
        },
    )

    def _unexpected_context(*_args, **_kwargs):
        raise AssertionError("RM route without artifact must not build dense context.")

    def _unexpected_runner():
        raise AssertionError("RM route without artifact must not import GN runner.")

    monkeypatch.setattr(rc, "_ensure_single_step_cached_context", _unexpected_context)
    monkeypatch.setattr(rc, "_load_gn_difference_runner_module", _unexpected_runner)

    result = rc._run_single_step_cached_request(request)

    assert result.error_msg is not None
    assert "laplace_rm requires a precomputed RM/GREIT artifact" in result.error_msg
    assert result.conductivity.size == 0
    assert result.metadata["rm_artifact_missing"] is True
    assert result.metadata["rm_route_pending_task"] == "T101"
    diagnostics = result.metadata["solver_diagnostics"]
    assert diagnostics["path"] == "single_step_cached_rm_missing_artifact"
    assert diagnostics["runtime"]["forward_solve_count"] == 0
    assert diagnostics["runtime"]["jacobian_rebuild_count"] == 0


def test_single_step_cached_request_resolves_greit_common_config_hot_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyeidors.inverse import precompute_greit_common_config

    warmup = precompute_greit_common_config("16e", artifact_dir=tmp_path)
    reference = np.linspace(1.0, 2.0, warmup.config.n_measurements, dtype=float)
    target = reference + np.linspace(
        0.01,
        0.02,
        warmup.config.n_measurements,
        dtype=float,
    )
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros_like(reference),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros_like(target),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=3,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
            "greit_common_config": "16e",
            "greit_common_config_dir": str(tmp_path),
            "device": "cpu",
            "n_elec": 16,
            "n_rings": 1,
            "radius": 0.18,
            "height": 0.16,
        },
    )

    def _unexpected_context(*_args, **_kwargs):
        raise AssertionError("common-config RM hot path must not build context.")

    def _unexpected_runner():
        raise AssertionError("common-config RM hot path must not import GN runner.")

    monkeypatch.setattr(rc, "_ensure_single_step_cached_context", _unexpected_context)
    monkeypatch.setattr(rc, "_load_gn_difference_runner_module", _unexpected_runner)

    result = rc._run_single_step_cached_request(request)

    expected_dv = target - reference
    expected_sigma = warmup.greit.rm @ expected_dv
    np.testing.assert_allclose(result.conductivity, expected_sigma)
    assert result.metadata["single_step_operator_space"] == "rm"
    assert result.metadata["online_hot_path"] == "rm_matmul"
    assert result.metadata["rm_artifact_path"] == str(warmup.artifact_path)
    diagnostics = result.metadata["solver_diagnostics"]
    assert diagnostics["path"] == "single_step_cached_rm"
    assert diagnostics["runtime"]["forward_solve_count"] == 0
    assert diagnostics["runtime"]["adjoint_solve_count"] == 0
    assert diagnostics["runtime"]["jacobian_rebuild_count"] == 0
    assert diagnostics["runtime"]["ksp_solve_count"] == 0
    assert diagnostics["rm_metadata"]["common_config_id"] == "16e"


def test_single_step_cached_greit_production_route_rejects_fixture_auto_warm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyeidors.inverse import greit_common_config

    cfg = greit_common_config("16e")
    reference = np.linspace(1.0, 2.0, cfg.n_measurements, dtype=float)
    target = reference + np.linspace(0.01, 0.02, cfg.n_measurements, dtype=float)
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros_like(reference),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros_like(target),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=3,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": "greit3d_rm",
            "rm_route_requires_artifact": True,
            "greit_common_config": "16e",
            "greit_common_config_dir": str(tmp_path),
            "greit_common_config_auto_warm": True,
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
            "device": "cpu",
            "n_elec": 16,
            "n_rings": 1,
            "radius": 0.18,
            "height": 0.16,
            "greit_official_fixture_scope": "requires registered EIDORS parity artifact",
            "greit_5936_protocol_scope": "production route rejects deterministic fixtures",
            "greit_official_equivalence_claim_allowed": False,
        },
    )

    def _unexpected_context(*_args, **_kwargs):
        raise AssertionError("GREIT common-config route must not build context.")

    def _unexpected_runner():
        raise AssertionError("GREIT common-config route must not import GN runner.")

    monkeypatch.setattr(rc, "_ensure_single_step_cached_context", _unexpected_context)
    monkeypatch.setattr(rc, "_load_gn_difference_runner_module", _unexpected_runner)

    result = rc._run_single_step_cached_request(request)

    assert result.error_msg
    assert result.metadata["rm_artifact_missing"] is True
    assert result.metadata["rm_artifact_required"] is True
    assert "registered EIDORS-parity artifact" in result.error_msg
    assert not (tmp_path / "greit3d_common_16e.h5").exists()


def test_single_step_cached_greit_official_artifact_uses_rec_geometry_and_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyeidors.inverse import GREIT_EIDORS_HDF5_SCHEMA, GREITRM

    artifact_path = tmp_path / "official_greit.h5"
    rm = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.25],
        ],
        dtype=float,
    )
    y = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, -0.25],
        ],
        dtype=float,
    )
    d = np.eye(2, dtype=float)
    rec_model = np.array(
        [
            [-0.2, -0.2, 0.0],
            [0.0, -0.2, 0.0],
            [-0.2, 0.0, 0.0],
            [0.2, 0.2, 0.1],
            [0.4, 0.2, 0.1],
            [0.2, 0.4, 0.1],
        ],
        dtype=float,
    )
    GREITRM(
        rm=rm,
        metadata=MappingProxyType(
            {
                "algorithm": "greit-3d",
                "artifact_schema": GREIT_EIDORS_HDF5_SCHEMA,
                "artifact_format": "hdf5",
                "eidors_parity": True,
                "fixture_only": False,
                "keep_model_components": True,
                "online_hot_path": "rm_matmul",
            }
        ),
        voxel_shape=(2, 1, 1),
        y=y,
        d=d,
        rec_model=rec_model,
    ).save(artifact_path)

    reference = np.array([2.0, 3.0, 4.0], dtype=float)
    target = reference + np.array([0.2, -0.1, 0.05], dtype=float)
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros_like(reference),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros_like(target),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=3,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": "greit3d_rm",
            "rm_route_requires_artifact": True,
            "greit_rm_path": str(artifact_path),
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
            "device": "cpu",
            "n_elec": 3,
            "n_rings": 1,
            "radius": 1.0,
            "height": 1.0,
        },
    )

    def _unexpected_context(*_args, **_kwargs):
        raise AssertionError("official GREIT artifact route must not build context.")

    def _unexpected_runner():
        raise AssertionError("official GREIT artifact route must not import GN runner.")

    monkeypatch.setattr(rc, "_ensure_single_step_cached_context", _unexpected_context)
    monkeypatch.setattr(rc, "_load_gn_difference_runner_module", _unexpected_runner)

    result = rc._run_single_step_cached_request(request)

    dv = target - reference
    expected = rm @ dv
    np.testing.assert_allclose(result.conductivity, expected)
    assert result.node_coords.shape == (16, 3)
    assert result.cell_connectivity.shape == (2, 8)
    assert result.metadata["rm_geometry_source"] == "greit_rec_model_centers"
    assert result.metadata["rm_fit_source"] == "greit_training_space_projection"
    np.testing.assert_allclose(result.simulated, y @ expected)
    diagnostics = result.metadata["solver_diagnostics"]
    assert diagnostics["path"] == "single_step_cached_rm"
    assert diagnostics["runtime"]["forward_solve_count"] == 0
    assert diagnostics["runtime"]["adjoint_solve_count"] == 0
    assert diagnostics["runtime"]["jacobian_rebuild_count"] == 0
    assert diagnostics["runtime"]["ksp_solve_count"] == 0
    assert diagnostics["rm_metadata"]["eidors_parity"] is True


def test_single_step_cached_greit_registry_hot_path_requires_exact_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyeidors.inverse import (
        GREIT_EIDORS_HDF5_SCHEMA,
        GREITRM,
        greit_artifact_signature,
        greit_artifact_signature_payload,
        register_greit_artifact,
    )

    reference = np.array([2.0, 3.0, 4.0], dtype=float)
    target = reference + np.array([0.2, -0.1, 0.05], dtype=float)
    base_meta = {
        "reconstruction_runtime": "single_step_cached",
        "simulation_inverse_route": "greit3d_rm",
        "rm_route_requires_artifact": True,
        "rm_auto_build": False,
        "greit_registry_auto_resolve": True,
        "greit_registry_dir": str(tmp_path),
        "difference_mode": "raw",
        "difference_orientation": "target_minus_reference",
        "device": "cpu",
        "n_elec": 3,
        "n_rings": 1,
        "radius": 1.0,
        "height": 1.0,
        "electrode_height_ratio": 0.2,
        "electrode_level_fractions": (0.25, 0.75),
        "electrode_layout": "ring_major",
        "measurement_protocol": "eidors_full_3d",
        "stim_pattern": "{ad}",
        "meas_pattern": "{ad}",
        "background_conductivity": 1.0,
        "contact_impedance": 0.0,
        "imgsz": (2, 1, 1),
        "target_radius": 0.2,
        "target_contrast": 1.0,
        "weight": 0.5,
        "artifact_schema": GREIT_EIDORS_HDF5_SCHEMA,
        "builder_backend": "native",
        "builder_semantic_version": "native-greit-finite-target-v1",
    }
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros_like(reference),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros_like(target),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=3,
        metadata=dict(base_meta),
    )
    runtime = rc._prepare_single_step_cached_runtime(request)
    config = rc._greit_registry_config_from_runtime(request, runtime)
    signature = greit_artifact_signature(config)
    payload = greit_artifact_signature_payload(config)
    artifact_path = tmp_path / "registered_greit.h5"
    rm = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.25],
        ],
        dtype=float,
    )
    y = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, -0.25],
        ],
        dtype=float,
    )
    d = np.eye(2, dtype=float)
    rec_model = np.array([[-0.2, -0.2, 0.0], [0.2, -0.2, 0.0]], dtype=float)
    GREITRM(
        rm=rm,
        metadata=MappingProxyType(
            {
                "algorithm": "greit-3d",
                "artifact_schema": GREIT_EIDORS_HDF5_SCHEMA,
                "artifact_format": "hdf5",
                "eidors_parity": True,
                "fixture_only": False,
                "keep_model_components": True,
                "online_hot_path": "rm_matmul",
                "greit_registry_signature": signature,
                "greit_registry_signature_payload": payload,
            }
        ),
        voxel_shape=(2, 1, 1),
        y=y,
        d=d,
        rec_model=rec_model,
    ).save(artifact_path)
    registered = register_greit_artifact(config, artifact_path, registry_dir=tmp_path)
    assert registered.signature == signature

    def _unexpected_context(*_args, **_kwargs):
        raise AssertionError("GREIT registry hit must not build context.")

    def _unexpected_runner():
        raise AssertionError("GREIT registry hit must not import GN runner.")

    monkeypatch.setattr(rc, "_ensure_single_step_cached_context", _unexpected_context)
    monkeypatch.setattr(rc, "_load_gn_difference_runner_module", _unexpected_runner)

    result = rc._run_single_step_cached_request(request)

    expected = rm @ (target - reference)
    np.testing.assert_allclose(result.conductivity, expected)
    assert result.metadata["greit_registry_signature"] == signature
    assert result.metadata["greit_registry_cache_status"] == "disk_hit"
    assert result.metadata["rm_artifact_path"] == str(artifact_path)

    bad_request = rc.ReconstructionRequest(
        reference_frame=request.reference_frame,
        target_frame=request.target_frame,
        mesh_dimension=3,
        metadata={**base_meta, "n_rings": 2},
    )
    bad_result = rc._run_single_step_cached_request(bad_request)
    assert bad_result.error_msg
    assert bad_result.metadata["rm_artifact_missing"] is True


def test_single_step_cached_request_uses_hardware_drive_metadata_for_context_and_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.common.gn_difference_runner as diff_runner

    base_meas = np.array([1.0, 2.0, 3.0], dtype=float)
    pred_diff = np.array([0.1, 0.2, 0.3], dtype=float)
    delta_sigma = np.array([0.4], dtype=float)
    cache_keys: list[tuple[object, ...]] = []
    build_kwargs: list[dict[str, object]] = []

    class _StubForwardModel:
        def fwd_solve(self, image):
            assert image.elem_data is not None
            return SimpleNamespace(meas=base_meas + pred_diff), None

    def _fake_build_shared_context(**kwargs):
        build_kwargs.append(dict(kwargs))
        return {
            "mesh": object(),
            "display_node_coords": np.array(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=float,
            ),
            "display_cell_connectivity": np.array([[0, 1, 2]], dtype=int),
            "operator_bundle": {
                "strict_solver_backend_effective": diff_runner.STRICT_SOLVER_BACKEND_MEASUREMENT,
            },
            "sigma_bg": np.ones_like(delta_sigma),
            "fwd_model": _StubForwardModel(),
            "base_meas": base_meas,
            "cache_build_seconds": {},
            "cache_miss_reasons": {},
            "cache_manager": None,
        }

    monkeypatch.setattr(
        rc,
        "_get_cached_fast_context",
        lambda cache_key: cache_keys.append(cache_key) or None,
    )
    monkeypatch.setattr(rc, "_put_cached_fast_context", lambda _cache_key, _ctx: None)
    monkeypatch.setattr(diff_runner, "build_shared_context", _fake_build_shared_context)
    monkeypatch.setattr(
        diff_runner,
        "_measurement_space_delta",
        lambda operator_bundle, rhs: delta_sigma,
    )

    request_100 = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "drive_mode": "total_current",
            "stim_amp_uA": 100,
            "step_size_calib": False,
        },
    )
    request_200 = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "drive_mode": "total_current",
            "stim_amp_uA": 200,
            "step_size_calib": False,
        },
    )

    result_100 = rc._run_single_step_cached_request(request_100)
    result_200 = rc._run_single_step_cached_request(request_200)

    assert len(cache_keys) == 2
    assert cache_keys[0] != cache_keys[1]
    assert build_kwargs[0]["drive_mode"] == "total_current"
    assert build_kwargs[0]["drive_value"] == pytest.approx(100e-6)
    assert build_kwargs[1]["drive_mode"] == "total_current"
    assert build_kwargs[1]["drive_value"] == pytest.approx(200e-6)
    assert result_100.metadata["drive_mode"] == "total_current"
    assert result_100.metadata["drive_value"] == pytest.approx(100e-6)
    assert result_200.metadata["drive_value"] == pytest.approx(200e-6)
    assert (
        build_kwargs[0]["single_step_algorithm_version"]
        == result_100.metadata["single_step_algorithm_version"]
    )
    assert (
        build_kwargs[0]["single_step_jacobian_math_convention"]
        == result_100.metadata["single_step_jacobian_math_convention"]
    )
    assert (
        build_kwargs[0]["single_step_projection_math_convention"]
        == result_100.metadata["single_step_projection_math_convention"]
    )
    assert (
        build_kwargs[0]["single_step_operator_math_convention"]
        == result_100.metadata["single_step_operator_math_convention"]
    )


def test_single_step_cached_request_scales_absolute_display_by_calibrated_alpha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.common.gn_difference_runner as diff_runner

    delta_sigma = np.array([2.0, -2.0], dtype=float)
    base_meas = np.array([1.0, 2.0, 3.0], dtype=float)
    captured_sigmas: list[np.ndarray] = []

    class _StubForwardModel:
        def fwd_solve(self, image):
            sigma = np.asarray(image.elem_data, dtype=float)
            captured_sigmas.append(sigma.copy())
            pred = np.array([sigma[0], sigma[1], sigma.mean()], dtype=float)
            return SimpleNamespace(meas=base_meas + pred), None

    ctx = {
        "mesh": object(),
        "display_node_coords": np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=float,
        ),
        "display_cell_connectivity": np.array([[0, 1, 2]], dtype=int),
        "operator_bundle": {
            "strict_solver_backend_effective": diff_runner.STRICT_SOLVER_BACKEND_MEASUREMENT,
        },
        "sigma_bg": np.ones_like(delta_sigma),
        "fwd_model": _StubForwardModel(),
        "base_meas": base_meas,
        "cache_build_seconds": {},
        "cache_miss_reasons": {},
        "cache_manager": None,
    }
    request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={"reconstruction_runtime": "single_step_cached"},
    )

    monkeypatch.setattr(rc, "_get_cached_fast_context", lambda _cache_key: ctx)
    monkeypatch.setattr(rc, "_put_cached_fast_context", lambda _cache_key, _ctx: None)
    monkeypatch.setattr(
        diff_runner,
        "_measurement_space_delta",
        lambda operator_bundle, rhs: delta_sigma,
    )
    monkeypatch.setattr(
        diff_runner,
        "_calibrate_step_size",
        lambda **kwargs: 0.25,
    )

    result = rc._run_single_step_cached_request(request)

    expected_update = delta_sigma * 0.25
    assert np.allclose(result.conductivity, np.ones_like(delta_sigma) + expected_update)
    assert np.allclose(captured_sigmas[-1], np.ones_like(delta_sigma) + expected_update)
    assert result.metadata["step_size_alpha"] == pytest.approx(0.25)


def _make_measurement_frame(real_values: list[float], index: int) -> FrameData:
    real = np.asarray(real_values, dtype=float)
    return FrameData(
        real=real,
        imag=np.zeros_like(real),
        timestamp=0.0,
        frame_index=index,
    )


def _make_alpha_policy_ctx(delta_sigma: np.ndarray, base_meas: np.ndarray) -> dict:
    import scripts.common.gn_difference_runner as diff_runner

    class _StubForwardModel:
        def fwd_solve(self, image):
            assert image.elem_data is not None
            return SimpleNamespace(meas=base_meas + 1.0), None

    return {
        "mesh": object(),
        "display_node_coords": np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=float,
        ),
        "display_cell_connectivity": np.array([[0, 1, 2]], dtype=int),
        "operator_bundle": {
            "strict_solver_backend_effective": diff_runner.STRICT_SOLVER_BACKEND_MEASUREMENT,
        },
        "sigma_bg": np.ones_like(delta_sigma),
        "fwd_model": _StubForwardModel(),
        "base_meas": base_meas,
        "cache_build_seconds": {},
        "cache_miss_reasons": {},
        "cache_manager": None,
    }


def _patch_alpha_policy_runtime(
    monkeypatch: pytest.MonkeyPatch,
    ctx: dict,
    delta_sigma: np.ndarray,
    calibrate_calls: list[dict],
    *,
    alpha: float = 0.25,
) -> None:
    import scripts.common.gn_difference_runner as diff_runner

    def _fake_calibrate(**kwargs):
        calibrate_calls.append(kwargs)
        return alpha

    monkeypatch.setattr(rc, "_get_cached_fast_context", lambda _cache_key: ctx)
    monkeypatch.setattr(rc, "_put_cached_fast_context", lambda _cache_key, _ctx: None)
    monkeypatch.setattr(
        diff_runner,
        "_measurement_space_delta",
        lambda operator_bundle, rhs: delta_sigma,
    )
    monkeypatch.setattr(diff_runner, "_calibrate_step_size", _fake_calibrate)


def _make_alpha_policy_request(
    reference: FrameData,
    target: FrameData,
    **meta_overrides: object,
) -> "rc.ReconstructionRequest":
    metadata: dict[str, object] = {"reconstruction_runtime": "single_step_cached"}
    metadata.update(meta_overrides)
    return rc.ReconstructionRequest(
        reference_frame=reference,
        target_frame=target,
        metadata=metadata,
    )


def test_single_step_cached_live_request_reuses_calibrated_alpha_per_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delta_sigma = np.array([0.5, -0.5], dtype=float)
    base_meas = np.array([1.0, 2.0, 3.0], dtype=float)
    ctx = _make_alpha_policy_ctx(delta_sigma, base_meas)
    calibrate_calls: list[dict] = []
    _patch_alpha_policy_runtime(monkeypatch, ctx, delta_sigma, calibrate_calls)

    reference = _make_measurement_frame([1.0, 2.0, 3.0], 0)
    first = rc._run_single_step_cached_request(
        _make_alpha_policy_request(
            reference,
            _make_measurement_frame([1.5, 2.5, 3.5], 1),
            request_source="hardware_auto_live",
        )
    )
    second = rc._run_single_step_cached_request(
        _make_alpha_policy_request(
            reference,
            _make_measurement_frame([2.0, 3.0, 4.0], 2),
            request_source="hardware_auto_live",
        )
    )

    assert len(calibrate_calls) == 1
    assert first.metadata["step_size_calib_policy"] == "cached_reference"
    assert first.metadata["step_size_alpha_source"] == "calibrated"
    assert first.metadata["step_size_alpha"] == pytest.approx(0.25)
    assert second.metadata["step_size_alpha_source"] == "cached"
    assert second.metadata["step_size_alpha"] == pytest.approx(0.25)


def test_single_step_cached_live_request_recalibrates_when_reference_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delta_sigma = np.array([0.5, -0.5], dtype=float)
    base_meas = np.array([1.0, 2.0, 3.0], dtype=float)
    ctx = _make_alpha_policy_ctx(delta_sigma, base_meas)
    calibrate_calls: list[dict] = []
    _patch_alpha_policy_runtime(monkeypatch, ctx, delta_sigma, calibrate_calls)

    target = _make_measurement_frame([1.5, 2.5, 3.5], 2)
    results = [
        rc._run_single_step_cached_request(
            _make_alpha_policy_request(
                _make_measurement_frame(reference, index),
                target,
                request_source="hardware_auto_live",
            )
        )
        for index, reference in enumerate(
            ([1.0, 2.0, 3.0], [9.0, 9.0, 9.0]),
        )
    ]

    assert len(calibrate_calls) == 2
    assert [r.metadata["step_size_alpha_source"] for r in results] == [
        "calibrated",
        "calibrated",
    ]


def test_single_step_cached_live_request_recalibrates_after_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delta_sigma = np.array([0.5, -0.5], dtype=float)
    base_meas = np.array([1.0, 2.0, 3.0], dtype=float)
    ctx = _make_alpha_policy_ctx(delta_sigma, base_meas)
    calibrate_calls: list[dict] = []
    _patch_alpha_policy_runtime(monkeypatch, ctx, delta_sigma, calibrate_calls)

    reference = _make_measurement_frame([1.0, 2.0, 3.0], 0)
    sources = []
    for index in range(3):
        result = rc._run_single_step_cached_request(
            _make_alpha_policy_request(
                reference,
                _make_measurement_frame([1.5 + index, 2.5, 3.5], index + 1),
                request_source="hardware_auto_live",
                step_size_recalib_interval=2,
            )
        )
        sources.append(result.metadata["step_size_alpha_source"])

    assert sources == ["calibrated", "cached", "calibrated"]
    assert len(calibrate_calls) == 2


def test_single_step_cached_request_default_policy_calibrates_every_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delta_sigma = np.array([0.5, -0.5], dtype=float)
    base_meas = np.array([1.0, 2.0, 3.0], dtype=float)
    ctx = _make_alpha_policy_ctx(delta_sigma, base_meas)
    calibrate_calls: list[dict] = []
    _patch_alpha_policy_runtime(monkeypatch, ctx, delta_sigma, calibrate_calls)

    reference = _make_measurement_frame([1.0, 2.0, 3.0], 0)
    for index in range(2):
        result = rc._run_single_step_cached_request(
            _make_alpha_policy_request(
                reference,
                _make_measurement_frame([1.5 + index, 2.5, 3.5], index + 1),
            )
        )
        assert result.metadata["step_size_calib_policy"] == "always"
        assert result.metadata["step_size_alpha_source"] == "calibrated"

    assert len(calibrate_calls) == 2
    assert "step_size_alpha_cache" not in ctx


def test_single_step_cached_request_skip_simulated_avoids_fwd_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delta_sigma = np.array([0.5, -0.5], dtype=float)
    base_meas = np.array([1.0, 2.0, 3.0], dtype=float)
    ctx = _make_alpha_policy_ctx(delta_sigma, base_meas)

    class _NoSolveForwardModel:
        def fwd_solve(self, image):
            raise AssertionError("fwd_solve must be skipped")

    ctx["fwd_model"] = _NoSolveForwardModel()
    calibrate_calls: list[dict] = []
    _patch_alpha_policy_runtime(monkeypatch, ctx, delta_sigma, calibrate_calls)

    result = rc._run_single_step_cached_request(
        _make_alpha_policy_request(
            _make_measurement_frame([1.0, 2.0, 3.0], 0),
            _make_measurement_frame([1.5, 2.5, 3.5], 1),
            step_size_calib=False,
            single_step_skip_simulated=True,
        )
    )

    assert result.error_msg is None
    assert result.simulated is None
    assert np.allclose(result.conductivity, np.ones_like(delta_sigma) + delta_sigma)


def test_single_step_cached_request_warmup_only_primes_context_without_solving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.common.gn_difference_runner as diff_runner

    build_calls: list[dict[str, object]] = []

    def _fake_build_shared_context(**kwargs):
        build_calls.append(dict(kwargs))
        return {
            "mesh": object(),
            "display_node_coords": np.array(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=float,
            ),
            "display_cell_connectivity": np.array([[0, 1, 2]], dtype=int),
            "operator_bundle": {
                "strict_solver_backend_effective": diff_runner.STRICT_SOLVER_BACKEND_MEASUREMENT,
            },
            "sigma_bg": np.array([1.0], dtype=float),
            "fwd_model": object(),
            "base_meas": np.array([0.0, 0.0, 0.0], dtype=float),
            "cache_build_seconds": {"mesh": 0.1},
            "cache_miss_reasons": {},
            "cache_manager": None,
        }

    monkeypatch.setattr(rc, "_get_cached_fast_context", lambda _cache_key: None)
    monkeypatch.setattr(rc, "_put_cached_fast_context", lambda _cache_key, _ctx: None)
    monkeypatch.setattr(diff_runner, "build_shared_context", _fake_build_shared_context)
    monkeypatch.setattr(
        diff_runner,
        "_measurement_space_delta",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("warmup should not solve")
        ),
    )

    request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "warmup_only": True,
        },
    )

    result = rc._run_single_step_cached_request(request)

    assert len(build_calls) == 1
    assert result.conductivity.size == 0
    assert result.metadata["cache_warmup_only"] is True
    assert (
        result.metadata["solver_diagnostics"]["strict_solver_backend_effective"]
        == "warmup_only"
    )


def test_single_step_cached_3d_context_uses_total_current_multiring_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.common.gn_difference_runner as diff_runner

    build_calls: list[dict[str, object]] = []

    def _fake_build_shared_context(**kwargs):
        build_calls.append(dict(kwargs))
        return {
            "mesh": object(),
            "display_node_coords": np.array(
                [[0.0, 0.0, -0.5], [1.0, 0.0, 0.5], [0.0, 1.0, 0.5]],
                dtype=float,
            ),
            "display_cell_connectivity": np.array([[0, 1, 2]], dtype=int),
            "operator_bundle": {
                "strict_solver_backend_effective": diff_runner.STRICT_SOLVER_BACKEND_MEASUREMENT,
            },
            "sigma_bg": np.array([1.0], dtype=float),
            "fwd_model": object(),
            "base_meas": np.array([0.0, 0.0, 0.0], dtype=float),
            "cache_build_seconds": {},
            "cache_miss_reasons": {},
            "cache_manager": None,
        }

    monkeypatch.setattr(rc, "_get_cached_fast_context", lambda _cache_key: None)
    monkeypatch.setattr(rc, "_put_cached_fast_context", lambda _cache_key, _ctx: None)
    monkeypatch.setattr(diff_runner, "build_shared_context", _fake_build_shared_context)

    request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        mesh_dimension=3,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "warmup_only": True,
            "mesh_dimension": 3,
            "n_elec": 8,
            "n_rings": 2,
            "drive_mode": "line_current_density",
        },
    )

    result = rc._run_single_step_cached_request(request)

    assert result.metadata["cache_warmup_only"] is True
    assert len(build_calls) == 1
    assert build_calls[0]["mesh_dim"] == 3
    assert build_calls[0]["n_elec"] == 8
    assert build_calls[0]["n_rings"] == 2
    assert build_calls[0]["drive_mode"] == "total_current"
    assert build_calls[0]["jacobian_representation"] == "linearized"


def test_gn_difference_runner_3d_multiring_loads_ring_ordered_mesh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.common.gn_difference_runner as diff_runner

    captured: dict[str, object] = {}

    def _fake_load_or_create_mesh(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after mesh kwargs")

    monkeypatch.setattr(diff_runner, "load_or_create_mesh", _fake_load_or_create_mesh)

    with pytest.raises(RuntimeError, match="stop after mesh kwargs"):
        diff_runner.build_shared_context(
            mesh_dir=str(tmp_path),
            mesh_name=None,
            mesh_dim=3,
            mesh_height=0.16,
            electrode_height_ratio=0.2,
            z_center=0.0,
            electrode_level_fractions=(0.25, 0.75),
            refinement=2,
            n_elec=8,
            n_rings=2,
            radius=0.18,
            drive_mode="line_current_density",
            drive_value=1.0e-5,
            solver_mode="fast",
        )

    assert captured["n_elec"] == 16
    assert captured["dimension"] == 3
    assert captured["electrode_layout"] == "ring_major"


def test_gn_difference_runner_complex_rm_build_only_preserves_complex_dtype(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.common.gn_difference_runner as diff_runner

    captured: dict[str, object] = {}
    node_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    cells = np.array([[0, 1, 2, 3]], dtype=np.int32)
    fake_mesh = SimpleNamespace(
        coordinates=lambda: node_coords,
        cells=lambda: cells,
        _pyeidors_mesh_cache_hit=False,
        _pyeidors_mesh_cache_layer="test",
        _pyeidors_mesh_cache_name="complex-rm-test",
    )

    class _FakeIndexMap:
        size_local = 2

    class _FakeDofMap:
        index_map = _FakeIndexMap()
        index_map_bs = 1

    class _FakeFunctionSpace:
        dofmap = _FakeDofMap()

    class _FakePatternManager:
        n_stim = 1
        n_meas_total = 2
        n_meas_per_stim = [2]

    class _FakeForwardModel:
        def __init__(self, **kwargs):
            captured["z"] = np.asarray(kwargs["z"])
            self.V_sigma = _FakeFunctionSpace()
            self.pattern_manager = _FakePatternManager()
            self._petsc_backend_info = {"petsc_device_effective": "cpu"}
            self._last_cache_lookup = {}

        def fwd_solve(self, _img):
            return (
                SimpleNamespace(
                    meas=np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64)
                ),
                None,
            )

    class _FakeJacobianAdapter:
        def __init__(self, *_args, **_kwargs):
            pass

        def calculate_from_image(self, _img):
            return np.array(
                [[1.0 + 0.5j, 2.0 - 1.0j], [0.25j, 3.0 + 0.75j]],
                dtype=np.complex64,
            )

    class _FakeCacheManager:
        def __init__(self, *_args, **_kwargs):
            pass

        def clear_name(self, *_args, **_kwargs):
            return None

        def get_or_compute_semantic(self, **kwargs):
            return (
                kwargs["compute_fn"](),
                SimpleNamespace(
                    hit=False,
                    layer="compute",
                    artifact=kwargs["artifact"],
                    key=kwargs["name"],
                ),
            )

        def stats(self):
            return {}

    monkeypatch.setattr(diff_runner, "CacheManager", _FakeCacheManager)
    monkeypatch.setattr(diff_runner, "load_or_create_mesh", lambda **_kwargs: fake_mesh)
    monkeypatch.setattr(diff_runner, "EITForwardModel", _FakeForwardModel)
    monkeypatch.setattr(diff_runner, "EidorsJacobianAdapter", _FakeJacobianAdapter)
    monkeypatch.setattr(
        diff_runner, "model_signature_from_forward_model", lambda _f: "m"
    )
    monkeypatch.setattr(
        diff_runner, "pattern_signature_from_forward_model", lambda _f: "p"
    )
    monkeypatch.setattr(
        diff_runner, "backend_signature_from_forward_model", lambda _f: "b"
    )
    monkeypatch.setattr(diff_runner, "detect_performance_capabilities", lambda: {})
    monkeypatch.setattr(
        diff_runner,
        "resolve_torch_device",
        lambda *_args, **_kwargs: SimpleNamespace(
            requested="cpu",
            effective="cpu",
            torch_device="cpu",
        ),
    )

    ctx = diff_runner.build_shared_context(
        mesh_dir=str(tmp_path),
        mesh_name=None,
        mesh_dim=3,
        mesh_height=0.16,
        electrode_height_ratio=0.2,
        z_center=0.0,
        refinement=2,
        n_elec=8,
        n_rings=2,
        radius=0.18,
        drive_mode="total_current",
        drive_value=1.0e-5,
        contact_impedance=0.01 + 0.05j,
        background_sigma=1.0 + 2.0j,
        solver_mode="fast",
        scalar_dtype="complex64",
        rm_build_only=True,
    )

    assert np.asarray(captured["z"]).dtype == np.dtype(np.complex64)
    assert ctx["sigma_bg"].dtype == np.dtype(np.complex64)
    assert ctx["base_meas"].dtype == np.dtype(np.complex64)
    assert ctx["J"].dtype == np.dtype(np.complex64)
    assert ctx["operator_bundle"]["mode"] == "rm_build_only"
    np.testing.assert_allclose(ctx["sigma_bg"], np.array([1.0 + 2.0j] * 2))


def test_v621_gn_difference_runner_loader_uses_packaged_runtime_without_repo_scripts(
    tmp_path: Path,
) -> None:
    script = """
from eit_app.controllers import reconstruction_controller as rc

rc._load_gn_difference_runner_module.cache_clear()
module = rc._load_gn_difference_runner_module()
if module.__name__ != "pyeidors.realtime.gn_difference_runner":
    raise SystemExit(f"unexpected runner module: {module.__name__}")
    """
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[2]
    existing_pythonpath = [
        entry
        for entry in env.get("PYTHONPATH", "").split(os.pathsep)
        if entry and Path(entry).resolve() != repo_root
    ]
    env["PYTHONPATH"] = os.pathsep.join([str(repo_root / "src"), *existing_pythonpath])
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_recover_nix_runtime_site_packages_restores_missing_runtime_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_paths = [
        "/nix/store/a-python3.13-fenics-dolfinx/lib/python3.13/site-packages",
        "/nix/store/b-python3.13-fenics-ufl/lib/python3.13/site-packages",
        "/nix/store/c-petsc/lib/python3.13/site-packages",
    ]
    original_sys_path = list(sys.path)
    original_pythonpath = os.environ.get("PYTHONPATH")

    def _fake_glob(pattern: str) -> list[str]:
        if "fenics-dolfinx" in pattern:
            return [fake_paths[0]]
        if "fenics-ufl" in pattern:
            return [fake_paths[1]]
        if "petsc" in pattern:
            return [fake_paths[2]]
        return []

    monkeypatch.setattr(rc.Path, "exists", lambda self: str(self) == "/nix/store")
    monkeypatch.setattr(rc.glob, "glob", _fake_glob)
    monkeypatch.setattr(rc.os.path, "isdir", lambda path: path in fake_paths)
    sys.path[:] = [entry for entry in original_sys_path if entry not in fake_paths]
    os.environ["PYTHONPATH"] = "/tmp/original-pythonpath"
    captured_sys_path: list[str] = []
    captured_pythonpath = ""
    try:
        added = rc._recover_nix_runtime_site_packages("ufl")
        captured_sys_path = list(sys.path)
        captured_pythonpath = os.environ["PYTHONPATH"]
    finally:
        sys.path[:] = original_sys_path
        if original_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = original_pythonpath

    assert added == tuple(reversed(fake_paths))
    assert captured_sys_path[: len(fake_paths)] == fake_paths
    assert captured_pythonpath.startswith(os.pathsep.join(reversed(fake_paths)))


def test_clear_reconstruction_system_cache_clears_both_runtime_caches() -> None:
    rc._SYSTEM_CACHE[("system",)] = object()
    rc._SYSTEM_CACHE_SIZES[("system",)] = 8
    rc._FAST_CONTEXT_CACHE[("fast",)] = object()
    rc._FAST_CONTEXT_CACHE_SIZES[("fast",)] = 8

    rc.clear_reconstruction_system_cache()

    assert not rc._SYSTEM_CACHE
    assert not rc._SYSTEM_CACHE_SIZES
    assert not rc._FAST_CONTEXT_CACHE
    assert not rc._FAST_CONTEXT_CACHE_SIZES


def test_v608_reconstruction_runtime_caches_skip_oversize_entries() -> None:
    rc.clear_reconstruction_system_cache()
    try:
        system = SimpleNamespace(
            mesh=None,
            fwd_model=SimpleNamespace(z=np.ones(8, dtype=np.float64)),
            _reconstruction_system_cache_max_bytes=1,
        )
        rc._put_cached_system(("system",), system)
        assert ("system",) not in rc._SYSTEM_CACHE
        assert ("system",) not in rc._SYSTEM_CACHE_SIZES

        ctx = {
            "operator_bundle": {"J": np.ones((4, 4), dtype=np.float64)},
            "sigma_bg": np.ones(4, dtype=np.float64),
            "single_step_context_cache_max_bytes": 1,
        }
        rc._put_cached_fast_context(("ctx",), ctx)
        assert ("ctx",) not in rc._FAST_CONTEXT_CACHE
        assert ("ctx",) not in rc._FAST_CONTEXT_CACHE_SIZES
        assert ctx["single_step_context_process_cache_stored"] is False
        assert ctx["single_step_context_process_cache_skip_reason"] == "entry_too_large"
        assert ctx["single_step_context_process_cache_bytes"] > 1
    finally:
        rc.clear_reconstruction_system_cache()


def test_v608_single_step_context_cache_eviction_uses_total_bytes() -> None:
    rc.clear_reconstruction_system_cache()
    try:
        first = {
            "operator_bundle": {"J": np.ones(8, dtype=np.float64)},
            "single_step_context_cache_max_bytes": 96,
        }
        second = {
            "operator_bundle": {"J": np.ones(8, dtype=np.float64)},
            "single_step_context_cache_max_bytes": 96,
        }

        rc._put_cached_fast_context(("first",), first)
        rc._put_cached_fast_context(("second",), second)

        assert ("first",) not in rc._FAST_CONTEXT_CACHE
        assert ("second",) in rc._FAST_CONTEXT_CACHE
        assert set(rc._FAST_CONTEXT_CACHE_SIZES) == {("second",)}
        assert second["single_step_context_process_cache_stored"] is True
    finally:
        rc.clear_reconstruction_system_cache()


def test_v609_rm_fit_jacobian_process_cache_eviction_uses_total_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with rc._RM_FIT_JACOBIAN_CACHE_LOCK:
        rc._RM_FIT_JACOBIAN_CACHE.clear()
        rc._RM_FIT_JACOBIAN_CACHE_SIZES.clear()
    monkeypatch.setattr(rc, "_RM_FIT_JACOBIAN_CACHE_MAX_BYTES", 96)
    try:
        first = np.ones((1, 8), dtype=np.float64)
        second = np.full((1, 8), 2.0, dtype=np.float64)

        assert rc._put_rm_fit_jacobian_cache("first", first) == "stored"
        assert rc._put_rm_fit_jacobian_cache("second", second) == "stored"

        with rc._RM_FIT_JACOBIAN_CACHE_LOCK:
            assert "first" not in rc._RM_FIT_JACOBIAN_CACHE
            assert "second" in rc._RM_FIT_JACOBIAN_CACHE
            assert set(rc._RM_FIT_JACOBIAN_CACHE_SIZES) == {"second"}
            assert sum(rc._RM_FIT_JACOBIAN_CACHE_SIZES.values()) <= 96
    finally:
        with rc._RM_FIT_JACOBIAN_CACHE_LOCK:
            rc._RM_FIT_JACOBIAN_CACHE.clear()
            rc._RM_FIT_JACOBIAN_CACHE_SIZES.clear()


def test_boundary_voltage_plot_keeps_recon_overlay_visible_for_tiny_fit() -> None:
    _get_app()
    widget = BoundaryVoltagePlotWidget(mode="hardware")
    measured = np.linspace(-1.0, 1.0, 208, dtype=float)
    reconstructed = 1.0e-6 * np.sin(np.linspace(0.0, 6.0 * np.pi, 208, dtype=float))

    widget.update_hardware_voltages(measured, reconstructed)

    assert widget._curve_primary.isVisible() is True
    assert widget._curve_primary_markers.isVisible() is True
    assert widget._curve_reconstructed_outline.isVisible() is True
    assert widget._curve_reconstructed.isVisible() is True
    assert widget._curve_reconstructed_markers.isVisible() is True
    primary_marker_x, primary_marker_y = widget._curve_primary_markers.getData()
    marker_x, marker_y = widget._curve_reconstructed_markers.getData()
    assert primary_marker_x is not None and primary_marker_y is not None
    assert marker_x is not None and marker_y is not None
    assert len(primary_marker_x) == measured.size
    assert len(marker_x) == reconstructed.size
    np.testing.assert_allclose(primary_marker_x, np.arange(1, measured.size + 1))
    np.testing.assert_allclose(primary_marker_y, measured)
    np.testing.assert_allclose(marker_y, reconstructed)
    assert float(marker_x[0]) == pytest.approx(1.0)
    assert float(marker_x[-1]) == pytest.approx(208.0)


def test_live_plot_marks_every_real_and_imag_channel_point() -> None:
    _get_app()
    widget = LivePlotWidget()
    frame = FrameData(
        real=np.linspace(-1.0, 1.0, 9, dtype=float),
        imag=np.linspace(2.0, -2.0, 9, dtype=float),
        timestamp=1.0,
        frame_index=0,
    )

    widget.update_frame(frame)

    real_marker_x, real_marker_y = widget._curve_real_markers.getData()
    imag_marker_x, imag_marker_y = widget._curve_imag_markers.getData()
    expected_x = np.arange(1, frame.real.size + 1, dtype=float)
    assert real_marker_x is not None and real_marker_y is not None
    assert imag_marker_x is not None and imag_marker_y is not None
    assert widget._curve_real_markers.isVisible() is True
    assert widget._curve_imag_markers.isVisible() is False
    np.testing.assert_allclose(real_marker_x, expected_x)
    np.testing.assert_allclose(real_marker_y, frame.real)
    np.testing.assert_allclose(imag_marker_x, expected_x)
    np.testing.assert_allclose(imag_marker_y, frame.imag)

    widget._show_imag.setChecked(True)
    _get_app().processEvents()

    assert widget._curve_imag_markers.isVisible() is True


def test_boundary_voltage_plot_truth_is_not_hidden_by_recon_outline() -> None:
    _get_app()
    widget = BoundaryVoltagePlotWidget(mode="simulation")

    widget.update_simulation_voltages(
        np.linspace(-1.0, 1.0, 16, dtype=float),
        np.linspace(-1.0, 1.0, 16, dtype=float),
    )

    assert widget._curve_primary.isVisible() is True
    assert widget._curve_reconstructed_outline.isVisible() is True
    assert widget._curve_reconstructed_outline.zValue() < widget._curve_primary.zValue()
    assert widget._curve_primary.zValue() < widget._curve_reconstructed.zValue()


def test_boundary_voltage_plot_rescales_y_range_for_new_simulation_data() -> None:
    _get_app()
    widget = BoundaryVoltagePlotWidget(mode="simulation")
    widget._plot_widget.setYRange(1000.0, 2000.0, padding=0.0)

    truth = np.array([1.0e-6, 2.0e-6, 3.0e-6], dtype=float)
    reconstructed = np.array([1.5e-6, -4.0e-6, 2.5e-6], dtype=float)
    widget.update_simulation_voltages(truth, reconstructed)

    _x_range, y_range = widget._plot_widget.getPlotItem().getViewBox().viewRange()
    assert y_range[0] < -4.0e-6
    assert y_range[1] > 3.0e-6
    primary_x, primary_y = widget._curve_primary.getData()
    recon_x, recon_y = widget._curve_reconstructed.getData()
    assert primary_x is not None and primary_y is not None
    assert recon_x is not None and recon_y is not None
    np.testing.assert_allclose(primary_y, truth)
    np.testing.assert_allclose(recon_y, reconstructed)


def test_boundary_voltage_plot_hides_recon_overlay_without_fit_data() -> None:
    _get_app()
    widget = BoundaryVoltagePlotWidget(mode="hardware")

    widget.update_hardware_voltages(np.linspace(-1.0, 1.0, 16, dtype=float), None)

    assert widget._curve_reconstructed_outline.isVisible() is False
    assert widget._curve_reconstructed.isVisible() is False
    assert widget._curve_reconstructed_markers.isVisible() is False
