"""Extra edge-path coverage for visualization helpers."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

import pyeidors.visualization.eit_plot_helpers as helper


def test_eidors_tick_vals_forced_high_branches(monkeypatch: pytest.MonkeyPatch):
    original_floor = helper.np.floor

    def _run_with_forced_second_floor(second_value: float):
        calls = {"n": 0}

        def fake_floor(x):
            calls["n"] += 1
            if calls["n"] == 1:
                return 0.0
            if calls["n"] == 2:
                return second_value
            if calls["n"] == 3:
                return 0.0
            return original_floor(x)

        monkeypatch.setattr(helper.np, "floor", fake_floor)
        ticks = helper.eidors_tick_vals(1.0, 0.0)
        assert ticks.size > 0

    _run_with_forced_second_floor(16.0)
    _run_with_forced_second_floor(12.0)
    _run_with_forced_second_floor(8.0)


def test_apply_ticks_and_matlab_short_formatter_edge_paths(
    monkeypatch: pytest.MonkeyPatch,
):
    cbar = mock.MagicMock()
    monkeypatch.setattr(
        helper, "eidors_tick_vals", lambda *args, **kwargs: np.array([], dtype=float)
    )
    helper.apply_eidors_ticks(cbar, vmin=-1.0, vmax=1.0)
    cbar.set_ticks.assert_not_called()

    cbar2 = mock.MagicMock()
    cbar2.ax.yaxis.get_offset_text.return_value = mock.MagicMock()
    helper.format_colorbar(cbar2, "matlab_short")
    assert cbar2.formatter(0.0, 0) == "0.0000"
    assert "e" in cbar2.formatter(1e5, 0)


def _make_overlay_mesh(
    *,
    association_table,
    facet_tags,
    coords: np.ndarray,
    connectivity_obj,
):
    raw_mesh = SimpleNamespace(
        topology=SimpleNamespace(
            dim=2,
            create_connectivity=lambda *_args: None,
            connectivity=lambda *_args: connectivity_obj,
        ),
        comm=SimpleNamespace(allreduce=lambda value, op=None: value),
    )
    return SimpleNamespace(
        association_table=association_table,
        facet_tags=facet_tags,
        mesh=raw_mesh,
        coordinates=lambda: np.asarray(coords, dtype=float),
    )


def test_extract_tags_and_overlay_error_paths(monkeypatch: pytest.MonkeyPatch):
    tags = helper.extract_electrode_tags(
        SimpleNamespace(association_table={2: 10, np.int32(5): 11, "x": "bad"})
    )
    assert tags == [10, 11]

    mesh_no_tags = _make_overlay_mesh(
        association_table={},
        facet_tags=None,
        coords=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        connectivity_obj=None,
    )
    with pytest.raises(RuntimeError, match="No electrode tags"):
        helper.overlay_electrode_labels(mock.MagicMock(), mesh_no_tags)

    mesh_no_facet = _make_overlay_mesh(
        association_table={"electrode_1": 2},
        facet_tags=None,
        coords=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        connectivity_obj=None,
    )
    with pytest.raises(RuntimeError, match="no facet tags"):
        helper.overlay_electrode_labels(mock.MagicMock(), mesh_no_facet)

    mesh_no_conn = _make_overlay_mesh(
        association_table={"electrode_1": 2},
        facet_tags=SimpleNamespace(
            indices=np.array([0], dtype=int), values=np.array([2], dtype=int)
        ),
        coords=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        connectivity_obj=None,
    )
    with pytest.raises(RuntimeError, match="Cannot read facet->vertex connectivity"):
        helper.overlay_electrode_labels(mock.MagicMock(), mesh_no_conn)


def test_overlay_electrode_labels_handles_empty_segments_and_zero_norm(
    monkeypatch: pytest.MonkeyPatch,
):
    def _fail_vstack(*_args, **_kwargs):
        raise AssertionError(
            "electrode label centroid assembly must not call np.vstack"
        )

    monkeypatch.setattr(helper.np, "vstack", _fail_vstack)
    assert "np.vstack" not in inspect.getsource(helper.overlay_electrode_labels)

    class _Connectivity:
        @staticmethod
        def links(_idx):
            return np.array([0, 1], dtype=np.int32)

    monkeypatch.setattr(
        helper.ufl, "Measure", lambda *args, **kwargs: lambda tag: float(tag)
    )
    monkeypatch.setattr(
        helper.fem,
        "Constant",
        lambda _mesh, value: helper._real_scalar(value, name="unit constant"),
    )
    monkeypatch.setattr(helper.fem, "form", lambda expr: expr)
    monkeypatch.setattr(helper.fem, "assemble_scalar", lambda expr: expr)

    mesh_empty = _make_overlay_mesh(
        association_table={"electrode_1": 2},
        facet_tags=SimpleNamespace(
            indices=np.array([0], dtype=int), values=np.array([99], dtype=int)
        ),
        coords=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        connectivity_obj=_Connectivity(),
    )
    ax_empty = mock.MagicMock()
    helper.overlay_electrode_labels(ax_empty, mesh_empty)
    ax_empty.plot.assert_not_called()
    ax_empty.text.assert_not_called()

    mesh_zero_norm = _make_overlay_mesh(
        association_table={"electrode_1": 2},
        facet_tags=SimpleNamespace(
            indices=np.array([0], dtype=int), values=np.array([2], dtype=int)
        ),
        coords=np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float),
        connectivity_obj=_Connectivity(),
    )
    ax_zero = mock.MagicMock()
    helper.overlay_electrode_labels(ax_zero, mesh_zero_norm)
    ax_zero.plot.assert_called()
    ax_zero.text.assert_called()
