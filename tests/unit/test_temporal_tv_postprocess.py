"""Tests for temporal smoothing + TV postprocess pipeline."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import pyeidors.inverse.postprocess.temporal as temporal_module
from pyeidors.inverse import (
    VoxelGrid,
    exponential_smooth_frames,
    moving_average_frames,
    postprocess_rm_frames,
)


def test_moving_average_frames_is_causal() -> None:
    frames = np.array(
        [
            [1.0, 10.0],
            [3.0, 20.0],
            [5.0, 40.0],
            [7.0, 80.0],
        ],
        dtype=float,
    )

    actual = moving_average_frames(frames, window=2)
    expected = np.array(
        [
            [1.0, 10.0],
            [2.0, 15.0],
            [4.0, 30.0],
            [6.0, 60.0],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(actual, expected)


def test_exponential_smooth_frames_uses_previous_output() -> None:
    frames = np.array(
        [
            [0.0, 10.0],
            [10.0, 20.0],
            [10.0, 40.0],
        ],
        dtype=float,
    )

    actual = exponential_smooth_frames(frames, alpha=0.25)
    expected = np.array(
        [
            [0.0, 10.0],
            [2.5, 12.5],
            [4.375, 19.375],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(actual, expected)


def test_postprocess_rm_frames_runs_temporal_then_3d_voxel_tv() -> None:
    mesh = VoxelGrid.from_bounds([0.0, 0.0, 0.0], [2.0, 2.0, 1.0], shape=(2, 2, 1))
    frames = np.array(
        [
            [0.0, 2.0, 0.0, 2.0],
            [2.0, 0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0, 2.0],
        ],
        dtype=float,
    )

    result = postprocess_rm_frames(
        frames,
        mesh,
        temporal="moving_average",
        moving_window=2,
        apply_tv=True,
        tv_weight=0.2,
        tv_max_iterations=30,
        tv_tolerance=1.0e-4,
        return_metadata=True,
    )

    assert result.values.shape == frames.shape
    assert result.metadata["schema"] == "pyeidors-rm-postprocess-v1"
    assert result.metadata["temporal"] == "moving_average"
    assert result.metadata["apply_tv"] is True
    assert result.metadata["tv_frame_count"] == frames.shape[0]
    assert result.metadata["n_parameters"] == mesh.num_cells()
    assert len(result.metadata["tv"]) == frames.shape[0]
    assert result.metadata["tv"][0]["method"] == "tv-pdhg"

    smoothed_only = moving_average_frames(frames, window=2)
    assert np.linalg.norm(result.values[1] - smoothed_only[1]) < np.linalg.norm(
        frames[1] - smoothed_only[1]
    )


def test_v293_postprocess_tv_direct_fills_refined_frame_rows(monkeypatch) -> None:
    def fail_vstack(*_args, **_kwargs):
        raise AssertionError("temporal TV postprocess must not call np.vstack")

    monkeypatch.setattr(temporal_module.np, "vstack", fail_vstack)
    mesh = VoxelGrid.from_bounds([0.0], [3.0], shape=(3,))
    frames = np.array(
        [[0.0, 2.0, 0.0], [1.0, 0.0, 1.0], [0.0, 2.0, 0.0]],
        dtype=float,
    )

    result = temporal_module.postprocess_rm_frames(
        frames,
        mesh,
        temporal="moving_average",
        moving_window=2,
        apply_tv=True,
        tv_weight=0.1,
        tv_max_iterations=5,
        tv_tolerance=1.0e-3,
        return_metadata=True,
    )

    assert result.values.shape == frames.shape
    assert result.metadata["tv_frame_count"] == frames.shape[0]
    assert "np.vstack" not in inspect.getsource(temporal_module.postprocess_rm_frames)


def test_v486_exponential_initial_uses_bounded_finite_scan() -> None:
    source = inspect.getsource(temporal_module.exponential_smooth_frames)

    assert "all_finite_values(previous)" in source
    assert "np.isfinite(previous).all()" not in source


def test_v554_temporal_postprocess_preserves_float32_and_direct_fills_ma() -> None:
    source = inspect.getsource(temporal_module.moving_average_frames)
    assert "denom =" not in source
    assert "csum[indices + 1]" not in source
    assert "dtype=np.float64" not in source

    frames = np.array(
        [
            [1.0, 10.0],
            [3.0, 20.0],
            [5.0, 40.0],
        ],
        dtype=np.float32,
    )
    mesh = VoxelGrid.from_bounds([0.0], [2.0], shape=(2,))

    moving = moving_average_frames(frames, window=2)
    exponential = exponential_smooth_frames(frames, alpha=0.25)
    postprocessed = postprocess_rm_frames(
        frames[:1],
        mesh,
        temporal="none",
        apply_tv=False,
        return_metadata=True,
    )

    assert moving.dtype == np.dtype(np.float32)
    assert exponential.dtype == np.dtype(np.float32)
    assert postprocessed.values.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(
        moving,
        np.array([[1.0, 10.0], [2.0, 15.0], [4.0, 30.0]], dtype=np.float32),
    )


def test_postprocess_rm_frames_exponential_without_tv_preserves_single_frame_shape() -> (
    None
):
    mesh = VoxelGrid.from_bounds([0.0], [4.0], shape=(4,))
    frame = np.array([0.0, 4.0, 2.0, 6.0], dtype=float)

    result = postprocess_rm_frames(
        frame,
        mesh,
        temporal="exponential",
        exponential_alpha=0.5,
        apply_tv=False,
        return_metadata=True,
    )

    assert result.values.shape == frame.shape
    np.testing.assert_allclose(result.values, frame)
    assert result.metadata["was_vector"] is True
    assert result.metadata["tv_frame_count"] == 0

    with pytest.raises(ValueError, match="temporal"):
        postprocess_rm_frames(frame, mesh, temporal="bad")
