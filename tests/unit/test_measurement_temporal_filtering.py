"""Tests for measurement-domain temporal filtering before RM reconstruction."""

from __future__ import annotations

import numpy as np
import pytest

from pyeidors.data.difference import normalize_time_difference
from pyeidors.data.temporal_filtering import filter_measurement_frames
from pyeidors.inverse import (
    reconstruct_difference_batch,
    reconstruct_temporal_difference_batch,
)


def test_filter_measurement_frames_moving_average_is_causal_and_keeps_timestamps():
    frames = np.array(
        [
            [1.0, 10.0],
            [3.0, 20.0],
            [5.0, 40.0],
        ],
        dtype=float,
    )
    timestamps = np.array([0.0, 0.1, 0.25], dtype=float)

    result = filter_measurement_frames(
        frames,
        temporal="moving_average",
        moving_window=2,
        timestamps=timestamps,
        sample_rate_hz=20.0,
        return_metadata=True,
    )

    np.testing.assert_allclose(
        result.values,
        np.array(
            [
                [1.0, 10.0],
                [2.0, 15.0],
                [4.0, 30.0],
            ],
            dtype=float,
        ),
    )
    assert result.metadata["timestamps"] == (0.0, 0.1, 0.25)
    assert result.metadata["timestamp_policy"] == "metadata_only_no_smoothing"
    assert result.metadata["final_state"]["frame_count"] == 3
    assert result.metadata["final_state"]["history_tail"] == ((5.0, 40.0),)

    resumed = filter_measurement_frames(
        np.array([[7.0, 80.0]], dtype=float),
        temporal="moving_average",
        moving_window=2,
        initial_state=result.metadata["final_state"],
        return_metadata=True,
    )
    np.testing.assert_allclose(resumed.values, np.array([[6.0, 60.0]], dtype=float))
    assert resumed.metadata["initial_state_used"] is True


def test_filter_measurement_frames_ema_state_matches_unsplit_batch():
    frames = np.array(
        [
            [0.0, 10.0],
            [10.0, 20.0],
            [10.0, 40.0],
            [30.0, 80.0],
        ],
        dtype=float,
    )

    full = filter_measurement_frames(
        frames,
        temporal="ema",
        exponential_alpha=0.25,
        return_metadata=True,
    )
    first = filter_measurement_frames(
        frames[:2],
        temporal="ema",
        exponential_alpha=0.25,
        return_metadata=True,
    )
    second = filter_measurement_frames(
        frames[2:],
        temporal="ema",
        exponential_alpha=0.25,
        initial_state=first.metadata["final_state"],
        return_metadata=True,
    )

    np.testing.assert_allclose(np.vstack([first.values, second.values]), full.values)
    assert second.metadata["final_state"]["frame_count"] == 4


def test_filter_measurement_frames_supports_bandpass_or_lockin_hooks():
    frames = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    seen_contexts = []

    def hook(values, context):
        seen_contexts.append(dict(context))
        return values * 2.0, {"gain": 2.0}

    result = filter_measurement_frames(
        frames,
        temporal="none",
        timestamps=[0.0, 0.5],
        hook=hook,
        hook_kind="lockin",
        return_metadata=True,
    )

    np.testing.assert_allclose(result.values, frames * 2.0)
    assert seen_contexts[0]["hook_kind"] == "lockin"
    assert seen_contexts[0]["timestamps"] == (0.0, 0.5)
    assert result.metadata["hook_applied"] is True
    assert result.metadata["hook_kind"] == "lockin"
    assert result.metadata["hook_metadata"]["gain"] == 2.0

    with pytest.raises(ValueError, match="output shape"):
        filter_measurement_frames(
            frames,
            hook=lambda values, context: values[:, :1],
            hook_kind="bandpass",
        )


def test_reconstruct_temporal_difference_batch_filters_before_contract_and_rm():
    rm = np.array([[1.0, 0.0, 2.0], [-1.0, 3.0, 0.5]], dtype=float)
    reference = np.array([2.0, 4.0, 1.0], dtype=float)
    targets = np.array(
        [
            [3.0, 8.0, 2.0],
            [1.0, 2.0, 5.0],
            [4.0, 6.0, 0.5],
        ],
        dtype=float,
    )
    mask = np.array([False, True, False], dtype=bool)
    weights = np.array([4.0, 9.0, 0.25], dtype=float)

    result = reconstruct_temporal_difference_batch(
        rm,
        targets,
        normalize=True,
        v_ref=reference,
        difference_orientation="reference_minus_target",
        temporal="moving_average",
        moving_window=2,
        channel_mask=mask,
        measurement_weights=weights,
        timestamps=[0.0, 0.1, 0.3],
        device="cpu",
        return_metadata=True,
    )

    normalized = np.vstack(
        [
            normalize_time_difference(
                row,
                reference,
                orientation="reference_minus_target",
            )
            for row in targets
        ]
    )
    filtered = np.vstack(
        [
            normalized[0],
            np.mean(normalized[:2], axis=0),
            np.mean(normalized[1:3], axis=0),
        ]
    )
    expected_payload = filtered.copy()
    expected_payload[:, mask] = 0.0
    masked_weights = weights.copy()
    masked_weights[mask] = 0.0
    expected_payload *= np.sqrt(masked_weights).reshape(1, -1)

    np.testing.assert_allclose(result.values, expected_payload @ rm.T)
    assert result.metadata["online_hot_path"] == "temporal_filter_plus_rm_matmul"
    assert result.metadata["rm_online_hot_path"] == "rm_matmul"
    assert result.metadata["forward_solve_count"] == 0
    assert result.metadata["adjoint_solve_count"] == 0
    assert result.metadata["ksp_solve_count"] == 0
    assert result.metadata["jacobian_rebuild_count"] == 0
    assert result.metadata["bad_channel_count"] == 1
    assert result.metadata["measurement_weight_kind"] == "diagonal"
    assert result.metadata["timestamps"] == (0.0, 0.1, 0.3)
    assert result.metadata["timestamp_policy"] == "metadata_only_no_smoothing"
    assert result.metadata["temporal_filter_state"]["frame_count"] == 3
    assert result.metadata["offline_rm_build_seconds"] == 0.0
    assert result.metadata["online_temporal_filter_seconds"] >= 0.0
    assert result.metadata["online_rm_apply_seconds"] >= 0.0


def test_reconstruct_difference_batch_exposes_difference_orientation_contract():
    rm = np.eye(2, dtype=float)
    reference = np.array([2.0, 4.0], dtype=float)
    target = np.array([3.0, 2.0], dtype=float)

    result = reconstruct_difference_batch(
        rm,
        target,
        normalize=True,
        v_ref=reference,
        difference_orientation="reference_minus_target",
        device="cpu",
    )

    np.testing.assert_allclose(
        result,
        normalize_time_difference(
            target,
            reference,
            orientation="reference_minus_target",
        ),
    )


def test_reconstruct_temporal_difference_batch_applies_raw_hook_before_rm():
    rm = np.eye(2, dtype=float)
    frames = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=float)

    result = reconstruct_temporal_difference_batch(
        rm,
        frames,
        normalize=False,
        temporal="none",
        filter_hook=lambda values, context: (values - 1.0, {"kind": "unit-test"}),
        hook_kind="bandpass",
        device="cpu",
        return_metadata=True,
    )

    np.testing.assert_allclose(result.values, frames - 1.0)
    filter_meta = result.metadata["temporal_filter_metadata"]
    assert filter_meta["hook_applied"] is True
    assert filter_meta["hook_kind"] == "bandpass"
    assert filter_meta["hook_metadata"]["kind"] == "unit-test"
