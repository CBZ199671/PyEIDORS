"""Tests for measurement dataset helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pyeidors.data.measurement_dataset import (
    MeasurementDataset,
    _parse_bool,
    _parse_direction,
)
from pyeidors.data.structures import PatternConfig
from pyeidors.electrodes.patterns import StimMeasPatternManager


def _metadata_for(pattern_cfg: PatternConfig, n_frames: int | None = None):
    metadata = {
        "n_elec": pattern_cfg.n_elec,
        "n_rings": pattern_cfg.n_rings,
        "stim_pattern": pattern_cfg.stim_pattern,
        "meas_pattern": pattern_cfg.meas_pattern,
        "drive_mode": pattern_cfg.drive_mode,
        "drive_value": pattern_cfg.drive_value,
        "geometry_scale_to_m": pattern_cfg.geometry_scale_to_m,
        "electrode_length_m_override": pattern_cfg.electrode_length_m_override,
        "use_meas_current": pattern_cfg.use_meas_current,
        "use_meas_current_next": pattern_cfg.use_meas_current_next,
        "rotate_meas": pattern_cfg.rotate_meas,
        "stim_direction": pattern_cfg.stim_direction,
        "meas_direction": pattern_cfg.meas_direction,
        "stim_first_positive": pattern_cfg.stim_first_positive,
    }
    if n_frames is not None:
        metadata["n_frames"] = int(n_frames)
    return metadata


def test_parse_bool_and_direction():
    assert _parse_bool("yes", False) is True
    assert _parse_bool("0", True) is False
    assert _parse_bool(None, True) is True
    assert _parse_bool(0, True) is False
    with pytest.raises(ValueError):
        _parse_bool("maybe", False)

    assert _parse_direction("cw", "ccw") == "cw"
    assert _parse_direction(None, "ccw") == "ccw"
    with pytest.raises(ValueError):
        _parse_direction("left", "ccw")


def test_measurement_dataset_happy_path():
    pattern_cfg = PatternConfig(n_elec=16, stim_pattern="{ad}", meas_pattern="{ad}")
    pattern_mgr = StimMeasPatternManager(pattern_cfg)
    measurements = np.arange(pattern_mgr.n_meas_total, dtype=float)
    metadata = _metadata_for(pattern_cfg, n_frames=1)

    dataset = MeasurementDataset.from_metadata(measurements, metadata, data_type="real")
    assert dataset.measurements.shape == (1, pattern_mgr.n_meas_total)
    assert dataset.n_meas_total == pattern_mgr.n_meas_total
    assert dataset.n_stim == pattern_mgr.n_stim

    frame = dataset.to_eit_data(frame_index=0, data_type="difference")
    assert frame.meas.shape[0] == pattern_mgr.n_meas_total
    assert frame.type == "difference"

    summaries = dataset.summary()
    assert summaries["n_frames"] == 1
    assert summaries["n_elec"] == 16

    all_frames = list(dataset.iter_frames(data_type="real"))
    assert len(all_frames) == 1


def test_measurement_dataset_validation_errors():
    pattern_cfg = PatternConfig(n_elec=16, stim_pattern="{ad}", meas_pattern="{ad}")
    pattern_mgr = StimMeasPatternManager(pattern_cfg)
    metadata = _metadata_for(pattern_cfg, n_frames=1)

    wrong_cols = np.zeros((1, pattern_mgr.n_meas_total + 1))
    with pytest.raises(ValueError):
        MeasurementDataset.from_metadata(wrong_cols, metadata)

    too_many_frames = np.zeros((2, pattern_mgr.n_meas_total))
    with pytest.raises(ValueError):
        MeasurementDataset.from_metadata(too_many_frames, metadata)

    with pytest.raises(KeyError):
        MeasurementDataset.from_metadata(
            np.zeros((1, pattern_mgr.n_meas_total)),
            {"n_elec": 16},
        )

    unsupported_metadata = dict(metadata)
    unsupported_metadata["amplitude"] = 1.0
    with pytest.raises(ValueError, match="amplitude"):
        MeasurementDataset.from_metadata(np.zeros((1, pattern_mgr.n_meas_total)), unsupported_metadata)

    dataset = MeasurementDataset.from_metadata(
        np.zeros((1, pattern_mgr.n_meas_total)),
        metadata,
    )
    with pytest.raises(IndexError):
        dataset.to_eit_data(frame_index=3)

    with pytest.raises(ValueError):
        MeasurementDataset._normalize_measurements(np.zeros((2, 3, 4)))
