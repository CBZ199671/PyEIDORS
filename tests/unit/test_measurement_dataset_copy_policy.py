"""Tests for MeasurementDataset copy policy semantics."""

from __future__ import annotations

import numpy as np
import pytest

from pyeidors.data.measurement_dataset import MeasurementDataset
from pyeidors.data.structures import PatternConfig
from pyeidors.electrodes.patterns import StimMeasPatternManager


def _build_dataset() -> MeasurementDataset:
    pattern_cfg = PatternConfig(n_elec=16, stim_pattern="{ad}", meas_pattern="{ad}")
    manager = StimMeasPatternManager(pattern_cfg)
    measurements = np.arange(manager.n_meas_total * 2, dtype=float).reshape(2, -1)
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
        "n_frames": 2,
    }
    return MeasurementDataset.from_metadata(measurements, metadata, data_type="real")


def test_view_policy_shares_memory_and_is_read_only():
    dataset = _build_dataset()
    eit_data = dataset.to_eit_data(copy_policy="view")

    assert np.shares_memory(eit_data.meas, dataset.measurements)
    assert np.shares_memory(eit_data.stim_pattern, dataset.stim_matrix)
    assert not eit_data.meas.flags.writeable
    assert not eit_data.stim_pattern.flags.writeable

    with pytest.raises(ValueError):
        eit_data.meas[0] = -1.0


def test_copy_policy_returns_detached_arrays():
    dataset = _build_dataset()
    eit_data = dataset.to_eit_data(copy_policy="copy")

    assert not np.shares_memory(eit_data.meas, dataset.measurements)
    assert not np.shares_memory(eit_data.stim_pattern, dataset.stim_matrix)

    original = float(dataset.measurements[0, 0])
    eit_data.meas[0] = original + 10.0
    assert dataset.measurements[0, 0] == original


def test_iter_frames_respects_copy_policy():
    dataset = _build_dataset()
    view_frames = list(dataset.iter_frames(copy_policy="view"))
    copy_frames = list(dataset.iter_frames(copy_policy="copy"))

    assert len(view_frames) == dataset.measurements.shape[0]
    assert np.shares_memory(view_frames[0].meas, dataset.measurements)
    assert not np.shares_memory(copy_frames[0].meas, dataset.measurements)


def test_copy_policy_validation():
    dataset = _build_dataset()
    with pytest.raises(ValueError, match="copy_policy"):
        dataset.to_eit_data(copy_policy="invalid")
