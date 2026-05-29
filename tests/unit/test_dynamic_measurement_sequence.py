"""Tests for dynamic measurement sequence contracts."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

import pyeidors.data.dynamic_sequence as dynamic_sequence_module
from pyeidors.data.dynamic_sequence import (
    DYNAMIC_MEASUREMENT_SEQUENCE_SCHEMA,
    DynamicMeasurementSequence,
    read_dynamic_measurement_sequence,
    write_dynamic_measurement_sequence,
)
from pyeidors.data.measurement_dataset import MeasurementDataset
from pyeidors.data.structures import PatternConfig
from pyeidors.electrodes.patterns import StimMeasPatternManager
from pyeidors.io.hdf5_artifacts import read_hdf5_artifact


def test_dynamic_measurement_sequence_carries_frame_metadata_and_contracts() -> None:
    frames = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.1, 2.2, 3.3],
            [0.9, 2.4, 3.5],
        ],
        dtype=float,
    )
    sequence = DynamicMeasurementSequence.from_arrays(
        frames,
        t=[0.0, 0.1, 0.25],
        sampling_rate_hz=20.0,
        frame_id=[100, 101, 103],
        reference_policy="fixed_first_frame",
        stim_meas_signature="stim-meas-sha256",
        bad_channel_mask=[False, True, False],
        measurement_weights=[4.0, 0.0, 1.0],
        frequency_hz=[1000.0, 1000.0, 1200.0],
        context_metadata={"domain": "plant", "protocol": "travelling-wave"},
        metadata={"operator": "unit-test"},
        data_type="difference",
    )

    assert sequence.shape == (3, 3)
    np.testing.assert_allclose(sequence.dt, [0.0, 0.1, 0.15])
    np.testing.assert_array_equal(
        sequence.bad_channel_mask,
        np.array(
            [
                [False, True, False],
                [False, True, False],
                [False, True, False],
            ],
            dtype=bool,
        ),
    )
    assert sequence.measurement_weight_kind == "diagonal"

    frame_meta = sequence.frame_metadata(2)
    assert frame_meta["frame_id"] == 103
    assert frame_meta["t"] == pytest.approx(0.25)
    assert frame_meta["dt"] == pytest.approx(0.15)
    assert frame_meta["sampling_rate_hz"] == pytest.approx(20.0)
    assert frame_meta["reference_policy"] == "fixed_first_frame"
    assert frame_meta["stim_meas_signature"] == "stim-meas-sha256"
    assert frame_meta["bad_channel_mask"] == (False, True, False)
    assert frame_meta["measurement_weight_kind"] == "diagonal"
    assert frame_meta["frequency_hz"] == pytest.approx(1200.0)
    assert frame_meta["context_metadata"]["domain"] == "plant"


def test_dynamic_bad_channel_mask_vector_broadcast_direct_fill() -> None:
    source = inspect.getsource(dynamic_sequence_module._bad_channel_mask_frames)
    assert "broadcast_to" not in source
    assert "np.copyto" in source

    mask = dynamic_sequence_module._bad_channel_mask_frames(
        [False, True, False],
        n_frames=2,
        n_measurements=3,
    )

    assert mask.flags.c_contiguous
    np.testing.assert_array_equal(
        mask,
        np.array([[False, True, False], [False, True, False]], dtype=bool),
    )


def test_v487_dynamic_sequence_numeric_guards_use_bounded_scans() -> None:
    frame_source = inspect.getsource(dynamic_sequence_module._frame_batch)
    timestamp_source = inspect.getsource(dynamic_sequence_module._timestamps)
    dt_source = inspect.getsource(dynamic_sequence_module._dt)
    weights_source = inspect.getsource(dynamic_sequence_module._measurement_weights)
    frequency_source = inspect.getsource(dynamic_sequence_module._frequency_hz)

    assert "all_finite_values(arr)" in frame_source
    assert "np.isfinite(arr).all()" not in frame_source
    assert "all_finite_values(values)" in timestamp_source
    assert "np.isfinite(values).all()" not in timestamp_source
    assert "all_finite_values(values)" in dt_source
    assert "np.isfinite(values).all()" not in dt_source
    assert "np.any(values < 0.0)" not in dt_source
    assert "all_finite_values(arr)" in weights_source
    assert "np.isfinite(arr).all()" not in weights_source
    assert "np.any(arr < 0.0)" not in weights_source
    assert "all_finite_values(values)" in frequency_source
    assert "np.isfinite(values).all()" not in frequency_source
    assert "np.any(values < 0.0)" not in frequency_source


def test_dynamic_measurement_sequence_hdf5_roundtrip(tmp_path: Path) -> None:
    weights = np.array(
        [
            [[2.0, 0.1], [0.1, 3.0]],
            [[4.0, 0.0], [0.0, 5.0]],
        ],
        dtype=float,
    )
    sequence = DynamicMeasurementSequence.from_arrays(
        [[1.0, 2.0], [3.0, 4.0]],
        t=[1.0, 1.2],
        dt=[0.0, 0.2],
        sampling_rate_hz=5.0,
        frame_id=[7, 8],
        reference_policy="rolling_previous_frame",
        stim_meas_signature="sig-123",
        bad_channel_mask=[[False, False], [True, False]],
        measurement_weights=weights,
        frequency_hz=60.0,
        context_metadata={"species": "arabidopsis"},
        metadata={"note": "roundtrip"},
    )

    path = write_dynamic_measurement_sequence(tmp_path / "sequence", sequence)
    assert path == tmp_path / "sequence.h5"
    assert not (tmp_path / "sequence.npz").exists()

    artifact = read_hdf5_artifact(path)
    assert artifact.schema == DYNAMIC_MEASUREMENT_SEQUENCE_SCHEMA
    assert artifact.metadata["artifact_format"] == "hdf5"
    assert artifact.metadata["package_role"] == "dynamic_measurement_sequence"
    assert artifact.metadata["measurement_weight_kind"] == "full_per_frame"

    loaded = read_dynamic_measurement_sequence(path)
    np.testing.assert_allclose(loaded.frames, sequence.frames)
    np.testing.assert_allclose(loaded.t, sequence.t)
    np.testing.assert_allclose(loaded.dt, sequence.dt)
    np.testing.assert_array_equal(loaded.frame_id, sequence.frame_id)
    np.testing.assert_array_equal(loaded.bad_channel_mask, sequence.bad_channel_mask)
    np.testing.assert_allclose(loaded.measurement_weights, sequence.measurement_weights)
    assert loaded.measurement_weight_kind == "full_per_frame"
    assert loaded.reference_policy == "rolling_previous_frame"
    assert loaded.context_metadata["species"] == "arabidopsis"
    assert loaded.metadata["note"] == "roundtrip"


def test_dynamic_measurement_sequence_validates_shapes_and_weights() -> None:
    with pytest.raises(FloatingPointError, match="frames"):
        DynamicMeasurementSequence.from_arrays([[1.0, np.nan]])
    with pytest.raises(ValueError, match="t length"):
        DynamicMeasurementSequence.from_arrays([[1.0, 2.0]], t=[0.0, 1.0])
    with pytest.raises(ValueError, match="dt must be non-negative"):
        DynamicMeasurementSequence.from_arrays([[1.0, 2.0]], dt=-0.1)
    with pytest.raises(ValueError, match="bad_channel_mask"):
        DynamicMeasurementSequence.from_arrays(
            [[1.0, 2.0]],
            bad_channel_mask=[[True, False], [False, False]],
        )
    with pytest.raises(ValueError, match="symmetric"):
        DynamicMeasurementSequence.from_arrays(
            [[1.0, 2.0]],
            measurement_weights=np.array([[1.0, 2.0], [0.0, 1.0]]),
        )


def test_measurement_dataset_remains_single_frame_compatible_with_sequence_metadata():
    pattern_cfg = PatternConfig(n_elec=4, stim_pattern="{ad}", meas_pattern="{ad}")
    manager = StimMeasPatternManager(pattern_cfg)
    frame = np.arange(manager.n_meas_total, dtype=float)
    sequence = DynamicMeasurementSequence.from_arrays(
        frame,
        sampling_rate_hz=100.0,
        frame_id=[42],
        reference_policy="fixed_baseline",
        stim_meas_signature="pattern-sig",
        bad_channel_mask=np.zeros(manager.n_meas_total, dtype=bool),
        measurement_weights=np.ones(manager.n_meas_total, dtype=float),
        frequency_hz=5000.0,
        metadata={"n_frames": 1},
    )
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
        "n_frames": sequence.n_frames,
        "sampling_rate_hz": sequence.sampling_rate_hz,
        "reference_policy": sequence.reference_policy,
        "stim_meas_signature": sequence.stim_meas_signature,
    }

    dataset = MeasurementDataset.from_metadata(sequence.frames.reshape(-1), metadata)
    eit_data = dataset.to_eit_data(frame_index=0)

    assert dataset.measurements.shape == (1, manager.n_meas_total)
    assert dataset.summary()["n_frames"] == 1
    np.testing.assert_allclose(eit_data.meas, frame)
    assert eit_data.n_meas == manager.n_meas_total
