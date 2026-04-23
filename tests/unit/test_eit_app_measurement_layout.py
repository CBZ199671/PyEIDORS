from __future__ import annotations

import math

import pytest

from eit_app.measurement_layout import (
    estimate_measurement_point_count,
    measurement_layout_from_config,
)


def test_measurement_layout_defaults_to_current_16_electrode_hardware() -> None:
    layout = measurement_layout_from_config()

    assert layout["n_elec"] == 16
    assert layout["n_rings"] == 1
    assert layout["stim_pattern"] == "{ad}"
    assert layout["meas_pattern"] == "{ad}"
    assert layout["points_per_frame"] == 208
    assert layout["total_electrodes"] == 16


def test_measurement_layout_can_reserve_for_32_electrode_adjacent_mode() -> None:
    layout = measurement_layout_from_config({"n_elec": 32})

    assert layout["n_elec"] == 32
    assert layout["points_per_frame"] == estimate_measurement_point_count(
        n_electrodes=32,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        use_meas_current=False,
        use_meas_current_next=0,
    )
    assert layout["points_per_frame"] == 928


def test_measurement_layout_counts_multiring_3d_adjacent_mode() -> None:
    layout = measurement_layout_from_config({"n_elec": 8, "n_rings": 2})

    assert layout["n_elec"] == 8
    assert layout["n_rings"] == 2
    assert layout["electrode_layout"] == "ring_major"
    assert layout["measurement_protocol"] == "eidors_full_3d"
    assert layout["total_electrodes"] == 16
    assert layout["points_per_frame"] == 208


def test_measurement_layout_uses_total_circumference_count_for_3d_zigzag() -> None:
    ring_major = measurement_layout_from_config(
        {
            "mesh_dimension": 3,
            "n_elec": 8,
            "n_rings": 2,
            "electrode_layout": "ring_major",
            "radius": 0.18,
            "electrode_coverage": 0.5,
        }
    )
    zigzag = measurement_layout_from_config(
        {
            "mesh_dimension": 3,
            "n_elec": 8,
            "n_rings": 2,
            "electrode_layout": "zigzag",
            "radius": 0.18,
            "electrode_coverage": 0.5,
        }
    )

    assert ring_major["electrode_length_m_override"] == pytest.approx(
        2.0 * math.pi * 0.18 * 0.5 / 8.0
    )
    assert zigzag["electrode_length_m_override"] == pytest.approx(
        2.0 * math.pi * 0.18 * 0.5 / 16.0
    )
    assert zigzag["points_per_frame"] == 208


def test_measurement_layout_counts_layer_local_25d_mode() -> None:
    layout = measurement_layout_from_config(
        {"n_elec": 8, "n_rings": 2, "measurement_protocol": "layer_local_2p5d"}
    )

    assert layout["total_electrodes"] == 16
    assert layout["points_per_frame"] == 80


def test_measurement_layout_counts_cross_layer_full_mode() -> None:
    layout = measurement_layout_from_config(
        {"n_elec": 8, "n_rings": 2, "measurement_protocol": "cross_layer_full"}
    )

    assert layout["total_electrodes"] == 16
    assert layout["points_per_frame"] == 152


def test_measurement_layout_counts_hybrid_full_3d_mode() -> None:
    layout = measurement_layout_from_config(
        {"n_elec": 8, "n_rings": 2, "measurement_protocol": "hybrid_full_3d"}
    )

    assert layout["total_electrodes"] == 16
    assert layout["points_per_frame"] == 456


def test_measurement_layout_counts_three_ring_hybrid_full_3d_mode() -> None:
    layout = measurement_layout_from_config(
        {"n_elec": 8, "n_rings": 3, "measurement_protocol": "hybrid_full_3d"}
    )

    assert layout["total_electrodes"] == 24
    assert layout["points_per_frame"] == 1368


def test_measurement_layout_accepts_explicit_points_override_for_future_protocols() -> (
    None
):
    layout = measurement_layout_from_config(
        {"n_elec": 32, "points_per_frame_override": 960}
    )

    assert layout["n_elec"] == 32
    assert layout["points_per_frame"] == 960


def test_explicit_electrode_length_overrides_stale_cached_coverage() -> None:
    layout = measurement_layout_from_config(
        {
            "n_elec": 16,
            "radius": 1.0,
            "geometry_scale_to_m": 1.0,
            "electrode_length_m_override": 0.020001,
            "electrode_coverage": 0.5,
        }
    )

    expected_coverage = 0.020001 / (2.0 * math.pi / 16.0)
    assert layout["electrode_length_m_override"] == pytest.approx(0.020001)
    assert layout["electrode_coverage"] == pytest.approx(expected_coverage)


def test_radius_change_recomputes_coverage_when_length_is_fixed() -> None:
    layout = measurement_layout_from_config(
        {
            "n_elec": 16,
            "radius": 1.5,
            "geometry_scale_to_m": 1.0,
            "electrode_length_m_override": 0.020001,
            "electrode_coverage": 0.5,
        }
    )

    expected_pitch = 2.0 * math.pi * 1.5 / 16.0
    assert layout["electrode_coverage"] == pytest.approx(0.020001 / expected_pitch)
