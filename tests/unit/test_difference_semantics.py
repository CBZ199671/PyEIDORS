"""Tests for difference-measurement projection semantics."""

from __future__ import annotations

import numpy as np

from pyeidors.data.difference import (
    build_difference_frames,
    build_difference_vector,
    normalize_time_difference,
    project_measurement_jacobian,
    project_measurement_vector,
)


def test_build_difference_vector_raw_normalized_and_orientation():
    reference = np.array([2.0, 4.0, 0.5], dtype=float)
    target = np.array([3.0, 10.0, 1.0], dtype=float)

    raw = build_difference_vector(target, reference, mode="raw")
    normalized = build_difference_vector(target, reference, mode="normalized")
    reversed_normalized = build_difference_vector(
        target,
        reference,
        mode="normalized",
        orientation="reference_minus_target",
    )

    assert np.allclose(raw, np.array([1.0, 6.0, 0.5], dtype=float))
    assert np.allclose(normalized, np.array([0.5, 1.5, 1.0], dtype=float))
    assert np.allclose(reversed_normalized, -normalized)


def test_normalize_time_difference_matches_existing_normalized_contract():
    reference = np.array([2.0, -4.0, 0.0], dtype=float)
    target = np.array([3.0, -2.0, 1.0], dtype=float)
    floor = 0.5

    expected = build_difference_vector(
        target,
        reference,
        mode="normalized",
        orientation="target_minus_reference",
        floor=floor,
    )

    np.testing.assert_allclose(
        normalize_time_difference(target, reference, floor=floor),
        expected,
    )
    np.testing.assert_allclose(
        normalize_time_difference(
            target,
            reference,
            floor=floor,
            orientation="reference_minus_target",
        ),
        -expected,
    )


def test_build_difference_frames_matches_rowwise_vector_contract():
    references = np.array(
        [[2.0, -4.0, 0.0], [1.0, 0.25, -0.5]],
        dtype=float,
    )
    targets = np.array(
        [[3.0, -2.0, 1.0], [1.5, 0.5, -1.5]],
        dtype=float,
    )
    floor = 0.5

    expected = np.vstack(
        [
            build_difference_vector(
                target,
                reference,
                mode="normalized",
                orientation="reference_minus_target",
                floor=floor,
            )
            for target, reference in zip(targets, references, strict=True)
        ]
    )

    actual = build_difference_frames(
        targets,
        references,
        mode="normalized",
        orientation="reference_minus_target",
        floor=floor,
    )

    assert actual.flags.c_contiguous
    np.testing.assert_allclose(actual, expected)


def test_project_measurement_vector_and_jacobian_in_difference_space():
    reference = np.array([2.0, 4.0], dtype=float)
    simulated = np.array([2.5, 5.0], dtype=float)
    jacobian = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=float)

    projected_vector = project_measurement_vector(
        simulated,
        measurement_type="difference",
        reference_meas=reference,
        difference_mode="normalized",
        difference_orientation="reference_minus_target",
    )
    projected_jacobian = project_measurement_jacobian(
        jacobian,
        measurement_type="difference",
        reference_meas=reference,
        difference_mode="normalized",
        difference_orientation="reference_minus_target",
    )

    assert np.allclose(projected_vector, np.array([-0.25, -0.25], dtype=float))
    assert np.allclose(
        projected_jacobian,
        np.array([[-1.0, -2.0], [-1.5, -2.0]], dtype=float),
    )
