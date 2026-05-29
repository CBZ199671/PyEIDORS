"""Tests for difference-measurement projection semantics."""

from __future__ import annotations

import inspect

import numpy as np

from pyeidors.data import difference as difference_module
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


def test_build_difference_frames_uses_one_output_buffer_for_projection():
    source = inspect.getsource(difference_module.build_difference_frames)
    assert "target_batch - reference_batch" not in source
    assert "diff = diff /" not in source
    assert "diff = -diff" not in source
    assert "out=diff" in source

    references = np.array([[2.0, 4.0], [1.0, 8.0]], dtype=float)
    targets = np.array([[3.0, 6.0], [2.0, 12.0]], dtype=float)
    actual = build_difference_frames(
        targets,
        references,
        mode="normalized",
        orientation="reference_minus_target",
    )

    assert actual.flags.c_contiguous
    assert not np.shares_memory(actual, targets)
    assert not np.shares_memory(actual, references)
    np.testing.assert_allclose(
        actual,
        np.array([[-0.5, -0.5], [-1.0, -0.5]], dtype=float),
    )


def test_v584_build_difference_frames_broadcasts_single_reference_vector() -> None:
    source = inspect.getsource(difference_module.build_difference_frames)
    assert "reference_raw.reshape(1, -1)" in source
    assert "np.copyto" not in source

    reference = np.array([2.0, 4.0, 1.0], dtype=np.float32)
    targets = np.array(
        [[3.0, 8.0, 1.5], [4.0, 12.0, 2.0]],
        dtype=np.float32,
    )

    actual = build_difference_frames(
        targets,
        reference,
        mode="normalized",
        orientation="reference_minus_target",
    )

    assert actual.dtype == np.dtype(np.float32)
    assert actual.flags.c_contiguous
    np.testing.assert_allclose(
        actual,
        np.array([[-0.5, -1.0, -0.5], [-1.0, -2.0, -1.0]], dtype=np.float32),
    )


def test_v558_difference_projection_preserves_float32_payloads() -> None:
    reference = np.array([2.0, 4.0, 1.0], dtype=np.float32)
    target = np.array([3.0, 8.0, 1.5], dtype=np.float32)
    references = np.vstack([reference, reference]).astype(np.float32)
    targets = np.vstack([target, target + np.float32(0.5)]).astype(np.float32)
    jacobian = np.array([[2.0, 4.0], [6.0, 8.0], [1.0, 3.0]], dtype=np.float32)

    vector = normalize_time_difference(target, reference)
    frames = build_difference_frames(targets, references, mode="normalized")
    projected_vector = project_measurement_vector(
        target,
        measurement_type="difference",
        reference_meas=reference,
        difference_mode="normalized",
    )
    projected_jacobian = project_measurement_jacobian(
        jacobian,
        measurement_type="difference",
        reference_meas=reference,
        difference_mode="normalized",
    )

    assert vector.dtype == np.dtype(np.float32)
    assert frames.dtype == np.dtype(np.float32)
    assert projected_vector.dtype == np.dtype(np.float32)
    assert projected_jacobian.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(vector, np.array([0.5, 1.0, 0.5], dtype=np.float32))


def test_v562_reference_floor_scans_use_input_sized_abs_work() -> None:
    reference = np.array([0.0, 4.0, 1.0], dtype=np.float32)
    target = np.array([1.0, 8.0, 1.5], dtype=np.float32)

    vector = build_difference_vector(target, reference, mode="normalized", floor=0.25)

    assert vector.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(vector, [4.0, 1.0, 0.5])
    assert difference_module._abs_work_dtype(reference) == np.dtype(np.float32)
    for helper in (
        difference_module._reference_has_near_zero,
        difference_module._clamp_reference_floor_in_place,
    ):
        source = inspect.getsource(helper)
        assert "abs_work = np.empty(block_size, dtype=np.float64)" not in source
        assert "dtype=_abs_work_dtype(arr)" in source


def test_v581_vector_difference_reuses_diff_buffer_when_floor_not_needed(
    monkeypatch,
) -> None:
    reference = np.array([2.0, 4.0, 1.0], dtype=np.float32)
    target = np.array([3.0, 8.0, 1.5], dtype=np.float32)

    def _unexpected_safe_reference(*_args, **_kwargs):
        raise AssertionError("nonzero references should not allocate a safe copy")

    monkeypatch.setattr(
        difference_module, "_safe_reference", _unexpected_safe_reference
    )

    actual = build_difference_vector(
        target,
        reference,
        mode="normalized",
        orientation="reference_minus_target",
        floor=0.25,
    )

    assert actual.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(actual, np.array([-0.5, -1.0, -0.5], dtype=np.float32))
    source = inspect.getsource(difference_module.build_difference_vector)
    assert "np.divide(diff, safe, out=diff)" in source
    assert "np.negative(diff, out=diff)" in source
    assert "diff = diff /" not in source
    assert "diff = -diff" not in source


def test_normalized_reference_floor_preserves_sign_and_complex_phase():
    reference = np.array([0.0, -1e-30, 2.0], dtype=float)
    target = np.array([1.0, 1.0, 3.0], dtype=float)

    actual = build_difference_vector(
        target,
        reference,
        mode="normalized",
        floor=0.5,
    )

    np.testing.assert_allclose(actual, np.array([2.0, -2.0, 0.5], dtype=float))

    complex_reference = np.array(
        [0.0 + 0.0j, 1e-30 + 1e-30j, 2.0 + 0.0j],
        dtype=np.complex128,
    )
    complex_target = complex_reference + np.array(
        [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j],
        dtype=np.complex128,
    )
    phase = (1.0 + 1.0j) / np.sqrt(2.0)

    complex_actual = build_difference_vector(
        complex_target,
        complex_reference,
        mode="normalized",
        floor=0.5,
    )

    np.testing.assert_allclose(
        complex_actual,
        np.array([2.0 + 0.0j, 1.0 / (0.5 * phase), 0.5 + 0.0j]),
    )


def test_v503_difference_reference_floor_uses_bounded_work_buffers():
    safe_source = inspect.getsource(difference_module._safe_reference)
    clamp_source = inspect.getsource(difference_module._clamp_reference_floor_in_place)
    frames_source = inspect.getsource(difference_module.build_difference_frames)

    assert "safe[small]" not in safe_source
    assert "safe[small]" not in frames_source
    assert "tiny = safe[small]" not in frames_source
    assert "tiny[nonzero]" not in frames_source
    assert "signs = np.sign(safe[small])" not in frames_source
    assert "np.abs(chunk, out=abs_chunk)" in clamp_source
    assert "np.copyto(chunk, replacement, where=small_chunk)" in clamp_source
    assert "_reference_has_near_zero(reference_batch, eps)" in frames_source


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


def test_project_measurement_jacobian_uses_single_output_buffer_for_normalized_copy():
    source = inspect.getsource(difference_module.project_measurement_jacobian)
    assert "projected = projected /" not in source
    assert "[:, None]" not in source
    assert "-projected" not in source
    assert "out=projected" in source

    jacobian = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=float)
    original = jacobian.copy()
    reference = np.array([2.0, 4.0], dtype=float)

    projected = project_measurement_jacobian(
        jacobian,
        measurement_type="difference",
        reference_meas=reference,
        difference_mode="normalized",
        difference_orientation="reference_minus_target",
    )

    assert not np.shares_memory(projected, jacobian)
    np.testing.assert_allclose(jacobian, original)
    np.testing.assert_allclose(
        projected,
        np.array([[-1.0, -2.0], [-1.5, -2.0]], dtype=float),
    )


def test_v582_jacobian_projection_skips_reference_copy_when_floor_not_needed(
    monkeypatch,
) -> None:
    reference = np.array([2.0, 4.0], dtype=np.float32)
    jacobian = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)

    def _unexpected_safe_reference(*_args, **_kwargs):
        raise AssertionError("nonzero references should not allocate a safe copy")

    monkeypatch.setattr(
        difference_module, "_safe_reference", _unexpected_safe_reference
    )

    projected = project_measurement_jacobian(
        jacobian,
        measurement_type="difference",
        reference_meas=reference,
        difference_mode="normalized",
        floor=0.25,
    )

    assert projected.dtype == np.dtype(np.float32)
    assert not np.shares_memory(projected, jacobian)
    np.testing.assert_allclose(
        projected,
        np.array([[1.0, 2.0], [1.5, 2.0]], dtype=np.float32),
    )
    source = inspect.getsource(difference_module.project_measurement_jacobian)
    assert "_reference_has_near_zero(reference, eps)" in source
