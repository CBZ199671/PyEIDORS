"""Tests for shared-memory measurement ring buffer."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from eit_app.acquisition.ring_buffer import FrameRingBuffer


def test_frame_ring_buffer_roundtrip_avoids_bytes_payload_copy() -> None:
    ring = FrameRingBuffer(capacity=2, n_meas=3, create=True)
    try:
        ring.write(
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([0.5, -1.0, 2.5], dtype=np.float64),
            timestamp=12.25,
            frame_index=7,
        )

        result = ring.read_latest()
        assert result is not None
        real, imag, timestamp, frame_index = result
        np.testing.assert_allclose(real, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(imag, [0.5, -1.0, 2.5])
        assert timestamp == pytest.approx(12.25)
        assert frame_index == 7
    finally:
        ring.close()
        ring.unlink()

    source = inspect.getsource(FrameRingBuffer.write) + inspect.getsource(
        FrameRingBuffer._read_slot
    )
    assert ".tobytes(" not in source
    assert "bytes(" not in source


def test_v560_frame_ring_buffer_write_does_not_widen_before_shared_memory_copy() -> (
    None
):
    ring = FrameRingBuffer(capacity=1, n_meas=3, create=True)
    try:
        real = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        component = ring._frame_component(real, name="real")
        assert np.shares_memory(component, real)
        assert component.dtype == np.float32

        ring.write(
            real,
            np.array([0.0, 0.25, 0.5], dtype=np.float32),
            timestamp=1.0,
            frame_index=1,
        )
        result = ring.read_latest()
        assert result is not None
        read_real, read_imag, _, _ = result
        assert read_real.dtype == np.float64
        assert read_imag.dtype == np.float64
        np.testing.assert_allclose(read_real, real)
        np.testing.assert_allclose(read_imag, [0.0, 0.25, 0.5])
    finally:
        ring.close()
        ring.unlink()

    source = inspect.getsource(FrameRingBuffer._frame_component)
    assert "dtype=np.float64" not in source


def test_frame_ring_buffer_rejects_wrong_component_length() -> None:
    ring = FrameRingBuffer(capacity=1, n_meas=3, create=True)
    try:
        with pytest.raises(ValueError, match="real frame length"):
            ring.write(np.ones(2), np.ones(3), timestamp=0.0, frame_index=1)
    finally:
        ring.close()
        ring.unlink()
