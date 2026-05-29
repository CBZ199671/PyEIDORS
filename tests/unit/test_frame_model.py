"""FrameData measurement-vector memory-budget regressions."""

from __future__ import annotations

import inspect

import numpy as np

from eit_app.models.frame_model import FrameData


def test_v563_frame_data_complex_and_magnitude_use_single_output_buffers() -> None:
    real = np.array([3.0, 5.0], dtype=np.float32)
    imag = np.array([4.0, 12.0], dtype=np.float32)
    frame = FrameData(real=real, imag=imag, timestamp=0.0, frame_index=1)

    mag = frame.to_measurement_vector("mag")
    complex_vector = frame.to_measurement_vector("complex")
    amplitude = frame.amplitude()

    assert mag.dtype == np.dtype(np.float32)
    assert complex_vector.dtype == np.dtype(np.complex64)
    assert amplitude.dtype == np.dtype(np.float32)
    assert not np.shares_memory(mag, real)
    assert not np.shares_memory(complex_vector, real)
    assert not np.shares_memory(complex_vector, imag)
    assert not np.shares_memory(amplitude, real)
    np.testing.assert_allclose(mag, [5.0, 13.0])
    np.testing.assert_allclose(complex_vector, real + 1j * imag)
    np.testing.assert_allclose(amplitude, np.array([11.0, 28.6], dtype=np.float32))

    source = inspect.getsource(FrameData.to_measurement_vector)
    amplitude_source = inspect.getsource(FrameData.amplitude)
    assert "self.real + 1j * self.imag" not in source
    assert "self.real.copy() + 1j * self.imag.copy()" not in source
    assert "np.hypot" in source
    assert "np.hypot" in amplitude_source
