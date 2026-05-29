from __future__ import annotations

import inspect

import numpy as np

import pyeidors.data.frame_io as frame_io_module


def test_v302_write_frame_csv_direct_fills_real_imag_columns(
    tmp_path,
    monkeypatch,
) -> None:
    def _fail_column_stack(*_args, **_kwargs):
        raise AssertionError("frame CSV writer must not call np.column_stack")

    monkeypatch.setattr(frame_io_module.np, "column_stack", _fail_column_stack)
    source = inspect.getsource(frame_io_module.write_frame_csv)
    assert "np.column_stack" not in source

    path = tmp_path / "frame.csv"
    frame_io_module.write_frame_csv(
        path,
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
    )
    real, imag = frame_io_module.read_frame_csv(path)
    np.testing.assert_allclose(real, [1.0, 2.0])
    np.testing.assert_allclose(imag, [3.0, 4.0])
