from __future__ import annotations

import numpy as np

from pyeidors.data.frame_io import read_frame_csv, read_legacy_frame_csv


def test_read_frame_csv_supports_legacy_four_column_format(tmp_path) -> None:
    path = tmp_path / "legacy.csv"
    np.savetxt(
        path,
        np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            dtype=np.float64,
        ),
        delimiter=",",
        fmt="%.6f",
    )

    real, imag = read_frame_csv(path)
    real_v0, imag_v0, real_legacy, imag_legacy = read_legacy_frame_csv(path)

    assert np.allclose(real, [3.0, 7.0])
    assert np.allclose(imag, [4.0, 8.0])
    assert np.allclose(real_v0, [1.0, 5.0])
    assert np.allclose(imag_v0, [2.0, 6.0])
    assert np.allclose(real_legacy, real)
    assert np.allclose(imag_legacy, imag)
