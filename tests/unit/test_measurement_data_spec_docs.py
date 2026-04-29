from __future__ import annotations

from pathlib import Path


DOC = Path(__file__).resolve().parents[2] / "docs" / "MEASUREMENT_DATA_SPEC.md"


def test_measurement_data_spec_recommends_hdf5_not_npz_default() -> None:
    text = DOC.read_text(encoding="utf-8")

    assert "**Recommended**: A single HDF5 package" in text
    assert "Legacy compatibility only" in text
    assert (
        "new production measurement, cache, reconstruction, GUI export, and benchmark artifacts should use HDF5"
        in text
    )
    assert "**Recommended**: A single `.npz`" not in text
    assert "For `.npz` format, use this naming scheme" not in text
