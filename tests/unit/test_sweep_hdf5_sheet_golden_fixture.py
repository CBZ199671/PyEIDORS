"""T81 phase 2c: HDF5 sheet-name fixture gate for sweep tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import pytest

from pyeidors.data._sweep_core import (
    SWEEP_TABLES_HDF5_SCHEMA,
    dataclass_csv_row,
    write_hdf5_row_tables,
)
from pyeidors.data.bucket_dense_experiments import (
    BUCKET_DENSE_FIELD_FIELDS,
    BUCKET_DENSE_SUMMARY_FIELDS,
    BUCKET_FULL256_COMPARE_SUMMARY_FIELDS,
)
from pyeidors.data.factor_sweep import CSV_FIELDS
from pyeidors.data.holdout_fit_diff import (
    FIELD_FIELDS,
    STRUCTURE_FIELDS,
    SUMMARY_FIELDS,
)
from pyeidors.io.hdf5_artifacts import read_hdf5_artifact


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "sweep_hdf5_tables"
    / "phase2a_table_names.h5"
)
SWEEP_HDF5_TABLE_COLUMNS = {
    "voltage_digit_summary": [
        "target_voltage_digits",
        "achieved_voltage_effective_digits",
        "voltage_rmse",
        "sigma_rmse",
        "sigma_relative_rmse",
        "sigma_mae",
        "sigma_max_abs_error",
        "sigma_effective_digits",
    ],
    "voltage_digit_field": [
        "target_voltage_digits",
        "cell_index",
        "sigma_true",
        "sigma_recon",
        "sigma_error",
        "abs_sigma_error",
    ],
    "factor_sweep_row": CSV_FIELDS,
    "bucket_dense_summary": BUCKET_DENSE_SUMMARY_FIELDS,
    "bucket_dense_field": BUCKET_DENSE_FIELD_FIELDS,
    "bucket_full256_summary": BUCKET_FULL256_COMPARE_SUMMARY_FIELDS,
    "holdout_fit_summary": SUMMARY_FIELDS,
    "holdout_fit_field": FIELD_FIELDS,
    "holdout_structure": STRUCTURE_FIELDS,
}


@dataclass(frozen=True)
class _TinyRow:
    name: str
    maybe: float | None
    enabled: bool

    def as_csv_row(self) -> dict[str, object]:
        return dataclass_csv_row(self)


def _empty_tables():
    return {
        table_name: (fieldnames, [])
        for table_name, fieldnames in SWEEP_HDF5_TABLE_COLUMNS.items()
    }


def _hdf5_layout(path: Path) -> tuple[str, ...]:
    entries: list[str] = []
    with h5py.File(path, "r") as handle:

        def visit(name: str, obj: Any) -> None:
            kind = "group" if isinstance(obj, h5py.Group) else "dataset"
            entries.append(f"{kind}:{name}")

        handle.visititems(visit)
    return tuple(entries)


def _table_contract(path: Path) -> tuple[str, tuple[str, ...], dict[str, list[str]]]:
    artifact = read_hdf5_artifact(path, lazy=True, verify_checksums=False)
    metadata = artifact.metadata
    return (
        artifact.schema,
        tuple(metadata["table_names"]),
        dict(metadata["table_columns"]),
    )


def _decode_string_array(values) -> list[list[str]]:
    decoded: list[list[str]] = []
    for row in values:
        decoded.append(
            [
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
                for item in row
            ]
        )
    return decoded


def test_phase2a_sweep_hdf5_sheet_names_match_golden_fixture(tmp_path: Path) -> None:
    generated = write_hdf5_row_tables(
        tmp_path / "phase2a_table_names.h5",
        _empty_tables(),
    )

    expected_layout = ["group:arrays"]
    expected_layout.extend(
        f"dataset:arrays/{table_name}"
        for table_name in sorted(SWEEP_HDF5_TABLE_COLUMNS)
    )
    assert list(_hdf5_layout(FIXTURE_PATH)) == expected_layout
    assert _hdf5_layout(generated) == _hdf5_layout(FIXTURE_PATH)
    assert _table_contract(generated) == _table_contract(FIXTURE_PATH)


def test_write_hdf5_row_tables_preserves_csv_cell_contract(tmp_path: Path) -> None:
    path = write_hdf5_row_tables(
        tmp_path / "tiny_tables.h5",
        {"tiny": (["name", "maybe", "enabled"], [_TinyRow("left", None, True)])},
    )

    artifact = read_hdf5_artifact(path)

    assert artifact.schema == SWEEP_TABLES_HDF5_SCHEMA
    assert artifact.metadata["package_role"] == "sweep_report_tables"
    assert artifact.metadata["table_columns"] == {"tiny": ["name", "maybe", "enabled"]}
    assert _decode_string_array(artifact.arrays["tiny"]) == [["left", "", "true"]]


def test_write_hdf5_row_tables_rejects_ambiguous_table_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="slash-free"):
        write_hdf5_row_tables(tmp_path / "bad.h5", {"bad/name": (["value"], [])})
