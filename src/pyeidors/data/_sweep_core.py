"""Shared sweep/report serialization helpers.

These helpers are intentionally small: they consolidate repeated CSV row,
CSV writer, and CLI table formatting code without imposing a base class on
paper-specific sweep dataclasses.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping as MappingABC
import csv
from dataclasses import dataclass
from dataclasses import fields, is_dataclass
import json
import math
from pathlib import Path
from typing import ClassVar, Iterable, Mapping, Protocol, Sequence, TypeVar

CSVCell = str | int | float
CaseT = TypeVar("CaseT")
RowT = TypeVar("RowT")
RECON_METRIC_FIELDS = (
    "sigma_rmse",
    "sigma_relative_rmse",
    "sigma_mae",
    "sigma_max_abs_error",
    "sigma_effective_digits",
)
STRUCTURE_METRIC_FIELDS = (
    "centroid_error",
    "equivalent_area",
    "eccentricity",
    "major_axis",
    "minor_axis",
    "artifact_area",
    "artifact_energy",
    "artifact_peak",
    *RECON_METRIC_FIELDS,
)
STRUCTURE_SUMMARY_METRIC_FIELDS = (
    "centroid_error",
    "eccentricity",
    "artifact_area",
    "artifact_energy",
    "artifact_peak",
    *RECON_METRIC_FIELDS,
)
SWEEP_TABLES_HDF5_SCHEMA = "pyeidors-sweep-tables-hdf5-v1"
SWEEP_TABLES_JSON_SCHEMA = "pyeidors-sweep-tables-json-v1"


class SupportsCSVRow(Protocol):
    """Structural protocol for sweep rows that expose CSV-ready mappings."""

    def as_csv_row(self) -> Mapping[str, object]:
        """Return one string-keyed CSV row."""


SweepRowLike = SupportsCSVRow | Mapping[str, object]


@dataclass(frozen=True)
class SweepTable:
    """Materialized report table with stable name, columns, and row order."""

    name: str
    fieldnames: tuple[str, ...]
    rows: tuple[SweepRowLike, ...]


class SweepRow:
    """Mixin for dataclass-backed report rows with stable CSV serialization."""

    csv_fieldnames: ClassVar[Sequence[str] | None] = None

    def as_csv_row(self) -> dict[str, CSVCell]:
        """Return CSV-ready cells using the row's dataclass field order."""

        return dataclass_csv_row(self, fieldnames=self.csv_fieldnames)


class ReconMetricRow(SweepRow):
    """Mixin for rows that expose reconstruction-quality metric columns."""

    recon_metric_fields: ClassVar[Sequence[str]] = RECON_METRIC_FIELDS

    def recon_metric_values(self) -> dict[str, float]:
        """Return reconstruction metric values keyed by their CSV field names."""

        return _numeric_field_values(self, self.recon_metric_fields)


class StructureMetricRow(ReconMetricRow):
    """Mixin for rows that expose spatial structure/artifact metric columns."""

    structure_metric_fields: ClassVar[Sequence[str]] = STRUCTURE_METRIC_FIELDS

    def structure_metric_values(self) -> dict[str, float]:
        """Return structure metric values keyed by their CSV field names."""

        return _numeric_field_values(self, self.structure_metric_fields)


@dataclass(frozen=True)
class StructureMetrics(StructureMetricRow):
    """Reusable structure/artifact metrics shared by sweep implementations."""

    centroid_error: float
    equivalent_area: float
    eccentricity: float
    major_axis: float
    minor_axis: float
    artifact_area: float
    artifact_energy: float
    artifact_peak: float
    sigma_rmse: float
    sigma_relative_rmse: float
    sigma_mae: float
    sigma_max_abs_error: float
    sigma_effective_digits: float


def csv_cell(value: object) -> CSVCell:
    """Normalize one Python value for sweep CSV output."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except (TypeError, ValueError):
            scalar = value
        else:
            if scalar is not value:
                return csv_cell(scalar)
    if isinstance(value, (str, int, float)):
        return value
    return str(value)


def dataclass_csv_row(
    row: object,
    *,
    fieldnames: Sequence[str] | None = None,
) -> dict[str, CSVCell]:
    """Return a CSV row from a dataclass using stable field declaration order."""

    if not is_dataclass(row):
        raise TypeError("dataclass_csv_row requires a dataclass instance")
    names = (
        tuple(str(name) for name in fieldnames)
        if fieldnames is not None
        else tuple(field.name for field in fields(row))
    )
    return {name: csv_cell(getattr(row, name)) for name in names}


def _numeric_field_values(row: object, fieldnames: Sequence[str]) -> dict[str, float]:
    return {str(name): float(getattr(row, str(name))) for name in fieldnames}


def write_csv_rows(
    path: str | Path,
    rows: Iterable[SweepRowLike],
    fieldnames: Sequence[str],
) -> Path:
    """Write sweep rows with a stable header and per-row ``as_csv_row``."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    field_list = [str(field) for field in fieldnames]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_list)
        writer.writeheader()
        for row in rows:
            mapping = _row_mapping(row)
            writer.writerow(
                {field: csv_cell(mapping.get(field, "")) for field in field_list}
            )
    return output


def run_sweep(
    cases: Iterable[CaseT],
    compute_row: Callable[[CaseT], RowT],
    dump_target: Callable[[Sequence[RowT]], object] | None = None,
) -> list[RowT]:
    """Run a deterministic case-to-row sweep and optionally dump its rows."""

    rows = [compute_row(case) for case in cases]
    if dump_target is not None:
        dump_target(rows)
    return rows


def write_hdf5_row_tables(
    path: str | Path,
    tables: Mapping[str, tuple[Sequence[str], Iterable[SweepRowLike]]],
    metadata: Mapping[str, object] | None = None,
    *,
    schema: str = SWEEP_TABLES_HDF5_SCHEMA,
) -> Path:
    """Write CSV-compatible sweep row tables to a HDF5 artifact.

    Table names become dataset names under the shared ``arrays`` group. Column
    names live in metadata so tests can lock sheet/dataset names separately from
    row-schema refactors.
    """

    materialized = _materialize_tables(tables)
    return _write_hdf5_materialized_tables(
        path,
        materialized,
        metadata=metadata,
        schema=schema,
    )


def write_json_row_tables(
    path: str | Path,
    tables: Mapping[str, tuple[Sequence[str], Iterable[SweepRowLike]]],
    metadata: Mapping[str, object] | None = None,
    *,
    schema: str = SWEEP_TABLES_JSON_SCHEMA,
) -> Path:
    """Write CSV-compatible sweep row tables to a deterministic JSON artifact."""

    materialized = _materialize_tables(tables)
    return _write_json_materialized_tables(
        path,
        materialized,
        metadata=metadata,
        schema=schema,
    )


def write_sweep_table_artifacts(
    *,
    tables: Mapping[str, tuple[Sequence[str], Iterable[SweepRowLike]]],
    hdf5_output: str | Path | None = None,
    json_output: str | Path | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, Path]:
    """Write optional shared HDF5/JSON report-table artifacts."""

    materialized = _materialize_tables(tables)
    written: dict[str, Path] = {}
    if hdf5_output is not None:
        written["hdf5"] = _write_hdf5_materialized_tables(
            hdf5_output,
            materialized,
            metadata=metadata,
        )
    if json_output is not None:
        written["json"] = _write_json_materialized_tables(
            json_output,
            materialized,
            metadata=metadata,
        )
    return written


def _write_hdf5_materialized_tables(
    path: str | Path,
    tables: Sequence[SweepTable],
    *,
    metadata: Mapping[str, object] | None = None,
    schema: str = SWEEP_TABLES_HDF5_SCHEMA,
) -> Path:
    import numpy as np

    from pyeidors.io.hdf5_artifacts import write_hdf5_artifact

    arrays: dict[str, object] = {}
    columns: dict[str, list[str]] = {}
    for table in tables:
        columns[table.name] = list(table.fieldnames)
        rendered_rows = [
            [
                str(csv_cell(_row_mapping(row).get(field, "")))
                for field in table.fieldnames
            ]
            for row in table.rows
        ]
        arrays[table.name] = (
            np.asarray(rendered_rows, dtype=np.str_)
            if rendered_rows
            else np.empty((0, len(table.fieldnames)), dtype=np.str_)
        )
    meta: dict[str, object] = {
        "package_role": "sweep_report_tables",
        "table_columns": columns,
        "table_names": sorted(columns),
    }
    if metadata:
        meta.update(dict(metadata))
    return write_hdf5_artifact(
        path,
        arrays,
        meta,
        schema=schema,
        compression=None,
        chunks=None,
    )


def _write_json_materialized_tables(
    path: str | Path,
    tables: Sequence[SweepTable],
    *,
    metadata: Mapping[str, object] | None = None,
    schema: str = SWEEP_TABLES_JSON_SCHEMA,
) -> Path:
    columns = {table.name: list(table.fieldnames) for table in tables}
    payload = {
        "schema": schema,
        "metadata": _json_ready(
            {
                "package_role": "sweep_report_tables",
                "table_columns": columns,
                "table_names": sorted(columns),
                **dict(metadata or {}),
            }
        ),
        "tables": {
            table.name: {
                "columns": list(table.fieldnames),
                "rows": [
                    [
                        _json_cell(_row_mapping(row).get(field, ""))
                        for field in table.fieldnames
                    ]
                    for row in table.rows
                ],
            }
            for table in tables
        },
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def format_aligned_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    limit: int | None = None,
    more_template: str = "... {remaining} more rows",
) -> str:
    """Format a pipe-separated, terminal-friendly aligned table."""

    header_list = [str(header) for header in headers]
    rendered = [[str(value) for value in row] for row in rows]
    if limit is None:
        display_rows = rendered
    else:
        display_rows = rendered[: max(0, int(limit))]
    widths = [
        max([len(header_list[idx])] + [len(row[idx]) for row in display_rows])
        for idx in range(len(header_list))
    ]
    lines = [
        " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(header_list)),
        "-+-".join("-" * width for width in widths),
    ]
    lines.extend(
        " | ".join(value.rjust(widths[idx]) for idx, value in enumerate(row))
        for row in display_rows
    )
    if limit is not None and len(rendered) > len(display_rows):
        lines.append(
            more_template.format(
                remaining=len(rendered) - len(display_rows),
                total=len(rendered),
                shown=len(display_rows),
            )
        )
    return "\n".join(lines)


def _hdf5_table_name(name: object) -> str:
    text = str(name)
    if not text or "/" in text:
        raise ValueError(f"HDF5 table name must be non-empty and slash-free: {text!r}")
    return text


def _materialize_tables(
    tables: Mapping[str, tuple[Sequence[str], Iterable[SweepRowLike]]],
) -> tuple[SweepTable, ...]:
    return tuple(
        SweepTable(
            name=_hdf5_table_name(table_name),
            fieldnames=tuple(str(field) for field in fieldnames),
            rows=tuple(rows),
        )
        for table_name, (fieldnames, rows) in sorted(
            tables.items(), key=lambda item: str(item[0])
        )
    )


def _row_mapping(row: SweepRowLike) -> Mapping[str, object]:
    if isinstance(row, MappingABC):
        return row
    return row.as_csv_row()


def _json_cell(value: object) -> object:
    cell = csv_cell(value)
    if isinstance(cell, float) and not math.isfinite(cell):
        return str(cell)
    return cell


def _json_ready(value: object) -> object:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, MappingABC):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except (TypeError, ValueError):
            scalar = value
        else:
            if scalar is not value:
                return _json_ready(scalar)
    return str(value)


__all__ = [
    "CSVCell",
    "RECON_METRIC_FIELDS",
    "STRUCTURE_METRIC_FIELDS",
    "STRUCTURE_SUMMARY_METRIC_FIELDS",
    "SWEEP_TABLES_HDF5_SCHEMA",
    "SWEEP_TABLES_JSON_SCHEMA",
    "ReconMetricRow",
    "StructureMetricRow",
    "StructureMetrics",
    "SupportsCSVRow",
    "SweepTable",
    "SweepRowLike",
    "SweepRow",
    "csv_cell",
    "dataclass_csv_row",
    "format_aligned_table",
    "run_sweep",
    "write_csv_rows",
    "write_hdf5_row_tables",
    "write_json_row_tables",
    "write_sweep_table_artifacts",
]
