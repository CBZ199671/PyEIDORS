"""Shared sweep/report serialization helpers.

These helpers are intentionally small: they consolidate repeated CSV row,
CSV writer, and CLI table formatting code without imposing a base class on
paper-specific sweep dataclasses.
"""

from __future__ import annotations

import csv
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence


CSVCell = str | int | float


class SupportsCSVRow(Protocol):
    """Structural protocol for sweep rows that expose CSV-ready mappings."""

    def as_csv_row(self) -> Mapping[str, object]:
        """Return one string-keyed CSV row."""


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


def write_csv_rows(
    path: str | Path,
    rows: Iterable[SupportsCSVRow],
    fieldnames: Sequence[str],
) -> Path:
    """Write sweep rows with a stable header and per-row ``as_csv_row``."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())
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


__all__ = [
    "CSVCell",
    "SupportsCSVRow",
    "csv_cell",
    "dataclass_csv_row",
    "format_aligned_table",
    "write_csv_rows",
]
