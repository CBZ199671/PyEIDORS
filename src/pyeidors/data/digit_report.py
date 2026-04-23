"""Markdown/CSV report tables for EIT digit experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import math
from typing import Iterable

from .adc_quantization import ideal_decimal_digits


EIT_DIGIT_FIELDS = [
    "bit",
    "ideal_decimal_digits",
    "voltage_rmse",
    "voltage_effective_digits",
    "sigma_rmse",
    "sigma_effective_digits",
    "hypothesis_delta_digits",
]

REPORT_FIELDS = [
    "scenario",
    "bit",
    "ideal_decimal_digits",
    "full_scale",
    "enob",
    "noise_std",
    "noise_relative",
    "voltage_rmse",
    "voltage_effective_digits",
    "sigma_rmse",
    "sigma_effective_digits",
    "hypothesis_delta_digits",
]


@dataclass(frozen=True)
class DigitReportCase:
    """One input CSV plus run settings that must be visible in the report."""

    label: str
    path: Path
    full_scale: float
    enob: float | None = None
    noise_std: float = 0.0
    noise_relative: float = 0.0


@dataclass(frozen=True)
class DigitReportRow:
    """One rendered row in the EIT digit report table."""

    scenario: str
    bit: int
    ideal_decimal_digits: float
    full_scale: float
    enob: float | None
    noise_std: float
    noise_relative: float
    voltage_rmse: float
    voltage_effective_digits: float
    sigma_rmse: float
    sigma_effective_digits: float
    hypothesis_delta_digits: float

    def as_csv_row(self) -> dict[str, float | int | str]:
        return {
            "scenario": self.scenario,
            "bit": self.bit,
            "ideal_decimal_digits": self.ideal_decimal_digits,
            "full_scale": self.full_scale,
            "enob": "" if self.enob is None else self.enob,
            "noise_std": self.noise_std,
            "noise_relative": self.noise_relative,
            "voltage_rmse": self.voltage_rmse,
            "voltage_effective_digits": self.voltage_effective_digits,
            "sigma_rmse": self.sigma_rmse,
            "sigma_effective_digits": self.sigma_effective_digits,
            "hypothesis_delta_digits": self.hypothesis_delta_digits,
        }


def _parse_float(value: str, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return parsed


def _validate_required_fields(fieldnames: Iterable[str] | None, *, path: Path) -> None:
    names = set(fieldnames or [])
    missing = [field for field in EIT_DIGIT_FIELDS if field not in names]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def _validate_digit_invariants(
    *,
    bit: int,
    ideal_digits: float,
    voltage_digits: float,
    sigma_digits: float,
    hypothesis_delta_digits: float,
    path: Path,
    row_number: int,
) -> None:
    expected_ideal = ideal_decimal_digits(bit)
    if not math.isclose(ideal_digits, expected_ideal, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            f"{path} row {row_number}: ideal_decimal_digits does not match "
            "bit * log10(2)"
        )

    if math.isfinite(voltage_digits) and math.isfinite(sigma_digits):
        expected_delta = sigma_digits - voltage_digits
        if not math.isclose(
            hypothesis_delta_digits,
            expected_delta,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{path} row {row_number}: hypothesis_delta_digits does not "
                "equal sigma_effective_digits - voltage_effective_digits"
            )


def read_eit_digit_case(case: DigitReportCase) -> list[DigitReportRow]:
    """Read one EIT digit CSV and attach visible run settings."""

    path = Path(case.path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _validate_required_fields(reader.fieldnames, path=path)
        rows: list[DigitReportRow] = []
        for row_number, payload in enumerate(reader, start=2):
            bit = int(payload["bit"])
            ideal_digits = _parse_float(
                payload["ideal_decimal_digits"],
                name="ideal_decimal_digits",
            )
            voltage_digits = _parse_float(
                payload["voltage_effective_digits"],
                name="voltage_effective_digits",
            )
            sigma_digits = _parse_float(
                payload["sigma_effective_digits"],
                name="sigma_effective_digits",
            )
            hypothesis_delta = _parse_float(
                payload["hypothesis_delta_digits"],
                name="hypothesis_delta_digits",
            )
            _validate_digit_invariants(
                bit=bit,
                ideal_digits=ideal_digits,
                voltage_digits=voltage_digits,
                sigma_digits=sigma_digits,
                hypothesis_delta_digits=hypothesis_delta,
                path=path,
                row_number=row_number,
            )
            rows.append(
                DigitReportRow(
                    scenario=str(case.label),
                    bit=bit,
                    ideal_decimal_digits=ideal_digits,
                    full_scale=float(case.full_scale),
                    enob=case.enob,
                    noise_std=float(case.noise_std),
                    noise_relative=float(case.noise_relative),
                    voltage_rmse=_parse_float(
                        payload["voltage_rmse"],
                        name="voltage_rmse",
                    ),
                    voltage_effective_digits=voltage_digits,
                    sigma_rmse=_parse_float(payload["sigma_rmse"], name="sigma_rmse"),
                    sigma_effective_digits=sigma_digits,
                    hypothesis_delta_digits=hypothesis_delta,
                )
            )
    return rows


def read_eit_digit_cases(cases: Iterable[DigitReportCase]) -> list[DigitReportRow]:
    rows: list[DigitReportRow] = []
    for case in cases:
        rows.extend(read_eit_digit_case(case))
    return rows


def _format_float(value: float | None) -> str:
    if value is None:
        return "nominal"
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.12g}"


def _escape_markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def format_markdown_table(rows: Iterable[DigitReportRow]) -> str:
    rendered = []
    for row in rows:
        rendered.append(
            [
                row.scenario,
                str(row.bit),
                _format_float(row.ideal_decimal_digits),
                _format_float(row.full_scale),
                _format_float(row.enob),
                _format_float(row.noise_std),
                _format_float(row.noise_relative),
                _format_float(row.voltage_rmse),
                _format_float(row.voltage_effective_digits),
                _format_float(row.sigma_rmse),
                _format_float(row.sigma_effective_digits),
                _format_float(row.hypothesis_delta_digits),
            ]
        )
    headers = REPORT_FIELDS
    if not rendered:
        widths = [len(header) for header in headers]
    else:
        widths = [
            max(len(headers[idx]), *(len(item[idx]) for item in rendered))
            for idx in range(len(headers))
        ]
    lines = [
        "| "
        + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers)))
        + " |",
        "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |",
    ]
    lines.extend(
        "| "
        + " | ".join(
            _escape_markdown_cell(value).rjust(widths[idx])
            for idx, value in enumerate(item)
        )
        + " |"
        for item in rendered
    )
    return "\n".join(lines)


def format_markdown_report(
    rows: Iterable[DigitReportRow],
    *,
    title: str = "EIT digit report table",
) -> str:
    row_list = list(rows)
    return (
        "\n\n".join(
            [
                f"# {title}",
                (
                    "Settings columns are included so ADC bit depth is not treated as "
                    "a direct conductivity digit conclusion. Use "
                    "`hypothesis_delta_digits` to evaluate the +1 digit hypothesis."
                ),
                format_markdown_table(row_list),
            ]
        )
        + "\n"
    )


def write_report_files(
    *,
    rows: Iterable[DigitReportRow],
    markdown_path: Path,
    csv_path: Path | None = None,
    title: str = "EIT digit report table",
) -> None:
    row_list = list(rows)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        format_markdown_report(row_list, title=title),
        encoding="utf-8",
    )
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
            writer.writeheader()
            for row in row_list:
                writer.writerow(row.as_csv_row())
