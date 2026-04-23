"""Plot helpers for EIT digit report tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager
import matplotlib.pyplot as plt

from .digit_report import REPORT_FIELDS, DigitReportRow


TIMES_NEW_ROMAN_FONT_FILES = [
    "/mnt/c/Windows/Fonts/times.ttf",
    "/mnt/c/Windows/Fonts/timesbd.ttf",
    "/mnt/c/Windows/Fonts/timesi.ttf",
    "/mnt/c/Windows/Fonts/timesbi.ttf",
]


def configure_times_new_roman() -> None:
    """Globally prefer Times New Roman for English text and Arabic numerals."""

    for font_file in TIMES_NEW_ROMAN_FONT_FILES:
        path = Path(font_file)
        if not path.exists():
            continue
        try:
            font_manager.fontManager.addfont(str(path))
        except Exception:
            pass
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "axes.unicode_minus": False,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _parse_float(value: str, *, optional: bool = False) -> float | None:
    text = str(value).strip()
    if optional and text == "":
        return None
    return float(text)


def read_digit_report_rows(path: Path) -> list[DigitReportRow]:
    """Read combined T9 report CSV rows for plotting."""

    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = [field for field in REPORT_FIELDS if field not in fields]
        if missing:
            raise ValueError(
                f"{source} is missing required columns: {', '.join(missing)}"
            )
        return [
            DigitReportRow(
                scenario=str(row["scenario"]),
                bit=int(row["bit"]),
                ideal_decimal_digits=float(row["ideal_decimal_digits"]),
                full_scale=float(row["full_scale"]),
                enob=_parse_float(row["enob"], optional=True),
                noise_std=float(row["noise_std"]),
                noise_relative=float(row["noise_relative"]),
                voltage_rmse=float(row["voltage_rmse"]),
                voltage_effective_digits=float(row["voltage_effective_digits"]),
                sigma_rmse=float(row["sigma_rmse"]),
                sigma_effective_digits=float(row["sigma_effective_digits"]),
                hypothesis_delta_digits=float(row["hypothesis_delta_digits"]),
            )
            for row in reader
        ]


def _group_by_scenario(
    rows: Iterable[DigitReportRow],
) -> dict[str, list[DigitReportRow]]:
    grouped: dict[str, list[DigitReportRow]] = {}
    for row in rows:
        grouped.setdefault(row.scenario, []).append(row)
    return {
        scenario: sorted(items, key=lambda item: item.bit)
        for scenario, items in grouped.items()
    }


def plot_digit_report(
    rows: Iterable[DigitReportRow],
    output_path: Path,
    *,
    title: str = "EIT digit hypothesis check",
    dpi: int = 200,
) -> Path:
    """Render voltage/sigma effective digits and hypothesis delta to PNG."""

    row_list = list(rows)
    if not row_list:
        raise ValueError("rows must not be empty")

    configure_times_new_roman()
    grouped = _group_by_scenario(row_list)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#17becf"]
    fig, (digit_ax, delta_ax) = plt.subplots(
        2,
        1,
        figsize=(9.0, 7.0),
        sharex=True,
        constrained_layout=True,
    )
    fig.suptitle(title, fontsize=14)

    for idx, (scenario, items) in enumerate(grouped.items()):
        color = colors[idx % len(colors)]
        bits = [item.bit for item in items]
        voltage_digits = [item.voltage_effective_digits for item in items]
        sigma_digits = [item.sigma_effective_digits for item in items]
        deltas = [item.hypothesis_delta_digits for item in items]
        digit_ax.plot(
            bits,
            voltage_digits,
            marker="o",
            linestyle="--",
            linewidth=1.7,
            color=color,
            label=f"{scenario} voltage",
        )
        digit_ax.plot(
            bits,
            sigma_digits,
            marker="s",
            linestyle="-",
            linewidth=1.9,
            color=color,
            label=f"{scenario} sigma",
        )
        delta_ax.plot(
            bits,
            deltas,
            marker="D",
            linewidth=1.8,
            color=color,
            label=scenario,
        )

    digit_ax.set_ylabel("Effective decimal digits")
    digit_ax.set_title("Voltage and conductivity digits")
    digit_ax.grid(True, alpha=0.28)
    digit_ax.legend(loc="best", fontsize=8, ncols=2)

    delta_ax.axhline(
        1.0, color="#444444", linewidth=1.0, linestyle=":", label="+1 target"
    )
    delta_ax.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
    delta_ax.set_xlabel("ADC bit")
    delta_ax.set_ylabel("Sigma digits - voltage digits")
    delta_ax.set_title("Hypothesis delta")
    delta_ax.grid(True, alpha=0.28)
    delta_ax.legend(loc="best", fontsize=8)

    output = output.with_suffix(".png")
    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output


def plot_digit_report_csv(
    *,
    input_csv: Path,
    output_path: Path,
    title: str = "EIT digit hypothesis check",
    dpi: int = 200,
) -> Path:
    """Read a T9 report CSV and render its T10 PNG plot."""

    return plot_digit_report(
        read_digit_report_rows(input_csv),
        output_path,
        title=title,
        dpi=dpi,
    )
