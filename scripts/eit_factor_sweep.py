#!/usr/bin/env python3
"""Run T15 controlled multi-factor EIT digit sweeps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

from pyeidors.data.factor_sweep import (
    CSV_FIELDS,
    FactorSweepRow,
    format_factor_sweep_report,
    plot_factor_sweep,
    run_factor_sweep,
)


def _format_float(value: float | None) -> str:
    if value is None:
        return "none"
    if np.isinf(value):
        return "inf"
    if np.isnan(value):
        return "nan"
    return f"{value:.12g}"


def _format_table(rows: list[FactorSweepRow], *, limit: int = 12) -> str:
    display_rows = rows[: int(limit)]
    fields = [
        "sweep",
        "changed_factor",
        "level",
        "fem_grid",
        "ridge",
        "target_voltage_digits",
        "enob",
        "noise_relative",
        "sigma_relative_rmse",
        "sigma_effective_digits",
    ]
    rendered = [
        [
            str(row.sweep),
            str(row.changed_factor),
            str(row.level),
            str(row.fem_grid),
            _format_float(row.ridge),
            str(row.target_voltage_digits),
            str(row.enob),
            _format_float(row.noise_relative),
            _format_float(row.sigma_relative_rmse),
            _format_float(row.sigma_effective_digits),
        ]
        for row in display_rows
    ]
    widths = [
        max(len(fields[idx]), *(len(row[idx]) for row in rendered))
        for idx in range(len(fields))
    ]
    lines = [
        " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(fields)),
        "-+-".join("-" * width for width in widths),
    ]
    lines.extend(
        " | ".join(value.rjust(widths[idx]) for idx, value in enumerate(row))
        for row in rendered
    )
    if len(rows) > len(display_rows):
        lines.append(f"... {len(rows) - len(display_rows)} more rows")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[FactorSweepRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one-factor-at-a-time sweeps plus a small fem_grid x ridge "
            "interaction sweep for 16e adjacent EIT digit experiments."
        ),
    )
    parser.add_argument(
        "--fem-grid-levels",
        nargs="+",
        type=int,
        default=[4, 6, 8],
        help="FEM grid levels for grid and grid-ridge sweeps.",
    )
    parser.add_argument(
        "--ridge-levels",
        nargs="+",
        type=float,
        default=[1e-4, 1e-3, 1e-2, 1e-1],
        help="Ridge lambda levels for ridge and grid-ridge sweeps.",
    )
    parser.add_argument(
        "--target-digits",
        nargs="+",
        type=int,
        default=[4, 5, 6, 7],
        help="Target voltage significant-digit levels.",
    )
    parser.add_argument(
        "--noise-relative-levels",
        nargs="+",
        type=float,
        default=[0.0, 0.001],
        help="RMS-relative voltage noise levels.",
    )
    parser.add_argument(
        "--enob-levels",
        nargs="+",
        default=["nominal", "11"],
        help="ENOB levels. Use 'nominal' for no ENOB clamp.",
    )
    parser.add_argument(
        "--forward-backend",
        choices=["surrogate", "pyeidors-fem"],
        default="pyeidors-fem",
        help="Forward model backend.",
    )
    parser.add_argument(
        "--fem-n-elec",
        type=int,
        default=16,
        help="PyEIDORS FEM electrode count.",
    )
    parser.add_argument(
        "--expected-fem-measurements",
        type=int,
        help="Expected PyEIDORS FEM measurement count.",
    )
    parser.add_argument(
        "--baseline-fem-grid",
        type=int,
        default=4,
        help="Baseline FEM grid.",
    )
    parser.add_argument(
        "--baseline-ridge",
        type=float,
        default=1e-2,
        help="Baseline ridge lambda.",
    )
    parser.add_argument(
        "--baseline-target-digits",
        type=int,
        default=6,
        help="Baseline target voltage significant digits.",
    )
    parser.add_argument(
        "--baseline-noise-relative",
        type=float,
        default=0.0,
        help="Baseline RMS-relative voltage noise.",
    )
    parser.add_argument(
        "--baseline-enob",
        default="nominal",
        help="Baseline ENOB level.",
    )
    parser.add_argument(
        "--full-scale",
        type=float,
        default=10.0,
        help="ADC full-scale range used for numeric ENOB levels.",
    )
    parser.add_argument(
        "--adc-bit",
        type=int,
        default=16,
        help="Nominal ADC bit depth used for numeric ENOB levels.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for voltage noise.",
    )
    parser.add_argument(
        "--inverse-backend",
        choices=["pyeidors-rm", "least-squares"],
        default="pyeidors-rm",
        help="Inverse backend for sigma reconstruction.",
    )
    parser.add_argument(
        "--rm-mode",
        choices=["tikhonov", "noser"],
        default="tikhonov",
        help="PyEIDORS RM regularization mode.",
    )
    parser.add_argument(
        "--rm-form",
        choices=["param", "measurement"],
        default="param",
        help="PyEIDORS RM construction form.",
    )
    parser.add_argument(
        "--n-measurements",
        type=int,
        default=16,
        help="Surrogate measurement count.",
    )
    parser.add_argument(
        "--n-parameters",
        type=int,
        default=8,
        help="Surrogate conductivity parameter count.",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=20260422,
        help="Surrogate model seed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/eit_factor_sweep_16e.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("outputs/eit_factor_sweep_16e.md"),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("outputs/eit_factor_sweep_16e.png"),
        help="PNG plot output path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG plot DPI.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = run_factor_sweep(
        fem_grid_levels=args.fem_grid_levels,
        ridge_levels=args.ridge_levels,
        target_digits=args.target_digits,
        noise_relative_levels=args.noise_relative_levels,
        enob_levels=args.enob_levels,
        forward_backend=args.forward_backend,
        n_elec=args.fem_n_elec,
        expected_measurements=args.expected_fem_measurements,
        baseline_fem_grid=args.baseline_fem_grid,
        baseline_ridge=args.baseline_ridge,
        baseline_target_digits=args.baseline_target_digits,
        baseline_noise_relative=args.baseline_noise_relative,
        baseline_enob=args.baseline_enob,
        full_scale_range=args.full_scale,
        adc_bit=args.adc_bit,
        seed=args.seed,
        inverse_backend=args.inverse_backend,
        rm_mode=args.rm_mode,
        rm_form=args.rm_form,
        n_measurements=args.n_measurements,
        n_parameters=args.n_parameters,
        model_seed=args.model_seed,
    )
    _write_csv(args.output, rows)
    report = format_factor_sweep_report(
        rows,
        full_scale_range=args.full_scale,
        adc_bit=args.adc_bit,
    )
    _write_text(args.report_output, report)
    plot_path = plot_factor_sweep(
        rows,
        args.plot_output,
        title="16e factor sweep"
        if args.forward_backend == "pyeidors-fem"
        else "Surrogate factor sweep",
        dpi=args.dpi,
    )

    print(
        "settings: "
        f"model={args.forward_backend}+{args.inverse_backend}, "
        f"fem_n_elec={args.fem_n_elec}, "
        f"baseline_grid={args.baseline_fem_grid}, "
        f"baseline_ridge={_format_float(args.baseline_ridge)}, "
        f"baseline_target_digits={args.baseline_target_digits}, "
        f"baseline_enob={args.baseline_enob}, "
        f"baseline_noise_relative={_format_float(args.baseline_noise_relative)}, "
        f"full_scale={_format_float(args.full_scale)}, "
        f"adc_bit={args.adc_bit}, "
        "stim_pattern={ad}, meas_pattern={ad}"
    )
    print(_format_table(rows))
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report_output}")
    print(f"Wrote {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
