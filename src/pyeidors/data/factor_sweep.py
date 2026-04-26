"""Controlled multi-factor sweeps for EIT digit reconstruction studies."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .adc_quantization import (
    add_voltage_noise,
    effective_digits_from_rmse,
    quantize_voltages,
    rmse,
)
from .eit_digit_metrics import (
    EITLinearizedModel,
    adjacent_measurement_count,
    build_pyeidors_fem_linearized_model,
    build_surrogate_linearized_model,
    reconstruct_linearized_sigma,
    sigma_true_from_anomaly_rule,
)
from .voltage_digit_sweep import keep_significant_digits


@dataclass(frozen=True)
class FactorSweepRow:
    """One row of the T15 multi-factor controlled-variable experiment."""

    sweep: str
    changed_factor: str
    level: str
    n_elec: int
    fem_grid: int
    ridge: float
    target_voltage_digits: int
    enob: str
    noise_relative: float
    noser_exponent: float
    n_measurements: int
    voltage_rmse: float
    achieved_voltage_effective_digits: float
    sigma_rmse: float
    sigma_relative_rmse: float
    sigma_mae: float
    sigma_max_abs_error: float
    sigma_effective_digits: float

    def as_csv_row(self) -> dict[str, float | int | str]:
        return {
            "sweep": self.sweep,
            "changed_factor": self.changed_factor,
            "level": self.level,
            "n_elec": self.n_elec,
            "fem_grid": self.fem_grid,
            "ridge": self.ridge,
            "target_voltage_digits": self.target_voltage_digits,
            "enob": self.enob,
            "noise_relative": self.noise_relative,
            "noser_exponent": self.noser_exponent,
            "n_measurements": self.n_measurements,
            "voltage_rmse": self.voltage_rmse,
            "achieved_voltage_effective_digits": self.achieved_voltage_effective_digits,
            "sigma_rmse": self.sigma_rmse,
            "sigma_relative_rmse": self.sigma_relative_rmse,
            "sigma_mae": self.sigma_mae,
            "sigma_max_abs_error": self.sigma_max_abs_error,
            "sigma_effective_digits": self.sigma_effective_digits,
        }


CSV_FIELDS = [
    "sweep",
    "changed_factor",
    "level",
    "n_elec",
    "fem_grid",
    "ridge",
    "target_voltage_digits",
    "enob",
    "noise_relative",
    "noser_exponent",
    "n_measurements",
    "voltage_rmse",
    "achieved_voltage_effective_digits",
    "sigma_rmse",
    "sigma_relative_rmse",
    "sigma_mae",
    "sigma_max_abs_error",
    "sigma_effective_digits",
]


def _as_float_vector(values: Iterable[float] | np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _positive_int_levels(values: Iterable[int], *, name: str) -> list[int]:
    levels = [int(value) for value in values]
    if not levels:
        raise ValueError(f"{name} must not be empty")
    if any(value <= 0 for value in levels):
        raise ValueError(f"{name} must all be positive")
    return levels


def _finite_float_levels(values: Iterable[float], *, name: str) -> list[float]:
    levels = [float(value) for value in values]
    if not levels:
        raise ValueError(f"{name} must not be empty")
    if any(not math.isfinite(value) for value in levels):
        raise ValueError(f"{name} must all be finite")
    return levels


def _non_negative_float_levels(values: Iterable[float], *, name: str) -> list[float]:
    levels = _finite_float_levels(values, name=name)
    if any(value < 0.0 for value in levels):
        raise ValueError(f"{name} must all be non-negative")
    return levels


def _positive_float_levels(values: Iterable[float], *, name: str) -> list[float]:
    levels = _finite_float_levels(values, name=name)
    if any(value <= 0.0 for value in levels):
        raise ValueError(f"{name} must all be positive")
    return levels


def normalize_enob_level(level: str | float | int | None) -> tuple[str, float | None]:
    """Normalize one ENOB level from CLI/report text."""

    if level is None:
        return "nominal", None
    text = str(level).strip().lower()
    if text in {"", "nominal", "none", "null"}:
        return "nominal", None
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("ENOB level must be nominal or a positive finite number")
    return f"{value:g}", value


def _relative_rmse(reference: np.ndarray, observed: np.ndarray) -> float:
    ref_rms = float(np.sqrt(np.mean(reference**2)))
    if ref_rms == 0.0:
        return math.nan
    return rmse(reference, observed) / ref_rms


def _format_level(value: int | float | str) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _build_model_for_grid(
    *,
    forward_backend: str,
    fem_grid: int,
    n_elec: int,
    expected_measurements: int | None,
    n_measurements: int,
    n_parameters: int,
    model_seed: int,
) -> EITLinearizedModel:
    backend = str(forward_backend).strip().lower()
    if backend in {"surrogate", "linear-surrogate"}:
        return build_surrogate_linearized_model(
            n_measurements=n_measurements,
            n_parameters=n_parameters,
            seed=model_seed + int(fem_grid),
        )
    if backend in {"pyeidors-fem", "fem"}:
        expected = (
            adjacent_measurement_count(n_elec)
            if expected_measurements is None
            else int(expected_measurements)
        )
        return build_pyeidors_fem_linearized_model(
            n_elec=n_elec,
            grid=fem_grid,
            expected_measurements=expected,
        )
    raise ValueError("forward_backend must be one of: surrogate, pyeidors-fem")


def _model_with_anomaly_rule(
    model: EITLinearizedModel,
    anomaly_rule: str,
) -> EITLinearizedModel:
    rule = str(anomaly_rule).strip().lower().replace("-", "_")
    sigma = sigma_true_from_anomaly_rule(
        model.sigma_reference.size,
        parameter_points=model.parameter_points,
        rule=rule,
    )
    if np.array_equal(sigma, model.sigma_true):
        return model
    if model.forward_solver is None:
        voltage_true = model.voltage_reference + model.sensitivity @ (
            sigma - model.sigma_reference
        )
    else:
        voltage_true = model.forward_solver(sigma)
    return replace(
        model,
        sigma_true=sigma,
        voltage_true=np.asarray(voltage_true, dtype=float),
        label=f"{model.label}:{rule}",
    )


def _measured_voltage(
    *,
    voltage_true: np.ndarray,
    target_voltage_digits: int,
    noise_relative: float,
    enob_value: float | None,
    full_scale_range: float,
    adc_bit: int,
    seed: int | None,
) -> np.ndarray:
    measured = keep_significant_digits(voltage_true, int(target_voltage_digits))
    if noise_relative > 0.0:
        measured = add_voltage_noise(
            measured,
            noise_relative=noise_relative,
            seed=seed,
        )
    if enob_value is not None:
        measured = quantize_voltages(
            measured,
            bit=int(adc_bit),
            full_scale_range=full_scale_range,
            enob=enob_value,
        )
    return measured


def _evaluate_case(
    *,
    model: EITLinearizedModel,
    sweep: str,
    changed_factor: str,
    level: str,
    n_elec: int,
    fem_grid: int,
    ridge: float,
    target_voltage_digits: int,
    enob_level: str | float | int | None,
    noise_relative: float,
    full_scale_range: float,
    adc_bit: int,
    seed: int | None,
    inverse_backend: str,
    rm_mode: str,
    rm_form: str,
    noser_exponent: float,
) -> FactorSweepRow:
    enob_label, enob_value = normalize_enob_level(enob_level)
    sigma_true = _as_float_vector(model.sigma_true, name="model.sigma_true")
    voltage_true = _as_float_vector(model.voltage_true, name="model.voltage_true")
    voltage_measured = _measured_voltage(
        voltage_true=voltage_true,
        target_voltage_digits=target_voltage_digits,
        noise_relative=float(noise_relative),
        enob_value=enob_value,
        full_scale_range=float(full_scale_range),
        adc_bit=int(adc_bit),
        seed=seed,
    )
    sigma_recon = reconstruct_linearized_sigma(
        model=model,
        voltages=voltage_measured,
        ridge=float(ridge),
        inverse_backend=inverse_backend,
        rm_mode=rm_mode,
        rm_form=rm_form,
        noser_exponent=noser_exponent,
    )
    error = sigma_recon - sigma_true
    abs_error = np.abs(error)
    return FactorSweepRow(
        sweep=str(sweep),
        changed_factor=str(changed_factor),
        level=str(level),
        n_elec=int(n_elec),
        fem_grid=int(fem_grid),
        ridge=float(ridge),
        target_voltage_digits=int(target_voltage_digits),
        enob=enob_label,
        noise_relative=float(noise_relative),
        noser_exponent=float(noser_exponent),
        n_measurements=int(model.n_measurements),
        voltage_rmse=rmse(voltage_true, voltage_measured),
        achieved_voltage_effective_digits=effective_digits_from_rmse(
            voltage_true,
            voltage_measured,
        ),
        sigma_rmse=rmse(sigma_true, sigma_recon),
        sigma_relative_rmse=_relative_rmse(sigma_true, sigma_recon),
        sigma_mae=float(np.mean(abs_error)),
        sigma_max_abs_error=float(np.max(abs_error)),
        sigma_effective_digits=effective_digits_from_rmse(sigma_true, sigma_recon),
    )


def run_factor_sweep(
    *,
    fem_grid_levels: Iterable[int],
    ridge_levels: Iterable[float],
    target_digits: Iterable[int],
    noise_relative_levels: Iterable[float],
    enob_levels: Iterable[str | float | int],
    full_scale_levels: Iterable[float] | None = None,
    rm_mode_levels: Iterable[str] | None = None,
    noser_exponent_levels: Iterable[float] | None = None,
    anomaly_rule_levels: Iterable[str] | None = None,
    forward_backend: str = "pyeidors-fem",
    n_elec: int = 16,
    expected_measurements: int | None = None,
    baseline_fem_grid: int = 4,
    baseline_ridge: float = 1e-2,
    baseline_target_digits: int = 6,
    baseline_noise_relative: float = 0.0,
    baseline_enob: str | float | int | None = "nominal",
    baseline_anomaly_rule: str = "default",
    full_scale_range: float = 10.0,
    adc_bit: int = 16,
    seed: int | None = 0,
    inverse_backend: str = "pyeidors-rm",
    rm_mode: str = "tikhonov",
    rm_form: str = "param",
    noser_exponent: float = 0.5,
    n_measurements: int = 16,
    n_parameters: int = 8,
    model_seed: int = 20260422,
) -> list[FactorSweepRow]:
    """Run T15/T17 single-factor and grid-ridge interaction sweeps."""

    grids = _positive_int_levels(fem_grid_levels, name="fem_grid_levels")
    ridges = _non_negative_float_levels(ridge_levels, name="ridge_levels")
    digits = _positive_int_levels(target_digits, name="target_digits")
    noise_levels = _non_negative_float_levels(
        noise_relative_levels,
        name="noise_relative_levels",
    )
    enobs = list(enob_levels)
    if not enobs:
        raise ValueError("enob_levels must not be empty")
    for level in enobs:
        normalize_enob_level(level)
    full_scales = (
        []
        if full_scale_levels is None
        else _non_negative_float_levels(full_scale_levels, name="full_scale_levels")
    )
    if any(value <= 0.0 for value in full_scales):
        raise ValueError("full_scale_levels must all be positive")
    rm_modes = (
        [] if rm_mode_levels is None else [str(value) for value in rm_mode_levels]
    )
    noser_exponents = (
        []
        if noser_exponent_levels is None
        else _positive_float_levels(
            noser_exponent_levels,
            name="noser_exponent_levels",
        )
    )
    anomaly_rules = (
        []
        if anomaly_rule_levels is None
        else [str(value) for value in anomaly_rule_levels]
    )

    full_scale = float(full_scale_range)
    if not math.isfinite(full_scale) or full_scale <= 0.0:
        raise ValueError("full_scale_range must be positive and finite")

    model_cache: dict[int, EITLinearizedModel] = {}
    anomaly_model_cache: dict[tuple[int, str], EITLinearizedModel] = {}

    def model_for(grid: int, anomaly_rule: str) -> EITLinearizedModel:
        grid_int = int(grid)
        rule = str(anomaly_rule).strip().lower().replace("-", "_")
        if grid_int not in model_cache:
            model_cache[grid_int] = _build_model_for_grid(
                forward_backend=forward_backend,
                fem_grid=grid_int,
                n_elec=n_elec,
                expected_measurements=expected_measurements,
                n_measurements=n_measurements,
                n_parameters=n_parameters,
                model_seed=model_seed,
            )
        key = (grid_int, rule)
        if key not in anomaly_model_cache:
            anomaly_model_cache[key] = _model_with_anomaly_rule(
                model_cache[grid_int],
                rule,
            )
        return anomaly_model_cache[key]

    rows: list[FactorSweepRow] = []

    def append_case(
        *,
        sweep: str,
        changed_factor: str,
        level: str,
        fem_grid: int,
        ridge: float,
        target_voltage_digits: int,
        enob_level: str | float | int | None,
        noise_relative: float,
        full_scale_range: float,
        rm_mode: str,
        noser_exponent: float,
        anomaly_rule: str,
    ) -> None:
        rows.append(
            _evaluate_case(
                model=model_for(int(fem_grid), anomaly_rule),
                sweep=sweep,
                changed_factor=changed_factor,
                level=level,
                n_elec=n_elec,
                fem_grid=int(fem_grid),
                ridge=float(ridge),
                target_voltage_digits=int(target_voltage_digits),
                enob_level=enob_level,
                noise_relative=float(noise_relative),
                full_scale_range=float(full_scale_range),
                adc_bit=int(adc_bit),
                seed=seed,
                inverse_backend=inverse_backend,
                rm_mode=rm_mode,
                rm_form=rm_form,
                noser_exponent=float(noser_exponent),
            )
        )

    append_case(
        sweep="baseline",
        changed_factor="baseline",
        level="baseline",
        fem_grid=baseline_fem_grid,
        ridge=baseline_ridge,
        target_voltage_digits=baseline_target_digits,
        enob_level=baseline_enob,
        noise_relative=baseline_noise_relative,
        full_scale_range=full_scale,
        rm_mode=rm_mode,
        noser_exponent=noser_exponent,
        anomaly_rule=baseline_anomaly_rule,
    )

    for grid in grids:
        append_case(
            sweep="single_factor",
            changed_factor="fem_grid",
            level=_format_level(grid),
            fem_grid=grid,
            ridge=baseline_ridge,
            target_voltage_digits=baseline_target_digits,
            enob_level=baseline_enob,
            noise_relative=baseline_noise_relative,
            full_scale_range=full_scale,
            rm_mode=rm_mode,
            noser_exponent=noser_exponent,
            anomaly_rule=baseline_anomaly_rule,
        )

    for ridge in ridges:
        append_case(
            sweep="single_factor",
            changed_factor="ridge",
            level=_format_level(ridge),
            fem_grid=baseline_fem_grid,
            ridge=ridge,
            target_voltage_digits=baseline_target_digits,
            enob_level=baseline_enob,
            noise_relative=baseline_noise_relative,
            full_scale_range=full_scale,
            rm_mode=rm_mode,
            noser_exponent=noser_exponent,
            anomaly_rule=baseline_anomaly_rule,
        )

    for digit_count in digits:
        append_case(
            sweep="single_factor",
            changed_factor="target_voltage_digits",
            level=_format_level(digit_count),
            fem_grid=baseline_fem_grid,
            ridge=baseline_ridge,
            target_voltage_digits=digit_count,
            enob_level=baseline_enob,
            noise_relative=baseline_noise_relative,
            full_scale_range=full_scale,
            rm_mode=rm_mode,
            noser_exponent=noser_exponent,
            anomaly_rule=baseline_anomaly_rule,
        )

    for noise in noise_levels:
        append_case(
            sweep="single_factor",
            changed_factor="noise_relative",
            level=_format_level(noise),
            fem_grid=baseline_fem_grid,
            ridge=baseline_ridge,
            target_voltage_digits=baseline_target_digits,
            enob_level=baseline_enob,
            noise_relative=noise,
            full_scale_range=full_scale,
            rm_mode=rm_mode,
            noser_exponent=noser_exponent,
            anomaly_rule=baseline_anomaly_rule,
        )

    for enob in enobs:
        enob_label, _ = normalize_enob_level(enob)
        append_case(
            sweep="single_factor",
            changed_factor="enob",
            level=enob_label,
            fem_grid=baseline_fem_grid,
            ridge=baseline_ridge,
            target_voltage_digits=baseline_target_digits,
            enob_level=enob,
            noise_relative=baseline_noise_relative,
            full_scale_range=full_scale,
            rm_mode=rm_mode,
            noser_exponent=noser_exponent,
            anomaly_rule=baseline_anomaly_rule,
        )

    for scale in full_scales:
        append_case(
            sweep="single_factor",
            changed_factor="full_scale",
            level=_format_level(scale),
            fem_grid=baseline_fem_grid,
            ridge=baseline_ridge,
            target_voltage_digits=baseline_target_digits,
            enob_level=baseline_enob,
            noise_relative=baseline_noise_relative,
            full_scale_range=scale,
            rm_mode=rm_mode,
            noser_exponent=noser_exponent,
            anomaly_rule=baseline_anomaly_rule,
        )

    for mode in rm_modes:
        append_case(
            sweep="single_factor",
            changed_factor="rm_mode",
            level=mode,
            fem_grid=baseline_fem_grid,
            ridge=baseline_ridge,
            target_voltage_digits=baseline_target_digits,
            enob_level=baseline_enob,
            noise_relative=baseline_noise_relative,
            full_scale_range=full_scale,
            rm_mode=mode,
            noser_exponent=noser_exponent,
            anomaly_rule=baseline_anomaly_rule,
        )

    for exponent in noser_exponents:
        append_case(
            sweep="single_factor",
            changed_factor="noser_exponent",
            level=_format_level(exponent),
            fem_grid=baseline_fem_grid,
            ridge=baseline_ridge,
            target_voltage_digits=baseline_target_digits,
            enob_level=baseline_enob,
            noise_relative=baseline_noise_relative,
            full_scale_range=full_scale,
            rm_mode="noser",
            noser_exponent=exponent,
            anomaly_rule=baseline_anomaly_rule,
        )

    for rule in anomaly_rules:
        append_case(
            sweep="single_factor",
            changed_factor="anomaly_rule",
            level=rule,
            fem_grid=baseline_fem_grid,
            ridge=baseline_ridge,
            target_voltage_digits=baseline_target_digits,
            enob_level=baseline_enob,
            noise_relative=baseline_noise_relative,
            full_scale_range=full_scale,
            rm_mode=rm_mode,
            noser_exponent=noser_exponent,
            anomaly_rule=rule,
        )

    for grid in grids:
        for ridge in ridges:
            append_case(
                sweep="grid_ridge_interaction",
                changed_factor="grid_x_ridge",
                level=f"grid={grid};ridge={ridge:g}",
                fem_grid=grid,
                ridge=ridge,
                target_voltage_digits=baseline_target_digits,
                enob_level=baseline_enob,
                noise_relative=baseline_noise_relative,
                full_scale_range=full_scale,
                rm_mode=rm_mode,
                noser_exponent=noser_exponent,
                anomaly_rule=baseline_anomaly_rule,
            )

    return rows


def _baseline_row(rows: Iterable[FactorSweepRow]) -> FactorSweepRow:
    for row in rows:
        if row.sweep == "baseline" and row.changed_factor == "baseline":
            return row
    raise ValueError("rows must include a baseline row")


def _delta_row(row: FactorSweepRow, baseline: FactorSweepRow) -> tuple[float, float]:
    return (
        row.sigma_relative_rmse - baseline.sigma_relative_rmse,
        row.sigma_effective_digits - baseline.sigma_effective_digits,
    )


def format_factor_sweep_report(
    rows: Iterable[FactorSweepRow],
    *,
    full_scale_range: float,
    adc_bit: int,
    title: str = "T15 多因素控制变量实验报告",
    rm_mode: str | None = None,
    noser_exponent: float | None = None,
    baseline_anomaly_rule: str | None = None,
) -> str:
    """Format the T15 CSV rows into a compact Markdown ranking report."""

    row_list = list(rows)
    if not row_list:
        raise ValueError("rows must not be empty")
    baseline = _baseline_row(row_list)
    single_rows = [row for row in row_list if row.sweep == "single_factor"]
    ranked = sorted(
        single_rows,
        key=lambda row: abs(_delta_row(row, baseline)[0]),
        reverse=True,
    )

    lines = [
        f"# {title}",
        "",
        "## 基准设置",
        "",
        "| 参数 | 值 |",
        "|---|---:|",
        f"| n_elec | {baseline.n_elec} |",
        f"| n_measurements | {baseline.n_measurements} |",
        f"| fem_grid | {baseline.fem_grid} |",
        f"| ridge | {baseline.ridge:.12g} |",
        f"| target_voltage_digits | {baseline.target_voltage_digits} |",
        f"| enob | {baseline.enob} |",
        f"| noise_relative | {baseline.noise_relative:.12g} |",
        f"| full_scale_range | {float(full_scale_range):.12g} |",
        f"| adc_bit | {int(adc_bit)} |",
    ]
    if rm_mode is not None:
        lines.append(f"| rm_mode | {rm_mode} |")
    if noser_exponent is not None:
        lines.append(f"| noser_exponent | {float(noser_exponent):.12g} |")
    if baseline_anomaly_rule is not None:
        lines.append(f"| anomaly_rule | {baseline_anomaly_rule} |")
    if any(row.changed_factor == "noser_exponent" for row in row_list):
        lines.extend(
            [
                "",
                "注：`noser_exponent` 行固定使用 `rm_mode=noser`，因为该指数对 "
                "`tikhonov` 无数学作用。",
            ]
        )
    lines.extend(
        [
            "",
            "## 主效应排序",
            "",
            "| changed_factor | level | sigma_relative_rmse | delta_sigma_relative_rmse | sigma_effective_digits | delta_sigma_effective_digits |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ranked:
        delta_rel, delta_digits = _delta_row(row, baseline)
        lines.append(
            f"| {row.changed_factor} | {row.level} | "
            f"{row.sigma_relative_rmse:.12g} | {delta_rel:.12g} | "
            f"{row.sigma_effective_digits:.12g} | {delta_digits:.12g} |"
        )

    interaction_rows = [
        row for row in row_list if row.sweep == "grid_ridge_interaction"
    ]
    grid_levels = sorted({row.fem_grid for row in interaction_rows})
    ridge_levels = sorted({row.ridge for row in interaction_rows})
    lookup = {
        (row.fem_grid, row.ridge): row.sigma_relative_rmse for row in interaction_rows
    }
    if interaction_rows:
        lines.extend(
            [
                "",
                "## grid × ridge 交互矩阵",
                "",
                "表内数值为 `sigma_relative_rmse`。",
                "",
                "| fem_grid \\ ridge | "
                + " | ".join(f"{ridge:.12g}" for ridge in ridge_levels)
                + " |",
                "|---:" + "".join("|---:" for _ in ridge_levels) + "|",
            ]
        )
        for grid in grid_levels:
            values = [lookup.get((grid, ridge), math.nan) for ridge in ridge_levels]
            lines.append(
                f"| {grid} | " + " | ".join(f"{value:.12g}" for value in values) + " |"
            )

    if ranked:
        top = ranked[0]
        delta_rel, delta_digits = _delta_row(top, baseline)
        lines.extend(
            [
                "",
                "## 结论提示",
                "",
                f"- 当前 sweep 中，相对基准影响最大的主效应是 `{top.changed_factor}={top.level}`。",
                f"- 其 `delta_sigma_relative_rmse = {delta_rel:.12g}`，`delta_sigma_effective_digits = {delta_digits:.12g}`。",
                "- 该排序只表示本次控制变量水平内的相对影响，不应外推为所有硬件和算法设置下的普遍规律。",
            ]
        )

    return "\n".join(lines) + "\n"


def plot_factor_sweep(
    rows: Iterable[FactorSweepRow],
    output_path: Path,
    *,
    title: str = "T15 factor sweep",
    dpi: int = 200,
) -> Path:
    """Render main-effect deltas and grid-ridge interaction to PNG."""

    row_list = list(rows)
    if not row_list:
        raise ValueError("rows must not be empty")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .digit_plot import configure_times_new_roman

    configure_times_new_roman()
    baseline = _baseline_row(row_list)
    single_rows = [row for row in row_list if row.sweep == "single_factor"]
    ranked = sorted(
        single_rows,
        key=lambda row: abs(_delta_row(row, baseline)[0]),
        reverse=True,
    )
    interaction_rows = [
        row for row in row_list if row.sweep == "grid_ridge_interaction"
    ]

    output = Path(output_path).with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, (delta_ax, interaction_ax) = plt.subplots(
        2,
        1,
        figsize=(10.5, 8.0),
        constrained_layout=True,
    )
    fig.suptitle(title, fontsize=14)

    labels = [f"{row.changed_factor}={row.level}" for row in ranked]
    deltas = [_delta_row(row, baseline)[0] for row in ranked]
    y_pos = np.arange(len(labels), dtype=float)
    colors = ["#d62728" if value >= 0.0 else "#2ca02c" for value in deltas]
    delta_ax.barh(y_pos, deltas, color=colors)
    delta_ax.axvline(0.0, color="#444444", linewidth=0.9)
    delta_ax.set_yticks(y_pos)
    delta_ax.set_yticklabels(labels, fontsize=8)
    delta_ax.invert_yaxis()
    delta_ax.set_xlabel("Delta sigma relative RMSE")
    delta_ax.set_title("Single-factor effects vs baseline")
    delta_ax.grid(True, axis="x", alpha=0.28)

    grid_levels = sorted({row.fem_grid for row in interaction_rows})
    ridge_levels = sorted({row.ridge for row in interaction_rows})
    for grid in grid_levels:
        grid_rows = sorted(
            [row for row in interaction_rows if row.fem_grid == grid],
            key=lambda row: row.ridge,
        )
        interaction_ax.plot(
            [row.ridge for row in grid_rows],
            [row.sigma_relative_rmse for row in grid_rows],
            marker="o",
            linewidth=1.7,
            label=f"grid={grid}",
        )
    if ridge_levels:
        interaction_ax.set_xscale("log")
    interaction_ax.set_xlabel("Ridge lambda")
    interaction_ax.set_ylabel("Sigma relative RMSE")
    interaction_ax.set_title("Grid x ridge interaction")
    interaction_ax.grid(True, alpha=0.28)
    interaction_ax.legend(loc="best", fontsize=8)

    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output
