"""Voltage significant-digit sweeps for EIT conductivity error studies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

from .adc_quantization import effective_digits_from_rmse, rmse
from .eit_digit_metrics import (
    EITLinearizedModel,
    adjacent_measurement_count,
    build_pyeidors_fem_linearized_model,
    build_surrogate_linearized_model,
    reconstruct_linearized_sigma,
)


DigitMethod = Literal["truncate", "round"]


@dataclass(frozen=True)
class VoltageDigitSweepSummary:
    """Summary metrics for one target voltage significant-digit setting."""

    target_voltage_digits: int
    achieved_voltage_effective_digits: float
    voltage_rmse: float
    sigma_rmse: float
    sigma_relative_rmse: float
    sigma_mae: float
    sigma_max_abs_error: float
    sigma_effective_digits: float

    def as_csv_row(self) -> dict[str, float | int]:
        return {
            "target_voltage_digits": self.target_voltage_digits,
            "achieved_voltage_effective_digits": self.achieved_voltage_effective_digits,
            "voltage_rmse": self.voltage_rmse,
            "sigma_rmse": self.sigma_rmse,
            "sigma_relative_rmse": self.sigma_relative_rmse,
            "sigma_mae": self.sigma_mae,
            "sigma_max_abs_error": self.sigma_max_abs_error,
            "sigma_effective_digits": self.sigma_effective_digits,
        }


@dataclass(frozen=True)
class VoltageDigitFieldRow:
    """Per-cell conductivity reconstruction error for one digit setting."""

    target_voltage_digits: int
    cell_index: int
    sigma_true: float
    sigma_recon: float
    sigma_error: float
    abs_sigma_error: float

    def as_csv_row(self) -> dict[str, float | int]:
        return {
            "target_voltage_digits": self.target_voltage_digits,
            "cell_index": self.cell_index,
            "sigma_true": self.sigma_true,
            "sigma_recon": self.sigma_recon,
            "sigma_error": self.sigma_error,
            "abs_sigma_error": self.abs_sigma_error,
        }


def _as_float_vector(values: Iterable[float] | np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _validate_target_digits(target_digits: Iterable[int]) -> list[int]:
    digits = [int(value) for value in target_digits]
    if not digits:
        raise ValueError("target_digits must not be empty")
    if any(value <= 0 for value in digits):
        raise ValueError("target_digits must all be positive")
    return digits


def keep_significant_digits(
    values: Iterable[float] | np.ndarray,
    digits: int,
    *,
    method: DigitMethod = "truncate",
) -> np.ndarray:
    """Keep a fixed number of decimal significant digits per value.

    The default truncates toward zero, matching the Word-table examples used
    in the precision study. ``method="round"`` is available for comparison.
    """

    digit_count = int(digits)
    if digit_count <= 0:
        raise ValueError("digits must be positive")
    if method not in {"truncate", "round"}:
        raise ValueError("method must be one of: truncate, round")

    arr = _as_float_vector(values, name="values")
    result = np.zeros_like(arr, dtype=float)
    nonzero = arr != 0.0
    if not np.any(nonzero):
        return result

    powers = np.floor(np.log10(np.abs(arr[nonzero]))).astype(int)
    scales = np.power(10.0, digit_count - powers - 1, dtype=float)
    scaled = arr[nonzero] * scales
    if method == "truncate":
        kept = np.trunc(scaled) / scales
    else:
        kept = np.round(scaled) / scales
    result[nonzero] = kept
    return result


def _relative_rmse(reference: np.ndarray, observed: np.ndarray) -> float:
    ref_rms = float(np.sqrt(np.mean(reference**2)))
    if ref_rms == 0.0:
        return math.nan
    return rmse(reference, observed) / ref_rms


def _summarize_sigma_error(
    *,
    target_digits: int,
    sigma_true: np.ndarray,
    sigma_recon: np.ndarray,
    voltage_true: np.ndarray,
    voltage_digit: np.ndarray,
) -> VoltageDigitSweepSummary:
    error = sigma_recon - sigma_true
    abs_error = np.abs(error)
    return VoltageDigitSweepSummary(
        target_voltage_digits=int(target_digits),
        achieved_voltage_effective_digits=effective_digits_from_rmse(
            voltage_true,
            voltage_digit,
        ),
        voltage_rmse=rmse(voltage_true, voltage_digit),
        sigma_rmse=rmse(sigma_true, sigma_recon),
        sigma_relative_rmse=_relative_rmse(sigma_true, sigma_recon),
        sigma_mae=float(np.mean(abs_error)),
        sigma_max_abs_error=float(np.max(abs_error)),
        sigma_effective_digits=effective_digits_from_rmse(sigma_true, sigma_recon),
    )


def _field_rows(
    *,
    target_digits: int,
    sigma_true: np.ndarray,
    sigma_recon: np.ndarray,
) -> list[VoltageDigitFieldRow]:
    error = sigma_recon - sigma_true
    return [
        VoltageDigitFieldRow(
            target_voltage_digits=int(target_digits),
            cell_index=int(index),
            sigma_true=float(true_value),
            sigma_recon=float(recon_value),
            sigma_error=float(error_value),
            abs_sigma_error=float(abs(error_value)),
        )
        for index, (true_value, recon_value, error_value) in enumerate(
            zip(sigma_true, sigma_recon, error, strict=True)
        )
    ]


def run_voltage_digit_sweep(
    *,
    model: EITLinearizedModel,
    target_digits: Iterable[int],
    ridge: float = 1e-8,
    inverse_backend: str = "pyeidors-rm",
    rm_mode: str = "tikhonov",
    rm_form: str = "param",
    noser_exponent: float = 0.5,
    digit_method: DigitMethod = "truncate",
) -> tuple[list[VoltageDigitSweepSummary], list[VoltageDigitFieldRow]]:
    """Control voltage significant digits and measure conductivity error."""

    sigma_true = _as_float_vector(model.sigma_true, name="model.sigma_true")
    voltage_true = _as_float_vector(model.voltage_true, name="model.voltage_true")
    summaries: list[VoltageDigitSweepSummary] = []
    field_rows: list[VoltageDigitFieldRow] = []

    for digit_count in _validate_target_digits(target_digits):
        voltage_digit = keep_significant_digits(
            voltage_true,
            digit_count,
            method=digit_method,
        )
        sigma_recon = reconstruct_linearized_sigma(
            model=model,
            voltages=voltage_digit,
            ridge=ridge,
            inverse_backend=inverse_backend,
            rm_mode=rm_mode,
            rm_form=rm_form,
            noser_exponent=noser_exponent,
        )
        if sigma_recon.shape != sigma_true.shape:
            raise RuntimeError("reconstructed sigma shape must match sigma_true")

        summaries.append(
            _summarize_sigma_error(
                target_digits=digit_count,
                sigma_true=sigma_true,
                sigma_recon=sigma_recon,
                voltage_true=voltage_true,
                voltage_digit=voltage_digit,
            )
        )
        field_rows.extend(
            _field_rows(
                target_digits=digit_count,
                sigma_true=sigma_true,
                sigma_recon=sigma_recon,
            )
        )

    return summaries, field_rows


def run_voltage_digit_sweep_from_backend(
    *,
    target_digits: Iterable[int],
    forward_backend: str = "pyeidors-fem",
    n_measurements: int = 16,
    n_parameters: int = 8,
    model_seed: int = 20260422,
    fem_n_elec: int = 16,
    fem_grid: int = 4,
    expected_fem_measurements: int | None = None,
    ridge: float = 1e-8,
    inverse_backend: str = "pyeidors-rm",
    rm_mode: str = "tikhonov",
    rm_form: str = "param",
    noser_exponent: float = 0.5,
    digit_method: DigitMethod = "truncate",
) -> tuple[
    EITLinearizedModel,
    list[VoltageDigitSweepSummary],
    list[VoltageDigitFieldRow],
]:
    """Build one model, then run a controlled voltage-digit sweep on it."""

    backend = str(forward_backend).strip().lower()
    if backend in {"surrogate", "linear-surrogate"}:
        model = build_surrogate_linearized_model(
            n_measurements=n_measurements,
            n_parameters=n_parameters,
            seed=model_seed,
        )
    elif backend in {"pyeidors-fem", "fem"}:
        expected_measurements = (
            adjacent_measurement_count(fem_n_elec)
            if expected_fem_measurements is None
            else int(expected_fem_measurements)
        )
        model = build_pyeidors_fem_linearized_model(
            n_elec=fem_n_elec,
            grid=fem_grid,
            expected_measurements=expected_measurements,
        )
    else:
        raise ValueError("forward_backend must be one of: surrogate, pyeidors-fem")

    summaries, field_rows = run_voltage_digit_sweep(
        model=model,
        target_digits=target_digits,
        ridge=ridge,
        inverse_backend=inverse_backend,
        rm_mode=rm_mode,
        rm_form=rm_form,
        noser_exponent=noser_exponent,
        digit_method=digit_method,
    )
    return model, summaries, field_rows


def plot_voltage_digit_sweep(
    summaries: Iterable[VoltageDigitSweepSummary],
    output_path: Path,
    *,
    title: str = "Voltage digit sweep",
    dpi: int = 200,
) -> Path:
    """Render voltage digit control and conductivity error metrics to PNG."""

    rows = list(summaries)
    if not rows:
        raise ValueError("summaries must not be empty")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .digit_plot import configure_times_new_roman

    configure_times_new_roman()
    output = Path(output_path).with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    targets = [row.target_voltage_digits for row in rows]
    achieved_voltage = [row.achieved_voltage_effective_digits for row in rows]
    sigma_digits = [row.sigma_effective_digits for row in rows]
    sigma_rmse = [row.sigma_rmse for row in rows]
    sigma_rel = [row.sigma_relative_rmse for row in rows]
    sigma_mae = [row.sigma_mae for row in rows]
    sigma_max = [row.sigma_max_abs_error for row in rows]

    fig, (digit_ax, error_ax) = plt.subplots(
        2,
        1,
        figsize=(9.0, 7.0),
        sharex=True,
        constrained_layout=True,
    )
    fig.suptitle(title, fontsize=14)

    digit_ax.plot(
        targets,
        achieved_voltage,
        marker="o",
        linewidth=1.9,
        color="#1f77b4",
        label="Achieved voltage digits",
    )
    digit_ax.plot(
        targets,
        sigma_digits,
        marker="s",
        linewidth=1.9,
        color="#d62728",
        label="Sigma effective digits",
    )
    digit_ax.plot(
        targets,
        targets,
        linestyle=":",
        linewidth=1.1,
        color="#444444",
        label="Target voltage digits",
    )
    digit_ax.set_ylabel("Effective decimal digits")
    digit_ax.set_title("Voltage control vs conductivity reconstruction")
    digit_ax.grid(True, alpha=0.28)
    digit_ax.legend(loc="best", fontsize=8)

    error_ax.plot(
        targets,
        sigma_rmse,
        marker="o",
        linewidth=1.8,
        color="#2ca02c",
        label="Sigma RMSE",
    )
    error_ax.plot(
        targets,
        sigma_mae,
        marker="s",
        linewidth=1.8,
        color="#9467bd",
        label="Sigma MAE",
    )
    error_ax.plot(
        targets,
        sigma_max,
        marker="D",
        linewidth=1.8,
        color="#8c564b",
        label="Sigma max abs error",
    )
    rel_ax = error_ax.twinx()
    rel_ax.plot(
        targets,
        sigma_rel,
        marker="^",
        linestyle="--",
        linewidth=1.5,
        color="#17becf",
        label="Relative RMSE",
    )
    error_ax.set_xlabel("Target voltage significant digits")
    error_ax.set_ylabel("Conductivity error")
    rel_ax.set_ylabel("Relative RMSE")
    error_ax.set_title("Conductivity distribution error")
    error_ax.grid(True, alpha=0.28)
    lines, labels = error_ax.get_legend_handles_labels()
    rel_lines, rel_labels = rel_ax.get_legend_handles_labels()
    error_ax.legend(lines + rel_lines, labels + rel_labels, loc="best", fontsize=8)

    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output
