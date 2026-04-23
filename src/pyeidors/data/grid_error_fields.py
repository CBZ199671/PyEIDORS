"""Conductivity field plots for grid-dependent EIT reconstruction errors."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .adc_quantization import effective_digits_from_rmse, rmse
from .eit_digit_metrics import (
    EITLinearizedModel,
    adjacent_measurement_count,
    build_pyeidors_fem_linearized_model,
    build_surrogate_linearized_model,
    reconstruct_linearized_sigma,
)
from .voltage_digit_sweep import keep_significant_digits


SUMMARY_FIELDS = [
    "fem_grid",
    "n_elec",
    "n_measurements",
    "n_parameters",
    "ridge",
    "target_voltage_digits",
    "voltage_rmse",
    "achieved_voltage_effective_digits",
    "sigma_rmse",
    "sigma_relative_rmse",
    "sigma_mae",
    "sigma_max_abs_error",
    "sigma_effective_digits",
    "max_abs_error_cell_index",
    "max_abs_error_x",
    "max_abs_error_y",
]

FIELD_FIELDS = [
    "fem_grid",
    "cell_index",
    "x",
    "y",
    "sigma_true",
    "sigma_recon",
    "sigma_error",
    "abs_sigma_error",
]


@dataclass(frozen=True)
class GridErrorSummary:
    """Summary metrics for one FEM grid conductivity-field comparison."""

    fem_grid: int
    n_elec: int
    n_measurements: int
    n_parameters: int
    ridge: float
    target_voltage_digits: int
    voltage_rmse: float
    achieved_voltage_effective_digits: float
    sigma_rmse: float
    sigma_relative_rmse: float
    sigma_mae: float
    sigma_max_abs_error: float
    sigma_effective_digits: float
    max_abs_error_cell_index: int
    max_abs_error_x: float
    max_abs_error_y: float

    def as_csv_row(self) -> dict[str, float | int]:
        return {
            "fem_grid": self.fem_grid,
            "n_elec": self.n_elec,
            "n_measurements": self.n_measurements,
            "n_parameters": self.n_parameters,
            "ridge": self.ridge,
            "target_voltage_digits": self.target_voltage_digits,
            "voltage_rmse": self.voltage_rmse,
            "achieved_voltage_effective_digits": self.achieved_voltage_effective_digits,
            "sigma_rmse": self.sigma_rmse,
            "sigma_relative_rmse": self.sigma_relative_rmse,
            "sigma_mae": self.sigma_mae,
            "sigma_max_abs_error": self.sigma_max_abs_error,
            "sigma_effective_digits": self.sigma_effective_digits,
            "max_abs_error_cell_index": self.max_abs_error_cell_index,
            "max_abs_error_x": self.max_abs_error_x,
            "max_abs_error_y": self.max_abs_error_y,
        }


@dataclass(frozen=True)
class GridErrorFieldRow:
    """Per-cell conductivity-field error for one FEM grid."""

    fem_grid: int
    cell_index: int
    x: float
    y: float
    sigma_true: float
    sigma_recon: float
    sigma_error: float
    abs_sigma_error: float

    def as_csv_row(self) -> dict[str, float | int]:
        return {
            "fem_grid": self.fem_grid,
            "cell_index": self.cell_index,
            "x": self.x,
            "y": self.y,
            "sigma_true": self.sigma_true,
            "sigma_recon": self.sigma_recon,
            "sigma_error": self.sigma_error,
            "abs_sigma_error": self.abs_sigma_error,
        }


@dataclass(frozen=True)
class GridErrorCase:
    """Full field data needed for one grid's report and plot."""

    summary: GridErrorSummary
    field_rows: list[GridErrorFieldRow]
    sigma_true: np.ndarray
    sigma_recon: np.ndarray
    sigma_error: np.ndarray
    parameter_points: np.ndarray
    mesh_points: np.ndarray | None = None
    mesh_cells: np.ndarray | None = None


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


def _relative_rmse(reference: np.ndarray, observed: np.ndarray) -> float:
    ref_rms = float(np.sqrt(np.mean(reference**2)))
    if ref_rms == 0.0:
        return math.nan
    return rmse(reference, observed) / ref_rms


def _fallback_parameter_points(n_parameters: int) -> np.ndarray:
    count = int(n_parameters)
    if count <= 0:
        raise ValueError("n_parameters must be positive")
    side = int(math.ceil(math.sqrt(count)))
    xs = (np.arange(side, dtype=float) + 0.5) / side
    ys = (np.arange(side, dtype=float) + 0.5) / side
    xx, yy = np.meshgrid(xs, ys[::-1], indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()])[:count]


def _model_parameter_points(model: EITLinearizedModel, n_parameters: int) -> np.ndarray:
    if model.parameter_points is None:
        return _fallback_parameter_points(n_parameters)
    points = np.asarray(model.parameter_points, dtype=float)
    if points.ndim != 2 or points.shape[0] != n_parameters or points.shape[1] < 2:
        raise ValueError("model parameter_points must have shape (n_parameters, >=2)")
    return points[:, :2].copy()


def _model_mesh_geometry(
    model: EITLinearizedModel,
    n_parameters: int,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if model.mesh_points is None or model.mesh_cells is None:
        return None, None
    points = np.asarray(model.mesh_points, dtype=float)
    cells = np.asarray(model.mesh_cells, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("model mesh_points must have shape (n_points, >=2)")
    if cells.ndim != 2 or cells.shape[0] != n_parameters:
        raise ValueError("model mesh_cells must have one row per parameter")
    return points[:, :2].copy(), cells.copy()


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


def evaluate_grid_error_case(
    *,
    model: EITLinearizedModel,
    fem_grid: int,
    n_elec: int,
    ridge: float,
    target_voltage_digits: int,
    inverse_backend: str = "pyeidors-rm",
    rm_mode: str = "tikhonov",
    rm_form: str = "param",
) -> GridErrorCase:
    """Reconstruct conductivity for one grid and return field-error data."""

    sigma_true = _as_float_vector(model.sigma_true, name="model.sigma_true")
    voltage_true = _as_float_vector(model.voltage_true, name="model.voltage_true")
    voltage_measured = keep_significant_digits(voltage_true, int(target_voltage_digits))
    sigma_recon = reconstruct_linearized_sigma(
        model=model,
        voltages=voltage_measured,
        ridge=float(ridge),
        inverse_backend=inverse_backend,
        rm_mode=rm_mode,
        rm_form=rm_form,
    )
    if sigma_recon.shape != sigma_true.shape:
        raise RuntimeError("reconstructed sigma shape must match sigma_true")

    sigma_error = sigma_recon - sigma_true
    abs_error = np.abs(sigma_error)
    points = _model_parameter_points(model, sigma_true.size)
    mesh_points, mesh_cells = _model_mesh_geometry(model, sigma_true.size)
    max_abs_index = int(np.argmax(abs_error))
    summary = GridErrorSummary(
        fem_grid=int(fem_grid),
        n_elec=int(n_elec),
        n_measurements=int(model.n_measurements),
        n_parameters=int(sigma_true.size),
        ridge=float(ridge),
        target_voltage_digits=int(target_voltage_digits),
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
        max_abs_error_cell_index=max_abs_index,
        max_abs_error_x=float(points[max_abs_index, 0]),
        max_abs_error_y=float(points[max_abs_index, 1]),
    )
    field_rows = [
        GridErrorFieldRow(
            fem_grid=int(fem_grid),
            cell_index=int(index),
            x=float(point[0]),
            y=float(point[1]),
            sigma_true=float(true_value),
            sigma_recon=float(recon_value),
            sigma_error=float(error_value),
            abs_sigma_error=float(abs(error_value)),
        )
        for index, (point, true_value, recon_value, error_value) in enumerate(
            zip(points, sigma_true, sigma_recon, sigma_error, strict=True)
        )
    ]
    return GridErrorCase(
        summary=summary,
        field_rows=field_rows,
        sigma_true=sigma_true,
        sigma_recon=sigma_recon,
        sigma_error=sigma_error,
        parameter_points=points,
        mesh_points=mesh_points,
        mesh_cells=mesh_cells,
    )


def run_grid_error_fields(
    *,
    fem_grid_levels: Iterable[int],
    forward_backend: str = "pyeidors-fem",
    n_elec: int = 16,
    expected_measurements: int | None = None,
    ridge: float = 1e-2,
    target_voltage_digits: int = 6,
    inverse_backend: str = "pyeidors-rm",
    rm_mode: str = "tikhonov",
    rm_form: str = "param",
    n_measurements: int = 16,
    n_parameters: int = 8,
    model_seed: int = 20260422,
) -> list[GridErrorCase]:
    """Build each grid independently and locate its reconstruction error field."""

    cases: list[GridErrorCase] = []
    for grid in _positive_int_levels(fem_grid_levels, name="fem_grid_levels"):
        model = _build_model_for_grid(
            forward_backend=forward_backend,
            fem_grid=grid,
            n_elec=n_elec,
            expected_measurements=expected_measurements,
            n_measurements=n_measurements,
            n_parameters=n_parameters,
            model_seed=model_seed,
        )
        cases.append(
            evaluate_grid_error_case(
                model=model,
                fem_grid=grid,
                n_elec=n_elec,
                ridge=ridge,
                target_voltage_digits=target_voltage_digits,
                inverse_backend=inverse_backend,
                rm_mode=rm_mode,
                rm_form=rm_form,
            )
        )
    return cases


def format_grid_error_report(
    cases: Iterable[GridErrorCase],
    *,
    title: str = "T18 网格电导率误差场报告",
) -> str:
    """Format grid field summaries and error-location notes as Markdown."""

    case_list = list(cases)
    if not case_list:
        raise ValueError("cases must not be empty")
    ranked = sorted(
        case_list,
        key=lambda case: case.summary.sigma_relative_rmse,
        reverse=True,
    )
    lines = [
        f"# {title}",
        "",
        "## 汇总",
        "",
        "| fem_grid | n_parameters | sigma_relative_rmse | sigma_effective_digits | sigma_mae | sigma_max_abs_error | max_abs_error_cell | max_abs_error_xy |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in case_list:
        row = case.summary
        lines.append(
            f"| {row.fem_grid} | {row.n_parameters} | "
            f"{row.sigma_relative_rmse:.12g} | {row.sigma_effective_digits:.12g} | "
            f"{row.sigma_mae:.12g} | {row.sigma_max_abs_error:.12g} | "
            f"{row.max_abs_error_cell_index} | "
            f"({row.max_abs_error_x:.4f}, {row.max_abs_error_y:.4f}) |"
        )

    lines.extend(
        [
            "",
            "## 误差定位",
            "",
            "| fem_grid | max_positive_error_cell | max_positive_error | xy | max_negative_error_cell | max_negative_error | xy |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case in case_list:
        errors = case.sigma_error
        max_pos = int(np.argmax(errors))
        max_neg = int(np.argmin(errors))
        pos_xy = case.parameter_points[max_pos]
        neg_xy = case.parameter_points[max_neg]
        lines.append(
            f"| {case.summary.fem_grid} | {max_pos} | {errors[max_pos]:.12g} | "
            f"({pos_xy[0]:.4f}, {pos_xy[1]:.4f}) | {max_neg} | "
            f"{errors[max_neg]:.12g} | ({neg_xy[0]:.4f}, {neg_xy[1]:.4f}) |"
        )

    top = ranked[0].summary
    lines.extend(
        [
            "",
            "## 结论提示",
            "",
            f"- 本组网格中 `fem_grid={top.fem_grid}` 的 `sigma_relative_rmse` 最大，值为 `{top.sigma_relative_rmse:.12g}`。",
            "- 不同网格 cell 数不同，本报告只比较汇总指标和场形态；不做跨网格逐单元相减。",
            "- 当前重建场接近参考电导率 1.0，误差形态主要跟 `sigma_true` 相对 1.0 的偏离位置一致。",
        ]
    )
    return "\n".join(lines) + "\n"


def _field_limits(cases: list[GridErrorCase]) -> tuple[float, float, float]:
    sigma_values = np.concatenate(
        [case.sigma_true for case in cases] + [case.sigma_recon for case in cases]
    )
    sigma_min = float(np.min(sigma_values))
    sigma_max = float(np.max(sigma_values))
    error_limit = float(max(np.max(np.abs(case.sigma_error)) for case in cases))
    return sigma_min, sigma_max, error_limit


def _draw_field(
    *,
    ax,
    case: GridErrorCase,
    values: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
):
    if (
        case.mesh_points is not None
        and case.mesh_cells is not None
        and case.mesh_cells.shape[1] == 3
    ):
        import matplotlib.tri as mtri

        triangulation = mtri.Triangulation(
            case.mesh_points[:, 0],
            case.mesh_points[:, 1],
            case.mesh_cells,
        )
        return ax.tripcolor(
            triangulation,
            facecolors=values,
            shading="flat",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolors="#ffffff",
            linewidth=0.15,
        )
    return ax.scatter(
        case.parameter_points[:, 0],
        case.parameter_points[:, 1],
        c=values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=42,
        marker="s",
        linewidths=0.0,
    )


def plot_grid_error_fields(
    cases: Iterable[GridErrorCase],
    output_path: Path,
    *,
    title: str = "Grid conductivity error fields",
    dpi: int = 200,
) -> Path:
    """Render ``sigma_true``, ``sigma_recon`` and ``sigma_error`` per grid."""

    case_list = list(cases)
    if not case_list:
        raise ValueError("cases must not be empty")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .digit_plot import configure_times_new_roman

    configure_times_new_roman()
    sigma_min, sigma_max, error_limit = _field_limits(case_list)
    output = Path(output_path).with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(case_list)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(10.5, max(3.0, 2.8 * n_rows)),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(title, fontsize=14)
    columns = [
        ("Sigma true", "sigma_true", "viridis", sigma_min, sigma_max),
        ("Sigma recon", "sigma_recon", "viridis", sigma_min, sigma_max),
        ("Sigma error", "sigma_error", "coolwarm", -error_limit, error_limit),
    ]
    for row_idx, case in enumerate(case_list):
        for col_idx, (label, attr, cmap, vmin, vmax) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            values = np.asarray(getattr(case, attr), dtype=float)
            image = _draw_field(
                ax=ax,
                case=case,
                values=values,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.set_title(f"grid={case.summary.fem_grid} {label}", fontsize=10)
            ax.grid(False)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)

    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output
