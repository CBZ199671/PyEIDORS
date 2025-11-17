#!/usr/bin/env python3
"""单步高斯-牛顿差分成像脚本。

该脚本基于 `scripts/run_synthetic_parity.py` 中的单步求解流程，
对实测差分数据运行一次与 EIDORS 对齐的 GN 单步重建，输出
导电率图像、测量/预测电压对比图以及误差指标，方便与原有
`run_difference_reconstruction.py` 的多迭代 GN 结果对比。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from fenics import Function
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC_PATH))

from pyeidors.core_system import EITSystem
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.data.measurement_dataset import MeasurementDataset
from pyeidors.inverse.jacobian.direct_jacobian import DirectJacobianCalculator
from pyeidors.visualization import create_visualizer
from pyeidors.electrodes.patterns import StimMeasPatternManager


try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("运行该脚本需要 PyYAML，请先安装: pip install pyyaml") from exc


# ---------------------------------------------------------------------------
# 工具函数


def compute_scale_bias(measured: np.ndarray, model: np.ndarray) -> tuple[float, float]:
    x = np.asarray(model, dtype=float)
    y = np.asarray(measured, dtype=float)
    denom = float(np.dot(x, x))
    if denom < 1e-18:
        return 1.0, float(y.mean())
    scale = float(np.dot(y, x) / denom)
    if abs(scale) < 1e-12:
        scale = 1.0 if scale >= 0 else -1.0
    bias = float(y.mean() - scale * x.mean())
    return scale, bias


def apply_calibration(vector: np.ndarray, scale: float, bias: float) -> np.ndarray:
    arr = np.asarray(vector, dtype=float)
    safe_scale = abs(scale)
    if safe_scale < np.finfo(float).eps:
        safe_scale = 1.0
    return (arr - bias) / safe_scale


def rescale_measurements(measurements: np.ndarray, source_amp: float, target_amp: Optional[float]) -> tuple[np.ndarray, float]:
    if target_amp is None or target_amp <= 0:
        return measurements, 1.0
    if source_amp <= 0:
        raise ValueError("metadata amplitude must be positive to rescale measurements")
    factor = target_amp / source_amp
    return measurements * factor, factor


def normalize_frame_signs(
    measurements: np.ndarray,
    mode: str,
    template: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if mode != "auto":
        return measurements, None

    if template is None:
        template = measurements[0]

    template_centered = template - np.mean(template)
    template_norm = np.linalg.norm(template_centered)
    if template_norm < np.finfo(float).eps:
        return measurements, np.ones(measurements.shape[0])

    signs = np.ones(measurements.shape[0])
    for idx, frame in enumerate(measurements):
        centered = frame - np.mean(frame)
        proj = np.dot(centered, template_centered)
        dominant = np.sign(proj)
        if dominant == 0:
            dominant = 1.0
        signs[idx] = dominant
        measurements[idx] = frame * dominant
    return measurements, signs


def sign_value(arg: str) -> float:
    return -1.0 if arg == "flip" else 1.0


def build_measurement_weights(
    reference_vector: np.ndarray,
    diff_vector: np.ndarray,
    strategy: str,
    floor: float,
    mode: str,
) -> Optional[np.ndarray]:
    if strategy == "none":
        return None
    if strategy == "eidors":
        reference = np.asarray(reference_vector, dtype=float)
        amplitudes = np.abs(reference)
        weights = amplitudes ** 2
        weights = np.where(np.isfinite(weights), weights, 0.0)
        return weights
    if strategy == "difference":
        reference = np.asarray(diff_vector, dtype=float)
        weights = np.maximum(np.abs(reference), floor)
        return weights

    reference = np.asarray(reference_vector, dtype=float)
    if mode == "difference":
        return np.ones_like(reference)

    weights = reference ** 2
    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = np.maximum(weights, floor)
    return weights


def compute_difference_vector(target: np.ndarray, reference: np.ndarray, mode: str) -> np.ndarray:
    if mode == "normalized":
        safe_ref = reference.copy()
        eps = np.finfo(float).eps
        mask = np.abs(safe_ref) < eps
        if np.any(mask):
            safe_ref[mask] = np.sign(safe_ref[mask]) * eps + eps
        return target / safe_ref - 1.0
    return target - reference


def compute_normalized_difference(
    reference_frame: np.ndarray,
    target_frame: np.ndarray,
    fwd_model,
    mode: str,
    measurement_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pattern_manager = getattr(fwd_model, "pattern_manager", None)
    if pattern_manager is None:
        raise RuntimeError("Forward model缺少 pattern_manager，无法执行差分预处理")

    # 应用 meas_select （EIDORS 相当于 filt_data）
    if measurement_mask is None:
        measurement_mask = np.ones_like(reference_frame, dtype=bool)

    ref = reference_frame[measurement_mask]
    tar = target_frame[measurement_mask]

    # 若模式为 normalized，执行 vi./vh - 1
    if mode == "normalized":
        safe_ref = ref.copy()
        eps = np.finfo(float).eps
        mask = np.abs(safe_ref) < eps
        if np.any(mask):
            safe_ref[mask] = np.sign(safe_ref[mask]) * eps + eps
        diff = tar / safe_ref - 1.0
    else:
        diff = tar - ref

    return diff, ref, tar


def compute_predicted_difference(
    system: EITSystem,
    image,
    scale: float,
    bias: float,
    reference_vector: np.ndarray,
    mode: str,
) -> np.ndarray:
    sim_data, _ = system.fwd_model.fwd_solve(image)
    calibrated = apply_calibration(sim_data.meas, scale, bias)
    return compute_difference_vector(calibrated, reference_vector, mode)


def solve_single_step_delta(
    system: EITSystem,
    baseline_image,
    diff_vector: np.ndarray,
    baseline_vector: np.ndarray,
    weight_reference: np.ndarray,
    mode: str,
    args: argparse.Namespace,
    weight_strategy: str,
) -> tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    jacobian_calculator = getattr(getattr(system, "reconstructor", None), "jacobian_calculator", None)
    if jacobian_calculator is None:
        jacobian_calculator = DirectJacobianCalculator(system.fwd_model)

    sigma_fn = Function(system.fwd_model.V_sigma)
    sigma_fn.vector()[:] = baseline_image.elem_data
    jacobian = jacobian_calculator.calculate(sigma_fn, method=args.single_step_jacobian_method)
    if getattr(getattr(system, "reconstructor", None), "negate_jacobian", False):
        jacobian = -jacobian

    weights = build_measurement_weights(
        weight_reference,
        diff_vector,
        weight_strategy,
        args.meas_weight_floor,
        mode,
    )
    dv = np.asarray(diff_vector, dtype=float)
    if weights is None:
        jacobian_weighted = jacobian
        dv_weighted = dv
    else:
        sqrt_weights = np.sqrt(weights)
        jacobian_weighted = jacobian * sqrt_weights[:, None]
        dv_weighted = dv * sqrt_weights

    diag_source = jacobian if weights is None else jacobian
    diag_entries = np.sum(diag_source ** 2, axis=0)
    diag_entries = np.maximum(diag_entries, args.noser_floor)
    noser_diag = diag_entries ** args.noser_exponent
    RtR = np.diag(noser_diag)

    hp = max(args.difference_hyperparameter, 0.0)
    lhs = jacobian_weighted.T @ jacobian_weighted + (hp ** 2) * RtR
    rhs = jacobian_weighted.T @ dv_weighted

    try:
        delta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        delta, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)

    linear_dump = {
        "jacobian": jacobian.copy(),
        "weights": None if weights is None else weights.copy(),
        "jacobian_weighted": jacobian_weighted.copy(),
        "dv_weighted": dv_weighted.copy(),
        "lhs": lhs.copy(),
        "rhs": rhs.copy(),
        "noser_diag": noser_diag.copy(),
    }

    return delta, weights, linear_dump


def build_single_step_image(system: EITSystem, baseline_image, delta: np.ndarray, step: float, bounds: Tuple[float, float]):
    min_cond, max_cond = bounds
    elem = baseline_image.elem_data + step * delta
    elem = np.clip(elem, min_cond, max_cond)
    return type(baseline_image)(elem_data=elem, fwd_model=system.fwd_model)


def optimize_step_size(
    system: EITSystem,
    baseline_image,
    delta: np.ndarray,
    target_diff: np.ndarray,
    reference_vector: np.ndarray,
    scale: float,
    bias: float,
    mode: str,
    args: argparse.Namespace,
) -> float:
    bounds = (args.step_size_min, args.step_size_max)

    def objective(step: float) -> float:
        image = build_single_step_image(system, baseline_image, delta, step, args.conductivity_bounds)
        pred = compute_predicted_difference(system, image, scale, bias, reference_vector, mode)
        residual = pred - target_diff
        return float(np.mean(residual ** 2))

    result = minimize_scalar(objective, bounds=bounds, method="bounded", options={"maxiter": args.step_size_maxiter})
    return float(result.x if result.success else 1.0)


def run_single_step_difference(
    system: EITSystem,
    baseline_image,
    diff_vector: np.ndarray,
    measured_reference: np.ndarray,
    simulated_reference: np.ndarray,
    scale: float,
    bias: float,
    mode: str,
    args: argparse.Namespace,
    weight_scale: float = 1.0,
) -> tuple[Any, np.ndarray, Dict[str, Any]]:
    if weight_scale != 1.0:
        measured_reference = measured_reference * weight_scale
    weight_reference = measured_reference if mode == "difference" else simulated_reference
    weight_strategy = "eidors" if args.strict_eidors else args.meas_weight_strategy
    delta, weights, linear_dump = solve_single_step_delta(
        system,
        baseline_image,
        diff_vector,
        measured_reference,
        weight_reference,
        mode,
        args,
        weight_strategy=weight_strategy,
    )
    step = 1.0
    reference_for_prediction = measured_reference
    if args.step_size_calibration:
        step = optimize_step_size(
            system,
            baseline_image,
            delta,
            diff_vector,
            reference_for_prediction,
            scale,
            bias,
            mode,
            args,
        )

    recon_image = build_single_step_image(system, baseline_image, delta, step, args.conductivity_bounds)
    predicted_diff = compute_predicted_difference(
        system,
        recon_image,
        scale,
        bias,
        reference_for_prediction,
        mode,
    )

    metadata: Dict[str, Any] = {
        "step_size": step,
        "jacobian_method": args.single_step_jacobian_method,
        "measurement_weight_strategy": args.meas_weight_strategy,
        "hyperparameter": args.difference_hyperparameter,
        "noser_exponent": args.noser_exponent,
        "conductivity_bounds": args.conductivity_bounds,
        "delta_norm": float(np.linalg.norm(delta)),
    }
    if weights is not None:
        metadata["weight_stats"] = {
            "min": float(np.min(weights)),
            "max": float(np.max(weights)),
            "median": float(np.median(weights)),
        }
    if linear_dump is not None:
        metadata["linear_system"] = linear_dump

    return recon_image, predicted_diff, metadata


def compute_metrics(measured: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    measured = measured.reshape(-1)
    predicted = predicted.reshape(-1)
    error = predicted - measured
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    max_abs = float(np.max(np.abs(error)))
    safe_measured = measured.copy()
    safe_measured[np.abs(safe_measured) < np.finfo(float).eps] = np.finfo(float).eps
    relative = np.abs(error / safe_measured) * 100.0
    corr = float(np.corrcoef(predicted, measured)[0, 1]) if np.std(measured) > 0 and np.std(predicted) > 0 else np.nan
    return {
        "rmse": rmse,
        "mae": mae,
        "max_abs_error": max_abs,
        "relative_error_pct": float(np.mean(relative)),
        "correlation": corr,
    }


def scale_to_fit(predicted: np.ndarray, measured: np.ndarray) -> float:
    """Return scalar s that minimizes ||s*predicted - measured||_2^2."""
    pred = np.asarray(predicted, dtype=float).reshape(-1)
    meas = np.asarray(measured, dtype=float).reshape(-1)
    denom = float(np.dot(pred, pred))
    if denom < 1e-18:
        return 1.0
    return float(np.dot(meas, pred) / denom)


def save_voltage_comparison(measured: np.ndarray, predicted: np.ndarray, output_path: Path, title: str, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    idx = np.arange(measured.size)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(idx, measured, label="Measured", linewidth=1.2)
    ax.plot(idx, predicted, label="Predicted", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Measurement index")
    ax.set_ylabel("Voltage")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_diff_comparison(measured: np.ndarray, predicted: np.ndarray, output_path: Path, dpi: int) -> None:
    """差分域对比：曲线 + 散点（同量纲 vi-vh vs pred_vi-pred_vh）。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    idx = np.arange(measured.size)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(idx, measured, "b-", lw=1.0, label="Measured diff")
    axes[0].plot(idx, predicted, "r--", lw=1.0, label="Predicted diff")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_xlabel("Measurement index")
    axes[0].set_ylabel("Voltage")
    axes[0].set_title("Difference curve")
    axes[1].scatter(measured, predicted, s=10, alpha=0.7)
    vmin = min(measured.min(), predicted.min())
    vmax = max(measured.max(), predicted.max())
    axes[1].plot([vmin, vmax], [vmin, vmax], "k--")
    axes[1].set_xlabel("Measured diff")
    axes[1].set_ylabel("Predicted diff")
    axes[1].grid(alpha=0.3)
    axes[1].set_title("Difference scatter")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# I/O 与主流程


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单步差分成像", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--npz", type=Path, help="包含 measurements 和 metadata 的 npz 文件")
    data_group.add_argument("--csv", type=Path, help="原始 CSV 数据文件")

    parser.add_argument("--metadata", type=Path, help="当使用 CSV 时需要指定的 YAML/JSON 元数据文件")
    parser.add_argument("--use-cols", type=int, nargs="+", default=[0, 2], help="从 CSV 选取哪些列作为帧（零基索引）")
    parser.add_argument("--delimiter", default=",", help="CSV 分隔符")
    parser.add_argument("--reference-frame", type=int, default=0, help="参考帧索引")
    parser.add_argument("--target-frame", type=int, default=1, help="目标帧索引")
    parser.add_argument("--calibration-frame", type=int, default=0, help="用于尺度校准的帧索引（-1 表示跳过）")

    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "difference_single_step", help="输出目录")
    parser.add_argument("--mesh-dir", type=Path, default=REPO_ROOT / "eit_meshes", help="网格目录")
    parser.add_argument("--mesh-name", type=str, help="指定网格名称（可选）")
    parser.add_argument("--refinement", type=int, default=12, help="生成或缓存网格使用的细化级别")
    parser.add_argument("--mesh-radius", type=float, default=1.0, help="网格半径 (用于缓存识别)")
    parser.add_argument("--electrode-coverage", type=float, default=0.5, help="电极覆盖率 (用于缓存识别)")
    parser.add_argument("--background-conductivity", type=float, default=0.08, help="均匀背景导电率 (S/m)")
    parser.add_argument("--figure-dpi", type=int, default=300, help="输出图像 DPI")

    parser.add_argument("--difference-mode", choices=["difference", "normalized"], default="difference", help="差分定义(EIDORS 同款 normalized 或简单差分)")
    parser.add_argument(
        "--diff-orientation",
        choices=["target_minus_reference", "reference_minus_target"],
        default="target_minus_reference",
        help="差分方向：目标-参考或参考-目标",
    )
    parser.add_argument("--frame-sign-mode", choices=["none", "auto"], default="none", help="是否自动统一每帧电压符号")
    parser.add_argument("--reference-sign", choices=["keep", "flip"], default="keep", help="是否翻转参考帧极性")
    parser.add_argument("--target-sign", choices=["keep", "flip"], default="keep", help="是否翻转目标帧极性")
    parser.add_argument("--disable-scale-to-fit", action="store_true", help="关闭预测差分的 post_calibration 缩放")
    parser.add_argument("--swap-reference-target", action="store_true", help="交换参考/目标帧，用于 CSV 顺序与约定相反的情况")
    parser.add_argument("--target-amplitude", type=float, default=None, help="可选: 将实测电流幅值重标到该幅度 (A)")
    parser.add_argument("--negate-jacobian", choices=["true", "false"], default="false", help="是否对雅可比取负以匹配符号")
    parser.add_argument("--pattern-amplitude", type=float, help="覆盖前向模型使用的激励电流幅度 (A)")
    parser.add_argument("--difference-hyperparameter", type=float, default=1e-2, help="单步求解中的 hp 参数")
    parser.add_argument("--noser-exponent", type=float, default=0.5, help="NOSER 对角指数")
    parser.add_argument("--noser-floor", type=float, default=1e-12, help="NOSER 对角线数值下限")
    parser.add_argument("--meas-weight-strategy", choices=["baseline", "difference", "none"], default="baseline", help="测量权重策略")
    parser.add_argument("--meas-weight-floor", type=float, default=1e-9, help="测量权重下限")
    parser.add_argument("--single-step-jacobian-method", choices=["efficient", "traditional"], default="efficient", help="雅可比计算方式")
    parser.add_argument("--conductivity-bounds", type=float, nargs=2, metavar=("MIN", "MAX"), default=(1e-6, 10.0), help="电导率裁剪范围")
    parser.add_argument("--step-size-calibration", action="store_true", help="是否执行步长一维搜索")
    parser.add_argument("--step-size-min", type=float, default=1e-5, help="步长搜索下界")
    parser.add_argument("--step-size-max", type=float, default=1e1, help="步长搜索上界")
    parser.add_argument("--step-size-maxiter", type=int, default=50, help="步长搜索最大迭代次数")
    parser.add_argument("--strict-eidors", action="store_true", help="严格按照 MATLAB/EIDORS 流程，固定帧顺序并仅对预测做 scale_to_fit")
    parser.add_argument("--stim-direction", choices=["ccw", "cw"], help="覆盖激励旋转方向（默认使用 metadata）")
    parser.add_argument("--meas-direction", choices=["ccw", "cw"], help="覆盖测量旋转方向（默认使用 metadata）")
    parser.set_defaults(stim_first_positive=None)
    parser.add_argument("--stim-first-positive", dest="stim_first_positive", action="store_true", help="指定激励对的第一个电极注入正向电流")
    parser.add_argument("--stim-first-negative", dest="stim_first_positive", action="store_false", help="指定激励对的第一个电极注入负向电流")
    parser.add_argument("--save-measured-diff", action="store_true", help="保存测量得到的差分向量 (target-reference) 以便与其他软件对比。")
    parser.add_argument("--dump-linear-system", action="store_true", help="保存最终求解使用的雅可比/权重/线性系统 (npz) 以便对照 MATLAB。")

    return parser.parse_args()


def load_dataset_from_npz(npz_path: Path) -> MeasurementDataset:
    data = np.load(npz_path, allow_pickle=True)
    if "measurements" not in data or "metadata" not in data:
        raise KeyError("npz 文件缺少 measurements 或 metadata 键")
    measurements = data["measurements"]
    metadata_raw = data["metadata"].item()
    metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
    metadata.setdefault("n_frames", int(measurements.shape[0]))
    return MeasurementDataset.from_metadata(measurements, metadata)


def load_dataset_from_csv(
    csv_path: Path,
    metadata_path: Path,
    use_cols: Optional[Sequence[int]],
    delimiter: str,
) -> MeasurementDataset:
    raw = np.loadtxt(csv_path, delimiter=delimiter)
    if raw.ndim == 1:
        raw = raw[:, np.newaxis]
    if use_cols:
        raw = raw[:, list(use_cols)]
    measurements = raw.T

    with metadata_path.open("r", encoding="utf-8") as fh:
        if metadata_path.suffix.lower() in {".yaml", ".yml"}:
            metadata = yaml.safe_load(fh)
        elif metadata_path.suffix.lower() == ".json":
            metadata = json.load(fh)
        else:
            raise ValueError("metadata 文件必须是 YAML 或 JSON")

    metadata = dict(metadata)
    metadata["n_frames"] = int(measurements.shape[0])
    return MeasurementDataset.from_metadata(measurements, metadata)


def load_dataset(args: argparse.Namespace) -> MeasurementDataset:
    if args.npz:
        return load_dataset_from_npz(args.npz)
    if args.csv:
        if args.metadata is None:
            raise ValueError("使用 CSV 时必须提供 --metadata")
        return load_dataset_from_csv(args.csv, args.metadata, args.use_cols, args.delimiter)
    raise RuntimeError("未提供任何输入数据")


def apply_pattern_overrides(
    dataset: MeasurementDataset,
    stim_direction: Optional[str],
    meas_direction: Optional[str],
    stim_first_positive: Optional[bool],
) -> MeasurementDataset:
    overrides: Dict[str, Any] = {}
    if stim_direction:
        overrides["stim_direction"] = stim_direction
    if meas_direction:
        overrides["meas_direction"] = meas_direction
    if stim_first_positive is not None:
        overrides["stim_first_positive"] = bool(stim_first_positive)

    if not overrides:
        return dataset

    new_config = replace(dataset.pattern_config, **overrides)
    pattern_manager = StimMeasPatternManager(new_config)
    expected = pattern_manager.n_meas_total
    actual = dataset.measurements.shape[1]
    if expected != actual:
        raise ValueError(
            "覆盖后的激励/测量模式与测量矩阵列数不匹配："
            f"期望 {expected} 列，实际 {actual} 列"
        )

    dataset.pattern_config = new_config
    dataset.stim_matrix = pattern_manager.stim_matrix.copy()
    dataset.n_stim = pattern_manager.n_stim
    dataset.n_meas_total = pattern_manager.n_meas_total
    dataset.n_meas_per_stim = tuple(pattern_manager.n_meas_per_stim)
    dataset.metadata = dict(dataset.metadata)
    dataset.metadata.update(
        {
            "stim_direction": new_config.stim_direction,
            "meas_direction": new_config.meas_direction,
            "stim_first_positive": new_config.stim_first_positive,
        }
    )
    return dataset


def save_conductivity_figures(recon_image, baseline_image, output_dir: Path, dpi: int, metadata: Dict[str, Any]) -> None:
    visualizer = create_visualizer()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_cond = visualizer.plot_conductivity(
        recon_image.fwd_model.mesh,
        recon_image.elem_data,
        title=None,
        save_path=str(output_dir / "reconstruction.png"),
        minimal=True,
    )
    plt.close(fig_cond)
    if baseline_image is not None:
        delta = recon_image.elem_data - baseline_image.elem_data
        limit = float(np.max(np.abs(delta)))
        if not np.isfinite(limit) or limit < 1e-18:
            limit = 1e-18
        fig_delta = visualizer.plot_conductivity(
            recon_image.fwd_model.mesh,
            delta,
            title=None,
            save_path=str(output_dir / "reconstruction_delta.png"),
            minimal=True,
            vmin=-limit,
            vmax=limit,
        )
        plt.close(fig_delta)


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args)
    dataset = apply_pattern_overrides(
        dataset,
        stim_direction=args.stim_direction,
        meas_direction=args.meas_direction,
        stim_first_positive=args.stim_first_positive,
    )
    if args.pattern_amplitude is not None:
        dataset.pattern_config.amplitude = float(args.pattern_amplitude)

    ref_idx = args.reference_frame
    tgt_idx = args.target_frame
    if not 0 <= ref_idx < dataset.measurements.shape[0]:
        raise IndexError("参考帧索引超出范围")
    if not 0 <= tgt_idx < dataset.measurements.shape[0]:
        raise IndexError("目标帧索引超出范围")
    if ref_idx == tgt_idx:
        raise ValueError("参考帧和目标帧必须不同")

    system = EITSystem(
        n_elec=dataset.n_elec,
        pattern_config=dataset.pattern_config,
        base_conductivity=args.background_conductivity,
    )
    mesh = load_or_create_mesh(
        mesh_dir=str(args.mesh_dir),
        mesh_name=args.mesh_name,
        n_elec=dataset.n_elec,
        refinement=max(args.refinement, 4),
        radius=args.mesh_radius,
        electrode_coverage=args.electrode_coverage,
    )
    system.setup(mesh=mesh)
    system.reconstructor.negate_jacobian = (args.negate_jacobian == "true")
    baseline_image = system.create_homogeneous_image()
    baseline_data = system.forward_solve(baseline_image)

    if args.strict_eidors:
        scale = 1.0
        bias = 0.0
        applied_scale = 1.0
        scale_sign = 1.0
        sim_reference_calibrated = baseline_data.meas.copy()
    else:
        scale = 1.0
        bias = 0.0
        applied_scale = 1.0
        scale_sign = 1.0
        sim_reference_calibrated = apply_calibration(baseline_data.meas, scale, bias)
        if args.calibration_frame >= 0:
            if args.calibration_frame >= dataset.measurements.shape[0]:
                raise IndexError("calibration_frame 超出范围")
            cal_vec = dataset.measurements[args.calibration_frame].copy()
            scale, bias = compute_scale_bias(cal_vec, baseline_data.meas)
            scale_sign = 1.0 if scale >= 0 else -1.0
            dataset.measurements = apply_calibration(dataset.measurements, scale, bias)
            applied_scale = abs(scale)
            sim_reference_calibrated = apply_calibration(baseline_data.meas, scale, bias)

    measurements = dataset.measurements.copy()
    orig_amp = float(dataset.metadata.get("amplitude", 1.0))
    if args.strict_eidors:
        amp_factor = 1.0
        frame_signs = None
        if args.target_amplitude:
            measurements, amp_factor = rescale_measurements(measurements, orig_amp, args.target_amplitude)
            dataset.measurements = measurements
    else:
        measurements, amp_factor = rescale_measurements(measurements, orig_amp, args.target_amplitude)
        template_frame = measurements[ref_idx].copy()
        measurements, frame_signs = normalize_frame_signs(measurements, args.frame_sign_mode, template_frame)
        dataset.measurements = measurements
    ref_sign = sign_value(args.reference_sign)
    tgt_sign = sign_value(args.target_sign)
    measurements[ref_idx] *= ref_sign
    measurements[tgt_idx] *= tgt_sign

    pattern_amp = float(getattr(dataset.pattern_config, "amplitude", orig_amp))
    effective_meas_amp = max(orig_amp * amp_factor, np.finfo(float).eps)
    weight_scale = pattern_amp / effective_meas_amp

    reference_vector = measurements[ref_idx].copy()
    target_vector = measurements[tgt_idx].copy()
    swapped = False
    if args.swap_reference_target:
        reference_vector, target_vector = target_vector, reference_vector
        swapped = True

    solver_mode = "difference" if args.strict_eidors else args.difference_mode

    if args.diff_orientation == "target_minus_reference":
        ref_vec = reference_vector
        tgt_vec = target_vector
        selected_orientation = "target_minus_reference"
    else:
        ref_vec = target_vector
        tgt_vec = reference_vector
        selected_orientation = "reference_minus_target"

    if args.strict_eidors:
        diff_vector = tgt_vec - ref_vec
        ref_used = ref_vec
        tar_used = tgt_vec
    else:
        diff_vector, ref_used, tar_used = compute_normalized_difference(
            ref_vec,
            tgt_vec,
            system.fwd_model,
            solver_mode,
        )

    recon_image, predicted_diff_raw, solver_meta = run_single_step_difference(
        system=system,
        baseline_image=baseline_image,
        diff_vector=diff_vector,
        measured_reference=ref_vec,
        simulated_reference=sim_reference_calibrated,
        scale=scale,
        bias=bias,
        mode=solver_mode,
        args=args,
        weight_scale=weight_scale,
    )

    if args.disable_scale_to_fit:
        post_scale, post_bias = 1.0, 0.0
        predicted_diff = predicted_diff_raw
    elif args.strict_eidors:
        post_scale = scale_to_fit(predicted_diff_raw, diff_vector)
        post_bias = 0.0
        predicted_diff = predicted_diff_raw * post_scale + post_bias
    else:
        post_scale, post_bias = compute_scale_bias(diff_vector, predicted_diff_raw)
        predicted_diff = predicted_diff_raw * post_scale + post_bias
    residual = float(np.linalg.norm(predicted_diff - diff_vector))

    linear_dump = solver_meta.pop("linear_system", None)

    measured_diff = tar_used - ref_used

    dataset_name = args.csv.stem if args.csv else args.npz.stem
    output_dir = args.output_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_measured_diff:
        np.savetxt(output_dir / "measured_difference.csv", measured_diff, delimiter=",")
    if args.dump_linear_system and linear_dump is not None:
        np.savez(output_dir / "linear_system.npz", **linear_dump)

    metadata = {
        "target_idx": tgt_idx,
        "reference_idx": ref_idx,
        "calibration": {
            "scale": scale,
            "applied_scale": applied_scale,
            "scale_sign": scale_sign,
            "bias": bias,
            "amplitude_factor": amp_factor,
        },
        "difference_mode": solver_mode,
        "frame_sign_mode": args.frame_sign_mode,
        "frame_signs": None if frame_signs is None else frame_signs.tolist(),
        "target_amplitude": args.target_amplitude,
        "negate_jacobian": args.negate_jacobian,
        "diff_orientation": selected_orientation,
        "post_calibration": {
            "scale": post_scale,
            "bias": post_bias,
        },
        "strict_eidors": args.strict_eidors,
        "sign_controls": {
            "reference": args.reference_sign,
            "target": args.target_sign,
            "swapped": swapped,
        },
    }
    save_conductivity_figures(recon_image, baseline_image, output_dir, args.figure_dpi, metadata)

    save_voltage_comparison(
        measured=diff_vector,
        predicted=predicted_diff,
        output_path=output_dir / "voltage_comparison.png",
        title="Single-step difference voltages",
        dpi=args.figure_dpi,
    )
    save_diff_comparison(
        measured=diff_vector,
        predicted=predicted_diff,
        output_path=output_dir / "diff_comparison.png",
        dpi=args.figure_dpi,
    )

    np.savetxt(output_dir / "predicted_difference_raw.csv", predicted_diff_raw, delimiter=",")
    np.savetxt(output_dir / "predicted_difference.csv", predicted_diff, delimiter=",")
    metrics = compute_metrics(diff_vector, predicted_diff)
    metrics.update({
        "solver": solver_meta,
        "difference_mode": args.difference_mode,
        "residual_norm": residual,
        "diff_orientation": selected_orientation,
        "post_calibration": {
            "scale": post_scale,
            "bias": post_bias,
        },
        "strict_eidors": args.strict_eidors,
    })
    (output_dir / "metrics.json").write_text(json.dumps({"difference": metrics}, indent=2))

    print(f"单步差分重建完成，输出目录: {output_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
