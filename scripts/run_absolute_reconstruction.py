#!/usr/bin/env python3
"""运行一次绝对成像重建并保存可视化结果。"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
from pathlib import Path
import dataclasses
from typing import Optional, Sequence, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("运行该脚本需要 PyYAML，请先安装: pip install pyyaml") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC_PATH))

from pyeidors.core_system import EITSystem
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.data.measurement_dataset import MeasurementDataset
from pyeidors.electrodes.patterns import StimMeasPatternManager
from pyeidors.visualization import create_visualizer

LOGGER = logging.getLogger("absolute_reconstruction")


def compute_scale_bias(measured: np.ndarray, model: np.ndarray) -> tuple[float, float]:
    """计算把模型响应缩放/平移到实测尺度的线性系数。"""
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


MODE_PRESETS: Dict[str, Dict[str, Any]] = {
    "accurate": {
        "max_iterations": 15,
        "regularization_param": 1e-2,
        "line_search_steps": 12,
        "convergence_tol": 5e-5,
        "measurement_weight_strategy": "scaled_baseline",
        "negate_jacobian": True,
        "max_step": 0.05,
    },
    "fast": {
        "max_iterations": 5,
        "regularization_param": 2e-2,
        "line_search_steps": 6,
        "convergence_tol": 1e-3,
        "measurement_weight_strategy": "scaled_baseline",
        "negate_jacobian": True,
    },
}


def apply_mode_preset(reconstructor, mode: str) -> None:
    """根据预设模式调整重建器参数。"""
    preset = MODE_PRESETS[mode]
    reconstructor.max_iterations = preset["max_iterations"]
    reconstructor.regularization_param = preset["regularization_param"]
    reconstructor.line_search_steps = preset["line_search_steps"]
    reconstructor.convergence_tol = preset["convergence_tol"]
    reconstructor.measurement_weight_strategy = preset.get("measurement_weight_strategy", "none")
    reconstructor.use_measurement_weights = reconstructor.measurement_weight_strategy != "none"
    if hasattr(reconstructor, "negate_jacobian"):
        reconstructor.negate_jacobian = preset.get("negate_jacobian", True)
    if hasattr(reconstructor, "max_step"):
        reconstructor.max_step = preset.get("max_step", 1.0)
    if hasattr(reconstructor, "min_step"):
        reconstructor.min_step = preset.get("min_step", 0.1)


def align_measurement_polarity(
    dataset: MeasurementDataset,
    baseline_vector: np.ndarray,
    mode: str,
    logger: logging.Logger = LOGGER,
) -> list[int]:
    """根据均匀场方向统一测量极性。"""

    if mode == "none":
        return []

    baseline = np.asarray(baseline_vector, dtype=float)
    if np.linalg.norm(baseline) < 1e-12:
        logger.warning("均匀场电压几乎为零，无法执行极性校正，已跳过。")
        return []

    flipped: list[int] = []
    for idx in range(dataset.measurements.shape[0]):
        frame = dataset.measurements[idx]
        dot = float(np.dot(frame, baseline))
        if dot < 0:
            dataset.measurements[idx] = -frame
            flipped.append(idx)

    if flipped:
        logger.info("极性校正：翻转帧索引 %s", flipped)
    else:
        logger.info("极性校正：所有帧已与均匀场方向一致。")

    return flipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行绝对成像重建",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--npz", type=Path, help="包含 measurements 和 metadata 的 npz 文件")
    data_group.add_argument("--csv", type=Path, help="原始 CSV 数据文件")

    parser.add_argument("--metadata", type=Path, help="当使用 CSV 时需要指定的 YAML/JSON 元数据文件")
    parser.add_argument(
        "--use-cols",
        type=int,
        nargs="+",
        default=[2],
        help="从 CSV 中选取哪些列作为帧（零基索引），默认仅使用当前帧实部",
    )
    parser.add_argument(
        "--use-both-components",
        action="store_true",
        help="当存在复数数据时，同时使用实部和虚部进行重建",
    )
    parser.add_argument("--delimiter", default=",", help="CSV 分隔符")
    parser.add_argument("--target-frame", type=int, default=0, help="要重建的帧索引")
    parser.add_argument(
        "--calibration-frame",
        type=int,
        default=0,
        help="用于尺度校准的帧索引（-1 表示跳过校准）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "absolute_reconstruction",
        help="保存输出的目录",
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=REPO_ROOT / "eit_meshes",
        help="FEniCS 网格所在目录",
    )
    parser.add_argument("--mesh-name", type=str, help="指定要加载的网格名称（不含扩展名）")
    parser.add_argument(
        "--fallback-mesh-size",
        type=float,
        default=0.08,
        help="缺少网格时使用的生成尺寸 (传递给 gmsh/workflows)",
    )
    parser.add_argument(
        "--refinement",
        type=int,
        default=12,
        help="load_or_create_mesh 使用的细化级别",
    )
    parser.add_argument(
        "--mesh-radius",
        type=float,
        default=1.0,
        help="网格半径 (用于缓存识别)",
    )
    parser.add_argument(
        "--electrode-coverage",
        type=float,
        default=0.5,
        help="电极覆盖率 (用于缓存识别)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="高斯牛顿最大迭代次数（仅在未选择 mode 时生效）",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=None,
        help="线搜索的最大步长（覆盖重建器的 max_step，默认 1.0）",
    )
    parser.add_argument(
        "--line-search-steps",
        type=int,
        default=None,
        help="线搜索的步数（覆盖重建器的 line_search_steps，默认 8）",
    )
    parser.add_argument(
        "--min-step",
        type=float,
        default=None,
        help="线搜索最小步长下限（避免步长被压到0，默认0.1）",
    )
    parser.add_argument(
        "--step-schedule",
        type=float,
        nargs="+",
        help="自定义每次迭代的固定步长序列，启用后跳过线搜索（例如: 5 1 0.5 0.2 0.1）",
    )
    parser.add_argument(
        "--regularization-param",
        type=float,
        help="覆盖默认的正则化参数 λ（若未指定，则使用 mode 或默认值）",
    )
    parser.add_argument(
        "--contact-impedance",
        type=float,
        default=1e-5,
        help="每个电极的接触阻抗 (Ω·m²)，与EIDORS默认值保持一致",
    )
    parser.add_argument(
        "--mode",
        choices=["accurate", "fast"],
        help="可选的预设重建模式：accurate更高精度，fast追求速度",
    )
    parser.add_argument(
        "--weight-strategy",
        choices=["none", "baseline", "scaled_baseline", "difference"],
        help="覆盖默认的测量加权策略",
    )
    parser.add_argument(
        "--measurement-scale",
        type=float,
        default=1.0,
        help="全局缩放实测数据的系数（<1 缩小实测，>1 放大量测），用于模型/硬件量纲对齐",
    )
    parser.add_argument(
        "--auto-measurement-scale",
        action="store_true",
        help="根据校准帧与均匀前向解，自动估计测量缩放系数并应用到所有帧",
    )
    parser.add_argument(
        "--auto-trim-percent",
        type=float,
        default=5.0,
        help="自动缩放时，去掉两端百分比后取均值(默认5%%)以抑制极端值",
    )
    parser.add_argument(
        "--auto-scale-min",
        type=float,
        default=1e-6,
        help="自动缩放系数的下限，避免过度缩小/放大",
    )
    parser.add_argument(
        "--auto-scale-max",
        type=float,
        default=1e6,
        help="自动缩放系数的上限，避免过度缩小/放大",
    )
    parser.add_argument(
        "--fit-scale-baseline",
        action="store_true",
        help="使用均匀场与目标帧的裁剪均值对齐测量尺度（单一缩放源，替代 auto_measurement_scale）",
    )
    parser.add_argument(
        "--raw-gn",
        action="store_true",
        help="开启最简管线：禁用极性对齐、测量缩放/自动缩放、线性校准、测量权重和扩展线搜索，仅保留核心 GN",
    )
    parser.add_argument(
        "--polarity-mode",
        choices=["auto", "none"],
        default="auto",
        help="测量极性对齐方式：auto 根据均匀场自动翻转，none 则跳过",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    parser.add_argument(
        "--convergence-tol",
        type=float,
        default=1e-4,
        help="收敛判据：导电率相对变化低于该阈值则提前停止",
    )
    parser.add_argument(
        "--min-iterations",
        type=int,
        default=1,
        help="至少执行的迭代步数（防止过早收敛）",
    )
    parser.add_argument(
        "--stim-scale",
        type=float,
        default=1.0,
        help="激励电流缩放因子（放大/缩小前向模型的激励电流），用于拉齐模型与实测的量纲",
    )
    parser.add_argument(
        "--auto-stim-scale",
        action="store_true",
        help="自动估计激励电流缩放，使均匀场预测幅值与实测目标帧量纲对齐（固定桶/电极尺寸，只调电流幅值）",
    )

    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="[%(levelname)s] %(message)s")


def load_dataset_from_npz(npz_path: Path) -> MeasurementDataset:
    LOGGER.info("加载 NPZ 数据: %s", npz_path)
    data = np.load(npz_path, allow_pickle=True)
    if "measurements" not in data or "metadata" not in data:
        raise KeyError("npz 文件缺少 measurements 或 metadata 键")
    measurements = data["measurements"]
    metadata_raw = data["metadata"].item()
    metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
    metadata.setdefault("n_frames", int(measurements.shape[0]))
    dataset = MeasurementDataset.from_metadata(measurements, metadata)
    LOGGER.debug("NPZ 数据摘要: %s", dataset.summary())
    return dataset


def load_dataset_from_csv(
    csv_path: Path,
    metadata_path: Path,
    use_cols: Optional[Sequence[int]],
        delimiter: str,
    use_both_components: bool,
) -> MeasurementDataset:
    LOGGER.info("加载 CSV 数据: %s", csv_path)
    raw = np.loadtxt(csv_path, delimiter=delimiter)
    if raw.ndim == 1:
        raw = raw[:, np.newaxis]
    if use_both_components and raw.shape[1] >= 4:
        raw = raw[:, [2, 3]]
    elif use_cols:
        raw = raw[:, list(use_cols)]
    measurements = raw.T

    LOGGER.info("读取元数据: %s", metadata_path)
    with metadata_path.open("r", encoding="utf-8") as fh:
        if metadata_path.suffix.lower() in {".yaml", ".yml"}:
            metadata = yaml.safe_load(fh)
        elif metadata_path.suffix.lower() == ".json":
            metadata = json.load(fh)
        else:
            raise ValueError("metadata 文件必须是 YAML 或 JSON 格式")

    metadata = dict(metadata)
    metadata["n_frames"] = int(measurements.shape[0])
    dataset = MeasurementDataset.from_metadata(measurements, metadata)
    LOGGER.debug("CSV 数据摘要: %s", dataset.summary())
    return dataset


def load_dataset(args: argparse.Namespace) -> MeasurementDataset:
    if args.npz:
        return load_dataset_from_npz(args.npz)
    if args.csv:
        if args.metadata is None:
            raise ValueError("使用 CSV 时必须通过 --metadata 指定元数据文件")
        return load_dataset_from_csv(
            args.csv,
            args.metadata,
            args.use_cols,
            args.delimiter,
            args.use_both_components,
        )
    raise RuntimeError("未提供输入数据")


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def trimmed_mean(arr: np.ndarray, trim_percent: float) -> float:
    trim = max(0.0, min(50.0, float(trim_percent)))
    if trim <= 0:
        return float(np.mean(arr))
    low, high = np.percentile(arr, [trim, 100 - trim])
    mask = (arr >= low) & (arr <= high)
    if not mask.any():
        return float(np.mean(arr))
    return float(np.mean(arr[mask]))


def _apply_physical_scale(
    measured: np.ndarray,
    simulated: np.ndarray,
    residual: np.ndarray,
    metadata: Dict[str, any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[float], float]:
    scale = metadata.get("calibration_scale")
    bias = metadata.get("calibration_bias", 0.0)
    if scale is None or abs(scale) < 1e-18:
        return measured, simulated, residual, None, bias
    # 防止异常放大导致绘图轴爆炸：若尺度过大/非有限则放弃还原物理标尺
    if not np.isfinite(scale) or abs(scale) > 1e5:
        LOGGER.warning("校准尺度过大(%.3e)或非有限，跳过物理标尺绘图", scale)
        return measured, simulated, residual, None, bias

    measured_phys = measured * scale + bias
    simulated_phys = simulated * scale + bias
    residual_phys = simulated_phys - measured_phys
    return measured_phys, simulated_phys, residual_phys, float(scale), float(bias)


def save_outputs(result, output_dir: Path, visualizer) -> None:
    ensure_output_dir(output_dir)
    mesh = result.conductivity_image.fwd_model.mesh

    display_values = result.metadata.get("display_values", result.conductivity)
    rescale_factor = float(result.metadata.get("conductivity_rescale_factor", 1.0))
    if rescale_factor != 0 and abs(rescale_factor - 1.0) > 1e-9:
        display_values = display_values / rescale_factor
    fig_cond = visualizer.plot_conductivity(
        mesh,
        display_values,
        title=None,
        save_path=str(output_dir / "reconstruction.png"),
        minimal=True,
    )
    plt.close(fig_cond)

    measured_plot, simulated_plot, residual_plot, scale, bias = _apply_physical_scale(
        result.measured,
        result.simulated,
        result.residual,
        result.metadata,
    )

    indices = np.arange(len(measured_plot))
    fig, ax = plt.subplots(figsize=visualizer.figsize)
    ax.plot(indices, measured_plot, label="实测边界电压", linewidth=1.5)
    ax.plot(indices, simulated_plot, label="预测边界电压", linewidth=1.5, alpha=0.8)
    ax.set_title(
        f"绝对测量对比 (相对误差={result.relative_error:.2e})"
        + (" [物理标尺]" if scale is not None else "")
    )
    ax.set_xlabel("测量索引")
    ax.set_ylabel("电压 (V)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "measurements_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig_res = visualizer.plot_measurements(
        residual_plot,
        title=("测量残差 [物理标尺]" if scale is not None else "测量残差"),
        save_path=str(output_dir / "measurements_residual.png"),
    )
    plt.close(fig_res)

    if scale is not None:
        np.savetxt(output_dir / "measured_physical_vector.txt", measured_plot)
        np.savetxt(output_dir / "predicted_physical_vector.txt", simulated_plot)
        np.savetxt(output_dir / "residual_physical_vector.txt", residual_plot)

    if result.residual_history:
        np.savetxt(output_dir / "residual_history.txt", np.array(result.residual_history))
    if result.sigma_change_history:
        np.savetxt(output_dir / "sigma_change_history.txt", np.array(result.sigma_change_history))


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    if args.raw_gn:
        # 最简模式：禁用所有缩放/权重/校准/极性对齐/线搜索扩展
        args.measurement_scale = 1.0
        args.auto_measurement_scale = False
        args.calibration_frame = -1
        args.weight_strategy = "none"
        args.max_step = 1.0
        args.line_search_steps = 1
        args.polarity_mode = "none"

    dataset = load_dataset(args)
    if args.measurement_scale != 1.0:
        LOGGER.info("应用 measurement_scale=%.3g 缩放测量数据", args.measurement_scale)
        dataset.measurements = dataset.measurements * float(args.measurement_scale)

    if args.target_frame >= dataset.measurements.shape[0]:
        raise IndexError(f"target_frame 超出范围 (available: 0~{dataset.measurements.shape[0]-1})")

    LOGGER.info("构建 EITSystem 并初始化")
    contact_impedance = np.full(dataset.n_elec, float(args.contact_impedance))

    # 预加载网格以避免重复加载
    mesh = load_or_create_mesh(
        mesh_dir=str(args.mesh_dir),
        n_elec=dataset.n_elec,
        refinement=max(args.refinement, 4),
        radius=args.mesh_radius,
        electrode_coverage=args.electrode_coverage,
    )

    stim_scale_applied = float(args.stim_scale)
    stim_scale_source = "manual"

    if args.auto_stim_scale and abs(args.stim_scale - 1.0) > 1e-9:
        LOGGER.warning("检测到 --stim-scale 已设置，跳过 --auto-stim-scale 以避免冲突。")
        args.auto_stim_scale = False

    # 自动估计激励缩放：用均匀场预测幅值与目标帧实测对齐（只调电流）
    if args.auto_stim_scale:
        temp_system = EITSystem(
            n_elec=dataset.n_elec,
            pattern_config=dataset.pattern_config,
            contact_impedance=contact_impedance,
        )
        temp_system.setup(mesh=mesh)
        baseline_temp = temp_system.forward_solve(temp_system.create_homogeneous_image())
        meas_abs = np.abs(dataset.measurements[args.target_frame])
        pred_abs = np.abs(baseline_temp.meas)
        meas_mean = trimmed_mean(meas_abs, args.auto_trim_percent)
        pred_mean = trimmed_mean(pred_abs, args.auto_trim_percent)
        if pred_mean > 0:
            stim_scale_applied = float(np.clip(meas_mean / pred_mean, args.auto_scale_min, args.auto_scale_max))
            stim_scale_source = "auto"
            LOGGER.info(
                "auto_stim_scale: meas_mean=%.3e, pred_mean=%.3e, stim_scale=%.3g",
                meas_mean,
                pred_mean,
                stim_scale_applied,
            )
        else:
            LOGGER.warning("auto_stim_scale 失败（pred_mean=0），使用 1.0")
            stim_scale_applied = 1.0

    # 可选：放大激励电流以对齐模型/测量量纲
    if stim_scale_applied != 1.0:
        old_amp = float(dataset.pattern_config.amplitude)
        new_amp = old_amp * float(stim_scale_applied)
        LOGGER.info(
            "%s 激励缩放 stim_scale=%.3g: amplitude %.3e -> %.3e",
            "应用" if stim_scale_source == "manual" else "自动估计",
            stim_scale_applied,
            old_amp,
            new_amp,
        )
        dataset.pattern_config = dataclasses.replace(dataset.pattern_config, amplitude=new_amp)
        # 重新生成 stim_matrix 以反映新的激励电流
        pattern_manager = StimMeasPatternManager(dataset.pattern_config)
        dataset.stim_matrix = pattern_manager.stim_matrix.copy()

    eit_system = EITSystem(
        n_elec=dataset.n_elec,
        pattern_config=dataset.pattern_config,
        contact_impedance=contact_impedance,
    )
    eit_system.setup(mesh=mesh)

    flipped_frames: list[int] = []

    if args.mode:
        LOGGER.info("应用预设模式: %s", args.mode)
        apply_mode_preset(eit_system.reconstructor, args.mode)
    else:
        if hasattr(eit_system.reconstructor, "max_iterations"):
            eit_system.reconstructor.max_iterations = args.max_iterations
    if args.max_step is not None and hasattr(eit_system.reconstructor, "max_step"):
        LOGGER.info("设置线搜索最大步长 max_step=%.3g", args.max_step)
        eit_system.reconstructor.max_step = float(args.max_step)
    if args.line_search_steps is not None and hasattr(
        eit_system.reconstructor, "line_search_steps"
    ):
        LOGGER.info("设置线搜索步数 line_search_steps=%d", args.line_search_steps)
        eit_system.reconstructor.line_search_steps = int(args.line_search_steps)
    if args.min_step is not None and hasattr(eit_system.reconstructor, "min_step"):
        LOGGER.info("设置线搜索最小步长 min_step=%.3g", args.min_step)
        eit_system.reconstructor.min_step = float(args.min_step)
    if args.step_schedule:
        LOGGER.info("使用自定义步长序列 (跳过线搜索): %s", args.step_schedule)
        eit_system.reconstructor.step_schedule = list(args.step_schedule)
    if args.regularization_param is not None:
        LOGGER.info("设置正则化参数 λ=%.3e", args.regularization_param)
        eit_system.reconstructor.regularization_param = float(args.regularization_param)
    if args.weight_strategy:
        LOGGER.info("覆盖测量权重策略: %s", args.weight_strategy)
        eit_system.reconstructor.measurement_weight_strategy = args.weight_strategy
        eit_system.reconstructor.use_measurement_weights = args.weight_strategy != "none"
    if hasattr(eit_system.reconstructor, "convergence_tol"):
        eit_system.reconstructor.convergence_tol = float(args.convergence_tol)
    if hasattr(eit_system.reconstructor, "min_iterations"):
        eit_system.reconstructor.min_iterations = int(args.min_iterations)

    baseline_image = eit_system.create_homogeneous_image()
    baseline_data = eit_system.forward_solve(baseline_image)

    flipped_frames = align_measurement_polarity(
        dataset, baseline_data.meas, args.polarity_mode, LOGGER
    )

    # 单一缩放源：优先 fit_scale_baseline，其次 auto_measurement_scale
    if args.fit_scale_baseline and args.measurement_scale == 1.0:
        meas = np.abs(dataset.measurements[args.target_frame])
        pred = np.abs(baseline_data.meas)
        meas_mean = trimmed_mean(meas, args.auto_trim_percent)
        pred_mean = trimmed_mean(pred, args.auto_trim_percent)
        if pred_mean > 0 and meas_mean > 0:
            scale = pred_mean / meas_mean
            scale = float(np.clip(scale, args.auto_scale_min, args.auto_scale_max))
            LOGGER.info("fit_scale_baseline: meas_mean=%.3e, pred_mean=%.3e, scale=%.3g", meas_mean, pred_mean, scale)
            dataset.measurements = dataset.measurements * scale
            dataset.metadata["fit_scale_baseline"] = scale
        else:
            LOGGER.warning("fit_scale_baseline 失败（均值为零），跳过。")
    elif args.auto_measurement_scale:
        if args.measurement_scale != 1.0:
            LOGGER.warning(
                "检测到 measurement_scale 已设置，跳过 auto_measurement_scale 以避免多重缩放"
            )
        else:
            meas = np.abs(dataset.measurements[args.target_frame])
            pred = np.abs(baseline_data.meas)
            meas_mean = trimmed_mean(meas, args.auto_trim_percent)
            pred_mean = trimmed_mean(pred, args.auto_trim_percent)
            if pred_mean > 0 and meas_mean > 0:
                auto_scale = pred_mean / meas_mean
                auto_scale = float(np.clip(auto_scale, args.auto_scale_min, args.auto_scale_max))
                LOGGER.info(
                    "自动测量缩放系数 auto_measurement_scale=%.3g (trim=%.1f%%，应用于所有帧)",
                    auto_scale,
                    args.auto_trim_percent,
                )
                dataset.measurements = dataset.measurements * auto_scale
                dataset.metadata["auto_measurement_scale"] = auto_scale
            else:
                LOGGER.warning("自动测量缩放失败（均值为零），跳过。")

    calibration_metadata: Dict[str, Any] = {}

    if args.calibration_frame >= 0:
        if args.calibration_frame >= dataset.measurements.shape[0]:
            raise IndexError("calibration_frame 超出范围")
        LOGGER.info("使用帧 %d 进行线性校准", args.calibration_frame)
        ref_raw = dataset.measurements[args.calibration_frame].copy()
        scale, bias = compute_scale_bias(ref_raw, baseline_data.meas)
        LOGGER.info("校准参数: scale=%.3e, bias=%.3e", scale, bias)
        if abs(scale) < 1e-12:
            scale = 1.0 if scale >= 0 else -1.0
        dataset.measurements = (dataset.measurements - bias) / scale
        calibration_metadata = {"calibration_scale": scale, "calibration_bias": bias}
    else:
        LOGGER.info("跳过尺度校准（未应用 scale/bias）")

    measurement_data = dataset.to_eit_data(args.target_frame, data_type="real")
    metadata = {"target_idx": args.target_frame, **calibration_metadata}
    if args.mode:
        metadata["mode"] = args.mode
    if args.weight_strategy:
        metadata["weight_strategy"] = args.weight_strategy
    metadata["stim_scale"] = stim_scale_applied
    metadata["stim_scale_source"] = stim_scale_source
    # 导电率物理值需除以激励放大系数
    metadata["conductivity_rescale_factor"] = stim_scale_applied if stim_scale_applied != 0 else 1.0
    metadata["polarity_mode"] = args.polarity_mode
    if flipped_frames:
        metadata["flipped_frames"] = flipped_frames

    result = eit_system.absolute_reconstruct(
        measurement_data=measurement_data,
        baseline_image=baseline_image,
        metadata=metadata,
    )

    # 将导电率除以激励缩放因子，使结果回到采集等效量纲
    if stim_scale_applied != 1.0:
        cond_scale = stim_scale_applied
        if hasattr(result, "conductivity"):
            result.conductivity = np.asarray(result.conductivity) / cond_scale
        if hasattr(result, "conductivity_image") and hasattr(result.conductivity_image, "elem_data"):
            result.conductivity_image.elem_data = result.conductivity_image.elem_data / cond_scale
        result.metadata["conductivity_scaled_by"] = 1.0 / cond_scale
        result.metadata["stim_scale_used"] = stim_scale_applied

    dataset_name = args.csv.stem if args.csv else args.npz.stem
    if args.mode:
        output_dir = args.output_dir / args.mode / dataset_name
    elif args.weight_strategy and args.weight_strategy != "none":
        output_dir = args.output_dir / f"weight_{args.weight_strategy}" / dataset_name
    else:
        output_dir = args.output_dir / dataset_name
    save_outputs(result, output_dir, create_visualizer())

    # Save the command used to run this script
    cmd = " ".join(shlex.quote(arg) for arg in sys.argv)
    with (output_dir / "command.txt").open("w") as f:
        f.write(cmd + "\n")

    LOGGER.info("绝对成像完成，结果输出到: %s", output_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
