#!/usr/bin/env python3
"""运行一次差分成像重建并保存可视化结果。"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

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
from pyeidors.visualization import create_visualizer

LOGGER = logging.getLogger("difference_reconstruction")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行差分成像重建",
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
        default=[0, 2],
        help="从 CSV 中选取哪些列作为帧（零基索引），默认使用基准实部 和 当前帧实部",
    )
    parser.add_argument("--delimiter", default=",", help="CSV 分隔符")
    parser.add_argument("--reference-frame", type=int, default=0, help="参考帧索引")
    parser.add_argument("--target-frame", type=int, default=1, help="目标帧索引")
    parser.add_argument(
        "--calibration-frame",
        type=int,
        default=0,
        help="用于尺度校准的帧索引（-1 表示跳过校准）",
    )
    parser.add_argument(
        "--target-amplitude",
        type=float,
        help="当设置时，先将全部帧缩放到该激励幅值，再进入 GN 迭代；结果会在保存前缩放回原单位",
    )
    parser.add_argument(
        "--pattern-amplitude",
        type=float,
        help="覆盖前向模型使用的激励幅值 (A)，未指定时沿用元数据",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "difference_reconstruction",
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
        help="缺少网格时使用的生成尺寸",
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
        help="网格半径 (传给 load_or_create_mesh，用于缓存识别)",
    )
    parser.add_argument(
        "--electrode-coverage",
        type=float,
        default=0.5,
        help="电极覆盖率 (传给 load_or_create_mesh)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="高斯牛顿最大迭代次数",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
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
) -> MeasurementDataset:
    LOGGER.info("加载 CSV 数据: %s", csv_path)
    raw = np.loadtxt(csv_path, delimiter=delimiter)
    if raw.ndim == 1:
        raw = raw[:, np.newaxis]
    if use_cols:
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
        return load_dataset_from_csv(args.csv, args.metadata, args.use_cols, args.delimiter)
    raise RuntimeError("未提供输入数据")


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_outputs(result, output_dir: Path, visualizer) -> None:
    ensure_output_dir(output_dir)
    mesh = result.conductivity_image.fwd_model.mesh

    display_values = result.metadata.get("display_values", result.conductivity)
    fig_cond = visualizer.plot_conductivity(
        mesh,
        display_values,
        title=None,
        save_path=str(output_dir / "reconstruction.png"),
        minimal=True,
    )
    plt.close(fig_cond)

    indices = np.arange(len(result.measured))
    fig, ax = plt.subplots(figsize=visualizer.figsize)
    ax.plot(indices, result.measured, label="差分实测", linewidth=1.5)
    ax.plot(indices, result.simulated, label="差分预测", linewidth=1.5, alpha=0.8)
    ax.set_title(f"差分测量对比 (相对误差={result.relative_error:.2e})")
    ax.set_xlabel("测量索引")
    ax.set_ylabel("电压差 (V)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "measurements_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig_res = visualizer.plot_measurements(
        result.residual,
        title="差分残差",
        save_path=str(output_dir / "measurements_residual.png"),
    )
    plt.close(fig_res)

    if result.residual_history:
        np.savetxt(output_dir / "residual_history.txt", np.array(result.residual_history))
    if result.sigma_change_history:
        np.savetxt(output_dir / "sigma_change_history.txt", np.array(result.sigma_change_history))


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    dataset = load_dataset(args)
    if not isinstance(dataset.metadata, dict):
        dataset.metadata = dict(dataset.metadata)

    source_amplitude = float(dataset.metadata.get("amplitude", getattr(dataset.pattern_config, "amplitude", 1.0)))
    pattern_override = args.pattern_amplitude
    if pattern_override is not None:
        dataset.pattern_config.amplitude = float(pattern_override)
    elif args.target_amplitude is not None:
        dataset.pattern_config.amplitude = float(args.target_amplitude)

    measurement_scale = 1.0
    target_amplitude = args.target_amplitude
    if target_amplitude is not None:
        if target_amplitude <= 0:
            raise ValueError("target_amplitude 必须为正数")
        if source_amplitude <= 0:
            raise ValueError("metadata 中的 amplitude 必须为正数才能进行缩放")
        measurement_scale = float(target_amplitude) / float(source_amplitude)
        LOGGER.info(
            "将在进入 GN 之前按幅值缩放测量: 源=%.3e A, 目标=%.3e A, 缩放系数=%.3e",
            source_amplitude,
            target_amplitude,
            measurement_scale,
        )

    if args.reference_frame >= dataset.measurements.shape[0] or args.reference_frame < 0:
        raise IndexError("参考帧索引超出范围")
    if args.target_frame >= dataset.measurements.shape[0] or args.target_frame < 0:
        raise IndexError("目标帧索引超出范围")
    if args.reference_frame == args.target_frame:
        raise ValueError("参考帧和目标帧必须不同")

    LOGGER.info("构建 EITSystem 并初始化")
    eit_system = EITSystem(n_elec=dataset.n_elec, pattern_config=dataset.pattern_config)
    mesh = load_or_create_mesh(
        mesh_dir=str(args.mesh_dir),
        n_elec=dataset.n_elec,
        refinement=max(args.refinement, 4),
        radius=args.mesh_radius,
        electrode_coverage=args.electrode_coverage,
    )
    eit_system.setup(mesh=mesh)

    if hasattr(eit_system.reconstructor, "max_iterations"):
        eit_system.reconstructor.max_iterations = args.max_iterations

    baseline_image = eit_system.create_homogeneous_image()
    baseline_data = eit_system.forward_solve(baseline_image)

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
    else:
        LOGGER.info("跳过尺度校准")

    if not np.isclose(measurement_scale, 1.0):
        dataset.measurements = dataset.measurements * measurement_scale
        dataset.metadata["amplitude_scaled_from"] = source_amplitude
        dataset.metadata["amplitude_scaled_to"] = target_amplitude
        dataset.metadata["measurement_scale_factor"] = measurement_scale

    reference_data = dataset.to_eit_data(args.reference_frame, data_type="reference")
    target_data = dataset.to_eit_data(args.target_frame, data_type="measurement")

    metadata = {"target_idx": args.target_frame, "reference_idx": args.reference_frame}
    if target_amplitude is not None:
        metadata["measurement_scaling"] = {
            "scale": measurement_scale,
            "source_amplitude": source_amplitude,
            "target_amplitude": target_amplitude,
        }

    result = eit_system.difference_reconstruct(
        measurement_data=target_data,
        reference_data=reference_data,
        metadata=metadata,
    )

    if not np.isclose(measurement_scale, 1.0):
        inv_scale = 1.0 / measurement_scale
        result.measured = result.measured * inv_scale
        result.simulated = result.simulated * inv_scale
        result.residual = result.residual * inv_scale
        if result.residual_history is not None:
            result.residual_history = [float(val * inv_scale) for val in result.residual_history]
        scaling_meta = result.metadata.setdefault("measurement_scaling", {})
        scaling_meta.update(
            {
                "scale": measurement_scale,
                "source_amplitude": source_amplitude,
                "target_amplitude": target_amplitude,
            }
        )

    output_dir = args.output_dir / (args.csv.stem if args.csv else args.npz.stem)
    ensure_output_dir(output_dir)
    np.savetxt(output_dir / "measured_difference.csv", result.measured, delimiter=",")
    np.savetxt(output_dir / "simulated_difference.csv", result.simulated, delimiter=",")
    np.savetxt(output_dir / "residual_difference.csv", result.residual, delimiter=",")
    save_outputs(result, output_dir, create_visualizer())
    LOGGER.info("差分成像完成，结果输出到: %s", output_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
