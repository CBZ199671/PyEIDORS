#!/usr/bin/env python3
"""按 EIDORS 参数风格执行一次 GN 绝对成像。

要点：
- 只用 CSV 第三列（零基索引 2）作为目标帧实部。
- 激励/测量模式与幅值来自配套 YAML 元数据。
- 网格：16 电极，圆柱半径 0.03 m，z_contact=1e-5，默认细化 12。
- 正则：NOSER，λ=0.02；GN 最大 15 次迭代，带回溯线搜索。
- 初始导电率：0.001 S/m。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from fenics import Function
import numpy as np

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("运行该脚本需要 PyYAML，请先安装: pip install pyyaml") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC_PATH))

from pyeidors.core_system import EITSystem  # noqa: E402
from pyeidors.data.measurement_dataset import MeasurementDataset  # noqa: E402
from pyeidors.data.structures import PatternConfig  # noqa: E402
from pyeidors.data.structures import EITData, EITImage  # noqa: E402
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh  # noqa: E402
from pyeidors.visualization import EITVisualizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于 PyEidors 的 GN 绝对成像（EIDORS 风格参数）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "data" / "measurements" / "tank" / "2025-11-14-22-18-02_1_10.00_50uA_3000Hz.csv",
        help="包含 4 列 (vh_real, vh_imag, vi_real, vi_imag) 的测量 CSV",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=REPO_ROOT / "data" / "measurements" / "tank" / "2025-11-14-22-18-02_1_10.00_50uA_3000Hz.yaml",
        help="与 CSV 对应的 YAML 元数据",
    )
    parser.add_argument(
        "--use-col",
        type=int,
        default=2,
        help="选择哪一列作为绝对重建输入（零基），默认第三列即 vi 实部",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "gn_absolute_eidors_style",
        help="结果输出目录",
    )
    parser.add_argument(
        "--mesh-radius",
        type=float,
        default=0.03,
        help="圆柱半径 (m)",
    )
    parser.add_argument(
        "--refinement",
        type=int,
        default=12,
        help="网格细化级别，传递给 load_or_create_mesh",
    )
    parser.add_argument(
        "--measurement-gain",
        type=float,
        default=10.0,
        help="测量放大器增益，CSV 中的电压会除以此值",
    )
    parser.add_argument(
        "--background-sigma",
        type=float,
        default=0.001,
        help="初始/背景导电率 (S/m)",
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=0.02,
        help="正则化参数 λ",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=15,
        help="GN 最大迭代次数",
    )
    parser.add_argument(
        "--contact-impedance",
        type=float,
        default=1e-5,
        help="接触阻抗 (Ω·m²)",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(fh)
        if path.suffix.lower() == ".json":
            return json.load(fh)
        raise ValueError("metadata 仅支持 YAML/JSON")


def load_measurement_vector(csv_path: Path, col_idx: int, measurement_gain: float = 1.0) -> np.ndarray:
    raw = np.loadtxt(csv_path, delimiter=",")
    if raw.ndim != 2 or raw.shape[1] <= col_idx:
        raise ValueError(f"CSV 形状 {raw.shape} 无法提取列 {col_idx}")
    # 形状: (n_meas, ) -> (1, n_meas) 以便 MeasurementDataset 接收
    frame = raw[:, col_idx].astype(float)
    # 除以测量增益得到实际电压
    if measurement_gain != 1.0:
        frame = frame / measurement_gain
    return frame.reshape(1, -1)


def build_dataset(measurements: np.ndarray, metadata: dict) -> MeasurementDataset:
    # 确保元数据字段齐全
    required = ["n_elec", "stim_pattern", "meas_pattern"]
    for key in required:
        if key not in metadata:
            raise KeyError(f"metadata 缺少必需字段: {key}")
    meta = dict(metadata)
    meta.setdefault("n_frames", int(measurements.shape[0]))
    return MeasurementDataset.from_metadata(measurements, meta)


def configure_reconstructor(system: EITSystem, lambda_: float = 0.02, max_iter: int = 15, background_sigma: float = 0.001) -> None:
    """把 GN 参数调成接近 EIDORS 样式。"""
    recon = system.reconstructor
    recon.max_iterations = max_iter
    recon.regularization_param = lambda_  # λ
    recon.line_search_steps = 12
    # 步长限制
    recon.max_step = 1.0
    recon.min_step = 1e-6  # 允许非常小的步长
    recon.convergence_tol = 1e-5
    recon.negate_jacobian = True
    recon.use_measurement_weights = True
    recon.measurement_weight_strategy = "scaled_baseline"
    # 避免导电率跌落过低导致电压爆掉
    recon.clip_values = (background_sigma * 0.1, background_sigma * 100)
    recon.min_iterations = 1
    # EIDORS 风格：使用先验误差项
    recon.use_prior_term = True


def run_reconstruction(
    csv_path: Path,
    metadata_path: Path,
    col_idx: int,
    output_dir: Path,
    mesh_radius: float,
    refinement: int,
    measurement_gain: float = 10.0,
    background_sigma: float = 0.001,
    lambda_: float = 0.02,
    max_iter: int = 15,
    contact_impedance: float = 1e-5,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] CSV 数据文件: {csv_path}")
    print(f"[INFO] YAML 元数据文件: {metadata_path}")
    metadata = load_metadata(metadata_path)
    measurements = load_measurement_vector(csv_path, col_idx, measurement_gain=measurement_gain)
    dataset = build_dataset(measurements, metadata)
    print(f"[INFO] Background sigma: {background_sigma}, lambda: {lambda_}, measurement_gain: {measurement_gain}")
    print(f"[INFO] Measurement range: [{measurements.min():.6e}, {measurements.max():.6e}]")

    # 用元数据构造 PatternConfig，保持激励/测量与采集一致
    pattern_config = dataset.pattern_config
    n_elec = pattern_config.n_elec

    # 生成/加载网格
    mesh = load_or_create_mesh(
        mesh_dir=str(REPO_ROOT / "eit_meshes"),
        mesh_name=None,
        n_elec=n_elec,
        radius=mesh_radius,
        refinement=refinement,
        electrode_coverage=float(metadata.get("electrode_coverage", 0.5)),
    )

    # EIT 系统配置：接触阻抗，NOSER 正则
    z_contact = np.ones(n_elec) * contact_impedance
    system = EITSystem(
        n_elec=n_elec,
        pattern_config=pattern_config,
        contact_impedance=z_contact,
        base_conductivity=background_sigma,
        regularization_type="noser",
        regularization_alpha=1.0,
        noser_exponent=0.5,  # EIDORS 风格
    )
    system.setup(mesh=mesh)
    configure_reconstructor(system, lambda_=lambda_, max_iter=max_iter, background_sigma=background_sigma)

    # 准备测量数据和初值
    eit_data: EITData = dataset.to_eit_data(frame_index=0, data_type="real")
    
    # 基线前向对比（仅用于信息输出，不做缩放）
    base_img = system.create_homogeneous_image(conductivity=background_sigma)
    base_forward, _ = system.fwd_model.fwd_solve(base_img)
    print(f"[INFO] Measured voltage range: [{eit_data.meas.min():.6e}, {eit_data.meas.max():.6e}]")
    print(f"[INFO] Model prediction range: [{base_forward.meas.min():.6e}, {base_forward.meas.max():.6e}]")
    scale_ratio = eit_data.meas.max() / (base_forward.meas.max() + 1e-12)
    print(f"[INFO] Meas/Model ratio: {scale_ratio:.2f} (如果远离1，请调整背景电导率或激励电流)")
    
    n_elements = len(Function(system.fwd_model.V_sigma).vector()[:])
    initial_sigma = np.full(n_elements, background_sigma, dtype=float)

    recon_result = system.reconstructor.reconstruct(
        measured_data=eit_data,
        initial_conductivity=initial_sigma,
        jacobian_method="efficient",
    )
    conductivity_fn = recon_result["conductivity"]
    conductivity_vec = conductivity_fn.vector()[:]

    # 前向预测用于曲线对比
    sim_data, _ = system.fwd_model.fwd_solve(EITImage(elem_data=conductivity_vec, fwd_model=system.fwd_model))
    measured_vec = eit_data.meas
    predicted_vec = sim_data.meas

    # 可视化
    visualizer = EITVisualizer(style="seaborn", figsize=(10, 8))
    fig_cond = visualizer.plot_conductivity(
        mesh,
        conductivity_fn,
        title="GN 绝对成像导电率分布",
        colormap="viridis",
    )
    fig_cond.savefig(output_dir / "conductivity.png", dpi=300, bbox_inches="tight")

    # 计算相关系数
    corr = np.corrcoef(measured_vec, predicted_vec)[0, 1]
    
    fig_cmp, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：曲线对比
    ax = axes[0]
    ax.plot(measured_vec, "b.-", label="实测", markersize=3)
    ax.plot(predicted_vec, "r--", label="预测")
    ax.set_xlabel("测量点索引")
    ax.set_ylabel("电压 (V)")
    ax.set_title("边界电压对比")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 右图：散点图 + 相关系数
    ax2 = axes[1]
    ax2.scatter(measured_vec, predicted_vec, s=15, alpha=0.7, c='steelblue')
    vmin = min(np.min(measured_vec), np.min(predicted_vec))
    vmax = max(np.max(measured_vec), np.max(predicted_vec))
    ax2.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1.5, label='y=x')
    ax2.set_xlabel("实测电压 (V)")
    ax2.set_ylabel("预测电压 (V)")
    ax2.set_title(f"散点拟合 (相关系数 r = {corr:.4f})")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_aspect('equal', adjustable='box')
    
    fig_cmp.tight_layout()
    fig_cmp.savefig(output_dir / "prediction_vs_measurement.png", dpi=300, bbox_inches="tight")

    # 保存关键数值
    np.savez(
        output_dir / "result_arrays.npz",
        conductivity=conductivity_vec,
        measured=measured_vec,
        predicted=predicted_vec,
        residual=np.asarray(predicted_vec) - np.asarray(measured_vec),
    )

    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "csv": str(csv_path),
                "metadata": str(metadata_path),
                "use_col": col_idx,
                "n_elec": n_elec,
                "mesh_radius": mesh_radius,
                "refinement": refinement,
                "regularization": "NOSER",
                "lambda": lambda_,
                "max_iterations": max_iter,
                "initial_sigma": background_sigma,
                "measurement_gain": measurement_gain,
                "contact_impedance": contact_impedance,
                "residual_history": recon_result.get("residual_history", []),
                "sigma_change_history": recon_result.get("sigma_change_history", []),
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] 完成 GN 绝对成像，结果已写入: {output_dir}")
    print(f"导电率图: {output_dir/'conductivity.png'}")
    print(f"预测 vs 实测: {output_dir/'prediction_vs_measurement.png'}")


def main() -> None:
    args = parse_args()
    run_reconstruction(
        csv_path=args.csv,
        metadata_path=args.metadata,
        col_idx=args.use_col,
        output_dir=args.output_dir,
        mesh_radius=args.mesh_radius,
        refinement=args.refinement,
        measurement_gain=args.measurement_gain,
        background_sigma=args.background_sigma,
        lambda_=args.lambda_,
        max_iter=args.max_iter,
        contact_impedance=args.contact_impedance,
    )


if __name__ == "__main__":
    main()
