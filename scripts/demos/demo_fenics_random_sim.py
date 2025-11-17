#!/usr/bin/env python3
"""最小化示范：使用 FEniCS 网格随机放置目标，完成一次正/逆 EIT 仿真。

流程：
1) 载入缓存的 16 电极圆域网格（eit_meshes/mesh_102070*），避免依赖 gmsh。
2) 构造相邻驱动/测量模式，接触阻抗与 MATLAB 示例一致（1e-6）。
3) 生成随机圆形异物（位置、半径、对比度随机），前向求得基线/目标电压。
4) 使用模块化 Gauss-Newton 重建（NOSER 正则，默认设置）得到导电率估计。
5) 计算与真值/测量的简单指标，并把结果写入 results/demo_random_fenics/*.npz。

运行：
    python scripts/demo_fenics_random_sim.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pyeidors.core_system import EITSystem
from pyeidors.data.structures import PatternConfig, EITImage
from pyeidors.data.synthetic_data import create_custom_phantom
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.visualization import create_visualizer

def cell_to_node(mesh, cell_values: np.ndarray) -> np.ndarray:
    """把单元值平均插值到节点，方便绘图。"""
    node_values = np.zeros(mesh.num_vertices())
    node_counts = np.zeros(mesh.num_vertices())
    for cell_idx, cell in enumerate(mesh.cells()):
        for v_idx in cell:
            node_values[v_idx] += cell_values[cell_idx]
            node_counts[v_idx] += 1
    node_counts[node_counts == 0] = 1
    node_values /= node_counts
    return node_values


def make_random_anomaly(rng: np.random.Generator) -> Dict:
    """随机生成一个圆形异物描述。"""
    # 半径取 [0.08, 0.18]，中心距原点不超过 0.5
    radius = float(rng.uniform(0.08, 0.18))
    angle = float(rng.uniform(0, 2 * math.pi))
    dist = float(rng.uniform(0.0, 0.5))
    center = (dist * math.cos(angle), dist * math.sin(angle))
    # 对比度：强化或减弱导电率
    contrast = float(rng.uniform(1.5, 3.0))
    if rng.random() < 0.35:  # 35% 概率改为电阻性（低导）
        contrast = float(rng.uniform(0.2, 0.8))
    return {"center": center, "radius": radius, "conductivity": contrast}


def main() -> None:
    rng = np.random.default_rng(20241116)

    # 1) 载入缓存网格（16 电极），避免 gmsh 依赖
    mesh = load_or_create_mesh(mesh_dir="eit_meshes", mesh_name="mesh_102070", n_elec=16)

    # 2) 构建系统（相邻驱动/测量，幅值 1；接触阻抗 1e-6）
    pattern_cfg = PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        amplitude=1.0,
        use_meas_current=False,
        rotate_meas=True,
    )
    contact_impedance = np.full(16, 1e-6, dtype=float)

    system = EITSystem(
        n_elec=16,
        pattern_config=pattern_cfg,
        contact_impedance=contact_impedance,
        base_conductivity=1.0,
        regularization_type="noser",
        regularization_alpha=1.0,
    )
    system.setup(mesh=mesh)

    n_elem = len(system.fwd_model.V_sigma.dofmap().dofs())

    # 3) 构造基线和随机异物
    sigma_bg = np.ones(n_elem, dtype=float)
    anomaly = make_random_anomaly(rng)
    sigma_true_fn = create_custom_phantom(
        system.fwd_model,
        background_conductivity=1.0,
        anomalies=[anomaly],
    )
    sigma_true = sigma_true_fn.vector()[:]

    img_bg = EITImage(elem_data=sigma_bg, fwd_model=system.fwd_model)
    img_true = EITImage(elem_data=sigma_true, fwd_model=system.fwd_model)

    # 4) 前向求解：基线与目标
    data_bg, _ = system.fwd_model.fwd_solve(img_bg)
    data_true, _ = system.fwd_model.fwd_solve(img_true)
    diff_meas = data_true.meas - data_bg.meas

    # 5) 单步 Gauss-Newton 重建（绝对重建，初始导电率=1）
    recon = system.reconstructor.reconstruct(
        data_true, initial_conductivity=1.0, jacobian_method="efficient"
    )
    sigma_est = recon["conductivity"].vector()[:]
    img_est = EITImage(elem_data=sigma_est, fwd_model=system.fwd_model)
    data_est, _ = system.fwd_model.fwd_solve(img_est)

    # 6) 误差指标
    meas_rmse = float(np.sqrt(np.mean((data_est.meas - data_true.meas) ** 2)))
    sigma_rmse = float(np.sqrt(np.mean((sigma_est - sigma_true) ** 2)))
    sigma_mae = float(np.mean(np.abs(sigma_est - sigma_true)))
    metrics: Dict[str, float] = {
        "meas_rmse": meas_rmse,
        "sigma_rmse": sigma_rmse,
        "sigma_mae": sigma_mae,
        "residual_final": float(recon["final_residual"]),
    }

    # 7) 保存结果
    out_dir = Path("results/demo_random_fenics")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "simulation_outputs.npz",
        sigma_bg=sigma_bg,
        sigma_true=sigma_true,
        sigma_est=sigma_est,
        anomaly_center=np.array(anomaly["center"]),
        anomaly_radius=anomaly["radius"],
        anomaly_conductivity=anomaly["conductivity"],
        meas_bg=data_bg.meas,
        meas_true=data_true.meas,
        meas_est=data_est.meas,
        diff_meas=diff_meas,
        metrics=np.array(list(metrics.values())),
        metric_names=np.array(list(metrics.keys())),
    )

    # 8) 可视化：真值 vs 重建导电率，测量对比
    viz = create_visualizer()
    sigma_true_nodes = cell_to_node(mesh, sigma_true) if len(sigma_true) == mesh.num_cells() else sigma_true
    sigma_est_nodes = cell_to_node(mesh, sigma_est) if len(sigma_est) == mesh.num_cells() else sigma_est
    fig_cmp = viz.plot_reconstruction_comparison(
        mesh,
        sigma_true_nodes,
        sigma_est_nodes,
        title="真值 vs 重建导电率"
    )
    fig_cmp.savefig(out_dir / "conductivity_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig_cmp)

    # 边界电压对比：真实目标 vs 重建预测（同测量空间）
    fig_v = plt.figure(figsize=(10, 4))
    ax1 = fig_v.add_subplot(1, 2, 1)
    ax1.scatter(data_true.meas, data_est.meas, s=14, alpha=0.7, label="预测 vs 真值")
    vmin = min(data_true.meas.min(), data_est.meas.min())
    vmax = max(data_true.meas.max(), data_est.meas.max())
    ax1.plot([vmin, vmax], [vmin, vmax], "r--", label="y = x")
    ax1.set_title("边界电压散点")
    ax1.set_xlabel("真值")
    ax1.set_ylabel("预测")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = fig_v.add_subplot(1, 2, 2)
    idx = np.arange(len(data_true.meas))
    ax2.plot(idx, data_true.meas, "b-", lw=1.2, label="真值")
    ax2.plot(idx, data_est.meas, "r--", lw=1.2, label="预测")
    ax2.set_title("边界电压序列")
    ax2.set_xlabel("测量索引")
    ax2.set_ylabel("电压")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig_v.suptitle("目标/预测边界电压对比", fontsize=13, fontweight="bold")
    fig_v.tight_layout()
    fig_v.savefig(out_dir / "voltage_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig_v)

    print("随机异物:", anomaly)
    print("指标:", metrics)
    print(f"结果已保存到 {out_dir / 'simulation_outputs.npz'}")
    print(f"导电率对比图: {out_dir / 'conductivity_comparison.png'}")
    print(f"电压对比图: {out_dir / 'voltage_comparison.png'}")


if __name__ == "__main__":
    main()
