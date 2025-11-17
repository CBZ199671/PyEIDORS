#!/usr/bin/env python3
"""基准对比：旧版 DirectJacobian vs 新版 EidorsStyleAdjointJacobian（CPU/torch）。

输出：
- 计算耗时
- 矩阵形状
- 相对误差（对齐符号后）
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pyeidors.data.structures import PatternConfig, EITImage
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.forward.eit_forward_model import EITForwardModel
from pyeidors.inverse.jacobian.direct_jacobian import DirectJacobianCalculator
from pyeidors.inverse.jacobian.adjoint_jacobian import EidorsStyleAdjointJacobian


def benchmark():
    mesh = load_or_create_mesh(mesh_dir="eit_meshes", mesh_name="mesh_102070", n_elec=16)
    pattern_cfg = PatternConfig(
        n_elec=16, stim_pattern="{ad}", meas_pattern="{ad}", amplitude=1.0, use_meas_current=False, rotate_meas=True
    )
    contact_impedance = np.full(16, 1e-6, dtype=float)
    fwd_model = EITForwardModel(n_elec=16, pattern_config=pattern_cfg, z=contact_impedance, mesh=mesh)

    n_elem = len(fwd_model.V_sigma.dofmap().dofs())
    sigma = np.ones(n_elem, dtype=float)
    img = EITImage(elem_data=sigma, fwd_model=fwd_model)

    # DirectJacobian (原实现)
    direct_calc = DirectJacobianCalculator(fwd_model)
    t0 = time.perf_counter()
    J_direct = direct_calc.calculate_from_image(img, method="efficient")
    t_direct = time.perf_counter() - t0

    # EidorsStyle adjoint CPU
    adj_cpu = EidorsStyleAdjointJacobian(fwd_model, use_torch=False)
    t0 = time.perf_counter()
    J_adj_cpu = adj_cpu.calculate_from_image(img)
    t_adj_cpu = time.perf_counter() - t0

    # EidorsStyle adjoint torch（若 GPU 可用则在 GPU）
    adj_torch = EidorsStyleAdjointJacobian(fwd_model, use_torch=True)
    t0 = time.perf_counter()
    J_adj_torch = adj_torch.calculate_from_image(img)
    t_adj_torch = time.perf_counter() - t0

    # 对齐符号：DirectJacobian 默认正号，EIDORS 风格带负号
    J_direct_aligned = -J_direct

    def rel_err(A, B):
        denom = np.linalg.norm(B.ravel()) + 1e-15
        return float(np.linalg.norm((A - B).ravel()) / denom)

    err_direct_vs_cpu = rel_err(J_direct_aligned, J_adj_cpu)
    err_cpu_vs_torch = rel_err(J_adj_cpu, J_adj_torch)

    print("形状: ", J_direct.shape)
    print(f"DirectJacobian 耗时: {t_direct:.4f}s")
    print(f"EidorsStyle adjoint CPU 耗时: {t_adj_cpu:.4f}s")
    print(f"EidorsStyle adjoint Torch 耗时: {t_adj_torch:.4f}s")
    print(f"Direct(取负) vs EidorsStyle CPU 相对误差: {err_direct_vs_cpu:.3e}")
    print(f"EidorsStyle CPU vs Torch 相对误差: {err_cpu_vs_torch:.3e}")


if __name__ == "__main__":
    benchmark()
