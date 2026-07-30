#!/usr/bin/env python3
"""生成面向教授的双语演示 Notebook。 / Build the bilingual walkthrough notebooks."""

from __future__ import annotations

import json
from pathlib import Path
import uuid


PACKAGE_DIR = Path(__file__).resolve().parent


def _markdown(text: str):
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": (text.strip() + "\n").splitlines(keepends=True),
    }


def _code(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": (text.strip() + "\n").splitlines(keepends=True),
    }


def _new_notebook() -> dict:
    return {
        "cells": [],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _shared_symbol_dictionary_markdown(framework: str):
    return _markdown(
        rf"""
## 符号与变量字典 / Symbol and variable dictionary

| 变量或符号 | 中文含义 | English meaning | 类型或维度 |
|---|---|---|---|
| `CASE_ID` | 当前正问题案例编号；默认 `X01` 是均匀背景电导率案例。 | Selected forward-problem case; default `X01` has uniform background conductivity. | `str` |
| `fixture` / `metadata_path` | 三个框架共用的规范输入：节点、三角形、电极边、逐单元电导率、接触阻抗和电流模式。 | Canonical input shared by all three frameworks: nodes, triangles, electrode edges, cell conductivity, contact impedance, and drives. | 路径和元数据 / paths and metadata |
| `{framework}_report` | {framework} 正式运行器保存的求解器、离散化、计时和原始电压报告。 | Solver, discretization, timing, and raw-voltage report saved by the official {framework} runner. | `dict` |
| $N$ | 有限元体节点数。 | Number of FEM body nodes. | `int` |
| $K$ | 三角形单元数。 | Number of triangular cells. | `int` |
| $L$ | 电极数。 | Number of electrodes. | `int` |
| $P$ | 同时求解的电流模式（右端项）数。 | Number of current patterns/right-hand sides. | `int` |
| `nodes` | 每个体节点的 $(x,y)$ 坐标。 | $(x,y)$ coordinate of every body node. | $N\times2$ |
| `cells` | 每个 P1 三角形的三个节点索引。 | Three node indices of every P1 triangle. | $K\times3$ |
| `tagged_edges` | 边界边的两个节点及其电极标签；标签 0 表示绝缘边。 | Two vertices and electrode label for each boundary edge; label 0 is insulating. | $E_b\times3$ |
| `cell_conductivity`, $\sigma_k$ | 单元电导率：第 $k$ 个三角形内的常数值；X01 全部为 $1/8$。 | Cell conductivity: the constant value in cell $k$; every X01 cell is $1/8$. | $K$ 个标量 / $K$ scalars |
| `contact_impedance`, $z_\ell$ | 第 $\ell$ 个电极的接触阻抗；X01 为 1。 | Contact impedance of electrode $\ell$; X01 uses 1. | $L$ 个标量 / $L$ scalars |
| `blocks.currents`, $I$ | 边界注入电流；每一列严格满足 $\mathbf{{1}}^\mathsf{{T}}I_{{:,p}}=0$。 | Injected current on the boundary; every column satisfies $\mathbf{{1}}^\mathsf{{T}}I_{{:,p}}=0$. | $L\times P$ |
| `blocks.robin_matrix`, $A_R$ | 体刚度矩阵加电极 Robin 边界质量项。 | Body stiffness matrix plus the electrode Robin boundary mass. | N×N 稀疏矩阵 |
| `blocks.coupling`, $C$ | 体节点自由度与常数电极电势之间的耦合。 | Coupling between body nodal DOFs and constant electrode voltages. | N×L 稀疏矩阵 |
| `blocks.electrode_matrix`, $D$ | 电极边界积分形成的电极块。 | Electrode block from boundary integrals. | L×L 稀疏矩阵 |
| $u$ | 每个电流模式对应的体内节点电势。 | Body nodal potential for each drive. | $N$×$P$ |
| $U$ | 每个电流模式对应的边界电极电压，也是本实验比较的主要输出。 | Boundary electrode voltage for each drive; the principal compared output. | $L$×$P$ |
| $\lambda$ | 传统 CEM 中实施 $\mathbf{{1}}^\mathsf{{T}}U=0$ 的拉格朗日乘子。 | Gauge multiplier enforcing $\mathbf{{1}}^\mathsf{{T}}U=0$ in Classic CEM. | $1$×$P$ |
| $Q$ | `float64` 求解中电极零和子空间的正交基。 | Orthonormal basis of the zero-sum electrode subspace in the `float64` solve. | $L$×$(L-1)$ |
| $y$ | 电极电压在零和基中的坐标，满足 $U=Qy$。 | Coordinates of electrode voltage in the zero-sum basis, with $U=Qy$. | $(L-1)$×$P$ |
| `response_basis`, $R$ | 解 $A_RR=CQ$ 得到的体响应基；代码用分解求解，不显式形成逆矩阵。 | Body response basis from $A_RR=CQ$; solved by a factorization without forming an inverse. | $N$×$(L-1)$ |
| `schur_action_basis` | $DQ-C^\mathsf{{T}}R$，即完整跨导算子作用在 $Q$ 上的结果。 | $DQ-C^\mathsf{{T}}R$, the full transconductance action on $Q$. | $L$×$(L-1)$ |
| `reduced_map`, $T_r$ | $Q^\mathsf{{T}}(DQ-C^\mathsf{{T}}R)$，Robin CEM 实际分解的小矩阵。 | $Q^\mathsf{{T}}(DQ-C^\mathsf{{T}}R)$, the small matrix actually factored by Robin CEM. | $(L-1)$×$(L-1)$ |
| `nnz` | 稀疏矩阵中非零元素数量，不是误差。 | Number of stored nonzeros in a sparse matrix; it is not an error metric. | `int` |
| `mesh_fingerprint` | 网格指纹：对规范节点、单元和带标签边界边计算的 SHA-256；三个报告必须一致。 | Mesh fingerprint: SHA-256 of canonical nodes, cells, and tagged boundary edges; all three reports must match it. | 64 位十六进制 / hex chars |

### 38 个案例究竟计算什么 / What the 38 cases compute

每个案例都是 **CEM 正问题**，不是逆问题重构：给定同一案例的网格、
逐单元电导率、电极、接触阻抗和零和电流模式，分别用传统 CEM 与 Robin CEM
计算体内节点电势 $u$ 和边界电极电压 $U$。每个框架必须先画出自己实际加载的
网格、$\sigma_k$ 和选定的边界注流，并通过相同网格指纹认证，之后结果才进入
跨框架比较。

Every case is a **CEM forward problem**, not an inverse reconstruction:
given the case's mesh, cell conductivity, electrodes, contact impedance, and
zero-sum drives, Classic and Robin CEM compute body potential $u$ and boundary
electrode voltage $U$. Each framework must first display the mesh,
$\sigma_k$, and a selected boundary drive that it actually loaded, and must
certify the same mesh fingerprint before its result enters the cross-framework
comparison.

| 案例 | 正问题设置 / Forward setting | 主要用途 / Purpose |
|---|---|---|
| X01–X16 | Q0/Q2 网格；16 电极；均匀有理 $\sigma$；多种有理 $z$；相邻和 skip-4 注流。 | 改变网格、物性范围、接触阻抗和注流跨度。 / Vary mesh, physical range, impedance, and drive span. |
| X17–X24 | Q0/Q2 网格；16 电极；左右两区 $\sigma=1/4$ 与 $1$；两种 $z$ 和注流。 | 表示已知内部非均匀待测物后的正向边界电压。 / Forward voltages for a known internal heterogeneity. |
| X25–X32 | Q0/Q2 网格；8 电极；均匀 $\sigma=1/4$；两种 $z$ 和注流。 | 检查电极数改变。 / Check a different electrode count. |
| X33–X38 | 更细 Q4 网格；16 电极；均匀 $\sigma=1/4$；三种 $z$ 和两种注流。 | 检查更大有理离散系统。 / Check the larger rational discretization. |
"""
    )


def build_pyeidors_notebook():
    notebook = _new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "PyEIDORS real float64 (Nix)",
            "language": "python",
            "name": "pyeidors-real-float64-nix",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    notebook["cells"] = [
        _markdown(
            r"""
# PyEIDORS/DOLFINx：经典 CEM 与 Robin CEM / Classic CEM versus Robin CEM

## 目标 / Goal

| 中文 | English |
|---|---|
| 从头到尾运行一个有理数精确案例，检查 DOLFINx 组装的 $A_R,C,D$ 块，分别求解传统增广 CEM 与约化 Robin/跨导 CEM，再与经过认证的有理数域 $\mathbb{Q}$ 精确电极电压比较，最后复现 38 个案例的报告汇总。 | Run one exact-rational case from top to bottom, inspect the DOLFINx-assembled $A_R,C,D$ blocks, solve the traditional augmented CEM and the reduced Robin/transconductance CEM, compare both with the certified exact voltage over $\mathbb{Q}$, and reproduce the 38-case report summary. |

| 默认案例 | Default case |
|---|---|
| `X01` 足够小，适合交互式调试和逐个查看矩阵。 | `X01` is intentionally small enough for interactive debugging and matrix inspection. |
"""
        ),
        _markdown(
            r"""
## 设置 / Setup

| 中文 | English |
|---|---|
| 必须从仓库根目录进入真实数 `float64` Nix 环境。 | Start from the repository root in the real-valued `float64` Nix profile. |

```bash
nix develop .#default --command jupyter lab \
  examples/cem_exact_extension_walkthrough/pyeidors_walkthrough.ipynb
```

| 中文 | English |
|---|---|
| VS Code 中先运行 `nix develop .#default --command python examples/cem_exact_extension_walkthrough/register_vscode_kernel.py`，重载窗口后选择 **Jupyter Kernel → PyEIDORS real float64 (Nix)**。不要选择裸的 `/nix/store/.../bin/python`。 | In VS Code, first run `nix develop .#default --command python examples/cem_exact_extension_walkthrough/register_vscode_kernel.py`, reload the window, and select **Jupyter Kernel → PyEIDORS real float64 (Nix)**. Do not select the raw `/nix/store/.../bin/python`. |

| 参数 | 中文 | English |
|---|---|---|
| `REGENERATE=False` | 复用已有认证结果，适合首次讲解。 | Reuse the certified result; recommended for the first walkthrough. |
| `REGENERATE=True` | 强制重新执行 DOLFINx 组装。 | Force a fresh DOLFINx assembly. |
"""
        ),
        _code(
            """
from pathlib import Path
import sys

NOTEBOOK_DIR = Path.cwd().resolve()
if NOTEBOOK_DIR.name != "cem_exact_extension_walkthrough":
    NOTEBOOK_DIR = Path("examples/cem_exact_extension_walkthrough").resolve()
REPO_ROOT = NOTEBOOK_DIR.parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src", NOTEBOOK_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from experiment_common import (  # noqa: E402
    build_classic_state,
    build_robin_state,
    exact_reference_metrics,
    formulation_diagnostics,
    load_assembled_blocks,
    load_csv_records,
    load_forward_fixture,
    load_portable_exact_reference,
    plot_forward_fixture,
    plot_forward_solution,
    solve_classic,
    solve_robin,
    summarize_accuracy_records,
)
from pyeidors_debug import ensure_pyeidors_case  # noqa: E402

for font_path in (
    Path("/mnt/c/Windows/Fonts/times.ttf"),
    Path("/mnt/c/Windows/Fonts/timesbd.ttf"),
    Path("/mnt/c/Windows/Fonts/msyh.ttc"),
):
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = ["Times New Roman", "Microsoft YaHei"]
"""
        ),
        _code(
            """
# 选择案例和是否重新组装 / Select the case and whether to reassemble.
CASE_ID = "X01"
REGENERATE = False
SUITE_OUTPUT = REPO_ROOT / "output" / "cem_exact_extension"
REFERENCE_PATH = NOTEBOOK_DIR / "fixtures" / "X01" / "exact_reference.json"
METRICS_PATH = NOTEBOOK_DIR / "expected" / "cem_exact_extension_metrics.csv"
FIGURE_DIR = NOTEBOOK_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
"""
        ),
        _shared_symbol_dictionary_markdown("PyEIDORS/DOLFINx"),
        _markdown(
            r"""
## 步骤 / Steps

### 1. 获取共享案例与 PyEIDORS 分块 / Obtain the shared case and PyEIDORS blocks

| 中文 | English |
|---|---|
| 共享夹具固定了节点、三角形、电极边、逐单元电导率、接触阻抗、电流模式、P1 阶次和真实数 `float64`。三个 FEM 框架导入完全相同的规范数据。 | The shared fixture fixes nodes, triangles, electrode edges, per-cell conductivity, contact impedance, current patterns, P1 order, and real `float64`. All three FEM frameworks import the same canonical data. |
"""
        ),
        _code(
            """
fixture, pyeidors_report = ensure_pyeidors_case(
    CASE_ID,
    SUITE_OUTPUT,
    regenerate=REGENERATE,
)
block_path = Path(fixture["case_dir"]) / "pyeidors_assembled_blocks.mat"
blocks = load_assembled_blocks(block_path)
forward_fixture = load_forward_fixture(
    Path(fixture["mat_path"]),
    Path(fixture["metadata_path"]),
)
solver_mesh_fingerprint = pyeidors_report["discretization"]["mesh_fingerprint"]
assert forward_fixture.mesh_fingerprint == solver_mesh_fingerprint

{
    "case": CASE_ID,
    "N_nodes": forward_fixture.nodes.shape[0],
    "K_cells": forward_fixture.cells.shape[0],
    "L_electrodes": forward_fixture.electrode_count,
    "P_current_patterns": forward_fixture.currents.shape[1],
    "potential_order": forward_fixture.potential_order,
    "scalar_dtype": forward_fixture.scalar_dtype,
    "conductivity_pattern": forward_fixture.conductivity_pattern,
    "unique_cell_conductivity": np.unique(forward_fixture.cell_conductivity),
    "contact_impedance_exact": forward_fixture.contact_impedance_exact,
    "first_current_pattern": forward_fixture.currents[:, 0],
    "current_column_sums": np.sum(forward_fixture.currents, axis=0),
    "A_R_shape": blocks.robin_matrix.shape,
    "A_R_nnz": blocks.robin_matrix.nnz,
    "C_shape": blocks.coupling.shape,
    "D_shape": blocks.electrode_matrix.shape,
    "I_shape": blocks.currents.shape,
    "mesh_fingerprint": solver_mesh_fingerprint,
}
"""
        ),
        _markdown(
            r"""
### 2. 显示公平的正问题条件 / Display the fair forward conditions

| 中文 | English |
|---|---|
| 左图显示实际加载的 P1 网格和每个三角形的电导率；右图在完全相同的边界上显示第一个电流模式的注入电极 `+I` 与回流电极 `−I`。X01 是均匀背景正问题，所以没有内部异常物；求解目标是在给定 $\sigma,z,I$ 后计算体电势 $u$ 与边界电压 $U$。 | The left panel shows the loaded P1 mesh and conductivity in every triangle. The right panel shows the injecting `+I` and returning `−I` electrodes for the first drive on the same boundary. X01 is a uniform-background forward problem with no interior anomaly; its task is to compute body potential $u$ and boundary voltage $U$ for prescribed $\sigma,z,I$. |
| 图题中的网格指纹来自规范节点、三角形和电极边；上一个单元已经断言它与 PyEIDORS 报告一致。 | The mesh fingerprint in the title is computed from canonical nodes, triangles, and electrode edges; the previous cell asserted that it matches the PyEIDORS report. |
"""
        ),
        _code(
            """
fairness_figure, fairness_axes = plot_forward_fixture(
    forward_fixture,
    current_column=0,
)
fairness_figure.savefig(
    FIGURE_DIR / f"{CASE_ID}_pyeidors_forward_setup.png",
    dpi=180,
    bbox_inches="tight",
)
fairness_figure
"""
        ),
        _markdown(
            r"""
### 3. 传统经典 CEM / Traditional Classic CEM

| 中文 | English |
|---|---|
| 经典方法把体内节点电势 $u$、电极电势 $U$ 和零均值约束的拉格朗日乘子 $\lambda$ 放入一个增广线性系统，一次分解后求解全部电流右端项。 | The Classic method places body potentials $u$, electrode voltages $U$, and the zero-mean gauge multiplier $\lambda$ in one augmented linear system, then solves all current right-hand sides after one factorization. |

$$
\begin{bmatrix}
A_R & C & 0 \\
C^\mathsf{T} & D & \mathbf{1} \\
0 & \mathbf{1}^\mathsf{T} & 0
\end{bmatrix}
\begin{bmatrix}
u \\ U \\ \lambda
\end{bmatrix}
=
\begin{bmatrix}
0 \\ I \\ 0
\end{bmatrix}.
$$

| 中文 | English |
|---|---|
| 在下一单元后暂停，重点查看 `classic_state.system_matrix`、`classic_state.factor`、`classic_solution.body_potential` 和 `classic_solution.electrode_voltage`。 | Stop after the next cell and inspect `classic_state.system_matrix`, `classic_state.factor`, `classic_solution.body_potential`, and `classic_solution.electrode_voltage`. |
"""
        ),
        _code(
            """
classic_state = build_classic_state(blocks)
classic_solution = solve_classic(classic_state, blocks.currents)

{
    "augmented_shape": classic_state.system_matrix.shape,
    "augmented_nnz": classic_state.system_matrix.nnz,
    "body_potential_shape": classic_solution.body_potential.shape,
    "electrode_voltage_shape": classic_solution.electrode_voltage.shape,
}
"""
        ),
        _markdown(
            r"""
### 4. Robin/跨导 CEM / Robin/transconductance CEM

| 中文 | English |
|---|---|
| $Q$ 是电极零和子空间的正交基，即 $Q^\mathsf{T}Q=I$ 且 $Q^\mathsf{T}\mathbf{1}=0$。先分解 $A_R$，消去体内未知量，再只在 $L-1$ 维电极子空间求解。 | $Q$ is an orthonormal basis of the zero-sum electrode subspace: $Q^\mathsf{T}Q=I$ and $Q^\mathsf{T}\mathbf{1}=0$. Factor $A_R$, eliminate the body unknowns, and solve only on the $L-1$ dimensional electrode subspace. |

$$
R=A_R^{-1}CQ,\qquad
T_r=Q^\mathsf{T}\left(DQ-C^\mathsf{T}R\right)
=Q^\mathsf{T}\left(D-C^\mathsf{T}A_R^{-1}C\right)Q.
$$

$$
T_r y=Q^\mathsf{T}I,\qquad
U=Qy,\qquad
u=-Ry.
$$

| 中文 | English |
|---|---|
| 代码不会直接求逆完整的奇异跨导矩阵；`response_basis` 对应 $R$，`schur_action_basis` 对应 $DQ-C^\mathsf{T}R$，`reduced_map` 对应 $T_r$。 | The code never directly inverts the full singular transconductance matrix; `response_basis` is $R$, `schur_action_basis` is $DQ-C^\mathsf{T}R$, and `reduced_map` is $T_r$. |
"""
        ),
        _code(
            """
robin_state = build_robin_state(blocks)
robin_solution = solve_robin(robin_state, blocks.currents)

{
    "Q_shape": robin_state.electrode_basis.shape,
    "response_basis_shape": robin_state.response_basis.shape,
    "Schur_action_shape": robin_state.schur_action_basis.shape,
    "reduced_map_shape": robin_state.reduced_map.shape,
    "Q_orthogonality_error": np.linalg.norm(
        robin_state.electrode_basis.T @ robin_state.electrode_basis
        - np.eye(blocks.electrode_count - 1)
    ),
    "Q_zero_sum_error": np.linalg.norm(
        np.ones(blocks.electrode_count) @ robin_state.electrode_basis
    ),
    "reduced_condition_number": np.linalg.cond(robin_state.reduced_map),
}
"""
        ),
        _markdown(
            """### 5. 比较两条浮点计算路径 / Compare the two floating-point routes

| 中文 | English |
|---|---|
| 两种方法在精确算术下等价，但矩阵分解、消元和乘法顺序不同，因此 `float64` 结果可能存在舍入级差异。 | The two methods are equivalent in exact arithmetic, but different factorization, elimination, and multiplication orders can produce roundoff-level differences in `float64`. |
"""
        ),
        _code(
            """
solutions = {
    "classic": classic_solution,
    "robin_transconductance": robin_solution,
}
diagnostics = formulation_diagnostics(blocks, solutions)
diagnostics
"""
        ),
        _markdown(
            r"""
### 6. 求解结果可视化 / Forward-result visualization

| 中文 | English |
|---|---|
| 上排使用同一色标显示第一个注流模式的 Classic 体电势、Robin 体电势和体电势差值 `Robin − Classic`；下排使用同一电压纵轴显示两条电极电压曲线，并单独放大舍入级电压差。 | The top row uses shared limits for the Classic body potential, Robin body potential, and signed `Robin − Classic` body-potential difference of the first drive. The bottom row uses shared voltage limits for both electrode traces and separately magnifies their roundoff-level voltage difference. |
| 这些图直接读取前面用于诊断的 `classic_solution` 与 `robin_solution`，没有重新求解或替换数据。 | These figures read the same `classic_solution` and `robin_solution` arrays used by the diagnostics; they do not rerun or replace the solve. |
"""
        ),
        _code(
            """
result_figure, result_axes = plot_forward_solution(
    forward_fixture,
    solutions,
    current_column=0,
)
result_figure.savefig(
    FIGURE_DIR / f"{CASE_ID}_pyeidors_classic_robin_results.png",
    dpi=180,
    bbox_inches="tight",
)
result_figure
"""
        ),
        _code(
            """
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].spy(classic_state.system_matrix, markersize=1.1, color="#475569")
axes[0].set_title("经典 CEM 增广矩阵 / Classic augmented CEM matrix")
axes[1].imshow(robin_state.reduced_map, cmap="cividis", aspect="auto")
axes[1].set_title("Robin 约化映射 / Robin reduced map $T_r$")
fig.tight_layout()
"""
        ),
        _markdown(
            r"""
### 7. 与有理数精确电压比较 / Compare with the exact rational voltage

| 中文 | English |
|---|---|
| X01 参考文件保存分数而不是舍入后的小数。候选 `float64` 数值先被解释为其精确二进制有理数，再以 100 位精度计算误差，因此评估过程不会再引入普通双精度减法误差。 | The X01 reference stores fractions rather than rounded decimal truth. Candidate `float64` values are promoted to their exact binary rational values before the 100-digit metric evaluation, so the metric calculation does not introduce another ordinary double-precision subtraction error. |

$$
\varepsilon_{\mathrm{truth}}
=\frac{\left\|U_{\mathrm{float64}}-U_{\mathbb{Q}}\right\|_F}
{\left\|U_{\mathbb{Q}}\right\|_F}.
$$
"""
        ),
        _code(
            """
exact_reference = load_portable_exact_reference(REFERENCE_PATH)
exact_metrics = {
    name: exact_reference_metrics(solution.electrode_voltage, exact_reference)
    for name, solution in solutions.items()
}
exact_metrics
"""
        ),
        _markdown(
            r"""
## 检查 / Checks

| 中文 | English |
|---|---|
| 只有经典有理数系统残差与 Robin 有理数系统残差都严格等于零、两种精确电压完全相同且电压规范残差严格为零时，参考解才被认证。 | The exact reference is certified only when the Classic rational residual and Robin rational residual are both exactly zero, both exact voltages are identical, and the voltage gauge residual is exactly zero. |
"""
        ),
        _code(
            """
exact_reference["certification"]
"""
        ),
        _code(
            """
stored_voltages = {
    name: np.asarray(values, dtype=np.float64)
    for name, values in pyeidors_report["raw_electrode_voltages"].items()
}
assert np.array_equal(
    classic_solution.electrode_voltage,
    stored_voltages["classic"],
)
assert np.array_equal(
    robin_solution.electrode_voltage,
    stored_voltages["robin_transconductance"],
)
assert exact_reference["certification"]["exact_classic_residual_zero"]
assert exact_reference["certification"]["exact_robin_residual_zero"]
assert exact_reference["certification"]["exact_classic_robin_identical"]
"All selected-case checks passed."
"""
        ),
        _markdown(
            """### 复现 38 个案例的报告数字 / Reproduce the 38-case report numbers

| 中文 | English |
|---|---|
| 下面的单元从冻结的 228 条精度记录重新计算几何平均误差、逐案例胜出次数和 Q4 网格排序。 | The next cell recomputes geometric-mean errors, per-case win counts, and the Q4 ordering from the 228 frozen accuracy records. |
"""
        ),
        _code(
            """
summary = summarize_accuracy_records(load_csv_records(METRICS_PATH))
{
    "record_count": summary["record_count"],
    "case_count": summary["case_count"],
    "geometric_means": summary["geometric_means"],
    "win_counts": summary["win_counts"],
    "q4_summary": summary["q4_summary"],
}
"""
        ),
        _markdown(
            r"""
## 后续步骤 / Next Steps

| 中文 | English |
|---|---|
| 1. 完成完整 `prepare` 后可以修改 `CASE_ID`。<br>2. 设置 `REGENERATE=True` 检查 DOLFINx 组装。<br>3. 在 `pyeidors_debug.py` 中设置断点逐行查看变量。<br>4. 对同一 MSH/JSON 运行 NGSolve Notebook，并对同一 MAT 运行 MATLAB 脚本。<br>5. 按 `README.md` 运行完整 `compare` 与 `timing`。 | 1. Change `CASE_ID` after the full `prepare` step.<br>2. Set `REGENERATE=True` to inspect DOLFINx assembly.<br>3. Set breakpoints in `pyeidors_debug.py` for line-by-line inspection.<br>4. Run the NGSolve notebook on the same MSH/JSON fixture and the MATLAB script on the same MAT fixture.<br>5. Run the full `compare` and `timing` commands from `README.md`. |
"""
        ),
    ]
    return notebook


def build_ngsolve_notebook():
    notebook = _new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "NGSolve float64",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    notebook["cells"] = [
        _markdown(
            r"""
# NGSolve：经典 CEM 与 Robin CEM / Classic CEM versus Robin CEM

## 目标 / Goal

| 中文 | English |
|---|---|
| 导入与 PyEIDORS、EIDORS 相同的 X01 Gmsh 网格，用 NGSolve 组装 $A_R,C,D$，逐步运行两种数学等价的 CEM 求解方法，再把 NGSolve 的 `float64` 电压与同一个 $\mathbb{Q}$ 有理数精确参考解比较。 | Import the same canonical X01 Gmsh mesh used by PyEIDORS and EIDORS, assemble $A_R,C,D$ with NGSolve, step through both mathematically equivalent CEM solvers, and compare NGSolve's `float64` voltage with the same certified exact reference over $\mathbb{Q}$. |
"""
        ),
        _markdown(
            r"""
## 设置 / Setup

| 中文 | English |
|---|---|
| 按 `README.md` 在隔离的 NGSolve 环境中启动本 Notebook。小型可移植 X01 MSH/JSON 夹具不依赖完整 38 案例预生成结果。 | Launch this notebook in the isolated NGSolve environment described in `README.md`. The portable X01 MSH/JSON fixture works before the complete 38-case suite is prepared. |
"""
        ),
        _code(
            """
from pathlib import Path
import json
import sys

NOTEBOOK_DIR = Path.cwd().resolve()
if NOTEBOOK_DIR.name != "cem_exact_extension_walkthrough":
    NOTEBOOK_DIR = Path("examples/cem_exact_extension_walkthrough").resolve()
REPO_ROOT = NOTEBOOK_DIR.parents[1]
for path in (REPO_ROOT, NOTEBOOK_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from experiment_common import (  # noqa: E402
    build_classic_state,
    build_robin_state,
    exact_reference_metrics,
    formulation_diagnostics,
    load_assembled_blocks,
    load_csv_records,
    load_forward_fixture,
    load_portable_exact_reference,
    plot_forward_fixture,
    plot_forward_solution,
    solve_classic,
    solve_robin,
    summarize_accuracy_records,
)
from ngsolve_debug import resolve_fixture  # noqa: E402
from scripts.benchmarks.ngsolve_cem_exact_extension_case import run_case  # noqa: E402

for font_path in (
    Path("/mnt/c/Windows/Fonts/times.ttf"),
    Path("/mnt/c/Windows/Fonts/timesbd.ttf"),
    Path("/mnt/c/Windows/Fonts/msyh.ttc"),
):
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = ["Times New Roman", "Microsoft YaHei"]
"""
        ),
        _code(
            """
# 选择案例和计时重复次数 / Select the case and timing repetitions.
CASE_ID = "X01"
REGENERATE = False
TIMING_REPEATS = 11
SUITE_OUTPUT = REPO_ROOT / "output" / "cem_exact_extension"
REFERENCE_PATH = NOTEBOOK_DIR / "fixtures" / "X01" / "exact_reference.json"
METRICS_PATH = NOTEBOOK_DIR / "expected" / "cem_exact_extension_metrics.csv"
FIGURE_DIR = NOTEBOOK_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
"""
        ),
        _shared_symbol_dictionary_markdown("NGSolve"),
        _markdown(
            r"""
## 步骤 / Steps

### 1. 用 NGSolve 组装规范夹具 / Assemble the canonical fixture with NGSolve

| 中文 | English |
|---|---|
| NGSolve 报告记录导入网格指纹、材料电导率摘要、P1 阶次和 `float64` 求解器元数据，用来证明三个框架使用同一个离散问题。 | The NGSolve report records the imported mesh fingerprint, material-conductivity digest, P1 order, and `float64` solver metadata, proving that the three frameworks use the same discrete problem. |
"""
        ),
        _code(
            """
mesh_path, metadata_path, case_dir = resolve_fixture(CASE_ID, SUITE_OUTPUT)
case_dir.mkdir(parents=True, exist_ok=True)
report_path = case_dir / "ngsolve_report.json"
block_path = case_dir / "ngsolve_assembled_blocks.mat"

if REGENERATE or not report_path.exists() or not block_path.exists():
    ngsolve_report = run_case(
        mesh_path,
        metadata_path,
        report_path,
        timing_repeats=TIMING_REPEATS,
    )
else:
    ngsolve_report = json.loads(report_path.read_text(encoding="utf-8"))

blocks = load_assembled_blocks(block_path)
forward_fixture = load_forward_fixture(
    metadata_path.with_suffix(".mat"),
    metadata_path,
)
solver_mesh_fingerprint = ngsolve_report["discretization"]["mesh_fingerprint"]
assert forward_fixture.mesh_fingerprint == solver_mesh_fingerprint
{
    "mesh": str(mesh_path),
    "N_nodes": forward_fixture.nodes.shape[0],
    "K_cells": forward_fixture.cells.shape[0],
    "L_electrodes": forward_fixture.electrode_count,
    "P_current_patterns": forward_fixture.currents.shape[1],
    "potential_order": forward_fixture.potential_order,
    "scalar_dtype": forward_fixture.scalar_dtype,
    "conductivity_pattern": forward_fixture.conductivity_pattern,
    "unique_cell_conductivity": np.unique(forward_fixture.cell_conductivity),
    "contact_impedance_exact": forward_fixture.contact_impedance_exact,
    "first_current_pattern": forward_fixture.currents[:, 0],
    "current_column_sums": np.sum(forward_fixture.currents, axis=0),
    "A_R_shape": blocks.robin_matrix.shape,
    "A_R_nnz": blocks.robin_matrix.nnz,
    "C_shape": blocks.coupling.shape,
    "D_shape": blocks.electrode_matrix.shape,
    "mesh_fingerprint": solver_mesh_fingerprint,
}
"""
        ),
        _markdown(
            r"""
### 2. 显示公平的正问题条件 / Display the fair forward conditions

| 中文 | English |
|---|---|
| NGSolve 使用的 MSH 与下面绘图读取的同名 MAT/JSON 由同一个规范有理夹具同时导出。左图显示 P1 网格和逐单元电导率，右图显示第一个边界电流模式。 | The NGSolve MSH and the same-stem MAT/JSON used below were exported together from one canonical rational fixture. The left panel shows the P1 mesh and per-cell conductivity; the right panel shows the first boundary-current pattern. |
| X01 计算的是均匀背景正问题：输入网格、$\sigma=1/8$、$z=1$ 和相邻电流模式，输出 $u$ 与电极电压 $U$；它不是逆问题重构。 | X01 solves a uniform-background forward problem: the mesh, $\sigma=1/8$, $z=1$, and adjacent drives are inputs, while $u$ and electrode voltage $U$ are outputs. It is not an inverse reconstruction. |
"""
        ),
        _code(
            """
fairness_figure, fairness_axes = plot_forward_fixture(
    forward_fixture,
    current_column=0,
)
fairness_figure.savefig(
    FIGURE_DIR / f"{CASE_ID}_ngsolve_forward_setup.png",
    dpi=180,
    bbox_inches="tight",
)
fairness_figure
"""
        ),
        _markdown(
            r"""
### 3. 传统经典 CEM / Traditional Classic CEM

| 中文 | English |
|---|---|
| 分解完整的增广稀疏矩阵，并一次求解所有电流列。 | Factor the complete augmented sparse matrix and solve all current columns. |

$$
\begin{bmatrix}
A_R & C & 0 \\
C^\mathsf{T} & D & \mathbf{1} \\
0 & \mathbf{1}^\mathsf{T} & 0
\end{bmatrix}
\begin{bmatrix}
u \\ U \\ \lambda
\end{bmatrix}
=
\begin{bmatrix}
0 \\ I \\ 0
\end{bmatrix}.
$$
"""
        ),
        _code(
            """
classic_state = build_classic_state(blocks)
classic_solution = solve_classic(classic_state, blocks.currents)
{
    "augmented_shape": classic_state.system_matrix.shape,
    "augmented_nnz": classic_state.system_matrix.nnz,
    "voltage_shape": classic_solution.electrode_voltage.shape,
}
"""
        ),
        _markdown(
            r"""
### 4. Robin/跨导 CEM / Robin/transconductance CEM

| 中文 | English |
|---|---|
| 分解 $A_R$，构造响应基 $R=A_R^{-1}CQ$ 和约化映射 $T_r$，再求解 $L-1$ 维电极问题。 | Factor $A_R$, construct the response basis $R=A_R^{-1}CQ$ and reduced map $T_r$, then solve the $L-1$ dimensional electrode problem. |

$$
R=A_R^{-1}CQ,\qquad
T_r=Q^\mathsf{T}\left(DQ-C^\mathsf{T}R\right)
=Q^\mathsf{T}\left(D-C^\mathsf{T}A_R^{-1}C\right)Q.
$$

$$
T_r y=Q^\mathsf{T}I,\qquad
U=Qy,\qquad
u=-Ry.
$$
"""
        ),
        _code(
            """
robin_state = build_robin_state(blocks)
robin_solution = solve_robin(robin_state, blocks.currents)
{
    "Q_shape": robin_state.electrode_basis.shape,
    "response_basis_shape": robin_state.response_basis.shape,
    "reduced_map_shape": robin_state.reduced_map.shape,
    "reduced_condition_number": np.linalg.cond(robin_state.reduced_map),
}
"""
        ),
        _markdown(
            """### 5. 浮点等价性诊断 / Floating-point equivalence diagnostics

| 中文 | English |
|---|---|
| 这里比较同一个 NGSolve 分块经过两条不同线性代数路径后得到的电极电压、体电势、缩放后向残差和零均值规范残差。 | This section compares electrode voltages, body potentials, scaled backward residuals, and zero-mean gauge residuals from the two algebraic routes applied to the same NGSolve blocks. |
"""
        ),
        _code(
            """
solutions = {
    "classic": classic_solution,
    "robin_transconductance": robin_solution,
}
diagnostics = formulation_diagnostics(blocks, solutions)
diagnostics
"""
        ),
        _markdown(
            r"""
### 6. 求解结果可视化 / Forward-result visualization

| 中文 | English |
|---|---|
| 上排比较同一个 NGSolve 分块得到的 Classic 体电势、Robin 体电势与体电势差值 `Robin − Classic`；下排显示两个电极电压序列及舍入级差值。Classic 与 Robin 主结果使用共同色标和共同纵轴，因此不能靠不同的自动缩放制造视觉差异。 | The top row compares Classic body potential, Robin body potential, and the `Robin − Classic` body-potential difference from the same NGSolve blocks. The bottom row shows both electrode-voltage sequences and their roundoff-level difference. Shared colour and voltage limits prevent separate autoscaling from manufacturing a visual discrepancy. |
| 图中数据就是上一单元 `diagnostics` 使用的 `solutions`，没有额外插值到另一张网格。 | The plotted data are the same `solutions` used by the preceding diagnostics, without interpolation to another mesh. |
"""
        ),
        _code(
            """
result_figure, result_axes = plot_forward_solution(
    forward_fixture,
    solutions,
    current_column=0,
)
result_figure.savefig(
    FIGURE_DIR / f"{CASE_ID}_ngsolve_classic_robin_results.png",
    dpi=180,
    bbox_inches="tight",
)
result_figure
"""
        ),
        _code(
            """
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].spy(classic_state.system_matrix, markersize=1.1, color="#475569")
axes[0].set_title("NGSolve 经典增广矩阵 / Classic augmented matrix")
axes[1].imshow(robin_state.reduced_map, cmap="cividis", aspect="auto")
axes[1].set_title("NGSolve Robin 约化映射 / Robin reduced map $T_r$")
fig.tight_layout()
"""
        ),
        _markdown(
            r"""### 7. 相对于有理数精确电压的误差 / Accuracy relative to the exact rational voltage

| 中文 | English |
|---|---|
| 与 PyEIDORS 使用相同的分数形式精确参考解；这样求解器误差不会被另一个双精度“真值”污染。 | Use the same fraction-valued exact reference as PyEIDORS, so solver error is not contaminated by another double-precision “truth”. |

$$
\varepsilon_{\mathrm{truth}}
=\frac{\left\|U_{\mathrm{float64}}-U_{\mathbb{Q}}\right\|_F}
{\left\|U_{\mathbb{Q}}\right\|_F}.
$$
"""
        ),
        _code(
            """
exact_reference = load_portable_exact_reference(REFERENCE_PATH)
exact_metrics = {
    name: exact_reference_metrics(solution.electrode_voltage, exact_reference)
    for name, solution in solutions.items()
}
exact_metrics
"""
        ),
        _markdown(
            r"""
## 检查 / Checks

| 中文 | English |
|---|---|
| 显式教学求解路径必须逐位复现正式 NGSolve 案例运行器保存的电极电压，否则立即失败。 | The explicit walkthrough path must reproduce the electrode voltage stored by the official NGSolve case runner exactly; otherwise it fails immediately. |
"""
        ),
        _code(
            """
stored_voltages = {
    name: np.asarray(values, dtype=np.float64)
    for name, values in ngsolve_report["raw_electrode_voltages"].items()
}
assert np.array_equal(
    classic_solution.electrode_voltage,
    stored_voltages["classic"],
)
assert np.array_equal(
    robin_solution.electrode_voltage,
    stored_voltages["robin_transconductance"],
)
"All selected-case checks passed."
"""
        ),
        _markdown(
            """### 复现 38 个案例的报告数字 / Reproduce the 38-case report numbers

| 中文 | English |
|---|---|
| 使用相同的冻结 228 条精度记录复算三个框架、两种公式的汇总指标。 | Recompute the three-framework, two-formulation summary from the same 228 frozen accuracy records. |
"""
        ),
        _code(
            """
summary = summarize_accuracy_records(load_csv_records(METRICS_PATH))
{
    "record_count": summary["record_count"],
    "case_count": summary["case_count"],
    "geometric_means": summary["geometric_means"],
    "win_counts": summary["win_counts"],
    "q4_summary": summary["q4_summary"],
}
"""
        ),
        _markdown(
            r"""
## 后续步骤 / Next Steps

| 中文 | English |
|---|---|
| 1. 先运行 PyEIDORS `prepare` 生成全部 38 个共享夹具。<br>2. 运行 `ngsolve_debug.py --all --timing-repeats 11`。<br>3. 在 `assemble_extension_blocks`、`build_classic_state` 和 `build_robin_state` 设置断点。<br>4. 只有三个框架的报告全部存在后才运行 Python `compare`。 | 1. Run PyEIDORS `prepare` to generate all 38 common fixtures.<br>2. Run `ngsolve_debug.py --all --timing-repeats 11`.<br>3. Set VS Code breakpoints in `assemble_extension_blocks`, `build_classic_state`, and `build_robin_state`.<br>4. Run Python `compare` only after all PyEIDORS, NGSolve, and EIDORS reports exist. |
"""
        ),
    ]
    return notebook


def build_exact_truth_notebook():
    notebook = _new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "PyEIDORS real float64 (Nix)",
            "language": "python",
            "name": "pyeidors-real-float64-nix",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    notebook["cells"] = [
        _markdown(
            r"""
# CEM 有理数精确真值：从输入到认证 / Exact rational CEM truth from inputs to certification

## 目标 / Goal

| 中文 | English |
|---|---|
| 不读取一个神秘的“小数真值”，而是从 X01 的有理坐标、整数拓扑、有理单元电导率、有理接触阻抗和整数电流开始，在 $\mathbb{Q}$ 上重新组装 CEM，并逐步得到经典 CEM 与 Robin CEM 完全相同的分数电极电压。 | Instead of loading a mysterious decimal “truth”, start from X01 rational coordinates, integer topology, rational cell conductivities, rational contact impedance, and integer currents; reassemble the CEM over $\mathbb{Q}$ and obtain exactly identical fractional electrode voltages from Classic and Robin CEM. |
| 最后把真实 `float64` PyEIDORS 电压与该分数矩阵比较，展示报告中的真值误差和缩放后向残差如何得到。 | Finally compare the actual `float64` PyEIDORS voltage with that fraction matrix and reproduce the truth error and scaled backward residual used by the report. |
"""
        ),
        _markdown(
            r"""
## 真值的适用范围 / Scope of the truth

| 中文 | English |
|---|---|
| 这里的“数学精确真值”是**固定有理 P1 有限维 CEM 线性系统的唯一精确解**，不是连续 PDE 的解析真值，也不是光滑真实圆域的解析解。 | Here “mathematical exact truth” means the **unique exact solution of one fixed rational finite-dimensional P1 CEM system**; it is not the analytic truth of the continuum PDE and not an analytic solution on a smooth physical disk. |
| 因为三个框架使用同一节点、单元、电极边、$\sigma,z,I$ 和 P1 离散，所以该真值可以隔离并比较组装/线性代数的浮点误差。若要研究连续物理误差，还必须另外做真实圆域网格加密与独立高阶参考实验。 | Because all three frameworks use the same nodes, cells, electrode edges, $\sigma,z,I$, and P1 discretization, this truth isolates floating-point assembly/linear-algebra error. Continuous-physics error requires a separate true-disk refinement study with an independent high-order reference. |
"""
        ),
        _markdown(
            r"""
## 设置 / Setup

| 中文 | English |
|---|---|
| 使用与 PyEIDORS Notebook 相同的真实数 `float64` Nix 内核；SymPy 只负责 $\mathbb{Q}$ 精确代数，DOLFINx 只用于生成待比较的 `float64` 结果。 | Use the same real `float64` Nix kernel as the PyEIDORS notebook. SymPy performs exact algebra over $\mathbb{Q}$; DOLFINx is used only to generate the `float64` candidate being compared. |

```bash
nix develop .#default --command jupyter lab \
  examples/cem_exact_extension_walkthrough/exact_rational_truth_walkthrough.ipynb
```
"""
        ),
        _code(
            """
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

NOTEBOOK_DIR = Path.cwd().resolve()
if NOTEBOOK_DIR.name != "cem_exact_extension_walkthrough":
    NOTEBOOK_DIR = Path("examples/cem_exact_extension_walkthrough").resolve()
REPO_ROOT = NOTEBOOK_DIR.parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src", NOTEBOOK_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from sympy import Matrix, QQ, SparseMatrix  # noqa: E402

from experiment_common import (  # noqa: E402
    build_classic_state,
    build_robin_state,
    exact_reference_metrics,
    load_assembled_blocks,
    load_forward_fixture,
    load_portable_exact_reference,
    plot_forward_fixture,
    plot_forward_solution,
    solve_classic,
    solve_robin,
)
from pyeidors_debug import ensure_pyeidors_case  # noqa: E402
from scripts.benchmarks.cem_exact_extension_suite import (  # noqa: E402
    EXTENSION_CASES,
    _exact_currents,
    _zero_sum_basis,
    assemble_exact_extension_cem,
    extension_case_cell_conductivities,
    extension_case_mesh,
    extension_current_patterns,
)

for font_path in (
    Path("/mnt/c/Windows/Fonts/times.ttf"),
    Path("/mnt/c/Windows/Fonts/timesbd.ttf"),
    Path("/mnt/c/Windows/Fonts/msyh.ttc"),
):
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = ["Times New Roman", "Microsoft YaHei"]
"""
        ),
        _code(
            """
# 教学案例和路径 / Teaching case and paths.
CASE_ID = "X01"
REGENERATE_FLOAT = False
SUITE_OUTPUT = REPO_ROOT / "output" / "cem_exact_extension"
PORTABLE_REFERENCE_PATH = (
    NOTEBOOK_DIR / "fixtures" / CASE_ID / "exact_reference.json"
)
PORTABLE_FIXTURE_DIR = NOTEBOOK_DIR / "fixtures" / CASE_ID / "common_mesh"
FIGURE_DIR = NOTEBOOK_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
"""
        ),
        _markdown(
            r"""
## 符号与变量字典 / Symbol and variable dictionary

| 变量或符号 | 中文含义 | English meaning | 类型或维度 |
|---|---|---|---|
| `case` | X01 的预注册物理/离散设置。 | Preregistered physical/discrete settings for X01. | `ExtensionCase` |
| `exact_nodes` | 不经过浮点舍入的节点坐标，每个坐标都是 `Fraction`。 | Node coordinates before floating-point rounding; every coordinate is a `Fraction`. | $N\times2$ |
| `cells` | P1 三角形拓扑，只有整数节点索引。 | P1 triangle topology containing only integer node indices. | $K\times3$ |
| `tagged_edges` | 边界边与电极标签。 | Boundary edges and electrode labels. | $E_b\times3$ |
| `cell_sigma`, $\sigma_k$ | 第 $k$ 个三角形的精确有理电导率。 | Exact rational conductivity of triangle $k$. | $K$ 个 `Fraction` |
| `case.contact_impedance`, $z$ | 精确有理接触阻抗。 | Exact rational contact impedance. | `Fraction` |
| `I_QQ`, $I$ | 精确整数注入电流矩阵，每列总和为零。 | Exact integer drive-current matrix; every column sums to zero. | $L$×$P$ |
| `A_R_QQ`, $A_R$ | 精确体刚度加 Robin 边界质量矩阵。 | Exact body stiffness plus Robin boundary mass matrix. | $N$×$N$ over $\mathbb{Q}$ |
| `C_QQ`, $C$ | 精确体节点/电极耦合矩阵。 | Exact body-node/electrode coupling matrix. | $N$×$L$ over $\mathbb{Q}$ |
| `D_QQ`, $D$ | 精确电极边界积分块。 | Exact electrode boundary-integral block. | $L$×$L$ over $\mathbb{Q}$ |
| `B_QQ`, $B$ | 有理零和基，列为 $e_j-e_L$；不用含平方根的正交基，保证所有元素仍在 $\mathbb{Q}$。 | Rational zero-sum basis with columns $e_j-e_L$; it avoids a square-root orthonormal basis so every entry remains in $\mathbb{Q}$. | $L$×$(L-1)$ |
| `classic_matrix_QQ`, $\mathcal{K}$ | 带零均值规范的传统 CEM 增广矩阵。 | Gauge-augmented Classic CEM matrix. | $(N+L+1)$×$(N+L+1)$ |
| `classic_basis_solution_QQ` | 对 $L-1$ 个基电流一次精确多右端求解得到的解基。 | Solution basis from one exact multi-RHS solve for $L-1$ basis currents. | $(N+L+1)$×$(L-1)$ |
| `response_QQ` | $A_R^{-1}C$ 的精确解；这里只是记号，代码调用求解器而不构造浮点逆矩阵。 | Exact solution of $A_R X=C$; the inverse is notation only and no floating inverse is formed. | $N$×$L$ |
| `reduced_map_QQ`, $T_r$ | $B^\mathsf{T}(D-C^\mathsf{T}A_R^{-1}C)B$。 | $B^\mathsf{T}(D-C^\mathsf{T}A_R^{-1}C)B$. | $(L-1)$×$(L-1)$ |
| `U_classic_QQ`, `U_robin_QQ` | 两种精确路径得到的电极电压；必须逐分数完全相同。 | Electrode voltages from the two exact routes; they must be identical fraction by fraction. | $L$×$P$ |
| `truth_sha256` | 对规范分数字符串矩阵计算的哈希，用来证明保存/读取没有改变真值。 | Hash of the canonical fraction-string matrix, proving serialization did not change the truth. | SHA-256 |
"""
        ),
        _markdown(
            r"""
## 步骤 / Steps

### 1. 构造并显示完全相同的正问题输入 / Build and display the identical forward input

| 中文 | English |
|---|---|
| X01 有 $N=33$ 个节点、$K=32$ 个三角形、$L=16$ 个电极和 $P=16$ 个相邻电流模式。所有单元的背景电导率为 $\sigma=1/8$，接触阻抗为 $z=1$；没有内部异常物。 | X01 has $N=33$ nodes, $K=32$ triangles, $L=16$ electrodes, and $P=16$ adjacent drive patterns. Every cell has background conductivity $\sigma=1/8$ and contact impedance $z=1$; there is no interior anomaly. |
"""
        ),
        _code(
            """
case = next(item for item in EXTENSION_CASES if item.case_id == CASE_ID)
exact_nodes, cells, tagged_edges, electrode_nodes, electrode_counts = (
    extension_case_mesh(case)
)
cell_sigma = extension_case_cell_conductivities(case, exact_nodes, cells)
current_patterns_float = extension_current_patterns(
    case.n_electrodes,
    case.drive_skip,
)
I_QQ = _exact_currents(current_patterns_float)

forward_fixture = load_forward_fixture(
    PORTABLE_FIXTURE_DIR / "cem_exact_extension_p1.mat",
    PORTABLE_FIXTURE_DIR / "cem_exact_extension_p1.json",
)
assert np.array_equal(
    forward_fixture.nodes,
    np.asarray([[float(x), float(y)] for x, y in exact_nodes]),
)
assert np.array_equal(forward_fixture.cells, cells)
assert np.array_equal(forward_fixture.tagged_edges, tagged_edges)
assert np.array_equal(
    forward_fixture.cell_conductivity,
    np.asarray(cell_sigma, dtype=np.float64),
)

{
    "case": case.case_id,
    "N_nodes": len(exact_nodes),
    "K_cells": cells.shape[0],
    "L_electrodes": case.n_electrodes,
    "P_current_patterns": I_QQ.cols,
    "conductivity_pattern": case.conductivity_pattern,
    "unique_exact_sigma": sorted(set(cell_sigma)),
    "exact_contact_impedance": case.contact_impedance,
    "mesh_fingerprint": forward_fixture.mesh_fingerprint,
}
"""
        ),
        _code(
            """
fairness_figure, fairness_axes = plot_forward_fixture(
    forward_fixture,
    current_column=0,
)
fairness_figure.savefig(
    FIGURE_DIR / f"{CASE_ID}_exact_forward_setup.png",
    dpi=180,
    bbox_inches="tight",
)
fairness_figure
"""
        ),
        _markdown(
            r"""
### 2. 证明输入确实属于有理数域 / Prove that the inputs lie in the rational field

| 中文 | English |
|---|---|
| `Fraction(p,q)` 保存整数分子和非零整数分母，不执行二进制浮点舍入。当前、几何和物性输入都先以这种形式构造；三角形拓扑和电极标签本来就是整数。 | `Fraction(p,q)` stores an integer numerator and a nonzero integer denominator without binary floating-point rounding. Geometry and material inputs are constructed in this form; triangle topology and electrode labels are already integers. |
"""
        ),
        _code(
            """
all_coordinates_are_fraction = all(
    isinstance(value, Fraction)
    for point in exact_nodes
    for value in point
)
all_conductivities_are_fraction = all(
    isinstance(value, Fraction)
    for value in cell_sigma
)
contact_impedance_is_fraction = isinstance(case.contact_impedance, Fraction)
all_currents_are_rational = all(
    value.is_Rational
    for value in I_QQ
)
all_current_columns_are_exactly_zero_sum = all(
    sum(I_QQ[row, column] for row in range(I_QQ.rows)) == 0
    for column in range(I_QQ.cols)
)
input_certification = {
    "all_coordinates_are_fraction": all_coordinates_are_fraction,
    "all_conductivities_are_fraction": all_conductivities_are_fraction,
    "contact_impedance_is_fraction": contact_impedance_is_fraction,
    "all_currents_are_rational": all_currents_are_rational,
    "all_current_columns_are_exactly_zero_sum": (
        all_current_columns_are_exactly_zero_sum
    ),
    "first_five_exact_nodes": exact_nodes[:5],
    "first_drive": list(I_QQ[:, 0]),
}
input_certification
"""
        ),
        _markdown(
            r"""
### 3. 用精确 P1 积分组装 $A_R,C,D$ / Assemble $A_R,C,D$ using exact P1 integrals

对三角形 $K$，P1 梯度为常数，因此体刚度条目可以只用有理数加减乘除：

$$
(A_\Omega^K)_{ij}
=\sigma_K\int_K\nabla\phi_i\cdot\nabla\phi_j\,dx
=\sigma_K\frac{b_i b_j+c_i c_j}{4|K|}.
$$

对一条长度为 $|e|$、接触阻抗为 $z_\ell$ 的电极边，精确边界条目为：

$$
\frac{|e|}{z_\ell}
\begin{bmatrix}
1/3 & 1/6\\
1/6 & 1/3
\end{bmatrix},
\qquad
C_{e,\ell}=-\frac{|e|}{2z_\ell}
\begin{bmatrix}1\\1\end{bmatrix},
\qquad
D_{\ell\ell}\mathrel{+}=\frac{|e|}{z_\ell}.
$$

| 中文 | English |
|---|---|
| 该专用有理网格保证所用电极边长也是可精确开平方的有理数；若不是，组装函数会立即拒绝案例。 | This purpose-built rational mesh guarantees rational electrode-edge lengths with exact square roots; assembly immediately rejects a case that violates this property. |
"""
        ),
        _code(
            """
A_R_QQ, C_QQ, D_QQ = assemble_exact_extension_cem(
    exact_nodes,
    cells,
    tagged_edges,
    cell_conductivities=cell_sigma,
    contact_impedance=case.contact_impedance,
    n_electrodes=case.n_electrodes,
)
block_domains = {
    "A_R_domain": str(A_R_QQ.to_DM().convert_to(QQ).domain),
    "C_domain": str(C_QQ.to_DM().convert_to(QQ).domain),
    "D_domain": str(D_QQ.to_DM().convert_to(QQ).domain),
    "A_R_shape": A_R_QQ.shape,
    "C_shape": C_QQ.shape,
    "D_shape": D_QQ.shape,
    "A_R_sample_fraction": A_R_QQ[0, 0],
    "C_sample_fraction": C_QQ[0, 0],
    "D_sample_fraction": D_QQ[0, 0],
}
block_domains
"""
        ),
        _markdown(
            r"""
### 4. 传统 CEM：在 $\mathbb{Q}$ 上精确 LU / Classic CEM: exact LU over $\mathbb{Q}$

$$
\mathcal{K}
=
\begin{bmatrix}
A_R & C & 0\\
C^\mathsf{T} & D & \mathbf{1}\\
0 & \mathbf{1}^\mathsf{T} & 0
\end{bmatrix},
\qquad
\mathcal{K}
\begin{bmatrix}u\\U\\\lambda\end{bmatrix}
=
\begin{bmatrix}0\\I\\0\end{bmatrix}.
$$

| 中文 | English |
|---|---|
| 为避免逐个电流模式重复分解，先对有理零和基 $B=[e_1-e_L,\ldots,e_{L-1}-e_L]$ 的 $L-1$ 个右端一起求解。因为每列 $I$ 的和为零，所以 $I=B\,I_{1:L-1,:}$。 | To avoid repeated factorizations, solve the $L-1$ right-hand sides of the rational zero-sum basis $B=[e_1-e_L,\ldots,e_{L-1}-e_L]$ together. Since every column of $I$ sums to zero, $I=B\,I_{1:L-1,:}$. |
| `DomainMatrix.convert_to(QQ).lu_solve(...)` 只执行整数/分数运算；没有容差、没有舍入、没有“足够接近零”。 | `DomainMatrix.convert_to(QQ).lu_solve(...)` performs only integer/fraction operations: no tolerance, no rounding, and no “close enough to zero”. |
"""
        ),
        _code(
            """
N = len(exact_nodes)
L = case.n_electrodes
P = I_QQ.cols
B_QQ = _zero_sum_basis(L)
assert B_QQ.T * Matrix.ones(L, 1) == Matrix.zeros(L - 1, 1)

classic_size = N + L + 1
classic_matrix_QQ = SparseMatrix.zeros(classic_size, classic_size)
classic_matrix_QQ[:N, :N] = A_R_QQ
classic_matrix_QQ[:N, N : N + L] = C_QQ
classic_matrix_QQ[N : N + L, :N] = C_QQ.T
classic_matrix_QQ[N : N + L, N : N + L] = D_QQ
for electrode in range(L):
    classic_matrix_QQ[N + electrode, classic_size - 1] = 1
    classic_matrix_QQ[classic_size - 1, N + electrode] = 1

classic_basis_rhs_QQ = SparseMatrix.zeros(classic_size, L - 1)
classic_basis_rhs_QQ[N : N + L, :] = B_QQ
classic_basis_solution_QQ = (
    classic_matrix_QQ.to_DM()
    .convert_to(QQ)
    .lu_solve(classic_basis_rhs_QQ.to_DM().convert_to(QQ))
    .to_Matrix()
)
# A square singular matrix makes exact DomainMatrix.lu_solve raise instead
# of returning a solution. Reaching this line therefore certifies invertibility.
classic_exact_lu_succeeded = True
classic_matrix_has_full_exact_rank = (
    classic_matrix_QQ.rows == classic_matrix_QQ.cols
    and classic_exact_lu_succeeded
)
classic_basis_residual_QQ = (
    classic_matrix_QQ * classic_basis_solution_QQ
    - classic_basis_rhs_QQ
)
classic_residual_is_exact_zero = all(
    value == 0
    for value in classic_basis_residual_QQ
)

current_coordinates_QQ = I_QQ[: L - 1, :]
assert B_QQ * current_coordinates_QQ == I_QQ
classic_solution_QQ = classic_basis_solution_QQ * current_coordinates_QQ
classic_rhs_QQ = SparseMatrix.zeros(classic_size, P)
classic_rhs_QQ[N : N + L, :] = I_QQ
assert classic_matrix_QQ * classic_solution_QQ == classic_rhs_QQ
U_classic_QQ = classic_solution_QQ[N : N + L, :]

{
    "classic_matrix_domain": str(
        classic_matrix_QQ.to_DM().convert_to(QQ).domain
    ),
    "classic_matrix_shape": classic_matrix_QQ.shape,
    "basis_rhs_shape": classic_basis_rhs_QQ.shape,
    "classic_residual_is_exact_zero": classic_residual_is_exact_zero,
    "sample_exact_voltage": U_classic_QQ[0, 0],
}
"""
        ),
        _markdown(
            r"""
### 5. Robin CEM：精确消去体未知量 / Robin CEM: eliminate body unknowns exactly

$$
T=D-C^\mathsf{T}A_R^{-1}C,
\qquad
T_r=B^\mathsf{T}TB.
$$

$$
T_r y=B^\mathsf{T}I,
\qquad
U=By,
\qquad
u=-A_R^{-1}CU.
$$

| 中文 | English |
|---|---|
| 代码仍然不显式构造逆矩阵；`lu_solve(C)` 表示精确求解 $A_RX=C$。Robin 路径的乘法/消元顺序与传统增广路径不同，因此它是独立的代数交叉认证。 | The code still does not form an inverse; `lu_solve(C)` means solving $A_RX=C$ exactly. Robin uses a different elimination and multiplication order from the augmented route, providing an independent algebraic cross-certification. |
"""
        ),
        _code(
            """
response_QQ = (
    A_R_QQ.to_DM()
    .convert_to(QQ)
    .lu_solve(C_QQ.to_DM().convert_to(QQ))
    .to_Matrix()
)
transconductance_QQ = D_QQ - C_QQ.T * response_QQ
reduced_map_QQ = B_QQ.T * transconductance_QQ * B_QQ
reduced_rhs_QQ = B_QQ.T * I_QQ
robin_coefficients_QQ = (
    reduced_map_QQ.to_DM()
    .convert_to(QQ)
    .lu_solve(reduced_rhs_QQ.to_DM().convert_to(QQ))
    .to_Matrix()
)
# The reduced map is square; successful exact LU certifies that it has rank L-1.
robin_exact_lu_succeeded = True
reduced_map_has_full_exact_rank = (
    reduced_map_QQ.rows == reduced_map_QQ.cols
    and robin_exact_lu_succeeded
)
robin_residual_QQ = (
    reduced_map_QQ * robin_coefficients_QQ
    - reduced_rhs_QQ
)
robin_residual_is_exact_zero = all(
    value == 0
    for value in robin_residual_QQ
)
U_robin_QQ = B_QQ * robin_coefficients_QQ
classic_robin_exactly_identical = U_classic_QQ == U_robin_QQ
exact_voltage_gauge_zero = all(
    sum(U_classic_QQ[row, column] for row in range(L)) == 0
    for column in range(P)
)

{
    "reduced_map_domain": str(
        reduced_map_QQ.to_DM().convert_to(QQ).domain
    ),
    "reduced_map_shape": reduced_map_QQ.shape,
    "robin_residual_is_exact_zero": robin_residual_is_exact_zero,
    "classic_robin_exactly_identical": classic_robin_exactly_identical,
    "exact_voltage_gauge_zero": exact_voltage_gauge_zero,
}
"""
        ),
        _markdown(
            r"""
### 6. 精确正问题结果可视化 / Visualize the exact forward result

| 中文 | English |
|---|---|
| 把已经在 $\mathbb{Q}$ 上求出的分数解仅为绘图转换成 `float64`。上排显示精确 Classic/Robin 体电势和它们的差值；下排显示精确电极电压。由于两条有理路径严格同解，差值图应为零。 | Convert the already-solved rational fractions to `float64` only for plotting. The top row shows the exact Classic/Robin body fields and their difference; the bottom row shows exact electrode voltages. Since both rational routes are strictly identical, the difference plots must be zero. |
| 这次转换不参与真值生成、哈希或误差计算；精确认证仍使用原始 SymPy `QQ` 分数对象。 | This conversion is not used to generate the truth, hash, or error metrics; certification still uses the original SymPy `QQ` fractions. |
"""
        ),
        _code(
            """
classic_body_QQ = classic_solution_QQ[:N, :]
robin_body_QQ = -(response_QQ * U_robin_QQ)
assert classic_body_QQ == robin_body_QQ

exact_plot_solutions = {
    "classic": SimpleNamespace(
        body_potential=np.asarray(classic_body_QQ, dtype=np.float64),
        electrode_voltage=np.asarray(U_classic_QQ, dtype=np.float64),
    ),
    "robin_transconductance": SimpleNamespace(
        body_potential=np.asarray(robin_body_QQ, dtype=np.float64),
        electrode_voltage=np.asarray(U_robin_QQ, dtype=np.float64),
    ),
}
exact_result_figure, exact_result_axes = plot_forward_solution(
    forward_fixture,
    exact_plot_solutions,
    current_column=0,
)
exact_result_figure.savefig(
    FIGURE_DIR / f"{CASE_ID}_exact_classic_robin_results.png",
    dpi=180,
    bbox_inches="tight",
)
exact_result_figure
"""
        ),
        _markdown(
            r"""
### 7. 证明解唯一并认证保存的分数真值 / Prove uniqueness and certify the stored fraction truth

唯一性与精确性的证据是：

$$
\operatorname{rank}_{\mathbb{Q}}(\mathcal{K})=N+L+1,
\qquad
\operatorname{rank}_{\mathbb{Q}}(T_r)=L-1,
$$

$$
\mathcal{K}X-B_{\mathrm{rhs}}=0,\qquad
T_rY-B^\mathsf{T}I=0,\qquad
U_{\mathrm{Classic}}-U_{\mathrm{Robin}}=0.
$$

| 中文 | English |
|---|---|
| 上式中的零是 SymPy 有理数对象的严格相等，不是小于某个容差。对方阵调用 `DomainMatrix.convert_to(QQ).lu_solve`：若矩阵奇异会抛出异常；精确 LU 成功就认证满秩与解唯一。随后严格零残差证明展示的分数矩阵正是该唯一解。这样不用再进行一次代价很高的通用行化简求秩。 | Every zero above is strict equality of SymPy rational objects, not a tolerance check. For a square matrix, `DomainMatrix.convert_to(QQ).lu_solve` raises if the matrix is singular; successful exact LU therefore certifies full rank and uniqueness. The exact-zero residual then proves that the displayed fraction matrix is that unique solution. This avoids a second expensive generic row reduction merely to recompute the rank. |
| 最后把每个分数规范化为 `"numerator/denominator"` 字符串并计算 SHA-256；它必须等于随包提供的参考文件哈希。 | Finally canonicalize every fraction as a `"numerator/denominator"` string and compute SHA-256; it must equal the hash in the portable reference file. |
"""
        ),
        _code(
            """
assert classic_matrix_has_full_exact_rank
assert reduced_map_has_full_exact_rank
truth_fraction_strings = [
    [str(U_classic_QQ[row, column]) for column in range(P)]
    for row in range(L)
]
truth_sha256 = hashlib.sha256(
    json.dumps(
        truth_fraction_strings,
        separators=(",", ":"),
    ).encode("ascii")
).hexdigest()
portable_reference = load_portable_exact_reference(PORTABLE_REFERENCE_PATH)
portable_truth_is_identical = (
    truth_fraction_strings == portable_reference["voltage"]
)
portable_hash_is_identical = (
    truth_sha256 == portable_reference["truth_sha256"]
)

certification_summary = {
    **input_certification,
    "classic_matrix_has_full_exact_rank": (
        classic_matrix_has_full_exact_rank
    ),
    "reduced_map_has_full_exact_rank": reduced_map_has_full_exact_rank,
    "classic_residual_is_exact_zero": classic_residual_is_exact_zero,
    "robin_residual_is_exact_zero": robin_residual_is_exact_zero,
    "classic_robin_exactly_identical": (
        classic_robin_exactly_identical
    ),
    "exact_voltage_gauge_zero": exact_voltage_gauge_zero,
    "portable_truth_is_identical": portable_truth_is_identical,
    "portable_hash_is_identical": portable_hash_is_identical,
    "truth_sha256": truth_sha256,
}
certification_summary
"""
        ),
        _markdown(
            r"""
### 8. 计算 `float64` 求解器相对于真值的误差 / Compute solver error relative to truth

候选 `float64` 数字不是用十进制文本近似转换，而是用 `Fraction.from_float` 取出该 IEEE-754 数值所代表的**精确二进制有理数**。随后在 100 位工作精度下评估：

$$
\varepsilon_{\mathrm{truth}}
=\frac{\|U_{\mathrm{float64}}-U_{\mathbb{Q}}\|_F}
{\|U_{\mathbb{Q}}\|_F}.
$$

缩放后向残差回答另一个问题：“候选解对输入的离散系统满足得有多好？”

$$
\eta
=\frac{\|T_r\widehat y-B^\mathsf{T}I\|_F}
{\|T_r\|_F\|\widehat y\|_F+\|B^\mathsf{T}I\|_F}.
$$

| 中文 | English |
|---|---|
| $\varepsilon_{\mathrm{truth}}$ 是前向误差，直接衡量求解器电压距离唯一精确解多远；$\eta$ 是后向残差，衡量候选结果满足方程的程度。残差小不保证前向误差最小，因为条件数会放大误差，所以两者必须同时报告。 | $\varepsilon_{\mathrm{truth}}$ is forward error, measuring distance to the unique exact voltage. $\eta$ is a backward residual, measuring equation satisfaction. A small residual does not guarantee the smallest forward error because conditioning can amplify error, so both must be reported. |
"""
        ),
        _code(
            """
float_fixture, pyeidors_report = ensure_pyeidors_case(
    CASE_ID,
    SUITE_OUTPUT,
    regenerate=REGENERATE_FLOAT,
)
float_blocks = load_assembled_blocks(
    Path(float_fixture["case_dir"]) / "pyeidors_assembled_blocks.mat"
)
float_classic_state = build_classic_state(float_blocks)
float_classic_solution = solve_classic(
    float_classic_state,
    float_blocks.currents,
)
float_robin_state = build_robin_state(float_blocks)
float_robin_solution = solve_robin(
    float_robin_state,
    float_blocks.currents,
)
sample_float64 = float(float_classic_solution.electrode_voltage[0, 0])
sample_float64_as_exact_fraction = Fraction.from_float(sample_float64)
float_metrics = {
    "classic": exact_reference_metrics(
        float_classic_solution.electrode_voltage,
        portable_reference,
    ),
    "robin_transconductance": exact_reference_metrics(
        float_robin_solution.electrode_voltage,
        portable_reference,
    ),
}
{
    "sample_float64": sample_float64,
    "sample_float64_as_exact_fraction": (
        sample_float64_as_exact_fraction
    ),
    "metrics": float_metrics,
}
"""
        ),
        _markdown(
            r"""
## 检查 / Checks

| 检查 | 中文含义 | English meaning |
|---|---|---|
| `all_*_are_fraction/rational` | 几何、物性和电流输入没有先经过不可逆的小数舍入。 | Geometry, material, and current inputs were not first irreversibly rounded. |
| `classic_matrix_has_full_exact_rank` | 带规范的传统 CEM 系统在 $\mathbb{Q}$ 上可逆，解唯一。 | The gauged Classic system is invertible over $\mathbb{Q}$, so its solution is unique. |
| `reduced_map_has_full_exact_rank` | Robin 零和约化系统在 $\mathbb{Q}$ 上可逆。 | The Robin zero-sum reduced system is invertible over $\mathbb{Q}$. |
| `classic_residual_is_exact_zero` | 传统路径的分数解严格满足方程。 | The fractional Classic solution satisfies its equation exactly. |
| `robin_residual_is_exact_zero` | Robin 路径的分数解严格满足方程。 | The fractional Robin solution satisfies its equation exactly. |
| `classic_robin_exactly_identical` | 两种不同代数路径得到逐分数相同的 $U$。 | Two distinct algebraic routes produce fraction-by-fraction identical $U$. |
| `exact_voltage_gauge_zero` | 每个电流模式的电极电压和严格为零。 | Electrode voltages sum exactly to zero for every drive. |
| `portable_hash_is_identical` | 保存到 JSON 后的真值与本次重新计算结果逐字节一致。 | The JSON truth is byte-canonically identical to the recomputed truth. |
"""
        ),
        _code(
            """
required_boolean_checks = {
    name: value
    for name, value in certification_summary.items()
    if isinstance(value, bool)
}
assert all(required_boolean_checks.values())
assert portable_reference["certification"] == {
    "exact_classic_residual_zero": True,
    "exact_robin_residual_zero": True,
    "exact_classic_robin_identical": True,
    "exact_voltage_gauge_zero": True,
}
required_boolean_checks
"""
        ),
        _markdown(
            r"""
## 后续步骤 / Next Steps

| 中文 | English |
|---|---|
| 1. 在 `CASE_ID="X01"` 下逐单元运行并展开任意分数。<br>2. 在 PyEIDORS/NGSolve Notebook 中对照同一网格图、指纹、$\sigma,z,I$。<br>3. 完整 38 案例的 Q0/Q2/Q4、均匀/非均匀电导率、8/16 电极和不同 $z$ 使用同一认证逻辑；大于等于 500 节点的 Q4 案例改用 `python-flint fmpq_mat.solve`，仍然在 $\mathbb{Q}$ 上严格求解和验证零残差。<br>4. 若研究连续 PDE 误差，应另做真实圆域加密实验，不能把本 Notebook 的离散真值误称为连续解析解。 | 1. Run cell-by-cell with `CASE_ID="X01"` and expand any fraction.<br>2. Cross-check the same mesh plot, fingerprint, $\sigma,z,I$ in the PyEIDORS and NGSolve notebooks.<br>3. All 38 Q0/Q2/Q4, uniform/heterogeneous conductivity, 8/16-electrode, and contact-impedance cases use the same certification logic. Q4 cases with at least 500 nodes use `python-flint fmpq_mat.solve`, still solving over $\mathbb{Q}$ with exact-zero residual checks.<br>4. Study continuum-PDE error separately on a true-disk refinement sequence; do not call this notebook's discrete truth a continuum analytic solution. |
"""
        ),
    ]
    return notebook


def main() -> int:
    outputs = {
        PACKAGE_DIR / "pyeidors_walkthrough.ipynb": build_pyeidors_notebook(),
        PACKAGE_DIR / "ngsolve_walkthrough.ipynb": build_ngsolve_notebook(),
        PACKAGE_DIR
        / "exact_rational_truth_walkthrough.ipynb": build_exact_truth_notebook(),
    }
    for path, notebook in outputs.items():
        path.write_text(
            json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
