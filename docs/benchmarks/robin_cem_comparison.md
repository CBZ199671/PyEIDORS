# Robin 边界 CEM：数学原理、PyEIDORS 移植与跨 FEM 对比

## 结论

Deakin、Adler 和 Lionheart 在 EIT 2026 论文《Complete electrode model via
Robin boundary conditions》中给出的算法，不是另一套电极物理模型，而是经典完整电极
模型（CEM）离散矩阵的精确 Schur 补形式。相同网格、函数空间、积分规则、材料参数和
线性代数精度下，两者应只相差舍入误差。

本项目的实测结果与此完全一致：

| 求解器 / 离散 | 标量精度 | 电极电压相对 L2（Robin 对经典） | 特征电阻曲线相对 L2 |
|---|---:|---:|---:|
| PyEIDORS / DOLFINx P1 | complex64 | 7.135e-7 | 3.096e-7 |
| NGSolve H1 P2 | float64 | 6.110e-15 | 2.627e-15 |
| EIDORS P1 | float64 | 5.189e-15 | 5.203e-16 |

三套求解器之间的原始 SI 特征电阻曲线则存在明显更大的离散差异：

| 求解器对（经典 CEM） | 原始曲线相对 L2 | 最大逐点相对差异 | 相关系数 |
|---|---:|---:|---:|
| EIDORS 对 NGSolve | 0.565% | 1.130% | 0.9999999984 |
| EIDORS 对 PyEIDORS | 1.155% | 2.452% | 0.9999999279 |
| NGSolve 对 PyEIDORS | 1.727% | 3.623% | 0.9999999057 |

Robin 公式下得到几乎相同的跨求解器数值。这说明当前可见差异主要来自网格密度、P1/P2
阶次、边界几何近似、积分实现以及 complex64/float64 精度，而不是 CEM 公式改变了物理。

## 根本数学原理

设体内节点电势为 `u`，`L` 个电极电压为 `U`，施加的净电极电流为 `I`，接触阻抗
边界项已经组装进 `A_R = K + B`。经典 CEM 的离散增广系统可写为

```text
[ A_R   C    0 ] [u]   [0]
[ C^T   D    1 ] [U] = [I]
[  0   1^T   0 ] [λ]   [0]
```

其中 `C = -G` 是节点与电极之间的边界耦合，`D` 是电极接触项，最后一行固定
`1^T U = 0` 的零均值规范。第一块行给出

```text
u = -A_R^{-1} C U.
```

代回电极方程后得到电极转导矩阵

```text
T = D - C^T A_R^{-1} C,       I = T U.
```

`T` 在常数电压方向上奇异，这正是电势规范自由度，而不是数值故障。令 `Q` 为
`1^T U = 0` 子空间的一组确定性正交基（本实现使用 Helmert 基），写成 `U = Qy`，则

```text
T_r = Q^T T Q,
T_r y = Q^T I,
U = Qy,
u = -A_R^{-1} C Qy.
```

算法只求解满秩的 `(L-1) × (L-1)` 矩阵 `T_r`，绝不对奇异的完整 `T` 求逆，也
不使用伪逆掩盖电极丢失、秩亏或病态问题。

对复电导率，互易 FEM 系统是复对称矩阵而不一定是 Hermitian 矩阵，因此上述乘积必须
使用非共轭转置 `T`（NumPy 的 `.T`、MATLAB 的 `.'`），不能替换为共轭转置。这一点与
EIDORS `system_mat_1st_order` 的源码注释和实现一致。

## PyEIDORS 移植

实现位于 `src/pyeidors/forward/robin_transconductance.py`，公开类为
`RobinTransconductanceForwardModel`。`EITSystem` 新增选择器：

```python
system = EITSystem(
    ...,
    cem_formulation="robin_transconductance",
)
```

不传该参数时仍使用 `cem_formulation="classic"`，原有行为不变。实现复用了 PyEIDORS
既有电极矩阵的 `B/C/D` 分块，从而保证经典与 Robin 路径具有完全一致的符号约定、
电极标签、接触阻抗单位和积分形式。

每个电导率状态执行以下操作：

1. 组装一次 `A_R`，建立一次稀疏求解器；
2. 一次性求解全部 `L-1` 个 `CQ` Robin 基响应；
3. 构造并检查 `T_r` 的有限性、秩、条件数、对称残差和响应残差；
4. 对任意批量平衡电流只求解小型 `T_r`，再恢复 `U` 和 `u`；
5. 按电导率指纹、后端和标量类型隔离缓存。

不平衡电流、非有限值、秩亏或超过精度相关阈值的病态转导矩阵会给出明确错误；不会
静默使用伪逆。PETSc 路径会记录真实残差、KSP 设置次数、基 RHS 次数及回退原因。

## 对比实验设计

共同物理参数沿用作者源码的圆盘设置：半径 `4 m`、背景电导率 `0.25 S/m`、接触阻抗
`1`、电极覆盖率 `0.7`。为控制三套运行时成本，本次使用 16 个电极，并施加
`k = 1...8` 的余弦/正弦空间电流模式。比较量为论文使用的特征电阻
`||U||₂ / ||I||₂`，保留原始 SI 值，不进行作者 notebook 中用于展示的事后拟合缩放。

离散信息如下：

| 求解器 | 网格 | 电势空间 | 电极积分 | 线性求解 |
|---|---|---|---|---|
| PyEIDORS | 652 点 / 1190 三角形 | DOLFINx P1 | facet forms | complex64 SciPy SuperLU |
| NGSolve | 2514 点 / 4611 单元，9638 DOF | H1 P2 | SymbolicBFI/LFI | NGSolve 组装，float64 SuperLU |
| EIDORS | 6393 点 / 12528 三角形 | P1 | system_mat_fields CEM | 官方经典解 + MATLAB Schur 解 |

NGSolve 阶次保留作者 notebook 的 P2；PyEIDORS 与 EIDORS 使用各自常规 P1。因此
“同一求解器内”结果用于检验公式等价性，“跨求解器”结果只用于量化实际 FEM 离散差异，
两类数字不能混为一谈。

单次计时仅作诊断，不作性能结论：PyEIDORS 经典/Robin 为约 `0.008/0.012 s`，NGSolve
代数经典/Robin 为约 `0.100/0.220 s`，EIDORS 官方经典/Robin Schur 阶段为约
`0.434/0.022 s`。三者包含的组装、缓存和求解阶段不同，不能直接据此宣称固定加速比。

## 复现

PyEIDORS（项目固定的 pure-Nix complex64 CUDA 开发环境）：

```bash
nix develop .#complex64-cuda --command \
  python scripts/benchmarks/compare_cem_formulations.py \
  --output-dir output/cem_formulation_comparison
```

NGSolve 脚本是作者 notebook 的无界面版本，需要独立 NGSolve Python 运行时；它不属于
PyEIDORS 的 runtime 依赖：

```bash
python scripts/benchmarks/ngsolve_cem_formulations.py \
  --output-dir output/cem_formulation_comparison
```

EIDORS 在 MATLAB 中运行：

```matlab
run('compare_with_Eidors/compare_cem_formulations.m')
```

最后用 `compare_cem_formulations.py --skip-pyeidors` 加载三个 CSV/JSON，可重新生成合并
CSV、机器可读报告和 Times New Roman 数字/英文图。实际命令及全部输入路径记录在聚合
JSON 中。

## 产物

- `output/cem_formulation_comparison/cem_formulation_comparison.csv`：三套求解器的原始数据；
- `output/cem_formulation_comparison/cem_formulation_comparison.json`：参数、离散元数据和两类误差；
- `output/cem_formulation_comparison/cem_formulation_comparison.png`：无事后缩放的对比图；
- `pyeidors_report.json`、`ngsolve_report.json`、`eidors_report.json`：各求解器独立诊断；
- `eidors_raw_voltages.mat`：EIDORS 可复核的原始电压矩阵。

作者提供的 `CEM-via-Robin-Boundary` 目录保持原样，未纳入或修改其源码；移植实现和对照
脚本均为项目内独立文件。
