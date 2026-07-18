# Robin 边界 CEM：数学原理、PyEIDORS 移植与公平跨 FEM 对比

## 结论

Deakin、Adler 和 Lionheart 在 EIT 2026 论文《Complete electrode model via
Robin boundary conditions》中给出的算法，不是另一套电极物理模型，而是经典完整电极
模型（CEM）离散矩阵的精确 Schur 补形式。相同网格、函数空间、积分规则、物理参数、
规范条件和标量精度下，两者应只相差舍入误差。

修正后的基准由 PyEIDORS 生成一次公共网格，再由 NGSolve 和 EIDORS 直接导入。三端均为
652 节点、1190 个 P1 三角形、112 条带电极编号的边界边、实数 `float64`；公共 SHA-256
网格指纹为：

```text
3ea76a1e81332ce6bbc49d4f170dbb55a07b2211c6cc74d5666a143bd9643088
```

严格同网格结果如下：

| 求解器 | 电极电压相对 L2（Robin 对经典） | 体内电势相对 L2 | 特征电阻曲线相对 L2 |
|---|---:|---:|---:|
| PyEIDORS / DOLFINx P1 float64 | 2.155e-15 | 2.432e-15 | 7.536e-16 |
| NGSolve P1 float64 | 2.255e-15 | 2.712e-15 | 2.644e-16 |
| EIDORS P1 float64 | 1.192e-15 | 1.407e-15 | 6.808e-16 |

跨求解器的经典 CEM 特征电阻曲线差异也降到了舍入误差量级：EIDORS 对 PyEIDORS 为
`1.800e-15`，NGSolve 对 PyEIDORS 为 `2.796e-14`，EIDORS 对 NGSolve 为
`2.951e-14`。Robin 结果分别为 `1.102e-15`、`2.845e-14`、`2.930e-14`。
这说明三套 FEM 在公共 P1 网格上的 CEM 离散实际上等价；此前约 `0.565%–1.727%` 的
跨求解器差异主要来自各自生成的不同网格和 NGSolve P2/PyEIDORS P1 的阶次差异。

此前 PyEIDORS 的 `7.135e-7` 结果无效于本次公平精度比较。`complex64` 的实部和虚部各是
`float32`，机器精度约 `1.19e-7`；即使输入物理量全为实数，其稀疏因子分解仍按单精度
复数进行。因此 `7.135e-7` 正是合理的单精度舍入量级，不是 Robin 数学算法错误。
本实验是实电导率问题，PyEIDORS 没有数学理由必须使用复数，故改用 real `float64` 的
Nix profile。若以后比较复导纳，三端应统一为 `complex128`，不能拿 `complex64` 与
`float64/complex128` 混比。

## 根本数学原理

设体内节点电势为 `u`，`L` 个电极电压为 `U`，施加的净电极电流为 `I`，接触阻抗
边界项已经组装进 `A_R = K + B`。经典 CEM 的离散增广系统为

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

`T` 在常数电压方向上奇异，这是电势规范自由度。令 `Q` 为 `1^T U = 0` 子空间的
确定性正交 Helmert 基，写成 `U = Qy`，则

```text
T_r = Q^T T Q,
T_r y = Q^T I,
U = Qy,
u = -A_R^{-1} C Qy.
```

算法只求解满秩的 `(L-1) × (L-1)` 矩阵 `T_r`，不对奇异的完整 `T` 求逆，也不使用
伪逆掩盖电极丢失、秩亏或病态问题。

对复电导率，互易 FEM 系统是复对称矩阵而不一定是 Hermitian 矩阵，所以上述乘积必须
使用非共轭转置（NumPy 的 `.T`、MATLAB 的 `.'`），不能替换为共轭转置。

## PyEIDORS 移植

生产实现位于 `src/pyeidors/forward/robin_transconductance.py`，公开类为
`RobinTransconductanceForwardModel`。`EITSystem` 的选择方式为：

```python
system = EITSystem(
    ...,
    cem_formulation="robin_transconductance",
)
```

不传该参数时仍使用 `cem_formulation="classic"`。实现复用既有电极矩阵的 `B/C/D`
分块，保证两条路径的电极标签、符号约定、接触阻抗单位和积分形式相同。

每个电导率状态执行：

1. 组装一次 `A_R`，建立一次稀疏求解器；
2. 一次性求解全部 `L-1` 个 `CQ` Robin 基响应；
3. 构造并检查 `T_r` 的有限性、秩、条件数、对称残差和响应残差；
4. 对任意批量平衡电流只求解小型 `T_r`，再恢复 `U` 和 `u`；
5. 按电导率指纹、后端和标量类型隔离缓存。

本次公平基准中的块矩阵算法还分别与 PyEIDORS 两个生产入口核对：经典入口差异为 `0`，
Robin 入口差异为 `7.545e-16`。EIDORS 公平经典块解与官方 `fwd_solve` 的差异为
`2.693e-15`。

## 公平网格与计时设计

共同物理参数为：半径 `4 m`、背景电导率 `0.25 S/m`、接触阻抗 `1`、16 个电极、
电极覆盖率 `0.7`，以及 `k=1...8` 的余弦/正弦平衡电流。比较量为原始 SI 特征电阻
`||U||₂ / ||I||₂`，不做拟合缩放。

PyEIDORS 输出两种完全同源的网格载体：

- ASCII Gmsh 2.2：NGSolve 导入后重新提取节点、单元和带电极标签的边并重算指纹；
- MATLAB MAT：EIDORS 直接赋值 `nodes/elems/boundary/electrode.nodes` 并核对数量与内容。

聚合器只有在三端同时满足 P1、`float64`、同一 64 位十六进制指纹、导入已验证、计时
作用域相同且无跨公式缓存复用时才接受结果。

速度比较以每端已经装配好的同一组 `A_R/C/D` 为输入，并把 FEM 组装单列报告：

- 冷启动：每次创建全新的公式状态，包含矩阵构造、稀疏/稠密因子分解和全部 16 个 RHS；
- 热求解：每个公式先独立填充一次自己的缓存，样本只包含相同 16 RHS 的求解与恢复；
- 计时前各做一次不保留因子的运行时预热，仅排除语言分派和分配器的一次性开销；
- 两个公式均重复 11 次，测量顺序逐次交替；
- 经典缓存绝不传给 Robin，Robin 缓存也绝不传给经典；
- 报告中保存全部样本、median、IQR、因子分解次数和缓存命中次数。

公平计时中位数如下（单位：秒/全部 16 RHS）：

| 求解器 | 经典冷启动 | Robin 冷启动 | 经典/Robin | 经典热求解 | Robin 热求解 | 经典/Robin |
|---|---:|---:|---:|---:|---:|---:|
| PyEIDORS | 0.002825 | 0.002049 | 1.379× | 0.000286 | 0.0000400 | 7.140× |
| NGSolve | 0.003000 | 0.002020 | 1.485× | 0.000367 | 0.0000478 | 7.692× |
| EIDORS | 0.001962 | 0.002378 | 0.825× | 0.000259 | 0.000113 | 2.302× |

`经典/Robin > 1` 表示 Robin 更快。EIDORS 的 Robin 冷启动因额外构造响应基和小型稠密
LU，约慢 21.2%；但缓存建立后，Robin 热求解约快 2.30 倍。PyEIDORS 和 NGSolve 在
冷、热两阶段的 Robin 都更快。这里比较的是相同块输入上的线性代数求解阶段，不应把
不同语言的 JIT、模型构造或网格生成时间混入公式加速比。

此前报告的 EIDORS `0.434/0.022 s`、PyEIDORS `0.008/0.012 s` 和 NGSolve
`0.100/0.220 s` 不再用于性能结论：它们混合了官方 `fwd_solve`、不同组装范围、不同
网格，以及第二次调用可能命中第一次产生的状态，计时口径不对称。

## 复现

先在 real float64 Nix profile 中生成公共网格并运行 PyEIDORS：

```bash
nix develop .#default --command \
  python scripts/benchmarks/compare_cem_formulations.py \
  --output-dir output/cem_formulation_comparison \
  --timing-repeats 11
```

NGSolve 使用独立 Python 运行时，但必须导入上一步的公共 MSH：

```bash
python scripts/benchmarks/ngsolve_cem_formulations.py \
  --output-dir output/cem_formulation_comparison \
  --mesh output/cem_formulation_comparison/common_mesh/cem_common_p1.msh \
  --mesh-metadata output/cem_formulation_comparison/common_mesh/cem_common_p1.json \
  --timing-repeats 11
```

EIDORS 从环境变量指定的公共 MAT 运行：

```matlab
setenv('CEM_BENCHMARK_OUTPUT_DIR', '<output/cem_formulation_comparison>');
setenv('CEM_COMMON_MESH_MAT', '<output/cem_formulation_comparison/common_mesh/cem_common_p1.mat>');
setenv('CEM_TIMING_REPEATS', '11');
run('compare_with_Eidors/compare_cem_formulations.m');
```

最后以 `--skip-pyeidors` 同时加载三个 CSV/JSON，聚合器会先执行严格公平性校验，再生成
合并报告和图。

## 产物

- `common_mesh/cem_common_p1.msh|mat|json`：公共网格与指纹；
- `cem_formulation_comparison.csv|json|png`：三端数值对比；
- `cem_formulation_timing.csv|png`：冷/热阶段的公平计时；
- `pyeidors_report.json`、`ngsolve_report.json`、`eidors_report.json`：完整样本与诊断；
- `eidors_raw_voltages.mat`：EIDORS 原始电势和电极电压。

作者提供的 `CEM-via-Robin-Boundary` 目录保持原样，未纳入或修改其源码；移植实现和对照
脚本均为项目内独立文件。
