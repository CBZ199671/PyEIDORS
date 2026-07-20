# 有理 P1 CEM 网格加密实验：三种 FEM 求解器的离散数学精度

## 技术摘要

本实验回答一个刻意收窄、但可以严格证明的问题：在三种 FEM 框架面对**同一个有限维
P1 CEM 离散问题**时，谁计算出的电极电压更接近该离散系统的数学精确解？

为避免单一小网格的偶然性，基线采用四级嵌套有理网格 Q0–Q3，节点/三角形数依次为
`33/32`、`65/96`、`129/192`、`257/448`。另在 Q1 上改变接触阻抗、电导率和激励，
共形成 8 个 case。每个 case 中，PyEIDORS/DOLFINx、EIDORS、NGSolve 使用相同节点、
三角形、电极边、P1 次数、物理参数、16 个 RHS、零均值规范和 real `float64`。

主要结果如下：

- 在 Q0–Q3 基线加密序列中，PyEIDORS 的 Robin 解 4/4 最接近精确解；经典解在
  Q0–Q2 第一，Q3 由 EIDORS 第一。Q3 经典中 PyEIDORS 的误差仅为 EIDORS 的
  `1.123×`，因此这是一个很小的排序反转，而不是数量级变化。
- 在全部 8 个 case 中，PyEIDORS 经典赢 6 组、Robin 赢 7 组；EIDORS 分别赢
  G7/G8 的经典解和 G4 的 Robin 解；NGSolve 没有赢得 case。
- 全部 8 组的误差几何平均为：PyEIDORS 经典 `5.167e-16`、Robin `3.904e-16`；
  EIDORS 经典 `6.160e-16`、Robin `6.181e-16`；NGSolve 经典 `4.882e-15`、Robin
  `4.775e-15`。
- 因此，在本实验矩阵内，PyEIDORS 的总体离散代数精度最好，EIDORS 次之，NGSolve
  第三。经典公式存在跨网格排序反转，全部 case 的 Robin 也因 G4 发生一次反转，
  所以不能把这一结果扩张成“任意 CEM 问题的普遍严格排名”。
- NGSolve 的误差在四级基线、两种公式中稳定为 `3.60e-15–4.22e-15`，约比
  PyEIDORS/EIDORS 高一个数量级，但仍然属于 `float64` 舍入误差尺度，而不是 FEM
  离散误差失控。

![Q0–Q3 三求解器对 QQ 精确解的前向误差与后向残差](../../output/cem_exact_accuracy/cem_exact_accuracy.png)

这张图只使用 Q0–Q3 基线。横轴是同一有理多边形上的嵌套网格级别和节点数；上排是
离数学精确电压的 forward error，下排是候选解满足精确约化系统的 scaled backward
residual。图中的线仅用于连接有序加密级别，不表示误差应当随网格单调变化：这里测量
的是浮点组装/求解误差，而不是连续 PDE 的网格收敛误差。

## “数学精确解”为什么确实是精确的

### 连续 CEM 与两种离散形式

连续 complete electrode model 为

\[
\nabla\!\cdot(\sigma\nabla u)=0 \quad\text{in }\Omega,
\]

\[
u+z_\ell\sigma\partial_nu=U_\ell \quad\text{on }e_\ell,
\qquad
\int_{e_\ell}\sigma\partial_nu\,ds=I_\ell,
\]

非电极边界满足零通量，并要求 \(\sum_\ell I_\ell=0\)。以
\(\sum_\ell U_\ell=0\) 固定电势规范。

对节点 P1 基函数 \(\phi_i\)，定义

\[
K_{ij}=\int_\Omega\sigma\nabla\phi_i\!\cdot\nabla\phi_j\,dx,
\]

\[
(A_R)_{ij}=K_{ij}+\sum_\ell z_\ell^{-1}
\int_{e_\ell}\phi_i\phi_j\,ds,
\quad
C_{i\ell}=-z_\ell^{-1}\int_{e_\ell}\phi_i\,ds,
\quad
D_{\ell\ell}=|e_\ell|/z_\ell.
\]

经典 CEM 直接求解带规范的增广块系统

\[
\begin{bmatrix}
A_R&C&0\\
C^T&D&\mathbf1\\
0&\mathbf1^T&0
\end{bmatrix}
\begin{bmatrix}u\\U\\\lambda\end{bmatrix}
=
\begin{bmatrix}0\\I\\0\end{bmatrix}.
\]

EIDORS 在本实验中确实使用这个经典块解：计时路径为 MATLAB sparse LU；官方
`fwd_solve` 只用于非计时一致性验证。

Robin/Schur 路径先由第一行得到 \(u=-A_R^{-1}CU\)，再消去体内自由度：

\[
S U=I,\qquad S=D-C^TA_R^{-1}C.
\]

取列空间为 \(\mathbf1^\perp\) 的有理基 \(R\)，令 \(U=Ry\)，则

\[
(R^TSR)y=R^TI,\qquad U=Ry.
\]

精确算术下，经典增广块解与 Robin/Schur 解只是同一线性系统的两种等价消元路径。

### 为什么所有离散系数都属于有理数域

固定外域由 16 组整数圆点构造的 32 个端点围成。端点统一除以 `8192`，因此坐标是
二进制 `float64` 可以精确表示的 dyadic rational。每个电极的整条弦长为

\[
|e_\ell|=\frac{5525}{32768}.
\]

Q2/Q3 把每条多边形边线性二分，新增点仍为有理数；Q1/Q3 在径向使用二进制分数层，
所有内部点也仍为 dyadic rational。注意：原始 32 个端点严格共圆，但边中点位于直弦
上；实验域始终是同一个**直边有理 32 边形**，不是 curved-element 真圆。

三角形面积、P1 梯度、体积分、电极边质量矩阵和耦合项由这些有理数经过有限次四则
运算得到。实验中的 \(\sigma\)、\(z\) 和电流也都是有理数，所以完整离散矩阵和 RHS
严格属于 \(\mathbb Q\)。

### 精确求解与认证条件

认证器显式把系数矩阵和全部 16 个 RHS 转换为相同的 SymPy `QQ DomainMatrix`，再使用
多右端 `lu_solve`；它不计算显式逆，也不使用任何 FEM 框架组装的浮点矩阵作为真值。
经典与 Robin 两条精确路径相互独立，每个 case 必须同时满足：

1. 经典增广系统的每个残差分数严格等于 0；
2. Robin 约化系统的每个残差分数严格等于 0；
3. 两条路径得到的 16×16 电极电压有理数矩阵逐项完全相同；
4. 每个 RHS 的电极电压和严格等于 0。

本次 8 个 case 全部通过四项认证。正的 \(\sigma,z\)、连通网格与零均值规范给出唯一
的规范化离散解；精确 LU 成功且残差为严格零。因此这里得到的是有限维有理 P1 系统的
数学精确解，而不是“80–128 位浮点近似解”。

## 受控实验矩阵

四级基线网格定义为

| 级别 | 每条原始边的子边数 | 径向层数 | 节点 | 三角形 |
|---|---:|---:|---:|---:|
| Q0 | 1 | 1 | 33 | 32 |
| Q1 | 1 | 2 | 65 | 96 |
| Q2 | 2 | 2 | 129 | 192 |
| Q3 | 2 | 4 | 257 | 448 |

Q0 的节点集合包含于 Q1，Q1 包含于 Q2，Q2 包含于 Q3。所有级别保持同一外多边形和
同一电极物理端点；边二分时电极积分被拆成等长子边，而不是改变电极覆盖范围。

完整 8 组实验为

| case | 网格 | \(\sigma\) | \(z\) | 激励 | 作用 |
|---|---|---:|---:|---|---|
| G1 | Q0 | 1/4 | 1 | adjacent | 基线加密 |
| G2 | Q1 | 1/4 | 1 | adjacent | 基线加密 |
| G3 | Q2 | 1/4 | 1 | adjacent | 基线加密 |
| G4 | Q1 | 1/4 | 1/8 | adjacent | 低接触阻抗稳健性 |
| G5 | Q1 | 1/4 | 8 | adjacent | 高接触阻抗稳健性 |
| G6 | Q1 | 1 | 1 | adjacent | 高电导率稳健性 |
| G7 | Q1 | 1/4 | 1 | skip-4 | 激励稳健性 |
| G8 | Q3 | 1/4 | 1 | adjacent | 基线加密 |

每组均包含 16 个整数零和 RHS。三个框架导入同一 MAT/MSH 连接关系；导入后重新计算
canonical SHA-256 mesh fingerprint，不匹配就拒绝比较。

## 四个核心指标的含义

### 经典 CEM 真值误差

\[
e_{C}=\frac{\|\widehat U_C-U_{\mathbb Q}\|_F}
{\|U_{\mathbb Q}\|_F}.
\]

这是 forward error，回答经典路径最终电极电压离离散数学精确解有多远。它综合包含
浮点组装、矩阵表示、因子分解和回代产生的误差。

### 经典 residual

把候选电压重新中心化并转换到相同零和基的系数 \(\widehat y\)，计算

\[
\eta_C=
\frac{\|S_{\mathbb Q}\widehat y-R^TI\|_F}
{\|S_{\mathbb Q}\|_F\|\widehat y\|_F+\|R^TI\|_F}.
\]

这是 scaled backward residual，回答候选解把精确离散方程满足到什么程度。它很小表示
候选解是某个邻近线性系统的精确解，但不保证 forward error 必然最小，因为还要经过
\(S^{-1}\) 并受条件数和残差方向影响。

### Robin 真值误差与 Robin residual

定义分别与上面相同，只把候选换成 Robin/Schur 路径的 \(\widehat U_R\)。两种公式都
针对同一个 \(U_{\mathbb Q}\) 和同一个精确约化系统计算，所以可以直接比较。

另外，`classic_robin_relative_l2` 只是同一求解器两条公式的内部差；内部差小不等于
两者到真值的距离小。`voltage_gauge_relative_residual` 衡量零均值规范误差；
`reduced_condition_number_2_estimate` 是精确约化矩阵转为 float64 后的 2-范数条件数
估计，本实验范围为 `18.36–77.74`。

## Q0–Q3 加密结果直接显示稳定部分和排序反转

### 经典 CEM forward error

| 网格 | PyEIDORS | EIDORS | NGSolve | 第一名 |
|---|---:|---:|---:|---|
| Q0 | `5.250e-16` | `5.363e-16` | `3.965e-15` | PyEIDORS |
| Q1 | `4.819e-16` | `4.820e-16` | `3.941e-15` | PyEIDORS |
| Q2 | `4.622e-16` | `1.194e-15` | `3.848e-15` | PyEIDORS |
| Q3 | `5.098e-16` | `4.539e-16` | `4.220e-15` | EIDORS |

四级几何平均分别为 PyEIDORS `4.941e-16`、EIDORS `6.119e-16`、NGSolve
`3.991e-15`。PyEIDORS 赢 3/4，但 Q3 出现真实且可复算的排序反转。因此经典路径不能
声称四级网格上存在统一严格顺序；可以声称 PyEIDORS 在该加密序列上的总体误差最小，
NGSolve 始终明显更高。

### Robin CEM forward error

| 网格 | PyEIDORS | EIDORS | NGSolve | 第一名 |
|---|---:|---:|---:|---|
| Q0 | `4.648e-16` | `6.449e-16` | `3.719e-15` | PyEIDORS |
| Q1 | `3.279e-16` | `4.699e-16` | `3.956e-15` | PyEIDORS |
| Q2 | `3.712e-16` | `1.384e-15` | `3.604e-15` | PyEIDORS |
| Q3 | `3.347e-16` | `4.945e-16` | `4.220e-15` | PyEIDORS |

四级几何平均分别为 PyEIDORS `3.710e-16`、EIDORS `6.749e-16`、NGSolve
`3.867e-15`。Robin 基线四级均保持 `PyEIDORS < EIDORS < NGSolve` 的严格 forward
error 顺序；NGSolve 相对最佳值约为 `8.0×–12.6×`。

## 全部 8 组稳健性结果

### 逐 case 第一名

| case | 经典 CEM 第一名 | Robin 第一名 |
|---|---|---|
| G1/Q0 | PyEIDORS `5.250e-16` | PyEIDORS `4.648e-16` |
| G2/Q1 | PyEIDORS `4.819e-16` | PyEIDORS `3.279e-16` |
| G3/Q2 | PyEIDORS `4.622e-16` | PyEIDORS `3.712e-16` |
| G4/低 z | PyEIDORS `8.125e-16` | EIDORS `8.608e-16` |
| G5/高 z | PyEIDORS `3.451e-16` | PyEIDORS `2.241e-16` |
| G6/高 \(\sigma\) | PyEIDORS `3.254e-16` | PyEIDORS `2.334e-16` |
| G7/skip-4 | EIDORS `6.836e-16` | PyEIDORS `5.766e-16` |
| G8/Q3 | EIDORS `4.539e-16` | PyEIDORS `3.347e-16` |

G2 经典中 PyEIDORS (`4.818697e-16`) 与 EIDORS (`4.819637e-16`) 几乎相同。
精确真值允许给出数值顺序，但这类 `~1e-19` 的差不能被解释为普遍算法优势。

### 聚合误差与 backward residual

| 求解器 | 经典 forward GM | 经典 residual GM | 经典胜场 | Robin forward GM | Robin residual GM | Robin 胜场 |
|---|---:|---:|---:|---:|---:|---:|
| PyEIDORS/DOLFINx | `5.167e-16` | `2.789e-17` | 6/8 | `3.904e-16` | `2.009e-17` | 7/8 |
| EIDORS | `6.160e-16` | `4.073e-17` | 2/8 | `6.181e-16` | `3.624e-17` | 1/8 |
| NGSolve | `4.882e-15` | `3.394e-16` | 0/8 | `4.775e-15` | `3.370e-16` | 0/8 |

forward 与 backward 两类聚合证据方向一致：PyEIDORS 最小，EIDORS 次之，NGSolve
第三。个别 case 的 forward 排名仍可因误差方向和条件数而反转，所以报告同时保留
逐 case 数据、胜场、中位数、最坏误差和原始电压，而不只给一个平均数。

## Classic 与 Robin 的公平速度比较

速度只在**同一求解器内部**比较公式，不把 MATLAB、SciPy 和 NGSolve 的绝对微秒数
包装成跨语言性能排名。每个样本求解相同 16 RHS，并采用：

- 预装配 \(A_R,C,D\)，assembly 单独报告；
- 11 个统计样本，每个样本批量 16 次操作后归一化；
- Classic/Robin 交替先后顺序，runtime/allocator 预热不计时；
- 冷态为“新建本公式状态 + 首次 16-RHS 求解”；
- setup 是同一冷态样本内配对测得的状态构建分量；
- 热态复用只复用本公式自己的 factor state；
- `cross_formulation_cache_reuse=false`，两种公式不共享求解结果或分解。

### 冷态一定比热态复用慢

| 求解器 | 公式 | 冷态 GM（µs） | 热态 GM（µs） | 冷/热 | 8 组最小冷/热 |
|---|---|---:|---:|---:|---:|
| PyEIDORS/DOLFINx | Classic | `784.50` | `34.62` | `22.66×` | `16.42×` |
| PyEIDORS/DOLFINx | Robin | `283.90` | `16.95` | `16.75×` | `10.77×` |
| NGSolve | Classic | `1017.30` | `44.31` | `22.96×` | `10.45×` |
| NGSolve | Robin | `281.24` | `14.94` | `18.82×` | `11.56×` |
| EIDORS | Classic | `384.02` | `36.18` | `10.61×` | `7.09×` |
| EIDORS | Robin | `433.03` | `6.61` | `65.54×` | `52.66×` |

全部 48 个“case × solver × formulation”组合均满足冷态中位数大于热态复用中位数，
因此不存在先前担心的“PyEIDORS 冷态反而更快”异常。

### 同阶段 Robin/Classic 比值

| 求解器 | 冷态 | setup | 热态复用 | Robin 更快 case 数（冷/setup/热） |
|---|---:|---:|---:|---:|
| PyEIDORS/DOLFINx | `0.362` | `0.328` | `0.490` | 8/8/8 |
| NGSolve | `0.276` | `0.264` | `0.337` | 8/8/8 |
| EIDORS | `1.128` | `1.371` | `0.183` | 1/8、0/8、8/8 |

比值小于 1 表示 Robin 更快。PyEIDORS 与 NGSolve 的 Robin 在三个阶段全部更快；
EIDORS 的 Robin setup 通常更贵，但状态建立后 15 维约化回代明显快于经典增广系统。
这些是小型预装配线性代数路径的固定开销结果，不应直接外推到大型 3D 系统。

## 为什么 EIDORS 的 Classic/Robin 内部差常更小

同一求解器的内部差

\[
\|U_R-U_C\|_F/\|U_C\|_F
\]

取决于两条浮点误差向量的大小和相关方向。MATLAB sparse/dense LU、SciPy SuperLU、
NGSolve 组装及不同 Schur 构造顺序会产生不同舍入路径。两条近似可能同向偏离真值，
于是内部差非常小，却不代表任一近似的绝对 forward error 最小。本实验直接用同一个
\(U_{\mathbb Q}\) 测量后，PyEIDORS 在 13/16 个 case×formulation 组合中第一，正好
说明“Classic/Robin 更接近”不能替代“更接近数学精确解”。

## 与真实圆域连续实验如何互相印证

两套实验必须分工，不能用一个排名覆盖另一个：

- 本有理实验拥有**离散代数精度**结论：同一个有限维 P1 问题组装/求解后，谁更接近
  该离散系统的精确解。
- 真实圆域实验拥有**连续总误差**结论：包括几何逼近、P1 空间离散、电极积分、组装
  舍入和线性求解的合成误差。

若连续参考解记为 \(U_*\)，某个网格的理想离散解记为 \(U_h^*\)，求解器输出为
\(\widehat U_{h,s}\)，则

\[
D_h=U_h^*-U_*,\qquad a_{h,s}=\widehat U_{h,s}-U_h^*,
\]

\[
\|\widehat U_{h,s}-U_*\|^2
=\|D_h\|^2+2\langle D_h,a_{h,s}\rangle+\|a_{h,s}\|^2.
\]

连续实验中 \(\|D_h\|\) 约为 `1e-3–1e-2`，而求解器间电压差仅约
`1e-15–1e-14`。交叉项可能因偶然抵消让代数误差更大的求解器出现略小的总误差。
因此，当两套实验排序不同时：

- 讨论“谁更准确地求解同一个离散矩阵”时，以本有理精确实验为准；
- 讨论“当前 P1 网格谁离连续物理解更近”时，以真实圆域总误差为准；
- 不能把有理多边形的排名称为连续真圆 PDE 的真值排名，也不能用连续总误差中
  `1e-15` 级的抵消反过来覆盖本实验的代数精度证据。

## 分享前验证、限制与下一步

状态：**可分享，但必须保留适用范围**。

已完成的验证包括：

- 8 case × 3 solver × 2 formulation = 48 条精度记录，键全部唯一；
- 48 条公平计时记录完整，48/48 满足 `cold median > warm median`；
- 24 份 solver report 均为 real `float64`、16×16 原始电压；
- 每个 case 的三个 solver report 均通过 case/schema、物理参数、P1 次数、原始形状和
  canonical mesh fingerprint 校验；
- 8/8 QQ 真值均通过经典零残差、Robin 零残差和精确电压恒等认证；
- 聚合几何平均和逐 case 第一名已从保存的逐行 metrics 独立复算；
- 最终 PNG 已检查 Times New Roman、对数尺度、图例、标签和裁切。

必须保留的限制：

1. 数学精确性属于有限维有理 P1 离散系统，不属于连续 PDE。
2. 网格规模为 33–257 节点，结论主要揭示当前小系统的舍入路径。
3. 域是固定有理直边 32 边形，不是真圆 curved geometry。
4. 当前为均匀标量有理 \(\sigma\)、统一有理 \(z\) 和两类整数激励。
5. PyEIDORS 与 EIDORS 的差常处在 `1e-16` 量级；总体证据可以排序，但不能包装成
   任意网格、参数、硬件和库版本上的普遍定理。

下一步是用本报告建立的证据层级修订真实圆域报告：同时展示共享连续参考下的名义
总误差排序、参考变体敏感性、求解器间电压分离以及离散误差与代数误差的抵消关系。

## 可复现产物

- [认证 JSON](../../output/cem_exact_accuracy/cem_exact_accuracy.json)
- [精度 CSV](../../output/cem_exact_accuracy/cem_exact_accuracy_metrics.csv)
- [公平计时 CSV](../../output/cem_exact_accuracy/cem_exact_timing_metrics.csv)
- [Q0–Q3 证据图](../../output/cem_exact_accuracy/cem_exact_accuracy.png)
- [suite manifest](../../output/cem_exact_accuracy/suite_manifest.json)
- [实验实现](../../scripts/benchmarks/cem_exact_reference_suite.py)
- [NGSolve 批处理 runner](../../scripts/benchmarks/ngsolve_cem_exact_suite.py)
- [EIDORS 批处理 runner](../../compare_with_Eidors/run_cem_exact_suite.m)
