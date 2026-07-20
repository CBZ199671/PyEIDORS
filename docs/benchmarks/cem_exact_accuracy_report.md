# 有理 P1 CEM 全因子网格加密实验：三种 FEM 求解器的离散数学精度

数据快照：2026-07-20。实验版本：`cem-exact-rational-suite-v3`。

## 技术摘要

本实验回答一个可以严格认证的问题：当 PyEIDORS/DOLFINx、EIDORS 和 NGSolve
面对**同一个有限维 P1 CEM 离散问题**时，哪个框架计算出的电极电压更接近该离散
系统的数学精确解？

实验采用平衡全因子设计：

\[
4\text{ 个网格}\times 2\text{ 个 }\sigma
\times 3\text{ 个 }z\times 2\text{ 种注流}=48\text{ 个 case}.
\]

每个 case 均比较 3 个求解器和 Classic/Robin 两种形式，因此共有 288 条精度记录和
288 条公平计时记录。三个框架逐 case 使用相同节点、三角形、电极边、物理参数、
16 个 RHS、P1 次数、零均值规范和 real `float64`。

主要结论如下：

- **总体离散代数精度仍为 PyEIDORS 第一、EIDORS 第二、NGSolve 第三。** Classic
  forward error 几何均值依次为 `6.202e-16`、`7.275e-16`、`5.564e-15`；
  Robin 依次为 `4.693e-16`、`7.262e-16`、`5.439e-15`。
- **PyEIDORS 的优势不是 48/48 无条件成立。** Classic 中 PyEIDORS 赢 29 例、
  EIDORS 赢 19 例；Robin 中分别赢 40 和 8 例。NGSolve 在两种形式的 48 例中
  始终第三。
- **网格和物理设置确实产生交互。** Classic 的 Q0、Q3 分层几何均值由 EIDORS
  略小；Q1、Q2 由 PyEIDORS 更小。Robin 的 Q0 由 EIDORS 略小，Q1–Q3 由
  PyEIDORS 更小。低接触阻抗设置 S02/S08 的四网格边际平均也由 EIDORS 最小。
- **Robin 对 PyEIDORS 的绝对精度改善最稳定。** PyEIDORS 的 Robin/Classic
  forward-error 几何均值比为 `0.757`，37/48 例 Robin 更小；EIDORS 为
  `0.998`、22/48；NGSolve 为 `0.977`、30/48。
- **所有真值均为严格有理数解。** 48/48 case 的 Classic 残差、Robin 残差、
  两形式电压差和电压规范均在 \(\mathbb Q\) 中逐项严格为零。

因此可以向安迪教授表述为：在本次 48 例平衡有理离散实验矩阵中，PyEIDORS 的总体
离散代数精度最优，EIDORS 次之，NGSolve 第三；但 PyEIDORS 与 EIDORS 会随网格和
设置发生小幅排序反转，不能扩张成任意 CEM 问题上的逐例定理。

## 两张图分别回答“有序加密”和“全因子交互”

![Q0–Q3 基线对 QQ 精确解的前向误差与后向残差](../../output/cem_exact_accuracy/cem_exact_accuracy.png)

上图只取基线设置 S01。横轴为同一有理 32 边形上的 Q0–Q3 嵌套网格；上排是到
数学精确电压的 forward error，下排是候选电压对精确约化系统的 scaled backward
residual。连线只表示有序网格级别，不暗示舍入误差应随网格单调收敛。

![48 例网格—设置全因子相对误差热图](../../output/cem_exact_accuracy/cem_exact_factorial_heatmap.png)

热图的每个单元固定一个“网格、\(\sigma\)、\(z\)、drive、formulation”，颜色为

\[
\log_{10}\!\left(\frac{e_{\mathrm{solver}}}{\min_s e_s}\right).
\]

零表示该单元的最佳求解器，颜色越深表示相对最佳值越大。该图使用同一色标，完整展示
12 个设置与 4 个网格的交互；它表达相对排名，不替代绝对误差表。

## “数学精确解”为什么确实是精确的

### Classic CEM 与 Robin/Schur CEM 是同一离散问题

连续 complete electrode model 为

\[
\nabla\!\cdot(\sigma\nabla u)=0\quad\text{in }\Omega,
\]

\[
u+z_\ell\sigma\partial_nu=U_\ell\quad\text{on }e_\ell,\qquad
\int_{e_\ell}\sigma\partial_nu\,ds=I_\ell,
\]

非电极边界满足零通量，\(\sum_\ell I_\ell=0\)，并以
\(\sum_\ell U_\ell=0\) 固定电势规范。

对节点 P1 基函数 \(\phi_i\)，令

\[
(A_R)_{ij}=\int_\Omega\sigma\nabla\phi_i\!\cdot\nabla\phi_j\,dx
+\sum_\ell z_\ell^{-1}\int_{e_\ell}\phi_i\phi_j\,ds,
\]

\[
C_{i\ell}=-z_\ell^{-1}\int_{e_\ell}\phi_i\,ds,\qquad
D_{\ell\ell}=|e_\ell|/z_\ell.
\]

Classic 路径直接求解带规范的增广块系统

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

Robin/Schur 路径消去体内自由度，得到

\[
S=D-C^TA_R^{-1}C,\qquad SU=I.
\]

取列空间为 \(\mathbf1^\perp\) 的有理基 \(R\)，令 \(U=Ry\)，则

\[
(R^TSR)y=R^TI.
\]

精确算术下，两条路径只是同一有限维方程的两种等价消元方式。

### 所有离散系数都严格属于 \(\mathbb Q\)

外边界的 32 个端点由整数圆点构造并统一除以 8192，坐标均为 dyadic rational，
且可被 binary64 精确表示。每条电极弦长为

\[
|e_\ell|=\frac{5525}{32768}.
\]

Q0–Q3 仅使用二分边和二进制分数径向层，新增坐标仍为有理数。三角形面积、P1 梯度、
体积分、电极边质量矩阵和耦合项均由有限次有理四则运算得到；本实验中的
\(\sigma\)、\(z\) 和电流也都是有理数。因此完整离散矩阵和 RHS 严格属于
\(\mathbb Q\)，没有“高精度浮点近似真值”这一层误差。

### 24 个精确基解为何可代表 48 个 case

系数矩阵只依赖 \((Q,\sigma,z)\)，不依赖注流模式；drive 只改变 RHS。16 电极的
零和电流空间维数为 15。认证器对每个不同 \((Q,\sigma,z)\)：

1. 构造 15 列有理零和基 \(R\)；
2. 分别用 SymPy `QQ DomainMatrix.lu_solve` 求 Classic 增广基解和 Robin 约化基解；
3. 对任意实际电流 \(I=R\alpha\)，用精确有理线性组合恢复实际解；
4. 对恢复后的实际 16 个 RHS 再逐项检查 Classic 残差、Robin 残差、两形式恒等和规范。

因此缓存只是利用线性性避免同一矩阵重复 LU，不是跨矩阵借用真值。本次运行严格记录
`misses=24`、`hits=24`、`currsize=24`，每个基解为 15 RHS。

正的 \(\sigma,z\)、连通网格和零均值规范保证规范化离散解唯一。精确 LU 成功且所有
认证等式在 \(\mathbb Q\) 中严格为零，所以这里得到的是有限维有理 P1 系统的数学精确解。

## 48 例平衡实验矩阵

四级网格保持同一外多边形与电极物理端点：

| 网格 | 原始边子边数 | 径向层数 | 节点 | 三角形 |
|---|---:|---:|---:|---:|
| Q0 | 1 | 1 | 33 | 32 |
| Q1 | 1 | 2 | 65 | 96 |
| Q2 | 2 | 2 | 129 | 192 |
| Q3 | 2 | 4 | 257 | 448 |

节点集合严格嵌套：Q0 \(\subset\) Q1 \(\subset\) Q2 \(\subset\) Q3。12 个设置为：

| 设置 | \(\sigma\) | \(z\) | 注流 |
|---|---:|---:|---|
| S01 | 1/4 | 1 | adjacent |
| S02 | 1/4 | 1/8 | adjacent |
| S03 | 1/4 | 8 | adjacent |
| S04 | 1 | 1 | adjacent |
| S05 | 1 | 1/8 | adjacent |
| S06 | 1 | 8 | adjacent |
| S07 | 1/4 | 1 | skip-4 |
| S08 | 1/4 | 1/8 | skip-4 |
| S09 | 1/4 | 8 | skip-4 |
| S10 | 1 | 1 | skip-4 |
| S11 | 1 | 1/8 | skip-4 |
| S12 | 1 | 8 | skip-4 |

每个设置在 Q0–Q3 上各出现一次，所以网格边际含 12 个等权 case，设置边际含 4 个
等权 case；不存在旧 8-case 设计中“Q1 设置更多”导致的聚合权重偏斜。

## 四个核心指标的含义

\[
e_C=\frac{\|\widehat U_C-U_{\mathbb Q}\|_F}{\|U_{\mathbb Q}\|_F},
\qquad
e_R=\frac{\|\widehat U_R-U_{\mathbb Q}\|_F}{\|U_{\mathbb Q}\|_F}.
\]

Classic/Robin 真值误差是 forward error，回答最终电极电压离同一离散数学精确解有
多远，综合包含浮点坐标/组装、矩阵表示、因子分解和回代误差。

将候选电压中心化并转换为零和基系数 \(\widehat y\)，计算

\[
\eta=
\frac{\|S_{\mathbb Q}\widehat y-R^TI\|_F}
{\|S_{\mathbb Q}\|_F\|\widehat y\|_F+\|R^TI\|_F}.
\]

对 Classic/Robin 候选分别得到相应 residual。scaled backward residual 衡量候选解把
精确离散方程满足到什么程度；它小不必然保证 forward error 最小，因为还受条件数和
残差方向影响。本实验约化矩阵的 float64 2-范数条件数估计为 `16.57–97.65`。
`classic_robin_relative_l2` 只衡量两条浮点路径的内部差，不等于到真值的距离。

## 总体结果稳定，但 PyEIDORS 与 EIDORS 存在逐例反转

| 求解器 | Classic forward GM | Classic residual GM | Classic 胜场 | Robin forward GM | Robin residual GM | Robin 胜场 |
|---|---:|---:|---:|---:|---:|---:|
| PyEIDORS/DOLFINx | `6.202e-16` | `3.216e-17` | 29/48 | `4.693e-16` | `2.487e-17` | 40/48 |
| EIDORS | `7.275e-16` | `4.383e-17` | 19/48 | `7.262e-16` | `4.158e-17` | 8/48 |
| NGSolve | `5.564e-15` | `3.403e-16` | 0/48 | `5.439e-15` | `3.362e-16` | 0/48 |

288 条记录只出现两种严格顺序：Classic 中
`PyEIDORS < EIDORS < NGSolve` 为 29 例，反转前两名为 19 例；Robin 分别为
40 和 8 例。NGSolve 在 96 个“case × formulation”比较中均为第三。

### 分网格结果显示交互，而不是单调网格趋势

| 形式 | 网格 | PyEIDORS GM | EIDORS GM | NGSolve GM | 胜场（Py/EID/NG） |
|---|---|---:|---:|---:|---:|
| Classic | Q0 | `6.270e-16` | `6.008e-16` | `5.540e-15` | 6/6/0 |
| Classic | Q1 | `5.258e-16` | `5.302e-16` | `5.670e-15` | 7/5/0 |
| Classic | Q2 | `6.439e-16` | `1.292e-15` | `5.617e-15` | 12/0/0 |
| Classic | Q3 | `6.967e-16` | `6.803e-16` | `5.431e-15` | 4/8/0 |
| Robin | Q0 | `5.834e-16` | `5.757e-16` | `5.395e-15` | 8/4/0 |
| Robin | Q1 | `3.586e-16` | `4.980e-16` | `5.621e-15` | 10/2/0 |
| Robin | Q2 | `5.198e-16` | `1.370e-15` | `5.333e-15` | 12/0/0 |
| Robin | Q3 | `4.460e-16` | `7.080e-16` | `5.411e-15` | 10/2/0 |

Q2 对 PyEIDORS 最有利；Q0 和 Classic Q3 则显示 EIDORS 的分层几何均值可略小。
这说明网格细化不会让舍入误差按固定方向单调变化。

### 分设置结果定位低接触阻抗反转

下表为每个设置跨 Q0–Q3 的 forward-error 几何均值；“第一”按未舍入数值判断。

| 设置 | Classic Py | Classic EID | Classic NG | 第一 | Robin Py | Robin EID | Robin NG | 第一 |
|---|---:|---:|---:|---|---:|---:|---:|---|
| S01 | `4.941e-16` | `6.119e-16` | `3.991e-15` | Py | `3.710e-16` | `6.749e-16` | `3.867e-15` | Py |
| S02 | `1.328e-15` | `1.109e-15` | `2.160e-14` | EID | `1.429e-15` | `1.021e-15` | `2.182e-14` | EID |
| S03 | `2.997e-16` | `4.800e-16` | `3.026e-15` | Py | `2.120e-16` | `4.818e-16` | `3.009e-15` | Py |
| S04 | `3.146e-16` | `4.682e-16` | `3.165e-15` | Py | `2.041e-16` | `5.045e-16` | `3.104e-15` | Py |
| S05 | `7.888e-16` | `7.970e-16` | `5.714e-15` | Py | `7.427e-16` | `8.697e-16` | `5.569e-15` | Py |
| S06 | `3.020e-16` | `4.579e-16` | `3.028e-15` | Py | `2.095e-16` | `4.329e-16` | `2.962e-15` | Py |
| S07 | `1.011e-15` | `1.039e-15` | `5.822e-15` | Py | `6.522e-16` | `1.073e-15` | `5.275e-15` | Py |
| S08 | `1.652e-15` | `1.573e-15` | `3.407e-14` | EID | `2.041e-15` | `1.248e-15` | `3.449e-14` | EID |
| S09 | `4.816e-16` | `5.554e-16` | `3.146e-15` | Py | `2.735e-16` | `5.507e-16` | `3.132e-15` | Py |
| S10 | `5.008e-16` | `6.571e-16` | `3.436e-15` | Py | `3.154e-16` | `6.728e-16` | `3.380e-15` | Py |
| S11 | `1.351e-15` | `1.468e-15` | `9.362e-15` | Py | `1.234e-15` | `1.581e-15` | `9.001e-15` | Py |
| S12 | `4.037e-16` | `4.507e-16` | `3.071e-15` | Py | `2.257e-16` | `4.348e-16` | `2.970e-15` | Py |

S02/S08 都是 \(\sigma=1/4,z=1/8\)，区别只在 adjacent/skip-4。低 \(z\) 使系统
误差整体变大，并使 EIDORS 的四网格边际平均优于 PyEIDORS。

## Robin 与 Classic 的绝对精度必须对同一真值比较

| 求解器 | Robin/Classic forward GM | Robin 更小 case | Classic/Robin 内部差 GM |
|---|---:|---:|---:|
| PyEIDORS/DOLFINx | `0.757` | 37/48 | `6.427e-16` |
| EIDORS | `0.998` | 22/48 | `4.570e-16` |
| NGSolve | `0.977` | 30/48 | `6.294e-16` |

EIDORS 的两条形式内部差最小，但它到 \(U_{\mathbb Q}\) 的总体误差并非最小。两条
浮点误差向量可以相关并同向偏离真值，内部相减会抵消共同误差；所以内部差不能代替
绝对真值误差。

## Classic 与 Robin 的公平速度比较

速度只在**同一求解器内部**比较。每个样本求解相同 16 RHS；预装配 \(A_R,C,D\)；
11 个样本批量 16 次后归一化；两形式交替先后；冷态为新建本形式状态加首次求解；
热态只复用本形式自己的 factor state；`cross_formulation_cache_reuse=false`。

| 求解器 | 形式 | 冷态 GM（µs） | 热态 GM（µs） | 冷/热 GM | 最小冷/热 |
|---|---|---:|---:|---:|---:|
| PyEIDORS/DOLFINx | Classic | `828.58` | `36.31` | `22.82×` | `10.78×` |
| PyEIDORS/DOLFINx | Robin | `297.18` | `16.61` | `17.89×` | `7.89×` |
| EIDORS | Classic | `377.52` | `37.48` | `10.07×` | `7.87×` |
| EIDORS | Robin | `430.82` | `6.03` | `71.46×` | `31.80×` |
| NGSolve | Classic | `1025.61` | `57.25` | `17.92×` | `9.46×` |
| NGSolve | Robin | `313.47` | `17.16` | `18.27×` | `6.21×` |

288/288 条计时记录均满足冷态中位数大于热态中位数。同阶段 Robin/Classic 比值为：

| 求解器 | 冷态 | setup | 热态 | Robin 更快 case（冷/setup/热） |
|---|---:|---:|---:|---:|
| PyEIDORS/DOLFINx | `0.359` | `0.332` | `0.457` | 48/48、48/48、48/48 |
| EIDORS | `1.141` | `1.405` | `0.161` | 0/48、0/48、48/48 |
| NGSolve | `0.306` | `0.301` | `0.300` | 48/48、48/48、48/48 |

PyEIDORS 和 NGSolve 的 Robin 在三个阶段全部更快；EIDORS 的 Robin 建态更贵，
但复用后 15 维约化回代明显更快。这是小型预装配路径结果，不能直接外推到大型 3D。

## 与真实圆域连续实验的证据边界

本有理实验拥有**离散代数精度**结论；真实圆域实验拥有**连续总误差**结论。若
\(U_*\) 是连续解，\(U_h^*\) 是理想离散解，\(\widehat U_{h,s}\) 是求解器输出，则

\[
D_h=U_h^*-U_*,\qquad a_{h,s}=\widehat U_{h,s}-U_h^*,
\]

\[
\|\widehat U_{h,s}-U_*\|^2
=\|D_h\|^2+2\langle D_h,a_{h,s}\rangle+\|a_{h,s}\|^2.
\]

连续离散误差比求解器代数差大许多，交叉项可能产生抵消。因此比较“谁更准确地求解
同一个离散问题”以本 QQ 实验为准；比较“谁离连续物理解更近”以连续参考实验为准。

## 分享前验证、限制与下一步

验证状态：**可分享，但必须保留适用范围**。

已独立复核：

- 48 case、288 精度记录、288 计时记录，复合键完整且唯一；
- 48/48 QQ 真值通过 Classic/Robin 零残差和精确电压恒等认证；
- QQ 缓存为 24 miss、24 hit，drive 不在矩阵缓存键中；
- 每个 case 的三个报告具有相同 canonical mesh fingerprint；
- 三个框架均为 real `float64`、P1、16×16 原始电极电压；
- 288/288 计时记录满足 `cold median > warm median > 0`；
- 总体、胜场、网格和设置边际均从逐行 JSON 独立复算；
- 两张 PNG 已检查字体、尺度、标签、色标和裁切。

必须保留的限制：

1. 数学精确性属于有限维有理 P1 系统，不属于连续 PDE。
2. 网格为 33–257 节点，主要揭示当前小系统的舍入路径。
3. 域是固定直边有理 32 边形，不是真圆 curved geometry。
4. 当前只覆盖均匀标量 \(\sigma\)、统一 \(z\) 与两类注流。
5. PyEIDORS 与 EIDORS 的差常处于 \(10^{-16}\) 尺度，不能外推为任意环境上的定理。

下一步可加入更多预先规定的有理 \(\sigma,z\) 水平、非均匀有理单元电导率和不同
电极数，同时保持逐 case 的共享网格与 QQ 认证。

## 可复现产物

- [认证 JSON](../../output/cem_exact_accuracy/cem_exact_accuracy.json)
- [精度 CSV](../../output/cem_exact_accuracy/cem_exact_accuracy_metrics.csv)
- [公平计时 CSV](../../output/cem_exact_accuracy/cem_exact_timing_metrics.csv)
- [Q0–Q3 基线图](../../output/cem_exact_accuracy/cem_exact_accuracy.png)
- [48 例全因子热图](../../output/cem_exact_accuracy/cem_exact_factorial_heatmap.png)
- [suite manifest](../../output/cem_exact_accuracy/suite_manifest.json)
- [实验实现](../../scripts/benchmarks/cem_exact_reference_suite.py)
- [NGSolve 批处理 runner](../../scripts/benchmarks/ngsolve_cem_exact_suite.py)
- [EIDORS 批处理 runner](../../compare_with_Eidors/run_cem_exact_suite.m)
