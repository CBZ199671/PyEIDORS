# 有理圆形 P1 CEM：三种 FEM 求解器的数学精确解、误差与公平计时对比

## 结论先行

本实验在 7 组严格受控的离散 CEM 问题上比较 EIDORS、NGSolve 和
PyEIDORS/DOLFINx。每组内三者使用同一组节点、三角形、边界电极边、材料参数、
接触阻抗、16 个电流 RHS、P1 电势空间、real `float64` 和零均值电极电压规范。
公共网格的 canonical SHA-256 fingerprint 不一致时，认证器会直接拒绝排名。

最重要的结果是：

- PyEIDORS 在经典 CEM 的 7 组中赢 6 组，在 Robin CEM 的 7 组中也赢 6 组；
  EIDORS 分别在 G7 经典和 G4 Robin 中最接近精确解。
- 两种公式都发生了求解器排名翻转，因此不支持“某个求解器在所有设置下永远最精确”
  的普遍结论。
- 按 7 组真值相对误差的几何平均，PyEIDORS 为经典 `5.066e-16`、Robin
  `3.812e-16`；EIDORS 为 `5.520e-16`、`5.414e-16`；NGSolve 为
  `5.012e-15`、`4.932e-15`。
- NGSolve 在这组小型离散系统上的误差大约比 PyEIDORS 高一个数量级，但仍处于
  `float64` 舍入误差量级；这不是 FEM 离散误差或连续 PDE 精度排名。
- EIDORS 的经典/Robin 内部差通常很小，却不等价于真值误差最小。内部一致性与绝对
  精度是两个不同问题。

![三求解器对 QQ 精确解的前向误差与后向残差](../../output/cem_exact_accuracy/cem_exact_accuracy.png)

## 这里的“数学精确解”究竟是什么

### 连续 CEM

对导电区域 \(\Omega\)、电极 \(e_\ell\)、电导率 \(\sigma>0\)、接触阻抗
\(z_\ell>0\)，经典 complete electrode model 为

\[
\nabla\cdot(\sigma\nabla u)=0 \quad\text{in }\Omega,
\]

\[
u+z_\ell\sigma\partial_n u=U_\ell \quad\text{on }e_\ell,
\qquad
\int_{e_\ell}\sigma\partial_n u\,ds=I_\ell,
\]

而非电极边界满足 \(\sigma\partial_nu=0\)。还要求
\(\sum_\ell I_\ell=0\)，并以 \(\sum_\ell U_\ell=0\) 固定电势规范。

### P1 离散经典块系统

对节点 P1 基函数 \(\phi_i\)，定义

\[
K_{ij}=\int_\Omega \sigma\nabla\phi_i\cdot\nabla\phi_j\,dx,
\]

\[
(A_R)_{ij}=K_{ij}+\sum_\ell z_\ell^{-1}
\int_{e_\ell}\phi_i\phi_j\,ds,
\quad
C_{i\ell}=-z_\ell^{-1}\int_{e_\ell}\phi_i\,ds,
\]

\[
D_{\ell\ell}=|e_\ell|/z_\ell.
\]

经典 CEM 的带规范增广系统为

\[
\begin{bmatrix}
A_R&C&0\\
C^T&D&\mathbf 1\\
0&\mathbf 1^T&0
\end{bmatrix}
\begin{bmatrix}u\\U\\\lambda\end{bmatrix}
=
\begin{bmatrix}0\\I\\0\end{bmatrix}.
\]

这里的“经典块解”就是直接求解这个增广系统。EIDORS 本实验并没有绕开经典块解：
计时路径明确使用 MATLAB sparse LU 求解上述增广矩阵；官方 `fwd_solve` 只用于
非计时验证。

### Robin/Schur 形式

由第一行得到 \(u=-A_R^{-1}CU\)，代回电极方程：

\[
S U=I,\qquad S=D-C^TA_R^{-1}C.
\]

取列空间为 \(\mathbf1^\perp\) 的有理基 \(R\)，令 \(U=Ry\)，则

\[
(R^TSR)y=R^TI,
\qquad U=Ry,
\qquad u=-A_R^{-1}CU.
\]

这就是被比较的 Robin transconductance/Schur 算法。精确算术下，它与经典增广
系统是同一个离散数学问题的等价消元；差异只可能来自有限精度组装与线性代数路径。

### 为什么可以得到严格的有理数解

外边界由 16 个整数圆点构造，每个电极使用与圆心向量正交的有理切向弦。所有坐标都
除以 `8192`，因而是二进制 `float64` 可精确表示的 dyadic rational。32 个外边界
端点严格共圆；0、1、2 层内部圆形环由外环乘以有理半径 `1/2`、`3/4` 得到。
这里是共圆直边 P1 多边形，不声称使用了 curved finite elements。

每条电极弦的精确长度为

\[
|e_\ell|=\frac{5525}{4\times8192}=\frac{5525}{32768},
\]

仍为有理数。三角形面积、P1 梯度、体积分、电极边质量矩阵、耦合向量和电极对角项
因此全都属于 \(\mathbb Q\)。7 组中的 \(\sigma\)、\(z\) 和电流同样为有理数。

认证器使用 SymPy DomainMatrix 支持的 \(\mathbb Q\) 精确消元，不把任何 FEM
求解器组装的矩阵作为真值输入，也不把高精度浮点数伪装成精确值。每组必须同时满足：

1. 经典增广系统的每个残差分数分子严格为 0；
2. Robin 约化系统的每个残差分数分子严格为 0；
3. 两条路径得到的 16×16 电极电压分数矩阵逐项完全相同；
4. 每个 RHS 的电极电压和严格为 0。

正的 \(\sigma,z\)、连通网格和零均值规范去除了 CEM 的常数零空间；实际精确逆存在。
因此得到的是该有限维有理 P1 系统的数学精确解。它不是连续圆域 PDE 的解析解，
也不消除 FEM 几何/空间离散误差。

## 实验矩阵

| 案例 | 内部圆环 | 节点 / 三角形 | \(\sigma\) | \(z\) | 激励 |
|---|---:|---:|---:|---:|---|
| G1 | 0 | 33 / 32 | 1/4 | 1 | adjacent |
| G2 | 1 | 65 / 96 | 1/4 | 1 | adjacent |
| G3 | 2 | 97 / 160 | 1/4 | 1 | adjacent |
| G4 | 1 | 65 / 96 | 1/4 | 1/8 | adjacent |
| G5 | 1 | 65 / 96 | 1/4 | 8 | adjacent |
| G6 | 1 | 65 / 96 | 1 | 1 | adjacent |
| G7 | 1 | 65 / 96 | 1/4 | 1 | skip-4 |

每组包含 16 个整数零和 RHS。三个框架均导入同一 MAT/MSH 表示；MAT connectivity
使用 1-based，内存和 Gmsh 使用 0-based，并在导入后重新计算 canonical fingerprint。

## 四类指标分别表示什么

### 1. 经典 CEM 真值误差

\[
e_{\mathrm{classic}}=
\frac{\|\widehat U_{\mathrm{classic}}-U_{\mathbb Q}\|_F}
{\|U_{\mathbb Q}\|_F}.
\]

它是 forward error：该求解器经典路径最终电极电压离离散精确解有多远。它综合反映
`float64` 组装舍入、矩阵表示、因子分解与回代误差。

### 2. 经典 residual

先把候选电压重新中心化到零均值规范，并转换为同一个有理零和基的系数
\(\widehat y\)，再计算

\[
\eta_{\mathrm{classic}}=
\frac{\|S_{\mathbb Q}\widehat y-R^TI\|_F}
{\|S_{\mathbb Q}\|_F\|\widehat y\|_F+\|R^TI\|_F}.
\]

这是 scaled backward residual：候选解把精确离散方程满足到什么程度。它很小表示
候选解是一个邻近线性系统的精确解，但不单独保证 forward error 最小；误差还受
条件数和残差方向影响。

### 3. Robin 真值误差

定义与经典真值误差相同，只是候选 \(\widehat U\) 来自 Robin/Schur 路径。它回答
“Robin 最终电极电压离同一个 \(U_{\mathbb Q}\) 多远”。

### 4. Robin residual

定义与经典 residual 相同，候选来自 Robin 路径。经典和 Robin 都对同一个精确约化
系统计算 residual，因此可以直接比较 backward stability。

另外报告：

- `classic_robin_relative_l2`：同一求解器两条公式的内部差；只表示实现一致性，
  不能代替真值误差；
- `voltage_gauge_relative_residual`：电极电压偏离零均值规范的程度；
- `reduced_condition_number_2_estimate`：精确约化矩阵转为 `float64` 后的 2-范数
  条件数估计，本实验约 `18.36–77.74`。

G4 Robin 是 forward/backward 区别的具体例子：EIDORS 的真值误差
`8.608e-16` 小于 PyEIDORS 的 `9.455e-16`，但 PyEIDORS 的 scaled residual
`2.583e-17` 又小于 EIDORS 的 `3.413e-17`。这不是矛盾；残差经过
\(S^{-1}\) 后，不同方向被放大的程度不同。

## 精度结果

### 逐案例第一名

| 案例 | 经典 CEM 第一名（真值误差） | Robin 第一名（真值误差） |
|---|---|---|
| G1 | PyEIDORS `5.250e-16` | PyEIDORS `4.648e-16` |
| G2 | PyEIDORS `4.819e-16` | PyEIDORS `3.279e-16` |
| G3 | PyEIDORS `3.974e-16` | PyEIDORS `2.689e-16` |
| G4 | PyEIDORS `8.125e-16` | EIDORS `8.608e-16` |
| G5 | PyEIDORS `3.451e-16` | PyEIDORS `2.241e-16` |
| G6 | PyEIDORS `3.254e-16` | PyEIDORS `2.334e-16` |
| G7 | EIDORS `6.836e-16` | PyEIDORS `5.766e-16` |

G2 经典中 PyEIDORS (`4.818697e-16`) 与 EIDORS (`4.819637e-16`) 极接近；
虽然精确分数真值允许给出严格数值顺序，但不应把这种极小差距解释成普遍算法优势。

### 7 组聚合

| 求解器 | 经典几何平均 | 经典胜场 | Robin 几何平均 | Robin 胜场 |
|---|---:|---:|---:|---:|
| PyEIDORS/DOLFINx | `5.066e-16` | 6/7 | `3.812e-16` | 6/7 |
| EIDORS | `5.520e-16` | 1/7 | `5.414e-16` | 1/7 |
| NGSolve | `5.012e-15` | 0/7 | `4.932e-15` | 0/7 |

在本实验矩阵内，可以说 PyEIDORS 的总体 forward error 最小，EIDORS 次之，
NGSolve 第三；不能说这个顺序对所有 CEM 网格和参数普遍成立。此前单个 fan fixture
得到 EIDORS 第一，并不与本结果冲突：单案例结论只属于当时的几何、激励和舍入路径。

## 经典与 Robin 的公平速度对比

速度只在同一求解器内部比较，不把 MATLAB、SciPy 和 NGSolve 的绝对微秒数直接作为
跨语言性能排名。每个样本求解相同的 16 RHS，并采用：

- 预装配 \(A_R,C,D\)，assembly 单独报告；
- 11 次重复，经典/Robin 交替先后顺序；
- 不计时 runtime/allocator 预热；
- 冷态：每次新建本公式因子状态并求解全部 RHS；
- 热态建态：每个公式独立建立一次因子状态；
- 热态复用：仅复用本公式自己的因子状态；
- 明确 `cross_formulation_cache_reuse=false`，经典结果不供 Robin 使用，反之亦然。

下表给出 7 组 `Robin time / classic time` 的几何平均；小于 1 表示 Robin 更快。

| 求解器 | 冷态 | 热态建态 | 热态复用 | Robin 更快的案例数（冷 / 建态 / 复用） |
|---|---:|---:|---:|---:|
| PyEIDORS/DOLFINx | `0.346` | `0.512` | `0.662` | 7 / 7 / 7 |
| NGSolve | `0.373` | `0.352` | `0.599` | 7 / 7 / 5 |
| EIDORS | `1.094` | `1.430` | `0.280` | 2 / 1 / 7 |

因此，在这些小矩阵上：

- PyEIDORS 的 Robin 路径冷态约少 `65%` 时间、热态建态约少 `49%`、热态复用
  约少 `34%`；
- NGSolve 的 Robin 冷态和建态均明显更快，热态复用总体更快，但 G1/G2 因微秒级
  开销和抖动略慢；
- EIDORS 的 Robin 冷态和建态总体略慢，因为需要 `A_R` sparse LU、15 个响应基 RHS
  和一个 dense reduced LU；一旦状态建好，15 维约化回代在 7/7 案例都显著快于经典
  增广系统复用。

所有计时同时保留中位数、IQR、最小值、最大值和原始 11 个样本。小网格微秒结果主要
说明当前实现的固定开销结构，不应外推为大型 2D/3D 稀疏系统的复杂度结论。

## 为什么 EIDORS 的经典/Robin 内部差常常更小

EIDORS 本实验确实使用经典增广块解。其内部差较小的合理解释是 MATLAB 对两个路径
中的 sparse/dense LU、矩阵构造、Helmert 基和回代采用了不同于 SciPy/NGSolve
装配路径的舍入顺序；这会改变两个近似解误差向量的相关性。内部差

\[
\|U_R-U_C\|/\|U_C\|
\]

小，只说明这两个近似向同一方向偏离或都非常接近，不说明它们各自到真值的距离更小。
本实验中 EIDORS 多次拥有最小内部差，而 PyEIDORS 的真值误差却在 12/14 个
案例×公式组合中最小，正好验证了这一点。

## 分享前验证与限制

### 验证结论

状态：**可分享，但必须保留范围说明**。

- 7 个案例、3 个求解器、2 个公式共 42 条精度记录完整；
- 42 条公平计时记录完整，全部为 11 次、16 RHS、交替顺序、无跨公式 cache；
- 每个案例的三个 solver report 均通过 case id、P1、`float64`、物理参数、raw shape
  和 canonical mesh fingerprint 校验；
- 7 个真值均满足 exact classic residual = 0、exact Robin residual = 0、
  exact classic/Robin voltage identity；
- 逐案例排名、胜场、几何平均、中位数和最坏误差均从保存的逐行 metrics 计算；
- 静态证据图已检查字体、对数尺度、图例、裁切和标签重叠。

### 必须保留的限制

1. 数学精确性属于有限维有理 P1 离散系统，不属于连续 PDE。
2. 网格仅为 33/65/97 节点的共圆直边多边形，没有完成连续问题的网格收敛研究。
3. 当前为均匀、标量、有理 \(\sigma\) 和统一有理 \(z\)，每电极一条边。
4. 排名属于这 7 组配置和当前软件/硬件；尤其 PyEIDORS 与 EIDORS 的差常处于
   `1e-16` 量级，不能包装成普遍优越性。
5. 速度是小型预装配线性代数路径的同求解器公式比较，不是三框架总体性能排名。

## 可复现产物

- [认证 JSON](../../output/cem_exact_accuracy/cem_exact_accuracy.json)
- [精度 CSV](../../output/cem_exact_accuracy/cem_exact_accuracy_metrics.csv)
- [公平计时 CSV](../../output/cem_exact_accuracy/cem_exact_timing_metrics.csv)
- [证据图 PNG](../../output/cem_exact_accuracy/cem_exact_accuracy.png)
- [suite manifest](../../output/cem_exact_accuracy/suite_manifest.json)
- [实验实现](../../scripts/benchmarks/cem_exact_reference_suite.py)
- [NGSolve 单案例 runner](../../scripts/benchmarks/ngsolve_cem_exact_case.py)
- [EIDORS runner](../../compare_with_Eidors/compare_cem_formulations.m)

下一步若要研究“连续物理问题谁更准确”，应在同一真实圆域上做网格加密序列，并建立
独立的高阶/外推连续参考解；那将回答 FEM 离散误差，而本报告回答的是固定 P1 离散
系统的组装与线性求解舍入误差。
