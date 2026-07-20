# 有理 CEM 扩展与低阻抗误差归因：三种 FEM 求解器的离散数学精度

数据快照：2026-07-20。扩展套件：`cem-exact-extension-v1`。低阻抗归因：
`cem-low-z-attribution-v1`。

## 技术摘要

本轮完成了两个预先规定的问题。

1. 在新增的有理参数、非均匀单元电导率、8 电极和更大 Q4 网格上，原有总体顺序
   是否保持？
2. 低接触阻抗下 PyEIDORS 与 EIDORS 的排序反转，主要来自组装实现还是线性代数
   后端？

38 个预注册 case 产生 228 条精度记录和 228 条公平计时记录。每个 case 的三个 FEM
框架共享节点、三角形、电极边、逐单元电导率、电流、规范、P1 次数和 real
`float64`，Classic 与 Robin 也共享同一组装输入。

结论不是“所有 case 都有一个不变排序”：

- 全部 38 例中，PyEIDORS/DOLFINx 在 Classic 赢 `27/38`，在 Robin 赢 `29/38`；
  EIDORS 分别赢 `11/38` 和 `9/38`；NGSolve 为 `0/38` 和 `0/38`。
- 更大的 Q4 网格给出更强、但仍限于本实验矩阵的证据：6 个 Q4 case 在 Classic 和
  Robin 下均为 `PyEIDORS/DOLFINx < EIDORS < NGSolve`，即 PyEIDORS `6/6`
  第一。
- 8 电极族是明确反例。Classic 的几何均值由 EIDORS 更小；Robin 胜场为
  PyEIDORS/EIDORS `4/4`，几何均值仍由 EIDORS 略小。因此不能把 Q4 结果外推成
  任意电极数上的逐 case 定理。
- 4 个预注册低阻抗/高敏感 case 全部判为“组装实现差异主导”。这里的“组装实现”
  是 PyEIDORS 与 EIDORS 在相同 SciPy/MATLAB 后端上的配对差异；其效应为
  `0.284–0.481` decade，线性后端为 `0.088–0.221` decade，纯前向/反向累加顺序
  为 `0.081–0.213` decade。

因此，对安迪教授最严谨的表述是：**在当前扩展有理离散矩阵中，PyEIDORS 的总体
舍入精度最优，NGSolve 稳定第三；更大 Q4 网格保持 PyEIDORS、EIDORS、NGSolve
的顺序，但跨电极数和全部物理设置不存在逐例普适排序。低阻抗反转主要与更广义的
组装/矩阵表示实现有关，不是 MATLAB sparse LU 与 SciPy SuperLU 的单独选择，也
不能只归因于局部贡献的累加先后。**

## 38 例扩展设计覆盖四类新证据

| 家族 | case 数 | 网格 | 设置 | 目的 |
|---|---:|---|---|---|
| `range` | 16 | Q0、Q2 | `sigma={1/8,4}` 或 `z={1/32,32}`；adjacent/skip-4 | 扩展有理参数范围 |
| `heterogeneous` | 8 | Q0、Q2 | 左半域 `sigma=1/4`、右半域 `sigma=1`；`z={1/8,1}` | 非均匀有理 DG0 单元电导率 |
| `electrode_count` | 8 | Q0、Q2 | 8 电极；`z={1/8,1}`；adjacent/skip-2 | 改变电极数和零和空间维数 |
| `large_q4` | 6 | Q4 | 16 电极；`z={1/32,1/8,1}`；adjacent/skip-4 | 更大有理网格 |

网格规模为：Q0 `33` 节点/`32` 三角形，Q2 `129` 节点/`192` 三角形，Q4
`513` 节点/`896` 三角形。外边界均为同一个有理 32 边形；Q2、Q4 只做有理二分，
不是重新拟合一个不同的圆。

每个 case 的逐单元电导率以源单元顺序生成 SHA-256。DOLFINx 通过
`original_cell_index` 映射回源顺序；NGSolve 按材料名而不是导入顺序映射；EIDORS
直接读同一 MAT 中的 `elem_data`。只有节点/单元/电极/电导率摘要全部相同，case
才进入排名。

## Q4 保持总体顺序，但 8 电极族否定普适排序

![38 个预注册 case 对逐 case QQ 真值的误差](../../output/cem_exact_extension/cem_exact_extension_accuracy.png)

图中横轴只是预注册 case 索引，不表示连续网格加密轨迹；纵轴是相对于各 case 自己
QQ 真值的相对 Frobenius 误差。曲线的作用是展示反转与数量级，不用于声称舍入误差
应随 case 索引单调变化。

### 全部 38 例

| 求解器 | Classic forward GM | Classic residual GM | 胜场 | Robin forward GM | Robin residual GM | 胜场 |
|---|---:|---:|---:|---:|---:|---:|
| PyEIDORS/DOLFINx | `1.109e-15` | `5.419e-17` | 27/38 | `1.060e-15` | `5.170e-17` | 29/38 |
| EIDORS | `1.694e-15` | `8.573e-17` | 11/38 | `1.735e-15` | `8.606e-17` | 9/38 |
| NGSolve | `1.120e-14` | `5.834e-16` | 0/38 | `1.075e-14` | `5.673e-16` | 0/38 |

### 更大的 Q4 网格

| 求解器 | Classic GM | Classic 胜场 | Robin GM | Robin 胜场 |
|---|---:|---:|---:|---:|
| PyEIDORS/DOLFINx | `2.062e-15` | 6/6 | `2.733e-15` | 6/6 |
| EIDORS | `6.513e-15` | 0/6 | `6.622e-15` | 0/6 |
| NGSolve | `2.722e-14` | 0/6 | `2.489e-14` | 0/6 |

6 个 Q4 case 的两个形式均出现同一严格顺序，而不只是几何均值顺序相同。这回答了
“更大有理网格是否保持总体顺序”：**在预注册 Q4 范围内保持；在全部 38 例与不同
电极数之间不保持普适逐例顺序。**

### 分家族反例与稳健性

| 家族 | 形式 | PyEIDORS GM / 胜场 | EIDORS GM / 胜场 | NGSolve GM / 胜场 |
|---|---|---:|---:|---:|
| range | Classic | `6.655e-16` / 13 | `1.074e-15` / 3 | `9.170e-15` / 0 |
| range | Robin | `6.626e-16` / 14 | `1.130e-15` / 2 | `8.925e-15` / 0 |
| heterogeneous | Classic | `1.259e-15` / 5 | `1.865e-15` / 3 | `8.080e-15` / 0 |
| heterogeneous | Robin | `1.033e-15` / 5 | `1.994e-15` / 3 | `7.728e-15` / 0 |
| 8 electrodes | Classic | `1.707e-15` / 3 | `1.392e-15` / 5 | `1.190e-14` / 0 |
| 8 electrodes | Robin | `1.365e-15` / 4 | `1.304e-15` / 4 | `1.157e-14` / 0 |

异质电导率没有改变总体第一/第三的结论，但 8 电极族显示 PyEIDORS 与 EIDORS 的
微小舍入差异会随系统维数、基排列和消元结构改变。这个反例是报告中必须保留的，
因为它阻止了不受数据支持的“求解器固有精度定理”。

## QQ 真值为什么是有限维离散系统的数学精确解

### Classic 与 Robin 是同一离散方程的两种消元

对 P1 基函数，记

\[
(A_R)_{ij}=\int_\Omega \sigma\nabla\phi_i\cdot\nabla\phi_j\,dx
+\sum_\ell z_\ell^{-1}\int_{e_\ell}\phi_i\phi_j\,ds,
\]

\[
C_{i\ell}=-z_\ell^{-1}\int_{e_\ell}\phi_i\,ds,
\qquad D_{\ell\ell}=|e_\ell|/z_\ell.
\]

Classic 直接解带零均值规范的增广块系统；Robin 先消去体节点未知量，形成

\[
S=D-C^TA_R^{-1}C.
\]

以列空间为 \(\mathbf1^\perp\) 的有理基 \(R\) 写成 \(U=Ry\)，最终约化系统为

\[
(R^TSR)y=R^TI.
\]

在精确算术下，两条路径必须得到完全相同的电极电压。

### 为什么矩阵严格属于有理数域

- 边界节点是整数圆点统一除以 8192，Q0/Q2/Q4 新坐标仍是 dyadic rational；
- P1 三角形梯度、面积、边界质量、耦合和电极长度都由有限次有理四则运算得到；
- `sigma`、逐单元异质 `sigma`、`z` 和电流均为有理数；
- 因而 Classic 增广矩阵、Robin Schur 矩阵和 RHS 全部严格属于 \(\mathbb Q\)。

Q0/Q2 使用 SymPy `DomainMatrix` 的 `QQ` 多 RHS `lu_solve`。Q4 的 Classic 系统达到
约 `530×530`，SymPy 路径不具备可恢复性能，因此改用隔离的
`python-flint 0.6.0` `fmpq_mat.solve`；这仍是任意精度有理数运算，不是更高精度浮点。
FLINT 写入原子 truth cache 后，主 Nix/Python 进程把结果重新载入为 SymPy 有理数，
再次逐项验证：

1. Classic 方程残差严格为零；
2. Robin 约化方程残差严格为零；
3. 两条路径的电极电压分数逐项完全相同；
4. 电压零均值规范严格满足。

正的电导率、正的接触阻抗、连通网格和零均值规范保证解唯一。因此这不是“80 位
浮点近似值”，而是该有限维 P1 离散矩阵在 \(\mathbb Q\) 上的唯一数学精确解。
它能认证组装与线性求解的舍入误差，但不等于真实圆域连续 PDE 的解析真值。

## 四个精度指标分别回答什么

对求解器给出的 Classic/Robin 电极电压 \(\widehat U_C,\widehat U_R\)，QQ 真值为
\(U_{\mathbb Q}\)：

\[
e_C=\frac{\|\widehat U_C-U_{\mathbb Q}\|_F}{\|U_{\mathbb Q}\|_F},
\qquad
e_R=\frac{\|\widehat U_R-U_{\mathbb Q}\|_F}{\|U_{\mathbb Q}\|_F}.
\]

- `Classic 真值误差`：Classic 浮点路径离离散精确电压多远；
- `Robin 真值误差`：Robin/Schur 浮点路径离同一离散精确电压多远；
- `Classic residual`：Classic 候选电压代回精确约化系统后的 scaled backward
  residual；
- `Robin residual`：Robin 候选电压代回同一精确约化系统后的 scaled backward
  residual。

scaled backward residual 使用

\[
\eta=\frac{\|S_{\mathbb Q}\widehat y-R^TI\|_F}
{\|S_{\mathbb Q}\|_F\|\widehat y\|_F+\|R^TI\|_F}.
\]

forward error 回答“答案离真值多远”；backward residual 回答“候选答案精确满足了一个
多接近原方程的系统”。残差更小不保证 forward error 必然更小，因为条件数和残差
方向也会放大或抑制误差。`Classic vs Robin` 的内部差也不能代替真值误差：两条路径
可以彼此非常接近，却同时偏离 QQ 真值。

## 低阻抗反转主要与广义组装实现有关

![低阻抗 case 的配对误差效应](../../output/cem_exact_extension/cem_low_z_attribution.png)

柱高是对 `log10(真值相对误差)` 的配对变化，单位为 decade；`0.301` decade 约等于
误差相差 2 倍。四个 case 覆盖 Q0 低 `z`、Q2 低 `z`、Q4 低 `z` 和 Q2 异质电导率。

| case | 组装实现 | 线性后端 | 纯累加顺序 | 判定 |
|---|---:|---:|---:|---|
| X05 | `0.284` | `0.093` | `0.184` | 组装实现主导 |
| X13 | `0.346` | `0.088` | `0.086` | 组装实现主导 |
| X33 | `0.317` | `0.151` | `0.081` | 组装实现主导 |
| X21 | `0.481` | `0.221` | `0.213` | 组装实现主导 |

归因实验不再比较“各框架原生结果”这一混合因素，而是保存各框架实际组装的
`A_R,C,D,I` 块，验证 canonical CSC SHA-256，然后交叉求解：

- 每套组装矩阵都分别通过 SciPy SuperLU 和 MATLAB sparse LU；
- 组装效应主指标只用 PyEIDORS↔EIDORS 配对，避免 NGSolve 的整体数量级差把
  “组装范围”人为放大；三者全范围仅作次要敏感性指标；
- 后端效应在固定 PyEIDORS/EIDORS 组装下比较 SciPy↔MATLAB；
- 纯顺序效应固定相同的精确局部贡献，只改变 forward/reverse 累加次序；
- Classic/Robin 两种形式的效应先分别计算，再对形式取平均；至少 3/4 case 超过
  其他效应 `0.05` decade 才允许下结论。

4/4 case 均满足阈值。因此，反转主要不是“EIDORS 使用 MATLAB LU 而 PyEIDORS
使用 SuperLU”导致的，也不是简单把单元循环倒序即可解释。更合理的范围是：不同
FEM 框架的局部积分核、单元/DOF 排列、稀疏插入与归并、矩阵存储转换和消元前表示
共同形成的组装实现差异。

这个结论是受控关联归因，不是对每一种低级操作完成了因果 ANOVA。若要进一步拆分，
需要在单一框架中逐项冻结 DOF 排列、积分值、稀疏插入顺序和矩阵格式。

## Classic 与 Robin 的计时采用配对冷态/热态协议

计时范围从已经共享并验证的 `A_R,C,D` 块开始，不包含网格生成和 FEM 积分。每个
case、求解器、形式均做 11 个重复；每个样本内部重复 16 次并取单次平均，Classic
与 Robin 交替先后。运行时预热不计时，两种形式不共享因子分解 cache。

- `cold`：从新状态构建、因子分解到求解该 case 全部 RHS；
- `setup`：冷态中的状态构建/因子分解部分；
- `warm reuse`：只复用本形式自己的已建状态并再次求解全部 RHS；
- `Robin/Classic < 1` 表示同一阶段 Robin 更快。

| 求解器 | Classic cold GM | Robin cold GM | R/C cold | Classic warm GM | Robin warm GM | R/C warm |
|---|---:|---:|---:|---:|---:|---:|
| PyEIDORS/DOLFINx | `875.5 us` | `316.7 us` | `0.362` | `34.41 us` | `17.69 us` | `0.514` |
| NGSolve | `1.118 ms` | `571.8 us` | `0.511` | `47.37 us` | `25.61 us` | `0.541` |
| EIDORS | `498.3 us` | `557.8 us` | `1.120` | `48.41 us` | `9.600 us` | `0.198` |

全部 228 条记录都满足 `cold median > warm reuse median`，所以此前出现过的
“PyEIDORS 冷态比热态快”反常结果没有进入本报告。冷态、setup 和 warm 是不同
语义，不能把一个阶段的比例当成另一个阶段的绝对速度。

## 稳健性、限制与可复现证据

- 独立 QA 从原始 228 条精度记录重算胜场，得到 Classic `27/11/0`、Robin
  `29/9/0`，与生成器汇总一致；
- 38/38 QQ truth 均通过精确残差零与两形式恒等认证；
- 低阻抗交叉矩阵共 48 条、纯顺序记录 16 条，同一 case/assembly 的两后端使用
  相同 block SHA；
- 228/228 计时记录通过 `cold > warm` 门禁；
- 图中英文与数字使用 Times New Roman，坐标与尺度未截断。

限制如下：

1. 排名只回答同一有限维有理 P1 矩阵的 floating-point forward accuracy，不回答
   真实圆域连续解误差；
2. Q4 只有 6 个预注册设置，足以回答本矩阵是否保持顺序，不足以证明任意更大网格；
3. 8 电极反例说明电极数是重要交互项；
4. 计时来自当前机器、库版本和单进程设置，适合比较本协议内 Classic/Robin，不能
   直接外推到其他硬件；
5. 低阻抗归因的“组装实现”仍是一个组合因素，需要更细的冻结实验才能拆成单一根因。

主要机器可读证据：

- `output/cem_exact_extension/cem_exact_extension_accuracy.json` / `.csv`
- `output/cem_exact_extension/cem_exact_extension_timing.json` / `.csv`
- `output/cem_exact_extension/cem_low_z_attribution.json` / `.csv`
- `output/cem_exact_extension/backend_cross_manifest.json`
- `output/cem_exact_extension/qq_cache/`

## 建议的下一步

1. 将 Q4 扩展为 Q4/Q5 或两个不同拓扑的同规模有理网格，并在提交前固定 case 清单；
2. 在 8、16、32 电极上做平衡电极数因子设计，验证 8 电极反例是系统性趋势还是
   当前设置交互；
3. 对低阻抗 case 逐项冻结局部积分值、DOF 排列、稀疏插入顺序和 CSC 排序，形成
   更细的因果消融；
4. 继续真实圆域连续精度实验，但把它明确作为 FEM 离散误差证据，与本报告的离散
   代数舍入精度并列，而不是用其中一个覆盖另一个。

## 仍待回答的问题

- Q4 的稳定顺序在 Q5、不同网格拓扑和 32 电极上是否继续保持？
- PyEIDORS/EIDORS 组装差异中，DOF/单元排列、积分核和稀疏归并各自贡献多少？
- 同一组装矩阵在不同 BLAS、SuperLU/MKL/PARDISO 版本和硬件上，后端效应是否仍
  小于组装效应？

这些问题需要新的预注册矩阵，不能从当前 38 例和 4 个归因 case 外推。
