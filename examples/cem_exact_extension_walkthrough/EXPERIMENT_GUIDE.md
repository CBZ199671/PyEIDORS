# 38 案例 CEM 实验的数学与计算细节 / Mathematical and computational details

## 1. 实验问题 / Experimental question

**中文说明：** 本实验把“数学等价性”“浮点绝对精度”和“运行时间”严格
分开。经典与 Robin 结果彼此接近只能证明两条浮点路径一致，不能代替与有理数
精确解的误差比较。

The experiment separates three different questions that can otherwise be
confused:

1. **Mathematical equivalence:** do the traditional augmented Complete
   Electrode Model (Classic CEM) and the Robin/transconductance formulation
   represent the same finite-dimensional problem?
2. **Floating-point accuracy:** when the two equivalent formulations are
   evaluated in `float64`, how close is each FEM implementation to the exact
   solution of its prescribed rational discrete problem?
3. **Runtime:** for the same preassembled $A_R,C,D$ blocks and all current
   right-hand sides, what are the cold and retained-state costs of the two
   solution strategies?

The experiment does **not** use agreement between Classic and Robin as a proxy
for absolute accuracy. Two floating-point algorithms can agree closely and
still be farther from the exact solution.

## 1.1 每个案例实际求解的正问题 / The forward problem solved by every case

**中文说明：** 38 个案例全部是 CEM 正问题，不是没有物理意义的裸矩阵测试。
每个案例的输入是规范节点和三角形网格、逐单元已知电导率
$\sigma_k$、带编号电极边、接触阻抗 $z_\ell$，以及满足总和为零的边界注入
电流 $I_{\ell p}$。输出是每个注流模式对应的体内节点电势 $u$ 和边界电极
电压 $U$。X01–X16 是均匀背景；X17–X24 在左右区域分别设置
$\sigma=1/4$ 与 $1$，表示放入一个已知非均匀区域后计算边界电压；
X25–X32 改为 8 电极；X33–X38 使用更细 Q4 网格。

All 38 cases are physical CEM forward problems, not bare matrix tests. The
inputs are canonical nodes and triangles, known cellwise conductivity
$\sigma_k$, labelled electrode edges, contact impedances $z_\ell$, and
zero-sum injected-current columns $I_{\ell p}$. The outputs are the body
nodal potential $u$ and boundary electrode voltage $U$ for every drive.
X01–X16 use uniform backgrounds; X17–X24 use left/right conductivities
$1/4$ and $1$ to represent a known internal heterogeneity; X25–X32 use
8 electrodes; X33–X38 use the finer Q4 mesh.

For every selected-case walkthrough, the executable script:

1. loads the canonical MAT/JSON fixture;
2. plots the actual P1 triangles coloured by $\sigma_k$;
3. overlays labelled electrodes and the selected $+I/-I$ boundary drive;
4. prints $N,K,L,P$, P1 order, real `float64`, $z$, and current-column sums;
5. asserts that the solver report and fixture have the same SHA-256 mesh
   fingerprint.

因此，“三个程序画出的图看起来相同”只是直观证据；相同的数组维度、逐项物理
输入以及网格指纹断言才是机器可检查的公平性证据。 / Thus matching plots are
visual evidence, while matching dimensions, physical arrays, and the asserted
mesh fingerprint provide the machine-checkable fairness certificate.

求解后，每个选定案例还绘制六个结果面板：Classic 与 Robin 的体电势使用
同一色标，体电势差使用以零为中心的对称色标；两个电极电压图使用相同纵轴，
最后单独放大 `Robin − Classic` 电压差。图直接读取进入残差与误差计算的同一
组解数组，不进行另一网格上的插值或二次求解。 / After solving, every
selected-case walkthrough renders six result panels. Classic and Robin body
fields share colour limits, their signed difference uses symmetric
zero-centred limits, the two electrode-voltage traces share a y-axis, and the
final panel magnifies `Robin − Classic`. The figures use the exact same solved
arrays that enter the residual and error calculations, without interpolation
onto another mesh or a second solve.

## 2. 连续完全电极模型 / Continuous Complete Electrode Model

**中文说明：** 两种离散算法都从同一个连续 CEM 出发；未知量是体内电势
$u$ 与电极电势 $U_\ell$，接触阻抗为 $z_\ell$，并施加电流守恒与电压零均值
规范。

Let $\Omega$ be the conducting body, $\sigma$ its conductivity, $E_\ell$
the $\ell$-th electrode, $z_\ell>0$ its contact impedance, $I_\ell$ the
applied current, $u$ the body potential, and $U_\ell$ the electrode
potential. The CEM is

$$
\nabla\cdot(\sigma\nabla u)=0\quad\text{in }\Omega,
$$

$$
u+z_\ell\sigma\frac{\partial u}{\partial n}=U_\ell
\quad\text{on }E_\ell,
$$

$$
\int_{E_\ell}\sigma\frac{\partial u}{\partial n}\,ds=I_\ell,
\qquad
\sigma\frac{\partial u}{\partial n}=0
\quad\text{on the gaps}.
$$

The physical compatibility and voltage gauge are

$$
\sum_{\ell=1}^{L}I_\ell=0,
\qquad
\sum_{\ell=1}^{L}U_\ell=0.
$$

Both numerical formulations below discretize these same equations.

## 3. 共享有限元分块 / Shared finite-element blocks

**中文说明：** 每个框架先在同一网格与物理参数上组装 $A_R,C,D$。框架内部
的经典/Robin 比较复用同一组分块，因此比较的是线性代数路径，而不是不同网格
或不同有限元组装。

For P1 nodal basis functions $\phi_i$, define the conductivity stiffness
matrix

$$
K_{ij}=\int_\Omega
\sigma\nabla\phi_i\cdot\nabla\phi_j\,dx.
$$

The electrode Robin mass contribution is

$$
B_{ij}=\sum_{\ell=1}^{L}\frac{1}{z_\ell}
\int_{E_\ell}\phi_i\phi_j\,ds,
\qquad
A_R=K+B.
$$

The body/electrode coupling and electrode block are

$$
C_{i\ell}=-\frac{1}{z_\ell}\int_{E_\ell}\phi_i\,ds,
\qquad
D_{\ell\ell}=\frac{|E_\ell|}{z_\ell}.
$$

All Classic/Robin comparisons inside one FEM implementation use the same
preassembled $A_R,C,D$. Therefore, the difference between the two
formulations is the linear-algebra route, not a changed mesh or a second FEM
assembly.

## 4. 传统增广经典 CEM / Traditional augmented Classic CEM

**中文说明：** 经典方法把 $u,U,\lambda$ 放入一个增广稀疏系统，用
$\lambda$ 强制 $\mathbf{1}^\mathsf{T}U=0$。

The zero-mean voltage constraint is imposed with a Lagrange multiplier
$\lambda$:

$$
\begin{bmatrix}
A_R & C & 0\\
C^\mathsf{T} & D & \mathbf{1}\\
0 & \mathbf{1}^\mathsf{T} & 0
\end{bmatrix}
\begin{bmatrix}
u\\U\\\lambda
\end{bmatrix}
=
\begin{bmatrix}
0\\I\\0
\end{bmatrix}.
$$

The complete sparse augmented matrix is factorized once for the warm state.
For a cold sample, the augmented matrix and factorization are rebuilt before
solving all current right-hand sides.

Inspectable implementation:

```python
classic_state = build_classic_state(blocks)
classic_solution = solve_classic(classic_state, blocks.currents)
```

The important objects are:

- `classic_state.system_matrix`
- `classic_state.factor`
- `classic_solution.body_potential`
- `classic_solution.electrode_voltage`

## 5. Robin/跨导 CEM / Robin/transconductance CEM

**中文说明：** Robin 方法先由第一块方程消去体内未知量，再通过零和基 $Q$
把奇异的完整电极空间约化到 $L-1$ 维非奇异子空间。

The first block row gives

$$
A_Ru+CU=0
\quad\Longrightarrow\quad
u=-A_R^{-1}CU.
$$

Substitution into the electrode equation gives the transconductance map

$$
T=D-C^\mathsf{T}A_R^{-1}C.
$$

$T$ is singular in the full electrode space because electrode potentials are
defined only up to a constant. It must not be directly inverted. Let
$Q\in\mathbb{R}^{L\times(L-1)}$ be a deterministic orthonormal basis of the
zero-sum space:

$$
Q^\mathsf{T}Q=I,\qquad Q^\mathsf{T}\mathbf{1}=0.
$$

Write $U=Qy$. The nonsingular reduced problem is

$$
T_r y=Q^\mathsf{T}I,
\qquad
T_r=Q^\mathsf{T}
\left(D-C^\mathsf{T}A_R^{-1}C\right)Q.
$$

The final fields are

$$
U=Qy,
\qquad
u=-A_R^{-1}CQy.
$$

The implementation first solves all $L-1$ response-basis right-hand sides
with one $A_R$ factorization:

$$
R=A_R^{-1}CQ.
$$

It then constructs and factorizes the small dense matrix

$$
T_r
=Q^\mathsf{T}(DQ-C^\mathsf{T}R)
=Q^\mathsf{T}(D-C^\mathsf{T}A_R^{-1}C)Q.
$$

Inspectable implementation:

```python
robin_state = build_robin_state(blocks)
robin_solution = solve_robin(robin_state, blocks.currents)
```

The important objects are:

- `robin_state.electrode_basis` — $Q$
- `robin_state.coupling_basis` — $CQ$
- `robin_state.response_basis` — $A_R^{-1}CQ$
- `robin_state.schur_action_basis` — $DQ-C^\mathsf{T}A_R^{-1}CQ$
- `robin_state.reduced_map` — $T_r$
- `robin_solution.electrode_voltage` — $U$

In the real-valued experiment, transpose and conjugate transpose are the same.
For complex reciprocal CEM systems, the project uses the reciprocal
nonconjugate transpose required by the underlying bilinear form.

## 6. 两个浮点结果为何不逐位相同 / Why results are not bit-identical

**中文说明：** 两种公式在精确算术下相等，但稀疏排序、主元选取、分解方式和
乘法顺序不同，因此 `float64` 中会出现舍入级差异。

The formulations are algebraically equivalent, but they execute different
floating-point operation sequences.

Classic CEM factorizes one larger indefinite augmented sparse matrix. Robin
CEM factorizes $A_R$, performs $L-1$ response solves, forms a Schur
complement, and factorizes a small dense reduced matrix. Sparse insertion
order, local finite-element accumulation order, pivoting, fill-reducing
ordering, and dense/sparse BLAS kernels therefore round intermediate values in
different orders.

Small $10^{-16}$–$10^{-14}$ differences are expected in `float64`.
Mathematical equivalence predicts equality in exact arithmetic, not
bit-for-bit equality between different floating-point factorizations.

## 7. 有理数域 $\mathbb{Q}$ 上的精确参考解 / Exact rational reference

### 7.1 有理几何与参数 / Rational geometry and parameters

**中文说明：** 网格坐标、电导率、接触阻抗和电流都属于 $\mathbb{Q}$，解析
P1 积分也保持有理数，因此离散矩阵的每个元素都是精确分数。

The exterior boundary is one fixed rational 32-gon. Q0, Q2, and Q4 are
dyadically refined versions of that same polygon:

| Level | Nodes | P1 triangles |
|---|---:|---:|
| Q0 | 33 | 32 |
| Q2 | 129 | 192 |
| Q4 | 513 | 896 |

Every coordinate, conductivity, contact impedance, and current entry is a
rational number. The selected polygon also has rational electrode edge
lengths. Analytic P1 triangle stiffness and electrode boundary integrals are
therefore rational.

Consequently, every entry in $A_R,C,D$, the Classic augmented matrix, and
the exact Robin reduced map belongs to the field

$$
\mathbb{Q}=\left\{\frac{p}{q}:p,q\in\mathbb{Z},q\ne0\right\}.
$$

### 7.2 精确线性代数 / Exact linear algebra

**中文说明：** Q0/Q2 使用 SymPy `DomainMatrix` 的 `QQ` LU，Q4 使用固定版本
FLINT `fmpq_mat.solve`。所有加减乘除都在有理数域完成，不进行浮点近似。

For Q0 and Q2, the suite uses SymPy `DomainMatrix` LU over `QQ`. For Q4, it
uses the pinned compiled `python-flint==0.6.0` `fmpq_mat.solve` backend.
Neither backend performs floating-point approximation.

The exact reference uses the rational zero-sum basis

$$
E=
\begin{bmatrix}
I_{L-1}\\-\mathbf{1}^\mathsf{T}
\end{bmatrix},
$$

rather than the orthonormal Helmert basis. Both bases span exactly the same
zero-sum electrode space. The rational basis avoids introducing square roots
into the exact computation.

For every case, the following integer/rational identities are checked:

$$
M_{\mathrm{classic}}X_{\mathrm{classic}}-B=0,
$$

$$
T_{r,\mathbb{Q}}Y-E^\mathsf{T}I=0,
$$

$$
U_{\mathrm{classic},\mathbb{Q}}
=U_{\mathrm{Robin},\mathbb{Q}},
\qquad
\mathbf{1}^\mathsf{T}U_{\mathbb{Q}}=0.
$$

“Zero” here means every rational numerator is exactly zero. This is why the
reference is the mathematical exact solution of the prescribed finite
dimensional discrete problem, not merely a high-precision approximation.

The reference is intentionally independent of the floating assembled matrices
exported by PyEIDORS, NGSolve, or EIDORS.

## 8. 误差指标 / Error metrics

**中文说明：** 真值前向误差回答“电压离数学精确解有多远”；缩放后向残差
回答“计算结果满足精确离散方程到什么程度”；规范残差检查电极电压零均值。

### 8.1 真值前向误差 / Forward truth error

$$
e_{\mathrm{forward}}
=
\frac{\|U_{\mathrm{float64}}-U_{\mathbb{Q}}\|_F}
{\|U_{\mathbb{Q}}\|_F}.
$$

This answers: **how close is the computed electrode voltage to the exact
discrete voltage?** It is the primary accuracy metric.

### 8.2 最大绝对误差 / Maximum absolute error

$$
e_{\max}=\max_{\ell,k}
\left|U_{\mathrm{float64},\ell k}-U_{\mathbb{Q},\ell k}\right|.
$$

This identifies the largest individual voltage-entry discrepancy.

### 8.3 精确约化系统的缩放后向残差 / Scaled backward residual

The candidate voltage is first centered to the zero-mean gauge and represented
in the exact rational basis. For

$$
T_{r,\mathbb{Q}}\widehat{Y}=B_{r,\mathbb{Q}},
$$

the reported residual is

$$
\eta_{\mathrm{backward}}
=
\frac{\|T_{r,\mathbb{Q}}\widehat{Y}-B_{r,\mathbb{Q}}\|_F}
{\|T_{r,\mathbb{Q}}\|_F\|\widehat{Y}\|_F+
 \|B_{r,\mathbb{Q}}\|_F}.
$$

This answers: **how accurately does the computed answer satisfy the exact
discrete equation?**

Forward error and backward residual are different:

- a small residual does not guarantee a small forward error when the system is
  ill-conditioned;
- a forward-accurate answer can have a slightly different residual due to
  scaling and rounding;
- both must therefore be reported.

### 8.4 规范残差 / Gauge residual

$$
e_{\mathrm{gauge}}
=
\frac{\|\mathbf{1}^\mathsf{T}U\|_2}{\|U\|_F}.
$$

This verifies the prescribed zero-mean electrode-voltage gauge.

### 8.5 同一求解器内的经典/Robin 差异 / Within-solver difference

$$
e_{\mathrm{C/R}}
=
\frac{\|U_{\mathrm{Robin}}-U_{\mathrm{Classic}}\|_F}
{\|U_{\mathrm{Classic}}\|_F}.
$$

This measures floating-point disagreement between two equivalent algorithms.
It is not an absolute-accuracy metric.

## 9. 38 个预注册案例 / The 38 preregistered cases

**中文说明：** 38 个案例组合 Q0/Q2/Q4 网格、均匀/非均匀电导率、不同接触
阻抗、电极数和刺激模式；每个案例在三个框架之间共享同一规范网格数据。

| Family | Cases | Meshes | Settings |
|---|---:|---|---|
| `range` | 16 | Q0, Q2 | $\sigma\in\{1/8,4\}$ or $z\in\{1/32,32\}$; adjacent/skip-4 |
| `heterogeneous` | 8 | Q0, Q2 | left $\sigma=1/4$, right $\sigma=1$; $z\in\{1/8,1\}$ |
| `electrode_count` | 8 | Q0, Q2 | 8 electrodes; $z\in\{1/8,1\}$; adjacent/skip-2 |
| `large_q4` | 6 | Q4 | 16 electrodes; $z\in\{1/32,1/8,1\}$; adjacent/skip-4 |

Each case fixes:

- node coordinates and triangle connectivity;
- tagged electrode/gap edges;
- P1 potential order;
- real `float64`;
- per-cell conductivity and its SHA-256 digest;
- contact impedance;
- current matrix;
- zero-mean voltage gauge.

The same canonical mesh arrays are imported by all three FEM frameworks.
DOLFINx maps source cell conductivities through `original_cell_index`;
NGSolve uses Gmsh physical materials; EIDORS reads the same MAT element order.

## 10. 冻结 CSV 复现的报告结果 / Results reproduced by frozen CSV

**中文说明：** 38 案例 × 3 框架 × 2 公式产生 228 条精度记录。下表数字由
`reproduce_report.py` 从逐案例记录重新计算，不是手工填写。

The 38 cases produce $38\times3\times2=228$ accuracy records.

| Solver | Classic forward-error GM | Classic wins | Robin forward-error GM | Robin wins |
|---|---:|---:|---:|---:|
| PyEIDORS/DOLFINx | $1.109\times10^{-15}$ | 27/38 | $1.060\times10^{-15}$ | 29/38 |
| EIDORS | $1.694\times10^{-15}$ | 11/38 | $1.735\times10^{-15}$ | 9/38 |
| NGSolve | $1.120\times10^{-14}$ | 0/38 | $1.075\times10^{-14}$ | 0/38 |

For all six Q4 cases and both formulations, the strict order is

$$
\text{PyEIDORS/DOLFINx}
<
\text{EIDORS}
<
\text{NGSolve}
$$

in forward truth error. This ordering is an observation for the preregistered
Q4 matrices, not a theorem for every mesh, electrode count, or physical
setting. The 8-electrode family contains counterexamples to a universal
per-case ordering.

The exact command

```bash
python reproduce_report.py
```

recomputes these values from the included 228-row accuracy CSV. It also
recomputes timing ratios from the included 228-row timing CSV.

## 11. 公平计时协议 / Fair timing protocol

**中文说明：** 两种公式从相同的预组装 $A_R,C,D$ 开始；冷态包含各自建态、
分解和全部右端项求解，热态只复用本公式自己的保留状态，不能跨公式借用缓存。

Assembly and mesh import are reported separately. The Classic/Robin timing
scope begins from identical preassembled $A_R,C,D$.

### 冷态 / Cold

A cold operation includes:

1. construction of a new formulation-specific state;
2. all required sparse/dense factorizations;
3. solution of the identical complete current matrix.

No factorization or formulation state is reused from the other formulation.

### 保留状态的热态复用 / Warm retained-state reuse

Each formulation creates and retains its own state once:

- Classic retains its augmented sparse LU;
- Robin retains $A_R$ LU, the response basis, and reduced dense LU.

The repeated timed operation then solves the same current matrix using only
that formulation's retained state.

The benchmark uses 11 repetitions, 16 operations per timing sample, alternating
Classic/Robin order, and reports median and IQR. Timing is hardware- and
library-dependent; a new computer should reproduce the protocol and numerical
relationships, not necessarily the original wall-clock seconds.

## 12. 各框架的组装方式 / Framework-specific assembly

### PyEIDORS/DOLFINx

DOLFINx assembles the P1 conductivity and boundary forms. The benchmark
extracts $A_R,C,D$, verifies the imported canonical mesh fingerprint, and
uses SciPy SuperLU for the controlled formulation comparison.

### NGSolve

NGSolve imports the canonical Gmsh 2.2 mesh, maps per-cell conductivities from
physical materials, assembles the P1 forms, and exports its own
$A_R,C,D$. The controlled formulation comparison again uses the shared
SciPy SuperLU block solver so that the observed cross-framework difference
primarily reflects FEM assembly and matrix representation.

### EIDORS

EIDORS directly imports the canonical node/element/electrode arrays, assembles
its official `system_mat_1st_order` CEM matrix, and extracts $A_R,C,D$.
MATLAB sparse LU is used for Classic, while MATLAB sparse $A_R$ LU plus a
dense reduced LU is used for Robin.

## 13. 解释边界 / Interpretation limits

**中文说明：** 本实验严格回答固定有限维离散问题的浮点精度，不等同于连续
真实圆域的离散误差；$10^{-15}$ 量级的框架排序也不应外推成软件的普遍定理。

This exact-rational experiment answers the accuracy of a finite-dimensional
P1 discretization. It does not identify the error relative to the continuous
true-circle CEM solution. Continuum accuracy requires a separate common
true-circle mesh-refinement sequence and an independent continuum reference.

Likewise, a solver ranking at $10^{-15}$ is sensitive to assembly order,
sparse storage order, pivoting, and backend versions. It should be reported as
an empirical result for the fixed protocol, not as a universal property of a
software package.
