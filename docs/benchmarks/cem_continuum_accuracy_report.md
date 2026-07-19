# 真实圆域 CEM：三种 FEM 的连续问题精度与网格收敛

## 技术结论

本实验回答连续物理问题，而非固定有限维矩阵的舍入误差。三种求解器在每一级均导入同一份 P1 网格；Classic 与 Robin 使用相同物理量、原始 SI 电压和零均值规范。

最细网格的 10 个 case/formulation 排名中，只有 0 个满足参考不确定度区间完全分离；其余必须报告并列，不能把区间重叠解释为严格精度排名。

全部 30/30 条收敛序列随网格加密单调下降，最后三级拟合阶位于 1.583–1.837。最细网格三个求解器误差的最大绝对 spread 为 1.803e-14，仅为对应连续参考不确定度的 3.226e-06；Classic/Robin 最大内部差为 4.372e-15。这说明可观测误差由共同 P1/直边圆域离散主导，而不是求解器品牌或两种等价 CEM 代数形式主导。

## 独立连续参考解为什么成立

均匀圆域内部满足 $\nabla\cdot(\sigma\nabla u)=0$。把调和解按圆周 Fourier 模式展开后，第 $n$ 个非零模式从边界电势到法向电流密度的系数为 $\sigma |n|/R$，所以逆映射为

$$\widehat u_n=\frac{R}{\sigma |n|}\widehat q_n,\qquad n\ne0.$$
该 Fourier 乘子就是圆域的解析 Neumann-to-Dirichlet 映射。总注入电流为零，因此 $q$ 的零 Fourier 模式为零；电势常数模式由 $\sum_lU_l=0$ 唯一确定。数值系统只施加电极上的 $u+zq=U_l$、间隙上的 $q=0$ 和 $\int_{E_l}q\,ds=I_l$。这等价于真实圆域上的连续 CEM，而非某个内部三角网格的离散方程。

边界电流使用周期 midpoint Fourier–Nyström 网格。电极端点与网格单元边界严格对齐，分辨率依次为 5120、10240、20480、40960；最后两组三网格经验阶 Richardson 外推之差定义参考不确定度。只有线性 residual、电流积分 residual、Robin residual、规范 residual 均不超过 $10^{-10}$ 且外推差不超过 $5\times10^{-6}$ 时才认证。连续参考从不读取 PyEIDORS、NGSolve 或 EIDORS 的组装矩阵。

### 参考认证

| Case | 最后观测阶 | 外推相对不确定度 | 最大约束 residual | 认证 |
|---|---:|---:|---:|:---:|
| C1 | 1.8667 | 6.621e-09 | 1.500e-12 | 是 |
| C2 | 1.8317 | 6.278e-08 | 1.338e-12 | 是 |
| C3 | 1.8737 | 8.975e-10 | 4.551e-13 | 是 |
| C4 | 1.8727 | 1.758e-09 | 1.932e-12 | 是 |
| C5 | 1.8644 | 5.048e-09 | 1.690e-12 | 是 |

## 真实圆域公共网格

网格由 Gmsh 真圆 CAD 圆弧生成，再导出为所有求解器共同使用的线性三角形。边界节点位于真实圆上，弦长和弦—圆弧 sagitta 随加密同步下降。

| Level | target h | nodes | cells | actual hmax | boundary chord | sagitta |
|---|---:|---:|---:|---:|---:|---:|
| H0 | 0.25000 | 290 | 530 | 1.56466e-01 | 1.37337e-01 | 2.36045e-03 |
| H1 | 0.12500 | 477 | 888 | 1.17742e-01 | 1.17742e-01 | 1.73439e-03 |
| H2 | 0.06250 | 1276 | 2438 | 6.97666e-02 | 5.88963e-02 | 4.33692e-04 |
| H3 | 0.03125 | 4240 | 8270 | 4.14765e-02 | 3.05421e-02 | 1.16609e-04 |

## 最细网格相对连续参考误差

| Case | Solver | Classic | Robin | Classic/Robin 内部差 |
|---|---|---:|---:|---:|
| C1 | PyEIDORS/DOLFINx | 3.670e-03 | 3.670e-03 | 1.733e-15 |
| C1 | NGSolve | 3.670e-03 | 3.670e-03 | 2.155e-15 |
| C1 | EIDORS | 3.670e-03 | 3.670e-03 | 1.310e-15 |
| C2 | PyEIDORS/DOLFINx | 1.366e-02 | 1.366e-02 | 3.838e-15 |
| C2 | NGSolve | 1.366e-02 | 1.366e-02 | 3.819e-15 |
| C2 | EIDORS | 1.366e-02 | 1.366e-02 | 2.760e-15 |
| C3 | PyEIDORS/DOLFINx | 6.016e-04 | 6.016e-04 | 8.628e-16 |
| C3 | NGSolve | 6.016e-04 | 6.016e-04 | 9.806e-16 |
| C3 | EIDORS | 6.016e-04 | 6.016e-04 | 4.381e-16 |
| C4 | PyEIDORS/DOLFINx | 1.170e-03 | 1.170e-03 | 1.174e-15 |
| C4 | NGSolve | 1.170e-03 | 1.170e-03 | 1.173e-15 |
| C4 | EIDORS | 1.170e-03 | 1.170e-03 | 8.899e-16 |
| C5 | PyEIDORS/DOLFINx | 2.562e-03 | 2.562e-03 | 2.543e-15 |
| C5 | NGSolve | 2.562e-03 | 2.562e-03 | 3.945e-15 |
| C5 | EIDORS | 2.562e-03 | 2.562e-03 | 1.634e-15 |

## 最细网格不确定度感知排名

| Case | Formulation | 严格顺序成立 | 最优并列集合 |
|---|---|:---:|---|
| C1 | classic | 否 | NGSolve, PyEIDORS/DOLFINx, EIDORS |
| C1 | robin_transconductance | 否 | NGSolve, PyEIDORS/DOLFINx, EIDORS |
| C2 | classic | 否 | NGSolve, PyEIDORS/DOLFINx, EIDORS |
| C2 | robin_transconductance | 否 | NGSolve, PyEIDORS/DOLFINx, EIDORS |
| C3 | classic | 否 | NGSolve, EIDORS, PyEIDORS/DOLFINx |
| C3 | robin_transconductance | 否 | NGSolve, EIDORS, PyEIDORS/DOLFINx |
| C4 | classic | 否 | NGSolve, EIDORS, PyEIDORS/DOLFINx |
| C4 | robin_transconductance | 否 | NGSolve, EIDORS, PyEIDORS/DOLFINx |
| C5 | classic | 否 | NGSolve, PyEIDORS/DOLFINx, EIDORS |
| C5 | robin_transconductance | 否 | NGSolve, PyEIDORS/DOLFINx, EIDORS |

## 收敛与 FEM 外推

| Case | Solver | Formulation | fitted p | finest error | FEM h→0 error |
|---|---|---|---:|---:|---:|
| C1 | PyEIDORS/DOLFINx | classic | 1.796 | 3.670e-03 | 2.429e-03 |
| C1 | PyEIDORS/DOLFINx | robin_transconductance | 1.796 | 3.670e-03 | 2.429e-03 |
| C1 | NGSolve | classic | 1.796 | 3.670e-03 | 2.429e-03 |
| C1 | NGSolve | robin_transconductance | 1.796 | 3.670e-03 | 2.429e-03 |
| C1 | EIDORS | classic | 1.796 | 3.670e-03 | 2.429e-03 |
| C1 | EIDORS | robin_transconductance | 1.796 | 3.670e-03 | 2.429e-03 |
| C2 | PyEIDORS/DOLFINx | classic | 1.583 | 1.366e-02 | 1.094e-02 |
| C2 | PyEIDORS/DOLFINx | robin_transconductance | 1.583 | 1.366e-02 | 1.094e-02 |
| C2 | NGSolve | classic | 1.583 | 1.366e-02 | 1.094e-02 |
| C2 | NGSolve | robin_transconductance | 1.583 | 1.366e-02 | 1.094e-02 |
| C2 | EIDORS | classic | 1.583 | 1.366e-02 | 1.094e-02 |
| C2 | EIDORS | robin_transconductance | 1.583 | 1.366e-02 | 1.094e-02 |
| C3 | PyEIDORS/DOLFINx | classic | 1.837 | 6.016e-04 | 3.977e-04 |
| C3 | PyEIDORS/DOLFINx | robin_transconductance | 1.837 | 6.016e-04 | 3.977e-04 |
| C3 | NGSolve | classic | 1.837 | 6.016e-04 | 3.977e-04 |
| C3 | NGSolve | robin_transconductance | 1.837 | 6.016e-04 | 3.977e-04 |
| C3 | EIDORS | classic | 1.837 | 6.016e-04 | 3.977e-04 |
| C3 | EIDORS | robin_transconductance | 1.837 | 6.016e-04 | 3.977e-04 |
| C4 | PyEIDORS/DOLFINx | classic | 1.837 | 1.170e-03 | 7.625e-04 |
| C4 | PyEIDORS/DOLFINx | robin_transconductance | 1.837 | 1.170e-03 | 7.625e-04 |
| C4 | NGSolve | classic | 1.837 | 1.170e-03 | 7.625e-04 |
| C4 | NGSolve | robin_transconductance | 1.837 | 1.170e-03 | 7.625e-04 |
| C4 | EIDORS | classic | 1.837 | 1.170e-03 | 7.625e-04 |
| C4 | EIDORS | robin_transconductance | 1.837 | 1.170e-03 | 7.625e-04 |
| C5 | PyEIDORS/DOLFINx | classic | 1.796 | 2.562e-03 | 1.536e-03 |
| C5 | PyEIDORS/DOLFINx | robin_transconductance | 1.796 | 2.562e-03 | 1.536e-03 |
| C5 | NGSolve | classic | 1.796 | 2.562e-03 | 1.536e-03 |
| C5 | NGSolve | robin_transconductance | 1.796 | 2.562e-03 | 1.536e-03 |
| C5 | EIDORS | classic | 1.796 | 2.562e-03 | 1.536e-03 |
| C5 | EIDORS | robin_transconductance | 1.796 | 2.562e-03 | 1.536e-03 |

## 误差含义

- `continuum_relative_l2`：$\|U_h-U_{cont}\|_F/\|U_{cont}\|_F$，包含 P1 场离散误差与直边三角形对圆域的几何误差。
- `classic_robin_relative_l2`：同一求解器、同一网格内两种代数实现的差；它主要反映浮点舍入，不是连续 FEM 误差。
- `fem_extrapolated_continuum_relative_l2`：由最后三级公共网格的电极电压独立 Richardson 外推后，与 Fourier–Nyström 连续参考比较。
- `reference_relative_uncertainty`：相邻两次连续参考外推结果的差，严格排名必须超过该不确定度。

## 限制

本套件针对均匀、各向同性二维圆域。非圆域、非均匀电导率和三维问题没有解析圆域 NtD 对角化，需要另一套独立高阶体积或边界参考。当前主比较固定为共同 P1/float64，因此回答的是受控离散精度，而不是各软件可用最高阶单元的能力上限。

## 可复现产物

- `cem_continuum_accuracy.json`：完整严格 JSON。
- `cem_continuum_accuracy_metrics.csv`：逐 case/mesh/solver/formulation 误差。
- `cem_continuum_convergence.png`：五组物理设置几何平均收敛图。
- `suite_manifest.json`：真实圆网格、指纹、物理配置和参考认证入口。
