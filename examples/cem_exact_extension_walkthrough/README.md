# 可复现的经典 CEM 与 Robin CEM 实验 / Reproducible CEM experiment

| 中文 | English |
|---|---|
| 本目录是面向教授、可直接运行的 38 案例有理数精确 CEM 报告配套材料。 | This folder is the professor-facing, executable companion to the 38-case exact-rational CEM report. |

| 模式 | 中文 | English |
|---|---|---|
| 教学/调试 | 运行一个小案例，在关键行暂停，并在 VS Code、Jupyter 或 MATLAB 中检查每个矩阵。 | Run one small case, stop at important lines, and inspect every matrix in VS Code, Jupyter, or MATLAB. |
| 完整复现 | 在 PyEIDORS/DOLFINx、NGSolve、EIDORS 中重跑全部 38 案例，重建有理数精确参考并重新生成 CSV/JSON。 | Rerun all 38 cases in the three frameworks, rebuild exact rational references, and regenerate CSV/JSON evidence. |

数学与实验细节 / Mathematical and experimental details:
[`EXPERIMENT_GUIDE.md`](EXPERIMENT_GUIDE.md).

## 文件说明 / Contents

| 文件 / File | 中文用途 | English purpose |
|---|---|---|
| `pyeidors_walkthrough.ipynb` | PyEIDORS/DOLFINx 双语教学 Notebook | Bilingual PyEIDORS/DOLFINx tutorial |
| `exact_rational_truth_walkthrough.ipynb` | 从有理输入重新组装、求解并认证数学精确真值 | Reassemble, solve, and certify the exact truth from rational inputs |
| `pyeidors_debug.py` | VS Code `# %%` 与普通调试脚本 | VS Code `# %%` and terminal debug script |
| `ngsolve_walkthrough.ipynb` | NGSolve 双语教学 Notebook | Bilingual NGSolve tutorial |
| `ngsolve_debug.py` | NGSolve 单案例与全套运行器 | NGSolve selected-case and full-suite runner |
| `eidors_selected_case_walkthrough.m` | MATLAB 分节双语演示 | Section-by-section bilingual MATLAB walkthrough |
| `run_eidors_38_cases.m` | 自动运行全部 38 个 EIDORS 案例 | Automated runner for all 38 EIDORS cases |
| `experiment_common.py` | 两个 Python 演示共用的显式分块代数 | Explicit block algebra shared by both Python walkthroughs |
| `reproduce_report.py` | 从 228 条精度与计时记录复算报告 | Recompute report tables from 228 accuracy and timing records |
| `expected/` | 冻结 CSV 证据 | Frozen CSV evidence |
| `fixtures/X01/` | 可移植共享网格与有理数精确夹具 | Portable shared mesh and exact-rational fixture |
| `.vscode/launch.json` | 调试配置示例 | Debug configurations |

三份 Notebook 与 MATLAB 演示都会显示同一个规范网格、逐单元电导率、
电极编号和选定边界电流模式，并打印 P1 阶次、`float64` 与网格指纹。
The three notebooks and MATLAB walkthrough all display the same canonical
mesh, per-cell conductivity, electrode labels, and selected boundary drive,
and report the P1 order, `float64` dtype, and mesh fingerprint.

三份 Notebook 还会显示同一个注流模式的六面板求解结果：Classic 体电势、
Robin 体电势、体电势差值、Classic 电极电压、Robin 电极电压以及电压差值。
执行后，高清 PNG 同时保存在 `figures/`。 / The notebooks also render a
six-panel solved-result comparison for the same drive and save high-resolution
PNG copies under `figures/`.

默认 X01 是均匀背景正问题：输入网格、$\sigma=1/8$、$z=1$ 和相邻电流，
输出体电势 $u$ 与边界电极电压 $U$，没有内部异常物，也不是逆问题重构。
38 案例中的 X17–X24 是非均匀 `left_right` 电导率正问题。 /
Default X01 is a uniform-background forward problem: mesh, $\sigma=1/8$,
$z=1$, and adjacent currents are inputs; body potential $u$ and boundary
voltage $U$ are outputs. It has no interior anomaly and is not an inverse
reconstruction. X17–X24 use heterogeneous `left_right` conductivity.

## 1. 立即复现报告数字 / Immediate report-number reproduction

下面的命令读取冻结证据，复算几何平均误差、胜出次数、Q4 排序和
Robin/Classic 计时比。 / The command recomputes geometric means, win counts,
Q4 ordering, and Robin/Classic timing ratios from frozen evidence.

```bash
cd /home/tom/workspace/PyEidors_wsl2
nix develop .#default --command \
  python examples/cem_exact_extension_walkthrough/reproduce_report.py
```

这里必须使用 `default`，因为本实验要求真实数 `float64`。重新生成结果时
不能使用常规 `complex64-cuda`。 / Use `default` because this experiment
requires real `float64`, not the normal `complex64-cuda` profile.

## 2. PyEIDORS 教学与调试 / PyEIDORS teaching and debugging

运行 X01 案例 / Run the selected X01 case:

```bash
cd /home/tom/workspace/PyEidors_wsl2
nix develop .#default --command \
  python examples/cem_exact_extension_walkthrough/pyeidors_debug.py \
  --case X01 --show-forward-setup --show-results
```

强制 DOLFINx 重新组装 / Force DOLFINx reassembly:

```bash
nix develop .#default --command \
  python examples/cem_exact_extension_walkthrough/pyeidors_debug.py \
  --case X01 --regenerate
```

打开 Notebook / Open the notebook:

```bash
nix develop .#default --command \
  jupyter lab \
  examples/cem_exact_extension_walkthrough/pyeidors_walkthrough.ipynb
```

在 WSL 的 VS Code 中只需注册一次 / Register once for VS Code in WSL:

```bash
nix develop .#default --command \
  python examples/cem_exact_extension_walkthrough/register_vscode_kernel.py
```

重载 VS Code 后选择 / Reload VS Code and choose:

```text
Select Kernel
  → Jupyter Kernel...
  → PyEIDORS real float64 (Nix)
```

不要在 `Python Environments` 中选择 `/nix/store/.../bin/python`。命名内核
每次都会重新进入 `nix develop .#default`，从而取得完整 DOLFINx/PETSc
真实 `float64` 环境。 / Do not select the raw Nix Python under
`Python Environments`; the named kernel re-enters the complete profile.

在 VS Code 中打开 `pyeidors_debug.py`，每个 `# %%` 均可独立运行。关键变量 /
Open `pyeidors_debug.py`; useful breakpoint variables:

- `blocks.robin_matrix` — Robin 体场矩阵 / body matrix $A_R$
- `blocks.coupling` — 耦合矩阵 / coupling $C$
- `blocks.electrode_matrix` — 电极块 / electrode block $D$
- `classic_state.system_matrix` — 经典 CEM 增广矩阵 / augmented matrix
- `robin_state.electrode_basis` — 零和 Helmert 基 / zero-sum basis $Q$
- `robin_state.response_basis` — 响应基 / response basis $A_R^{-1}CQ$
- `robin_state.schur_action_basis` — Schur 作用 / $DQ-C^\mathsf{T}A_R^{-1}CQ$
- `robin_state.reduced_map` — 约化映射 / reduced map $T_r$
- `classic_solution.electrode_voltage` and
  `robin_solution.electrode_voltage`

## 3. 有理数精确真值教学 / Exact rational truth tutorial

打开配套 Notebook / Open the companion notebook:

```bash
cd /home/tom/workspace/PyEidors_wsl2
nix develop .#default --command \
  jupyter lab \
  examples/cem_exact_extension_walkthrough/exact_rational_truth_walkthrough.ipynb
```

该 Notebook 不只读取 `exact_reference.json`，而是从 X01 的 `Fraction`
坐标、整数拓扑、有理 $\sigma,z$ 和整数 $I$ 开始，重新执行：

1. 精确 P1 体积分与电极边界积分，得到 $A_R,C,D\in\mathbb Q$；
2. `DomainMatrix.convert_to(QQ).lu_solve` 传统增广多右端求解；
3. 独立 Robin/Schur 精确消元；
4. 满秩、严格零残差、零均值规范、Classic/Robin 逐分数同解；
5. 分数字符串 SHA-256 与可移植参考文件一致；
6. `Fraction.from_float` 加 100 位指标计算，得到真值相对误差和缩放后向残差。

The notebook rebuilds the exact solution from rational inputs, rather than
merely loading JSON. Its “truth” is the unique exact solution of the fixed
finite-dimensional rational P1 CEM system, not an analytic continuum-PDE
solution.

## 4. 用 PyEIDORS 准备 38 个共享案例 / Prepare all 38 shared cases

生成全部共享 MAT/MSH/JSON 夹具和 PyEIDORS 报告 /
Generate all common fixtures and PyEIDORS reports:

```bash
cd /home/tom/workspace/PyEidors_wsl2
nix develop .#default --command \
  python scripts/benchmarks/cem_exact_extension_suite.py prepare \
  --output-dir output/cem_exact_extension
```

生成结构 / Generated structure:

```text
output/cem_exact_extension/
  suite_manifest.json
  cases/
    X01_.../
      common_mesh/cem_exact_extension_p1.{mat,msh,json}
      pyeidors_assembled_blocks.mat
      pyeidors_report.json
    ...
    X38_.../
```

## 5. NGSolve 教学与调试 / NGSolve teaching and debugging

NGSolve 使用隔离环境，避免改变 PyEIDORS 依赖。 /
NGSolve uses an isolated environment to avoid changing PyEIDORS dependencies:

```bash
cd /home/tom/workspace/PyEidors_wsl2
uv run --no-project --python /usr/bin/python3 \
  --with ngsolve==6.2.2606 \
  --with scipy --with numpy --with mpmath \
  python examples/cem_exact_extension_walkthrough/ngsolve_debug.py \
  --case X01 --regenerate --show-forward-setup --show-results
```

在同一隔离环境中打开 NGSolve Notebook /
Open the NGSolve notebook in the same isolated environment:

```bash
uv run --no-project --python /usr/bin/python3 \
  --with ngsolve==6.2.2606 \
  --with scipy --with numpy --with mpmath \
  --with jupyter --with matplotlib \
  jupyter lab \
  examples/cem_exact_extension_walkthrough/ngsolve_walkthrough.ipynb
```

运行全部 38 个案例 / Run all 38 cases:

```bash
uv run --no-project --python /usr/bin/python3 \
  --with ngsolve==6.2.2606 \
  --with scipy --with numpy --with mpmath \
  python examples/cem_exact_extension_walkthrough/ngsolve_debug.py \
  --all --suite-output output/cem_exact_extension \
  --timing-repeats 11
```

## 6. EIDORS/MATLAB 双语演示 / Bilingual walkthrough

逐节打开并运行 / Open and run section-by-section:

```text
examples/cem_exact_extension_walkthrough/eidors_selected_case_walkthrough.m
```

如果 MATLAB 路径中没有 EIDORS，请在第 1 节设置 `eidors_startup`。脚本会把
$A_R,C,D,Q,T_r,u,U$ 与所有残差保留在工作区。 / Set `eidors_startup`
when needed; all matrices and residuals remain in the MATLAB workspace.

PyEIDORS 准备好共享案例后运行完整 EIDORS 套件 /
Run the complete EIDORS suite after preparation:

```matlab
repo = "\\wsl.localhost\Ubuntu-22.04\home\tom\workspace\PyEidors_wsl2";
startup = "C:\path\to\eidors\startup.m";
run_eidors_38_cases(repo, startup, 11);
```

## 7. 重建有理数精确比较与报告 / Rebuild exact-QQ comparison

三个求解器均写出报告后 / After all three solvers write their reports:

```bash
cd /home/tom/workspace/PyEidors_wsl2
nix develop .#default --command \
  python scripts/benchmarks/cem_exact_extension_suite.py compare \
  --output-dir output/cem_exact_extension

nix develop .#default --command \
  python scripts/benchmarks/cem_exact_extension_suite.py timing \
  --output-dir output/cem_exact_extension
```

从新结果复算汇总表 / Recompute headline tables from the new run:

```bash
nix develop .#default --command \
  python examples/cem_exact_extension_walkthrough/reproduce_report.py \
  --metrics output/cem_exact_extension/cem_exact_extension_metrics.csv \
  --timing output/cem_exact_extension/cem_exact_extension_timing.csv \
  --json-output output/cem_exact_extension/professor_summary.json
```

Q4 使用隔离并固定版本的 `python-flint==0.6.0` `fmpq_mat.solve`；Q0、Q2
使用 SymPy 在 `QQ` 上的 `DomainMatrix`。 / Q4 uses pinned FLINT; Q0 and Q2
use SymPy `DomainMatrix` over `QQ`.

## 8. VS Code 调试 / Debugging

推荐流程 / Recommended workflow:

1. Open this repository through VS Code Remote–WSL.
2. For PyEIDORS, start VS Code from `nix develop .#default`.
3. Open this folder as the workspace if you want to use its
   `.vscode/launch.json` directly.
4. Set a breakpoint immediately after `blocks = load_assembled_blocks(...)`.
5. Step into `build_classic_state`, `build_robin_state`, `solve_classic`, and
   `solve_robin`.
6. Use the Data Viewer for dense arrays and the Debug Console for sparse
   matrices, for example:

   ```python
   blocks.robin_matrix.shape
   blocks.robin_matrix.nnz
   robin_state.reduced_map
   np.linalg.cond(robin_state.reduced_map)
   ```

For NGSolve, use the `# %%` cells or select the isolated NGSolve interpreter
created by the `uv run --no-project` command.
