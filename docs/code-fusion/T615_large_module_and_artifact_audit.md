# T615 超大模块与版本资产审计

日期：2026-08-01  
范围：只读评估；本审计未修改求解器/GUI 行为，未删除 `reports/` 或 `pictures/` 资产。

## 结论

- 四个最大模块共 21,029 行，精选 Ruff 复杂度族合计 98 项；“代码已相当干净、主要精简空间不在 `src/`”不成立。
- 禁止对四文件做大块机械拆分。`_solve_with_petsc` 影响级别 CRITICAL；`_prepare_single_step_cached_runtime` 和 `_on_run_sim_inverse` 为 HIGH。
- `greit.py` 有最清晰的纯函数边界，适合作为第一个实际拆分试点；PETSc 解法器路径最后动。
- 四大文件之外，2,536 行的 `gauss_newton_linear_system.py` 包含全仓最极端的 `_solve_linear_system_fast`（C901=140，47 branches，527 statements），应单独立项，不应藏在四大文件拆分中。
- 当前决策：32 个 tracked reports 全保留；9 个 README 直接消费的图片保留；3 个无引用图片进入后续视觉/来源复核，本次不删。

## 可复现基线

行数使用 `wc -l`；复杂度仅统计 `C901,PLR0911,PLR0912,PLR0915`，不代表 Ruff 全规则结果。

| 模块 | 行数 | C901 | returns | branches | statements | 主要热点 |
|------|-----:|-----:|--------:|---------:|-----------:|----------|
| `src/eit_app/controllers/reconstruction_controller.py` | 6,463 | 10 | 7 | 6 | 8 | `_prepare_single_step_cached_runtime`: C901=19, statements=149 |
| `src/eit_app/ui/main_window.py` | 5,602 | 10 | 6 | 7 | 6 | `_on_run_sim_inverse`: C901=44, statements=235 |
| `src/pyeidors/inverse/greit.py` | 4,633 | 6 | 3 | 3 | 2 | 目标几何/训练构建职责混合 |
| `src/pyeidors/forward/eit_forward_model.py` | 4,331 | 9 | 4 | 6 | 5 | `_solve_with_petsc`: C901=28, statements=132 |

## GitNexus 影响半径

2026-08-01 刷新索引后执行 upstream impact（含测试）。图结果只是下限；动态导入、monkeypatch、Qt signal 和私有 mixin 调度可能不在图中。

| 候选边界 | 风险 | 直接/总影响 | 流程/模块 | 裁决 |
|----------|------|-------------|-----------|------|
| `EITForwardModel._solve_with_petsc` | CRITICAL | 4 / 29 | 3 / 7 | 先锁定诊断键、fallback reason、KSP session 计数；最后拆 |
| `_prepare_single_step_cached_runtime` | HIGH | 2 / 15 | 1 / 4 | 先提取纯 metadata schema/defaults，保持原函数签名为门面 |
| `EITWorkstation._on_run_sim_inverse` | HIGH | 19 / 19 | 0 / 1 | 仅提取纯 request builder；Qt 状态/signal/结果呈现留在 window |
| `build_3d_greit_rm` | LOW* | 0 / 0 | 0 / 0 | `*` 公开 API + 动态 registry 风险；保留 `greit.py` 兼容门面 |
| `_solve_linear_system_fast` | LOW* | 0 / 0 | 0 / 0 | `*` 与 7 个静态测试文件相矛盾；视为索引缺口，不是安全证明 |

## 建议拆分顺序

### 0. 共同前置门禁

1. 锁定现有公开签名、cache/signature payload、诊断键、fallback reason、Qt signal 顺序。
2. 为每个被移动分支先加 characterization test，再做一个边界/一个提交。
3. 原公开/被 monkeypatch 的符号保留薄门面；不在同一批中重命名、搬文件并改语义。
4. 每批运行对应 §V 测试、全 unit suite 和 GitNexus `detect_changes()`。

### 1. `greit.py`（最先）

- 拟议 `greit_targets.py`：目标分布、finite-target 几何、节点/单元坐标归一化。
- 拟议 `greit_training.py`：`Y/D/PJt/M`、weight search、native training pipeline。
- `greit.py` 保留 `GREITRM`、`build_3d_greit_rm`、metrics 门面并 re-export；`greit_registry.py` 不再扩展训练数学。
- 核心门禁：V29/V30/V41/V50/V63 与 `test_greit_rm.py`/`test_greit_noise_figure.py`/官方 fixture gate。

### 2. `reconstruction_controller.py`

- 拟议 `reconstruction_runtime_config.py`：metadata defaults、标准化、route policy，全部保持纯函数。
- 拟议 `reconstruction_artifact_runtime.py`：RM/GREIT artifact 解析、自动构建、缓存命中决策。
- 原 `_prepare_single_step_cached_runtime` 和 `execute_reconstruction_request_in_backend` 保留稳定门面；`batch_reconstruction_controller.run` 是必验证流程。

### 3. `main_window.py`

- 拟议 `ui/simulation_inverse_flow.py`：从 forward result + panel config 纯构建 `ReconstructionRequest`。
- `_on_run_sim_inverse` 仅留 stale check、loading/running 状态、worker 启动与结果呈现。
- 禁止把 Qt widget 引用或 signal 携入纯 builder；保持 GUI smoke 中 RM/GREIT/sparse 路由与错误文案。

### 4. `eit_forward_model.py`（最后）

- 先提取无副作用的 KSP 观测快照/诊断组装，再考虑 `petsc_solve_runtime.py` 执行器。
- `EITForwardModel._solve_with_petsc` 长期保留门面；不同批改 MatSolve 政策、CUDA fallback、gauge fix 或 session reuse。
- 核心门禁：V1/V2/V3/V13/V14/V19/V24/V42/V43，以及 KSP session/mat-solve/branch suite。

### 独立优先债务：`_solve_linear_system_fast`

不做“大函数切四段”。先把 solver/preconditioner/backend 路由固定为可比较的决策表/纯 policy，保持原函数为调度门面；以 `test_gn_fast_linear_solver.py`、runtime contract/golden/helper/branch 与 native-complex 测试作为基线。

## `reports/` 决策

| 项 | 实测 | 决策/理由 |
|----|------|-----------|
| tracked 文件 | 32 | 全保留；合计仅 936 KiB，承载 parity/runtime/互操作证据 |
| 直接消费 | V21/V50/V63、`METHOD_ROADMAP.md`、`eidors_greit_source_map.json` 等 | 属可审计来源，不按“生成物”统一删除 |
| ignore 政策 | `.gitignore` 整体忽略 `reports/` | 风险：新证据需 `git add -f`，易形成 SPEC 引用但未跟踪；需显式 allowlist/provenance 规则 |

当前检出缺失的权威证据路径：

1. V47: `reports/benchmarks/forward_spd_gamg_cuda_48e_repeat2_20260421.json`
2. V46: `reports/runtime_benchmarks/dual_model_rm_v1_20260421/summary.json`
3. V48: `reports/runtime_benchmarks/lazy_48e_spd_gamg_cuda_b4_20260421/summary.json`
4. V50: `reports/eidors_greit_fixtures/reduced_48e_5936_eidors_greit_fixture.mat`

后续不应伪造/重算以冒充原证据；应在 T616 中选择：跟踪小型权威 summary/manifest，或把大产物改为外部存储的 checksum + provenance 引用，并修正不存在路径。

## `pictures/` 决策

12 个 tracked 文件共 50 MiB，未配置 Git LFS/filter。README 直接消费 9 个：Fig.1/3/4/5/6、3 个 benchmark PNG 和 `reconstruction_iterations.gif`，全部保留。

| 无引用候选 | 大小 | 建议 |
|------------|-----:|------|
| `Fig. 2. Functional Workflow.png` | 4.6 MiB | drop 候选；若论文/文档仍需，先建立消费链和来源说明 |
| `Fig. 7. pinn_pyeidors_eit_infographic_white.png` | 5.3 MiB | drop 候选；先与 SVG 视觉对等性验证 |
| `Fig. 7. pinn_pyeidors_eit_infographic_white.svg` | 28 KiB | 若具备来源/授权，作为可编辑源保留；否则与 PNG 同删 |

大型但正在消费的 Fig.3（9.3 MiB）/Fig.5（20 MiB）不直接删；如要压缩，需单独做像素尺寸、透明度、文字可读性和 README 渲染的视觉 QA。

## 后续决策点

- T616 可直接执行证据路径/ignore 政策修复和 3 个图片候选复核。
- 四模块真正拆分会修改 HIGH/CRITICAL 符号，必须由维护者选定先后顺序后再建独立任务；不从本审计自动扩张到实施。
