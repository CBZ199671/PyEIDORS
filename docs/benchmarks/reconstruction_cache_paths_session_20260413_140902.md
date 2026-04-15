# 重构缓存路径实测报告

时间：2026-04-14
环境：WSL2 Ubuntu 22.04 / `nix develop` 默认 CPU 壳
数据：`/home/tom/workspace/PyEidors_wsl2/data/measurements/test_for_gui/session_20260413_140902`

## 结论

这次实测证明，“强 cache 没有生效”并不是事实。
真实情况是：

- 当前 GUI 默认实时重构链：约 `1.53 fps`
- 关闭步长搜索后的 GN 严格链：约 `1.73 fps`
- GN fast 模式：约 `16.74 fps`
- 强 cache 单步全路径：约 `33.82 fps`
- 强 cache 纯核心算子：约 `128.58 fps`
- GUI 切到强 cache 单步路径并启用实时线程配置后：约 `53.28 fps`

也就是说，用户记忆里的“几十到上百 fps”并非错觉，它对应的是**强 cache 单步路径**，而不是当前 GUI 正在调用的默认 GN 严格链。

## 关键数字

来源：
`/home/tom/workspace/PyEidors_wsl2/docs/benchmarks/reconstruction_cache_paths_session_20260413_140902.json`

| 路径 | 冷启动 | 暖态均值 | FPS |
|---|---:|---:|---:|
| GUI 默认链 | 2295.89 ms | 654.62 ms | 1.53 |
| GN 默认链 | 681.01 ms | 624.97 ms | 1.60 |
| GN 关闭步长搜索 | 591.86 ms | 577.37 ms | 1.73 |
| GN fast 模式 | 283.60 ms | 59.73 ms | 16.74 |
| 强 cache 单步全路径 | 48.25 ms | 29.57 ms | 33.82 |
| 强 cache 纯核心算子 | 无单独冷启动 | 7.78 ms | 128.58 |

## 集成到 GUI 后的真实热态结果

在把硬件实时重构入口切到：

- `/home/tom/workspace/PyEidors_wsl2/src/eit_app/controllers/reconstruction_controller.py` 的 `single_step_cached`

并在 GUI 启动前默认设置实时线程环境：

- `OMP_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`
- `BLIS_NUM_THREADS=1`

之后，同一份真实 session 再测到的热态结果是：

| 路径 | 暖态均值 | FPS |
|---|---:|---:|
| GUI 强 cache 单步路径（集成后） | 18.77 ms | 53.28 |

这说明之前“GUI 明明切了强 cache，为什么还是只有个位数 fps”的真正原因，不是 cache 失效，而是**GUI 进程的数值线程配置没有收成实时模式**。

## 为什么 GUI 现在只有 1.53 fps

### 1. GUI 并没有走“强 cache 单步路径”

当前硬件实时重构入口在：

- `/home/tom/workspace/PyEidors_wsl2/src/eit_app/controllers/reconstruction_controller.py`

它会构建/复用 `EITSystem`，然后调用：

- `system.difference_reconstruct(...)`

继续进入：

- `/home/tom/workspace/PyEidors_wsl2/src/pyeidors/core_system_facade.py`
- `/home/tom/workspace/PyEidors_wsl2/src/pyeidors/inverse/workflows/difference.py`
- `/home/tom/workspace/PyEidors_wsl2/src/pyeidors/core_system.py`
- `/home/tom/workspace/PyEidors_wsl2/src/pyeidors/inverse/solvers/gauss_newton_engine.py`
- `/home/tom/workspace/PyEidors_wsl2/src/pyeidors/inverse/solvers/gauss_newton_runtime.py`

这是一条**完整 GN 运行时链**。

而用户记忆里的“强 cache”路径来自：

- `/home/tom/workspace/PyEidors_wsl2/scripts/common/gn_difference_runner.py`

这里缓存并复用了：

- `base_meas`
- `jacobian`
- `Jt`
- `NOSER diag`
- `A`
- `LU`

本质上是“单步算子已组装 + 已因式分解”的快路径。

### 2. GUI 默认链还保留了昂贵的步长搜索

在：

- `/home/tom/workspace/PyEidors_wsl2/src/pyeidors/core_system.py`

默认差分 preset `eidors_one_step_noser` 会把：

- `difference_step_size_mode`

设成 `optimize`，除非显式覆盖。

实测 `gn_default` 的诊断里可以看到：

- `difference_step_size.mode = "optimize"`
- `eval_count = 19`

这意味着每帧还会额外做多次目标函数评估。

### 3. GUI 默认链还是 strict solver

当前 GUI 的 `run_reconstruction_request(...)` 没有显式把 `solver_mode` 切到 `fast`。
所以 `EITSystem` 默认走的是 `strict`。

这也是为什么：

- GUI 默认链：`1.53 fps`
- GN fast 模式：`16.74 fps`

中间直接差了一个数量级。

### 4. cache 是命中的，但命中的对象不等于“每帧只做 8ms 计算”

GUI 默认链的 cache 统计里：

- `process_hits = 900`
- `process_misses = 10`
- `hit_rate ≈ 0.986`

说明 cache 并没有失效。
问题在于：**即使 cache 命中，GUI 当前路径仍然会执行完整 GN 运行时逻辑**，所以单帧仍然要几百毫秒。

## cProfile 看到的瓶颈

暖态 GUI 单帧 profile 的主要热点：

- `_solve_strict_path`
- `torch._C._linalg.linalg_solve`
- 多次 `fwd_solve`
- `_apply_difference_step_size`
- `scipy.optimize.minimize_scalar`

这说明当前慢点主要来自：

1. strict 线性求解
2. 额外 forward validate
3. difference step-size optimize

而不是图像渲染。渲染链之前已经单独测过，缓存后纯渲染大约 `133 fps`。

## 一个很重要的补充发现

GUI 请求里的：

- `mesh_refinement = 4`

并不等于最终真的拿 `ref4` mesh。
当前 GUI 会先换算 mesh size，再反推出 effective refinement，所以这次真实 benchmark 里：

- GUI 请求 refinement: `4`
- GUI 实际 refinement: `8`

这点在：

- `/home/tom/workspace/PyEidors_wsl2/src/eit_app/controllers/reconstruction_controller.py`

里可以看到。

这也是为什么“用户以为自己在测 ref4，其实实际跑的是 ref8”。

## 对当前项目最直接的工程结论

如果目标是实时成像：

- 现在渲染已经不是第一瓶颈
- 真正应该切换的是**计算路径**

优先级建议：

1. 让 GUI 实时重构链优先走 `solver_mode="fast"`
2. 对实时模式单独接入强 cache 单步路径，而不是继续复用完整 GN 默认链
3. 保留完整 GN 严格链作为“高精度/离线/手动重构”模式

## 复现实测命令

```bash
cd /home/tom/workspace/PyEidors_wsl2
nix develop -c python scripts/benchmarks/benchmark_reconstruction_cache_paths.py \
  --session-dir data/measurements/test_for_gui/session_20260413_140902 \
  --mesh-dir eit_meshes \
  --mesh-refinement 4 \
  --compute-cycles 2 \
  --output-json docs/benchmarks/reconstruction_cache_paths_session_20260413_140902.json
```
