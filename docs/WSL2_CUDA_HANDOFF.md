# PyEIDORS mac 封版与 WSL2/CUDA 迁移交接

## 1. 当前结论

当前 Apple Silicon + macOS + Nix + `uv` 版本已经完成一轮完整的 3D CPU 路线性能收敛。

固定结论如下：

- 当前最佳交付路径是 `D_combined`：`woodbury + jacobian block autotune + auto/cholmod-precond`
- `ROM/POD + Inexact GN + Low-rank` 保留为 `experimental`，默认关闭
- 当前机器上继续做 CPU 侧微优化，已经很难带来端到端总时长的质变
- 如果目标是让 3D EIT 正逆问题出现明显的端到端提速，下一阶段最有希望的方向是迁移到带 CUDA 的 WSL2/Linux 机器，并把 forward / assembly / inverse pipeline 的核心路径 GPU 化

## 2. 当前封版版本的推荐使用方式

### 默认 3D fast 路线

- `solver_mode=fast`
- `fast_linear_path=auto`
- `preconditioner=auto`
- `jacobian_block_tune=auto`
- `rom_mode=off`
- `inexact_mode=off`
- `lowrank_mode=off`

### 当前 fair compare 结论

最新本机 fair compare 报告显示：

- `ref_1`
  - `A_baseline`: `16.596s`
  - `D_combined`: `16.644s`，总时长约 `0.997x`
  - `E_fused`: `16.582s`，总时长约 `1.001x`
- `ref_2`
  - `A_baseline`: `40.462s`
  - `D_combined`: `39.787s`，总时长约 `1.017x`
  - `E_fused`: `40.198s`，总时长约 `1.007x`

阶段收益依然明确：

- `D_combined` 对线性求解有约 `10x-11x` 的改善
- `D_combined` 对 Jacobian assembly 有约 `1.8x` 的改善
- 但当前端到端总时长已经更多受 `forward solve / outer loop / line search / repeated solves` 主导

因此当前版本封版的逻辑是：

- 保留真正能稳定支持交付的主路径
- 保留实验路径作为研究资产
- 不再把实验路径当作默认交付方案

## 3. 当前代码结构与迁移优先切入点

### 3.1 建议保持稳定的模块

这些模块是下一台机器继续开发时应尽量保持接口稳定的基座：

- `scripts/run_reconstruction_unified.py`
  - 统一 CLI 合约
  - 当前 3D 默认策略入口
- `src/pyeidors/core_system.py`
  - 系统装配和求解器接入总入口
- `src/pyeidors/forward/eit_forward_model.py`
  - forward solve 主实现
- `src/pyeidors/inverse/solvers/gauss_newton_engine.py`
  - GN 迭代求解器接口
- `src/pyeidors/inverse/solvers/gauss_newton_runtime.py`
  - fast/strict/experimental 路径分发与 diagnostics
- `src/pyeidors/inverse/jacobian/direct_jacobian.py`
  - Jacobian 组装主路径
- `src/pyeidors/perf/policy.py`
  - 当前性能默认值、profile 与 gate 约定

### 3.2 当前主交付路径

当前主交付路径是 CPU 友好的稀疏/测量空间路线：

- NOSER 对角正则时优先 `woodbury` 测量空间解
- 稀疏/非对角路径走 `pcg` + `preconditioner`
- Jacobian 装配走 block autotune
- CHOLMOD 只作为辅助预条件和加速路径，不是主总时长卖点

### 3.3 实验路径

以下目录和开关属于研究路径，不应作为下一阶段 GPU 化的默认基础：

- `src/pyeidors/inverse/reduced/`
- `rom_mode`
- `inexact_mode`
- `lowrank_mode`
- `E_fused` benchmark profile

它们应保留，但默认不要打开。未来只有在 GPU 主路径成熟后，再评估是否把 reduced-order 思路叠加回去。

## 4. 当前 CPU 路线的瓶颈结论

从本机 fair compare 和阶段测试可以得到较稳定的结论：

1. 线性求解已经不是唯一主瓶颈。
2. Jacobian assembly 已经被显著压缩，但仍不是决定性瓶颈。
3. 端到端总时长更可能被以下环节主导：
   - 多次 forward solve
   - 外层 GN 迭代控制
   - line search 导致的重复计算
   - CPU 上有限元装配与稀疏算子应用的整体吞吐

这也是为什么当前机器上继续做 CPU 代码级微调，很难得到“质变”提速。

## 5. WSL2/CUDA 迁移前置条件

下一台机器建议满足：

- Windows + WSL2 Ubuntu，或原生 Linux
- NVIDIA GPU，显存足够覆盖目标 3D 网格规模
- CUDA toolkit 与驱动版本匹配
- 能构建或安装支持 GPU 路径的科学计算栈
- 优先使用 Linux 环境，不建议把下一阶段 GPU 化建立在 macOS 上

建议在迁移前固定以下事项：

- 当前仓库提交点与 benchmark 报告
- 当前 CLI 契约
- 当前 test/gate 基线
- 当前 mesh / pattern / measurement 数据样例

## 6. GPU 化的推荐推进顺序

本轮不实现 CUDA，但建议下一阶段按如下顺序推进。

### Phase 1: Forward 优先

先让 forward path 受益，而不是一开始就全面重写 inverse。

优先目标：

- 系统矩阵组装与多 RHS solve
- 激励批处理
- 反复 forward 调用的缓冲复用
- 减少 CPU/GPU 间往返拷贝

原因：

- 当前端到端总时长更受 forward/outer-loop 主导
- forward 一旦提速，absolute 与 difference 都能受益

### Phase 2: Jacobian / adjoint 组装

在 forward 稳定后，再推进：

- Jacobian 所需梯度/投影核函数 GPU 化
- adjoint/measurement projection 路径 GPU 化
- 保持当前 diagnostics 和 benchmark 口径，便于与 CPU 路径做 A/B

### Phase 3: Inverse runtime GPU 化

最后再把 inverse runtime 主链迁移过去：

- fast linear path 在 GPU 端重建
- 迭代控制、line search、缓存策略保持现有接口
- 保持 `strict` 作为 CPU 回退或数值对照路径

## 7. 建议保持不变的契约

为了让下一阶段迁移成本最低，建议以下契约保持稳定：

- `run_reconstruction_unified.py` 的主要 CLI 参数与输出目录结构
- `EITSystem` 对外的求解入口
- benchmark 报告 JSON 的主字段
- `D_combined` / `E_fused` 等 profile 命名
- diagnostics 中的 `fast_solver_path`、`fallback_reason`、`rom_enabled_effective` 等字段

这样可以在 GPU 版本落地后继续沿用当前 benchmark/gate 体系做同代码 A/B 对比。

## 8. 哪些代码适合下一阶段重构为 GPU 后端

优先候选：

- `src/pyeidors/forward/eit_forward_model.py`
- `src/pyeidors/inverse/jacobian/direct_jacobian.py`
- `src/pyeidors/inverse/solvers/gauss_newton_runtime.py`

不建议首先重构：

- CLI 层
- cache key / signature 机制
- benchmark/gate 报告结构
- current `D_combined` 默认策略定义

## 9. 与 WCCN / 复值网络的后续衔接

这部分不属于本轮实现范围，只做交接记录。

后续建议路线：

1. 用当前 PyEIDORS 版本在本机和未来 CUDA 机器上稳定生成训练数据
2. 在 `WCCN` 中构建复值神经网络并用这些数据训练
3. 再考虑把两个项目打通，形成一个受 PyEIDORS 物理结构与约束限制的复值网络求解框架

在这个阶段，PyEIDORS 的价值主要是：

- 生成高质量训练数据
- 提供 forward / inverse 的物理约束与可解释基线
- 提供 benchmark 与数值对照框架

## 10. 当前 handoff 的执行建议

把仓库拷贝到下一台机器后，建议先做三件事：

1. 先在新机器上复现当前 CPU 路径，确保 benchmark/gate 与当前封版版本一致。
2. 再只替换 forward 相关后端，建立第一条 GPU A/B 路径。
3. 只有在 forward GPU 化带来稳定收益后，再推进 Jacobian 与 inverse runtime 的 GPU 迁移。

当前版本的目标已经完成：它是一个适合交付、适合迁移、适合作为下一阶段 GPU 化起点的 CPU 封版版本。
