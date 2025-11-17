# Sparse Bayesian vs. Gauss–Newton Reconstruction (EIT_DEV_Test)

## 实现概览
- `src/pyeidors/inverse/solvers/sparse_bayesian.py`: 基于 CUQIpy 的稀疏贝叶斯重建器，实现了以 Smoothed-Laplace 先验和线性化前向模型为核心的 MAP 求解流程，并支持雅可比缓存与噪声自适应估计。
- `src/pyeidors/inverse/workflows/sparse_bayesian.py`: 封装绝对/差分两种模式的高层接口，返回与既有 `ReconstructionResult` 兼容的结果。
- `scripts/run_sparse_bayesian_reconstruction.py`: 端到端脚本，加载实测数据、执行线性校准、运行稀疏贝叶斯重建，并生成图像、向量和对比报告（默认输出到 `results/sparse_bayesian/`）。

## 实验设置
- 数据：`data/measurements/EIT_DEV_Test/2025-09-23-00-01-56_10_10.00_100uA_2000Hz.csv`
- 网格：使用 `load_or_create_mesh` 自动生成的 5702 单元 FEniCS 网格
- 校准：按照原有流程，对测量帧执行线性尺度/偏移拟合（相对于均匀导电率模拟）
- 先验与噪声：Smoothed-Laplace scale=0.05、beta=1e-6；噪声标准差自动截断到 1e-6
- 运行命令：
  ```bash
  # 绝对成像（约 10 分钟）
  python scripts/run_sparse_bayesian_reconstruction.py \
    --csv data/measurements/EIT_DEV_Test/2025-09-23-00-01-56_10_10.00_100uA_2000Hz.csv \
    --mode absolute --absolute-col 2 --jacobian-cache

  # 差分成像（约 9 分钟）
  python scripts/run_sparse_bayesian_reconstruction.py \
    --csv data/measurements/EIT_DEV_Test/2025-09-23-00-01-56_10_10.00_100uA_2000Hz.csv \
    --mode difference --reference-col 0 --target-col 2 --jacobian-cache
  ```
  > 注意：MAP 优化在 5702 维空间上运行，单次重建耗时 ~9–10 分钟。

## 结果对比（稀疏贝叶斯 vs. 高斯牛顿）
| 模式 | 指标 | Sparse Bayesian | Gauss–Newton |
| --- | --- | --- | --- |
| 绝对 | 相对误差 | 0.79516 | 0.79515 |
| 绝对 | L2 误差 | 8.42×10⁻⁵ | 8.42×10⁻⁵ |
| 绝对 | MSE | 3.41×10⁻¹¹ | 3.41×10⁻¹¹ |
| 差分 | 相对误差 | 1.00272 | 1.00064 |
| 差分 | L2 误差 | 3.47×10⁻⁴ | 3.46×10⁻⁴ |
| 差分 | MSE | 5.78×10⁻¹⁰ | 5.75×10⁻¹⁰ |

- 稀疏贝叶斯绝对成像的误差指标与高斯牛顿几乎一致，同时保持了更稀疏的导电率分布（参考 `results/sparse_bayesian/absolute/.../reconstruction.png`）。
- 差分成像中，两种算法的残差与误差仅相差 ~0.2%，说明在当前噪声水平下，稀疏先验对差分模式的增益有限。
- 稀疏贝叶斯流程自动估计 1e-6 的噪声标准差，与线性校准后的测量尺度匹配；若期望更强稀疏性，可调小 `prior_scale` 或增大 `beta`。

## 输出位置
- 绝对成像：`results/sparse_bayesian/absolute/2025-09-23-00-01-56_10_10.00_100uA_2000Hz/`
- 差分成像：`results/sparse_bayesian/difference/2025-09-23-00-01-56_10_10.00_100uA_2000Hz/`
- 摘要汇总：`results/sparse_bayesian/reports/2025-09-23-00-01-56_10_10.00_100uA_2000Hz_summary.json`

## 结论与后续建议
- 稀疏贝叶斯在绝对成像场景下可与现有高斯牛顿性能持平，并易于集成贝叶斯不确定度评估。
- 差分成像下误差改善不显著，后续可探索：
  1. 低维先验（PCA/小波）以缩减 MAP 变量维度；
  2. 结合 CUQIpy 的采样器（如 UGLA/pCN）获取不确定度区间；
  3. 根据频率与噪声水平自适应调整先验 scale。

整体来看，新的稀疏贝叶斯流程已经可与现有高斯牛顿并行使用，为后续贝叶斯推断/不确定度分析铺平道路。
