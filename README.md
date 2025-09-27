# PyEidors

基于FEniCS的电阻抗成像 (EIT) 正逆问题求解系统，提供类似EIDORS的Python化实现并整合PyTorch加速。

## 项目概览

- 面向研究与工程实践，覆盖网格生成、前向建模、雅可比计算、正则化和高斯牛顿重建的完整链路。
- 采用模块化设计，核心系统 `EITSystem` 负责协调几何、前向与逆问题组件，方便替换或扩展各环节。
- 支持GMsh+meshio+FEniCS的网格工作流，内置激励/测量模式管理器与合成数据、可视化工具。
- 提供示例、测试和报告资料，帮助验证电极布局、网格质量和端到端重建流程。

## 系统架构总览

```
网格加载/生成 ──► 前向模型 (Complete Electrode Model)
                      │
                      ▼
              雅可比计算与正则化
                      │
                      ▼
              模块化高斯牛顿重建
                      │
                      ▼
        可视化 · 合成数据 · 结果分析
```

- 几何模块准备FEniCS兼容的网格（可加载现有H5/XDMF或基于GMsh实时生成）。
- `EITForwardModel` 构建有限元离散、应用激励/测量模式并输出电极电压与测量值。
- `DirectJacobianCalculator` 与正则化模块提供灵活的敏度矩阵和惩罚项。
- `ModularGaussNewtonReconstructor` 借助PyTorch实现GPU/CPU自适应的高斯牛顿迭代，完成逆问题求解。

## 核心模块与用途

- `src/pyeidors/core_system.py`: `EITSystem` 聚合前向模型、雅可比、正则化和求解器，提供 `setup`, `forward_solve`, `inverse_solve` 等统一接口。
- `src/pyeidors/forward/eit_forward_model.py`: 完全电极模型实现，封装FEniCS函数空间、系统矩阵装配与测量模式应用。
- `src/pyeidors/electrodes/patterns.py`: 激励/测量模式生成与筛选，支持相邻/对跷等常见策略，并输出测量子集。
- `src/pyeidors/inverse/jacobian/direct_jacobian.py`: 基于伴随场的雅可比计算，提供传统与高效两种模式以匹配不同场景。
- `src/pyeidors/inverse/regularization/smoothness.py`: 拉普拉斯型平滑正则化，同时示例化Tikhonov与TV接口。
- `src/pyeidors/inverse/solvers/gauss_newton.py`: 模块化高斯牛顿求解器，内置线搜索、裁剪约束、残差记录和正则矩阵缓存。
- `src/pyeidors/data/structures.py`: 项目统一的数据结构（`PatternConfig`, `EITData`, `EITImage`, `MeshConfig` 等）。
- `src/pyeidors/data/synthetic_data.py`: 生成包含噪声控制与自定义幻象的合成数据集，便于验证算法链路。
- `src/pyeidors/geometry/*.py`: 覆盖网格加载 (`mesh_loader.py`)、GMsh生成与FEniCS转换 (`mesh_generator.py`, `simple_mesh_generator.py`, `optimized_mesh_generator.py`) 以及转换工具。
- `src/pyeidors/visualization/eit_plots.py`: 提供网格、导电率、测量数据及重建对比的可视化接口。
- `src/pyeidors/utils/chinese_font_config.py`: 为matplotlib配置中文字体，方便中文报告输出。

## 典型工作流程

1. 使用 `MeshLoader` 加载 `eit_meshes/` 内的H5/XDMF网格，或通过 `create_eit_mesh` / `create_simple_eit_mesh` 动态生成。
2. 实例化 `EITSystem` 并调用 `setup` 完成网格、前向模型与求解器初始化。
3. 通过 `create_homogeneous_image`/`add_phantom` 或 `synthetic_data` 模块构造导电率分布与测量数据。
4. 调用 `forward_solve` 生成模拟测量，或使用 `inverse_solve` 在给定测量 + 参考数据下执行高斯牛顿重建。
5. 借助 `visualization` 模块绘制网格、测量分布、重建结果或收敛曲线，并结合 `tests/results` 核验链路。

## 数据、可视化与测试

- 合成数据: `create_synthetic_data` 支持设置噪声水平、异常位置与导电率，返回清洁/含噪数据及信噪比指标。
- 可视化: `EITVisualizer` 内置网格、电导率、测量、重建对比及收敛曲线绘制，可输出PNG报告。
- 测试: `tests/unit/test_complete_eit_system.py` 提供端到端流程验证，`tests/unit/test_optimized_mesh_generator.py` 等覆盖几何及电极布局。
- 示例: `examples/basic_usage.py` 演示模块结构、环境检查及系统初始化步骤。

## 环境说明

本项目基于Docker环境开发，使用以下核心组件：

- **FEniCS**: ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30
- **CUQIpy**: 1.3.0
- **CUQIpy-FEniCS**: 0.8.0
- **PyTorch**: 2.7.1+cu128 (GPU支持)
- **Python**: 3.10+ (通过Docker提供)

以下安装方式基于FEniCS官方镜像，亦可使用仓库中的 `Dockerfile` 构建全量环境或直接拉取预制镜像。

## Docker环境设置

```bash
# 启动容器
docker run -ti \
  --gpus all \
  --shm-size=24g \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network=host \
  --cpus=20 \
  --memory=28g \
  -v "D:\workspace\PyEIDORS:/root/shared" \
  -w /root/shared \
  --name pyeidors \
  ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30

# （可选）安装中文字体依赖
apt-get update && apt-get install ttf-wqy-zenhei

# 安装CUQIpy和CUQIpy-FEniCS
pip install cuqipy cuqipy-fenics

# 创建虚拟环境
python3 -m venv /opt/final_venv --system-site-packages
source /opt/final_venv/bin/activate

# 安装PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
