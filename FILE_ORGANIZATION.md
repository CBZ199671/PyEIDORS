# PyEidors 文件系统整理说明

## 📁 目录结构

### 🏗️ 核心源码
```
src/
├── pyeidors/                    # 主要包目录
│   ├── core_system.py          # 核心EIT系统类
│   ├── data/                   # 数据结构、合成/实测数据工具
│   ├── electrodes/             # 电极和激励模式
│   ├── forward/                # 前向问题求解
│   ├── inverse/                # 逆问题重建
│   ├── geometry/               # 网格生成和处理
│   │   ├── optimized_mesh_generator.py  # 优化的网格生成器 ⭐
│   │   ├── mesh_loader.py      # 网格加载器
│   │   └── ...
│   ├── visualization/          # 可视化工具
│   └── utils/                  # 实用工具
└── pyeidors.egg-info/          # 包安装信息
```

### 🧪 测试系统
```
tests/
├── unit/                       # 单元测试
│   ├── test_electrode_position_y_axis.py  # y轴电极位置测试 ⭐
│   ├── test_optimized_mesh_generator.py   # 优化网格测试
│   ├── test_real_mesh_generation.py       # 真实网格测试
│   └── ...
├── integration/                # 集成测试 (待添加)
├── test_geometry/              # 几何测试
│   └── test4.py               # 参考实现
└── run_all_tests.py           # 测试运行器
```

### 🎨 演示和可视化
```
demos/
├── demo_y_axis_electrodes.py          # y轴电极演示 ⭐
├── demo_optimized_mesh.py             # 优化网格演示
├── y_axis_electrode_demo.png          # y轴电极图像 ⭐
├── electrode_position_comparison.png   # 位置对比图像 ⭐
├── electrode_positions_demo.png       # 电极配置演示
├── mesh_generation_demo.png           # 网格生成演示
└── mesh_quality_demo.png              # 网格质量分析
```

### 📊 结果和数据
```
results/
├── meshes/                     # 生成的网格文件
│   ├── *.msh                  # GMsh网格文件
│   ├── *.xdmf                 # XDMF格式文件
│   └── *.ini                  # 关联表文件
├── mesh_generation/           # 网格生成测试结果
├── visualizations/           # 可视化结果 (待整理)
├── test_measurements.png     # 测试测量结果
└── test_report.md           # 测试报告
```

### 📝 文档和报告
```
reports/
├── OPTIMIZED_MESH_GENERATOR_REPORT.md  # 优化网格报告 ⭐
├── FINAL_PROJECT_SUMMARY.md           # 项目总结报告
├── source/measurement_data_spec.md    # 实测数据格式规范
└── ...
```

### 🗂️ 配置和示例
```
examples/                       # 使用示例
├── basic/                     # 基础示例
├── advanced/                  # 高级示例
├── notebooks/                 # Jupyter笔记本
└── basic_usage.py            # 基本使用示例

data/                          # 项目数据
├── measurements/              # 测量数据
├── meshes/                   # 预制网格
└── phantoms/                 # 幻象数据

configs/                      # 配置文件 (待添加)
scripts/                      # 实用脚本
├── run_absolute_reconstruction.py          # 绝对成像重建入口
├── run_difference_reconstruction.py        # 差分成像重建入口
archived/                     # 归档文件
├── README_FINAL.md           # 旧版文档
└── ...
```

## 🚀 最新改进

### ⭐ 电极位置优化
- **文件**: `src/pyeidors/geometry/optimized_mesh_generator.py`
- **改进**: 电极默认初始位置从x轴正半轴改为y轴正半轴
- **测试**: `tests/unit/test_electrode_position_y_axis.py`
- **演示**: `demos/demo_y_axis_electrodes.py`

### 🎯 关键特性
1. **ElectrodePosition类**：精确计算电极位置，支持y轴起始
2. **可视化对比**：展示修改前后的电极位置差异
3. **完整测试**：验证角度计算、顺序排列、旋转效果

## 📋 文件清理日志

### ✅ 已整理的文件类型
- **测试文件** → `tests/unit/`
- **演示脚本** → `demos/`
- **图像文件** → `demos/`
- **报告文档** → `reports/`
- **网格结果** → `results/meshes/`
- **测试结果** → `results/`
- **归档文档** → `archived/`

### 🎯 整理原则
1. **按功能分类**：测试、演示、结果、文档分别存放
2. **保持层次**：unit/integration测试分离
3. **便于维护**：相关文件集中管理
4. **清晰命名**：文件名反映内容和用途

## 🔧 使用指南

### 运行测试
```bash
# 运行电极位置测试
python tests/unit/test_electrode_position_y_axis.py

# 运行所有测试
python tests/run_all_tests.py
```

### 查看演示
```bash
# 运行y轴电极演示
python demos/demo_y_axis_electrodes.py

# 查看生成的图像
ls demos/*.png
```

### 导入优化网格生成器
```python
from pyeidors.geometry.optimized_mesh_generator import (
    ElectrodePosition, OptimizedMeshConfig, 
    OptimizedMeshGenerator, create_eit_mesh
)

# 创建y轴起始的16电极配置
elec_pos = ElectrodePosition(L=16, coverage=0.5)
mesh = create_eit_mesh(n_elec=16, electrode_coverage=0.5)
```

---

**整理时间**: 2025年7月4日  
**主要改进**: 电极位置y轴起始 + 文件系统整理  
**测试状态**: 100%通过  
**文档状态**: 完整更新  

🎊 **文件系统整理完成！** 🎊
