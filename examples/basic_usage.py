#!/usr/bin/env python3
"""
PyEidors基本使用示例
演示如何使用模块化的EIT系统进行前向求解和逆问题重建
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def basic_usage_example():
    """基本使用示例"""
    print("=== PyEidors基本使用示例 ===")
    
    # 导入必要的模块
    from pyeidors import EITSystem, check_environment
    from pyeidors.data.structures import PatternConfig, MeshConfig
    
    # 检查环境
    print("1. 环境检查")
    env = check_environment()
    print(f"   FEniCS可用: {env['fenics_available']}")
    print(f"   PyTorch可用: {env['torch_available']}")
    print(f"   CUDA可用: {env['cuda_available']}")
    if env['torch_available']:
        print(f"   PyTorch版本: {env['torch_version']}")
        print(f"   GPU数量: {env['cuda_device_count']}")
    print()
    
    # 配置EIT系统
    print("2. 配置EIT系统")
    n_elec = 16  # 16个电极
    
    # 激励测量模式配置
    pattern_config = PatternConfig(
        n_elec=n_elec,
        stim_pattern='{ad}',  # 相邻激励模式
        meas_pattern='{ad}',  # 相邻测量模式
        amplitude=1.0         # 激励电流幅值
    )
    
    # 网格配置
    mesh_config = MeshConfig(
        radius=1.0,          # 圆形域半径
        refinement=8,        # 网格细化级别
        mesh_size=0.1       # 网格尺寸
    )
    
    # 创建EIT系统
    eit_system = EITSystem(
        n_elec=n_elec,
        pattern_config=pattern_config,
        mesh_config=mesh_config
    )
    
    print(f"   电极数量: {n_elec}")
    print(f"   激励模式: {pattern_config.stim_pattern}")
    print(f"   测量模式: {pattern_config.meas_pattern}")
    print()
    
    # 获取系统信息
    print("3. 系统信息")
    info = eit_system.get_system_info()
    for key, value in info.items():
        if key != 'pattern_config' and key != 'mesh_config':
            print(f"   {key}: {value}")
    print()
    
    # 注意：实际使用时需要提供网格对象
    print("4. 注意事项")
    print("   - 当前版本需要外部提供网格对象")
    print("   - 可以使用现有的网格文件或自定义网格生成器")
    print("   - 示例网格文件位于 eit_meshes/ 目录")
    print()

def show_module_structure():
    """显示模块结构"""
    print("=== PyEidors模块结构 ===")
    
    structure = {
        "pyeidors/": {
            "__init__.py": "主模块入口，环境检查",
            "core_system.py": "核心EIT系统类",
            "data/": {
                "structures.py": "数据结构定义 (EITData, EITImage, 配置类)",
                "synthetic_data.py": "合成数据生成"
            },
            "forward/": {
                "eit_forward_model.py": "EIT前向模型 (完全电极模型)"
            },
            "inverse/": {
                "jacobian/": {
                    "base_jacobian.py": "雅可比计算器基类",
                    "direct_jacobian.py": "直接方法雅可比计算器"
                },
                "regularization/": {
                    "base_regularization.py": "正则化基类",
                    "smoothness.py": "平滑性正则化"
                },
                "solvers/": {
                    "gauss_newton.py": "模块化高斯牛顿求解器"
                }
            },
            "electrodes/": {
                "patterns.py": "激励测量模式管理器"
            },
            "geometry/": {
                "mesh_generator.py": "网格生成器",
                "mesh_converter.py": "网格格式转换器"
            },
            "utils/": "实用工具函数",
            "visualization/": "可视化模块"
        }
    }
    
    def print_structure(struct, indent=0):
        for key, value in struct.items():
            print("  " * indent + f"├── {key}")
            if isinstance(value, dict):
                print_structure(value, indent + 1)
            else:
                print("  " * (indent + 1) + f"    {value}")
    
    print_structure(structure)
    print()

def show_key_features():
    """显示关键特性"""
    print("=== PyEidors关键特性 ===")
    
    features = [
        "🔧 模块化设计",
        "   - 独立的前向模型、逆问题求解器、正则化模块",
        "   - 可插拔的雅可比计算器和正则化策略",
        "   - 清晰的数据结构定义",
        "",
        "⚡ 性能优化",
        "   - PyTorch GPU加速支持",
        "   - 高效的雅可比矩阵计算",
        "   - 稀疏矩阵操作优化",
        "",
        "🧮 数值方法",
        "   - 完全电极模型 (Complete Electrode Model)",
        "   - 高斯牛顿迭代求解",
        "   - 多种正则化策略 (Tikhonov, 平滑性, 全变分)",
        "",
        "🔬 科学计算",
        "   - 基于FEniCS有限元框架",
        "   - 支持自定义网格和边界条件",
        "   - 兼容标准EIT数据格式",
        "",
        "📊 可扩展性",
        "   - 支持多种激励测量模式",
        "   - 可集成CUQI贝叶斯推断框架",
        "   - 灵活的可视化接口"
    ]
    
    for feature in features:
        print(feature)
    print()

if __name__ == "__main__":
    basic_usage_example()
    show_module_structure()
    show_key_features()
    
    print("=== 下一步 ===")
    print("1. 提供网格对象以完成系统初始化")
    print("2. 运行前向求解验证模型正确性")
    print("3. 测试逆问题重建算法")
    print("4. 添加可视化和数据保存功能")
    print("5. 集成更多正则化和求解策略")