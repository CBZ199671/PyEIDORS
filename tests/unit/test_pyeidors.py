#!/usr/bin/env python3
"""
PyEidors模块测试脚本
测试各个模块的基本功能
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """测试模块导入"""
    print("测试模块导入...")
    
    try:
        import pyeidors
        print("✓ pyeidors主模块导入成功")
        
        # 检查环境
        env_info = pyeidors.check_environment()
        print(f"✓ 环境检查: {env_info}")
        
    except ImportError as e:
        print(f"✗ 模块导入失败: {e}")
        return False
    
    try:
        from pyeidors.core_system import EITSystem
        print("✓ EITSystem导入成功")
    except ImportError as e:
        print(f"✗ EITSystem导入失败: {e}")
        return False
    
    try:
        from pyeidors.data.structures import PatternConfig, EITData, EITImage, MeshConfig, ElectrodePosition
        print("✓ 数据结构导入成功")
    except ImportError as e:
        print(f"✗ 数据结构导入失败: {e}")
        return False
    
    try:
        from pyeidors.forward.eit_forward_model import EITForwardModel
        print("✓ 前向模型导入成功")
    except ImportError as e:
        print(f"✗ 前向模型导入失败: {e}")
        return False
    
    try:
        from pyeidors.inverse.solvers.gauss_newton import ModularGaussNewtonReconstructor
        print("✓ 高斯牛顿求解器导入成功")
    except ImportError as e:
        print(f"✗ 高斯牛顿求解器导入失败: {e}")
        return False
    
    try:
        from pyeidors.inverse.jacobian.direct_jacobian import DirectJacobianCalculator
        print("✓ 雅可比计算器导入成功")
    except ImportError as e:
        print(f"✗ 雅可比计算器导入失败: {e}")
        return False
    
    try:
        from pyeidors.inverse.regularization.smoothness import SmoothnessRegularization
        print("✓ 平滑性正则化导入成功")
    except ImportError as e:
        print(f"✗ 平滑性正则化导入失败: {e}")
        return False
    
    try:
        from pyeidors.electrodes.patterns import StimMeasPatternManager
        print("✓ 激励测量模式管理器导入成功")
    except ImportError as e:
        print(f"✗ 激励测量模式管理器导入失败: {e}")
        return False
    
    return True

def test_data_structures():
    """测试数据结构"""
    print("\n测试数据结构...")
    
    try:
        from pyeidors.data.structures import PatternConfig, EITData, EITImage, MeshConfig, ElectrodePosition
        
        # 测试PatternConfig
        config = PatternConfig(n_elec=16)
        print(f"✓ PatternConfig创建成功: {config}")
        
        # 测试EITData
        data = EITData(
            meas=np.random.rand(10),
            stim_pattern=np.random.rand(16, 4),
            n_elec=16,
            n_stim=4,
            n_meas=10
        )
        print(f"✓ EITData创建成功: {data.type}")
        
        # 测试EITImage
        img = EITImage(elem_data=np.ones(100), fwd_model=None)
        conductivity = img.get_conductivity()
        print(f"✓ EITImage创建成功，导电率形状: {conductivity.shape}")
        
        # 测试MeshConfig
        mesh_config = MeshConfig(radius=1.0, refinement=8)
        print(f"✓ MeshConfig创建成功: {mesh_config}")
        
        # 测试ElectrodePosition
        electrode_pos = ElectrodePosition.create_circular(n_elec=16)
        print(f"✓ ElectrodePosition创建成功，电极数量: {electrode_pos.L}")
        
        return True
    except Exception as e:
        print(f"✗ 数据结构测试失败: {e}")
        return False

def test_eit_system():
    """测试EIT系统"""
    print("\n测试EIT系统...")
    
    try:
        from pyeidors.core_system import EITSystem
        from pyeidors.data.structures import PatternConfig, MeshConfig
        
        # 创建EIT系统
        pattern_config = PatternConfig(n_elec=16)
        mesh_config = MeshConfig()
        
        eit_system = EITSystem(
            n_elec=16,
            pattern_config=pattern_config,
            mesh_config=mesh_config
        )
        
        print(f"✓ EIT系统创建成功")
        
        # 获取系统信息
        info = eit_system.get_system_info()
        print(f"✓ 系统信息: {info}")
        
        # 测试创建均匀图像（这会在setup之前失败，这是预期的）
        try:
            img = eit_system.create_homogeneous_image()
            print("✗ 这不应该成功，因为系统未初始化")
        except RuntimeError as e:
            print(f"✓ 正确捕获了未初始化错误: {e}")
        
        return True
    except Exception as e:
        print(f"✗ EIT系统测试失败: {e}")
        return False

def test_pattern_manager():
    """测试激励测量模式管理器"""
    print("\n测试激励测量模式管理器...")
    
    try:
        from pyeidors.electrodes.patterns import StimMeasPatternManager
        from pyeidors.data.structures import PatternConfig
        
        config = PatternConfig(n_elec=16)
        manager = StimMeasPatternManager(config)
        
        print(f"✓ 模式管理器创建成功")
        print(f"✓ 激励数量: {manager.n_stim}")
        print(f"✓ 总测量数量: {manager.n_meas_total}")
        print(f"✓ 激励矩阵形状: {manager.stim_matrix.shape}")
        
        return True
    except Exception as e:
        print(f"✗ 模式管理器测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("=== PyEidors模块测试开始 ===")
    
    tests = [
        test_imports,
        test_data_structures,
        test_eit_system,
        test_pattern_manager
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n=== 测试结果汇总 ===")
    total = len(results)
    passed = sum(results)
    failed = total - passed
    
    print(f"总测试数: {total}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    
    if failed == 0:
        print("✓ 所有测试通过！")
        return True
    else:
        print(f"✗ 有{failed}个测试失败")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)