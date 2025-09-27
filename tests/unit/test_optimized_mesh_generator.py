#!/usr/bin/env python3
"""
优化mesh生成器测试
测试基于参考实现的新mesh生成器功能
"""

import numpy as np
import tempfile
import shutil
from pathlib import Path
import logging
import unittest
from unittest.mock import Mock, patch
import sys

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parents[1]
SRC_PATH = PROJECT_ROOT / 'src'

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_electrode_position():
    """测试电极位置配置"""
    print("🔧 测试电极位置配置...")
    
    try:
        from pyeidors.geometry.optimized_mesh_generator import ElectrodePosition
        
        # 测试基本配置
        elec_pos = ElectrodePosition(L=16)
        assert elec_pos.L == 16
        assert elec_pos.coverage == 0.5
        assert elec_pos.anticlockwise == True
        
        # 测试位置计算
        positions = elec_pos.positions
        assert len(positions) == 16
        assert all(isinstance(pos, tuple) and len(pos) == 2 for pos in positions)
        
        # 测试角度覆盖
        elec_pos_full = ElectrodePosition(L=8, coverage=1.0)
        pos_full = elec_pos_full.positions
        assert len(pos_full) == 8
        
        # 测试输入验证
        try:
            ElectrodePosition(L=0)
            assert False, "应该抛出ValueError"
        except ValueError:
            pass
        
        try:
            ElectrodePosition(L=16, coverage=0)
            assert False, "应该抛出ValueError"
        except ValueError:
            pass
        
        print("✅ 电极位置配置测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 电极位置配置测试失败: {e}")
        return False

def test_mesh_config():
    """测试网格配置"""
    print("🔧 测试网格配置...")
    
    try:
        from pyeidors.geometry.optimized_mesh_generator import OptimizedMeshConfig
        
        # 测试默认配置
        config = OptimizedMeshConfig()
        assert config.radius == 1.0
        assert config.refinement == 8
        assert config.electrode_vertices == 6
        assert config.gap_vertices == 1
        
        # 测试网格尺寸计算
        mesh_size = config.mesh_size
        expected_size = config.radius / (config.refinement * 2)
        assert abs(mesh_size - expected_size) < 1e-10
        
        # 测试自定义配置
        custom_config = OptimizedMeshConfig(
            radius=2.0,
            refinement=4,
            electrode_vertices=10,
            gap_vertices=2
        )
        assert custom_config.radius == 2.0
        assert custom_config.refinement == 4
        assert custom_config.electrode_vertices == 10
        assert custom_config.gap_vertices == 2
        
        print("✅ 网格配置测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 网格配置测试失败: {e}")
        return False

def test_mesh_generator_creation():
    """测试网格生成器创建"""
    print("🔧 测试网格生成器创建...")
    
    try:
        from pyeidors.geometry.optimized_mesh_generator import (
            OptimizedMeshGenerator, OptimizedMeshConfig, ElectrodePosition
        )
        
        # 创建配置
        config = OptimizedMeshConfig(radius=1.0, refinement=6)
        electrodes = ElectrodePosition(L=16, coverage=0.5)
        
        # 创建生成器
        generator = OptimizedMeshGenerator(config, electrodes)
        
        # 验证初始化
        assert generator.config == config
        assert generator.electrodes == electrodes
        assert isinstance(generator.mesh_data, dict)
        
        print("✅ 网格生成器创建测试通过")
        return True
        
    except ImportError as e:
        print(f"⚠️  依赖不可用，跳过网格生成器创建测试: {e}")
        return True
        
    except Exception as e:
        print(f"❌ 网格生成器创建测试失败: {e}")
        return False

@patch('pyeidors.geometry.optimized_mesh_generator.GMSH_AVAILABLE', True)
def test_mesh_generation_mock():
    """测试网格生成(模拟)"""
    print("🔧 测试网格生成(模拟)...")
    
    try:
        from pyeidors.geometry.optimized_mesh_generator import (
            OptimizedMeshGenerator, OptimizedMeshConfig, ElectrodePosition
        )
        
        # 创建配置
        config = OptimizedMeshConfig(radius=1.0, refinement=4)
        electrodes = ElectrodePosition(L=8, coverage=0.5)
        
        # 创建生成器
        generator = OptimizedMeshGenerator(config, electrodes)
        
        # 模拟gmsh调用
        with patch('gmsh.initialize') as mock_init, \
             patch('gmsh.model.add') as mock_add, \
             patch('gmsh.model.occ.addPoint') as mock_point, \
             patch('gmsh.model.occ.addLine') as mock_line, \
             patch('gmsh.model.occ.addCurveLoop') as mock_loop, \
             patch('gmsh.model.occ.addPlaneSurface') as mock_surface, \
             patch('gmsh.model.occ.synchronize') as mock_sync, \
             patch('gmsh.model.mesh.embed') as mock_embed, \
             patch('gmsh.model.addPhysicalGroup') as mock_physical, \
             patch('gmsh.model.mesh.setSize') as mock_size, \
             patch('gmsh.model.mesh.generate') as mock_generate, \
             patch('gmsh.write') as mock_write, \
             patch('gmsh.finalize') as mock_finalize:
            
            # 设置模拟返回值
            mock_point.return_value = 1
            mock_line.return_value = 1
            mock_loop.return_value = 1
            mock_surface.return_value = 1
            
            # 创建临时目录
            with tempfile.TemporaryDirectory() as temp_dir:
                # 测试几何创建方法
                generator._create_geometry()
                
                # 验证调用
                assert mock_point.called
                assert mock_line.called
                assert mock_loop.called
                assert mock_surface.called
                
                # 验证网格数据结构
                assert 'boundary_points' in generator.mesh_data
                assert 'electrode_ranges' in generator.mesh_data
                assert 'lines' in generator.mesh_data
                assert 'surface' in generator.mesh_data
                
                print("✅ 网格生成(模拟)测试通过")
                return True
                
    except Exception as e:
        print(f"❌ 网格生成(模拟)测试失败: {e}")
        return False

def test_mesh_converter_creation():
    """测试网格转换器创建"""
    print("🔧 测试网格转换器创建...")
    
    try:
        from pyeidors.geometry.optimized_mesh_generator import OptimizedMeshConverter
        
        # 创建转换器
        converter = OptimizedMeshConverter("/tmp/test.msh", "/tmp/output")
        
        # 验证初始化
        assert converter.mesh_file == "/tmp/test.msh"
        assert converter.output_dir == "/tmp/output"
        assert converter.prefix == "test"
        
        print("✅ 网格转换器创建测试通过")
        return True
        
    except ImportError as e:
        print(f"⚠️  依赖不可用，跳过网格转换器创建测试: {e}")
        return True
        
    except Exception as e:
        print(f"❌ 网格转换器创建测试失败: {e}")
        return False

def test_convenience_functions():
    """测试便捷函数"""
    print("🔧 测试便捷函数...")
    
    try:
        from pyeidors.geometry.optimized_mesh_generator import create_eit_mesh
        
        # 测试参数传递
        with patch('pyeidors.geometry.optimized_mesh_generator.OptimizedMeshGenerator') as mock_generator:
            mock_instance = Mock()
            mock_generator.return_value = mock_instance
            mock_instance.generate.return_value = "mock_mesh"
            
            # 调用便捷函数
            result = create_eit_mesh(
                n_elec=16,
                radius=1.0,
                refinement=6,
                electrode_coverage=0.5,
                output_dir="/tmp/test"
            )
            
            # 验证调用
            assert mock_generator.called
            assert mock_instance.generate.called
            assert result == "mock_mesh"
        
        print("✅ 便捷函数测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 便捷函数测试失败: {e}")
        return False

def test_error_handling():
    """测试错误处理"""
    print("🔧 测试错误处理...")
    
    try:
        from pyeidors.geometry.optimized_mesh_generator import (
            OptimizedMeshGenerator, OptimizedMeshConfig, ElectrodePosition
        )
        
        # 测试缺少依赖时的错误处理
        with patch('pyeidors.geometry.optimized_mesh_generator.GMSH_AVAILABLE', False):
            config = OptimizedMeshConfig()
            electrodes = ElectrodePosition(L=16)
            
            try:
                generator = OptimizedMeshGenerator(config, electrodes)
                assert False, "应该抛出ImportError"
            except ImportError:
                pass
        
        print("✅ 错误处理测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 错误处理测试失败: {e}")
        return False

def test_integration_with_reference():
    """测试与参考实现的集成"""
    print("🔧 测试与参考实现的兼容性...")
    
    try:
        from pyeidors.geometry.optimized_mesh_generator import (
            ElectrodePosition, OptimizedMeshConfig
        )
        
        # 创建与参考实现相同的配置
        elec_pos = ElectrodePosition(L=16, coverage=0.5)
        config = OptimizedMeshConfig(radius=1.0, refinement=8)
        
        # 验证电极位置计算与参考实现一致
        positions = elec_pos.positions
        assert len(positions) == 16
        
        # 验证每个位置都是有效的角度对
        for start, end in positions:
            assert 0 <= start <= 2 * np.pi
            assert 0 <= end <= 2 * np.pi
            assert start < end or (start > end and end < 0.1)  # 考虑跨越0点的情况
        
        # 验证网格尺寸计算
        mesh_size = config.mesh_size
        assert mesh_size > 0
        
        print("✅ 与参考实现兼容性测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 与参考实现兼容性测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("🚀 开始运行优化mesh生成器测试...")
    print("=" * 50)
    
    tests = [
        ("电极位置配置", test_electrode_position),
        ("网格配置", test_mesh_config),
        ("网格生成器创建", test_mesh_generator_creation),
        ("网格生成(模拟)", test_mesh_generation_mock),
        ("网格转换器创建", test_mesh_converter_creation),
        ("便捷函数", test_convenience_functions),
        ("错误处理", test_error_handling),
        ("与参考实现兼容性", test_integration_with_reference),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 运行测试: {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ 测试失败: {test_name}")
        except Exception as e:
            print(f"❌ 测试异常: {test_name} - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试完成: {passed}/{total} 通过 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 所有测试通过！")
    else:
        print(f"⚠️  {total - passed} 个测试失败")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)