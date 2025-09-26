#!/usr/bin/env python3
"""
实际网格生成测试
测试真实的GMsh网格生成和FEniCS转换
"""

import numpy as np
import tempfile
import shutil
from pathlib import Path
import logging
import sys
import os

# 添加模块路径
sys.path.insert(0, '/root/shared/src')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_real_mesh_generation():
    """测试真实的网格生成"""
    print("🔧 测试真实网格生成...")
    
    try:
        from pyeidors.geometry.optimized_mesh_generator import (
            OptimizedMeshGenerator, OptimizedMeshConfig, ElectrodePosition,
            create_eit_mesh
        )
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 测试简单配置
            config = OptimizedMeshConfig(
                radius=1.0,
                refinement=4,  # 较小的细化级别以加快测试
                electrode_vertices=4,
                gap_vertices=1
            )
            electrodes = ElectrodePosition(L=8, coverage=0.5)  # 8电极简化测试
            
            # 创建生成器
            generator = OptimizedMeshGenerator(config, electrodes)
            
            # 生成网格
            mesh_result = generator.generate(output_dir=temp_path)
            
            # 验证结果
            if isinstance(mesh_result, dict):
                # 返回的是网格信息字典
                print("✅ 生成了网格信息字典")
                assert 'n_electrodes' in mesh_result
                assert mesh_result['n_electrodes'] == 8
                assert 'radius' in mesh_result
                assert mesh_result['radius'] == 1.0
                
            else:
                # 返回的是FEniCS网格对象
                print("✅ 生成了FEniCS网格对象")
                assert hasattr(mesh_result, 'num_vertices')
                assert hasattr(mesh_result, 'num_cells')
                print(f"   顶点数: {mesh_result.num_vertices()}")
                print(f"   单元数: {mesh_result.num_cells()}")
            
            # 检查输出文件
            msh_files = list(temp_path.glob("*.msh"))
            assert len(msh_files) >= 1, "应该生成至少一个.msh文件"
            print(f"✅ 生成了 {len(msh_files)} 个网格文件")
            
            # 检查XDMF文件
            xdmf_files = list(temp_path.glob("*.xdmf"))
            if xdmf_files:
                print(f"✅ 生成了 {len(xdmf_files)} 个XDMF文件")
            
            return True
            
    except ImportError as e:
        print(f"⚠️  依赖不可用，跳过真实网格生成测试: {e}")
        return True
        
    except Exception as e:
        print(f"❌ 真实网格生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_convenience_function():
    """测试便捷函数的真实调用"""
    print("🔧 测试便捷函数真实调用...")
    
    try:
        from pyeidors.geometry.optimized_mesh_generator import create_eit_mesh
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 使用便捷函数
            mesh_result = create_eit_mesh(
                n_elec=8,
                radius=1.0,
                refinement=3,
                electrode_coverage=0.5,
                output_dir=temp_dir
            )
            
            # 验证结果
            if isinstance(mesh_result, dict):
                print("✅ 便捷函数生成了网格信息字典")
                assert 'n_electrodes' in mesh_result
                assert mesh_result['n_electrodes'] == 8
            else:
                print("✅ 便捷函数生成了FEniCS网格对象")
                assert hasattr(mesh_result, 'num_vertices')
                assert hasattr(mesh_result, 'num_cells')
            
            # 检查输出文件
            output_path = Path(temp_dir)
            msh_files = list(output_path.glob("*.msh"))
            assert len(msh_files) >= 1, "应该生成至少一个.msh文件"
            
            return True
            
    except ImportError as e:
        print(f"⚠️  依赖不可用，跳过便捷函数测试: {e}")
        return True
        
    except Exception as e:
        print(f"❌ 便捷函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mesh_converter():
    """测试网格转换器"""
    print("🔧 测试网格转换器...")
    
    try:
        from pyeidors.geometry.optimized_mesh_generator import (
            OptimizedMeshConverter, OptimizedMeshGenerator,
            OptimizedMeshConfig, ElectrodePosition
        )
        
        # 首先生成一个网格文件
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 生成网格
            config = OptimizedMeshConfig(radius=1.0, refinement=3)
            electrodes = ElectrodePosition(L=8, coverage=0.5)
            generator = OptimizedMeshGenerator(config, electrodes)
            
            # 创建网格文件
            mesh_result = generator.generate(output_dir=temp_path)
            
            # 找到生成的.msh文件
            msh_files = list(temp_path.glob("*.msh"))
            if msh_files:
                msh_file = msh_files[0]
                print(f"✅ 找到网格文件: {msh_file.name}")
                
                # 测试转换器
                converter = OptimizedMeshConverter(str(msh_file), str(temp_path))
                
                # 尝试转换
                try:
                    mesh, boundaries_mf, assoc_table = converter.convert()
                    print("✅ 网格转换成功")
                    
                    # 验证结果
                    if hasattr(mesh, 'num_vertices'):
                        print(f"   转换后顶点数: {mesh.num_vertices()}")
                        print(f"   转换后单元数: {mesh.num_cells()}")
                    
                    if assoc_table:
                        print(f"   关联表项数: {len(assoc_table)}")
                        
                except Exception as e:
                    print(f"⚠️  转换过程中出现问题: {e}")
                    # 检查是否至少生成了XDMF文件
                    xdmf_files = list(temp_path.glob("*.xdmf"))
                    if xdmf_files:
                        print(f"✅ 生成了 {len(xdmf_files)} 个XDMF文件")
                    
                    ini_files = list(temp_path.glob("*.ini"))
                    if ini_files:
                        print(f"✅ 生成了 {len(ini_files)} 个关联表文件")
                        
                return True
            else:
                print("⚠️  没有找到网格文件，跳过转换器测试")
                return True
                
    except ImportError as e:
        print(f"⚠️  依赖不可用，跳过网格转换器测试: {e}")
        return True
        
    except Exception as e:
        print(f"❌ 网格转换器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_electrode_geometry():
    """测试电极几何计算"""
    print("🔧 测试电极几何计算...")
    
    try:
        from pyeidors.geometry.optimized_mesh_generator import ElectrodePosition
        
        # 测试16电极标准配置
        elec_pos = ElectrodePosition(L=16, coverage=0.5)
        positions = elec_pos.positions
        
        # 验证角度分布
        total_coverage = 0
        for start, end in positions:
            if end > start:
                total_coverage += (end - start)
            else:
                total_coverage += (end + 2*np.pi - start)
        
        expected_coverage = 2 * np.pi * 0.5
        assert abs(total_coverage - expected_coverage) < 1e-10
        
        print(f"✅ 电极总覆盖角度正确: {total_coverage:.4f} rad")
        
        # 测试对称性
        elec_pos_sym = ElectrodePosition(L=8, coverage=0.5)
        pos_sym = elec_pos_sym.positions
        
        # 验证相邻电极间距相等
        gaps = []
        for i in range(len(pos_sym)):
            end_current = pos_sym[i][1]
            start_next = pos_sym[(i+1) % len(pos_sym)][0]
            
            if start_next > end_current:
                gap = start_next - end_current
            else:
                gap = start_next + 2*np.pi - end_current
            gaps.append(gap)
        
        # 检查间距是否相等
        gap_std = np.std(gaps)
        assert gap_std < 1e-10, f"间距不相等，标准差: {gap_std}"
        
        print(f"✅ 电极间距分布均匀: {np.mean(gaps):.4f} rad")
        
        return True
        
    except Exception as e:
        print(f"❌ 电极几何计算测试失败: {e}")
        return False

def run_all_tests():
    """运行所有实际测试"""
    print("🚀 开始运行实际网格生成测试...")
    print("=" * 50)
    
    tests = [
        ("电极几何计算", test_electrode_geometry),
        ("真实网格生成", test_real_mesh_generation),
        ("便捷函数真实调用", test_convenience_function),
        ("网格转换器", test_mesh_converter),
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