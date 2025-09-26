#!/usr/bin/env python3
"""PyEidors模块化测试"""

import numpy as np

def test_imports():
    """测试所有模块导入"""
    print("🧪 测试模块导入...")
    
    try:
        from pyeidors.data.structures import MeshConfig, ElectrodePosition, PatternConfig
        print("✅ 数据结构模块导入成功")
    except Exception as e:
        print(f"❌ 数据结构模块导入失败: {e}")
        return False
    
    try:
        from pyeidors.electrodes.patterns import StimMeasPatternManager
        print("✅ 电极模式模块导入成功")
    except Exception as e:
        print(f"❌ 电极模式模块导入失败: {e}")
        return False
    
    try:
        from pyeidors.geometry.mesh_generator import MeshGenerator
        print("✅ 网格生成模块导入成功")
    except Exception as e:
        print(f"❌ 网格生成模块导入失败: {e}")
        return False
    
    try:
        from pyeidors.forward.eit_forward_model import EITForwardModel
        print("✅ 前向模型模块导入成功")
    except Exception as e:
        print(f"❌ 前向模型模块导入失败: {e}")
        return False
    
    try:
        from pyeidors.inverse.solvers.gauss_newton import StandardGaussNewtonReconstructor
        print("✅ 逆问题求解器模块导入成功")
    except Exception as e:
        print(f"❌ 逆问题求解器模块导入失败: {e}")
        return False
    
    try:
        from pyeidors.data.synthetic_data import create_synthetic_data
        print("✅ 合成数据模块导入成功")
    except Exception as e:
        print(f"❌ 合成数据模块导入失败: {e}")
        return False
        
    return True


def test_basic_workflow():
    """测试基本工作流程"""
    print("\n🔧 测试基本工作流程...")
    
    try:
        # 导入所需模块
        from pyeidors.data.structures import MeshConfig, ElectrodePosition, PatternConfig
        from pyeidors.geometry.mesh_generator import MeshGenerator
        from pyeidors.electrodes.patterns import StimMeasPatternManager
        from pyeidors.forward.eit_forward_model import EITForwardModel
        from pyeidors.inverse.solvers.gauss_newton import StandardGaussNewtonReconstructor
        from pyeidors.data.synthetic_data import create_synthetic_data
        
        # 1. 创建配置
        n_elec = 16
        mesh_config = MeshConfig(radius=1.0, refinement=6, electrode_vertices=4)
        electrode_config = ElectrodePosition(L=n_elec, coverage=0.5)
        pattern_config = PatternConfig(
            n_elec=n_elec,
            stim_pattern='{ad}',
            meas_pattern='{ad}',
            amplitude=1.0,
            use_meas_current=False
        )
        
        print("✅ 配置创建成功")
        
        # 2. 生成网格
        generator = MeshGenerator(mesh_config, electrode_config)
        mesh = generator.generate()
        
        print(f"✅ 网格生成成功: {mesh.num_cells()}个单元")
        
        # 3. 创建激励测量模式管理器
        pattern_manager = StimMeasPatternManager(pattern_config)
        
        print(f"✅ 模式管理器创建成功: {pattern_manager.n_stim}个激励, {pattern_manager.n_meas_total}次测量")
        
        # 4. 创建前向模型
        z = np.full(n_elec, 1e-6)  # 接触阻抗
        fwd_model = EITForwardModel(n_elec, pattern_config, z, mesh)
        
        print("✅ 前向模型创建成功")
        
        # 5. 生成合成数据
        synthetic_data = create_synthetic_data(
            fwd_model=fwd_model,
            inclusion_conductivity=2.5,
            background_conductivity=1.0,
            noise_level=0.02,
            center=(-0.3, 0.1),
            radius=0.3
        )
        
        print(f"✅ 合成数据生成成功: SNR = {synthetic_data['snr_db']:.1f} dB")
        
        # 6. 创建重建器（但不执行重建以节省时间）
        reconstructor = StandardGaussNewtonReconstructor(
            fwd_model=fwd_model,
            max_iterations=5,  # 减少迭代次数以节省时间
            convergence_tol=1e-3,
            regularization_param=0.01,
            verbose=False
        )
        
        print("✅ 重建器创建成功")
        
        print("✅ 基本工作流程测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 基本工作流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("🎯 PyEidors模块化测试")
    print("=" * 50)
    
    # 测试导入
    if not test_imports():
        print("\n❌ 导入测试失败，请检查模块结构")
        return
    
    # 测试基本工作流程
    if not test_basic_workflow():
        print("\n❌ 工作流程测试失败")
        return
    
    print("\n🎉 所有测试通过! PyEidors模块化重构成功!")


if __name__ == "__main__":
    main()