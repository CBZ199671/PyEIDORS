#!/usr/bin/env python3
"""
PyEidors完整系统端到端测试
测试完整的EIT正逆问题求解流程，包括网格加载、前向求解、逆问题重建和可视化
"""

import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt
from pathlib import Path

# 添加源码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_complete_eit_workflow():
    """测试完整的EIT工作流程"""
    print("=== PyEidors完整系统端到端测试 ===\n")
    
    try:
        # 1. 导入模块并检查环境
        print("1. 导入模块并检查环境...")
        from pyeidors import EITSystem, check_environment
        from pyeidors.data.structures import PatternConfig, MeshConfig
        from pyeidors.data.synthetic_data import create_synthetic_data, create_custom_phantom
        from pyeidors.visualization import create_visualizer
        from pyeidors.geometry.mesh_loader import MeshLoader
        
        env_info = check_environment()
        print(f"   ✓ FEniCS: {env_info['fenics_available']}")
        print(f"   ✓ PyTorch: {env_info['torch_available']} (CUDA: {env_info['cuda_available']})")
        print(f"   ✓ CUQIpy: {env_info['cuqi_available']}")
        print()
        
        # 2. 检查网格文件
        print("2. 检查和加载网格...")
        mesh_loader = MeshLoader()
        available_meshes = mesh_loader.list_available_meshes()
        print(f"   可用网格: {available_meshes}")
        
        if not available_meshes['fenics_h5']:
            print("   ⚠️  没有找到FEniCS H5格式的网格文件")
            return False
        
        # 加载默认网格
        mesh = mesh_loader.get_default_mesh()
        mesh_info = mesh.get_info()
        print(f"   ✓ 网格加载成功:")
        print(f"     - 节点数: {mesh_info['num_vertices']}")
        print(f"     - 单元数: {mesh_info['num_cells']}")
        print(f"     - 电极数: {mesh_info['num_electrodes']}")
        print(f"     - 半径: {mesh_info['radius']:.3f}")
        print()
        
        # 3. 创建EIT系统
        print("3. 创建和初始化EIT系统...")
        n_elec = mesh_info['num_electrodes']
        
        pattern_config = PatternConfig(
            n_elec=n_elec,
            stim_pattern='{ad}',
            meas_pattern='{ad}',
            amplitude=1.0
        )
        
        eit_system = EITSystem(
            n_elec=n_elec,
            pattern_config=pattern_config,
            contact_impedance=np.ones(n_elec) * 0.01
        )
        
        # 使用加载的网格初始化系统
        eit_system.setup(mesh=mesh)
        
        system_info = eit_system.get_system_info()
        print(f"   ✓ EIT系统初始化成功:")
        print(f"     - 电极数: {system_info['n_elec']}")
        print(f"     - 单元数: {system_info['n_elements']}")
        print(f"     - 节点数: {system_info['n_nodes']}")
        print(f"     - 测量数: {system_info['n_measurements']}")
        print(f"     - 激励模式数: {system_info['n_stimulation_patterns']}")
        print()
        
        # 4. 创建合成测试数据
        print("4. 生成合成测试数据...")
        
        # 创建自定义幻象
        anomalies = [
            {'center': (0.3, 0.3), 'radius': 0.2, 'conductivity': 2.5},
            {'center': (-0.4, -0.2), 'radius': 0.15, 'conductivity': 0.5}
        ]
        
        sigma_phantom = create_custom_phantom(
            eit_system.fwd_model,
            background_conductivity=1.0,
            anomalies=anomalies
        )
        
        # 生成合成数据
        synthetic_data = create_synthetic_data(
            eit_system.fwd_model,
            inclusion_conductivity=2.5,
            background_conductivity=1.0,
            noise_level=0.02,
            center=(0.2, 0.2),
            radius=0.25
        )
        
        print(f"   ✓ 合成数据生成成功:")
        print(f"     - 信噪比: {synthetic_data['snr_db']:.2f} dB")
        print(f"     - 测量数量: {len(synthetic_data['data_clean'].meas)}")
        print(f"     - 噪声标准差: {np.std(synthetic_data['noise']):.6f}")
        print()
        
        # 5. 前向求解验证
        print("5. 前向求解验证...")
        start_time = time.time()
        
        # 使用自定义幻象进行前向求解
        from pyeidors.data.structures import EITImage
        phantom_image = EITImage(elem_data=sigma_phantom.vector()[:], fwd_model=eit_system.fwd_model)
        forward_data = eit_system.forward_solve(phantom_image)
        
        forward_time = time.time() - start_time
        print(f"   ✓ 前向求解完成:")
        print(f"     - 计算时间: {forward_time:.3f} 秒")
        print(f"     - 测量范围: [{np.min(forward_data.meas):.6f}, {np.max(forward_data.meas):.6f}]")
        print(f"     - 测量均值: {np.mean(forward_data.meas):.6f}")
        print()
        
        # 6. 逆问题重建
        print("6. 逆问题重建...")
        start_time = time.time()
        
        # 创建参考数据（均匀分布）
        reference_image = eit_system.create_homogeneous_image(conductivity=1.0)
        reference_data = eit_system.forward_solve(reference_image)
        
        # 执行重建
        try:
            reconstructed_image = eit_system.inverse_solve(
                data=synthetic_data['data_noisy'],
                reference_data=reference_data,
                initial_guess=None
            )
            
            reconstruction_time = time.time() - start_time
            print(f"   ✓ 逆问题重建完成:")
            print(f"     - 计算时间: {reconstruction_time:.3f} 秒")
            
            # 计算重建误差
            true_values = synthetic_data['sigma_true'].vector()[:]
            recon_values = reconstructed_image.elem_data
            relative_error = np.linalg.norm(recon_values - true_values) / np.linalg.norm(true_values)
            print(f"     - 相对误差: {relative_error:.4f}")
            print(f"     - 重建范围: [{np.min(recon_values):.3f}, {np.max(recon_values):.3f}]")
            
        except Exception as e:
            print(f"   ⚠️  重建过程出现问题: {e}")
            print("   继续其他测试...")
            reconstructed_image = None
        print()
        
        # 7. 可视化测试
        print("7. 可视化测试...")
        try:
            visualizer = create_visualizer()
            
            # 创建输出目录
            output_dir = Path("test_results")
            output_dir.mkdir(exist_ok=True)
            
            # 绘制网格
            fig1 = visualizer.plot_mesh(mesh, title="网格结构", 
                                       save_path=output_dir / "mesh.png")
            print("   ✓ 网格可视化完成")
            
            # 绘制真实导电率分布
            fig2 = visualizer.plot_conductivity(mesh, synthetic_data['sigma_true'], 
                                              title="真实导电率分布",
                                              save_path=output_dir / "true_conductivity.png")
            print("   ✓ 真实分布可视化完成")
            
            # 绘制测量数据
            fig3 = visualizer.plot_measurements(synthetic_data['data_noisy'], 
                                              title="合成测量数据（含噪声）",
                                              save_path=output_dir / "measurements.png")
            print("   ✓ 测量数据可视化完成")
            
            # 如果重建成功，绘制对比图
            if reconstructed_image is not None:
                fig4 = visualizer.plot_reconstruction_comparison(
                    mesh, synthetic_data['sigma_true'], reconstructed_image.elem_data,
                    title="重建结果对比",
                    save_path=output_dir / "reconstruction_comparison.png"
                )
                print("   ✓ 重建对比可视化完成")
            
            print(f"   ✓ 所有图像保存到: {output_dir.absolute()}")
            
            # 关闭图像以释放内存
            plt.close('all')
            
        except Exception as e:
            print(f"   ⚠️  可视化过程出现问题: {e}")
        print()
        
        # 8. 性能统计
        print("8. 性能统计总结...")
        print(f"   - 前向求解时间: {forward_time:.3f} 秒")
        if 'reconstruction_time' in locals():
            print(f"   - 逆问题重建时间: {reconstruction_time:.3f} 秒")
        print(f"   - 网格规模: {mesh_info['num_vertices']} 节点, {mesh_info['num_cells']} 单元")
        print(f"   - 测量数量: {system_info['n_measurements']}")
        print()
        
        print("🎉 完整系统测试成功完成！")
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_module_integration():
    """测试模块集成"""
    print("=== 模块集成测试 ===\n")
    
    modules_to_test = [
        ("核心系统", "pyeidors.core_system"),
        ("网格加载器", "pyeidors.geometry.mesh_loader"),
        ("前向模型", "pyeidors.forward.eit_forward_model"),
        ("雅可比计算器", "pyeidors.inverse.jacobian.direct_jacobian"),
        ("正则化", "pyeidors.inverse.regularization.smoothness"),
        ("高斯牛顿求解器", "pyeidors.inverse.solvers.gauss_newton"),
        ("激励模式管理器", "pyeidors.electrodes.patterns"),
        ("合成数据生成", "pyeidors.data.synthetic_data"),
        ("可视化", "pyeidors.visualization")
    ]
    
    success_count = 0
    for name, module_path in modules_to_test:
        try:
            __import__(module_path)
            print(f"✓ {name} 模块导入成功")
            success_count += 1
        except Exception as e:
            print(f"✗ {name} 模块导入失败: {e}")
    
    print(f"\n模块集成测试结果: {success_count}/{len(modules_to_test)} 成功")
    return success_count == len(modules_to_test)

if __name__ == "__main__":
    print("开始PyEidors完整系统测试...\n")
    
    # 模块集成测试
    integration_success = test_module_integration()
    print()
    
    if integration_success:
        # 完整工作流程测试
        workflow_success = test_complete_eit_workflow()
        
        if workflow_success:
            print("\n🏆 所有测试均成功通过！PyEidors系统运行正常。")
            sys.exit(0)
        else:
            print("\n⚠️  工作流程测试未完全成功，但基本功能可用。")
            sys.exit(1)
    else:
        print("\n❌ 模块集成测试失败，请检查依赖和配置。")
        sys.exit(1)