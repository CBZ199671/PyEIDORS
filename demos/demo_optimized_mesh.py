#!/usr/bin/env python3
"""
优化网格生成器演示
展示基于参考实现的新网格生成器功能
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import sys

DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent
SRC_PATH = PROJECT_ROOT / 'src'

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from pyeidors.utils.chinese_font_config import configure_chinese_font
    configure_chinese_font()
except ImportError:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False


def demo_electrode_positions():
    """演示电极位置配置"""
    print("🔬 演示电极位置配置")
    print("=" * 40)
    
    from pyeidors.geometry.optimized_mesh_generator import ElectrodePosition
    
    configs = [
        ("标准16电极", ElectrodePosition(L=16, coverage=0.5)),
        ("紧凑16电极", ElectrodePosition(L=16, coverage=0.3)),
        ("宽电极", ElectrodePosition(L=16, coverage=0.8)),
        ("8电极", ElectrodePosition(L=8, coverage=0.5)),
        ("32电极", ElectrodePosition(L=32, coverage=0.5)),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (name, config) in enumerate(configs):
        if i >= len(axes):
            break
        ax = axes[i]
        positions = config.positions
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.3)
        
        for j, (start, end) in enumerate(positions):
            theta_elec = np.linspace(start, end, 20)
            x_elec = np.cos(theta_elec)
            y_elec = np.sin(theta_elec)
            ax.plot(x_elec, y_elec, 'b-', linewidth=3, label='电极' if j == 0 else '')
            mid_angle = (start + end) / 2
            ax.text(1.1*np.cos(mid_angle), 1.1*np.sin(mid_angle), str(j + 1), ha='center', va='center', fontsize=8)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(f'{name}\n{config.L}电极, 覆盖率{config.coverage}')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend()
    
    if len(configs) < len(axes):
        axes[-1].set_visible(False)
    
    plt.tight_layout()
    output_path = DEMO_DIR / 'electrode_positions_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ 电极位置配置演示完成，保存为 electrode_positions_demo.png")


def demo_mesh_generation():
    """演示网格生成"""
    print("\n🔬 演示网格生成")
    print("=" * 40)
    
    from pyeidors.geometry.optimized_mesh_generator import (
        OptimizedMeshGenerator, OptimizedMeshConfig, ElectrodePosition
    )
    
    configs = [
        ("粗糙网格", OptimizedMeshConfig(radius=1.0, refinement=2)),
        ("中等网格", OptimizedMeshConfig(radius=1.0, refinement=4)),
        ("精细网格", OptimizedMeshConfig(radius=1.0, refinement=6)),
    ]
    
    electrodes = ElectrodePosition(L=16, coverage=0.5)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (name, config) in enumerate(configs):
            print(f"   生成{name}...")
            generator = OptimizedMeshGenerator(config, electrodes)
            mesh = generator.generate(output_dir=temp_path)
            ax = axes[i]
            
            if hasattr(mesh, 'coordinates'):
                coords = mesh.coordinates()
                cells = mesh.cells()
                for cell in cells:
                    triangle = coords[cell]
                    triangle = np.vstack([triangle, triangle[0]])
                    ax.plot(triangle[:, 0], triangle[:, 1], 'b-', alpha=0.3, linewidth=0.5)
                ax.plot(coords[:, 0], coords[:, 1], 'ro', markersize=1, alpha=0.6)
                n_vertices = mesh.num_vertices()
                n_cells = mesh.num_cells()
            else:
                n_vertices = "N/A"
                n_cells = "N/A"
                ax.text(0, 0, f"网格生成成功\n但无法可视化\n({type(mesh).__name__})", ha='center', va='center', fontsize=12)
            
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
            
            positions = electrodes.positions
            for start, end in positions:
                theta_elec = np.linspace(start, end, 20)
                x_elec = np.cos(theta_elec)
                y_elec = np.sin(theta_elec)
                ax.plot(x_elec, y_elec, 'r-', linewidth=3)
            
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect('equal')
            ax.set_title(f'{name}\n顶点: {n_vertices}, 单元: {n_cells}')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = DEMO_DIR / 'mesh_generation_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ 网格生成演示完成，保存为 mesh_generation_demo.png")


def demo_convenience_functions():
    """演示便捷函数"""
    print("\n🔬 演示便捷函数")
    print("=" * 40)
    
    from pyeidors.geometry.optimized_mesh_generator import create_eit_mesh
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print("   使用便捷函数创建标准EIT网格...")
        mesh = create_eit_mesh(
            n_elec=16,
            radius=1.0,
            refinement=5,
            electrode_coverage=0.5,
            output_dir=temp_dir
        )
        
        if hasattr(mesh, 'num_vertices'):
            print(f"   ✅ 成功创建网格: {mesh.num_vertices()}个顶点, {mesh.num_cells()}个单元")
        else:
            print(f"   ✅ 成功创建网格信息: {type(mesh).__name__}")
        
        output_path = Path(temp_dir)
        msh_files = list(output_path.glob('*.msh'))
        xdmf_files = list(output_path.glob('*.xdmf'))
        ini_files = list(output_path.glob('*.ini'))
        print(f"   📁 生成文件: {len(msh_files)} .msh, {len(xdmf_files)} .xdmf, {len(ini_files)} .ini")


def demo_mesh_quality():
    """演示网格质量分析"""
    print("\n🔬 演示网格质量分析")
    print("=" * 40)
    
    from pyeidors.geometry.optimized_mesh_generator import create_eit_mesh
    
    with tempfile.TemporaryDirectory() as temp_dir:
        mesh_configs = [
            ("基础网格", {"refinement": 3}),
            ("标准网格", {"refinement": 5}),
            ("高质量网格", {"refinement": 7}),
        ]
        
        results = []
        
        for name, config in mesh_configs:
            print(f"   生成{name}...")
            mesh = create_eit_mesh(
                n_elec=16,
                radius=1.0,
                electrode_coverage=0.5,
                output_dir=temp_dir,
                **config
            )
            
            if hasattr(mesh, 'num_vertices'):
                n_vertices = mesh.num_vertices()
                n_cells = mesh.num_cells()
                area = np.pi
                density = n_cells / area
                results.append({
                    'name': name,
                    'vertices': n_vertices,
                    'cells': n_cells,
                    'density': density,
                    'refinement': config['refinement']
                })
                print(f"     顶点: {n_vertices}, 单元: {n_cells}, 密度: {density:.1f} cells/unit²")
        
        if results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            names = [r['name'] for r in results]
            vertices = [r['vertices'] for r in results]
            cells = [r['cells'] for r in results]
            x = np.arange(len(names))
            width = 0.35
            
            ax1.bar(x - width/2, vertices, width, label='顶点数', alpha=0.7)
            ax1.bar(x + width/2, cells, width, label='单元数', alpha=0.7)
            ax1.set_xlabel('网格配置')
            ax1.set_ylabel('数量')
            ax1.set_title('网格规模对比')
            ax1.set_xticks(x)
            ax1.set_xticklabels(names)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            refinements = [r['refinement'] for r in results]
            ax2.plot(refinements, cells, 'bo-', markersize=8)
            ax2.set_xlabel('细化级别')
            ax2.set_ylabel('单元数')
            ax2.set_title('细化级别与网格密度的关系')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path = DEMO_DIR / 'mesh_quality_demo.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print("   ✅ 网格质量分析完成，保存为 mesh_quality_demo.png")


def run_demo():
    """运行完整演示"""
    print("🎬 优化网格生成器演示")
    print("=" * 60)
    
    try:
        demo_electrode_positions()
        demo_mesh_generation()
        demo_convenience_functions()
        demo_mesh_quality()
        
        print("\n🎉 所有演示完成！")
        print("📊 生成的文件:")
        print("   - electrode_positions_demo.png: 电极位置配置对比")
        print("   - mesh_generation_demo.png: 不同精度网格对比")
        print("   - mesh_quality_demo.png: 网格质量分析")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_demo()
