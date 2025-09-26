#!/usr/bin/env python3
"""
演示修改后的电极位置（y轴正半轴起始）
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin
import sys
import os

# 添加模块路径
sys.path.insert(0, '/root/shared/src')

# 设置中文字体支持
try:
    from pyeidors.utils.chinese_font_config import configure_chinese_font
    configure_chinese_font()
except ImportError:
    # 备选方案
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

def demo_y_axis_electrodes():
    """演示y轴正半轴起始的电极位置"""
    print("🎨 生成y轴正半轴起始电极位置演示...")
    
    from pyeidors.geometry.optimized_mesh_generator import ElectrodePosition
    
    # 创建不同配置的电极
    configs = [
        ("8电极系统", ElectrodePosition(L=8, coverage=0.5)),
        ("16电极系统", ElectrodePosition(L=16, coverage=0.5)),
        ("16电极紧凑", ElectrodePosition(L=16, coverage=0.3)),
        ("旋转30°", ElectrodePosition(L=8, coverage=0.5, rotation=pi/6)),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, (name, config) in enumerate(configs):
        ax = axes[i]
        
        # 获取电极位置
        positions = config.positions
        
        # 绘制圆周
        theta = np.linspace(0, 2*pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.3, linewidth=1)
        
        # 绘制坐标轴
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # 标记y轴正半轴
        ax.arrow(0, 0, 0, 1.2, head_width=0.05, head_length=0.05, 
                fc='red', ec='red', alpha=0.7)
        ax.text(0.1, 1.1, 'Y+', fontsize=12, color='red', weight='bold')
        
        # 绘制电极
        for j, (start, end) in enumerate(positions):
            # 电极弧线
            theta_elec = np.linspace(start, end, 20)
            x_elec = np.cos(theta_elec)
            y_elec = np.sin(theta_elec)
            ax.plot(x_elec, y_elec, 'b-', linewidth=4, alpha=0.8)
            
            # 电极编号
            mid_angle = (start + end) / 2
            label_radius = 1.15
            x_label = label_radius * np.cos(mid_angle)
            y_label = label_radius * np.sin(mid_angle)
            
            # 特殊标记第一个电极
            if j == 0:
                ax.plot(x_label, y_label, 'ro', markersize=8)
                ax.text(x_label, y_label-0.15, f'{j+1}', ha='center', va='center', 
                       fontsize=10, weight='bold', color='red',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            else:
                ax.text(x_label, y_label, f'{j+1}', ha='center', va='center', 
                       fontsize=9, color='blue')
        
        # 绘制电极中心连线显示顺序
        centers_x, centers_y = [], []
        for start, end in positions:
            mid_angle = (start + end) / 2
            centers_x.append(np.cos(mid_angle))
            centers_y.append(np.sin(mid_angle))
        
        # 连接相邻电极中心
        for j in range(len(centers_x)):
            next_j = (j + 1) % len(centers_x)
            ax.plot([centers_x[j], centers_x[next_j]], [centers_y[j], centers_y[next_j]], 
                   'g--', alpha=0.3, linewidth=1)
        
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect('equal')
        ax.set_title(f'{name}\n第1个电极从Y+开始（红色标记）')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/root/shared/demos/y_axis_electrode_demo.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ y轴电极位置演示完成，保存为 y_axis_electrode_demo.png")

def demo_before_after_comparison():
    """对比修改前后的电极位置"""
    print("🎨 生成修改前后对比演示...")
    
    from pyeidors.geometry.optimized_mesh_generator import ElectrodePosition
    
    # 模拟修改前的电极位置（从x轴正半轴开始）
    def old_positions(L, coverage):
        electrode_size = 2 * pi / L * coverage
        gap_size = 2 * pi / L * (1 - coverage)
        positions = []
        for i in range(L):
            start = electrode_size * i + gap_size * i  # 从0开始
            end = electrode_size * (i + 1) + gap_size * i
            positions.append((start, end))
        return positions
    
    # 创建对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    L = 8  # 使用8电极便于观察
    coverage = 0.5
    
    # 左图：修改前（从x轴正半轴开始）
    old_pos = old_positions(L, coverage)
    
    theta = np.linspace(0, 2*pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # 标记x轴正半轴
    ax1.arrow(0, 0, 1.2, 0, head_width=0.05, head_length=0.05, 
             fc='red', ec='red', alpha=0.7)
    ax1.text(1.1, 0.1, 'X+', fontsize=12, color='red', weight='bold')
    
    for j, (start, end) in enumerate(old_pos):
        theta_elec = np.linspace(start, end, 20)
        x_elec = np.cos(theta_elec)
        y_elec = np.sin(theta_elec)
        ax1.plot(x_elec, y_elec, 'b-', linewidth=4, alpha=0.8)
        
        mid_angle = (start + end) / 2
        x_label = 1.15 * np.cos(mid_angle)
        y_label = 1.15 * np.sin(mid_angle)
        
        if j == 0:
            ax1.plot(x_label, y_label, 'ro', markersize=8)
            ax1.text(x_label+0.15, y_label, f'{j+1}', ha='center', va='center', 
                    fontsize=10, weight='bold', color='red',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        else:
            ax1.text(x_label, y_label, f'{j+1}', ha='center', va='center', 
                    fontsize=9, color='blue')
    
    ax1.set_xlim(-1.4, 1.4)
    ax1.set_ylim(-1.4, 1.4)
    ax1.set_aspect('equal')
    ax1.set_title('修改前：第1个电极从X+开始\n（传统方式）')
    ax1.grid(True, alpha=0.3)
    
    # 右图：修改后（从y轴正半轴开始）
    new_config = ElectrodePosition(L=L, coverage=coverage)
    new_pos = new_config.positions
    
    ax2.plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # 标记y轴正半轴
    ax2.arrow(0, 0, 0, 1.2, head_width=0.05, head_length=0.05, 
             fc='red', ec='red', alpha=0.7)
    ax2.text(0.1, 1.1, 'Y+', fontsize=12, color='red', weight='bold')
    
    for j, (start, end) in enumerate(new_pos):
        theta_elec = np.linspace(start, end, 20)
        x_elec = np.cos(theta_elec)
        y_elec = np.sin(theta_elec)
        ax2.plot(x_elec, y_elec, 'b-', linewidth=4, alpha=0.8)
        
        mid_angle = (start + end) / 2
        x_label = 1.15 * np.cos(mid_angle)
        y_label = 1.15 * np.sin(mid_angle)
        
        if j == 0:
            ax2.plot(x_label, y_label, 'ro', markersize=8)
            ax2.text(x_label, y_label-0.15, f'{j+1}', ha='center', va='center', 
                    fontsize=10, weight='bold', color='red',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        else:
            ax2.text(x_label, y_label, f'{j+1}', ha='center', va='center', 
                    fontsize=9, color='blue')
    
    ax2.set_xlim(-1.4, 1.4)
    ax2.set_ylim(-1.4, 1.4)
    ax2.set_aspect('equal')
    ax2.set_title('修改后：第1个电极从Y+开始\n（改进方式）')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/root/shared/demos/electrode_position_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ 修改前后对比演示完成，保存为 electrode_position_comparison.png")

def run_demo():
    """运行完整演示"""
    print("🎬 电极y轴初始位置演示")
    print("=" * 50)
    
    try:
        demo_y_axis_electrodes()
        demo_before_after_comparison()
        
        print("\n🎉 所有演示完成！")
        print("📊 生成的文件:")
        print("   - y_axis_electrode_demo.png: y轴起始电极配置演示")
        print("   - electrode_position_comparison.png: 修改前后对比")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_demo()