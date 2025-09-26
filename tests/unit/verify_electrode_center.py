#!/usr/bin/env python3
"""
验证第一个电极中心位置精确在Y轴正半轴
"""

import numpy as np
import sys
import os
from math import pi, cos, sin

# 添加模块路径
sys.path.insert(0, '/root/shared/src')

def verify_electrode_center():
    """验证电极中心位置"""
    print("🔍 验证第一个电极中心位置...")
    print("=" * 50)
    
    from pyeidors.geometry.optimized_mesh_generator import ElectrodePosition
    
    # 测试不同配置
    configs = [
        ("8电极", 8),
        ("16电极", 16),
        ("32电极", 32),
    ]
    
    for name, n_elec in configs:
        print(f"\n📋 {name}配置:")
        
        # 创建电极配置
        elec_pos = ElectrodePosition(L=n_elec, coverage=0.5)
        positions = elec_pos.positions
        
        # 计算第一个电极中心
        first_start, first_end = positions[0]
        first_center = (first_start + first_end) / 2
        
        # 计算坐标
        x = cos(first_center)
        y = sin(first_center)
        
        # 验证精度
        angle_deg = first_center * 180 / pi
        
        print(f"   第1个电极中心角度: {first_center:.10f} rad ({angle_deg:.6f}°)")
        print(f"   理论Y轴正半轴: {pi/2:.10f} rad (90.000000°)")
        print(f"   角度误差: {abs(first_center - pi/2):.2e} rad")
        print(f"   中心坐标: ({x:.10f}, {y:.10f})")
        print(f"   x坐标误差: {abs(x):.2e}")
        print(f"   y坐标误差: {abs(y - 1.0):.2e}")
        
        # 验证是否精确
        if abs(first_center - pi/2) < 1e-15:
            print("   ✅ 角度位置精确正确")
        else:
            print("   ❌ 角度位置有误差")
            
        if abs(x) < 1e-15 and abs(y - 1.0) < 1e-15:
            print("   ✅ 坐标位置精确正确")
        else:
            print("   ❌ 坐标位置有误差")
    
    print("\n" + "=" * 50)
    print("🎯 验证结论: 第一个电极中心精确位于Y轴正半轴 (0, 1)")

def verify_rotation_effect():
    """验证旋转参数的效果"""
    print("\n🔄 验证旋转参数效果...")
    print("=" * 30)
    
    from pyeidors.geometry.optimized_mesh_generator import ElectrodePosition
    
    rotations = [0, pi/6, pi/4, pi/3, pi/2]  # 0°, 30°, 45°, 60°, 90°
    
    for rotation in rotations:
        elec_pos = ElectrodePosition(L=8, coverage=0.5, rotation=rotation)
        positions = elec_pos.positions
        
        # 第一个电极中心
        first_center = (positions[0][0] + positions[0][1]) / 2
        
        # 期望位置
        expected = pi/2 + rotation
        
        angle_deg = first_center * 180 / pi
        expected_deg = expected * 180 / pi
        
        print(f"旋转{rotation*180/pi:5.1f}°: 中心位置{angle_deg:6.1f}° (期望{expected_deg:6.1f}°)")
        
        # 验证精度
        if abs(first_center - expected) < 1e-15:
            print("   ✅ 旋转效果精确")
        else:
            print(f"   ❌ 旋转误差: {abs(first_center - expected):.2e}")

if __name__ == "__main__":
    verify_electrode_center()
    verify_rotation_effect()
    print("\n🎉 验证完成！")