#!/usr/bin/env python3
"""
测试电极默认位置在y轴正半轴
"""

import numpy as np
import sys
from pathlib import Path
from math import pi, cos, sin

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parents[1]
SRC_PATH = PROJECT_ROOT / 'src'

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def test_electrode_y_axis_start():
    """测试电极默认初始位置在y轴正半轴"""
    print("🔧 测试电极y轴初始位置...")
    
    try:
        from pyeidors.geometry.optimized_mesh_generator import ElectrodePosition
        
        elec_pos = ElectrodePosition(L=16, coverage=0.5, rotation=0.0)
        positions = elec_pos.positions
        first_electrode_start, first_electrode_end = positions[0]
        first_electrode_center = (first_electrode_start + first_electrode_end) / 2
        expected_center = pi / 2
        
        print(f"   第一个电极中心角度: {first_electrode_center:.6f} rad ({first_electrode_center*180/pi:.3f}°)")
        print(f"   期望角度: {expected_center:.6f} rad ({expected_center*180/pi:.3f}°)")
        
        angle_diff = abs(first_electrode_center - expected_center)
        assert angle_diff < 1e-10, f"第一个电极中心应该精确在y轴正半轴: 差值{angle_diff}"
        
        x_center = cos(first_electrode_center)
        y_center = sin(first_electrode_center)
        print(f"   第一个电极中心坐标: ({x_center:.4f}, {y_center:.4f})")
        assert abs(x_center) < 1e-10, f"x坐标应该精确为0: {x_center}"
        assert abs(y_center - 1.0) < 1e-10, f"y坐标应该精确为1: {y_center}"
        
        print("✅ 电极y轴初始位置测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 电极y轴初始位置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_electrode_sequence():
    """测试电极按逆时针顺序排列"""
    print("🔧 测试电极逆时针顺序...")
    
    try:
        from pyeidors.geometry.optimized_mesh_generator import ElectrodePosition
        
        elec_pos = ElectrodePosition(L=8, coverage=0.5, rotation=0.0)
        positions = elec_pos.positions
        centers = []
        for start, end in positions:
            centers.append((start + end) / 2)
        
        print("   电极中心角度:")
        for i, center in enumerate(centers):
            degree = center * 180 / pi
            x, y = cos(center), sin(center)
            print(f"     电极{i+1}: {center:.4f} rad ({degree:.1f}°) -> ({x:.3f}, {y:.3f})")
        
        for i in range(1, len(centers)):
            if centers[i] < centers[i - 1]:
                centers[i] += 2 * pi
            assert centers[i] > centers[i - 1], f"电极{i+1}角度小于电极{i}: {centers[i]} < {centers[i-1]}"
        
        first_center = centers[0]
        expected_first = pi / 2
        assert abs(first_center - expected_first) < 0.2, f"第一个电极不在顶部: {first_center}"
        
        print("✅ 电极逆时针顺序测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 电极逆时针顺序测试失败: {e}")
        return False


def test_rotation_effect():
    """测试旋转参数的效果"""
    print("🔧 测试旋转参数效果...")
    
    try:
        from pyeidors.geometry.optimized_mesh_generator import ElectrodePosition
        
        elec_pos_no_rot = ElectrodePosition(L=8, coverage=0.5, rotation=0.0)
        elec_pos_rot = ElectrodePosition(L=8, coverage=0.5, rotation=pi / 4)
        
        center_no_rot = (elec_pos_no_rot.positions[0][0] + elec_pos_no_rot.positions[0][1]) / 2
        center_rot = (elec_pos_rot.positions[0][0] + elec_pos_rot.positions[0][1]) / 2
        expected_diff = pi / 4
        actual_diff = center_rot - center_no_rot
        
        print(f"   无旋转第一个电极中心: {center_no_rot:.4f} rad ({center_no_rot*180/pi:.1f}°)")
        print(f"   旋转后第一个电极中心: {center_rot:.4f} rad ({center_rot*180/pi:.1f}°)")
        print(f"   角度差: {actual_diff:.4f} rad ({actual_diff*180/pi:.1f}°)")
        
        assert abs(actual_diff - expected_diff) < 0.01, f"旋转效果不正确: {actual_diff} vs {expected_diff}"
        
        print("✅ 旋转参数效果测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 旋转参数效果测试失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("🚀 开始测试电极y轴初始位置...")
    print("=" * 50)
    
    tests = [
        ("电极y轴初始位置", test_electrode_y_axis_start),
        ("电极逆时针顺序", test_electrode_sequence),
        ("旋转参数效果", test_rotation_effect),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 运行测试: {test_name}")
        try:
            if test_func():
                passed += 1
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
