#!/usr/bin/env python3
"""
Test electrode default position on positive Y-axis.
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
    """Test electrode default starting position on positive Y-axis."""
    print("🔧 Testing electrode Y-axis starting position...")

    try:
        from pyeidors.geometry.optimized_mesh_generator import ElectrodePosition

        elec_pos = ElectrodePosition(L=16, coverage=0.5, rotation=0.0)
        positions = elec_pos.positions
        first_electrode_start, first_electrode_end = positions[0]
        first_electrode_center = (first_electrode_start + first_electrode_end) / 2
        expected_center = pi / 2

        print(f"   First electrode center angle: {first_electrode_center:.6f} rad ({first_electrode_center*180/pi:.3f}°)")
        print(f"   Expected angle: {expected_center:.6f} rad ({expected_center*180/pi:.3f}°)")

        angle_diff = abs(first_electrode_center - expected_center)
        assert angle_diff < 1e-10, f"First electrode center should be exactly on positive Y-axis: diff {angle_diff}"

        x_center = cos(first_electrode_center)
        y_center = sin(first_electrode_center)
        print(f"   First electrode center coordinates: ({x_center:.4f}, {y_center:.4f})")
        assert abs(x_center) < 1e-10, f"x coordinate should be exactly 0: {x_center}"
        assert abs(y_center - 1.0) < 1e-10, f"y coordinate should be exactly 1: {y_center}"

        print("✅ Electrode Y-axis starting position test passed")
        return True

    except Exception as e:
        print(f"❌ Electrode Y-axis starting position test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_electrode_sequence():
    """Test electrodes arranged in counter-clockwise order."""
    print("🔧 Testing electrode counter-clockwise order...")

    try:
        from pyeidors.geometry.optimized_mesh_generator import ElectrodePosition

        elec_pos = ElectrodePosition(L=8, coverage=0.5, rotation=0.0)
        positions = elec_pos.positions
        centers = []
        for start, end in positions:
            centers.append((start + end) / 2)

        print("   Electrode center angles:")
        for i, center in enumerate(centers):
            degree = center * 180 / pi
            x, y = cos(center), sin(center)
            print(f"     Electrode {i+1}: {center:.4f} rad ({degree:.1f}°) -> ({x:.3f}, {y:.3f})")

        for i in range(1, len(centers)):
            if centers[i] < centers[i - 1]:
                centers[i] += 2 * pi
            assert centers[i] > centers[i - 1], f"Electrode {i+1} angle less than electrode {i}: {centers[i]} < {centers[i-1]}"

        first_center = centers[0]
        expected_first = pi / 2
        assert abs(first_center - expected_first) < 0.2, f"First electrode not at top: {first_center}"

        print("✅ Electrode counter-clockwise order test passed")
        return True

    except Exception as e:
        print(f"❌ Electrode counter-clockwise order test failed: {e}")
        return False


def test_rotation_effect():
    """Test rotation parameter effect."""
    print("🔧 Testing rotation parameter effect...")

    try:
        from pyeidors.geometry.optimized_mesh_generator import ElectrodePosition

        elec_pos_no_rot = ElectrodePosition(L=8, coverage=0.5, rotation=0.0)
        elec_pos_rot = ElectrodePosition(L=8, coverage=0.5, rotation=pi / 4)

        center_no_rot = (elec_pos_no_rot.positions[0][0] + elec_pos_no_rot.positions[0][1]) / 2
        center_rot = (elec_pos_rot.positions[0][0] + elec_pos_rot.positions[0][1]) / 2
        expected_diff = pi / 4
        actual_diff = center_rot - center_no_rot

        print(f"   No rotation first electrode center: {center_no_rot:.4f} rad ({center_no_rot*180/pi:.1f}°)")
        print(f"   Rotated first electrode center: {center_rot:.4f} rad ({center_rot*180/pi:.1f}°)")
        print(f"   Angle difference: {actual_diff:.4f} rad ({actual_diff*180/pi:.1f}°)")

        assert abs(actual_diff - expected_diff) < 0.01, f"Rotation effect incorrect: {actual_diff} vs {expected_diff}"

        print("✅ Rotation parameter effect test passed")
        return True

    except Exception as e:
        print(f"❌ Rotation parameter effect test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("🚀 Starting electrode Y-axis starting position tests...")
    print("=" * 50)

    tests = [
        ("Electrode Y-axis Starting Position", test_electrode_y_axis_start),
        ("Electrode Counter-clockwise Order", test_electrode_sequence),
        ("Rotation Parameter Effect", test_rotation_effect),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Running test: {test_name}")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test exception: {test_name} - {e}")

    print("\n" + "=" * 50)
    print(f"📊 Tests complete: {passed}/{total} passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {total - passed} test(s) failed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
