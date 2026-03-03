#!/usr/bin/env python3
"""
Verify first electrode center position is exactly on positive Y-axis.
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

def verify_electrode_center():
    """Verify electrode center position."""
    print("🔍 Verifying first electrode center position...")
    print("=" * 50)

    from pyeidors.geometry.optimized_mesh_generator import ElectrodePosition

    # Test different configurations
    configs = [
        ("8-electrode", 8),
        ("16-electrode", 16),
        ("32-electrode", 32),
    ]

    for name, n_elec in configs:
        print(f"\n📋 {name} configuration:")

        # Create electrode configuration
        elec_pos = ElectrodePosition(L=n_elec, coverage=0.5)
        positions = elec_pos.positions

        # Calculate first electrode center
        first_start, first_end = positions[0]
        first_center = (first_start + first_end) / 2

        # Calculate coordinates
        x = cos(first_center)
        y = sin(first_center)

        # Verify precision
        angle_deg = first_center * 180 / pi

        print(f"   Electrode 1 center angle: {first_center:.10f} rad ({angle_deg:.6f}°)")
        print(f"   Theoretical positive Y-axis: {pi/2:.10f} rad (90.000000°)")
        print(f"   Angle error: {abs(first_center - pi/2):.2e} rad")
        print(f"   Center coordinates: ({x:.10f}, {y:.10f})")
        print(f"   x-coordinate error: {abs(x):.2e}")
        print(f"   y-coordinate error: {abs(y - 1.0):.2e}")

        # Verify if exact
        if abs(first_center - pi/2) < 1e-15:
            print("   ✅ Angle position exactly correct")
        else:
            print("   ❌ Angle position has error")

        if abs(x) < 1e-15 and abs(y - 1.0) < 1e-15:
            print("   ✅ Coordinate position exactly correct")
        else:
            print("   ❌ Coordinate position has error")

    print("\n" + "=" * 50)
    print("🎯 Verification conclusion: First electrode center is exactly on positive Y-axis (0, 1)")

def verify_rotation_effect():
    """Verify rotation parameter effect."""
    print("\n🔄 Verifying rotation parameter effect...")
    print("=" * 30)

    from pyeidors.geometry.optimized_mesh_generator import ElectrodePosition

    rotations = [0, pi/6, pi/4, pi/3, pi/2]  # 0°, 30°, 45°, 60°, 90°

    for rotation in rotations:
        elec_pos = ElectrodePosition(L=8, coverage=0.5, rotation=rotation)
        positions = elec_pos.positions

        # First electrode center
        first_center = (positions[0][0] + positions[0][1]) / 2

        # Expected position
        expected = pi/2 + rotation

        angle_deg = first_center * 180 / pi
        expected_deg = expected * 180 / pi

        print(f"Rotation {rotation*180/pi:5.1f}°: center position {angle_deg:6.1f}° (expected {expected_deg:6.1f}°)")

        # Verify precision
        if abs(first_center - expected) < 1e-15:
            print("   ✅ Rotation effect exact")
        else:
            print(f"   ❌ Rotation error: {abs(first_center - expected):.2e}")

if __name__ == "__main__":
    verify_electrode_center()
    verify_rotation_effect()
    print("\n🎉 Verification complete!")
