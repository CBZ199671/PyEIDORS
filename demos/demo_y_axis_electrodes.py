#!/usr/bin/env python3
"""
Demonstrate modified electrode positions (starting from positive Y-axis).
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent
SRC_PATH = PROJECT_ROOT / 'src'

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Configure font support
try:
    from pyeidors.utils.chinese_font_config import configure_chinese_font
    configure_chinese_font()
except ImportError:
    # Fallback
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

def demo_y_axis_electrodes():
    """Demonstrate electrode positions starting from positive Y-axis."""
    print("🎨 Generating Y-axis starting electrode position demo...")

    from pyeidors.geometry.optimized_mesh_generator import ElectrodePosition

    # Create electrodes with different configurations
    configs = [
        ("8-electrode system", ElectrodePosition(L=8, coverage=0.5)),
        ("16-electrode system", ElectrodePosition(L=16, coverage=0.5)),
        ("16-electrode compact", ElectrodePosition(L=16, coverage=0.3)),
        ("Rotated 30°", ElectrodePosition(L=8, coverage=0.5, rotation=pi/6)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for i, (name, config) in enumerate(configs):
        ax = axes[i]

        # Get electrode positions
        positions = config.positions

        # Draw circle boundary
        theta = np.linspace(0, 2*pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.3, linewidth=1)

        # Draw coordinate axes
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        # Mark positive Y-axis
        ax.arrow(0, 0, 0, 1.2, head_width=0.05, head_length=0.05,
                fc='red', ec='red', alpha=0.7)
        ax.text(0.1, 1.1, 'Y+', fontsize=12, color='red', weight='bold')

        # Draw electrodes
        for j, (start, end) in enumerate(positions):
            # Electrode arc
            theta_elec = np.linspace(start, end, 20)
            x_elec = np.cos(theta_elec)
            y_elec = np.sin(theta_elec)
            ax.plot(x_elec, y_elec, 'b-', linewidth=4, alpha=0.8)

            # Electrode number
            mid_angle = (start + end) / 2
            label_radius = 1.15
            x_label = label_radius * np.cos(mid_angle)
            y_label = label_radius * np.sin(mid_angle)

            # Special marker for first electrode
            if j == 0:
                ax.plot(x_label, y_label, 'ro', markersize=8)
                ax.text(x_label, y_label-0.15, f'{j+1}', ha='center', va='center',
                       fontsize=10, weight='bold', color='red',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            else:
                ax.text(x_label, y_label, f'{j+1}', ha='center', va='center',
                       fontsize=9, color='blue')

        # Draw lines connecting electrode centers to show order
        centers_x, centers_y = [], []
        for start, end in positions:
            mid_angle = (start + end) / 2
            centers_x.append(np.cos(mid_angle))
            centers_y.append(np.sin(mid_angle))

        # Connect adjacent electrode centers
        for j in range(len(centers_x)):
            next_j = (j + 1) % len(centers_x)
            ax.plot([centers_x[j], centers_x[next_j]], [centers_y[j], centers_y[next_j]],
                   'g--', alpha=0.3, linewidth=1)

        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect('equal')
        ax.set_title(f'{name}\nElectrode 1 starts from Y+ (red marker)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = DEMO_DIR / 'y_axis_electrode_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print("✅ Y-axis electrode position demo complete, saved as y_axis_electrode_demo.png")

def demo_before_after_comparison():
    """Compare electrode positions before and after modification."""
    print("🎨 Generating before/after comparison demo...")

    from pyeidors.geometry.optimized_mesh_generator import ElectrodePosition

    # Simulate old electrode positions (starting from positive X-axis)
    def old_positions(L, coverage):
        electrode_size = 2 * pi / L * coverage
        gap_size = 2 * pi / L * (1 - coverage)
        positions = []
        for i in range(L):
            start = electrode_size * i + gap_size * i  # Start from 0
            end = electrode_size * (i + 1) + gap_size * i
            positions.append((start, end))
        return positions

    # Create comparison figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    L = 8  # Use 8 electrodes for easier observation
    coverage = 0.5

    # Left plot: Before modification (starting from positive X-axis)
    old_pos = old_positions(L, coverage)

    theta = np.linspace(0, 2*pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Mark positive X-axis
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
    ax1.set_title('Before: Electrode 1 starts from X+\n(Traditional approach)')
    ax1.grid(True, alpha=0.3)

    # Right plot: After modification (starting from positive Y-axis)
    new_config = ElectrodePosition(L=L, coverage=coverage)
    new_pos = new_config.positions

    ax2.plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Mark positive Y-axis
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
    ax2.set_title('After: Electrode 1 starts from Y+\n(Improved approach)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = DEMO_DIR / 'electrode_position_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print("✅ Before/after comparison demo complete, saved as electrode_position_comparison.png")

def run_demo():
    """Run complete demo."""
    print("🎬 Electrode Y-axis Starting Position Demo")
    print("=" * 50)

    try:
        demo_y_axis_electrodes()
        demo_before_after_comparison()

        print("\n🎉 All demos complete!")
        print("📊 Generated files:")
        print("   - y_axis_electrode_demo.png: Y-axis starting electrode configuration demo")
        print("   - electrode_position_comparison.png: Before/after comparison")

    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_demo()
