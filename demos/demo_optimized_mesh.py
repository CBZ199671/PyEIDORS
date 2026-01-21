#!/usr/bin/env python3
"""
Optimized Mesh Generator Demo
Demonstrates the new mesh generator functionality based on reference implementation.
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
    """Demonstrate electrode position configuration."""
    print("🔬 Demonstrating electrode position configuration")
    print("=" * 40)

    from pyeidors.geometry.optimized_mesh_generator import ElectrodePosition

    configs = [
        ("Standard 16-electrode", ElectrodePosition(L=16, coverage=0.5)),
        ("Compact 16-electrode", ElectrodePosition(L=16, coverage=0.3)),
        ("Wide electrodes", ElectrodePosition(L=16, coverage=0.8)),
        ("8-electrode", ElectrodePosition(L=8, coverage=0.5)),
        ("32-electrode", ElectrodePosition(L=32, coverage=0.5)),
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
            ax.plot(x_elec, y_elec, 'b-', linewidth=3, label='Electrode' if j == 0 else '')
            mid_angle = (start + end) / 2
            ax.text(1.1*np.cos(mid_angle), 1.1*np.sin(mid_angle), str(j + 1), ha='center', va='center', fontsize=8)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(f'{name}\n{config.L} electrodes, coverage {config.coverage}')
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend()

    if len(configs) < len(axes):
        axes[-1].set_visible(False)

    plt.tight_layout()
    output_path = DEMO_DIR / 'electrode_positions_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print("✅ Electrode position configuration demo complete, saved as electrode_positions_demo.png")


def demo_mesh_generation():
    """Demonstrate mesh generation."""
    print("\n🔬 Demonstrating mesh generation")
    print("=" * 40)

    from pyeidors.geometry.optimized_mesh_generator import (
        OptimizedMeshGenerator, OptimizedMeshConfig, ElectrodePosition
    )

    configs = [
        ("Coarse mesh", OptimizedMeshConfig(radius=1.0, refinement=2)),
        ("Medium mesh", OptimizedMeshConfig(radius=1.0, refinement=4)),
        ("Fine mesh", OptimizedMeshConfig(radius=1.0, refinement=6)),
    ]

    electrodes = ElectrodePosition(L=16, coverage=0.5)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, (name, config) in enumerate(configs):
            print(f"   Generating {name}...")
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
                ax.text(0, 0, f"Mesh generated successfully\nbut cannot be visualized\n({type(mesh).__name__})", ha='center', va='center', fontsize=12)

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
            ax.set_title(f'{name}\nVertices: {n_vertices}, Cells: {n_cells}')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = DEMO_DIR / 'mesh_generation_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print("✅ Mesh generation demo complete, saved as mesh_generation_demo.png")


def demo_convenience_functions():
    """Demonstrate convenience functions."""
    print("\n🔬 Demonstrating convenience functions")
    print("=" * 40)

    from pyeidors.geometry.optimized_mesh_generator import create_eit_mesh

    with tempfile.TemporaryDirectory() as temp_dir:
        print("   Creating standard EIT mesh using convenience function...")
        mesh = create_eit_mesh(
            n_elec=16,
            radius=1.0,
            refinement=5,
            electrode_coverage=0.5,
            output_dir=temp_dir
        )

        if hasattr(mesh, 'num_vertices'):
            print(f"   ✅ Successfully created mesh: {mesh.num_vertices()} vertices, {mesh.num_cells()} cells")
        else:
            print(f"   ✅ Successfully created mesh info: {type(mesh).__name__}")

        output_path = Path(temp_dir)
        msh_files = list(output_path.glob('*.msh'))
        xdmf_files = list(output_path.glob('*.xdmf'))
        ini_files = list(output_path.glob('*.ini'))
        print(f"   📁 Generated files: {len(msh_files)} .msh, {len(xdmf_files)} .xdmf, {len(ini_files)} .ini")


def demo_mesh_quality():
    """Demonstrate mesh quality analysis."""
    print("\n🔬 Demonstrating mesh quality analysis")
    print("=" * 40)

    from pyeidors.geometry.optimized_mesh_generator import create_eit_mesh

    with tempfile.TemporaryDirectory() as temp_dir:
        mesh_configs = [
            ("Basic mesh", {"refinement": 3}),
            ("Standard mesh", {"refinement": 5}),
            ("High quality mesh", {"refinement": 7}),
        ]

        results = []

        for name, config in mesh_configs:
            print(f"   Generating {name}...")
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
                print(f"     Vertices: {n_vertices}, Cells: {n_cells}, Density: {density:.1f} cells/unit²")

        if results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            names = [r['name'] for r in results]
            vertices = [r['vertices'] for r in results]
            cells = [r['cells'] for r in results]
            x = np.arange(len(names))
            width = 0.35

            ax1.bar(x - width/2, vertices, width, label='Vertices', alpha=0.7)
            ax1.bar(x + width/2, cells, width, label='Cells', alpha=0.7)
            ax1.set_xlabel('Mesh Configuration')
            ax1.set_ylabel('Count')
            ax1.set_title('Mesh Size Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(names)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            refinements = [r['refinement'] for r in results]
            ax2.plot(refinements, cells, 'bo-', markersize=8)
            ax2.set_xlabel('Refinement Level')
            ax2.set_ylabel('Cell Count')
            ax2.set_title('Relationship Between Refinement Level and Mesh Density')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            output_path = DEMO_DIR / 'mesh_quality_demo.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            print("   ✅ Mesh quality analysis complete, saved as mesh_quality_demo.png")


def run_demo():
    """Run complete demo."""
    print("🎬 Optimized Mesh Generator Demo")
    print("=" * 60)

    try:
        demo_electrode_positions()
        demo_mesh_generation()
        demo_convenience_functions()
        demo_mesh_quality()

        print("\n🎉 All demos complete!")
        print("📊 Generated files:")
        print("   - electrode_positions_demo.png: Electrode position configuration comparison")
        print("   - mesh_generation_demo.png: Different mesh refinement comparison")
        print("   - mesh_quality_demo.png: Mesh quality analysis")

    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_demo()
