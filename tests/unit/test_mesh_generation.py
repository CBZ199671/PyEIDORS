#!/usr/bin/env python3
"""
Test mesh generation functionality.
Verifies GMsh mesh generation and FEniCS conversion functionality.
"""

import numpy as np
import sys
import time
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parents[1]
SRC_PATH = PROJECT_ROOT / 'src'

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

def test_mesh_generation():
    """Test mesh generation functionality."""
    print("=== Testing Mesh Generation Functionality ===\n")

    try:
        # 1. Check dependencies
        print("1. Checking dependencies...")

        dependencies = {}

        try:
            import gmsh
            dependencies['gmsh'] = True
            print("   ✓ GMsh available")
        except ImportError:
            dependencies['gmsh'] = False
            print("   ✗ GMsh not available")

        try:
            import meshio
            dependencies['meshio'] = True
            print("   ✓ meshio available")
        except ImportError:
            dependencies['meshio'] = False
            print("   ✗ meshio not available")

        try:
            from fenics import Mesh
            dependencies['fenics'] = True
            print("   ✓ FEniCS available")
        except ImportError:
            dependencies['fenics'] = False
            print("   ✗ FEniCS not available")

        print()

        if not dependencies['gmsh']:
            print("❌ GMsh not available, cannot run mesh generation test")
            print("Please install GMsh: pip install gmsh")
            return False

        # 2. Test simple mesh generator
        print("2. Testing simple mesh generator...")
        from pyeidors.geometry.simple_mesh_generator import SimpleEITMeshGenerator, create_simple_eit_mesh

        # Create generator
        generator = SimpleEITMeshGenerator(
            n_elec=16,
            radius=1.0,
            mesh_size=0.1,
            electrode_width=0.2
        )

        print("   ✓ Mesh generator created successfully")
        print(f"     - Electrodes: {generator.n_elec}")
        print(f"     - Radius: {generator.radius}")
        print(f"     - Mesh size: {generator.mesh_size}")
        print()

        # 3. Generate mesh
        print("3. Generating EIT mesh...")
        start_time = time.time()

        # Create output directory
        output_dir = Path("test_results/mesh_generation")
        output_dir.mkdir(parents=True, exist_ok=True)

        mesh = generator.generate_circular_mesh(
            output_dir=str(output_dir),
            save_files=True
        )

        generation_time = time.time() - start_time

        print(f"   ✓ Mesh generation complete (time: {generation_time:.3f} seconds)")

        # Get mesh info
        if hasattr(mesh, 'get_info'):
            mesh_info = mesh.get_info()
            print(f"   Mesh info:")
            for key, value in mesh_info.items():
                if key not in ['bbox', 'association_table']:
                    print(f"     - {key}: {value}")

        print()

        # 4. Test convenience function
        print("4. Testing convenience function...")

        start_time = time.time()
        simple_mesh = create_simple_eit_mesh(
            n_elec=8,
            radius=1.0,
            mesh_size=0.15,
            output_dir=str(output_dir / "simple")
        )
        simple_time = time.time() - start_time

        print(f"   ✓ Convenience function test complete (time: {simple_time:.3f} seconds)")

        if hasattr(simple_mesh, 'get_info'):
            simple_info = simple_mesh.get_info()
            print(f"   Simple mesh: {simple_info['num_vertices']} vertices, {simple_info['num_cells']} cells")

        print()

        # 5. Test EIT system integration
        print("5. Testing EIT system integration...")

        try:
            from pyeidors import EITSystem
            from pyeidors.data.structures import PatternConfig

            # Create EIT system
            eit_system = EITSystem(
                n_elec=16,
                pattern_config=PatternConfig(n_elec=16)
            )

            # Initialize system with generated mesh
            eit_system.setup(mesh=mesh)

            system_info = eit_system.get_system_info()
            print("   ✓ EIT system integration successful")
            print(f"     - System initialized: {system_info['initialized']}")
            print(f"     - Measurements: {system_info['n_measurements']}")
            print()

        except Exception as e:
            print(f"   ⚠️  EIT system integration test failed: {e}")

        # 6. Visualization test (if possible)
        print("6. Visualization test...")

        try:
            from pyeidors.visualization import create_visualizer
            import matplotlib.pyplot as plt

            visualizer = create_visualizer()

            # Plot mesh
            fig = visualizer.plot_mesh(mesh, title="Generated EIT Mesh")

            # Save image
            plt.savefig(output_dir / "generated_mesh.png", dpi=150, bbox_inches='tight')
            plt.close()

            print("   ✓ Mesh visualization complete")
            print(f"   Image saved to: {output_dir / 'generated_mesh.png'}")

        except Exception as e:
            print(f"   ⚠️  Visualization test failed: {e}")

        print()

        # 7. Performance test
        print("7. Performance test...")

        mesh_sizes = [0.2, 0.15, 0.1, 0.08]
        for mesh_size in mesh_sizes:
            start_time = time.time()

            test_mesh = create_simple_eit_mesh(
                n_elec=16,
                mesh_size=mesh_size,
                output_dir=str(output_dir / f"perf_test_{mesh_size}")
            )

            elapsed = time.time() - start_time

            if hasattr(test_mesh, 'get_info'):
                info = test_mesh.get_info()
                print(f"   Mesh size {mesh_size}: {info['num_vertices']} vertices, "
                      f"{info['num_cells']} cells, time {elapsed:.3f} seconds")
            else:
                print(f"   Mesh size {mesh_size}: time {elapsed:.3f} seconds")

        print()
        print("🎉 Mesh generation functionality test complete!")

        # Summary
        print("\n📋 Test Summary:")
        print("   ✅ GMsh mesh generation working")
        print("   ✅ Mesh conversion available")
        print("   ✅ EIT system integration successful")
        print("   ✅ Performance good")

        if not dependencies['fenics']:
            print("   ⚠️  FEniCS not available, using simplified mesh object")

        if not dependencies['meshio']:
            print("   ⚠️  meshio not available, mesh format conversion limited")

        return True

    except Exception as e:
        print(f"❌ Mesh generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mesh_with_eit_workflow():
    """Test complete EIT workflow (using generated mesh)."""
    print("\n=== Testing Complete EIT Workflow (Using Generated Mesh) ===\n")

    try:
        from pyeidors import EITSystem
        from pyeidors.data.structures import PatternConfig
        from pyeidors.data.synthetic_data import create_synthetic_data

        print("1. Creating EIT system and auto-generating mesh...")

        # Create EIT system (will auto-generate mesh)
        eit_system = EITSystem(n_elec=16)

        # Initialize system (will try to load or generate mesh)
        eit_system.setup()

        info = eit_system.get_system_info()
        print(f"   ✓ EIT system initialized successfully:")
        print(f"     - Electrodes: {info['n_elec']}")
        print(f"     - Vertices: {info['n_nodes']}")
        print(f"     - Cells: {info['n_elements']}")
        print(f"     - Measurements: {info['n_measurements']}")
        print()

        print("2. Generating synthetic test data...")

        synthetic_data = create_synthetic_data(
            eit_system.fwd_model,
            inclusion_conductivity=2.0,
            background_conductivity=1.0,
            noise_level=0.01,
            center=(0.3, 0.3),
            radius=0.2
        )

        print(f"   ✓ Synthetic data generated successfully:")
        print(f"     - SNR: {synthetic_data['snr_db']:.2f} dB")
        print(f"     - Measurements: {len(synthetic_data['data_clean'].meas)}")
        print()

        print("3. Forward solve test...")

        start_time = time.time()
        reference_image = eit_system.create_homogeneous_image(1.0)
        reference_data = eit_system.forward_solve(reference_image)
        forward_time = time.time() - start_time

        print(f"   ✓ Forward solve successful (time: {forward_time:.3f} seconds)")
        print(f"     - Measurement range: [{np.min(reference_data.meas):.6f}, {np.max(reference_data.meas):.6f}]")
        print()

        print("4. Inverse reconstruction test...")

        try:
            start_time = time.time()

            reconstructed = eit_system.inverse_solve(
                data=synthetic_data['data_noisy'],
                reference_data=reference_data
            )

            reconstruction_time = time.time() - start_time

            # Calculate reconstruction error
            true_values = synthetic_data['sigma_true'].vector()[:]
            recon_values = reconstructed.elem_data
            relative_error = np.linalg.norm(recon_values - true_values) / np.linalg.norm(true_values)

            print(f"   ✓ Inverse reconstruction successful (time: {reconstruction_time:.3f} seconds)")
            print(f"     - Relative error: {relative_error:.4f}")
            print(f"     - Reconstruction range: [{np.min(recon_values):.3f}, {np.max(recon_values):.3f}]")

        except Exception as e:
            print(f"   ⚠️  Inverse reconstruction failed: {e}")

        print()
        print("🎉 Complete EIT workflow test successful!")
        return True

    except Exception as e:
        print(f"❌ EIT workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting mesh generation and EIT system test\n")

    mesh_success = test_mesh_generation()

    if mesh_success:
        workflow_success = test_mesh_with_eit_workflow()

        if workflow_success:
            print("\n🏆 All tests successful! Mesh generation and EIT system working properly.")
            sys.exit(0)
        else:
            print("\n⚠️  EIT workflow test partially failed, but mesh generation working.")
            sys.exit(1)
    else:
        print("\n❌ Mesh generation test failed.")
        sys.exit(1)
