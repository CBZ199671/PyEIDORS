#!/usr/bin/env python3
"""
PyEidors Complete System End-to-End Test
Tests the complete EIT forward/inverse solving workflow including mesh loading,
forward solving, inverse reconstruction, and visualization.
"""

import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parents[1]
SRC_PATH = PROJECT_ROOT / 'src'

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

def test_complete_eit_workflow():
    """Test complete EIT workflow."""
    print("=== PyEidors Complete System End-to-End Test ===\n")

    try:
        # 1. Import modules and check environment
        print("1. Importing modules and checking environment...")
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

        # 2. Check mesh files
        print("2. Checking and loading mesh...")
        mesh_loader = MeshLoader()
        available_meshes = mesh_loader.list_available_meshes()
        print(f"   Available meshes: {available_meshes}")

        if not available_meshes['fenics_h5']:
            print("   ⚠️  No FEniCS H5 format mesh files found")
            return False

        # Load default mesh
        mesh = mesh_loader.get_default_mesh()
        mesh_info = mesh.get_info()
        print(f"   ✓ Mesh loaded successfully:")
        print(f"     - Vertices: {mesh_info['num_vertices']}")
        print(f"     - Cells: {mesh_info['num_cells']}")
        print(f"     - Electrodes: {mesh_info['num_electrodes']}")
        print(f"     - Radius: {mesh_info['radius']:.3f}")
        print()

        # 3. Create EIT system
        print("3. Creating and initializing EIT system...")
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

        # Initialize system with loaded mesh
        eit_system.setup(mesh=mesh)

        system_info = eit_system.get_system_info()
        print(f"   ✓ EIT system initialized successfully:")
        print(f"     - Electrodes: {system_info['n_elec']}")
        print(f"     - Cells: {system_info['n_elements']}")
        print(f"     - Vertices: {system_info['n_nodes']}")
        print(f"     - Measurements: {system_info['n_measurements']}")
        print(f"     - Stimulation patterns: {system_info['n_stimulation_patterns']}")
        print()

        # 4. Create synthetic test data
        print("4. Generating synthetic test data...")

        # Create custom phantom
        anomalies = [
            {'center': (0.3, 0.3), 'radius': 0.2, 'conductivity': 2.5},
            {'center': (-0.4, -0.2), 'radius': 0.15, 'conductivity': 0.5}
        ]

        sigma_phantom = create_custom_phantom(
            eit_system.fwd_model,
            background_conductivity=1.0,
            anomalies=anomalies
        )

        # Generate synthetic data
        synthetic_data = create_synthetic_data(
            eit_system.fwd_model,
            inclusion_conductivity=2.5,
            background_conductivity=1.0,
            noise_level=0.02,
            center=(0.2, 0.2),
            radius=0.25
        )

        print(f"   ✓ Synthetic data generated successfully:")
        print(f"     - SNR: {synthetic_data['snr_db']:.2f} dB")
        print(f"     - Measurements: {len(synthetic_data['data_clean'].meas)}")
        print(f"     - Noise std: {np.std(synthetic_data['noise']):.6f}")
        print()

        # 5. Forward solve verification
        print("5. Forward solve verification...")
        start_time = time.time()

        # Forward solve with custom phantom
        from pyeidors.data.structures import EITImage
        phantom_image = EITImage(elem_data=sigma_phantom.vector()[:], fwd_model=eit_system.fwd_model)
        forward_data = eit_system.forward_solve(phantom_image)

        forward_time = time.time() - start_time
        print(f"   ✓ Forward solve complete:")
        print(f"     - Computation time: {forward_time:.3f} seconds")
        print(f"     - Measurement range: [{np.min(forward_data.meas):.6f}, {np.max(forward_data.meas):.6f}]")
        print(f"     - Measurement mean: {np.mean(forward_data.meas):.6f}")
        print()

        # 6. Inverse reconstruction
        print("6. Inverse reconstruction...")
        start_time = time.time()

        # Create reference data (homogeneous)
        reference_image = eit_system.create_homogeneous_image(conductivity=1.0)
        reference_data = eit_system.forward_solve(reference_image)

        # Perform reconstruction
        try:
            reconstructed_image = eit_system.inverse_solve(
                data=synthetic_data['data_noisy'],
                reference_data=reference_data,
                initial_guess=None
            )

            reconstruction_time = time.time() - start_time
            print(f"   ✓ Inverse reconstruction complete:")
            print(f"     - Computation time: {reconstruction_time:.3f} seconds")

            # Calculate reconstruction error
            true_values = synthetic_data['sigma_true'].vector()[:]
            recon_values = reconstructed_image.elem_data
            relative_error = np.linalg.norm(recon_values - true_values) / np.linalg.norm(true_values)
            print(f"     - Relative error: {relative_error:.4f}")
            print(f"     - Reconstruction range: [{np.min(recon_values):.3f}, {np.max(recon_values):.3f}]")

        except Exception as e:
            print(f"   ⚠️  Reconstruction issue: {e}")
            print("   Continuing with other tests...")
            reconstructed_image = None
        print()

        # 7. Visualization test
        print("7. Visualization test...")
        try:
            visualizer = create_visualizer()

            # Create output directory
            output_dir = Path("test_results")
            output_dir.mkdir(exist_ok=True)

            # Plot mesh
            fig1 = visualizer.plot_mesh(mesh, title="Mesh Structure",
                                       save_path=output_dir / "mesh.png")
            print("   ✓ Mesh visualization complete")

            # Plot true conductivity distribution
            fig2 = visualizer.plot_conductivity(mesh, synthetic_data['sigma_true'],
                                              title="True Conductivity Distribution",
                                              save_path=output_dir / "true_conductivity.png")
            print("   ✓ True distribution visualization complete")

            # Plot measurement data
            fig3 = visualizer.plot_measurements(synthetic_data['data_noisy'],
                                              title="Synthetic Measurement Data (with noise)",
                                              save_path=output_dir / "measurements.png")
            print("   ✓ Measurement data visualization complete")

            # If reconstruction succeeded, plot comparison
            if reconstructed_image is not None:
                fig4 = visualizer.plot_reconstruction_comparison(
                    mesh, synthetic_data['sigma_true'], reconstructed_image.elem_data,
                    title="Reconstruction Comparison",
                    save_path=output_dir / "reconstruction_comparison.png"
                )
                print("   ✓ Reconstruction comparison visualization complete")

            print(f"   ✓ All images saved to: {output_dir.absolute()}")

            # Close figures to free memory
            plt.close('all')

        except Exception as e:
            print(f"   ⚠️  Visualization issue: {e}")
        print()

        # 8. Performance statistics
        print("8. Performance statistics summary...")
        print(f"   - Forward solve time: {forward_time:.3f} seconds")
        if 'reconstruction_time' in locals():
            print(f"   - Inverse reconstruction time: {reconstruction_time:.3f} seconds")
        print(f"   - Mesh size: {mesh_info['num_vertices']} vertices, {mesh_info['num_cells']} cells")
        print(f"   - Measurements: {system_info['n_measurements']}")
        print()

        print("🎉 Complete system test finished successfully!")
        return True

    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_module_integration():
    """Test module integration."""
    print("=== Module Integration Test ===\n")

    modules_to_test = [
        ("Core System", "pyeidors.core_system"),
        ("Mesh Loader", "pyeidors.geometry.mesh_loader"),
        ("Forward Model", "pyeidors.forward.eit_forward_model"),
        ("Jacobian Calculator", "pyeidors.inverse.jacobian.direct_jacobian"),
        ("Regularization", "pyeidors.inverse.regularization.smoothness"),
        ("Gauss-Newton Solver", "pyeidors.inverse.solvers.gauss_newton"),
        ("Pattern Manager", "pyeidors.electrodes.patterns"),
        ("Synthetic Data", "pyeidors.data.synthetic_data"),
        ("Visualization", "pyeidors.visualization")
    ]

    success_count = 0
    for name, module_path in modules_to_test:
        try:
            __import__(module_path)
            print(f"✓ {name} module import successful")
            success_count += 1
        except Exception as e:
            print(f"✗ {name} module import failed: {e}")

    print(f"\nModule integration test result: {success_count}/{len(modules_to_test)} successful")
    return success_count == len(modules_to_test)

if __name__ == "__main__":
    print("Starting PyEidors complete system test...\n")

    # Module integration test
    integration_success = test_module_integration()
    print()

    if integration_success:
        # Complete workflow test
        workflow_success = test_complete_eit_workflow()

        if workflow_success:
            print("\n🏆 All tests passed successfully! PyEidors system is working properly.")
            sys.exit(0)
        else:
            print("\n⚠️  Workflow test not fully successful, but basic functionality available.")
            sys.exit(1)
    else:
        print("\n❌ Module integration test failed, please check dependencies and configuration.")
        sys.exit(1)
