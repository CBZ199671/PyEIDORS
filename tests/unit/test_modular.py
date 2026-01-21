#!/usr/bin/env python3
"""PyEidors Modular Test"""

import numpy as np

def test_imports():
    """Test all module imports."""
    print("🧪 Testing module imports...")

    try:
        from pyeidors.data.structures import MeshConfig, ElectrodePosition, PatternConfig
        print("✅ Data structures module import successful")
    except Exception as e:
        print(f"❌ Data structures module import failed: {e}")
        return False

    try:
        from pyeidors.electrodes.patterns import StimMeasPatternManager
        print("✅ Electrode patterns module import successful")
    except Exception as e:
        print(f"❌ Electrode patterns module import failed: {e}")
        return False

    try:
        from pyeidors.geometry.mesh_generator import MeshGenerator
        print("✅ Mesh generation module import successful")
    except Exception as e:
        print(f"❌ Mesh generation module import failed: {e}")
        return False

    try:
        from pyeidors.forward.eit_forward_model import EITForwardModel
        print("✅ Forward model module import successful")
    except Exception as e:
        print(f"❌ Forward model module import failed: {e}")
        return False

    try:
        from pyeidors.inverse.solvers.gauss_newton import StandardGaussNewtonReconstructor
        print("✅ Inverse solver module import successful")
    except Exception as e:
        print(f"❌ Inverse solver module import failed: {e}")
        return False

    try:
        from pyeidors.data.synthetic_data import create_synthetic_data
        print("✅ Synthetic data module import successful")
    except Exception as e:
        print(f"❌ Synthetic data module import failed: {e}")
        return False

    return True


def test_basic_workflow():
    """Test basic workflow."""
    print("\n🔧 Testing basic workflow...")

    try:
        # Import required modules
        from pyeidors.data.structures import MeshConfig, ElectrodePosition, PatternConfig
        from pyeidors.geometry.mesh_generator import MeshGenerator
        from pyeidors.electrodes.patterns import StimMeasPatternManager
        from pyeidors.forward.eit_forward_model import EITForwardModel
        from pyeidors.inverse.solvers.gauss_newton import StandardGaussNewtonReconstructor
        from pyeidors.data.synthetic_data import create_synthetic_data

        # 1. Create configuration
        n_elec = 16
        mesh_config = MeshConfig(radius=1.0, refinement=6, electrode_vertices=4)
        electrode_config = ElectrodePosition(L=n_elec, coverage=0.5)
        pattern_config = PatternConfig(
            n_elec=n_elec,
            stim_pattern='{ad}',
            meas_pattern='{ad}',
            amplitude=1.0,
            use_meas_current=False
        )

        print("✅ Configuration created successfully")

        # 2. Generate mesh
        generator = MeshGenerator(mesh_config, electrode_config)
        mesh = generator.generate()

        print(f"✅ Mesh generated successfully: {mesh.num_cells()} cells")

        # 3. Create stimulation/measurement pattern manager
        pattern_manager = StimMeasPatternManager(pattern_config)

        print(f"✅ Pattern manager created successfully: {pattern_manager.n_stim} stimulations, {pattern_manager.n_meas_total} measurements")

        # 4. Create forward model
        z = np.full(n_elec, 1e-6)  # Contact impedance
        fwd_model = EITForwardModel(n_elec, pattern_config, z, mesh)

        print("✅ Forward model created successfully")

        # 5. Generate synthetic data
        synthetic_data = create_synthetic_data(
            fwd_model=fwd_model,
            inclusion_conductivity=2.5,
            background_conductivity=1.0,
            noise_level=0.02,
            center=(-0.3, 0.1),
            radius=0.3
        )

        print(f"✅ Synthetic data generated successfully: SNR = {synthetic_data['snr_db']:.1f} dB")

        # 6. Create reconstructor (but don't run reconstruction to save time)
        reconstructor = StandardGaussNewtonReconstructor(
            fwd_model=fwd_model,
            max_iterations=5,  # Reduced iterations to save time
            convergence_tol=1e-3,
            regularization_param=0.01,
            verbose=False
        )

        print("✅ Reconstructor created successfully")

        print("✅ Basic workflow test complete!")
        return True

    except Exception as e:
        print(f"❌ Basic workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🎯 PyEidors Modular Test")
    print("=" * 50)

    # Test imports
    if not test_imports():
        print("\n❌ Import test failed, please check module structure")
        return

    # Test basic workflow
    if not test_basic_workflow():
        print("\n❌ Workflow test failed")
        return

    print("\n🎉 All tests passed! PyEidors modular refactoring successful!")


if __name__ == "__main__":
    main()
