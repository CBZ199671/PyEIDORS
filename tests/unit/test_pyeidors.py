#!/usr/bin/env python3
"""
PyEidors Module Test Script
Tests basic functionality of each module.
"""

import numpy as np
import sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parents[1]
SRC_PATH = PROJECT_ROOT / 'src'

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

def test_imports():
    """Test module imports."""
    print("Testing module imports...")

    try:
        import pyeidors
        print("✓ pyeidors main module import successful")

        # Check environment
        env_info = pyeidors.check_environment()
        print(f"✓ Environment check: {env_info}")

    except ImportError as e:
        print(f"✗ Module import failed: {e}")
        return False

    try:
        from pyeidors.core_system import EITSystem
        print("✓ EITSystem import successful")
    except ImportError as e:
        print(f"✗ EITSystem import failed: {e}")
        return False

    try:
        from pyeidors.data.structures import PatternConfig, EITData, EITImage, MeshConfig, ElectrodePosition
        print("✓ Data structures import successful")
    except ImportError as e:
        print(f"✗ Data structures import failed: {e}")
        return False

    try:
        from pyeidors.forward.eit_forward_model import EITForwardModel
        print("✓ Forward model import successful")
    except ImportError as e:
        print(f"✗ Forward model import failed: {e}")
        return False

    try:
        from pyeidors.inverse.solvers.gauss_newton import ModularGaussNewtonReconstructor
        print("✓ Gauss-Newton solver import successful")
    except ImportError as e:
        print(f"✗ Gauss-Newton solver import failed: {e}")
        return False

    try:
        from pyeidors.inverse.jacobian.direct_jacobian import DirectJacobianCalculator
        print("✓ Jacobian calculator import successful")
    except ImportError as e:
        print(f"✗ Jacobian calculator import failed: {e}")
        return False

    try:
        from pyeidors.inverse.regularization.smoothness import SmoothnessRegularization
        print("✓ Smoothness regularization import successful")
    except ImportError as e:
        print(f"✗ Smoothness regularization import failed: {e}")
        return False

    try:
        from pyeidors.electrodes.patterns import StimMeasPatternManager
        print("✓ Stimulation/measurement pattern manager import successful")
    except ImportError as e:
        print(f"✗ Stimulation/measurement pattern manager import failed: {e}")
        return False

    return True

def test_data_structures():
    """Test data structures."""
    print("\nTesting data structures...")

    try:
        from pyeidors.data.structures import PatternConfig, EITData, EITImage, MeshConfig, ElectrodePosition

        # Test PatternConfig
        config = PatternConfig(n_elec=16)
        print(f"✓ PatternConfig created successfully: {config}")

        # Test EITData
        data = EITData(
            meas=np.random.rand(10),
            stim_pattern=np.random.rand(16, 4),
            n_elec=16,
            n_stim=4,
            n_meas=10
        )
        print(f"✓ EITData created successfully: {data.type}")

        # Test EITImage
        img = EITImage(elem_data=np.ones(100), fwd_model=None)
        conductivity = img.get_conductivity()
        print(f"✓ EITImage created successfully, conductivity shape: {conductivity.shape}")

        # Test MeshConfig
        mesh_config = MeshConfig(radius=1.0, refinement=8)
        print(f"✓ MeshConfig created successfully: {mesh_config}")

        # Test ElectrodePosition
        electrode_pos = ElectrodePosition.create_circular(n_elec=16)
        print(f"✓ ElectrodePosition created successfully, electrode count: {electrode_pos.L}")

        return True
    except Exception as e:
        print(f"✗ Data structures test failed: {e}")
        return False

def test_eit_system():
    """Test EIT system."""
    print("\nTesting EIT system...")

    try:
        from pyeidors.core_system import EITSystem
        from pyeidors.data.structures import PatternConfig, MeshConfig

        # Create EIT system
        pattern_config = PatternConfig(n_elec=16)
        mesh_config = MeshConfig()

        eit_system = EITSystem(
            n_elec=16,
            pattern_config=pattern_config,
            mesh_config=mesh_config
        )

        print(f"✓ EIT system created successfully")

        # Get system info
        info = eit_system.get_system_info()
        print(f"✓ System info: {info}")

        # Test creating homogeneous image (this will fail before setup, as expected)
        try:
            img = eit_system.create_homogeneous_image()
            print("✗ This should not succeed because system is not initialized")
        except RuntimeError as e:
            print(f"✓ Correctly caught uninitialized error: {e}")

        return True
    except Exception as e:
        print(f"✗ EIT system test failed: {e}")
        return False

def test_pattern_manager():
    """Test stimulation/measurement pattern manager."""
    print("\nTesting stimulation/measurement pattern manager...")

    try:
        from pyeidors.electrodes.patterns import StimMeasPatternManager
        from pyeidors.data.structures import PatternConfig

        config = PatternConfig(n_elec=16)
        manager = StimMeasPatternManager(config)

        print(f"✓ Pattern manager created successfully")
        print(f"✓ Number of stimulations: {manager.n_stim}")
        print(f"✓ Total measurements: {manager.n_meas_total}")
        print(f"✓ Stimulation matrix shape: {manager.stim_matrix.shape}")

        return True
    except Exception as e:
        print(f"✗ Pattern manager test failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("=== PyEidors Module Test Started ===")

    tests = [
        test_imports,
        test_data_structures,
        test_eit_system,
        test_pattern_manager
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n=== Test Results Summary ===")
    total = len(results)
    passed = sum(results)
    failed = total - passed

    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("✓ All tests passed!")
        return True
    else:
        print(f"✗ {failed} test(s) failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
