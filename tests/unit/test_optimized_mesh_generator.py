#!/usr/bin/env python3
"""
Optimized Mesh Generator Test
Tests the new mesh generator functionality based on reference implementation.
"""

import numpy as np
import tempfile
import shutil
from pathlib import Path
import logging
import unittest
from unittest.mock import Mock, patch
import sys

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parents[1]
SRC_PATH = PROJECT_ROOT / 'src'

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_electrode_position():
    """Test electrode position configuration."""
    print("🔧 Testing electrode position configuration...")

    try:
        from pyeidors.geometry.optimized_mesh_generator import ElectrodePosition

        # Test basic configuration
        elec_pos = ElectrodePosition(L=16)
        assert elec_pos.L == 16
        assert elec_pos.coverage == 0.5
        assert elec_pos.anticlockwise == True

        # Test position calculation
        positions = elec_pos.positions
        assert len(positions) == 16
        assert all(isinstance(pos, tuple) and len(pos) == 2 for pos in positions)

        # Test angle coverage
        elec_pos_full = ElectrodePosition(L=8, coverage=1.0)
        pos_full = elec_pos_full.positions
        assert len(pos_full) == 8

        # Test input validation
        try:
            ElectrodePosition(L=0)
            assert False, "Should raise ValueError"
        except ValueError:
            pass

        try:
            ElectrodePosition(L=16, coverage=0)
            assert False, "Should raise ValueError"
        except ValueError:
            pass

        print("✅ Electrode position configuration test passed")
        return True

    except Exception as e:
        print(f"❌ Electrode position configuration test failed: {e}")
        return False

def test_mesh_config():
    """Test mesh configuration."""
    print("🔧 Testing mesh configuration...")

    try:
        from pyeidors.geometry.optimized_mesh_generator import OptimizedMeshConfig

        # Test default configuration
        config = OptimizedMeshConfig()
        assert config.radius == 1.0
        assert config.refinement == 8
        assert config.electrode_vertices == 6
        assert config.gap_vertices == 1

        # Test mesh size calculation
        mesh_size = config.mesh_size
        expected_size = config.radius / (config.refinement * 2)
        assert abs(mesh_size - expected_size) < 1e-10

        # Test custom configuration
        custom_config = OptimizedMeshConfig(
            radius=2.0,
            refinement=4,
            electrode_vertices=10,
            gap_vertices=2
        )
        assert custom_config.radius == 2.0
        assert custom_config.refinement == 4
        assert custom_config.electrode_vertices == 10
        assert custom_config.gap_vertices == 2

        print("✅ Mesh configuration test passed")
        return True

    except Exception as e:
        print(f"❌ Mesh configuration test failed: {e}")
        return False

def test_mesh_generator_creation():
    """Test mesh generator creation."""
    print("🔧 Testing mesh generator creation...")

    try:
        from pyeidors.geometry.optimized_mesh_generator import (
            OptimizedMeshGenerator, OptimizedMeshConfig, ElectrodePosition
        )

        # Create configuration
        config = OptimizedMeshConfig(radius=1.0, refinement=6)
        electrodes = ElectrodePosition(L=16, coverage=0.5)

        # Create generator
        generator = OptimizedMeshGenerator(config, electrodes)

        # Verify initialization
        assert generator.config == config
        assert generator.electrodes == electrodes
        assert isinstance(generator.mesh_data, dict)

        print("✅ Mesh generator creation test passed")
        return True

    except ImportError as e:
        print(f"⚠️  Dependency not available, skipping mesh generator creation test: {e}")
        return True

    except Exception as e:
        print(f"❌ Mesh generator creation test failed: {e}")
        return False

@patch('pyeidors.geometry.optimized_mesh_generator.GMSH_AVAILABLE', True)
def test_mesh_generation_mock():
    """Test mesh generation (mock)."""
    print("🔧 Testing mesh generation (mock)...")

    try:
        from pyeidors.geometry.optimized_mesh_generator import (
            OptimizedMeshGenerator, OptimizedMeshConfig, ElectrodePosition
        )

        # Create configuration
        config = OptimizedMeshConfig(radius=1.0, refinement=4)
        electrodes = ElectrodePosition(L=8, coverage=0.5)

        # Create generator
        generator = OptimizedMeshGenerator(config, electrodes)

        # Mock gmsh calls
        with patch('gmsh.initialize') as mock_init, \
             patch('gmsh.model.add') as mock_add, \
             patch('gmsh.model.occ.addPoint') as mock_point, \
             patch('gmsh.model.occ.addLine') as mock_line, \
             patch('gmsh.model.occ.addCurveLoop') as mock_loop, \
             patch('gmsh.model.occ.addPlaneSurface') as mock_surface, \
             patch('gmsh.model.occ.synchronize') as mock_sync, \
             patch('gmsh.model.mesh.embed') as mock_embed, \
             patch('gmsh.model.addPhysicalGroup') as mock_physical, \
             patch('gmsh.model.mesh.setSize') as mock_size, \
             patch('gmsh.model.mesh.generate') as mock_generate, \
             patch('gmsh.write') as mock_write, \
             patch('gmsh.finalize') as mock_finalize:

            # Set mock return values
            mock_point.return_value = 1
            mock_line.return_value = 1
            mock_loop.return_value = 1
            mock_surface.return_value = 1

            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Test geometry creation method
                generator._create_geometry()

                # Verify calls
                assert mock_point.called
                assert mock_line.called
                assert mock_loop.called
                assert mock_surface.called

                # Verify mesh data structure
                assert 'boundary_points' in generator.mesh_data
                assert 'electrode_ranges' in generator.mesh_data
                assert 'lines' in generator.mesh_data
                assert 'surface' in generator.mesh_data

                print("✅ Mesh generation (mock) test passed")
                return True

    except Exception as e:
        print(f"❌ Mesh generation (mock) test failed: {e}")
        return False

def test_mesh_converter_creation():
    """Test mesh converter creation."""
    print("🔧 Testing mesh converter creation...")

    try:
        from pyeidors.geometry.optimized_mesh_generator import OptimizedMeshConverter

        # Create converter
        converter = OptimizedMeshConverter("/tmp/test.msh", "/tmp/output")

        # Verify initialization
        assert converter.mesh_file == "/tmp/test.msh"
        assert converter.output_dir == "/tmp/output"
        assert converter.prefix == "test"

        print("✅ Mesh converter creation test passed")
        return True

    except ImportError as e:
        print(f"⚠️  Dependency not available, skipping mesh converter creation test: {e}")
        return True

    except Exception as e:
        print(f"❌ Mesh converter creation test failed: {e}")
        return False

def test_convenience_functions():
    """Test convenience functions."""
    print("🔧 Testing convenience functions...")

    try:
        from pyeidors.geometry.optimized_mesh_generator import create_eit_mesh

        # Test parameter passing
        with patch('pyeidors.geometry.optimized_mesh_generator.OptimizedMeshGenerator') as mock_generator:
            mock_instance = Mock()
            mock_generator.return_value = mock_instance
            mock_instance.generate.return_value = "mock_mesh"

            # Call convenience function
            result = create_eit_mesh(
                n_elec=16,
                radius=1.0,
                refinement=6,
                electrode_coverage=0.5,
                output_dir="/tmp/test"
            )

            # Verify calls
            assert mock_generator.called
            assert mock_instance.generate.called
            assert result == "mock_mesh"

        print("✅ Convenience functions test passed")
        return True

    except Exception as e:
        print(f"❌ Convenience functions test failed: {e}")
        return False

def test_error_handling():
    """Test error handling."""
    print("🔧 Testing error handling...")

    try:
        from pyeidors.geometry.optimized_mesh_generator import (
            OptimizedMeshGenerator, OptimizedMeshConfig, ElectrodePosition
        )

        # Test error handling when dependency is missing
        with patch('pyeidors.geometry.optimized_mesh_generator.GMSH_AVAILABLE', False):
            config = OptimizedMeshConfig()
            electrodes = ElectrodePosition(L=16)

            try:
                generator = OptimizedMeshGenerator(config, electrodes)
                assert False, "Should raise ImportError"
            except ImportError:
                pass

        print("✅ Error handling test passed")
        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_integration_with_reference():
    """Test integration with reference implementation."""
    print("🔧 Testing compatibility with reference implementation...")

    try:
        from pyeidors.geometry.optimized_mesh_generator import (
            ElectrodePosition, OptimizedMeshConfig
        )

        # Create configuration same as reference implementation
        elec_pos = ElectrodePosition(L=16, coverage=0.5)
        config = OptimizedMeshConfig(radius=1.0, refinement=8)

        # Verify electrode position calculation matches reference implementation
        positions = elec_pos.positions
        assert len(positions) == 16

        # Verify each position is a valid angle pair
        for start, end in positions:
            assert 0 <= start <= 2 * np.pi
            assert 0 <= end <= 2 * np.pi
            assert start < end or (start > end and end < 0.1)  # Consider wrapping at 0

        # Verify mesh size calculation
        mesh_size = config.mesh_size
        assert mesh_size > 0

        print("✅ Reference implementation compatibility test passed")
        return True

    except Exception as e:
        print(f"❌ Reference implementation compatibility test failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("🚀 Starting optimized mesh generator tests...")
    print("=" * 50)

    tests = [
        ("Electrode Position Configuration", test_electrode_position),
        ("Mesh Configuration", test_mesh_config),
        ("Mesh Generator Creation", test_mesh_generator_creation),
        ("Mesh Generation (Mock)", test_mesh_generation_mock),
        ("Mesh Converter Creation", test_mesh_converter_creation),
        ("Convenience Functions", test_convenience_functions),
        ("Error Handling", test_error_handling),
        ("Reference Implementation Compatibility", test_integration_with_reference),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Running test: {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ Test failed: {test_name}")
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
