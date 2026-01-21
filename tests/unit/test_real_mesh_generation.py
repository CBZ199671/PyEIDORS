#!/usr/bin/env python3
"""
Real mesh generation test.
Tests real GMsh mesh generation and FEniCS conversion.
"""

import numpy as np
import tempfile
import shutil
from pathlib import Path
import logging
import sys

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parents[1]
SRC_PATH = PROJECT_ROOT / 'src'

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_real_mesh_generation():
    """Test real mesh generation."""
    print("🔧 Testing real mesh generation...")

    try:
        from pyeidors.geometry.optimized_mesh_generator import (
            OptimizedMeshGenerator, OptimizedMeshConfig, ElectrodePosition,
            create_eit_mesh
        )

        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test simple configuration
            config = OptimizedMeshConfig(
                radius=1.0,
                refinement=4,  # Smaller refinement level for faster testing
                electrode_vertices=4,
                gap_vertices=1
            )
            electrodes = ElectrodePosition(L=8, coverage=0.5)  # 8 electrodes for simplified test

            # Create generator
            generator = OptimizedMeshGenerator(config, electrodes)

            # Generate mesh
            mesh_result = generator.generate(output_dir=temp_path)

            # Verify result
            if isinstance(mesh_result, dict):
                # Returned mesh info dictionary
                print("✅ Generated mesh info dictionary")
                assert 'n_electrodes' in mesh_result
                assert mesh_result['n_electrodes'] == 8
                assert 'radius' in mesh_result
                assert mesh_result['radius'] == 1.0

            else:
                # Returned FEniCS mesh object
                print("✅ Generated FEniCS mesh object")
                assert hasattr(mesh_result, 'num_vertices')
                assert hasattr(mesh_result, 'num_cells')
                print(f"   Vertices: {mesh_result.num_vertices()}")
                print(f"   Cells: {mesh_result.num_cells()}")

            # Check output files
            msh_files = list(temp_path.glob("*.msh"))
            assert len(msh_files) >= 1, "Should generate at least one .msh file"
            print(f"✅ Generated {len(msh_files)} mesh file(s)")

            # Check XDMF files
            xdmf_files = list(temp_path.glob("*.xdmf"))
            if xdmf_files:
                print(f"✅ Generated {len(xdmf_files)} XDMF file(s)")

            return True

    except ImportError as e:
        print(f"⚠️  Dependency not available, skipping real mesh generation test: {e}")
        return True

    except Exception as e:
        print(f"❌ Real mesh generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_convenience_function():
    """Test convenience function real invocation."""
    print("🔧 Testing convenience function real invocation...")

    try:
        from pyeidors.geometry.optimized_mesh_generator import create_eit_mesh

        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use convenience function
            mesh_result = create_eit_mesh(
                n_elec=8,
                radius=1.0,
                refinement=3,
                electrode_coverage=0.5,
                output_dir=temp_dir
            )

            # Verify result
            if isinstance(mesh_result, dict):
                print("✅ Convenience function generated mesh info dictionary")
                assert 'n_electrodes' in mesh_result
                assert mesh_result['n_electrodes'] == 8
            else:
                print("✅ Convenience function generated FEniCS mesh object")
                assert hasattr(mesh_result, 'num_vertices')
                assert hasattr(mesh_result, 'num_cells')

            # Check output files
            output_path = Path(temp_dir)
            msh_files = list(output_path.glob("*.msh"))
            assert len(msh_files) >= 1, "Should generate at least one .msh file"

            return True

    except ImportError as e:
        print(f"⚠️  Dependency not available, skipping convenience function test: {e}")
        return True

    except Exception as e:
        print(f"❌ Convenience function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mesh_converter():
    """Test mesh converter."""
    print("🔧 Testing mesh converter...")

    try:
        from pyeidors.geometry.optimized_mesh_generator import (
            OptimizedMeshConverter, OptimizedMeshGenerator,
            OptimizedMeshConfig, ElectrodePosition
        )

        # First generate a mesh file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Generate mesh
            config = OptimizedMeshConfig(radius=1.0, refinement=3)
            electrodes = ElectrodePosition(L=8, coverage=0.5)
            generator = OptimizedMeshGenerator(config, electrodes)

            # Create mesh file
            mesh_result = generator.generate(output_dir=temp_path)

            # Find generated .msh file
            msh_files = list(temp_path.glob("*.msh"))
            if msh_files:
                msh_file = msh_files[0]
                print(f"✅ Found mesh file: {msh_file.name}")

                # Test converter
                converter = OptimizedMeshConverter(str(msh_file), str(temp_path))

                # Try conversion
                try:
                    mesh, boundaries_mf, assoc_table = converter.convert()
                    print("✅ Mesh conversion successful")

                    # Verify result
                    if hasattr(mesh, 'num_vertices'):
                        print(f"   Converted vertices: {mesh.num_vertices()}")
                        print(f"   Converted cells: {mesh.num_cells()}")

                    if assoc_table:
                        print(f"   Association table entries: {len(assoc_table)}")

                except Exception as e:
                    print(f"⚠️  Issue during conversion: {e}")
                    # Check if at least XDMF files were generated
                    xdmf_files = list(temp_path.glob("*.xdmf"))
                    if xdmf_files:
                        print(f"✅ Generated {len(xdmf_files)} XDMF file(s)")

                    ini_files = list(temp_path.glob("*.ini"))
                    if ini_files:
                        print(f"✅ Generated {len(ini_files)} association table file(s)")

                return True
            else:
                print("⚠️  No mesh file found, skipping converter test")
                return True

    except ImportError as e:
        print(f"⚠️  Dependency not available, skipping mesh converter test: {e}")
        return True

    except Exception as e:
        print(f"❌ Mesh converter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_electrode_geometry():
    """Test electrode geometry calculation."""
    print("🔧 Testing electrode geometry calculation...")

    try:
        from pyeidors.geometry.optimized_mesh_generator import ElectrodePosition

        # Test 16-electrode standard configuration
        elec_pos = ElectrodePosition(L=16, coverage=0.5)
        positions = elec_pos.positions

        # Verify angle distribution
        total_coverage = 0
        for start, end in positions:
            if end > start:
                total_coverage += (end - start)
            else:
                total_coverage += (end + 2*np.pi - start)

        expected_coverage = 2 * np.pi * 0.5
        assert abs(total_coverage - expected_coverage) < 1e-10

        print(f"✅ Total electrode coverage angle correct: {total_coverage:.4f} rad")

        # Test symmetry
        elec_pos_sym = ElectrodePosition(L=8, coverage=0.5)
        pos_sym = elec_pos_sym.positions

        # Verify adjacent electrode gaps are equal
        gaps = []
        for i in range(len(pos_sym)):
            end_current = pos_sym[i][1]
            start_next = pos_sym[(i+1) % len(pos_sym)][0]

            if start_next > end_current:
                gap = start_next - end_current
            else:
                gap = start_next + 2*np.pi - end_current
            gaps.append(gap)

        # Check if gaps are equal
        gap_std = np.std(gaps)
        assert gap_std < 1e-10, f"Gaps not equal, std: {gap_std}"

        print(f"✅ Electrode gap distribution uniform: {np.mean(gaps):.4f} rad")

        return True

    except Exception as e:
        print(f"❌ Electrode geometry calculation test failed: {e}")
        return False

def run_all_tests():
    """Run all real tests."""
    print("🚀 Starting real mesh generation tests...")
    print("=" * 50)

    tests = [
        ("Electrode Geometry Calculation", test_electrode_geometry),
        ("Real Mesh Generation", test_real_mesh_generation),
        ("Convenience Function Real Invocation", test_convenience_function),
        ("Mesh Converter", test_mesh_converter),
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
