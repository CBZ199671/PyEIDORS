#!/usr/bin/env python3
"""
PyEIDORS Basic Usage Example
Demonstrates how to use the modular EIT system for forward solving and inverse reconstruction.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def basic_usage_example():
    """Basic usage example."""
    print("=== PyEIDORS Basic Usage Example ===")

    # Import required modules
    from pyeidors import EITSystem, check_environment
    from pyeidors.data.structures import PatternConfig, MeshConfig

    # Check environment
    print("1. Environment Check")
    env = check_environment()
    print(f"   FEniCS available: {env['fenics_available']}")
    print(f"   PyTorch available: {env['torch_available']}")
    print(f"   CUDA available: {env['cuda_available']}")
    if env['torch_available']:
        print(f"   PyTorch version: {env['torch_version']}")
        print(f"   GPU count: {env['cuda_device_count']}")
    print()

    # Configure EIT system
    print("2. Configure EIT System")
    n_elec = 16  # 16 electrodes

    # Stimulation/measurement pattern configuration
    pattern_config = PatternConfig(
        n_elec=n_elec,
        stim_pattern='{ad}',  # Adjacent stimulation pattern
        meas_pattern='{ad}',  # Adjacent measurement pattern
        amplitude=1.0         # Stimulation current amplitude
    )

    # Mesh configuration
    mesh_config = MeshConfig(
        radius=1.0,          # Circular domain radius
        refinement=8,        # Mesh refinement level
        mesh_size=0.1       # Mesh element size
    )

    # Create EIT system
    eit_system = EITSystem(
        n_elec=n_elec,
        pattern_config=pattern_config,
        mesh_config=mesh_config
    )

    print(f"   Number of electrodes: {n_elec}")
    print(f"   Stimulation pattern: {pattern_config.stim_pattern}")
    print(f"   Measurement pattern: {pattern_config.meas_pattern}")
    print()

    # Get system information
    print("3. System Information")
    info = eit_system.get_system_info()
    for key, value in info.items():
        if key != 'pattern_config' and key != 'mesh_config':
            print(f"   {key}: {value}")
    print()

    # Notes for actual usage
    print("4. Notes")
    print("   - Current version requires externally provided mesh object")
    print("   - Can use existing mesh files or custom mesh generator")
    print("   - Example mesh files located in eit_meshes/ directory")
    print()

def show_module_structure():
    """Show module structure."""
    print("=== PyEIDORS Module Structure ===")

    structure = {
        "pyeidors/": {
            "__init__.py": "Main module entry, environment check",
            "core_system.py": "Core EIT system class",
            "data/": {
                "structures.py": "Data structure definitions (EITData, EITImage, config classes)",
                "synthetic_data.py": "Synthetic data generation"
            },
            "forward/": {
                "eit_forward_model.py": "EIT forward model (Complete Electrode Model)"
            },
            "inverse/": {
                "jacobian/": {
                    "base_jacobian.py": "Jacobian calculator base class",
                    "direct_jacobian.py": "Direct method Jacobian calculator"
                },
                "regularization/": {
                    "base_regularization.py": "Regularization base class",
                    "smoothness.py": "Smoothness regularization"
                },
                "solvers/": {
                    "gauss_newton.py": "Modular Gauss-Newton solver"
                }
            },
            "electrodes/": {
                "patterns.py": "Stimulation/measurement pattern manager"
            },
            "geometry/": {
                "mesh_generator.py": "Mesh generator",
                "mesh_converter.py": "Mesh format converter"
            },
            "utils/": "Utility functions",
            "visualization/": "Visualization module"
        }
    }

    def print_structure(struct, indent=0):
        for key, value in struct.items():
            print("  " * indent + f"├── {key}")
            if isinstance(value, dict):
                print_structure(value, indent + 1)
            else:
                print("  " * (indent + 1) + f"    {value}")

    print_structure(structure)
    print()

def show_key_features():
    """Show key features."""
    print("=== PyEIDORS Key Features ===")

    features = [
        "🔧 Modular Design",
        "   - Independent forward model, inverse solver, regularization modules",
        "   - Pluggable Jacobian calculators and regularization strategies",
        "   - Clear data structure definitions",
        "",
        "⚡ Performance Optimization",
        "   - PyTorch GPU acceleration support",
        "   - Efficient Jacobian matrix computation",
        "   - Sparse matrix operation optimization",
        "",
        "🧮 Numerical Methods",
        "   - Complete Electrode Model (CEM)",
        "   - Gauss-Newton iterative solver",
        "   - Multiple regularization strategies (Tikhonov, smoothness, total variation)",
        "",
        "🔬 Scientific Computing",
        "   - Based on FEniCS finite element framework",
        "   - Support for custom meshes and boundary conditions",
        "   - Compatible with standard EIT data formats",
        "",
        "📊 Extensibility",
        "   - Support for multiple stimulation/measurement patterns",
        "   - Integrable with CUQI Bayesian inference framework",
        "   - Flexible visualization interface"
    ]

    for feature in features:
        print(feature)
    print()

if __name__ == "__main__":
    basic_usage_example()
    show_module_structure()
    show_key_features()

    print("=== Next Steps ===")
    print("1. Provide mesh object to complete system initialization")
    print("2. Run forward solve to verify model correctness")
    print("3. Test inverse reconstruction algorithms")
    print("4. Add visualization and data saving functionality")
    print("5. Integrate more regularization and solving strategies")
