"""T91 — gmsh + meshio import is deferred until cache miss.

V72 carves out subprocess use exactly for "behavior under test is process
isolation or import-failure handling": cold-start lazy-import is the canonical
isolation case here. Run a single subprocess per generator module to keep
total cost under the unit-suite budget.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pyeidors.geometry.mesh3d_generator as _mesh3d
import pyeidors.geometry.optimized_mesh_generator as _opt


def test_optimized_mesh_generator_exposes_lazy_gmsh_loader() -> None:
    assert hasattr(_opt, "_ensure_gmsh")
    assert callable(_opt._ensure_gmsh)
    assert isinstance(_opt.GMSH_AVAILABLE, bool)


def test_mesh3d_generator_exposes_lazy_gmsh_and_meshio_loader() -> None:
    assert hasattr(_mesh3d, "_ensure_gmsh")
    assert hasattr(_mesh3d, "_ensure_meshio")
    assert isinstance(_mesh3d.GMSH_AVAILABLE, bool)
    assert isinstance(_mesh3d.MESHIO_AVAILABLE, bool)


def _cold_start_loaded_modules(module: str) -> set[str]:
    script = textwrap.dedent(
        f"""
        import importlib
        import sys
        importlib.import_module({module!r})
        print("\\n".join(sorted(sys.modules)))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(result.stdout.splitlines())


def test_optimized_mesh_generator_import_does_not_load_system_gmsh() -> None:
    loaded = _cold_start_loaded_modules("pyeidors.geometry.optimized_mesh_generator")
    assert "gmsh" not in loaded, (
        "optimized_mesh_generator import eagerly loaded the system gmsh "
        "library; ensure all gmsh imports are deferred via _ensure_gmsh()."
    )


def test_mesh3d_generator_import_does_not_load_system_gmsh_or_meshio() -> None:
    loaded = _cold_start_loaded_modules("pyeidors.geometry.mesh3d_generator")
    assert "gmsh" not in loaded, (
        "mesh3d_generator import eagerly loaded the system gmsh library; "
        "defer via _ensure_gmsh()."
    )
    assert "meshio" not in loaded, (
        "mesh3d_generator import eagerly loaded meshio; defer via _ensure_meshio()."
    )


def test_mesh_generator_import_does_not_load_system_gmsh() -> None:
    loaded = _cold_start_loaded_modules("pyeidors.geometry.mesh_generator")
    assert "gmsh" not in loaded, (
        "legacy mesh_generator import eagerly loaded the system gmsh "
        "library; defer via _ensure_gmsh()."
    )
