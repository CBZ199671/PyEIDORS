"""T91 — gmsh + meshio import is deferred until cache miss.

V72 carves out subprocess use exactly for "behavior under test is process
isolation or import-failure handling": cold-start lazy-import is the canonical
isolation case here. Run a single subprocess per generator module to keep
total cost under the unit-suite budget.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pyeidors.geometry.mesh_converter as _converter
import pyeidors.geometry.mesh3d_generator as _mesh3d
import pyeidors.geometry.mesh_generator as _mesh2d
import pyeidors.geometry.optimized_mesh_generator as _opt


def test_mesh_modules_expose_lazy_gmsh_loaders() -> None:
    for module in (_converter, _mesh2d, _mesh3d, _opt):
        assert hasattr(module, "_ensure_gmsh")
        assert callable(module._ensure_gmsh)
        assert isinstance(module.GMSH_AVAILABLE, bool)


def test_mesh3d_generator_exposes_lazy_gmsh_and_meshio_loader() -> None:
    assert hasattr(_mesh3d, "_ensure_meshio")
    assert isinstance(_mesh3d.MESHIO_AVAILABLE, bool)


def _cold_start_probe(module: str) -> dict[str, object]:
    script = textwrap.dedent(
        f"""
        import importlib
        import json
        import sys
        module = importlib.import_module({module!r})
        print(json.dumps({{
            "modules": sorted(sys.modules),
            "gmsh_bound": getattr(module, "gmsh", None) is not None,
            "gmshio_bound": getattr(module, "gmshio", None) is not None,
            "meshio_bound": getattr(module, "meshio", None) is not None,
        }}))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _assert_gmsh_stack_not_bound(module_name: str, probe: dict[str, object]) -> None:
    loaded = set(probe["modules"])
    assert "gmsh" not in loaded, (
        f"{module_name} import eagerly loaded the system gmsh library; "
        "defer via _ensure_gmsh()."
    )
    assert probe["gmshio_bound"] is False, (
        f"{module_name} import eagerly bound dolfinx.io.gmsh to module.gmshio; "
        "defer via _ensure_gmsh()."
    )
    assert probe["gmsh_bound"] is False, (
        f"{module_name} import eagerly bound gmsh to module.gmsh; "
        "defer via _ensure_gmsh()."
    )
    assert "mpi4py" not in loaded, (
        f"{module_name} import eagerly loaded mpi4py; defer until mesh IO."
    )


def test_geometry_package_import_is_lazy_light() -> None:
    probe = _cold_start_probe("pyeidors.geometry")
    loaded = set(probe["modules"])

    assert "pyeidors.geometry.mesh_generator" not in loaded
    assert "pyeidors.geometry.mesh_converter" not in loaded
    assert "pyeidors.geometry.derived_cache" not in loaded
    assert "mpi4py" not in loaded
    assert "dolfinx" not in loaded


def test_optimized_mesh_generator_import_does_not_load_gmsh_stack() -> None:
    module_name = "pyeidors.geometry.optimized_mesh_generator"
    probe = _cold_start_probe(module_name)
    _assert_gmsh_stack_not_bound(module_name, probe)


def test_mesh_converter_import_does_not_load_gmsh_stack() -> None:
    module_name = "pyeidors.geometry.mesh_converter"
    probe = _cold_start_probe(module_name)
    _assert_gmsh_stack_not_bound(module_name, probe)


def test_mesh3d_generator_import_does_not_load_gmsh_stack_or_meshio() -> None:
    module_name = "pyeidors.geometry.mesh3d_generator"
    probe = _cold_start_probe(module_name)
    _assert_gmsh_stack_not_bound(module_name, probe)
    assert "meshio" not in set(probe["modules"]), (
        "mesh3d_generator import eagerly loaded meshio; defer via _ensure_meshio()."
    )
    assert probe["meshio_bound"] is False, (
        "mesh3d_generator import eagerly bound meshio; defer via _ensure_meshio()."
    )


def test_mesh_generator_import_does_not_load_gmsh_stack() -> None:
    module_name = "pyeidors.geometry.mesh_generator"
    probe = _cold_start_probe(module_name)
    _assert_gmsh_stack_not_bound(module_name, probe)
