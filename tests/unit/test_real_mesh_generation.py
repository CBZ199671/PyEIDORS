"""Additional real gmsh-path checks."""

from __future__ import annotations

import os
import subprocess
import sys

from pyeidors.geometry.mesh_loader import MeshLoader


def test_load_or_create_mesh_prefers_cached_msh(gmsh_mesh_artifacts):
    code = f"""
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
mesh = load_or_create_mesh(
    mesh_dir={str(gmsh_mesh_artifacts["mesh_dir"])!r},
    mesh_name={gmsh_mesh_artifacts["mesh_name"]!r},
    n_elec=16,
)
assert mesh.mesh_file == {str(gmsh_mesh_artifacts["msh_file"])!r}
assert mesh.num_cells() > 0
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1"},
    )
    assert proc.returncode == 0, proc.stderr


def test_mesh_loader_default_requires_msh(tmp_path):
    empty_dir = tmp_path / "empty_meshes"
    empty_dir.mkdir(parents=True, exist_ok=True)
    loader = MeshLoader(mesh_dir=str(empty_dir))
    try:
        loader.get_default_mesh()
    except FileNotFoundError as exc:
        assert ".msh caches" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError when no .msh cache exists")


def test_mesh_listing_reports_supported_formats(gmsh_mesh_artifacts):
    loader = MeshLoader(mesh_dir=str(gmsh_mesh_artifacts["mesh_dir"]))
    listing = loader.list_available_meshes()
    assert "msh" in listing and gmsh_mesh_artifacts["mesh_name"] in listing["msh"]
    assert "xdmf" in listing
    assert "numpy" in listing
