"""Mandatory gmsh real-path integration tests.

These tests intentionally exercise gmsh in a subprocess so that a low-level gmsh
abort on one platform does not crash the whole pytest worker.
"""

from __future__ import annotations

from pathlib import Path

from tests.utils import run_python


def test_gmsh_create_eit_mesh_subprocess(tmp_path):
    mesh_dir = tmp_path / "gmsh_subprocess"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh_name = "mesh_subproc"

    code = f"""
from pyeidors.geometry.optimized_mesh_generator import create_eit_mesh
mesh = create_eit_mesh(
    n_elec=8,
    radius=1.0,
    refinement=3,
    electrode_coverage=0.5,
    output_dir={str(mesh_dir)!r},
    mesh_name={mesh_name!r},
)
assert mesh.num_cells() > 0
assert mesh.num_vertices() > 0
print(mesh.mesh_file)
"""

    proc = run_python(code)
    assert proc.returncode == 0, (
        f"gmsh subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    msh_file = mesh_dir / f"{mesh_name}.msh"
    assoc_file = mesh_dir / f"{mesh_name}_association_table.ini"
    assert msh_file.exists()
    assert assoc_file.exists()
