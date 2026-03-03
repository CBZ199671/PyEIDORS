"""Unit tests for optimized gmsh generator helpers."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from pyeidors.geometry.optimized_mesh_generator import (
    ElectrodePosition,
    OptimizedMeshConfig,
    _build_cache_name,
)


def test_electrode_position_validation_and_order():
    pos = ElectrodePosition(L=16, coverage=0.5)
    centers = [0.5 * (s + e) for s, e in pos.positions]
    assert len(centers) == 16
    assert np.isclose(centers[0], np.pi / 2, atol=1e-12)

    with pytest.raises(ValueError):
        ElectrodePosition(L=0)
    with pytest.raises(ValueError):
        ElectrodePosition(L=8, coverage=0.0)


def test_mesh_config_size_formula():
    cfg = OptimizedMeshConfig(radius=2.0, refinement=5)
    assert np.isclose(cfg.mesh_size, 2.0 / 10.0)


def test_cache_name_encoding():
    name = _build_cache_name(n_elec=16, radius=1.25, refinement=6, electrode_coverage=0.5)
    assert name.startswith("mesh_16e_")
    assert "_ref6_" in name


def test_converter_handles_existing_msh(gmsh_mesh_artifacts):
    code = f"""
from pyeidors.geometry.optimized_mesh_generator import OptimizedMeshConverter
converter = OptimizedMeshConverter(
    mesh_file={str(gmsh_mesh_artifacts["msh_file"])!r},
    output_dir={str(gmsh_mesh_artifacts["mesh_dir"])!r},
)
mesh, facet_tags, association = converter.convert()
assert mesh.num_cells() > 0
assert facet_tags is not None
assert "domain" in association
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1"},
    )
    assert proc.returncode == 0, proc.stderr
