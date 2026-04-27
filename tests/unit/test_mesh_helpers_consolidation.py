"""T78 phase 1 entrance gate: cache-name + converter consolidation.

Freezes the byte-stable cache filename helpers and the
``MeshConverter`` / ``OptimizedMeshConverter`` shared body that T78
phase 1 consolidated. Cache filenames are persisted to disk and act as
keys, so any drift breaks every cached HDF5/XDMF artifact downstream.
The converter parity test guards against the two converter classes
silently diverging again after the subclass extraction.

Phase 2 of T78 (cylinder generator base extraction + structured
sidecar consolidation) is intentionally out of scope for this gate.
"""

from __future__ import annotations

import inspect

import pytest

from pyeidors.electrodes.layout import (
    ELECTRODE_LAYOUT_RING_MAJOR,
    ELECTRODE_LAYOUT_ZIGZAG,
)
from pyeidors.geometry import _helpers as helpers
from pyeidors.geometry import optimized_mesh_generator as opt_gen
from pyeidors.geometry.mesh_converter import MeshConverter
from pyeidors.geometry.optimized_mesh_generator import OptimizedMeshConverter


# ---------------------------------------------------------------------------
# format_float_compact: byte-stable formatting used inside cache names.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (1.0, "1"),
        (1.25, "1p25"),
        (0.5, "0p5"),
        (1.234567, "1p234567"),
        (1.2345678, "1p234568"),  # rounded at 6dp
        (10.0, "10"),
        (0.0, "0"),
        (1.50000000, "1p5"),
    ],
)
def test_format_float_compact_known_values(value: float, expected: str) -> None:
    assert helpers.format_float_compact(value) == expected


def test_format_float_compact_legacy_alias_matches() -> None:
    """Module-level ``_format_float`` alias still resolves to the canonical helper."""
    assert opt_gen._format_float is helpers.format_float_compact


# ---------------------------------------------------------------------------
# build_mesh_cache_name: 2D cache filename.
# ---------------------------------------------------------------------------


def test_build_mesh_cache_name_byte_stable_for_known_inputs() -> None:
    name = helpers.build_mesh_cache_name(
        n_elec=16, radius=1.25, refinement=6, electrode_coverage=0.5
    )
    assert name == "mesh_16e_r1p25_ref6_cov0p5"


def test_build_mesh_cache_name_legacy_alias_matches_canonical() -> None:
    assert opt_gen._build_cache_name is helpers.build_mesh_cache_name
    legacy = opt_gen._build_cache_name(
        n_elec=32, radius=1.0, refinement=8, electrode_coverage=0.4
    )
    canonical = helpers.build_mesh_cache_name(
        n_elec=32, radius=1.0, refinement=8, electrode_coverage=0.4
    )
    assert legacy == canonical == "mesh_32e_r1_ref8_cov0p4"


# ---------------------------------------------------------------------------
# build_mesh_cache_name_3d: 3D cache filename includes mesh-family / version.
# ---------------------------------------------------------------------------


def test_build_mesh_cache_name_3d_byte_stable_for_known_inputs() -> None:
    name = helpers.build_mesh_cache_name_3d(
        n_elec=48,
        radius=0.25,
        height=0.2,
        refinement=4,
        electrode_coverage=0.5,
        electrode_height_ratio=0.6,
        electrode_level_fractions=(0.3, 0.5, 0.7),
        z_center=0.1,
        mesh_family="tetra",
        geometry_version="geomv2",
        generator_revision="g3d4",
    )
    assert name == (
        "mesh3d_"
        "48e_r0p25_h0p2_ref4_cov0p5_ehr0p6_lev0p3-0p5-0p7_zc0p1_"
        "elring_major_cftetra_geomv2_g3d4"
    )


def test_build_mesh_cache_name_3d_layout_normalization_round_trips() -> None:
    """``electrode_layout`` argument flows through ``normalize_electrode_layout``."""
    ring_major = helpers.build_mesh_cache_name_3d(
        n_elec=16,
        radius=1.0,
        height=1.0,
        refinement=2,
        electrode_coverage=0.5,
        electrode_height_ratio=0.5,
        electrode_level_fractions=(0.5,),
        z_center=0.0,
        mesh_family="tetra",
        geometry_version="legacy",
        generator_revision="g3d1",
        electrode_layout=ELECTRODE_LAYOUT_RING_MAJOR,
    )
    zigzag = helpers.build_mesh_cache_name_3d(
        n_elec=16,
        radius=1.0,
        height=1.0,
        refinement=2,
        electrode_coverage=0.5,
        electrode_height_ratio=0.5,
        electrode_level_fractions=(0.5,),
        z_center=0.0,
        mesh_family="tetra",
        geometry_version="legacy",
        generator_revision="g3d1",
        electrode_layout=ELECTRODE_LAYOUT_ZIGZAG,
    )
    assert "ring_major" in ring_major
    assert "zigzag" in zigzag
    assert ring_major != zigzag


def test_build_mesh_cache_name_3d_legacy_alias_matches_canonical() -> None:
    assert opt_gen._build_cache_name_3d is helpers.build_mesh_cache_name_3d


# ---------------------------------------------------------------------------
# Converter parity: OptimizedMeshConverter is now a thin MeshConverter subclass.
# ---------------------------------------------------------------------------


def test_optimized_mesh_converter_is_meshconverter_subclass() -> None:
    assert issubclass(OptimizedMeshConverter, MeshConverter)


def test_optimized_mesh_converter_only_overrides_init() -> None:
    """The two-class fusion stays honest: only ``__init__`` may diverge."""
    own_methods = {
        name for name, value in vars(OptimizedMeshConverter).items() if callable(value)
    }
    assert own_methods == {"__init__"}, (
        f"OptimizedMeshConverter must only override __init__; got {own_methods!r}"
    )


def test_meshconverter_accepts_radius_provider_keyword() -> None:
    """Canonical MeshConverter exposes the ``radius_provider`` hook OptimizedMeshConverter consumes."""
    sig = inspect.signature(MeshConverter.__init__)
    assert "radius_provider" in sig.parameters
    param = sig.parameters["radius_provider"]
    assert param.kind == inspect.Parameter.KEYWORD_ONLY
    assert param.default is None


def test_optimized_mesh_converter_init_wires_estimate_radius() -> None:
    """OptimizedMeshConverter forwards ``estimate_radius`` as the radius provider."""
    from pyeidors.femx import estimate_radius

    converter = OptimizedMeshConverter.__new__(OptimizedMeshConverter)
    # __init__ tries to build paths from real strings; bypass with __new__ +
    # manual super invocation to keep the test pure (no filesystem touch).
    MeshConverter.__init__(
        converter,
        "ignored.msh",
        "ignored_dir",
        gdim=2,
        radius_provider=estimate_radius,
    )
    assert converter._radius_provider is estimate_radius
