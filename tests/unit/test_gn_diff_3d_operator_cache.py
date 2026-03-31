"""3D GN-difference cache warm-start tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pyeidors.geometry.mesh3d_generator import GMSH_AVAILABLE

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_PATH = REPO_ROOT / "scripts"
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

from common import gn_difference_runner

OPERATOR_CACHE_KEYS = (
    "operator_jt",
    "operator_noser",
    "operator_A",
    "operator_lu",
)


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
def test_gn_difference_3d_context_cache_hits_and_background_invalidation(tmp_path):
    cache_dir = tmp_path / "cache3d"
    kwargs = dict(
        mesh_dir=str(tmp_path / "mesh_cache"),
        mesh_name=None,
        mesh_dim=3,
        mesh_height=0.08,
        electrode_height_ratio=0.2,
        z_center=0.0,
        refinement=2,
        n_elec=4,
        radius=0.1,
        drive_value=1.0,
        contact_impedance=1e-6,
        lam=0.1,
        cache_scope="both",
        cache_dir=str(cache_dir),
        cache_clear_names=[],
    )

    cold_ctx = gn_difference_runner.build_shared_context(
        background_sigma=1.0,
        **kwargs,
    )
    assert cold_ctx["cache_lookups"]["jacobian"]["hit"] is False
    for key in OPERATOR_CACHE_KEYS:
        assert cold_ctx["cache_lookups"][key]["hit"] is False

    warm_ctx = gn_difference_runner.build_shared_context(
        background_sigma=1.0,
        **kwargs,
    )
    assert warm_ctx["cache_lookups"]["jacobian"]["hit"] is True
    for key in OPERATOR_CACHE_KEYS:
        assert warm_ctx["cache_lookups"][key]["hit"] is True

    changed_ctx = gn_difference_runner.build_shared_context(
        background_sigma=1.0001,
        **kwargs,
    )
    assert changed_ctx["cache_lookups"]["jacobian"]["hit"] is False
    for key in OPERATOR_CACHE_KEYS:
        assert changed_ctx["cache_lookups"][key]["hit"] is False
