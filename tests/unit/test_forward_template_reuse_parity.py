"""T92 — opt-in PETSc full-matrix template reuse must match the legacy AXPY path.

Default ``forward_template_reuse=False`` keeps ``DIFFERENT_NONZERO_PATTERN``
AXPY (the contract every persisted CEM artifact and §V36 RM cache signature
depends on). The new opt-in path duplicates a cached union-pattern template
and applies M + K via ``SAME_NONZERO_PATTERN`` AXPY, which lets PETSc skip
the symbolic phase. This gate locks numerical parity bit-for-bit on a small
real CEM 2D mesh + asserts the template is invalidated on session disposal.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyeidors.data.structures import PatternConfig
from pyeidors.forward.eit_forward_model import EITForwardModel
from pyeidors.forward.process_setup_cache import clear_process_forward_setup_cache
from pyeidors.geometry.optimized_mesh_generator import (
    GMSH_AVAILABLE,
    create_eit_mesh,
)


@pytest.fixture(scope="module")
def _small_cem_mesh(tmp_path_factory):
    if not GMSH_AVAILABLE:
        pytest.skip("gmsh python bindings not available")
    return create_eit_mesh(
        n_elec=8,
        radius=1.0,
        refinement=3,
        electrode_coverage=0.5,
        output_dir=str(tmp_path_factory.mktemp("t92_template_mesh")),
        mesh_name="t92_template_2d",
    )


def _build_model(mesh, *, template_reuse: bool) -> EITForwardModel:
    pattern = PatternConfig(
        n_elec=8,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="total_current",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
    )
    return EITForwardModel(
        n_elec=8,
        pattern_config=pattern,
        z=np.full(8, 1e-5, dtype=float),
        mesh=mesh,
        linear_backend="petsc",
        backend_config={
            "forward_template_reuse": template_reuse,
            "petsc_device": "cpu",
        },
    )


def _full_matrix_dense(mat) -> np.ndarray:
    indptr, indices, data = mat.getValuesCSR()
    nrows, ncols = mat.getSize()
    dense = np.zeros((int(nrows), int(ncols)), dtype=np.float64)
    for r in range(int(nrows)):
        for k in range(int(indptr[r]), int(indptr[r + 1])):
            dense[r, int(indices[k])] += float(np.real_if_close(data[k]))
    return dense


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
def test_template_reuse_matches_legacy_axpy_full_matrix(_small_cem_mesh):
    clear_process_forward_setup_cache()
    legacy = _build_model(_small_cem_mesh, template_reuse=False)
    reuse = _build_model(_small_cem_mesh, template_reuse=True)

    assert legacy.backend_config.forward_template_reuse is False
    assert reuse.backend_config.forward_template_reuse is True

    from dolfinx import fem as _fem

    sigma_a = _fem.Function(legacy.V_sigma)
    sigma_a.x.array[:] = 1.0
    sigma_b = _fem.Function(reuse.V_sigma)
    sigma_b.x.array[:] = 1.0

    legacy_mat = legacy._create_full_matrix_petsc(sigma_a)
    reuse_first = reuse._create_full_matrix_petsc(sigma_b)
    legacy_dense = _full_matrix_dense(legacy_mat)
    reuse_first_dense = _full_matrix_dense(reuse_first)
    np.testing.assert_allclose(reuse_first_dense, legacy_dense, atol=0.0, rtol=0.0)

    sigma_a.x.array[:] = 1.5
    sigma_b.x.array[:] = 1.5
    legacy_mat2 = legacy._create_full_matrix_petsc(sigma_a)
    reuse_second = reuse._create_full_matrix_petsc(sigma_b)
    legacy_dense2 = _full_matrix_dense(legacy_mat2)
    reuse_second_dense = _full_matrix_dense(reuse_second)
    np.testing.assert_allclose(reuse_second_dense, legacy_dense2, atol=0.0, rtol=0.0)
    assert reuse._full_matrix_template is not None


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
def test_template_invalidated_on_session_disposal(_small_cem_mesh):
    clear_process_forward_setup_cache()
    reuse = _build_model(_small_cem_mesh, template_reuse=True)
    from dolfinx import fem as _fem

    sigma = _fem.Function(reuse.V_sigma)
    sigma.x.array[:] = 1.0
    reuse._create_full_matrix_petsc(sigma)
    assert reuse._full_matrix_template is not None

    reuse._dispose_forward_ksp_session(reuse._forward_ksp_session)
    assert reuse._full_matrix_template is None
    assert reuse._full_matrix_template_fingerprint is None


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
def test_default_is_off_and_legacy_path_unchanged(_small_cem_mesh):
    clear_process_forward_setup_cache()
    default = _build_model(_small_cem_mesh, template_reuse=False)
    assert default.backend_config.forward_template_reuse is False
    from dolfinx import fem as _fem

    sigma = _fem.Function(default.V_sigma)
    sigma.x.array[:] = 1.0
    default._create_full_matrix_petsc(sigma)
    assert default._full_matrix_template is None
