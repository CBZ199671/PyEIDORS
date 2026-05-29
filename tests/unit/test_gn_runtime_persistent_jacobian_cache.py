"""T6 — GN runtime gate for the persistent across-iteration Jacobian cache.

The cache is opt-in (``persistent_jacobian_cache=True`` on
``GaussNewtonReconstructor``). When enabled, repeated reconstructions
on the same conductivity / mesh / signatures hit the
:mod:`pyeidors.inverse.jacobian.process_jacobian_cache` LRU and skip a
fresh ``jacobian_calculator.calculate`` call. The default (``False``)
keeps every existing call site behaviourally identical.

This module covers:

- second reconstruct hits the cache → ``calculate`` count drops to 0
- mesh content hash differs → cache miss → fresh ``calculate`` call
- calculator identity differs → cache miss → fresh ``calculate`` call
- toggle off → no cache entry recorded, no behavioural change
- ``_last_persistent_jacobian_lookup`` / runtime diagnostic surface
"""

from __future__ import annotations

import inspect
import numpy as np
import pytest
from dolfinx import fem

from pyeidors.data.structures import EITImage, PatternConfig
from pyeidors.forward.eit_forward_model import EITForwardModel
from pyeidors.forward.process_setup_cache import clear_process_forward_setup_cache
from pyeidors.inverse.jacobian import (
    clear_process_jacobian_cache,
    process_jacobian_cache_stats,
)
from pyeidors.inverse.jacobian.adjoint_jacobian import EidorsJacobianAdapter
from pyeidors.inverse.regularization.smoothness import TikhonovRegularization
from pyeidors.inverse.solvers.gauss_newton import GaussNewtonReconstructor
from pyeidors.inverse.solvers import gauss_newton_runtime as gn_runtime_module

# Reuse the shared 2D unit-square fixture maker that other forward smoke
# tests already validate.
from tests.unit.test_gn_linearized_real_smoke import _make_tagged_unit_square


@pytest.fixture(autouse=True)
def _isolate_caches():
    clear_process_jacobian_cache()
    clear_process_forward_setup_cache()
    yield
    clear_process_jacobian_cache()
    clear_process_forward_setup_cache()


def _build_fwd(*, n_elec: int = 4):
    pattern = PatternConfig(
        n_elec=n_elec,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="total_current",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
    )
    return EITForwardModel(
        n_elec=n_elec,
        pattern_config=pattern,
        z=np.full(n_elec, 1e-5, dtype=np.float64),
        mesh=_make_tagged_unit_square(n_elec=n_elec),
        linear_backend="scipy",
    )


def _build_reconstructor(fwd, *, persistent: bool):
    return GaussNewtonReconstructor(
        fwd_model=fwd,
        regularization=TikhonovRegularization(fwd, alpha=1.0),
        max_iterations=1,
        min_iterations=1,
        regularization_param=0.1,
        solver_mode="fast",
        line_search_mode="fast",
        fast_linear_path="pcg",
        preconditioner="diag",
        verbose=False,
        clip_values=None,
        min_step=1.0,
        negate_jacobian=False,
        persistent_jacobian_cache=persistent,
    )


def _wrap_calculate_count(reconstructor):
    """Patch jacobian_calculator.calculate to count invocations."""
    original = reconstructor.jacobian_calculator.calculate
    counter = {"calls": 0}

    def _counted(*args, **kwargs):
        counter["calls"] += 1
        return original(*args, **kwargs)

    reconstructor.jacobian_calculator.calculate = _counted
    return counter


def test_persistent_cache_skips_recompute_on_repeat_run():
    fwd = _build_fwd()
    base = np.ones(fem.Function(fwd.V_sigma).x.array.size, dtype=np.float64)
    measured, _ = fwd.fwd_solve(EITImage(elem_data=base, fwd_model=fwd))

    reconstructor = _build_reconstructor(fwd, persistent=True)
    counter = _wrap_calculate_count(reconstructor)

    out_first = reconstructor.reconstruct(
        measured, initial_conductivity=base.copy(), jacobian_method="efficient"
    )
    assert counter["calls"] >= 1
    lookup_first = out_first.diagnostics["backend_info"][
        "persistent_jacobian_cache_lookup"
    ]
    assert lookup_first.get("hit") is False
    assert lookup_first.get("stored") is True
    assert lookup_first.get("key")
    assert process_jacobian_cache_stats()["items"] >= 1

    calls_after_first = counter["calls"]
    out_second = reconstructor.reconstruct(
        measured, initial_conductivity=base.copy(), jacobian_method="efficient"
    )
    lookup_second = out_second.diagnostics["backend_info"][
        "persistent_jacobian_cache_lookup"
    ]
    assert lookup_second.get("hit") is True
    assert counter["calls"] == calls_after_first
    source = inspect.getsource(gn_runtime_module._calculate_iteration_jacobian)
    assert "np.array(cached, copy=True)" not in source
    assert "jacobian = cached" in source


def test_persistent_cache_key_tracks_calculator_identity_after_swap():
    fwd = _build_fwd()
    base = np.ones(fem.Function(fwd.V_sigma).x.array.size, dtype=np.float64)
    measured, _ = fwd.fwd_solve(EITImage(elem_data=base, fwd_model=fwd))

    reconstructor = _build_reconstructor(fwd, persistent=True)
    direct_counter = _wrap_calculate_count(reconstructor)
    out_direct = reconstructor.reconstruct(
        measured, initial_conductivity=base.copy(), jacobian_method="efficient"
    )
    direct_lookup = out_direct.diagnostics["backend_info"][
        "persistent_jacobian_cache_lookup"
    ]
    assert direct_counter["calls"] >= 1
    assert direct_lookup.get("stored") is True
    direct_key = direct_lookup.get("key")
    assert direct_key

    reconstructor.set_jacobian_calculator(EidorsJacobianAdapter(fwd, use_torch=False))
    adapter_counter = _wrap_calculate_count(reconstructor)
    out_adapter = reconstructor.reconstruct(
        measured, initial_conductivity=base.copy(), jacobian_method="efficient"
    )
    adapter_lookup = out_adapter.diagnostics["backend_info"][
        "persistent_jacobian_cache_lookup"
    ]

    assert adapter_counter["calls"] >= 1
    assert adapter_lookup.get("hit") is False
    assert adapter_lookup.get("stored") is True
    assert adapter_lookup.get("key") != direct_key
    assert process_jacobian_cache_stats()["items"] == 2


def test_persistent_cache_off_default_no_cache_entry_or_lookup():
    fwd = _build_fwd()
    base = np.ones(fem.Function(fwd.V_sigma).x.array.size, dtype=np.float64)
    measured, _ = fwd.fwd_solve(EITImage(elem_data=base, fwd_model=fwd))

    reconstructor = _build_reconstructor(fwd, persistent=False)
    assert reconstructor.persistent_jacobian_cache is False

    counter = _wrap_calculate_count(reconstructor)
    out = reconstructor.reconstruct(
        measured, initial_conductivity=base.copy(), jacobian_method="efficient"
    )
    assert counter["calls"] >= 1
    lookup = out.diagnostics["backend_info"]["persistent_jacobian_cache_lookup"]
    assert lookup.get("key") is None
    assert lookup.get("stored") is False
    assert process_jacobian_cache_stats()["items"] == 0

    calls_first = counter["calls"]
    reconstructor.reconstruct(
        measured, initial_conductivity=base.copy(), jacobian_method="efficient"
    )
    # Cache is off → second call recomputes.
    assert counter["calls"] > calls_first


def test_persistent_cache_distinct_mesh_misses():
    fwd_a = _build_fwd(n_elec=4)
    base_a = np.ones(fem.Function(fwd_a.V_sigma).x.array.size, dtype=np.float64)
    measured_a, _ = fwd_a.fwd_solve(EITImage(elem_data=base_a, fwd_model=fwd_a))
    rec_a = _build_reconstructor(fwd_a, persistent=True)
    rec_a.reconstruct(
        measured_a, initial_conductivity=base_a.copy(), jacobian_method="efficient"
    )
    cache_items_after_a = process_jacobian_cache_stats()["items"]
    assert cache_items_after_a == 1

    # Build a different mesh (different facet tag layout / different cells).
    # n_elec = 6 forces a different `_hash_mesh_content` because facet tags
    # change shape, but actually that may overlap. Let's create a distinct
    # in-memory mesh with different refinement to force a new content hash.
    from dolfinx import mesh as dmesh
    from mpi4py import MPI
    import numpy as np_mod

    different_mesh = dmesh.create_unit_square(MPI.COMM_WORLD, 5, 5)
    fdim = different_mesh.topology.dim - 1
    boundary_facets = dmesh.locate_entities_boundary(
        different_mesh,
        fdim,
        lambda x: np_mod.full(x.shape[1], True, dtype=bool),
    ).astype(np_mod.int32)
    different_mesh.topology.create_connectivity(fdim, 0)
    f2v = different_mesh.topology.connectivity(fdim, 0)
    coords = different_mesh.geometry.x[:, :2]
    centroids = np_mod.zeros((boundary_facets.size, 2), dtype=np_mod.float64)
    for idx, facet in enumerate(boundary_facets):
        centroids[idx, :] = coords[f2v.links(int(facet))].mean(axis=0)
    x = centroids[:, 0]
    y = centroids[:, 1]
    eps = 1e-10
    t = np_mod.zeros_like(x)
    left = np_mod.isclose(x, 0.0, atol=eps)
    top = (~left) & np_mod.isclose(y, 1.0, atol=eps)
    right = (~left) & (~top) & np_mod.isclose(x, 1.0, atol=eps)
    bottom = (~left) & (~top) & (~right) & np_mod.isclose(y, 0.0, atol=eps)
    t[left] = y[left]
    t[top] = 1.0 + x[top]
    t[right] = 2.0 + (1.0 - y[right])
    t[bottom] = 3.0 + (1.0 - x[bottom])
    n_elec = 4
    tags = (
        np_mod.floor(np_mod.clip(t, 0.0, 4.0 - eps) / (4.0 / float(n_elec))).astype(
            np_mod.int32
        )
        + 2
    ).astype(np_mod.int32)
    order = np_mod.argsort(boundary_facets)
    facet_tags = dmesh.meshtags(
        different_mesh, fdim, boundary_facets[order], tags[order]
    )
    from pyeidors.femx import build_eit_mesh

    eit_mesh_b = build_eit_mesh(
        different_mesh,
        facet_tags=facet_tags,
        association_table={f"electrode_{i + 1}": i + 2 for i in range(n_elec)},
        radius=1.0,
    )
    fwd_b = EITForwardModel(
        n_elec=n_elec,
        pattern_config=PatternConfig(
            n_elec=n_elec,
            stim_pattern="{ad}",
            meas_pattern="{ad}",
            drive_mode="total_current",
            drive_value=1.0,
            geometry_scale_to_m=1.0,
        ),
        z=np.full(n_elec, 1e-5, dtype=np.float64),
        mesh=eit_mesh_b,
        linear_backend="scipy",
    )
    base_b = np.ones(fem.Function(fwd_b.V_sigma).x.array.size, dtype=np.float64)
    measured_b, _ = fwd_b.fwd_solve(EITImage(elem_data=base_b, fwd_model=fwd_b))
    rec_b = _build_reconstructor(fwd_b, persistent=True)
    out_b = rec_b.reconstruct(
        measured_b, initial_conductivity=base_b.copy(), jacobian_method="efficient"
    )
    # Different mesh content hash → cache miss + new entry
    assert process_jacobian_cache_stats()["items"] == 2
    lookup_b = out_b.diagnostics["backend_info"]["persistent_jacobian_cache_lookup"]
    assert lookup_b.get("hit") is False
    assert lookup_b.get("stored") is True


def test_persistent_cache_skipped_for_operator_jacobian_method():
    fwd = _build_fwd()
    base = np.ones(fem.Function(fwd.V_sigma).x.array.size, dtype=np.float64)
    measured, _ = fwd.fwd_solve(EITImage(elem_data=base, fwd_model=fwd))
    reconstructor = _build_reconstructor(fwd, persistent=True)

    out = reconstructor.reconstruct(
        measured,
        initial_conductivity=base.copy(),
        jacobian_method="linearized",
    )
    # Operator path never enters the cache code path → no entries stored.
    assert process_jacobian_cache_stats()["items"] == 0
    lookup = out.diagnostics["backend_info"].get("persistent_jacobian_cache_lookup", {})
    assert lookup.get("reason") == "operator_jacobian"
    assert lookup.get("stored", False) is False
    assert lookup.get("hit", False) is False


def test_persistent_cache_lookup_reset_when_operator_follows_dense_run():
    fwd = _build_fwd()
    base = np.ones(fem.Function(fwd.V_sigma).x.array.size, dtype=np.float64)
    measured, _ = fwd.fwd_solve(EITImage(elem_data=base, fwd_model=fwd))
    reconstructor = _build_reconstructor(fwd, persistent=True)

    dense = reconstructor.reconstruct(
        measured,
        initial_conductivity=base.copy(),
        jacobian_method="efficient",
    )
    dense_lookup = dense.diagnostics["backend_info"]["persistent_jacobian_cache_lookup"]
    assert dense_lookup.get("key")
    assert dense_lookup.get("stored") is True

    operator = reconstructor.reconstruct(
        measured,
        initial_conductivity=base.copy(),
        jacobian_method="linearized",
    )
    lookup = operator.diagnostics["backend_info"]["persistent_jacobian_cache_lookup"]
    assert lookup.get("reason") == "operator_jacobian"
    assert lookup.get("key") is None
    assert lookup.get("stored") is False
    assert lookup.get("hit") is False
    assert process_jacobian_cache_stats()["items"] == 1
