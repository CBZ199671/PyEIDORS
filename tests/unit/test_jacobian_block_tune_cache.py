"""Tests for Jacobian block-size auto-tuning cache behavior."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pyeidors.cache import CacheManager, CachePolicy
import pyeidors.inverse.jacobian.direct_jacobian as direct_jacobian_module
from pyeidors.inverse.jacobian.direct_jacobian import DirectJacobianCalculator


def _make_calc(cache_manager, *, candidates=(64, 128, 256)):
    calc = DirectJacobianCalculator.__new__(DirectJacobianCalculator)
    calc.gdim = 3
    calc.block_tune_mode = "auto"
    calc.block_size = 0
    calc.block_candidates = tuple(int(v) for v in candidates)
    calc._resolved_block_size = None
    calc._block_tune_source = "unset"
    calc.cell_areas = np.ones(512, dtype=float)
    calc.fwd_model = SimpleNamespace(
        pattern_manager=SimpleNamespace(n_meas_per_stim=[4]),
        cache_manager=cache_manager,
    )
    return calc


def test_block_tune_cache_hits_and_invalidates_on_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(
        direct_jacobian_module,
        "model_signature_from_forward_model",
        lambda _fm: "model-sig",
    )
    monkeypatch.setattr(
        direct_jacobian_module,
        "pattern_signature_from_forward_model",
        lambda _fm: "pattern-sig",
    )
    monkeypatch.setattr(
        direct_jacobian_module,
        "backend_signature_from_forward_model",
        lambda _fm: "backend-sig",
    )

    grad_u_all = [np.random.default_rng(0).standard_normal((512, 3))]
    adjoint = [
        np.random.default_rng(1 + idx).standard_normal((512, 3))
        for idx in range(4)
    ]

    cache_policy = CachePolicy(process_max_bytes=8 * 1024 * 1024, disk_max_bytes=64 * 1024 * 1024)
    manager_a = CacheManager(scope="both", cache_dir=tmp_path / "cache", policy=cache_policy)
    calc_a = _make_calc(manager_a, candidates=(64, 128, 256))
    block_a = calc_a._resolve_block_size(grad_u_all, adjoint, 512)
    info_a = calc_a.block_tuning_info()
    assert block_a in {64, 128, 256}
    assert info_a["tune_source"] == "compute"
    assert "assembly_elapsed_only" in info_a

    manager_b = CacheManager(scope="both", cache_dir=tmp_path / "cache", policy=cache_policy)
    calc_b = _make_calc(manager_b, candidates=(64, 128, 256))
    block_b = calc_b._resolve_block_size(grad_u_all, adjoint, 512)
    info_b = calc_b.block_tuning_info()
    assert block_b == block_a
    assert info_b["tune_source"] in {"disk", "process"}
    assert "assembly_elapsed_only" in info_b

    calc_c = _make_calc(manager_b, candidates=(32, 64))
    _ = calc_c._resolve_block_size(grad_u_all, adjoint, 512)
    info_c = calc_c.block_tuning_info()
    assert info_c["tune_source"] == "compute"
