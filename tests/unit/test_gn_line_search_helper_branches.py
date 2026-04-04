"""Additional branch coverage for Gauss-Newton line-search helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import pyeidors.inverse.solvers.gauss_newton_line_search as line_search_module
from pyeidors.inverse.solvers.gauss_newton_line_search import (
    calc_perturb_limits,
    line_search_torch,
    update_perturb_eidors_style,
)


def _reconstructor(measurements, *, perturb=None):
    values = [np.asarray(v, dtype=float) for v in measurements]
    calls = {"count": 0}

    def _fwd_solve(_img):
        idx = min(calls["count"], len(values) - 1)
        calls["count"] += 1
        return SimpleNamespace(meas=values[idx]), None

    reconstructor = SimpleNamespace(
        max_step=1.0,
        _line_search_perturb=np.array([0.0, 0.5, 1.0], dtype=float) if perturb is None else np.asarray(perturb, dtype=float),
        clip_values=None,
        fwd_model=SimpleNamespace(fwd_solve=_fwd_solve),
        device="cpu",
        _torch_dtype=torch.float64,
        _measurement_space_type="real",
        _difference_reference_meas=None,
        _difference_mode_effective="raw",
        difference_mode="raw",
        _difference_orientation_effective="target_minus_reference",
        difference_orientation="target_minus_reference",
        use_prior_term=False,
        R_torch=torch.eye(1, dtype=torch.float64),
        convergence_tol=1e-3,
    )
    reconstructor._calls = calls
    return reconstructor


def test_calc_perturb_limits_and_update_heuristics_cover_remaining_branches(monkeypatch: pytest.MonkeyPatch):
    reconstructor = _reconstructor([np.array([0.0], dtype=float)], perturb=[0.2, 2.0])
    perturb = calc_perturb_limits(
        reconstructor,
        x=np.array([1.0, -1.0], dtype=float),
        dx=np.array([0.5, -0.25], dtype=float),
    )
    assert perturb[0] == 0.0
    assert perturb[-1] <= 1.0 + 1e-12

    monkeypatch.setattr(np.random, "randn", lambda n: np.zeros(int(n), dtype=float))

    reconstructor._line_search_perturb = np.array([0.0, 0.1, 0.2], dtype=float)
    update_perturb_eidors_style(
        reconstructor,
        chosen_step=0.0,
        perturb=np.array([0.0, 0.1, 0.2], dtype=float),
        mlist=np.array([1.0, 1.2, 1.3], dtype=float),
        valid_idx=np.array([0, 1, 2], dtype=int),
    )
    np.testing.assert_allclose(reconstructor._line_search_perturb, np.array([0.0, 0.01, 0.02], dtype=float))

    reconstructor._line_search_perturb = np.array([0.0, 0.1, 0.2], dtype=float)
    update_perturb_eidors_style(
        reconstructor,
        chosen_step=0.0,
        perturb=np.array([0.0, 0.1, 0.2], dtype=float),
        mlist=np.array([1.0, 1.01, 1.02], dtype=float),
        valid_idx=np.array([0, 1, 2], dtype=int),
    )
    np.testing.assert_allclose(reconstructor._line_search_perturb, np.array([0.0, 0.5, 1.0], dtype=float))

    reconstructor._line_search_perturb = np.array([0.0, 0.02, 0.05], dtype=float)
    update_perturb_eidors_style(
        reconstructor,
        chosen_step=0.02,
        perturb=np.array([0.0, 0.02, 0.05], dtype=float),
        mlist=np.array([1.0, 0.999, 0.998], dtype=float),
        valid_idx=np.array([0, 1, 2], dtype=int),
    )
    np.testing.assert_allclose(reconstructor._line_search_perturb, np.array([0.0, 0.2, 0.5], dtype=float))

    negative = _reconstructor([np.array([0.0], dtype=float)], perturb=[-1.0])
    neg_perturb = calc_perturb_limits(
        negative,
        x=np.array([1.0], dtype=float),
        dx=np.array([1.0], dtype=float),
    )
    assert neg_perturb[0] == 0.0
    assert neg_perturb.shape == (4,)

    scaled = _reconstructor([np.array([0.0], dtype=float)], perturb=[0.0, 1e-5, 1e-2])
    scaled_perturb = calc_perturb_limits(
        scaled,
        x=np.array([4.5e6], dtype=float),
        dx=np.array([1e-8], dtype=float),
    )
    assert scaled_perturb[0] == 0.0
    assert np.all(np.diff(scaled_perturb[1:]) > 0.0)
    assert scaled_perturb[-1] <= 1.0 + 1e-12


def test_line_search_torch_handles_inf_objective_break_and_no_valid_idx(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(line_search_module, "function_get_array", lambda sigma_current: np.asarray(sigma_current.array, dtype=float))
    monkeypatch.setattr(line_search_module, "project_measurement_vector", lambda values, **kwargs: np.asarray(values, dtype=float))

    sigma_current = SimpleNamespace(array=np.array([1.0], dtype=float))
    delta = torch.tensor([0.25], dtype=torch.float64)
    target = torch.tensor([0.0], dtype=torch.float64)

    huge = _reconstructor([np.array([np.inf], dtype=float)])
    step_huge = line_search_torch(
        huge,
        sigma_current,
        delta,
        target,
        current_weighted_residual=1.0,
        retry=5,
    )
    assert step_huge == 0.0
    assert huge._calls["count"] == 1

    empty = _reconstructor([np.array([0.0], dtype=float)], perturb=[0.0])
    step_empty = line_search_torch(
        empty,
        sigma_current,
        delta,
        target,
        current_weighted_residual=np.inf,
        retry=5,
    )
    assert step_empty == 0.0
    assert empty._calls["count"] == 0
