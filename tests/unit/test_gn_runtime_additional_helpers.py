"""Additional branch coverage for Gauss-Newton runtime helper functions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import pyeidors.inverse.solvers.gauss_newton_runtime as gn_runtime


def _recon(**overrides):
    recon = SimpleNamespace(
        device="cpu",
        _torch_dtype=torch.float64,
        n_elements=3,
        clip_values=None,
        best_homog_mode="off",
        _measurement_space_type="real",
        difference_mode="normalized",
        difference_orientation="target_minus_reference",
        difference_step_size_mode="off",
        difference_step_size_value=None,
        difference_step_size_bounds=(0.0, 4.0),
        difference_step_size_fmin_options={},
        active_preset_name="eidors_one_step_noser",
        use_prior_term=True,
        R_torch=torch.eye(2, dtype=torch.float64),
        regularization_param=0.1,
        solver_mode="strict",
        line_search_mode="full",
        max_step=1.0,
        min_step=None,
        step_schedule=None,
        _meas_weight_sqrt=None,
        verbose=False,
        fwd_model=SimpleNamespace(
            fwd_solve=lambda _img: (SimpleNamespace(meas=np.array([1.0, 1.0], dtype=float)), None),
        ),
    )
    for key, value in overrides.items():
        setattr(recon, key, value)
    return recon


def test_best_homog_bounds_and_estimate_exception(monkeypatch: pytest.MonkeyPatch):
    recon_scalar = _recon(clip_values=None)
    assert gn_runtime._best_homog_bounds(recon_scalar, 2.0) == pytest.approx((0.4, 10.0))

    recon_clipped = _recon(clip_values=(0.5, 1.5))
    assert gn_runtime._best_homog_bounds(recon_clipped, np.array([2.0, 3.0], dtype=float)) == (0.5, 1.5)

    monkeypatch.setattr(
        gn_runtime,
        "minimize_scalar",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("opt failed")),
    )
    info = gn_runtime._estimate_best_homogeneous_conductivity(
        _recon(best_homog_mode="optimize"),
        measured_vector=np.array([1.0, 2.0], dtype=float),
        initial_conductivity=1.0,
    )
    assert info["applied"] is False
    assert info["reason"] == "optimization_failed:RuntimeError"

    diff_info = gn_runtime._estimate_best_homogeneous_conductivity(
        _recon(best_homog_mode="optimize", _measurement_space_type="difference"),
        measured_vector=np.array([1.0, 2.0], dtype=float),
        initial_conductivity=1.0,
    )
    assert diff_info["reason"] == "difference_measurement_space"


def test_prepare_prior_and_difference_step_size_reason_paths(monkeypatch: pytest.MonkeyPatch):
    recon = _recon()
    prior_torch = gn_runtime._prepare_prior(
        recon,
        prior_data=np.array([0.2, 0.3, 0.4], dtype=float),
        initial_conductivity=1.0,
    )
    np.testing.assert_allclose(recon._prior_data, np.array([0.2, 0.3, 0.4], dtype=float))
    assert prior_torch.shape == (3,)

    sigma_final = np.array([1.0, 1.1, 1.2], dtype=float)
    measured = np.array([0.1, 0.2], dtype=float)

    _, preset_info = gn_runtime._apply_difference_step_size(
        _recon(
            _measurement_space_type="difference",
            difference_step_size_mode="optimize",
            active_preset_name="sphere_multistep_noser",
        ),
        sigma_final=sigma_final,
        measured_vector=measured,
    )
    assert preset_info["reason"] == "preset_not_one_step"

    _, missing_prior_info = gn_runtime._apply_difference_step_size(
        _recon(_measurement_space_type="difference", difference_step_size_mode="optimize"),
        sigma_final=sigma_final,
        measured_vector=measured,
    )
    assert missing_prior_info["reason"] == "missing_prior"

    _, shape_info = gn_runtime._apply_difference_step_size(
        _recon(
            _measurement_space_type="difference",
            difference_step_size_mode="optimize",
            _prior_data=np.array([1.0, 1.1], dtype=float),
        ),
        sigma_final=sigma_final,
        measured_vector=measured,
    )
    assert shape_info["reason"] == "missing_prior"

    _, zero_info = gn_runtime._apply_difference_step_size(
        _recon(
            _measurement_space_type="difference",
            difference_step_size_mode="optimize",
            _prior_data=sigma_final.copy(),
        ),
        sigma_final=sigma_final,
        measured_vector=measured,
    )
    assert zero_info["reason"] == "zero_delta"

    monkeypatch.setattr(
        gn_runtime,
        "minimize_scalar",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad optimize")),
    )
    _, exc_info = gn_runtime._apply_difference_step_size(
        _recon(
            _measurement_space_type="difference",
            difference_step_size_mode="optimize",
            _prior_data=np.array([0.9, 1.0, 1.1], dtype=float),
        ),
        sigma_final=sigma_final,
        measured_vector=measured,
    )
    assert exc_info["reason"] == "optimization_failed:RuntimeError"


def test_compute_objective_build_linear_system_and_step_selection(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        gn_runtime,
        "_apply_regularization_np",
        lambda reconstructor, vec: np.array([2.0, 4.0], dtype=float),
    )
    meas_misfit, prior_misfit, total, rt_r_de = gn_runtime._compute_objective(
        _recon(R_torch=None),
        weighted_residual_torch=torch.tensor([1.0, 2.0], dtype=torch.float64),
        de_current=torch.tensor([0.5, 1.0], dtype=torch.float64),
        lambda_eff=0.25,
        iteration=0,
    )
    assert meas_misfit == pytest.approx(2.5)
    assert prior_misfit == pytest.approx(0.625)
    assert total == pytest.approx(3.125)
    np.testing.assert_allclose(rt_r_de.detach().cpu().numpy(), np.array([2.0, 4.0], dtype=float))

    recon_linear = _recon(R_torch=torch.eye(2, dtype=torch.float64), use_prior_term=True)
    A, b = gn_runtime._build_linear_system(
        recon_linear,
        JTJ=torch.eye(2, dtype=torch.float64),
        JTr=torch.tensor([0.5, -0.2], dtype=torch.float64),
        de_torch=torch.tensor([1.0, 2.0], dtype=torch.float64),
        lambda_eff=0.5,
        iteration=0,
        RtR_de=None,
    )
    np.testing.assert_allclose(A.detach().cpu().numpy(), np.array([[1.5, 0.0], [0.0, 1.5]], dtype=float))
    np.testing.assert_allclose(b.detach().cpu().numpy(), np.array([-1.0, -0.8], dtype=float))

    quick = gn_runtime._select_step_size(
        _recon(solver_mode="fast", line_search_mode="fast", max_step=0.2, min_step=0.4),
        iteration=0,
        sigma_current=object(),
        delta_sigma_torch=torch.tensor([0.1], dtype=torch.float64),
        meas_torch=torch.tensor([0.1], dtype=torch.float64),
        residual_norm_weighted=1.0,
        prior_torch=torch.tensor([0.0], dtype=torch.float64),
        lambda_eff=0.1,
    )
    assert quick == pytest.approx(0.4)

    recon_line = _recon(min_step=0.1)
    recon_line._line_search_torch = lambda *args, **kwargs: 0.05
    clamped = gn_runtime._select_step_size(
        recon_line,
        iteration=0,
        sigma_current=object(),
        delta_sigma_torch=torch.tensor([0.1], dtype=torch.float64),
        meas_torch=torch.tensor([0.1], dtype=torch.float64),
        residual_norm_weighted=1.0,
        prior_torch=torch.tensor([0.0], dtype=torch.float64),
        lambda_eff=0.1,
    )
    assert clamped == pytest.approx(0.1)

    one_step = gn_runtime._select_step_size(
        _recon(
            _measurement_space_type="difference",
            active_preset_name="eidors_demo3d_tv",
            max_iterations=1,
        ),
        iteration=0,
        sigma_current=object(),
        delta_sigma_torch=torch.tensor([0.1], dtype=torch.float64),
        meas_torch=torch.tensor([0.1], dtype=torch.float64),
        residual_norm_weighted=1.0,
        prior_torch=torch.tensor([0.0], dtype=torch.float64),
        lambda_eff=0.1,
    )
    assert one_step == pytest.approx(1.0)


def test_torch_cg_solver_and_linear_system_fallbacks(monkeypatch: pytest.MonkeyPatch):
    sol = gn_runtime._solve_linear_system_torch_cg(
        torch.diag(torch.tensor([2.0, 3.0], dtype=torch.float64)),
        torch.tensor([2.0, 3.0], dtype=torch.float64),
    )
    np.testing.assert_allclose(sol.detach().cpu().numpy(), np.array([1.0, 1.0], dtype=float), atol=1e-10)

    with pytest.raises(RuntimeError, match="did not converge"):
        gn_runtime._solve_linear_system_torch_cg(
            torch.zeros((2, 2), dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            max_iter=2,
        )

    with pytest.raises(RuntimeError, match="did not converge"):
        gn_runtime._solve_linear_system_torch_cg(
            torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64),
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
            max_iter=1,
        )

    calls = {"count": 0}

    def _solve_once_then_regularized(A, b):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("singular")
        return torch.tensor([0.5, 0.25], dtype=torch.float64)

    monkeypatch.setattr(gn_runtime.torch.linalg, "solve", _solve_once_then_regularized)
    delta, delta_norm = gn_runtime._solve_linear_system(
        _recon(R_torch=torch.eye(2, dtype=torch.float64), regularization_param=0.2),
        A=torch.eye(2, dtype=torch.float64),
        b=torch.tensor([1.0, 0.5], dtype=torch.float64),
        JTJ=torch.eye(2, dtype=torch.float64),
        iteration=0,
    )
    np.testing.assert_allclose(delta.detach().cpu().numpy(), np.array([0.5, 0.25], dtype=float))
    assert delta_norm == pytest.approx(np.linalg.norm([0.5, 0.25]))

    monkeypatch.setattr(
        gn_runtime.torch.linalg,
        "solve",
        lambda A, b: (_ for _ in ()).throw(RuntimeError("singular again")),
    )
    monkeypatch.setattr(
        gn_runtime,
        "_solve_linear_system_torch_cg",
        lambda A, b, **kwargs: torch.tensor([0.2, 0.3], dtype=torch.float64),
    )
    delta_cg, _ = gn_runtime._solve_linear_system(
        _recon(R_torch=torch.eye(2, dtype=torch.float64), regularization_param=0.2),
        A=torch.eye(2, dtype=torch.float64),
        b=torch.tensor([1.0, 0.5], dtype=torch.float64),
        JTJ=torch.eye(2, dtype=torch.float64),
        iteration=0,
    )
    np.testing.assert_allclose(delta_cg.detach().cpu().numpy(), np.array([0.2, 0.3], dtype=float))

    monkeypatch.setattr(
        gn_runtime.torch.linalg,
        "solve",
        lambda A, b: (_ for _ in ()).throw(RuntimeError("missing libtorch_cuda_linalg runtime")),
    )
    monkeypatch.setattr(
        gn_runtime,
        "_solve_linear_system_torch_cg",
        lambda A, b, **kwargs: torch.tensor([0.7, 0.8], dtype=torch.float64),
    )
    fake_cuda_A = SimpleNamespace(device=SimpleNamespace(type="cuda"), shape=(2, 2))
    delta_cuda, _ = gn_runtime._solve_linear_system(
        _recon(R_torch=torch.eye(2, dtype=torch.float64), regularization_param=0.2),
        A=fake_cuda_A,
        b=torch.tensor([1.0, 0.5], dtype=torch.float64),
        JTJ=torch.eye(2, dtype=torch.float64),
        iteration=1,
    )
    np.testing.assert_allclose(delta_cuda.detach().cpu().numpy(), np.array([0.7, 0.8], dtype=float))


def test_maybe_rollback_rolls_back_and_stops(monkeypatch: pytest.MonkeyPatch):
    restored = {}
    monkeypatch.setattr(
        gn_runtime,
        "function_set_array",
        lambda _fn, values: restored.setdefault("values", np.asarray(values, dtype=float).copy()),
    )

    rolled_back, should_stop, count = gn_runtime._maybe_rollback(
        _recon(verbose=False),
        sigma_current=object(),
        sigma_old_values=np.array([1.0, 1.1], dtype=float),
        residual_norm=2.0,
        prev_residual=1.0,
        residual_history=[1.0, 2.0],
        sigma_change_history=[0.2, 0.3],
        consecutive_rollbacks=1,
        max_consecutive_rollbacks=2,
    )
    assert rolled_back is True
    assert should_stop is True
    assert count == 2
    np.testing.assert_allclose(restored["values"], np.array([1.0, 1.1], dtype=float))


def test_maybe_rollback_verbose_messages(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(
        gn_runtime,
        "function_set_array",
        lambda *_args, **_kwargs: None,
    )

    rolled_back, should_stop, count = gn_runtime._maybe_rollback(
        _recon(verbose=True),
        sigma_current=object(),
        sigma_old_values=np.array([1.0, 1.1], dtype=float),
        residual_norm=2.0,
        prev_residual=1.0,
        residual_history=[1.0, 2.0],
        sigma_change_history=[0.2, 0.3],
        consecutive_rollbacks=0,
        max_consecutive_rollbacks=1,
    )
    assert rolled_back is True
    assert should_stop is True
    assert count == 1
    out = capsys.readouterr().out
    assert "rolling back step" in out
    assert "terminating early" in out


def test_torch_cg_nonfinite_rz_new_and_maybe_rollback_continue_paths(monkeypatch: pytest.MonkeyPatch):
    values = iter(
        [
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(float("nan"), dtype=torch.float64),
        ]
    )
    monkeypatch.setattr(gn_runtime.torch, "dot", lambda _a, _b: next(values))
    with pytest.raises(RuntimeError, match="did not converge"):
        gn_runtime._solve_linear_system_torch_cg(
            torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            max_iter=4,
        )

    monkeypatch.setattr(
        gn_runtime,
        "function_set_array",
        lambda *_args, **_kwargs: None,
    )
    rolled_back, should_stop, count = gn_runtime._maybe_rollback(
        _recon(verbose=False),
        sigma_current=object(),
        sigma_old_values=np.array([1.0, 1.1], dtype=float),
        residual_norm=2.0,
        prev_residual=1.0,
        residual_history=[1.0, 2.0],
        sigma_change_history=[0.2, 0.3],
        consecutive_rollbacks=0,
        max_consecutive_rollbacks=3,
    )
    assert rolled_back is True
    assert should_stop is False
    assert count == 1
