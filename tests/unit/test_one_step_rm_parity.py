"""Parity gates between one-step RM helpers and legacy dense NOSER GN."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pyeidors.data.difference import (
    build_difference_vector,
    project_measurement_jacobian,
)
from pyeidors.inverse.reconstruction_matrix import build_one_step_rm
from pyeidors.inverse.regularization import smoothness as smooth_module
from pyeidors.inverse.regularization.smoothness import NOSERRegularization
from pyeidors.inverse.solvers import gauss_newton_runtime as gn_runtime


class _FakeSigmaFunction:
    def __init__(self, _space):
        self.x = SimpleNamespace(array=np.zeros(3, dtype=np.float64))


def _fake_forward_model(n_elements: int) -> SimpleNamespace:
    return SimpleNamespace(
        mesh=SimpleNamespace(),
        V_sigma=SimpleNamespace(
            dofmap=SimpleNamespace(
                index_map=SimpleNamespace(size_local=int(n_elements)),
                index_map_bs=1,
            )
        ),
    )


def _legacy_dense_noser_delta(
    projected_jacobian: np.ndarray,
    difference_vector: np.ndarray,
    *,
    hyperparameter: float,
    monkeypatch: pytest.MonkeyPatch,
) -> np.ndarray:
    """Run the old dense GN/NOSER linear-system contract on a tiny problem."""

    projected_jacobian = np.asarray(projected_jacobian, dtype=np.float64)
    difference_vector = np.asarray(difference_vector, dtype=np.float64).reshape(-1)
    n_parameters = int(projected_jacobian.shape[1])

    monkeypatch.setattr(smooth_module.fem, "Function", _FakeSigmaFunction)
    regularization = NOSERRegularization(
        _fake_forward_model(n_parameters),
        jacobian_calculator=SimpleNamespace(
            calculate=lambda _sigma: projected_jacobian.copy()
        ),
        base_conductivity=1.0,
        alpha=1.0,
        exponent=1.0,
        floor=1e-12,
        adaptive_floor=False,
    )
    regularization_matrix = regularization.get_regularization_matrix().toarray()

    reconstructor = SimpleNamespace(
        R_torch=torch.as_tensor(regularization_matrix, dtype=torch.float64),
        regularization_param=float(hyperparameter) ** 2,
        use_prior_term=True,
    )
    residual = -difference_vector
    j_torch = torch.as_tensor(projected_jacobian, dtype=torch.float64)
    residual_torch = torch.as_tensor(residual, dtype=torch.float64)
    de_current = torch.zeros(n_parameters, dtype=torch.float64)
    jtj = j_torch.T @ j_torch
    jtr = j_torch.T @ residual_torch

    system, rhs = gn_runtime._build_linear_system(
        reconstructor,
        jtj,
        jtr,
        de_current,
        reconstructor.regularization_param,
        iteration=0,
    )
    delta, _delta_norm = gn_runtime._solve_linear_system(
        reconstructor,
        system,
        rhs,
        jtj,
        iteration=0,
    )
    return delta.detach().cpu().numpy()


@pytest.mark.parametrize("difference_mode", ["raw", "normalized"])
def test_one_step_noser_rm_matches_legacy_dense_eidors_baseline(
    difference_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    absolute_jacobian = np.array(
        [
            [0.9, -0.15, 0.35],
            [0.25, 0.7, -0.2],
            [-0.4, 0.15, 0.8],
            [0.5, -0.3, 0.45],
        ],
        dtype=np.float64,
    )
    reference = np.array([2.0, -3.0, 4.0, 5.0], dtype=np.float64)
    true_delta = np.array([0.12, -0.07, 0.09], dtype=np.float64)
    target = reference + absolute_jacobian @ true_delta
    hyperparameter = 0.23

    projected_jacobian = project_measurement_jacobian(
        absolute_jacobian,
        measurement_type="difference",
        reference_meas=reference,
        difference_mode=difference_mode,
        difference_orientation="target_minus_reference",
    )
    difference_vector = build_difference_vector(
        target,
        reference,
        mode=difference_mode,
        orientation="target_minus_reference",
    )

    rm = build_one_step_rm(
        projected_jacobian,
        lambda_=hyperparameter,
        mode="noser",
        form="param",
    )
    legacy_delta = _legacy_dense_noser_delta(
        projected_jacobian,
        difference_vector,
        hyperparameter=hyperparameter,
        monkeypatch=monkeypatch,
    )

    np.testing.assert_allclose(
        rm @ difference_vector,
        legacy_delta,
        rtol=1e-11,
        atol=1e-12,
    )


def test_eidors_one_step_controls_do_not_rescale_rm_delta() -> None:
    reconstructor = SimpleNamespace(
        _measurement_space_type="difference",
        active_preset_name="eidors_one_step_noser",
        max_iterations=1,
        difference_step_size_mode="off",
    )

    step = gn_runtime._select_step_size(
        reconstructor,
        iteration=0,
        sigma_current=object(),
        delta_sigma_torch=torch.ones(3, dtype=torch.float64),
        meas_torch=torch.ones(4, dtype=torch.float64),
        residual_norm_weighted=1.0,
        prior_torch=torch.zeros(3, dtype=torch.float64),
        lambda_eff=0.01,
    )
    sigma_final = np.array([1.1, 0.95, 1.02], dtype=np.float64)
    scaled, info = gn_runtime._apply_difference_step_size(
        reconstructor,
        sigma_final=sigma_final,
        measured_vector=np.ones(4, dtype=np.float64),
    )

    assert step == 1.0
    np.testing.assert_allclose(scaled, sigma_final)
    assert info["mode"] == "off"
    assert info["reason"] == "disabled"
