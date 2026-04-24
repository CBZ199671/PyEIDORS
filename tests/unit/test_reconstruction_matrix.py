"""Tests for online reconstruction-matrix helpers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from pyeidors.data.difference import normalize_time_difference
from pyeidors.inverse import VoxelGrid
from pyeidors.inverse.prior import as_rtr_prior, graph_curvature_prior, graph_laplacian
from pyeidors.inverse.reconstruction_matrix import (
    build_one_step_rm,
    reconstruct_difference,
    reconstruct_difference_batch,
)


def test_build_one_step_rm_tikhonov_matches_dense_formula() -> None:
    jacobian = np.array([[1.0, 0.5], [0.0, 2.0], [1.0, -1.0]], dtype=float)
    lam = 0.25

    result = build_one_step_rm(
        jacobian,
        lambda_=lam,
        mode="tikhonov",
        return_metadata=True,
    )
    expected = np.linalg.solve(
        jacobian.T @ jacobian + lam**2 * np.eye(2),
        jacobian.T,
    )

    np.testing.assert_allclose(result.rm, expected)
    assert result.shape == (2, 3)
    assert result.metadata["mode"] == "tikhonov"
    assert result.metadata["form"] == "param"
    assert result.metadata["regularization_source"] == "identity"
    assert result.metadata["condition_estimate"] >= 1.0


def test_build_one_step_rm_noser_defaults_to_eidors_sqrt_diag_jtj() -> None:
    jacobian = np.array([[1.0, 2.0], [3.0, 0.5], [0.0, 1.0]], dtype=float)
    lam = 0.1

    rm = build_one_step_rm(jacobian, lambda_=lam, mode="noser")
    noser = np.diag(np.sum(jacobian * jacobian, axis=0) ** 0.5)
    expected = np.linalg.solve(jacobian.T @ jacobian + lam**2 * noser, jacobian.T)

    np.testing.assert_allclose(rm, expected)


def test_build_one_step_rm_noser_supports_legacy_exponent_one() -> None:
    jacobian = np.array([[1.0, 2.0], [3.0, 0.5], [0.0, 1.0]], dtype=float)
    lam = 0.1

    rm = build_one_step_rm(
        jacobian,
        lambda_=lam,
        mode="noser",
        noser_exponent=1.0,
    )
    noser = np.diag(np.sum(jacobian * jacobian, axis=0))
    expected = np.linalg.solve(jacobian.T @ jacobian + lam**2 * noser, jacobian.T)

    np.testing.assert_allclose(rm, expected)


def test_build_one_step_rm_laplace_accepts_sparse_regularization() -> None:
    jacobian = np.array(
        [[1.0, 0.0, 0.5], [0.0, 1.0, -0.25], [1.0, 1.0, 0.0]],
        dtype=float,
    )
    laplace = sparse.csr_matrix(
        np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
    )
    lam = 0.2

    result = build_one_step_rm(
        jacobian,
        regularization=laplace,
        lambda_=lam,
        mode="laplace",
        return_metadata=True,
    )
    expected = np.linalg.solve(
        jacobian.T @ jacobian + lam**2 * laplace.toarray(),
        jacobian.T,
    )

    np.testing.assert_allclose(result.rm, expected)
    assert result.metadata["regularization_source"] == "provided_laplace"
    assert result.metadata["RtR_kind"] == "sparse"
    assert result.metadata["RtR_signature_hash"]
    assert result.metadata["RtR_metadata"]["name"] == "laplace"


def test_build_one_step_rm_accepts_rtr_prior_contract() -> None:
    jacobian = np.array(
        [[1.0, 0.0, 0.5], [0.0, 1.0, -0.25], [1.0, 1.0, 0.0]],
        dtype=float,
    )
    laplace = sparse.csr_matrix(
        np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
    )
    prior = as_rtr_prior(laplace, n_parameters=3, name="laplace-prior")
    lam = 0.2

    result = build_one_step_rm(
        jacobian,
        regularization=prior,
        lambda_=lam,
        mode="laplace",
        return_metadata=True,
    )
    expected = np.linalg.solve(
        jacobian.T @ jacobian + lam**2 * laplace.toarray(),
        jacobian.T,
    )

    np.testing.assert_allclose(result.rm, expected)
    assert result.metadata["RtR_signature_hash"] == prior.signature_hash
    assert result.metadata["RtR_kind"] == prior.kind


@pytest.mark.parametrize("mode", ["curvature", "graph_ltl"])
def test_build_one_step_rm_curvature_mode_matches_laplace_but_keeps_distinct_signature(
    mode: str,
) -> None:
    mesh = VoxelGrid.from_bounds([0.0], [4.0], shape=(4,))
    jacobian = np.array(
        [[1.0, 0.0, 0.5, -0.25], [0.5, 2.0, -1.0, 0.75]],
        dtype=float,
    )
    laplace = graph_laplacian(mesh)
    curvature = graph_curvature_prior(mesh)
    lam = 0.3

    laplace_rm = build_one_step_rm(
        jacobian,
        regularization=laplace,
        lambda_=lam,
        mode="laplace",
        return_metadata=True,
    )
    curvature_rm = build_one_step_rm(
        jacobian,
        regularization=curvature
        if mode == "curvature"
        else curvature.as_RtR(dense=False),
        lambda_=lam,
        mode=mode,
        return_metadata=True,
    )

    np.testing.assert_allclose(curvature_rm.rm, laplace_rm.rm)
    assert curvature_rm.metadata["regularization_type"] == mode
    assert curvature_rm.metadata["regularization_source"] == "provided_graph_ltl"
    assert (
        curvature_rm.metadata["RtR_signature_hash"]
        != laplace_rm.metadata["RtR_signature_hash"]
    )
    assert curvature_rm.metadata["RtR_metadata"]["signature_hint"] == "graph_ltl"


@pytest.mark.parametrize("mode", ["tikhonov", "noser", "laplace"])
def test_build_one_step_rm_uses_official_hp2_rtr_weighted_fixture(mode: str) -> None:
    jacobian = np.array(
        [[1.0, -0.25, 0.5], [0.0, 1.5, -1.0], [2.0, 0.75, 0.25]],
        dtype=float,
    )
    weights = np.array([4.0, 0.25, 2.0], dtype=float)
    laplace = sparse.csr_matrix(
        np.array(
            [
                [1.0, -1.0, 0.0],
                [-1.0, 2.0, -1.0],
                [0.0, -1.0, 1.0],
            ],
            dtype=float,
        )
    )
    hp = 0.17
    regularization = laplace if mode == "laplace" else None

    result = build_one_step_rm(
        jacobian,
        regularization=regularization,
        lambda_=hp,
        mode=mode,
        measurement_weights=weights,
        return_metadata=True,
    )

    weighted_j = np.diag(np.sqrt(weights)) @ jacobian
    if mode == "noser":
        rtr = np.diag(np.sum(weighted_j * weighted_j, axis=0) ** 0.5)
    elif mode == "laplace":
        rtr = laplace.toarray()
    else:
        rtr = np.eye(jacobian.shape[1])
    expected_lhs = weighted_j.T @ weighted_j + hp**2 * rtr
    expected_rm = np.linalg.solve(expected_lhs, weighted_j.T)

    np.testing.assert_allclose(result.rm, expected_rm)
    assert result.metadata["algorithm"] == "one-step-gn"
    assert result.metadata["solver_family"] == "gauss-newton"
    assert result.metadata["regularization_type"] == mode
    assert result.metadata["normal_equation_formula"] == "JtWJ_plus_hp2_RtR"
    assert result.metadata["regularization_matrix_role"] == "RtR"
    assert result.metadata["hyperparameter_name"] == "hp"
    assert result.metadata["hp"] == pytest.approx(hp)
    assert result.metadata["hp_squared"] == pytest.approx(hp**2)
    assert result.metadata["lambda_squared"] == pytest.approx(hp**2)
    assert result.metadata["RtR_shape"] == rtr.shape
    assert result.metadata["RtR_nnz"] == int(np.count_nonzero(rtr))
    assert result.metadata["RtR_signature_hash"]
    assert result.metadata["RtR_metadata"]["schema"] == "pyeidors-rtr-prior-v1"
    if mode == "noser":
        assert result.metadata["noser_exponent"] == pytest.approx(0.5)


def test_build_one_step_rm_measurement_form_matches_param_for_tikhonov() -> None:
    jacobian = np.array(
        [[1.0, 0.0, 0.5, -0.25], [0.5, 2.0, -1.0, 0.75]],
        dtype=float,
    )
    lam = 0.3

    param_rm = build_one_step_rm(jacobian, lambda_=lam, mode="tikhonov")
    measurement = build_one_step_rm(
        jacobian,
        lambda_=lam,
        mode="tikhonov",
        form="measurement",
        return_metadata=True,
    )

    np.testing.assert_allclose(measurement.rm, param_rm, rtol=1e-10, atol=1e-12)
    assert measurement.metadata["form"] == "measurement"
    assert measurement.metadata["inversion_dimension"] == "measurement"
    assert measurement.metadata["system_shape"] == (2, 2)
    assert measurement.metadata["prior_inverse_solver"] == "solve"


def test_build_one_step_rm_measurement_form_matches_param_for_noser() -> None:
    jacobian = np.array(
        [[1.0, 2.0, 0.5], [3.0, 0.5, -1.0]],
        dtype=float,
    )
    lam = 0.15

    param_rm = build_one_step_rm(jacobian, lambda_=lam, mode="noser")
    measurement_rm = build_one_step_rm(
        jacobian,
        lambda_=lam,
        mode="noser",
        form="measurement",
    )

    np.testing.assert_allclose(measurement_rm, param_rm, rtol=1e-6, atol=1e-8)


def test_build_one_step_rm_measurement_form_matches_param_for_spd_laplace() -> None:
    jacobian = np.array(
        [[1.0, 0.0, 0.5], [0.0, 1.0, -0.25]],
        dtype=float,
    )
    laplace = sparse.csr_matrix(
        np.array(
            [
                [1.5, -1.0, 0.0],
                [-1.0, 2.5, -1.0],
                [0.0, -1.0, 1.5],
            ],
            dtype=float,
        )
    )
    lam = 0.2

    param_rm = build_one_step_rm(
        jacobian,
        regularization=laplace,
        lambda_=lam,
        mode="laplace",
    )
    measurement_rm = build_one_step_rm(
        jacobian,
        regularization=laplace,
        lambda_=lam,
        mode="laplace",
        form="measurement",
    )

    np.testing.assert_allclose(measurement_rm, param_rm, rtol=1e-7, atol=1e-9)


def test_build_one_step_rm_measurement_form_matches_param_for_curvature() -> None:
    mesh = VoxelGrid.from_bounds([0.0], [4.0], shape=(4,))
    jacobian = np.array(
        [[1.0, 0.0, 0.5, -0.25], [0.5, 2.0, -1.0, 0.75]],
        dtype=float,
    )
    curvature = as_rtr_prior(
        graph_curvature_prior(mesh).as_RtR(dense=False)
        + sparse.eye(4, format="csr") * 1e-9,
        name="curvature",
        metadata={"signature_hint": "graph_ltl"},
    )
    lam = 0.3

    param_rm = build_one_step_rm(
        jacobian,
        regularization=curvature,
        lambda_=lam,
        mode="curvature",
    )
    measurement_rm = build_one_step_rm(
        jacobian,
        regularization=curvature,
        lambda_=lam,
        mode="curvature",
        form="measurement",
    )

    np.testing.assert_allclose(measurement_rm, param_rm, rtol=1e-6, atol=1e-8)


def test_build_one_step_rm_measurement_form_accepts_measurement_regularization() -> (
    None
):
    jacobian = np.array([[1.0, 0.5, 0.0], [0.25, -1.0, 2.0]], dtype=float)
    rn = np.diag([2.0, 3.0])
    lam = 0.4

    rm = build_one_step_rm(
        jacobian,
        lambda_=lam,
        mode="tikhonov",
        form="measurement",
        measurement_regularization=rn,
        return_metadata=True,
    )
    expected = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + lam**2 * rn)

    np.testing.assert_allclose(rm.rm, expected)
    assert rm.metadata["measurement_regularization_source"] == "provided"


def test_build_one_step_rm_applies_bad_channels_and_weights_consistently() -> None:
    jacobian = np.array(
        [[1.0, 0.0], [5.0, 5.0], [0.0, 2.0]],
        dtype=float,
    )
    dv = np.array([2.0, 100.0, -1.0], dtype=float)
    mask = np.array([False, True, False], dtype=bool)
    weights = np.array([4.0, 9.0, 1.0], dtype=float)
    lam = 0.2

    result = build_one_step_rm(
        jacobian,
        lambda_=lam,
        mode="tikhonov",
        channel_mask=mask,
        measurement_weights=weights,
        return_metadata=True,
    )
    recon = reconstruct_difference(
        result.rm,
        dv,
        normalize=False,
        channel_mask=mask,
        measurement_weights=weights,
    )

    sqrt_w = np.diag(np.sqrt([4.0, 0.0, 1.0]))
    masked_j = jacobian.copy()
    masked_j[1, :] = 0.0
    masked_dv = dv.copy()
    masked_dv[1] = 0.0
    weighted_j = sqrt_w @ masked_j
    weighted_dv = sqrt_w @ masked_dv
    expected_rm = np.linalg.solve(
        weighted_j.T @ weighted_j + lam**2 * np.eye(2),
        weighted_j.T,
    )

    np.testing.assert_allclose(result.rm, expected_rm)
    np.testing.assert_allclose(recon, expected_rm @ weighted_dv)
    assert result.metadata["bad_channel_count"] == 1
    assert result.metadata["measurement_weight_kind"] == "diagonal"


def test_build_one_step_rm_measurement_form_honors_same_weight_contract() -> None:
    jacobian = np.array(
        [[1.0, 0.0, 0.5], [0.0, 2.0, -1.0], [1.0, -0.5, 0.25]],
        dtype=float,
    )
    mask = np.array([False, False, True], dtype=bool)
    weights = np.array([2.0, 0.5, 7.0], dtype=float)
    lam = 0.3

    param_rm = build_one_step_rm(
        jacobian,
        lambda_=lam,
        mode="tikhonov",
        channel_mask=mask,
        measurement_weights=weights,
    )
    measurement_rm = build_one_step_rm(
        jacobian,
        lambda_=lam,
        mode="tikhonov",
        form="measurement",
        channel_mask=mask,
        measurement_weights=weights,
    )

    np.testing.assert_allclose(measurement_rm, param_rm, rtol=1e-10, atol=1e-12)


def test_build_one_step_rm_rejects_invalid_form_and_measurement_rn_shape() -> None:
    with pytest.raises(ValueError, match="form must be"):
        build_one_step_rm(np.eye(2), lambda_=0.1, form="bad")
    with pytest.raises(ValueError, match="measurement_regularization"):
        build_one_step_rm(
            np.eye(2),
            lambda_=0.1,
            form="measurement",
            measurement_regularization=np.eye(3),
        )


def test_reconstruct_difference_applies_rm_to_normalized_time_difference() -> None:
    rm = np.array([[1.0, 2.0, -1.0], [0.5, 0.0, 4.0]], dtype=float)
    reference = np.array([2.0, 4.0, -2.0], dtype=float)
    target = np.array([3.0, 8.0, -1.0], dtype=float)

    expected_dv = normalize_time_difference(target, reference)
    expected = rm @ expected_dv

    np.testing.assert_allclose(
        reconstruct_difference(rm, target, normalize=True, v_ref=reference),
        expected,
    )


def test_reconstruct_difference_accepts_preprojected_dv_and_sparse_rm() -> None:
    rm = sparse.csr_matrix(np.array([[1.0, 0.0, 2.0], [0.0, -1.0, 1.0]], dtype=float))
    dv = np.array([0.25, -0.5, 2.0], dtype=float)

    np.testing.assert_allclose(
        reconstruct_difference(rm, dv, normalize=False),
        np.array([4.25, 2.5], dtype=float),
    )


def test_reconstruct_difference_batch_reuses_normalization_and_weight_contract() -> (
    None
):
    rm = np.array([[1.0, 0.0, 2.0], [-1.0, 3.0, 0.5]], dtype=float)
    reference = np.array([2.0, 4.0, 1.0], dtype=float)
    targets = np.array([[3.0, 8.0, 2.0], [1.0, 2.0, 5.0]], dtype=float)
    mask = np.array([False, True, False], dtype=bool)
    weights = np.array([4.0, 9.0, 0.25], dtype=float)

    result = reconstruct_difference_batch(
        rm,
        targets,
        normalize=True,
        v_ref=reference,
        channel_mask=mask,
        measurement_weights=weights,
        device="cpu",
        return_metadata=True,
    )

    normalized = np.vstack(
        [normalize_time_difference(row, reference) for row in targets]
    )
    normalized[:, 1] = 0.0
    weighted = normalized @ np.diag(np.sqrt([4.0, 0.0, 0.25]))
    np.testing.assert_allclose(result.values, weighted @ rm.T)
    assert result.metadata["batched"] is True
    assert result.metadata["device_effective"] == "cpu"


def test_reconstruct_difference_batch_matches_rowwise_contract_with_large_diagonal() -> (
    None
):
    n_measurements = 32
    n_frames = 7
    rm = np.arange(4 * n_measurements, dtype=float).reshape(4, n_measurements) / 100.0
    reference = np.linspace(2.0, 4.0, n_measurements)
    targets = np.vstack(
        [reference * (1.0 + 0.001 * (idx + 1)) for idx in range(n_frames)]
    )
    mask = np.zeros(n_measurements, dtype=bool)
    mask[::5] = True
    weights = np.linspace(0.5, 2.0, n_measurements)

    result = reconstruct_difference_batch(
        rm,
        targets,
        normalize=True,
        v_ref=reference,
        channel_mask=mask,
        measurement_weights=weights,
        device="cpu",
        return_metadata=True,
    )

    normalized = np.vstack(
        [normalize_time_difference(row, reference) for row in targets]
    )
    expected_payload = normalized.copy()
    expected_payload[:, mask] = 0.0
    masked_weights = weights.copy()
    masked_weights[mask] = 0.0
    expected_payload *= np.sqrt(masked_weights).reshape(1, -1)
    np.testing.assert_allclose(result.values, expected_payload @ rm.T)
    assert result.metadata["forward_solve_count"] == 0
    assert result.metadata["ksp_solve_count"] == 0


def test_reconstruct_difference_batch_accepts_single_frame_vector() -> None:
    rm = np.array([[1.0, 2.0], [3.0, -1.0]], dtype=float)
    dv = np.array([0.5, 2.0], dtype=float)

    out = reconstruct_difference_batch(rm, dv, normalize=False, device="cpu")

    np.testing.assert_allclose(out, rm @ dv)


def test_reconstruct_difference_validates_shapes_and_finite_values() -> None:
    rm = np.ones((2, 3), dtype=float)
    with pytest.raises(ValueError, match="measurement dimension"):
        reconstruct_difference(rm, np.ones(2), normalize=False)
    with pytest.raises(FloatingPointError, match="dv contains non-finite"):
        reconstruct_difference(rm, np.array([1.0, np.nan, 2.0]), normalize=False)
    with pytest.raises(ValueError, match="rm must be a 2D"):
        reconstruct_difference(np.ones(3), np.ones(3), normalize=False)
