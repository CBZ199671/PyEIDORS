"""Tests for joint inverse block metadata contracts."""

from __future__ import annotations

import numpy as np
import pytest

from pyeidors.inverse.block_system import (
    build_electrode_movement_jacobian,
    build_sigma_contact_block_metadata,
    make_block_diagonal_inverse_action,
    prior_movement,
    scale_contact_impedance_update,
)


def test_sigma_contact_metadata_shapes_and_fieldsplit_plan():
    metadata = build_sigma_contact_block_metadata(
        n_sigma=12,
        n_contact=4,
        n_measurements=20,
        fieldsplit_type="schur",
    )

    assert metadata.total_size == 16
    assert metadata.block("sigma").slice == slice(0, 12)
    assert metadata.block("z_contact").slice == slice(12, 16)
    assert metadata.block_slices() == {
        "sigma": slice(0, 12),
        "z_contact": slice(12, 16),
    }

    coupling_shapes = {coupling.name: coupling.shape for coupling in metadata.couplings}
    assert coupling_shapes["H_sigma_z"] == (12, 4)
    assert coupling_shapes["H_z_sigma"] == (4, 12)
    assert coupling_shapes["J_sigma"] == (20, 12)
    assert coupling_shapes["J_z_contact"] == (20, 4)

    plan = metadata.fieldsplit_plan()
    assert plan["pc_type"] == "fieldsplit"
    assert plan["pc_fieldsplit_type"] == "schur"
    assert plan["schur_approximation"] == "diag-z-and-prior-sigma"
    assert plan["upgrade_path"] == ["block-diagonal", "multiplicative", "schur"]
    assert plan["blocks"][0]["name"] == "sigma"
    assert plan["blocks"][1]["name"] == "z_contact"


def test_sigma_contact_movement_metadata_extends_fieldsplit_plan():
    metadata = build_sigma_contact_block_metadata(
        n_sigma=12,
        n_contact=4,
        n_movement=12,
        n_measurements=20,
        fieldsplit_type="multiplicative",
    )

    assert metadata.total_size == 28
    assert metadata.block("sigma").slice == slice(0, 12)
    assert metadata.block("z_contact").slice == slice(12, 16)
    assert metadata.block("e").slice == slice(16, 28)
    assert metadata.block("e").regularization == "prior_movement"
    assert metadata.block("e").metadata["dofs_per_electrode"] == 3.0

    coupling_shapes = {coupling.name: coupling.shape for coupling in metadata.couplings}
    assert coupling_shapes["H_sigma_e"] == (12, 12)
    assert coupling_shapes["H_e_sigma"] == (12, 12)
    assert coupling_shapes["H_z_e"] == (4, 12)
    assert coupling_shapes["H_e_z"] == (12, 4)
    assert coupling_shapes["H_ee"] == (12, 12)
    assert coupling_shapes["J_e"] == (20, 12)

    plan = metadata.fieldsplit_plan()
    assert plan["pc_type"] == "fieldsplit"
    assert plan["pc_fieldsplit_type"] == "multiplicative"
    assert [block["name"] for block in plan["blocks"]] == ["sigma", "z_contact", "e"]
    assert plan["blocks"][2]["regularization"] == "prior_movement"
    assert "schur" in plan["upgrade_path"]


def test_block_diagonal_inverse_action_is_shape_safe():
    metadata = build_sigma_contact_block_metadata(
        n_sigma=3,
        n_contact=2,
        n_measurements=7,
    )
    action = make_block_diagonal_inverse_action(
        metadata,
        sigma_inverse_action=lambda x: np.asarray(x, dtype=float) / 2.0,
        contact_inverse_action=lambda x: np.asarray(x, dtype=float) / 10.0,
    )

    out = action(np.array([2.0, 4.0, 6.0, 10.0, 20.0], dtype=float))
    np.testing.assert_allclose(out, np.array([1.0, 2.0, 3.0, 1.0, 2.0]))

    with pytest.raises(ValueError, match="Expected vector length 5"):
        action(np.ones(4, dtype=float))


def test_block_diagonal_inverse_action_handles_movement_block():
    metadata = build_sigma_contact_block_metadata(
        n_sigma=2,
        n_contact=1,
        n_movement=3,
    )
    with pytest.raises(ValueError, match="movement_inverse_action is required"):
        make_block_diagonal_inverse_action(
            metadata,
            sigma_inverse_action=lambda x: np.asarray(x, dtype=float),
            contact_inverse_action=lambda x: np.asarray(x, dtype=float),
        )

    action = make_block_diagonal_inverse_action(
        metadata,
        sigma_inverse_action=lambda x: np.asarray(x, dtype=float) + 1.0,
        contact_inverse_action=lambda x: np.asarray(x, dtype=float) * 2.0,
        movement_inverse_action=lambda x: -np.asarray(x, dtype=float),
    )

    out = action(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float))
    np.testing.assert_allclose(out, np.array([2.0, 3.0, 6.0, -4.0, -5.0, -6.0]))

    bad_movement = make_block_diagonal_inverse_action(
        metadata,
        sigma_inverse_action=lambda x: np.asarray(x, dtype=float),
        contact_inverse_action=lambda x: np.asarray(x, dtype=float),
        movement_inverse_action=lambda _x: np.ones(2, dtype=float),
    )
    with pytest.raises(ValueError, match="movement inverse action returned length"):
        bad_movement(np.ones(6, dtype=float))


def test_block_diagonal_inverse_action_validates_subblock_outputs():
    metadata = build_sigma_contact_block_metadata(n_sigma=3, n_contact=2)
    bad_sigma = make_block_diagonal_inverse_action(
        metadata,
        sigma_inverse_action=lambda _x: np.ones(2, dtype=float),
        contact_inverse_action=lambda x: np.asarray(x, dtype=float),
    )
    with pytest.raises(ValueError, match="sigma inverse action returned length"):
        bad_sigma(np.ones(5, dtype=float))

    bad_contact = make_block_diagonal_inverse_action(
        metadata,
        sigma_inverse_action=lambda x: np.asarray(x, dtype=float),
        contact_inverse_action=lambda _x: np.ones(1, dtype=float),
    )
    with pytest.raises(ValueError, match="contact inverse action returned length"):
        bad_contact(np.ones(5, dtype=float))


def test_sigma_contact_metadata_rejects_invalid_sizes_and_modes():
    with pytest.raises(ValueError, match="n_sigma must be positive"):
        build_sigma_contact_block_metadata(n_sigma=0, n_contact=4)
    with pytest.raises(ValueError, match="n_contact must be positive"):
        build_sigma_contact_block_metadata(n_sigma=4, n_contact=0)
    with pytest.raises(ValueError, match="n_measurements must be non-negative"):
        build_sigma_contact_block_metadata(n_sigma=4, n_contact=2, n_measurements=-1)
    with pytest.raises(ValueError, match="n_movement must be positive"):
        build_sigma_contact_block_metadata(n_sigma=4, n_contact=2, n_movement=0)
    with pytest.raises(ValueError, match="fieldsplit_type"):
        build_sigma_contact_block_metadata(
            n_sigma=4, n_contact=2, fieldsplit_type="bad"
        )
    with pytest.raises(KeyError, match="Unknown parameter block"):
        build_sigma_contact_block_metadata(n_sigma=4, n_contact=2).block("missing")


def test_electrode_movement_jacobian_finite_difference_orientations():
    baseline = np.array([10.0, 20.0, 30.0], dtype=float)
    perturbed = np.array(
        [
            [10.1, 20.4, 29.7],
            [9.8, 20.2, 30.6],
        ],
        dtype=float,
    )

    jacobian = build_electrode_movement_jacobian(
        baseline,
        perturbed,
        np.array([0.1, 0.2], dtype=float),
    )

    expected = np.array(
        [
            [1.0, -1.0],
            [4.0, 1.0],
            [-3.0, 3.0],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(jacobian, expected)

    jacobian_measurement_major = build_electrode_movement_jacobian(
        baseline,
        perturbed.T,
        np.array([0.1, 0.2], dtype=float),
        orientation="measurement-major",
    )
    np.testing.assert_allclose(jacobian_measurement_major, expected)

    with pytest.raises(ValueError, match="orientation"):
        build_electrode_movement_jacobian(baseline, perturbed, 0.1, orientation="bad")
    with pytest.raises(FloatingPointError, match="finite and non-zero"):
        build_electrode_movement_jacobian(baseline, perturbed, [0.1, 0.0])


def test_prior_movement_returns_positive_diagonal_sparse_matrix():
    prior = prior_movement(4, weight=2.5, floor=0.5)

    assert prior.shape == (4, 4)
    assert prior.getformat() == "csr"
    np.testing.assert_allclose(prior.diagonal(), np.full(4, 3.0))

    with pytest.raises(ValueError, match="n_movement must be positive"):
        prior_movement(0)
    with pytest.raises(ValueError, match="weight must be positive"):
        prior_movement(3, weight=0.0)
    with pytest.raises(ValueError, match="floor must be non-negative"):
        prior_movement(3, floor=-1.0)


def test_scale_contact_impedance_update_is_finite_positive_and_limited():
    current = np.array([1e-3, 2e-3], dtype=float)
    delta = np.array([2e-3, -10e-3], dtype=float)

    updated, step = scale_contact_impedance_update(
        current,
        delta,
        max_relative_step=0.5,
        floor=1e-6,
    )

    assert 0.0 < step < 1.0
    assert np.isfinite(updated).all()
    assert np.all(updated >= 1e-6)
    np.testing.assert_allclose(updated, current + step * delta)

    unchanged, unchanged_step = scale_contact_impedance_update(
        current, np.zeros_like(current)
    )
    np.testing.assert_allclose(unchanged, current)
    assert unchanged_step == 1.0

    with pytest.raises(ValueError, match="shape mismatch"):
        scale_contact_impedance_update(current, np.ones(3, dtype=float))
    with pytest.raises(FloatingPointError, match="non-finite"):
        scale_contact_impedance_update(current, np.array([np.nan, 0.0]))
    with pytest.raises(ValueError, match="max_relative_step"):
        scale_contact_impedance_update(current, delta, max_relative_step=0.0)
