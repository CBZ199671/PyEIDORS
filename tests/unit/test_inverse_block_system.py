"""Tests for joint inverse block metadata contracts."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import pyeidors.inverse.block_system as block_system
from pyeidors.inverse.block_system import (
    assemble_sigma_contact_normal_system,
    build_electrode_movement_jacobian,
    build_sigma_contact_block_metadata,
    configure_petsc_fieldsplit_solver,
    make_block_diagonal_inverse_action,
    prior_movement,
    scale_contact_impedance_update,
    solve_sigma_contact_fieldsplit,
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


def test_sigma_contact_normal_system_and_scipy_solve_match_dense_reference():
    j_sigma = np.array(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    j_contact = np.array([[1.0], [0.5], [0.0], [1.0]], dtype=float)
    residual = np.array([1.0, -0.5, 0.25, 2.0], dtype=float)
    measurement_weights = np.array([1.0, 2.0, 0.5, 1.5], dtype=float)

    system = assemble_sigma_contact_normal_system(
        j_sigma,
        j_contact,
        residual,
        sigma_regularization=np.array([0.3, 0.4], dtype=float),
        contact_regularization=0.2,
        measurement_weights=measurement_weights,
        fieldsplit_type="multiplicative",
    )

    assert system.shape == (3, 3)
    assert system.metadata.block("sigma").slice == slice(0, 2)
    assert system.metadata.block("z_contact").slice == slice(2, 3)
    assert system.diagnostics["measurement_weights"] == "diagonal"
    expected = np.linalg.solve(system.matrix.toarray(), system.rhs)

    result = solve_sigma_contact_fieldsplit(
        j_sigma,
        j_contact,
        residual,
        sigma_regularization=np.array([0.3, 0.4], dtype=float),
        contact_regularization=0.2,
        measurement_weights=measurement_weights,
        fieldsplit_type="multiplicative",
        backend="scipy",
    )

    assert result.backend == "scipy"
    assert result.converged
    assert result.fieldsplit_type == "multiplicative"
    assert result.residual_norm < 1e-10
    np.testing.assert_allclose(result.solution, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.sigma_delta, expected[:2])
    np.testing.assert_allclose(result.contact_delta, expected[2:])


def test_v305_sigma_contact_normal_rhs_direct_fills_without_concatenate():
    j_sigma = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    j_contact = np.array([[2.0], [0.0], [1.0]], dtype=float)
    residual = np.array([1.0, 2.0, 3.0], dtype=float)

    system = assemble_sigma_contact_normal_system(j_sigma, j_contact, residual)

    np.testing.assert_allclose(system.rhs, np.array([4.0, 5.0, 5.0], dtype=float))
    assert system.rhs.flags.c_contiguous
    assert "np.concatenate" not in inspect.getsource(
        assemble_sigma_contact_normal_system
    )


def test_v481_block_system_finite_guards_use_bounded_scanner():
    checked_functions = (
        block_system._finite_vector,
        block_system._as_csr_matrix,
        block_system._regularization_to_csr,
        block_system._measurement_weights_to_csr,
        block_system.assemble_sigma_contact_normal_system,
        block_system._solve_sigma_contact_with_petsc,
        block_system._solve_sigma_contact_with_scipy,
        block_system.build_electrode_movement_jacobian,
        block_system.scale_contact_impedance_update,
    )
    old_payload_scans = (
        "np.isfinite(arr).all()",
        "np.isfinite(matrix.data).all()",
        "np.isfinite(rhs).all()",
        "np.isfinite(solution).all()",
        "np.isfinite(out).all()",
        "np.isfinite(baseline).all()",
        "np.isfinite(perturbed).all()",
        "np.isfinite(steps).all()",
        "np.isfinite(z).all()",
        "np.isfinite(delta).all()",
        "np.isfinite(updated).all()",
        "np.any(arr < 0.0)",
        "np.any(steps == 0.0)",
    )

    for func in checked_functions:
        source = inspect.getsource(func)
        assert "all_finite_values(" in source
        for old_payload_scan in old_payload_scans:
            assert old_payload_scan not in source


class _FakePC:
    def __init__(self) -> None:
        self.type = None
        self.fieldsplit_type = None
        self.schur_fact_type = None
        self.field_splits = None

    def setType(self, pc_type):
        self.type = pc_type

    def setFieldSplitType(self, fieldsplit_type):
        self.fieldsplit_type = fieldsplit_type

    def setFieldSplitSchurFactType(self, fact_type):
        self.schur_fact_type = fact_type

    def setFieldSplitIS(self, *field_splits):
        self.field_splits = field_splits


class _FakeKSP:
    comm = "comm-self"

    def __init__(self) -> None:
        self.type = None
        self.pc = _FakePC()
        self.tolerances = {}

    def setType(self, ksp_type):
        self.type = ksp_type

    def getPC(self):
        return self.pc

    def setTolerances(self, **kwargs):
        self.tolerances.update(kwargs)


class _FakePETSc:
    COMM_SELF = "comm-self"

    class PC:
        class CompositeType:
            ADDITIVE = "additive-enum"
            MULTIPLICATIVE = "multiplicative-enum"
            SCHUR = "schur-enum"

        class SchurFactType:
            FULL = "full-enum"

    class IS:
        def createStride(self, size, *, first, step, comm):
            return ("stride", int(size), int(first), int(step), comm)


@pytest.mark.parametrize(
    ("fieldsplit_type", "expected_enum"),
    [
        ("additive", "additive-enum"),
        ("multiplicative", "multiplicative-enum"),
        ("schur", "schur-enum"),
    ],
)
def test_configure_petsc_fieldsplit_solver_uses_block_slices(
    fieldsplit_type,
    expected_enum,
):
    metadata = build_sigma_contact_block_metadata(
        n_sigma=3,
        n_contact=2,
        n_measurements=5,
        fieldsplit_type=fieldsplit_type,
    )
    ksp = _FakeKSP()

    plan = configure_petsc_fieldsplit_solver(
        ksp,
        metadata,
        petsc_module=_FakePETSc,
        rtol=1e-7,
        maxiter=25,
    )

    assert ksp.type == "gmres"
    assert ksp.pc.type == "fieldsplit"
    assert ksp.pc.fieldsplit_type == expected_enum
    assert ksp.pc.field_splits == (
        ("sigma", ("stride", 3, 0, 1, "comm-self")),
        ("z_contact", ("stride", 2, 3, 1, "comm-self")),
    )
    assert ksp.tolerances == {"rtol": 1e-7, "max_it": 25}
    assert plan["pc_fieldsplit_type"] == fieldsplit_type
    if fieldsplit_type == "schur":
        assert ksp.pc.schur_fact_type == "full-enum"
    else:
        assert ksp.pc.schur_fact_type is None


def test_sigma_contact_fieldsplit_petsc_request_falls_back_when_unavailable(
    monkeypatch,
):
    monkeypatch.setattr(block_system, "_PETSc", None)

    result = solve_sigma_contact_fieldsplit(
        np.eye(2, dtype=float),
        np.ones((2, 1), dtype=float),
        np.array([1.0, 2.0], dtype=float),
        sigma_regularization=0.1,
        contact_regularization=0.2,
        backend="petsc",
    )

    assert result.backend == "scipy"
    assert result.diagnostics["backend_requested"] == "petsc"
    assert result.diagnostics["petsc_fallback_reason"] == "petsc_backend_unavailable"


def test_sigma_contact_joint_solver_validates_shapes_and_weights():
    with pytest.raises(ValueError, match="same measurement row count"):
        assemble_sigma_contact_normal_system(
            np.ones((2, 2), dtype=float),
            np.ones((3, 1), dtype=float),
            np.ones(2, dtype=float),
        )

    with pytest.raises(
        ValueError, match="measurement_weights matrix must be symmetric"
    ):
        assemble_sigma_contact_normal_system(
            np.eye(2, dtype=float),
            np.ones((2, 1), dtype=float),
            np.ones(2, dtype=float),
            measurement_weights=np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float),
        )

    with pytest.raises(ValueError, match="backend must be one of"):
        solve_sigma_contact_fieldsplit(
            np.eye(2, dtype=float),
            np.ones((2, 1), dtype=float),
            np.ones(2, dtype=float),
            backend="bad",
        )


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
