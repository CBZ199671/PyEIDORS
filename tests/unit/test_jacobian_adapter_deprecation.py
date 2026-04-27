"""T75 Stage 5: ``EidorsStyleAdjointJacobian`` alias deprecation contract.

The class historically named ``EidorsStyleAdjointJacobian`` was renamed
to ``EidorsJacobianAdapter`` once it shrank to a thin EIDORS-canonical
sign-flip wrapper around the shared assembly core. The legacy name
remains importable for one release cycle so external callers can
migrate, but constructing it MUST emit a
:class:`DeprecationWarning` so the migration is visible. The alias also
MUST stay behaviorally identical to the canonical class — same MRO,
same sign convention, same dense ``calculate(σ)`` output. These tests
freeze that contract until the alias is removed in the next cycle.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from dolfinx import fem

from pyeidors.inverse.jacobian.adjoint_jacobian import (
    EidorsJacobianAdapter,
    EidorsStyleAdjointJacobian,
)


def _baseline_sigma(system):
    sigma = fem.Function(system.fwd_model.V_sigma)
    sigma.x.array[:] = 1.0
    return sigma


def test_legacy_alias_subclasses_canonical_adapter() -> None:
    assert issubclass(EidorsStyleAdjointJacobian, EidorsJacobianAdapter)
    assert (
        EidorsStyleAdjointJacobian.sign_convention
        == EidorsJacobianAdapter.sign_convention
        == "-dV/dsigma_eidors_canonical"
    )


def test_legacy_alias_construction_emits_deprecation_warning(eit_system) -> None:
    with pytest.warns(DeprecationWarning, match="EidorsStyleAdjointJacobian"):
        legacy = EidorsStyleAdjointJacobian(eit_system.fwd_model, use_torch=False)

    assert isinstance(legacy, EidorsJacobianAdapter)


def test_canonical_adapter_construction_silent(eit_system) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        EidorsJacobianAdapter(eit_system.fwd_model, use_torch=False)


def test_alias_and_canonical_produce_identical_jacobian(eit_system) -> None:
    sigma = _baseline_sigma(eit_system)
    canonical = EidorsJacobianAdapter(eit_system.fwd_model, use_torch=False)
    with pytest.warns(DeprecationWarning):
        legacy = EidorsStyleAdjointJacobian(eit_system.fwd_model, use_torch=False)

    np.testing.assert_allclose(
        canonical.calculate(sigma),
        legacy.calculate(sigma),
        rtol=1e-12,
        atol=1e-14,
    )
