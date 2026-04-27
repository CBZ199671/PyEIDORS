"""T73 (V73): Direct vs Adjoint Jacobian sign-convention contract.

Frozen contract for the two coexisting calculators:
- ``DirectJacobianCalculator`` uses the PyEIDORS runtime sign convention
  ``J = +∂V/∂σ``. Combined with ``rhs = -jtr`` in
  ``gauss_newton_runtime.py`` this yields physical δσ matching EIDORS.
- ``EidorsJacobianAdapter`` uses the EIDORS canonical physical
  convention ``J = -∂V/∂σ`` (matches ``calc_jacobian_adjoint.m``'s
  final ``J = -J;`` step).

For any σ the two MUST satisfy ``Direct.calculate(σ) == -Adjoint.calculate(σ)``.
The absolute-value parity is asserted as an auxiliary sanity check.
``linearize(σ).to_dense()`` MUST obey the same signed parity.

These tests freeze the contract so future refactors (Path C: shared core +
sign adapter) cannot silently merge the calculators without preserving the
documented sign convention pairing with the GN runtime / RM build.
"""

from __future__ import annotations

import numpy as np
from dolfinx import fem

from pyeidors.inverse.jacobian.adjoint_jacobian import EidorsJacobianAdapter
from pyeidors.inverse.jacobian.direct_jacobian import DirectJacobianCalculator


def _baseline_sigma(system):
    sigma = fem.Function(system.fwd_model.V_sigma)
    sigma.x.array[:] = 1.0
    return sigma


def test_direct_calculate_equals_negative_adjoint_calculate(eit_system) -> None:
    """V73: signed parity ``Direct == -Adjoint`` on dense ``calculate`` path."""

    sigma = _baseline_sigma(eit_system)
    direct = DirectJacobianCalculator(eit_system.fwd_model)
    adjoint = EidorsJacobianAdapter(eit_system.fwd_model, use_torch=False)

    j_direct = direct.calculate(sigma)
    j_adjoint = adjoint.calculate(sigma)

    np.testing.assert_allclose(
        j_direct,
        -j_adjoint,
        rtol=1.0e-8,
        atol=1.0e-12,
        err_msg="Direct(σ) must equal -Adjoint(σ) per V73 sign convention.",
    )


def test_direct_and_adjoint_calculate_have_equal_magnitude(eit_system) -> None:
    """V73 aux: ``|Direct| == |Adjoint|`` (only the sign differs, not magnitude)."""

    sigma = _baseline_sigma(eit_system)
    direct = DirectJacobianCalculator(eit_system.fwd_model)
    adjoint = EidorsJacobianAdapter(eit_system.fwd_model, use_torch=False)

    j_direct = direct.calculate(sigma)
    j_adjoint = adjoint.calculate(sigma)

    np.testing.assert_allclose(
        np.abs(j_direct),
        np.abs(j_adjoint),
        rtol=1.0e-8,
        atol=1.0e-12,
        err_msg="Magnitudes must match; only the sign convention differs.",
    )


def test_direct_and_adjoint_linearize_to_dense_signed_parity(eit_system) -> None:
    """V73: ``linearize().to_dense()`` obeys the same ``Direct == -Adjoint`` contract."""

    sigma = _baseline_sigma(eit_system)
    direct = DirectJacobianCalculator(eit_system.fwd_model)
    adjoint = EidorsJacobianAdapter(eit_system.fwd_model, use_torch=False)

    direct_dense = direct.linearize(sigma).to_dense()
    adjoint_dense = adjoint.linearize(sigma).to_dense()

    np.testing.assert_allclose(
        direct_dense,
        -adjoint_dense,
        rtol=1.0e-8,
        atol=1.0e-12,
        err_msg="linearize().to_dense() must respect the V73 signed parity.",
    )


def test_calculator_sign_convention_metadata(eit_system) -> None:
    """V73: each calculator advertises its sign convention via class attribute."""

    sigma = _baseline_sigma(eit_system)
    direct = DirectJacobianCalculator(eit_system.fwd_model)
    adjoint = EidorsJacobianAdapter(eit_system.fwd_model, use_torch=False)

    assert direct.sign_convention == "+dV/dsigma_pyeidors_runtime"
    assert adjoint.sign_convention == "-dV/dsigma_eidors_canonical"

    direct_lin = direct.linearize(sigma)
    adjoint_lin = adjoint.linearize(sigma)
    assert float(direct_lin.sign) == 1.0
    assert float(adjoint_lin.sign) == -1.0
