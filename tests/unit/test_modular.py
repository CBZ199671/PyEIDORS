"""Jacobian and regularization unit/integration checks."""

from __future__ import annotations

import numpy as np
from dolfinx import fem

from pyeidors.inverse.jacobian.adjoint_jacobian import EidorsStyleAdjointJacobian
from pyeidors.inverse.jacobian.direct_jacobian import DirectJacobianCalculator
from pyeidors.inverse.regularization.smoothness import (
    NOSERRegularization,
    SmoothnessRegularization,
    TikhonovRegularization,
)


def _baseline_sigma(system):
    sigma = fem.Function(system.fwd_model.V_sigma)
    sigma.x.array[:] = 1.0
    return sigma


def test_direct_and_adjoint_jacobian_shapes(eit_system):
    sigma = _baseline_sigma(eit_system)
    direct = DirectJacobianCalculator(eit_system.fwd_model)
    adjoint = EidorsStyleAdjointJacobian(eit_system.fwd_model, use_torch=False)

    j_direct = direct.calculate(sigma)
    j_adjoint = adjoint.calculate(sigma)

    assert j_direct.shape == j_adjoint.shape
    assert j_direct.shape[0] == eit_system.fwd_model.pattern_manager.n_meas_total
    assert j_direct.shape[1] == sigma.x.array.size
    assert np.isfinite(j_direct).all()
    assert np.isfinite(j_adjoint).all()


def test_regularization_matrix_shapes(eit_system):
    sigma = _baseline_sigma(eit_system)
    jac = DirectJacobianCalculator(eit_system.fwd_model)

    noser = NOSERRegularization(eit_system.fwd_model, jacobian_calculator=jac, base_conductivity=1.0)
    smooth = SmoothnessRegularization(eit_system.fwd_model, alpha=0.5)
    tik = TikhonovRegularization(eit_system.fwd_model, alpha=0.25)

    for mat in (noser.get_regularization_matrix(), smooth.get_regularization_matrix(), tik.get_regularization_matrix()):
        assert mat.shape[0] == mat.shape[1] == sigma.x.array.size
        assert np.isfinite(mat).all()
