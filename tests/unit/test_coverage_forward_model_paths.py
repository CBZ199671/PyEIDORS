"""Tests for eit_forward_model and cuda_structured_backend uncovered paths."""

from __future__ import annotations


import numpy as np
import pytest

_TEST_STACK_IMPORT_ERROR = None
try:
    from pyeidors.data.structures import EITImage
except Exception as exc:
    _TEST_STACK_IMPORT_ERROR = exc


def _skip_if_no_stack():
    if _TEST_STACK_IMPORT_ERROR is not None:
        pytest.skip(f"requires DOLFINx: {_TEST_STACK_IMPORT_ERROR}")


class TestForwardModelPaths:
    """Cover eit_forward_model.py uncovered lines using the eit_system fixture."""

    def test_forward_solve_basic(self, eit_system):
        """Exercise forward solve to cover assembly/solve paths."""
        _skip_if_no_stack()
        fwd = eit_system.fwd_model
        n_cells = fwd.V_sigma.dofmap.index_map.size_local
        sigma = np.ones(n_cells, dtype=float)
        img = EITImage(elem_data=sigma, fwd_model=fwd)
        data, info = fwd.fwd_solve(img)
        assert data.meas is not None
        assert len(data.meas) > 0

    def test_forward_solve_with_anomaly(self, eit_system):
        """Cover conductivity update paths."""
        _skip_if_no_stack()
        fwd = eit_system.fwd_model
        n_cells = fwd.V_sigma.dofmap.index_map.size_local
        sigma = np.ones(n_cells, dtype=float)
        sigma[: n_cells // 4] = 2.0  # anomaly
        img = EITImage(elem_data=sigma, fwd_model=fwd)
        data, info = fwd.fwd_solve(img)
        assert data.meas is not None


class TestCudaStructuredBackendValidation:
    """Cover cuda_structured_backend.py validation lines."""

    def test_unsupported_mesh_dim(self):
        try:
            from pyeidors.forward.cuda_structured_backend import _validate_preconditions
        except ImportError:
            pytest.skip("cuda_structured_backend not available")

        with pytest.raises((ValueError, RuntimeError)):
            _validate_preconditions(
                mesh_dim=1,
                scalar_type="float64",
                mpi_size=1,
                mesh_file="/nonexistent.msh",
                mesh_family="hex",
                geometry_version="geomv2",
            )
